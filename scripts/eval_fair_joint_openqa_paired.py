"""Paired five-dataset evaluation of matched Engram and fair joint training.

The baseline and candidate were each fit on the same 20M Wikipedia-token
target stage from the same deterministic direct-reader initialization.  This
script loads them sequentially so one LUMI GCD is sufficient.
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_dual_reader_openqa_paired import (
    TASKS,
    compact_metrics,
    compare_task_results,
    load_task,
    set_reader_mode,
)
from scripts.eval_openqa import (
    TASK_SCALAR_METRICS,
    _is_tri_reader_config,
    evaluate_openqa,
    evaluate_truthfulqa,
    get_model_max_context,
    load_adaptor_runtime_config,
    setup_condition,
    task_scalar_score,
)
from scripts.tri_memory_oracle import build_all_oracles, capture_ratio


DUAL_MODES = ("engram_baseline", "joint_engram_only", "joint_both", "joint_routed")
TRI_MODE_MAP = {
    "tri_E": "engram_only",
    "tri_GE": "generated_from_engram_only",
    "tri_GH": "generated_from_context_only",
    "tri_E_GE": "e_ge",
    "tri_E_GH": "e_gh",
    "tri_GE_GH": "ge_gh",
    "tri_reader_hard": "tri_routed",
    "tri_reader_soft": "tri_soft_fused",
    "tri_reader_advantage": "tri_advantage_routed",
    "tri_safe_routed": "tri_safe_routed",
    "tri_subset_reader_hard": "tri_subset_routed",
    "tri_subset_reader_soft": "tri_subset_soft_fused",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--baseline-adaptor-dir", required=True)
    parser.add_argument("--baseline-adaptor-checkpoint", default=None)
    parser.add_argument("--joint-adaptor-dir", required=True)
    parser.add_argument("--joint-adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument(
        "--candidate-modes",
        nargs="+",
        default=None,
        help="Optional runtime result names to evaluate for a correction pass.",
    )
    parser.add_argument("--triviaqa-config", choices=["rc.nocontext"], default="rc.nocontext")
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--reasoning-mode", choices=["vanilla", "cot"], default="vanilla")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")
    return args


def _condition_args(args, adaptor_dir: str, checkpoint: str | None):
    values = vars(args).copy()
    values.update({
        "adaptor_dir": adaptor_dir,
        "adaptor_checkpoint": checkpoint,
        "dual_reader_mode": "both",
    })
    return argparse.Namespace(**values)


def _candidate_modes(adaptor_dir: str) -> tuple[dict[str, str], bool]:
    """Select the runtime matrix from the saved adaptor architecture."""
    config = load_adaptor_runtime_config(adaptor_dir)
    if _is_tri_reader_config(config):
        candidate_modes = dict(TRI_MODE_MAP)
        advantage = config.get("advantage_reader")
        if not isinstance(advantage, dict) or not advantage.get("enabled"):
            candidate_modes.pop("tri_reader_advantage", None)
        return candidate_modes, True
    return {
        "joint_engram_only": "engram_only",
        # Diagnostic for the already-trained dual-reader checkpoint: the
        # generated branch is evaluated as a standalone GE expert.  This is
        # runtime-only and does not retrain the historical baseline.
        "joint_generated_only": "generated_only",
        "joint_both": "both",
        "joint_routed": "routed",
    }, False


def _evaluate_tasks(
    wrapper,
    set_canon_fn,
    task_data: dict,
    args,
    device: torch.device,
) -> dict:
    tokenizer = wrapper.tokenizer
    max_context = get_model_max_context(wrapper, args.max_context_length)
    results = {}
    for task in args.tasks:
        started = time.time()
        if task == "truthfulqa":
            metrics = evaluate_truthfulqa(
                wrapper=wrapper,
                set_canon_fn=set_canon_fn,
                tokenizer=tokenizer,
                examples=task_data[task],
                device=device,
                max_context_length=max_context,
                retain_all_examples=True,
            )
        else:
            metrics = evaluate_openqa(
                wrapper=wrapper,
                set_canon_fn=set_canon_fn,
                tokenizer=tokenizer,
                task_name=task,
                examples=task_data[task],
                device=device,
                max_new_tokens=args.max_new_tokens,
                max_context_length=max_context,
                reasoning_mode=args.reasoning_mode,
                retain_all_predictions=True,
            )
        metrics["elapsed_s"] = time.time() - started
        results[task] = {"metrics": metrics}
        print(f"{task}: " + json.dumps(compact_metrics(task, metrics)), flush=True)
    return results


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cleanup_wrapper(wrapper) -> None:
    wrapper.cleanup()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    task_data = {}
    task_meta = {}
    for task in args.tasks:
        examples, dataset_meta = load_task(task, args.triviaqa_config)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        task_data[task] = examples
        task_meta[task] = {
            "dataset": dataset_meta,
            "n_examples": len(examples),
            "scalar_metric": TASK_SCALAR_METRICS[task],
        }
        print(f"Loaded {task}: {len(examples)} examples from {dataset_meta}", flush=True)

    evaluation = {}
    baseline_args = _condition_args(
        args, args.baseline_adaptor_dir, args.baseline_adaptor_checkpoint
    )
    baseline, baseline_canon = setup_condition(
        baseline_args, "transferred", device, dtype
    )
    print("Evaluating matched 20M-token Engram baseline", flush=True)
    evaluation["engram_baseline"] = _evaluate_tasks(
        baseline, baseline_canon, task_data, args, device
    )
    _cleanup_wrapper(baseline)
    del baseline
    _release_cuda_memory()

    joint_args = _condition_args(
        args, args.joint_adaptor_dir, args.joint_adaptor_checkpoint
    )
    joint, joint_canon = setup_condition(joint_args, "transferred", device, dtype)
    candidate_mode_map, is_tri = _candidate_modes(args.joint_adaptor_dir)
    if args.candidate_modes is not None:
        unknown = [name for name in args.candidate_modes if name not in candidate_mode_map]
        if unknown:
            raise ValueError(
                f"Unsupported candidate modes for this checkpoint: {unknown}; "
                f"available={list(candidate_mode_map)}"
            )
        candidate_mode_map = {
            name: candidate_mode_map[name] for name in args.candidate_modes
        }
    for result_mode, runtime_mode in candidate_mode_map.items():
        set_reader_mode(joint, runtime_mode)
        print(f"Evaluating {result_mode}", flush=True)
        evaluation[result_mode] = _evaluate_tasks(
            joint, joint_canon, task_data, args, device
        )
    _cleanup_wrapper(joint)
    del joint
    _release_cuda_memory()

    comparisons = {}
    if args.bootstrap_samples:
        for task in args.tasks:
            baseline_metrics = evaluation["engram_baseline"][task]["metrics"]
            comparisons[task] = {
                mode: compare_task_results(
                    task,
                    baseline_metrics,
                    evaluation[mode][task]["metrics"],
                    candidate_name=mode,
                    seed=args.seed,
                    bootstrap_samples=args.bootstrap_samples,
                )
                for mode in candidate_mode_map
            }

    oracles = {}
    capture = {}
    source_capture = {}
    runtime_capture = {}
    if is_tri:
        oracles = {
            task: build_all_oracles(task, evaluation) for task in args.tasks
        }
        if args.bootstrap_samples:
            for task in args.tasks:
                baseline_metrics = evaluation["engram_baseline"][task]["metrics"]
                comparisons[task].update(
                    {
                        oracle_name: compare_task_results(
                            task,
                            baseline_metrics,
                            oracle_metrics,
                            candidate_name=oracle_name,
                            seed=args.seed,
                            bootstrap_samples=args.bootstrap_samples,
                        )
                        for oracle_name, oracle_metrics in oracles[task].items()
                    }
                )

    modes = ("engram_baseline", *candidate_mode_map.keys())
    scalar_summary = {
        mode: {
            task: task_scalar_score(task, evaluation[mode][task]["metrics"])
            for task in args.tasks
        }
        for mode in modes
    }
    for mode in modes:
        scalar_summary[mode]["macro_average"] = float(
            np.mean([scalar_summary[mode][task] for task in args.tasks])
        )

    oracle_summary = {}
    if is_tri:
        oracle_summary = {
            oracle_name: {
                task: task_scalar_score(task, oracles[task][oracle_name])
                for task in args.tasks
            }
            for oracle_name in next(iter(oracles.values()))
        }
        for oracle_name in oracle_summary:
            oracle_summary[oracle_name]["macro_average"] = float(
                np.mean([oracle_summary[oracle_name][task] for task in args.tasks])
            )
        for reader_name in (
            "tri_reader_hard",
            "tri_reader_soft",
            "tri_reader_advantage",
            "tri_subset_reader_hard",
            "tri_subset_reader_soft",
        ):
            if reader_name not in scalar_summary:
                continue
            capture[reader_name] = {}
            source_capture[reader_name] = {}
            runtime_capture[reader_name] = {}
            for task in args.tasks:
                capture[reader_name][task] = capture_ratio(
                    scalar_summary[reader_name][task],
                    scalar_summary["engram_baseline"][task],
                    oracle_summary["oracle_E_GE_GH"][task],
                )
                source_capture[reader_name][task] = capture_ratio(
                    scalar_summary[reader_name][task],
                    scalar_summary["engram_baseline"][task],
                    oracle_summary["oracle_source_E_GE_GH"][task],
                )
                runtime_capture[reader_name][task] = capture_ratio(
                    scalar_summary[reader_name][task],
                    scalar_summary["engram_baseline"][task],
                    oracle_summary["oracle_all_nonempty_subsets"][task],
                )
            capture[reader_name]["macro_average"] = capture_ratio(
                scalar_summary[reader_name]["macro_average"],
                scalar_summary["engram_baseline"]["macro_average"],
                oracle_summary["oracle_E_GE_GH"]["macro_average"],
            )
            source_capture[reader_name]["macro_average"] = capture_ratio(
                scalar_summary[reader_name]["macro_average"],
                scalar_summary["engram_baseline"]["macro_average"],
                oracle_summary["oracle_source_E_GE_GH"]["macro_average"],
            )
            runtime_capture[reader_name]["macro_average"] = capture_ratio(
                scalar_summary[reader_name]["macro_average"],
                scalar_summary["engram_baseline"]["macro_average"],
                oracle_summary["oracle_all_nonempty_subsets"]["macro_average"],
            )

    results = {
        "target_model": args.target_model,
        "training_control": {
            "baseline_tokens": 20_000_000,
            "joint_tokens": 20_000_000,
            "training_corpus": "wikipedia-2021-causal-next-token-only",
            "same_direct_engram_initialization": True,
            "candidate_design": "tri_reader" if is_tri else "dual_reader",
        },
        "seed": args.seed,
        "tasks": task_meta,
        "modes": list(modes),
        "evaluation": evaluation,
        "oracles": oracles,
        "paired_comparisons": comparisons,
        "bootstrap_completed": bool(args.bootstrap_samples),
        "scalar_summary": scalar_summary,
        "oracle_scalar_summary": oracle_summary,
        "reader_capture_ratio": capture,
        "reader_capture_ratio_source_selection": source_capture,
        "reader_capture_ratio_all_nonempty_subsets": runtime_capture,
        "completed": True,
    }
    with open(output_dir / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    print("ATHENA_FAIR_JOINT_PAIRED_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
