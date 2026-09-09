"""Paired five-dataset evaluation of E, GE, GH, and their learned Reader."""

from __future__ import annotations

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
    evaluate_openqa,
    evaluate_truthfulqa,
    get_model_max_context,
    setup_condition,
    task_scalar_score,
)
from scripts.tri_memory_oracle import build_all_oracles, capture_ratio


MODES = (
    "engram_baseline",
    "tri_E",
    "tri_GE",
    "tri_GH",
    "tri_E_GE",
    "tri_E_GH",
    "tri_GE_GH",
    "tri_reader_hard",
    "tri_reader_soft",
    "tri_reader_advantage",
    "tri_subset_reader_hard",
    "tri_subset_reader_soft",
)
RUNTIME_MODES = {
    "tri_E": "engram_only",
    "tri_GE": "generated_from_engram_only",
    "tri_GH": "generated_from_context_only",
    "tri_E_GE": "e_ge",
    "tri_E_GH": "e_gh",
    "tri_GE_GH": "ge_gh",
    "tri_reader_hard": "tri_routed",
    "tri_reader_soft": "tri_soft_fused",
    "tri_reader_advantage": "tri_advantage_routed",
    "tri_subset_reader_hard": "tri_subset_routed",
    "tri_subset_reader_soft": "tri_subset_soft_fused",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--baseline-adaptor-dir", required=True)
    parser.add_argument("--baseline-adaptor-checkpoint", default=None)
    parser.add_argument("--tri-adaptor-dir", required=True)
    parser.add_argument("--tri-adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
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
    values.update(
        {
            "adaptor_dir": adaptor_dir,
            "adaptor_checkpoint": checkpoint,
            "dual_reader_mode": "both",
        }
    )
    return argparse.Namespace(**values)


def _evaluate_tasks(wrapper, set_canon_fn, task_data, args, device):
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


def _release(wrapper):
    wrapper.cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _supports_advantage_reader(wrapper) -> bool:
    adaptors = (
        list(wrapper.adaptor)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList)
        else [wrapper.adaptor]
    )
    return bool(adaptors) and all(
        getattr(adaptor, "advantage_router", None) is not None
        for adaptor in adaptors
    )


def _scalar(task: str, metrics: dict) -> float:
    return task_scalar_score(task, metrics)


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
    _release(baseline)
    del baseline

    tri_args = _condition_args(args, args.tri_adaptor_dir, args.tri_adaptor_checkpoint)
    tri, tri_canon = setup_condition(tri_args, "transferred", device, dtype)
    runtime_modes = dict(RUNTIME_MODES)
    if not _supports_advantage_reader(tri):
        runtime_modes.pop("tri_reader_advantage", None)
        print(
            "Skipping tri_reader_advantage: checkpoint has no configured advantage head",
            flush=True,
        )
    for result_mode, runtime_mode in runtime_modes.items():
        set_reader_mode(tri, runtime_mode)
        print(f"Evaluating {result_mode} ({runtime_mode})", flush=True)
        evaluation[result_mode] = _evaluate_tasks(
            tri, tri_canon, task_data, args, device
        )
    _release(tri)
    del tri

    oracles = {
        task: build_all_oracles(task, evaluation) for task in args.tasks
    }
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
                for mode in runtime_modes
            }
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

    evaluated_modes = ("engram_baseline", *runtime_modes.keys())
    scalar_summary = {
        mode: {
            task: _scalar(task, evaluation[mode][task]["metrics"])
            for task in args.tasks
        }
        for mode in evaluated_modes
    }
    for mode in evaluated_modes:
        scalar_summary[mode]["macro_average"] = float(
            np.mean([scalar_summary[mode][task] for task in args.tasks])
        )
    oracle_summary = {
        oracle_name: {
            task: _scalar(task, oracles[task][oracle_name])
            for task in args.tasks
        }
        for oracle_name in next(iter(oracles.values()))
    }
    for oracle_name in oracle_summary:
        oracle_summary[oracle_name]["macro_average"] = float(
            np.mean([oracle_summary[oracle_name][task] for task in args.tasks])
        )

    capture = {}
    source_capture = {}
    runtime_capture = {}
    reader_modes = tuple(
        reader_name
        for reader_name in (
            "tri_reader_hard",
            "tri_reader_soft",
            "tri_reader_advantage",
            "tri_subset_reader_hard",
            "tri_subset_reader_soft",
        )
        if reader_name in scalar_summary
    )
    for task in args.tasks:
        oracle = oracle_summary["oracle_E_GE_GH"][task]
        source_oracle = oracle_summary["oracle_source_E_GE_GH"][task]
        runtime_oracle = oracle_summary["oracle_all_nonempty_subsets"][task]
        capture[task] = {
            reader_name: {
                "relative_to_matched_engram": capture_ratio(
                    scalar_summary[reader_name][task],
                    scalar_summary["engram_baseline"][task],
                    oracle,
                ),
                "relative_to_joint_tri_E": capture_ratio(
                    scalar_summary[reader_name][task],
                    scalar_summary["tri_E"][task],
                    oracle,
                ),
            }
            for reader_name in reader_modes
        }
        source_capture[task] = {
            reader_name: capture_ratio(
                scalar_summary[reader_name][task],
                scalar_summary["engram_baseline"][task],
                source_oracle,
            )
            for reader_name in reader_modes
        }
        runtime_capture[task] = {
            reader_name: capture_ratio(
                scalar_summary[reader_name][task],
                scalar_summary["engram_baseline"][task],
                runtime_oracle,
            )
            for reader_name in reader_modes
        }
    capture["macro_average"] = {
        reader_name: {
            "relative_to_matched_engram": capture_ratio(
                scalar_summary[reader_name]["macro_average"],
                scalar_summary["engram_baseline"]["macro_average"],
                oracle_summary["oracle_E_GE_GH"]["macro_average"],
            ),
            "relative_to_joint_tri_E": capture_ratio(
                scalar_summary[reader_name]["macro_average"],
                scalar_summary["tri_E"]["macro_average"],
                oracle_summary["oracle_E_GE_GH"]["macro_average"],
            ),
        }
        for reader_name in reader_modes
    }
    source_capture["macro_average"] = {
        reader_name: capture_ratio(
            scalar_summary[reader_name]["macro_average"],
            scalar_summary["engram_baseline"]["macro_average"],
            oracle_summary["oracle_source_E_GE_GH"]["macro_average"],
        )
        for reader_name in reader_modes
    }
    runtime_capture["macro_average"] = {
        reader_name: capture_ratio(
            scalar_summary[reader_name]["macro_average"],
            scalar_summary["engram_baseline"]["macro_average"],
            oracle_summary["oracle_all_nonempty_subsets"]["macro_average"],
        )
        for reader_name in reader_modes
    }

    results = {
        "target_model": args.target_model,
        "training_control": {
            "baseline_tokens": 20_000_000,
            "tri_tokens": 20_000_000,
            "training_corpus": "wikipedia-2021-causal-next-token-only",
            "downstream_training": False,
            "same_direct_engram_initialization": True,
        },
        "seed": args.seed,
        "tasks": task_meta,
        "modes": list(evaluated_modes),
        "evaluation": evaluation,
        "oracles": oracles,
        "oracle_note": (
            "Gold-aware per-example diagnostics only; labels were not used by "
            "training or the learned Reader. Source oracles select one expert; "
            "the all-nonempty-subsets oracle also includes pair/triple endpoints."
        ),
        "paired_comparisons": comparisons,
        "bootstrap_completed": bool(args.bootstrap_samples),
        "scalar_summary": scalar_summary,
        "oracle_scalar_summary": oracle_summary,
        "reader_oracle_capture_ratio": capture,
        "reader_oracle_capture_ratio_source_selection": source_capture,
        "reader_oracle_capture_ratio_all_nonempty_subsets": runtime_capture,
        "completed": True,
    }
    with open(output_dir / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    print("ATHENA_TRI_MEMORY_PAIRED_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
