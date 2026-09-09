"""Paired cross-dataset evaluation of a routed dual-reader checkpoint.

The checkpoint is evaluated in three runtime-only modes: direct Engram,
unrouted Engram plus generated residual, and the learned safe router.  No
benchmark examples are used for training or router selection.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_openqa import (
    TASK_LOADERS,
    TASK_SCALAR_METRICS,
    evaluate_openqa,
    evaluate_truthfulqa,
    get_model_max_context,
    load_triviaqa_examples,
    setup_condition,
    task_scalar_score,
)


TASKS = ("nq", "webqa", "triviaqa", "truthfulqa", "hotpotqa")
DEFAULT_TASKS = ("webqa", "triviaqa", "truthfulqa", "hotpotqa")
MODES = ("engram_only", "both", "routed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--triviaqa-config", choices=["rc.nocontext"], default="rc.nocontext")
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--reasoning-mode", choices=["vanilla", "cot"], default="vanilla")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Compute paired bootstrap inline; use 0 when a CPU post-processing job will do it",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")
    return args


def set_reader_mode(wrapper, mode: str) -> None:
    adaptors = (
        list(wrapper.adaptor)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList)
        else [wrapper.adaptor]
    )
    for adaptor in adaptors:
        tri_setter = getattr(adaptor, "set_tri_reader_mode", None)
        # GenerativeMemoryAdaptor exposes the tri-reader compatibility method
        # even for legacy dual-reader checkpoints.  Dispatch by the saved
        # architecture, not by method presence; otherwise a dual checkpoint
        # crashes when a runtime-only generated_only correction is requested.
        is_tri = tri_setter is not None and getattr(
            adaptor, "fusion_type", None
        ) == "tri_reader"
        if is_tri:
            tri_setter(mode)
            continue
        if getattr(adaptor, "router", None) is None:
            raise RuntimeError("The checkpoint does not contain a trained safe router")
        else:
            adaptor.set_dual_reader_mode(mode)


def _interval(values: list[float]) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def paired_bootstrap(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict:
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    if baseline_values.shape != candidate_values.shape or baseline_values.ndim != 1:
        raise ValueError("Paired metric arrays must be one-dimensional and equally sized")
    if baseline_values.size == 0:
        raise ValueError("Cannot bootstrap an empty evaluation")

    delta = candidate_values - baseline_values
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(samples):
        indices = rng.integers(0, len(delta), size=len(delta))
        bootstrap.append(float(delta[indices].mean()))
    return {
        "delta": float(delta.mean()),
        "delta_95pct_paired_bootstrap": _interval(bootstrap),
        "improved": int((delta > 0).sum()),
        "regressed": int((delta < 0).sum()),
        "tied": int((delta == 0).sum()),
    }


def _check_pairing(baseline_examples: list[dict], candidate_examples: list[dict]) -> None:
    if len(baseline_examples) != len(candidate_examples):
        raise ValueError("Paired evaluations have different lengths")
    if any(
        left["question"] != right["question"]
        for left, right in zip(baseline_examples, candidate_examples)
    ):
        raise ValueError("Paired evaluations are not in the same question order")


def compare_openqa_results(
    baseline_metrics: dict,
    candidate_metrics: dict,
    *,
    candidate_name: str,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    baseline = baseline_metrics["sample_predictions"]
    candidate = candidate_metrics["sample_predictions"]
    _check_pairing(baseline, candidate)
    result = {
        "n_examples": len(baseline),
        "candidate": candidate_name,
        "baseline": "engram_only",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "metrics": {},
    }
    for offset, (metric, key) in enumerate((("em", "correct"), ("f1", "f1"))):
        stats = paired_bootstrap(
            np.asarray([float(row[key]) for row in baseline]),
            np.asarray([float(row[key]) for row in candidate]),
            seed=seed + offset,
            samples=bootstrap_samples,
        )
        result["metrics"][metric] = stats
    return result


def compare_truthfulqa_results(
    baseline_metrics: dict,
    candidate_metrics: dict,
    *,
    candidate_name: str,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    baseline = baseline_metrics["sample_examples"]
    candidate = candidate_metrics["sample_examples"]
    _check_pairing(baseline, candidate)
    result = {
        "n_examples": len(baseline),
        "candidate": candidate_name,
        "baseline": "engram_only",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "metrics": {},
    }
    metric_keys = (("mc1", "MC1"), ("mc2", "MC2"), ("mc3", "MC3"))
    for offset, (metric, key) in enumerate(metric_keys):
        stats = paired_bootstrap(
            np.asarray([float(row["metrics"][key]) for row in baseline]),
            np.asarray([float(row["metrics"][key]) for row in candidate]),
            seed=seed + offset,
            samples=bootstrap_samples,
        )
        result["metrics"][metric] = stats

    baseline_avg = np.asarray([
        np.mean([float(row["metrics"][key]) for _, key in metric_keys])
        for row in baseline
    ])
    candidate_avg = np.asarray([
        np.mean([float(row["metrics"][key]) for _, key in metric_keys])
        for row in candidate
    ])
    result["metrics"]["mc_avg"] = paired_bootstrap(
        baseline_avg,
        candidate_avg,
        seed=seed + len(metric_keys),
        samples=bootstrap_samples,
    )
    return result


def compare_task_results(
    task: str,
    baseline_metrics: dict,
    candidate_metrics: dict,
    *,
    candidate_name: str,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    compare_fn = compare_truthfulqa_results if task == "truthfulqa" else compare_openqa_results
    return compare_fn(
        baseline_metrics,
        candidate_metrics,
        candidate_name=candidate_name,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )


def compact_metrics(task: str, metrics: dict) -> dict:
    if task == "truthfulqa":
        keys = ("mc1", "mc2", "mc3", "mc_avg", "total", "elapsed_s")
    else:
        keys = ("em", "f1", "correct", "total", "elapsed_s")
    return {key: metrics[key] for key in keys}


def load_task(task: str, triviaqa_config: str):
    if task == "triviaqa":
        return load_triviaqa_examples(triviaqa_config)
    return TASK_LOADERS[task]()


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
    wrapper, set_canon_fn = setup_condition(args, "transferred", device, dtype)
    max_context = get_model_max_context(wrapper, args.max_context_length)
    tokenizer = wrapper.tokenizer

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

    evaluation = {mode: {} for mode in MODES}
    for mode in MODES:
        set_reader_mode(wrapper, mode)
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
            evaluation[mode][task] = {"metrics": metrics}
            print(
                f"{task} {mode}: " + json.dumps(compact_metrics(task, metrics)),
                flush=True,
            )

    comparisons = {}
    if args.bootstrap_samples:
        for task in args.tasks:
            baseline = evaluation["engram_only"][task]["metrics"]
            comparisons[task] = {
                mode: compare_task_results(
                    task,
                    baseline,
                    evaluation[mode][task]["metrics"],
                    candidate_name=mode,
                    seed=args.seed,
                    bootstrap_samples=args.bootstrap_samples,
                )
                for mode in ("both", "routed")
            }

    scalar_summary = {
        mode: {
            task: task_scalar_score(task, evaluation[mode][task]["metrics"])
            for task in args.tasks
        }
        for mode in MODES
    }
    for mode in MODES:
        scalar_summary[mode]["macro_average"] = float(
            np.mean([scalar_summary[mode][task] for task in args.tasks])
        )

    results = {
        "target_model": args.target_model,
        "seed": args.seed,
        "tasks": task_meta,
        "modes": list(MODES),
        "evaluation": evaluation,
        "paired_comparisons": comparisons,
        "bootstrap_completed": bool(args.bootstrap_samples),
        "scalar_summary": scalar_summary,
        "completed": True,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)
    if comparisons:
        print("PAIRED_COMPARISONS " + json.dumps(comparisons), flush=True)
    else:
        print("PAIRWISE_PREDICTIONS_READY_FOR_CPU_BOOTSTRAP", flush=True)
    print("ATHENA_DUAL_READER_CROSS_DATASET_PAIRED_EVAL_COMPLETE", flush=True)
    wrapper.cleanup()


if __name__ == "__main__":
    main()
