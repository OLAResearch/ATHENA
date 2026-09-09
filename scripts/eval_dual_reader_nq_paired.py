"""Paired full-NQ evaluation of a dual-reader residual checkpoint."""

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
    evaluate_openqa,
    get_model_max_context,
    load_nq_examples,
    setup_condition,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_reader_mode(wrapper, mode: str) -> None:
    adaptors = (
        list(wrapper.adaptor)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList)
        else [wrapper.adaptor]
    )
    for adaptor in adaptors:
        adaptor.set_dual_reader_mode(mode)


def compact_metrics(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in ("em", "f1", "correct", "total", "elapsed_s")
    }


def compare_reader_results(
    baseline_result: dict, candidate_result: dict, candidate_name: str, seed: int
) -> dict:
    baseline = baseline_result["nq"]["metrics"]["sample_predictions"]
    candidate = candidate_result["nq"]["metrics"]["sample_predictions"]
    if len(baseline) != len(candidate):
        raise ValueError("Paired NQ evaluations have different lengths")
    if any(
        left["question"] != right["question"]
        for left, right in zip(baseline, candidate)
    ):
        raise ValueError("Paired NQ evaluations are not in the same order")

    em_delta = np.asarray([
        float(right["correct"]) - float(left["correct"])
        for left, right in zip(baseline, candidate)
    ])
    f1_delta = np.asarray([
        float(right["f1"]) - float(left["f1"])
        for left, right in zip(baseline, candidate)
    ])
    rng = np.random.default_rng(seed)
    em_bootstrap = []
    f1_bootstrap = []
    for _ in range(2000):
        sample = rng.integers(0, len(em_delta), size=len(em_delta))
        em_bootstrap.append(float(em_delta[sample].mean()))
        f1_bootstrap.append(float(f1_delta[sample].mean()))

    def interval(values):
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "n_examples": len(baseline),
        "candidate": candidate_name,
        "baseline": "engram_only",
        "em_delta": float(em_delta.mean()),
        "f1_delta": float(f1_delta.mean()),
        "em_delta_95pct_paired_bootstrap": interval(em_bootstrap),
        "f1_delta_95pct_paired_bootstrap": interval(f1_bootstrap),
        "em_improved": int((em_delta > 0).sum()),
        "em_regressed": int((em_delta < 0).sum()),
        "em_tied": int((em_delta == 0).sum()),
        "f1_improved": int((f1_delta > 0).sum()),
        "f1_regressed": int((f1_delta < 0).sum()),
        "f1_tied": int((f1_delta == 0).sum()),
        "bootstrap_samples": 2000,
        "seed": seed,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    results_path = output_dir / "results.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    wrapper, set_canon_fn = setup_condition(args, "transferred", device, dtype)
    examples, dataset_meta = load_nq_examples()
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    max_context = get_model_max_context(wrapper, None)

    evaluation = {}
    adaptors = (
        list(wrapper.adaptor)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList)
        else [wrapper.adaptor]
    )
    # The generated branch is a residual corrector, not a standalone expert.
    # Evaluate the exact baseline, un-routed residual, and learned safe route.
    modes = ["engram_only", "both"]
    if all(getattr(adaptor, "router", None) is not None for adaptor in adaptors):
        modes.append("routed")
    for mode in modes:
        set_reader_mode(wrapper, mode)
        started = time.time()
        metrics = evaluate_openqa(
            wrapper=wrapper,
            set_canon_fn=set_canon_fn,
            tokenizer=wrapper.tokenizer,
            task_name="nq",
            examples=examples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            max_context_length=max_context,
            retain_all_predictions=True,
        )
        metrics["elapsed_s"] = time.time() - started
        evaluation[mode] = {"nq": {"metrics": metrics}}
        print(f"NQ {mode}: " + json.dumps(compact_metrics(metrics)))

    comparisons = {
        mode: compare_reader_results(
            evaluation["engram_only"], evaluation[mode], mode, args.seed
        )
        for mode in modes
        if mode != "engram_only"
    }
    results = {
        "dataset": dataset_meta,
        "n_examples": len(examples),
        "metric": "token_f1_primary_with_exact_match_secondary",
        "evaluation": evaluation,
        "paired_comparisons": comparisons,
        "completed": True,
    }
    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print("PAIRED_COMPARISONS " + json.dumps(comparisons))
    print("ATHENA_DUAL_READER_NQ_PAIRED_EVAL_COMPLETE")
    wrapper.cleanup()


if __name__ == "__main__":
    main()
