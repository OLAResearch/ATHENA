"""Compute paired cross-reader confidence intervals on CPU."""

import argparse
import json
from pathlib import Path

from scripts.eval_dual_reader_openqa_paired import compare_task_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-results", nargs="+", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def output_path(input_path: Path) -> Path:
    return input_path.with_name("results_with_bootstrap.json")


def build_comparisons(results: dict, *, samples: int, seed: int) -> dict:
    if results.get("completed") is not True:
        raise ValueError("GPU evaluation is not marked complete")
    if set(results.get("modes", [])) != {"engram_only", "both", "routed"}:
        raise ValueError("GPU evaluation does not contain all three reader modes")

    comparisons = {}
    for task in results["tasks"]:
        baseline = results["evaluation"]["engram_only"][task]["metrics"]
        comparisons[task] = {
            mode: compare_task_results(
                task,
                baseline,
                results["evaluation"][mode][task]["metrics"],
                candidate_name=mode,
                seed=seed,
                bootstrap_samples=samples,
            )
            for mode in ("both", "routed")
        }
    return comparisons


def main():
    args = parse_args()
    inputs = [path.resolve() for path in args.input_results]
    outputs = [output_path(path) for path in inputs]
    if len(set(inputs)) != len(inputs):
        raise ValueError("Input result paths must be unique")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {existing}")

    for input_path, destination in zip(inputs, outputs):
        with open(input_path) as handle:
            results = json.load(handle)
        comparisons = build_comparisons(
            results,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        results["paired_comparisons"] = comparisons
        results["bootstrap_completed"] = True
        results["bootstrap_samples"] = args.bootstrap_samples
        results["bootstrap_seed"] = args.seed
        with open(destination, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"BOOTSTRAP_COMPLETE {destination} " + json.dumps(comparisons), flush=True)

    print("ATHENA_DUAL_READER_CROSS_DATASET_BOOTSTRAP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
