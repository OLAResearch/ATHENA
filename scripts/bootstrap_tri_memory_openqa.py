"""Add paired bootstrap intervals to tri-memory evaluation results."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_dual_reader_openqa_paired import compare_task_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    for raw_path in args.results:
        path = Path(raw_path)
        with open(path) as handle:
            results = json.load(handle)
        if not results.get("completed"):
            raise ValueError(f"Incomplete evaluation: {path}")
        comparisons = {}
        for task in results["tasks"]:
            baseline = results["evaluation"]["engram_baseline"][task]["metrics"]
            candidates = {
                mode: results["evaluation"][mode][task]["metrics"]
                for mode in results.get("modes", [])
                if mode != "engram_baseline" and mode in results["evaluation"]
            }
            candidates.update(results["oracles"][task])
            comparisons[task] = {
                name: compare_task_results(
                    task,
                    baseline,
                    metrics,
                    candidate_name=name,
                    seed=args.seed,
                    bootstrap_samples=args.bootstrap_samples,
                )
                for name, metrics in candidates.items()
            }
        results["paired_comparisons"] = comparisons
        results["bootstrap_completed"] = True
        results["bootstrap_samples"] = args.bootstrap_samples
        output = path.with_name("results_with_bootstrap.json")
        with open(output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Wrote {output}")
    print("ATHENA_TRI_MEMORY_BOOTSTRAP_COMPLETE")


if __name__ == "__main__":
    main()
