"""Validate full tri-reader evaluation artifacts without importing GPU libraries."""

import argparse
import json
import math
from pathlib import Path


TASK_COUNTS = {"nq": 3609, "webqa": 2032, "triviaqa": 17944, "truthfulqa": 817, "hotpotqa": 7405}
EXPECTED_MODES = {
    "engram_baseline", "tri_E", "tri_GE", "tri_GH", "tri_E_GE", "tri_E_GH",
    "tri_GE_GH", "tri_reader_hard", "tri_reader_soft", "tri_reader_advantage",
    "tri_safe_routed", "tri_subset_reader_hard", "tri_subset_reader_soft",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(results, task):
    count = TASK_COUNTS[task]
    require(results.get("completed") is True, "Evaluation is not complete")
    require(results.get("bootstrap_completed") is False, "Expected evaluation before bootstrap")
    require(set(results["tasks"]) == {task}, "Unexpected task set")
    require(results["tasks"][task]["n_examples"] == count, "Incomplete dataset")
    require(set(results["evaluation"]) == EXPECTED_MODES, "Expected all 13 tri-reader modes")
    require(set(results["modes"]) == EXPECTED_MODES and len(results["modes"]) == 13,
            "Mode metadata does not match the 13 evaluated modes")
    keys = ("mc1", "mc2", "mc3", "mc_avg") if task == "truthfulqa" else ("em", "f1")
    sample_key = "sample_examples" if task == "truthfulqa" else "sample_predictions"
    summary = {}
    for mode in sorted(EXPECTED_MODES):
        require(set(results["evaluation"][mode]) == {task}, f"{mode}: unexpected tasks")
        metrics = results["evaluation"][mode][task]["metrics"]
        require(metrics["total"] == count, f"{mode}: incomplete metric count")
        require(len(metrics[sample_key]) == count, f"{mode}: incomplete saved predictions")
        require(all(math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1 for key in keys),
                f"{mode}: invalid metrics")
        require(math.isfinite(metrics["elapsed_s"]) and metrics["elapsed_s"] >= 0,
                f"{mode}: invalid elapsed time")
        summary[mode] = {key: metrics[key] for key in (*keys, "total", "elapsed_s")}
    return {task: summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("task", choices=TASK_COUNTS)
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.results.read_text()), args.task)))
    print(f"ATHENA_LUMI_FAIR_JOINT_EVAL_COMPLETE {args.task}")


if __name__ == "__main__":
    main()
