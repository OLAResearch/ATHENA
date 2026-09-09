"""Recompute strict tri-memory oracles without rerunning model inference.

The model evaluation files contain all paired endpoint predictions.  This
post-processing pass deliberately writes a new file and never overwrites the
original evaluation or bootstrap artifact.  In particular, the seven-way
upper bound contains exactly the seven fixed subset endpoints and never the
learned subset Reader whose capture is being measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_dual_reader_openqa_paired import compare_task_results
from scripts.eval_openqa import task_scalar_score
from scripts.tri_memory_oracle import build_all_oracles, capture_ratio


READER_MODES = (
    "tri_reader_hard",
    "tri_reader_soft",
    "tri_reader_advantage",
    "tri_subset_reader_hard",
    "tri_subset_reader_soft",
)


def _recompute(path: Path, bootstrap_samples: int, seed: int) -> Path:
    with path.open() as handle:
        results = json.load(handle)
    if not results.get("completed"):
        raise ValueError(f"Incomplete evaluation: {path}")

    corrected_oracles = {}
    corrected_comparisons = {}
    for task in results["tasks"]:
        evaluation = results["evaluation"]
        corrected_oracles[task] = build_all_oracles(task, evaluation)
        baseline = evaluation["engram_baseline"][task]["metrics"]
        candidates = {
            mode: evaluation[mode][task]["metrics"]
            for mode in results.get("modes", [])
            if mode != "engram_baseline" and mode in evaluation
        }
        candidates.update(corrected_oracles[task])
        corrected_comparisons[task] = {
            name: compare_task_results(
                task,
                baseline,
                metrics,
                candidate_name=name,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            )
            for name, metrics in candidates.items()
        }

    scalar_summary = {}
    for mode in results["modes"]:
        scalar_summary[mode] = {
            task: task_scalar_score(
                task, results["evaluation"][mode][task]["metrics"]
            )
            for task in results["tasks"]
        }
        scalar_summary[mode]["macro_average"] = sum(
            scalar_summary[mode][task] for task in results["tasks"]
        ) / len(results["tasks"])

    oracle_scalar_summary = {}
    for oracle_name in next(iter(corrected_oracles.values())):
        oracle_scalar_summary[oracle_name] = {
            task: task_scalar_score(task, corrected_oracles[task][oracle_name])
            for task in results["tasks"]
        }
        oracle_scalar_summary[oracle_name]["macro_average"] = sum(
            oracle_scalar_summary[oracle_name][task] for task in results["tasks"]
        ) / len(results["tasks"])

    capture = {}
    source_capture = {}
    runtime_capture = {}
    for reader_name in READER_MODES:
        if reader_name not in scalar_summary:
            continue
        capture[reader_name] = {}
        source_capture[reader_name] = {}
        runtime_capture[reader_name] = {}
        for task in results["tasks"]:
            baseline = scalar_summary["engram_baseline"][task]
            reader = scalar_summary[reader_name][task]
            capture[reader_name][task] = capture_ratio(
                reader, baseline, oracle_scalar_summary["oracle_E_GE_GH"][task]
            )
            source_capture[reader_name][task] = capture_ratio(
                reader,
                baseline,
                oracle_scalar_summary["oracle_source_E_GE_GH"][task],
            )
            runtime_capture[reader_name][task] = capture_ratio(
                reader,
                baseline,
                oracle_scalar_summary["oracle_all_nonempty_subsets"][task],
            )
        for target, target_oracle in (
            (capture, "oracle_E_GE_GH"),
            (source_capture, "oracle_source_E_GE_GH"),
            (runtime_capture, "oracle_all_nonempty_subsets"),
        ):
            target[reader_name]["macro_average"] = capture_ratio(
                scalar_summary[reader_name]["macro_average"],
                scalar_summary["engram_baseline"]["macro_average"],
                oracle_scalar_summary[target_oracle]["macro_average"],
            )

    corrected = dict(results)
    corrected["oracles"] = corrected_oracles
    corrected["paired_comparisons"] = corrected_comparisons
    corrected["oracle_scalar_summary"] = oracle_scalar_summary
    corrected["scalar_summary"] = scalar_summary
    corrected["reader_capture_ratio"] = capture
    corrected["reader_capture_ratio_source_selection"] = source_capture
    corrected["reader_capture_ratio_all_nonempty_subsets"] = runtime_capture
    corrected["oracle_definition"] = (
        "strict_fixed_seven_endpoints_excluding_learned_subset_reader"
    )
    corrected["corrected_oracle_completed"] = True
    corrected["corrected_oracle_bootstrap_samples"] = bootstrap_samples
    corrected["corrected_oracle_seed"] = seed

    output = path.with_name("results_with_corrected_oracle.json")
    with output.open("w") as handle:
        json.dump(corrected, handle, indent=2)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    for raw_path in args.results:
        _recompute(Path(raw_path), args.bootstrap_samples, args.seed)
    print("ATHENA_TRI_MEMORY_CORRECTED_ORACLE_COMPLETE")


if __name__ == "__main__":
    main()
