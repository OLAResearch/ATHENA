"""Build Table 3 aggregates and small-sample paired tests from real JSONs.

The script deliberately keeps two inferential units separate:
* seed-level tests use paired seeds and are only emitted when both rows have
  the same seed set; and
* single-run comparisons against the reported Base use an exact sign-flip
  permutation over the five tasks.  That is a task-level diagnostic, not a
  substitute for independent model seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


TASKS = ("nq", "webqa", "triviaqa", "truthfulqa", "hotpotqa")


def _payload_for_task(data: dict, task: str, mode: str | None) -> dict:
    evaluation = data.get("evaluation", {})
    if mode and mode in evaluation:
        evaluation = evaluation[mode]
    elif task not in evaluation and len(evaluation) == 1:
        evaluation = next(iter(evaluation.values()))
    payload = evaluation.get(task, evaluation.get("metrics", evaluation))
    if isinstance(payload, dict) and "metrics" in payload:
        payload = payload["metrics"]
    return payload


def read_task_value(path: str, task: str, mode: str | None) -> float:
    data = json.loads(Path(path).read_text())
    metrics = _payload_for_task(data, task, mode)
    key = "mc_avg" if task == "truthfulqa" else "f1"
    if key not in metrics:
        raise KeyError(f"{path}: missing metrics.{key} for {task}")
    value = float(metrics[key])
    return value * 100.0 if value <= 1.0 else value


def exact_sign_flip_pvalue(deltas: list[float]) -> float:
    """Two-sided exact sign-flip p-value for five paired task deltas."""
    if not deltas:
        return float("nan")
    observed = abs(sum(deltas) / len(deltas))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        values.append(abs(sum(s * d for s, d in zip(signs, deltas)) / len(deltas)))
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def mean_std(values: list[float]) -> dict:
    result = {"n": len(values), "mean": statistics.mean(values) if values else None}
    result["std"] = statistics.stdev(values) if len(values) > 1 else None
    return result


def seed_tests(left: dict[int, float], right: dict[int, float]) -> dict:
    seeds = sorted(set(left) & set(right))
    diffs = [left[s] - right[s] for s in seeds]
    out = {"seeds": seeds, "n": len(seeds), "mean_difference": statistics.mean(diffs) if diffs else None}
    if len(diffs) < 2:
        out.update({"paired_t_p": None, "wilcoxon_p": None})
        return out
    try:
        from scipy.stats import ttest_rel, wilcoxon

        lvals = [left[s] for s in seeds]
        rvals = [right[s] for s in seeds]
        out["paired_t_p"] = float(ttest_rel(lvals, rvals).pvalue)
        out["wilcoxon_p"] = float(wilcoxon(lvals, rvals, zero_method="wilcox", alternative="two-sided").pvalue)
    except (ImportError, ValueError):
        out.update({"paired_t_p": None, "wilcoxon_p": None})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="JSON list of row specifications")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    rows = []
    by_condition: dict[str, dict[int, float]] = {}
    base = None
    for row in spec:
        values = {
            task: read_task_value(row["task_files"][task], task, row.get("mode"))
            for task in TASKS
        }
        average = statistics.mean(values.values())
        record = {"condition": row["condition"], "seed": row.get("seed"), "task_values": values, "average": average}
        rows.append(record)
        if row.get("seed") is not None:
            by_condition.setdefault(row["condition"], {})[int(row["seed"])] = average
        if row.get("is_base"):
            base = values

    tests = {}
    if base is not None:
        for row in rows:
            deltas = [row["task_values"][task] - base[task] for task in TASKS]
            tests[row["condition"]] = {
                "mean_delta": statistics.mean(deltas),
                "exact_task_sign_flip_p": exact_sign_flip_pvalue(deltas),
                "deltas": dict(zip(TASKS, deltas)),
            }
    seed_summary = {condition: mean_std(list(values.values())) for condition, values in by_condition.items()}
    seed_pairwise = {}
    conditions = sorted(by_condition)
    for left, right in itertools.combinations(conditions, 2):
        seed_pairwise[f"{left}__vs__{right}"] = seed_tests(by_condition[left], by_condition[right])
    output = {"tasks": TASKS, "rows": rows, "single_run_task_tests_vs_base": tests, "seed_summary": seed_summary, "seed_pairwise_tests": seed_pairwise}
    Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
