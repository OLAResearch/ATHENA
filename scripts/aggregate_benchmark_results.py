"""Aggregate ATHENA/MemGen benchmark result JSON files into one table.

The evaluator outputs intentionally use different task-facing metrics:
OpenQA reports F1/EM, code reports functional Pass@k, and ALFWorld reports
episode SR (and GC-SR when the environment exposes it).  This utility keeps
those metrics in separate columns and never substitutes a lexical code score
for a functional Pass@k score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_ORDER = (
    "alfworld",
    "triviaqa",
    "popqa",
    "kodcode",
    "bigcodebench",
    "gpqa",
    "gsm8k",
    "math",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_table"))
    return parser.parse_args()


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _metric(metrics: dict[str, Any], *keys: str) -> Any:
    value = _first(metrics, *keys)
    if value is not None:
        return value
    nested = metrics.get("pass_at_k") or metrics.get("pass@k")
    if isinstance(nested, dict):
        return _first(nested, *keys)
    return None


def _method(path: Path, data: dict[str, Any]) -> str:
    text = "/".join(path.parts).lower()
    mode = str(data.get("dual_reader_mode") or "").lower()
    condition = str(data.get("condition") or "").lower()
    if "cot" in text:
        return "cot"
    if mode == "generated_only" or "generated_memory" in text or "genmem" in text:
        return "generated-memory-only"
    if condition == "baseline" or "baseline" in text or "vanilla" in text:
        return "vanilla"
    if mode == "both" or condition in {"transferred", "both"}:
        return "both"
    if mode == "engram_only" or "engram_only" in text:
        return "engram-only"
    return condition or "unknown"


def _model(data: dict[str, Any], path: Path) -> str:
    model = data.get("target_model") or data.get("model")
    if model:
        return str(model)
    for part in path.parts:
        low = part.lower()
        if low in {"mistral-7b", "smollm3-3b", "qwen3-8b"}:
            return part
    return "unknown"


def _count(task_data: dict[str, Any], metrics: dict[str, Any]) -> int | None:
    value = _first(task_data, "evaluated_count", "full_count", "n_examples")
    if value is None:
        value = _first(metrics, "total", "n_examples")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_metrics(task: str, task_data: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Fill task-native fields that older result writers omitted."""
    normalized = dict(metrics)
    if task == "alfworld" and normalized.get("success_rate") is None:
        successes = normalized.get("successful_episodes")
        total = normalized.get("n_examples")
        if successes is not None and total:
            normalized["success_rate"] = float(successes) / float(total)
    return normalized


def _row(task: str, task_data: dict[str, Any], metrics: dict[str, Any], path: Path, data: dict[str, Any]) -> dict[str, Any]:
    metrics = _normalize_metrics(task, task_data, metrics)
    return {
        "model": _model(data, path),
        "method": _method(path, data),
        "task": task,
        "n": _count(task_data, metrics),
        "acc": _metric(metrics, "acc", "accuracy"),
        "em": _metric(metrics, "em", "exact_match"),
        "f1": _metric(metrics, "f1"),
        "pass@1": _metric(metrics, "pass@1", "pass_at_1"),
        "pass@5": _metric(metrics, "pass@5", "pass_at_5"),
        "pass@10": _metric(metrics, "pass@10", "pass_at_10"),
        "sr": _metric(metrics, "success_rate", "sr"),
        "gc_sr": _metric(metrics, "goal_condition_success_rate", "gc_sr"),
        "source": str(path),
    }


def collect_file(path: Path, root: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    rel = path.relative_to(root)
    rows: list[dict[str, Any]] = []

    if data.get("task") == "alfworld" and isinstance(data.get("metrics"), dict):
        rows.append(_row("alfworld", data, data["metrics"], rel, data))
        return rows

    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        return rows
    for task, task_data in tasks.items():
        if not isinstance(task_data, dict):
            continue
        metrics = task_data.get("metrics")
        if not isinstance(metrics, dict):
            # Older OpenQA output stores metrics under the condition name.
            conditions = data.get("conditions") or []
            condition = conditions[0] if conditions else None
            if condition and isinstance(task_data.get(condition), dict):
                metrics = task_data[condition]
            else:
                metrics = next(
                    (value for value in task_data.values() if isinstance(value, dict) and "f1" in value),
                    None,
                )
        if isinstance(metrics, dict):
            rows.append(_row(task, task_data, metrics, rel, data))
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if 0 <= value <= 1 else f"{value:.4f}"
    return str(value)


def write_outputs(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (TASK_ORDER.index(row["task"]) if row["task"] in TASK_ORDER else 99, row["model"], row["method"], row["source"]))
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    columns = ["model", "method", "task", "n", "acc", "em", "f1", "pass@1", "pass@5", "pass@10", "sr", "gc_sr", "source"]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = ["# Benchmark results", "", header, separator]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[column]) for column in columns) + " |")
    (output / "table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.root.rglob("*.json")):
        rows.extend(collect_file(path, args.root))
    write_outputs(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
