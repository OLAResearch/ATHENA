#!/usr/bin/env python3
"""Merge deterministic KodCode generation shards without touching baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Directory names are lexicographically ordered, so shard-1334-2000
    # sorts before shard-667-1334.  Order by the recorded slice offset
    # instead; this also validates the payload rather than trusting names.
    shard_files = sorted(
        args.shard_root.glob("shard-*/results.json"),
        key=lambda path: int(
            json.loads(path.read_text())["tasks"]["kodcode"]["slice_start"]
        ),
    )
    if not shard_files:
        raise FileNotFoundError(f"No shard results under {args.shard_root}")

    merged_rows: list[dict] = []
    shard_results: list[tuple[int, int, dict]] = []
    expected_start = 0
    for result_path in shard_files:
        payload = json.loads(result_path.read_text())
        if payload.get("completed") is not True:
            raise RuntimeError(f"Incomplete shard: {result_path}")
        task = payload.get("tasks", {}).get("kodcode")
        if not task:
            raise RuntimeError(f"Shard lacks kodcode task: {result_path}")
        start = int(task["slice_start"])
        end = int(task["slice_end"])
        if start != expected_start or end <= start:
            raise RuntimeError(
                f"Non-contiguous KodCode shard {result_path}: "
                f"expected start {expected_start}, got [{start}:{end}]"
            )
        sample_path = Path(task["metrics"]["samples_file"])
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        rows = [json.loads(line) for line in sample_path.read_text().splitlines() if line]
        if len(rows) != end - start:
            raise RuntimeError(
                f"Shard row count mismatch in {sample_path}: "
                f"expected {end - start}, got {len(rows)}"
            )
        merged_rows.extend(rows)
        shard_results.append((start, end, task))
        expected_start = end

    full_count = int(shard_results[0][2]["full_count"])
    if expected_start != full_count or len(merged_rows) != full_count:
        raise RuntimeError(
            f"Expected complete KodCode coverage [0:{full_count}], "
            f"got [0:{expected_start}] with {len(merged_rows)} rows"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "kodcode_samples.jsonl"
    with samples_path.open("w") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row) + "\n")

    total = sum(int(task["evaluated_count"]) for _, _, task in shard_results)
    em = sum(float(task["metrics"]["em"]) * int(task["evaluated_count"]) for _, _, task in shard_results) / total
    f1 = sum(float(task["metrics"]["f1"]) * int(task["evaluated_count"]) for _, _, task in shard_results) / total
    first_task = shard_results[0][2]
    final = {
        "target_model": "mistralai/Mistral-7B-v0.3",
        "condition": "transferred",
        "architecture": "generated_memory+engram+dual_reader",
        "dual_reader_mode": "tri_advantage_routed",
        "reasoning_mode": "vanilla",
        "tasks": {
            "kodcode": {
                "dataset": first_task["dataset"],
                "full_count": full_count,
                "evaluated_count": total,
                "slice_start": 0,
                "slice_end": full_count,
                "metrics": {
                    "acc": None,
                    "em": em,
                    "f1": f1,
                    "metric_note": "EM/F1 are lexical diagnostics; acc awaits isolated functional execution",
                    "total": total,
                    "num_samples": 1,
                    "pass_k": [1],
                    "samples_file": str(samples_path),
                },
            }
        },
        "completed": True,
        "merged_from": [str(path) for path in shard_files],
    }
    (args.output_dir / "results.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"completed": True, "total": total, "em": em, "f1": f1}))


if __name__ == "__main__":
    main()
