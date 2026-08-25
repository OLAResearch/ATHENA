"""Merge full-data code-generation shards without changing their samples.

Each shard is produced by ``eval_memgen_tasks.py`` with ``--start-index`` and
``--max-examples``.  This utility concatenates the JSONL records in official
split order and writes a single result envelope that can be passed to the
existing isolated functional evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _merge_task(task: str, shard_dirs: list[Path], output_dir: Path) -> dict:
    ordered = []
    for shard_dir in shard_dirs:
        result = _read_json(shard_dir / "results.json")
        item = result["tasks"][task]
        ordered.append((int(item["slice_start"]), int(item["slice_end"]), result, item))
    ordered.sort(key=lambda value: value[0])
    if not ordered or ordered[0][0] != 0:
        raise ValueError(f"{task}: shards must start at index 0")

    full_count = int(ordered[0][3]["full_count"])
    cursor = 0
    records = []
    for start, end, _result, item in ordered:
        if start != cursor or end <= start or int(item["full_count"]) != full_count:
            raise ValueError(f"{task}: non-contiguous or inconsistent shard {start}:{end}")
        sample_path = Path(item["metrics"]["samples_file"])
        if not sample_path.is_absolute():
            sample_path = shard_dirs[0] / sample_path
        if not sample_path.is_file():
            sample_path = next(
                (shard_dir / sample_path.name for shard_dir in shard_dirs if (shard_dir / sample_path.name).is_file()),
                sample_path,
            )
        with sample_path.open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
        cursor = end
    if cursor != full_count or len(records) != full_count:
        raise ValueError(f"{task}: merged {len(records)} records, expected {full_count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"{task}_samples.jsonl"
    with samples_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    total = sum(int(item["evaluated_count"]) for *_unused, item in ordered)
    weighted_em = sum(float(item["metrics"]["em"]) * int(item["evaluated_count"]) for *_unused, item in ordered) / total
    weighted_f1 = sum(float(item["metrics"]["f1"]) * int(item["evaluated_count"]) for *_unused, item in ordered) / total
    num_samples = {int(item["metrics"].get("num_samples", 1)) for *_unused, item in ordered}
    if len(num_samples) != 1:
        raise ValueError(f"{task}: shards disagree on num_samples: {num_samples}")
    return {
        "dataset": ordered[0][2]["tasks"][task]["dataset"],
        "full_count": full_count,
        "evaluated_count": full_count,
        "slice_start": 0,
        "slice_end": full_count,
        "metrics": {
            "acc": None,
            "em": weighted_em,
            "f1": weighted_f1,
            "metric_note": "EM/F1 are lexical diagnostics; acc awaits isolated functional execution",
            "total": full_count,
            "num_samples": num_samples.pop(),
            "pass_k": [1, 5, 10],
            "samples_file": str(samples_path),
        },
    }


def merge(task_dirs: dict[str, list[Path]], output_dir: Path) -> Path:
    merged_tasks = {
        task: _merge_task(task, dirs, output_dir)
        for task, dirs in task_dirs.items()
    }
    first = next(iter(merged_tasks.values()))
    envelope = {
        "target_model": "mistralai/Mistral-7B-v0.3",
        "condition": "baseline",
        "architecture": "vanilla_backbone",
        "dual_reader_mode": None,
        "reasoning_mode": "vanilla",
        "tasks": merged_tasks,
        "completed": True,
        "merged_from_shards": True,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(envelope, indent=2) + "\n")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kodcode-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--bigcodebench-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path = merge(
        {"kodcode": args.kodcode_shards, "bigcodebench": args.bigcodebench_shards},
        args.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
