"""Aggregate isolated one-threshold Yahoo sweep artifacts without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.3)


def _threshold_specs(raw: str) -> tuple[tuple[str, str], ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value < 0.0 for value in values):
        raise ValueError("thresholds must contain non-negative numbers")
    return tuple(
        (f"{value:.6g}", f"{value:.6g}".replace(".", "p"))
        for value in values
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    args = parser.parse_args()
    expected = _threshold_specs(args.thresholds)

    evaluations: dict[str, dict] = {}
    checkpoints = []
    for key, tag in expected:
        result_file = args.input_dir / f"threshold-{tag}" / "results.json"
        # Older array launchers retained the decimal suffix for integer values.
        if not result_file.is_file() and float(key).is_integer():
            result_file = args.input_dir / f"threshold-{tag}p0" / "results.json"
        if not result_file.is_file():
            raise FileNotFoundError(result_file)
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        if key not in summary:
            raise ValueError(f"missing threshold {key} in {result_file}")
        evaluations[key] = payload["evaluation"][key]
        checkpoints.append(payload.get("checkpoint"))

    if len({json.dumps(value, sort_keys=True) for value in checkpoints}) != 1:
        raise ValueError("threshold shards do not use the same checkpoint metadata")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "task": "yahoo",
        "protocol": "domain_conditional_pmi",
        "dataset": "yahoo_answers_topics/test",
        "dataset_size": 60000,
        "conditions": {"reader_mode": "tri_advantage_routed", "max_scale": 1.0},
        "thresholds": [float(key) for key, _ in expected],
        "labels_used_only_for_accuracy": True,
        "checkpoint": checkpoints[0],
        "evaluation": evaluations,
        "summary": {key: value["accuracy"] for key, value in evaluations.items()},
        "sharded_inference": True,
        "batch_size": 8,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "status.json").write_text(
        json.dumps({"stage": "complete", "summary": output["summary"]}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
