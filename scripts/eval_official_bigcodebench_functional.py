#!/usr/bin/env python3
"""Run the official BigCodeBench local instruct/full/pass@1 evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parallel", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not args.samples.is_file():
        raise FileNotFoundError(args.samples)
    args.output.mkdir(parents=True, exist_ok=False)

    from bigcodebench.evaluate import evaluate

    official_samples = args.output / "samples.jsonl"
    official_samples.write_text(args.samples.read_text())
    evaluate(
        split="instruct",
        subset="full",
        samples=str(official_samples),
        execution="local",
        pass_k="1",
        save_pass_rate=True,
        calibrated=False,
        parallel=args.parallel,
        min_time_limit=1,
    )
    pass_path = Path(str(official_samples).replace(".jsonl", "_pass_at_k.json"))
    eval_path = Path(str(official_samples).replace(".jsonl", "_eval_results.json"))
    if not pass_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError("Official BigCodeBench evaluator did not emit expected outputs")
    pass_data = json.loads(pass_path.read_text())
    summary = {
        "completed": True,
        "dataset": "bigcode/bigcodebench",
        "dataset_version": "v0.1.4",
        "mode": "instruct",
        "subset": "full",
        "pass_k": "1",
        "accuracy": float(pass_data["pass@1"]),
        "official_pass_at_k": str(pass_path),
        "official_eval_results": str(eval_path),
        "groundtruth_pass_rate": pass_data.get("gt_pass_rate"),
        "failed_groundtruth_tasks": pass_data.get("failed_tasks", []),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("OFFICIAL_BIGCODEBENCH_FUNCTIONAL_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
