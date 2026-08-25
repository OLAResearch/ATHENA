#!/usr/bin/env python3
"""Merge the canonical TriviaQA OpenQA result with static-task results.

TriviaQA deliberately comes from ``eval_openqa.py`` so it retains the
historical validation split, prompt, tokenization, and EM/F1 implementation.
PopQA and GPQA continue to use ``eval_memgen_tasks.py``.  This adapter keeps
the result schema consumed by the existing benchmark-table scripts.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triviaqa-results", required=True)
    parser.add_argument("--other-results", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--reasoning-mode", choices=["vanilla", "cot"], default="vanilla")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def build_triviaqa_result(data: dict) -> dict:
    task = data["tasks"]["triviaqa"]
    condition = task.get("baseline")
    if condition is None:
        raise KeyError("TriviaQA OpenQA result does not contain the baseline condition")
    total = int(task["n_examples"])
    if int(condition.get("total", total)) != total:
        raise ValueError(
            f"TriviaQA count mismatch: metadata={total}, metrics={condition.get('total')}"
        )
    return {
        "dataset": task["dataset"],
        "full_count": total,
        "evaluated_count": total,
        "slice_start": 0,
        "slice_end": total,
        "metrics": {
            # Older canonical eval_openqa JSONs recorded only EM/F1.  For
            # OpenQA, exact-match accuracy is EM, so accept both schemas.
            "acc": condition.get("acc", condition["em"]),
            "em": condition["em"],
            "f1": condition["f1"],
            "correct": condition.get("correct", 0),
            "total": total,
            "sample_predictions": condition.get("sample_predictions", []),
            "metric_note": "EM/F1 and exact-match accuracy use eval_openqa.py",
        },
    }


def main():
    args = parse_args()
    triviaqa = load_json(args.triviaqa_results)
    other = load_json(args.other_results)

    if triviaqa.get("conditions") != ["baseline"]:
        raise ValueError(f"Expected only the baseline OpenQA condition, got {triviaqa.get('conditions')}")
    if set(other.get("tasks", {})) != {"popqa", "gpqa"}:
        raise ValueError(f"Expected PopQA and GPQA results, got {set(other.get('tasks', {}))}")
    if other.get("condition") != "baseline":
        raise ValueError(f"Expected baseline static results, got {other.get('condition')}")

    final = {
        "target_model": args.target_model,
        "condition": "baseline",
        "architecture": "vanilla_backbone",
        "dual_reader_mode": None,
        "reasoning_mode": args.reasoning_mode,
        "protocols": {
            "triviaqa": "eval_openqa.py",
            "popqa": "eval_memgen_tasks.py",
            "gpqa": "eval_memgen_tasks.py",
        },
        "tasks": {
            "triviaqa": build_triviaqa_result(triviaqa),
            "popqa": other["tasks"]["popqa"],
            "gpqa": other["tasks"]["gpqa"],
        },
        "completed": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(final, handle, indent=2)
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
