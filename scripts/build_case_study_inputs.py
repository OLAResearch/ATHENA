#!/usr/bin/env python3
"""Select clean, post-hoc HotpotQA case-study candidates.

The selection uses an already completed evaluation only to decide which rows
are worth displaying.  It is not used by the router and is explicitly
recorded as post-hoc selection in the output artifact.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SOURCE_TO_MODE = {
    "E": "tri_E",
    "GE": "tri_GE",
    "GH": "tri_GH",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _looks_clean(prediction: dict) -> bool:
    raw = _clean(prediction.get("raw_prediction", prediction.get("prediction", "")))
    diagnostics = prediction.get("prediction_diagnostics", {})
    raw_diag = diagnostics.get("raw", {}) if isinstance(diagnostics, dict) else {}
    if not raw or len(raw) > 180:
        return False
    if raw_diag.get("repetition_count", 0) > 2:
        return False
    if "; ;" in raw or raw.count("Question:") > 0:
        return False
    return True


def _prediction_map(results: dict, mode: str) -> dict[str, dict]:
    task_payload = results["evaluation"][mode]["hotpotqa"]
    metrics = task_payload.get("metrics", task_payload)
    return {
        str(row["question"]): row
        for row in metrics.get("sample_predictions", [])
    }


def select_candidates(results: dict, max_per_source: int) -> list[dict]:
    base = _prediction_map(results, "engram_baseline")
    router = _prediction_map(results, "tri_reader_advantage")
    selected = []
    for source, mode in SOURCE_TO_MODE.items():
        target = _prediction_map(results, mode)
        rows = []
        for question, target_row in target.items():
            base_row = base.get(question)
            router_row = router.get(question)
            if not base_row or not router_row:
                continue
            if base_row.get("correct", False) or not target_row.get("correct", False):
                continue
            if not _looks_clean(target_row) or not _looks_clean(router_row):
                continue
            answers = [str(answer).strip() for answer in target_row.get("answers", []) if str(answer).strip()]
            if not answers:
                continue
            # Prefer concise, exact answers and avoid examples whose existing
            # target prediction contains a long explanation.
            target_text = _clean(target_row.get("prediction", ""))
            score = (
                0 if target_text.lower() in {answer.lower() for answer in answers} else 1,
                len(target_text),
                len(question),
            )
            rows.append((score, {
                "question": question,
                "answers": answers,
                "preferred_source": source,
                "selection_mode": mode,
                "selection_protocol": (
                    "post-hoc from completed HotpotQA predictions: base exact-match false "
                    "and preferred single-reader exact-match true; no labels are used by routing"
                ),
                "prior_base": {
                    "prediction": base_row.get("prediction", ""),
                    "correct": bool(base_row.get("correct", False)),
                    "f1": float(base_row.get("f1", 0.0)),
                },
                "prior_preferred_reader": {
                    "prediction": target_row.get("prediction", ""),
                    "correct": bool(target_row.get("correct", False)),
                    "f1": float(target_row.get("f1", 0.0)),
                },
                "prior_router": {
                    "prediction": router_row.get("prediction", ""),
                    "correct": bool(router_row.get("correct", False)),
                    "f1": float(router_row.get("f1", 0.0)),
                },
            }))
        rows.sort(key=lambda item: item[0])
        selected.extend(row for _, row in rows[:max_per_source])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-source", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_per_source < 1:
        raise ValueError("--max-per-source must be positive")
    with open(args.formal_results) as handle:
        results = json.load(handle)
    candidates = select_candidates(results, args.max_per_source)
    if not candidates:
        raise RuntimeError("No clean post-hoc case-study candidates were found")
    payload = {
        "dataset": "HotpotQA distractor validation",
        "source_results": args.formal_results,
        "selection_is_post_hoc": True,
        "labels_used_for_router": False,
        "selection_note": (
            "Answer labels are used only to filter an already completed result table "
            "for presentation; the new inference run does not use labels for routing."
        ),
        "candidates": candidates,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    counts = {source: sum(row["preferred_source"] == source for row in candidates) for source in ("E", "GE", "GH")}
    print(json.dumps({"output": str(output), "n_candidates": len(candidates), "by_source": counts}))


if __name__ == "__main__":
    main()
