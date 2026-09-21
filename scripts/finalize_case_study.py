#!/usr/bin/env python3
"""Finalize a case study from an already completed inference artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _select_showcase(rows):
    by_source = {name: [] for name in ("E", "GE", "GH")}
    for row in rows:
        outputs = row["outputs"]
        router = outputs["router"]
        if outputs["base"]["correct"] or outputs["rag"]["correct"]:
            continue
        if not router["correct"]:
            continue
        source = router.get("route_trace", {}).get("argmax_source")
        prediction = str(router.get("prediction", ""))
        if source not in by_source or not prediction or "; ;" in prediction:
            continue
        by_source[source].append({**row, "showcase_basis": "router_correct_realised_argmax"})
    selected = []
    for source in ("E", "GE", "GH"):
        if by_source[source]:
            selected.append(by_source[source][0])
    for source in ("E", "GE", "GH"):
        for row in by_source[source][1:]:
            if len(selected) >= 3:
                break
            selected.append(row)
        if len(selected) >= 3:
            break
    grouped = {name: [] for name in ("E", "GE", "GH")}
    for row in selected:
        grouped[row["outputs"]["router"]["route_trace"]["argmax_source"]].append(row)
    grouped["router"] = selected[:3]
    return grouped


def _render_markdown(payload):
    lines = [
        "# MemoryAthena case study",
        "",
        "Post-hoc qualitative slice of HotpotQA distractor validation.",
        "Exactly three rows are shown where Base and RAG are wrong and the router answer is correct.",
        "RAG receives the dataset distractor documents in dataset order without supporting-fact labels.",
        "",
        "| Realised argmax | Question | Base | RAG | Fixed-path diagnostic | Router output | Gold |",
        "|---|---|---|---|---|---|---|",
    ]
    for source in ("E", "GE", "GH"):
        for row in payload["selected_showcase"].get(source, []):
            out = row["outputs"]
            question = row["question"].replace("|", "\\|")
            lines.append(
                f"| {source} | {question} | {out['base']['prediction']} | "
                f"{out['rag']['prediction']} | {out[source]['prediction']} | "
                f"{out['router']['prediction']} | {'; '.join(row['answers'])} |"
            )
    lines += [
        "",
        "The final router output is the method result. Fixed-path outputs are diagnostics;",
        "the realised argmax is reported separately and is not claimed to be a standalone-reader guarantee.",
        "Labels were used only after inference for scoring and post-hoc selection.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--source-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_results = Path(args.source_results)
    source_candidates = Path(args.source_candidates)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    payload = json.loads(source_results.read_text())
    rows = payload.get("candidates", [])
    print(
        json.dumps({
            "source_results": str(source_results),
            "source_exists": source_results.exists(),
            "n_candidates": len(rows),
            "router_correct": sum(bool(r.get("outputs", {}).get("router", {}).get("correct")) for r in rows),
            "base_rag_wrong_router_correct": sum(
                bool(r.get("outputs", {}).get("router", {}).get("correct"))
                and not r.get("outputs", {}).get("base", {}).get("correct")
                and not r.get("outputs", {}).get("rag", {}).get("correct")
                for r in rows
            ),
        }),
        flush=True,
    )
    selected = _select_showcase(payload["candidates"])
    if len(selected["router"]) != 3:
        raise RuntimeError(f"Expected exactly 3 router-correct rows, found {len(selected['router'])}")
    for row in selected["router"]:
        outputs = row["outputs"]
        assert not outputs["base"]["correct"]
        assert not outputs["rag"]["correct"]
        assert outputs["router"]["correct"]
        source = outputs["router"]["route_trace"]["argmax_source"]
        assert source in ("E", "GE", "GH")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_candidates, output_dir.parent / "candidates.json")
    payload["selected_showcase"] = selected
    payload["selected_counts"] = {name: len(values) for name, values in selected.items()}
    payload["final_showcase_requirement"] = (
        "Exactly three rows: Base and RAG incorrect, router correct, and realised argmax path correct."
    )
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "status.json").write_text(json.dumps({
        "stage": "complete",
        "n_candidates": len(payload["candidates"]),
        "selected_counts": payload["selected_counts"],
        "router_correct_examples": 3,
        "labels_used_for_routing": False,
    }, indent=2) + "\n")
    (output_dir / "case_study.md").write_text(_render_markdown(payload))
    print(json.dumps({"output_dir": str(output_dir), "selected_counts": payload["selected_counts"]}))


if __name__ == "__main__":
    main()
