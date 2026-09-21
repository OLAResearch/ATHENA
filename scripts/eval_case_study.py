#!/usr/bin/env python3
"""Run a label-free, paired HotpotQA case study for MemoryAthena.

The RAG condition receives the dataset's distractor context in its original
order.  It does not read supporting-fact annotations.  The final showcase is
selected only after inference, so the code reports the selection rule and all
candidate outputs rather than hiding failed candidates.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_dual_reader_openqa_paired import set_reader_mode
from scripts.eval_openqa import (
    build_openqa_prompt,
    exact_match,
    f1_score,
    greedy_generate,
    setup_condition,
    get_model_max_context,
)
from scripts.routing_proportion import RoutingProportionCollector, _latest_weights


SOURCE_MODES = {
    "E": "engram_only",
    "GE": "generated_from_engram_only",
    "GH": "generated_from_context_only",
    "router": "tri_advantage_routed",
}


class TraceCollector:
    """Keep realised router weights for generated answer positions."""

    def __init__(self):
        self.values = []

    def add_last_token(self, wrapper):
        weights = _latest_weights(wrapper)
        if weights is not None:
            self.values.append(weights[:, -1, :].detach().float().cpu().mean(dim=0).tolist())

    def report(self) -> dict:
        if not self.values:
            return {
                "source_order": ["E", "GE", "GH"],
                "positions": 0,
                "weighted_source_mass": {name: 0.0 for name in ("E", "GE", "GH")},
                "argmax_source_fraction": {name: 0.0 for name in ("E", "GE", "GH")},
                "argmax_source": None,
                "generated_admission_rate": None,
                "exact_e_fallback_rate": None,
                "mean_alpha": None,
            }
        collector = RoutingProportionCollector()
        collector.add_weights(torch.tensor(self.values).unsqueeze(0))
        report = collector.report()
        report["argmax_source"] = max(
            report["argmax_source_fraction"],
            key=report["argmax_source_fraction"].get,
        )
        return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", required=True)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalise_question(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def load_hotpotqa_with_context():
    from datasets import load_dataset

    errors = []
    for name, config in (("hotpotqa/hotpot_qa", "distractor"), ("hotpot_qa", "distractor")):
        try:
            dataset = load_dataset(name, config, split="validation", trust_remote_code=True)
            rows = {}
            for row in dataset:
                question = str(row["question"])
                context = row.get("context", [])
                rows[_normalise_question(question)] = {
                    "question": question,
                    "answer": str(row.get("answer", "")),
                    "context": context,
                }
            return rows, {"dataset": name, "config": config, "split": "validation"}
        except Exception as exc:
            errors.append(f"{name}/{config}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Unable to load HotpotQA distractor validation:\n" + "\n".join(errors))


def format_rag_context(context, max_chars=18000) -> str:
    blocks = []
    for index, item in enumerate(context or [], start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sentences = item
        title = str(title).strip()
        text = " ".join(str(sentence).strip() for sentence in sentences)
        blocks.append(f"[{index}] {title}: {text}")
    text = "\n".join(blocks)
    return text[:max_chars]


def build_rag_prompt(question: str, context) -> str:
    return (
        "Refer to the retrieved background documents and answer the question.\n"
        "Background:\n"
        f"{format_rag_context(context)}\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def _score(prediction: str, answers: list[str]) -> dict:
    em = any(exact_match(prediction, answer) for answer in answers)
    f1 = max((f1_score(prediction, answer)[0] for answer in answers), default=0.0)
    return {"correct": bool(em), "exact_match": float(em), "f1": float(f1)}


def _run_generation(wrapper, canon_fn, prompt, *, max_context, args, trace=None):
    raw = greedy_generate(
        wrapper,
        wrapper.tokenizer,
        prompt,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        canon_fn,
        max_new_tokens=args.max_new_tokens,
        max_context_length=max_context,
        official_tokenization=True,
        # Some Mistral generations begin with a formatting newline.  The
        # shared helper's historical stop-at-newline behavior would split at
        # that first newline and return an empty answer, which was especially
        # common for the longer RAG prompt.
        stop_at_newline=False,
        routing_collector=trace,
    )
    for line in str(raw).splitlines():
        line = line.strip()
        if line:
            return line
    return str(raw).strip()


def _condition_args(args):
    values = vars(args).copy()
    values["dual_reader_mode"] = "auto"
    return argparse.Namespace(**values)


def _run_case(candidate, base, joint, base_canon, joint_canon, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_context = get_model_max_context(base, args.max_context_length)
    question = candidate["question"]
    answers = candidate["answers"]
    raw_base = _run_generation(base, base_canon, build_openqa_prompt(question), max_context=max_context, args=args)
    raw_rag = _run_generation(base, base_canon, build_rag_prompt(question, candidate["context"]), max_context=max_context, args=args)
    outputs = {
        "base": {
            "prediction": raw_base,
            **_score(raw_base, answers),
            "prompt_type": "bare_question",
        },
        "rag": {
            "prediction": raw_rag,
            **_score(raw_rag, answers),
            "prompt_type": "HotpotQA distractor context in dataset order",
        },
    }
    for name, mode in SOURCE_MODES.items():
        set_reader_mode(joint, mode)
        trace = TraceCollector() if name == "router" else None
        raw = _run_generation(joint, joint_canon, build_openqa_prompt(question), max_context=max_context, args=args, trace=trace)
        outputs[name] = {
            "prediction": raw,
            **_score(raw, answers),
            "prompt_type": "bare_question_with_memory_path",
        }
        if trace is not None:
            outputs[name]["route_trace"] = trace.report()
    return outputs


def _select_showcase(rows):
    by_source = {name: [] for name in ("E", "GE", "GH")}
    for row in rows:
        router = row["outputs"]["router"]
        if row["outputs"]["base"]["correct"] or row["outputs"]["rag"]["correct"]:
            continue
        # A displayed row must be a genuine end-to-end router success.  In
        # particular, do not present a fixed-reader intervention as evidence
        # that the router selected that pathway.
        if not router["correct"]:
            continue
        argmax_source = router.get("route_trace", {}).get("argmax_source")
        if argmax_source not in by_source:
            continue
        prediction = str(router.get("prediction", ""))
        if not prediction or "; ;" in prediction:
            continue
        by_source[argmax_source].append({
            **row,
            "showcase_basis": "router_correct_realised_argmax",
        })
    # Prefer route diversity when it exists, but the scientific requirement is
    # three verified end-to-end successes, not an invented one-per-source quota.
    # The realised argmax is retained in every row so concentration on one
    # pathway remains visible rather than being hidden by post-processing.
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
    result = {name: [] for name in ("E", "GE", "GH")}
    for row in selected:
        result[row["outputs"]["router"]["route_trace"]["argmax_source"]].append(row)
    result["router"] = selected[:3]
    return result


def render_markdown(payload: dict) -> str:
    lines = [
        "# MemoryAthena case study",
        "",
        "This case study is a post-hoc qualitative slice of HotpotQA distractor validation.",
        "It contains exactly three end-to-end router successes, with the realised route",
        "reported for every row.",
        "The RAG prompt receives the dataset's distractor documents in their original order",
        "without supporting-fact annotations. Labels are used only after inference to score",
        "answers and select showcase rows; they are never used to configure the router.",
        "",
        "## Showcase rows",
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
                f"{out['router']['prediction']} | "
                f"{'; '.join(row['answers'])} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "Each displayed row requires the bare model and RAG to be incorrect, while the",
        "router's final answer is exactly correct. The fixed-path output is shown only as",
        "a diagnostic; the realised argmax is reported separately and is not presented as",
        "a standalone-reader guarantee. The complete machine-readable result",
        "retains every evaluated candidate, including failures, so the qualitative table is",
        "not a substitute for the aggregate benchmark metrics.",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    candidate_payload = json.loads(Path(args.candidate_json).read_text())
    dataset_rows, dataset_meta = load_hotpotqa_with_context()
    candidates = []
    for item in candidate_payload["candidates"]:
        row = dataset_rows.get(_normalise_question(item["question"]))
        if row is None:
            raise KeyError(f"Candidate missing from HotpotQA validation: {item['question']}")
        candidate = dict(item)
        candidate["context"] = row["context"]
        candidate["context_document_count"] = len(row["context"])
        candidates.append(candidate)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_args = _condition_args(args)
    base, base_canon = setup_condition(base_args, "baseline", device, dtype)
    joint, joint_canon = setup_condition(base_args, "transferred", device, dtype)
    rows = []
    try:
        for index, candidate in enumerate(candidates):
            print(f"case {index + 1}/{len(candidates)}: {candidate['question']}", flush=True)
            rows.append({
                **candidate,
                "outputs": _run_case(candidate, base, joint, base_canon, joint_canon, args),
            })
    finally:
        for wrapper in (base, joint):
            cleanup = getattr(wrapper, "cleanup", None)
            if cleanup is not None:
                cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected = _select_showcase(rows)
    if len(selected["router"]) != 3:
        raise RuntimeError(
            "Case study requires exactly three router-correct examples; "
            f"found {len(selected['router'])}"
        )
    result = {
        "status": "complete",
        "dataset": dataset_meta,
        "candidate_source": args.candidate_json,
        "selection_is_post_hoc": True,
        "labels_used_for_routing": False,
        "rag_protocol": {
            "context_source": "HotpotQA distractor validation context",
            "context_order": "dataset order",
            "supporting_fact_labels_used": False,
            "retriever": "none; fixed dataset distractor contexts",
        },
        "conditions": ["base", "rag", "E", "GE", "GH", "router"],
        "source_order": ["E", "GE", "GH"],
        "candidates": rows,
        "selected_showcase": selected,
        "selected_counts": {source: len(values) for source, values in selected.items()},
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "status.json").write_text(json.dumps({
        "stage": "complete",
        "n_candidates": len(rows),
        "selected_counts": result["selected_counts"],
        "labels_used_for_routing": False,
    }, indent=2) + "\n")
    (output_dir / "case_study.md").write_text(render_markdown(result))
    print(json.dumps({"output_dir": str(output_dir), "n_candidates": len(rows), "selected_counts": result["selected_counts"]}))


if __name__ == "__main__":
    main()
