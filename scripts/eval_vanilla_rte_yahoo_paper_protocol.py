"""Evaluate Vanilla Mistral on the paper-aligned RTE and Yahoo protocols.

The prompts and verbalizers are copied from the public kNN-Prompt task
loaders used by the MLP Memory comparison.  This runner is intentionally
baseline-only: no Engram memory, reader, adaptor, or downstream labels are
loaded during inference.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.eval_general_nlp_halueval import _example, evaluate_task
from scripts.eval_openqa import get_model_max_context, setup_condition


SOURCE_URL = "https://github.com/swj0419/kNN_prompt/tree/main/task_data"
RTE_PROMPT = " true or false?\n answer:"
YAHOO_CHOICES = [
    "society",
    "science",
    "health",
    "education",
    "computer",
    "sports",
    "business",
    "entertainment",
    "family",
    "politics",
]


def _load_rte(path: Path) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            # This preserves the original loader's test-time spacing exactly.
            context = f" {row['premise']}\n question: {row['hypothesis']} {RTE_PROMPT}"
            label = 0 if row["label"] == "entailment" else 1
            examples.append(_example(context, ["true", "false"], label, RTE_PROMPT))
    return examples


def _load_yahoo() -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("yahoo_answers_topics", split="test")
    examples = []
    for row in dataset:
        title = str(row.get("question_title") or "")
        content = str(row.get("question_content") or "")
        answer = str(row.get("best_answer") or "")
        context = f"title: {title} content: {content} answer: {answer} topic:"
        label = int(row["topic"] if "topic" in row else row["label"])
        examples.append(_example(context, YAHOO_CHOICES, label, "topic:"))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--rte-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-forward-sequences",
        type=int,
        default=10,
        help="Choice rows per forward pass; 10 is safe for one Yahoo example on one MI250X GCD.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_forward_sequences <= 0:
        parser.error("--batch-size and --max-forward-sequences must be positive")

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = {
        "rte": _load_rte(Path(args.rte_data)),
        "yahoo": _load_yahoo(),
    }
    print("Loaded task sizes:", {name: len(rows) for name, rows in tasks.items()}, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    baseline_args = SimpleNamespace(target_model=args.target_model, dual_reader_mode="auto")
    wrapper, canon = setup_condition(baseline_args, "baseline", device, dtype)
    max_context = get_model_max_context(wrapper, None)
    results = {}
    try:
        for task, examples in tasks.items():
            print(f"Evaluating {task} ({len(examples)} examples)...", flush=True)
            results[task] = evaluate_task(
                wrapper,
                canon,
                examples,
                device,
                max_context,
                pmi=True,
                batch_size=args.batch_size,
                max_forward_sequences=args.max_forward_sequences,
            )
            print(
                f"vanilla/{task}: {results[task]['correct']}/{results[task]['total']} "
                f"= {results[task]['accuracy']:.6f}",
                flush=True,
            )
    finally:
        cleanup = getattr(wrapper, "cleanup", None)
        if cleanup is not None:
            cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "method": "vanilla_mistral_baseline",
        "target_model": args.target_model,
        "tasks": ["rte", "yahoo"],
        "evaluation_split": {
            "rte": "kNN_prompt/task_data/rte/val.jsonl",
            "yahoo": "yahoo_answers_topics/test",
        },
        "source_url": SOURCE_URL,
        "protocol": "domain_conditional_pmi",
        "prompt_protocol": {
            "rte": "<premise>\\n question: <hypothesis>  true or false?\\n answer:",
            "rte_choices": ["true", "false"],
            "yahoo": "title: <title> content: <content> answer: <best_answer> topic:",
            "yahoo_choices": YAHOO_CHOICES,
        },
        "memory_loaded": False,
        "adaptor_loaded": False,
        "results": results,
    }
    result_file = output_dir / "results.json"
    result_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {result_file}", flush=True)


if __name__ == "__main__":
    main()
