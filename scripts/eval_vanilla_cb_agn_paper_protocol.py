"""Evaluate Vanilla Mistral on the paper-aligned CB and AGN protocols.

The public kNN-Prompt loaders used by the MLP Memory comparison use cloze
prompts that differ from ATHENA's task-name prompts.  This runner keeps the
two remaining tasks isolated and baseline-only: no memory, adaptor, training,
or downstream labels are used during inference.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.eval_general_nlp_halueval import evaluate_task
from scripts.eval_openqa import get_model_max_context, setup_condition


SOURCE_URL = "https://github.com/swj0419/kNN_prompt/tree/main/task_data"
# Keep the leading blank in the verbalizers.  The public kNN-Prompt loader
# concatenates ``premise + hypothesis`` directly; using a generic helper that
# appends a blank to the premise creates a two-blank prompt and changes the
# next-token distribution substantially on CB.
CB_CHOICES = [" true", " false", " neither"]
AGN_CHOICES = [" world", " sports", " business", " science"]


def _cloze_example(context: str, choices: list[str], label: int, domain_context: str) -> dict:
    return {
        "context": context,
        "domain_context": domain_context,
        "choices": choices,
        "label": int(label),
    }


def _load_cb(path: Path) -> list[dict]:
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = label_map[row["label"]]
            context = (
                f' question: Given that "{row["premise"]}" Is '
                f'"{row["hypothesis"]}" true, false, or neither?\n answer:'
            )
            examples.append(_cloze_example(context, CB_CHOICES, label, " the answer is:"))
    return examples


def _load_agn(path: Path) -> list[dict]:
    examples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row["Class Index"]) - 1
            context = f'{row["Title"]} \n {row["Description"]} topic:'
            examples.append(_cloze_example(context, AGN_CHOICES, label, " topic:"))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--cb-data", required=True)
    parser.add_argument("--agn-data", required=True)
    parser.add_argument("--tasks", nargs="+", choices=("cb", "agn"), default=["cb", "agn"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-forward-sequences", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_forward_sequences <= 0:
        parser.error("batch and forward sequence sizes must be positive")

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_loaders = {
        "cb": lambda: _load_cb(Path(args.cb_data)),
        "agn": lambda: _load_agn(Path(args.agn_data)),
    }
    tasks = {name: task_loaders[name]() for name in args.tasks}
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
        "tasks": list(args.tasks),
        "evaluation_split": {
            "cb": "kNN_prompt/task_data/cb/dev.jsonl",
            "agn": "kNN_prompt/task_data/agn/test.csv",
        },
        "source_url": SOURCE_URL,
        "protocol": "domain_conditional_pmi",
        "prompt_protocol": {
            "cb": 'question: Given that "<premise>" Is "<hypothesis>" true, false, or neither?\\n answer:',
            "cb_choices": CB_CHOICES,
            "agn": "<title> \\n <description> topic:",
            "agn_choices": AGN_CHOICES,
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
