"""Evaluate bare Mistral on the paper-aligned general-NLP prompt protocol.

The MLP Memory table follows the kNN-Prompt-style cloze prompts rather than
the task-name prompts used by the first ATHENA pilot.  This runner keeps the
protocol isolated from the existing tri-reader evaluator and records the
exact source files, prompts, verbalizers, and split sizes in its result.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.eval_general_nlp_halueval import _example, evaluate_task
from scripts.eval_openqa import get_model_max_context, setup_condition


TASKS = ("sst2", "mr", "cr", "rt", "hyp")
SOURCE_URL = "https://github.com/swj0419/kNN_prompt/tree/main/task_data"


def _load_sentiment_csv(path: Path, choices: list[str]) -> list[dict]:
    examples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row["label"])
            examples.append(
                _example(
                    f"{row['text']} It was",
                    choices,
                    label,
                    "It was",
                )
            )
    return examples


def _load_sst2(path: Path) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw_label, sentence = line.rstrip("\n").split("\t", 1)
            # The source SST-2 dev file stores 3/4/5 sentiment scores.  The
            # original kNN-Prompt loader removes the neutral score (3) and
            # maps 4/5 to positive and 2/1 to negative.
            score = int(raw_label[-1]) - 3
            if score == 0:
                continue
            label = 1 if score > 0 else 0
            examples.append(
                _example(
                    f"{sentence} It was",
                    ["terrible", "great"],
                    label,
                    "It was",
                )
            )
    return examples


def _load_rt(path: Path) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = 1 if row["output"].strip().lower() == "positive" else 0
            examples.append(
                _example(
                    f"{row['input']} It was",
                    ["terrible", "great"],
                    label,
                    "It was",
                )
            )
    return examples


def _load_hyp(path: Path) -> list[dict]:
    examples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            examples.append(
                _example(
                    f"{row['text']}\n neutral or partisan? Answer:",
                    ["neutral", "partisan"],
                    int(row["label"]),
                    "\n neutral or partisan? Answer:",
                )
            )
    return examples


def load_tasks(
    task_data_dir: Path,
    *,
    cr_choices: tuple[str, str] = ("terrible", "great"),
) -> dict[str, list[dict]]:
    """Load the paper-aligned suite with an explicit CR verbalizer.

    CR is especially sensitive to the verbalizer.  The original aligned run
    uses ``terrible/great`` for CR, matching MR and RT.  Keeping this as an
    explicit argument prevents a later repair from silently changing the
    protocol to ``negative/positive`` and invalidating comparisons.
    """
    if len(cr_choices) != 2 or not all(cr_choices):
        raise ValueError("cr_choices must contain two non-empty verbalizers")
    return {
        "sst2": _load_sst2(task_data_dir / "sst2" / "dev.tsv"),
        "mr": _load_sentiment_csv(task_data_dir / "mr" / "test.csv", ["terrible", "great"]),
        "cr": _load_sentiment_csv(task_data_dir / "cr" / "test.csv", list(cr_choices)),
        "rt": _load_rt(task_data_dir / "rotten_tomatoes" / "test.jsonl"),
        "hyp": _load_hyp(task_data_dir / "hyp" / "test.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--task-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cr-choices",
        nargs=2,
        default=["terrible", "great"],
        metavar=("NEGATIVE", "POSITIVE"),
        help="Two CR verbalizers; the aligned protocol uses terrible great.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_data_dir = Path(args.task_data_dir)
    cr_choices = (str(args.cr_choices[0]), str(args.cr_choices[1]))
    tasks = load_tasks(task_data_dir, cr_choices=cr_choices)
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
        for task in TASKS:
            print(f"Evaluating {task} ({len(tasks[task])} examples)...", flush=True)
            results[task] = evaluate_task(
                wrapper,
                canon,
                tasks[task],
                device,
                max_context,
                pmi=True,
                batch_size=args.batch_size,
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
        "tasks": list(TASKS),
        "evaluation_split": {
            "sst2": "kNN_prompt/task_data/sst2/dev.tsv",
            "mr": "kNN_prompt/task_data/mr/test.csv",
            "cr": "kNN_prompt/task_data/cr/test.csv",
            "rt": "kNN_prompt/task_data/rotten_tomatoes/test.jsonl",
            "hyp": "kNN_prompt/task_data/hyp/test.csv",
        },
        "source_url": SOURCE_URL,
        "protocol": "domain_conditional_pmi",
        "prompt_protocol": {
            "sst2": {"prompt": "<input> It was", "choices": ["terrible", "great"]},
            "mr": {"prompt": "<input> It was", "choices": ["terrible", "great"]},
            "cr": {"prompt": "<input> It was", "choices": list(cr_choices)},
            "rt": {"prompt": "<input> It was", "choices": ["terrible", "great"]},
            "hyp": "<article>\\n neutral or partisan? Answer:",
            "hyp_choices": ["neutral", "partisan"],
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
