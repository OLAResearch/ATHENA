"""Run one exactly aligned nine-task evaluation for all ATHENA conditions.

The task formatting follows the public kNN-Prompt loaders used by the MLP
Memory comparison.  The same examples, verbalizers, domain-conditional PMI
score, and next-token synonym aggregation are used for the bare Mistral baseline,
Engram-only, and the three-reader advantage router.  No downstream labels are
used to configure either memory condition; labels are consumed only when
computing the final accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from engram.tri_memory import TriMemoryAdaptor
from scripts.eval_openqa import get_model_max_context, setup_condition


TASKS = ("sst2", "mr", "cr", "rt", "hyp", "cb", "rte", "agn", "yahoo")
PAPER_TARGETS = {
    "sst2": 0.8121,
    "mr": 0.7535,
    "cr": 0.6230,
    "rt": 0.7495,
    "hyp": 0.5542,
    "cb": 0.6964,
    "rte": 0.5957,
    "agn": 0.7595,
    "yahoo": 0.5636,
}
SOURCE_URL = "https://github.com/swj0419/kNN_prompt/tree/main/task_data"


DEFAULT_LABEL_SYNONYMS = {
    "sentiment": [[" terrible"], [" great"]],
    "hyp": [[" neutral"], [" partisan"]],
    "cb": [[" true"], [" false"], [" neither"]],
    "agn": [[" world"], [" sports"], [" business"], [" science"]],
    "yahoo": [
        [" society"], [" science"], [" health"], [" education"], [" computer"],
        [" sports"], [" business"], [" entertainment"], [" family"], [" politics"],
    ],
}


def _paper_example(
    context: str,
    choices: list[str],
    label: int,
    domain: str,
    label_synonyms: list[list[str]],
) -> dict:
    # Keep leading blanks in both the prompt and verbalizers.  The public
    # loader concatenates premise + hypothesis directly, so normalising either
    # side would silently change the first answer-token distribution.
    if not choices or any(not choice.startswith(" ") for choice in choices):
        raise ValueError("paper verbalizers must start with one leading space")
    return {
        "context": context,
        "domain_context": domain,
        "choices": choices,
        "label_synonyms": label_synonyms,
        "label": int(label),
    }


def _load_label_file(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as handle:
        rows = [[f" {word}" for word in line.strip().split(", ")] for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"label file is empty: {path}")
    min_len = min(map(len, rows))
    return [row[:min_len] for row in rows]


def load_paper_label_assets(asset_dir: Path | None = None) -> dict[str, list[list[str]]]:
    """Load the exact synonym assets used by the public dCPMI evaluator."""
    if asset_dir is None:
        return {name: [list(row) for row in rows] for name, rows in DEFAULT_LABEL_SYNONYMS.items()}
    sentiment = _load_label_file(asset_dir / "label_names_sentidict.txt")
    agn = _load_label_file(asset_dir / "label_names_kb.txt")
    with (asset_dir / "yahoo_label.json").open(encoding="utf-8") as handle:
        yahoo_topics = json.load(handle)
    yahoo_order = [
        "Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
        "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
        "Family & Relationships", "Politics & Government",
    ]
    yahoo = [[f" {word}" for word in yahoo_topics[name]] for name in yahoo_order]
    return {
        "sentiment": sentiment,
        "hyp": [[" neutral", " fair", " objective"], [" partisan", " biased", " unfair"]],
        "cb": [
            [" true", " yes", " accurate", " correct", " faithful"],
            [" false", " no", " incorrect", " wrong", " untrue", " unfaithful"],
            [" neither"],
        ],
        "agn": agn,
        "yahoo": yahoo,
    }


def _load_sst2(path: Path, labels: dict[str, list[list[str]]] | None = None) -> list[dict]:
    labels = labels or load_paper_label_assets()
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw_label, sentence = line.rstrip("\n").split("\t", 1)
            score = int(raw_label[-1]) - 3
            if score == 0:
                continue
            examples.append(
                _paper_example(
                    f"{sentence} The sentence has a tone that is",
                    [" terrible", " great"],
                    1 if score > 0 else 0,
                    " The sentence has a tone that is",
                    labels["sentiment"],
                )
            )
    return examples


def _load_sentiment_csv(
    path: Path,
    labels: dict[str, list[list[str]]] | None = None,
    prompt: str = "The sentence has a tone that is",
) -> list[dict]:
    labels = labels or load_paper_label_assets()
    examples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            examples.append(
                _paper_example(
                    f"{row['text']} {prompt}" if not prompt.startswith(" ") else f"{row['text']}{prompt}",
                    [" terrible", " great"],
                    int(row["label"]),
                    prompt,
                    labels["sentiment"],
                )
            )
    return examples


def _load_rt(path: Path, labels: dict[str, list[list[str]]] | None = None) -> list[dict]:
    labels = labels or load_paper_label_assets()
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(
                _paper_example(
                    f"{row['input']} It is",
                    [" terrible", " great"],
                    1 if row["output"].strip().lower() == "positive" else 0,
                    " It is",
                    labels["sentiment"],
                )
            )
    return examples


def _load_hyp(path: Path, labels: dict[str, list[list[str]]] | None = None) -> list[dict]:
    labels = labels or load_paper_label_assets()
    examples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            examples.append(
                _paper_example(
                    f"{row['text'].strip()}\n neutral or partisan? Answer:",
                    [" neutral", " partisan"],
                    int(row["label"]),
                    "\n neutral or partisan? Answer:",
                    labels["hyp"],
                )
            )
    return examples


def _load_cb(path: Path, labels: dict[str, list[list[str]]] | None = None) -> list[dict]:
    labels = labels or load_paper_label_assets()
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
    choices = [" true", " false", " neither"]
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            context = (
                f'premise: {row["premise"]}\n'
                f'hypothesis: {row["hypothesis"]}\n'
                "answer: The hypothesis is"
            )
            examples.append(_paper_example(context, choices, label_map[row["label"]], "answer: The hypothesis is", labels["cb"]))
    return examples


def _load_rte(path: Path) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            context = (
                f" {row['premise']}\nquestion: {row['hypothesis']} "
                "true or false?\nanswer:"
            )
            label = 0 if row["label"] == "entailment" else 1
            examples.append(
                _paper_example(
                    context, [" true", " false"], label, "true or false?\nanswer:"
                    , [[" true", " yes", " accurate", " correct", " faithful"],
                       [" false", " no", " incorrect", " wrong", " untrue", " unfaithful"]]
                )
            )
    return examples


def _load_agn(path: Path, labels: dict[str, list[list[str]]] | None = None) -> list[dict]:
    labels = labels or load_paper_label_assets()
    choices = [" world", " sports", " business", " science"]
    examples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            examples.append(
                _paper_example(
                    f"title: {row['Title']}\nsummary: {row['Description']}\ntopic:",
                    choices,
                    int(row["Class Index"]) - 1,
                    "topic:",
                    labels["agn"],
                )
            )
    return examples


def _load_yahoo(labels: dict[str, list[list[str]]] | None = None) -> list[dict]:
    labels = labels or load_paper_label_assets()
    from datasets import load_dataset

    choices = [
        " society",
        " science",
        " health",
        " education",
        " computer",
        " sports",
        " business",
        " entertainment",
        " family",
        " politics",
    ]
    dataset = load_dataset("yahoo_answers_topics", split="test")
    examples = []
    for row in dataset:
        title = str(row.get("question_title") or "")
        content = str(row.get("question_content") or "")
        answer = str(row.get("best_answer") or "")
        examples.append(
            _paper_example(
                f"question: {title}\nanswer: {answer}\nthe topic of the question and answer is",
                choices,
                int(row["topic"] if "topic" in row else row["label"]),
                "the topic of the question and answer is",
                labels["yahoo"],
            )
        )
    return examples


def load_tasks(
    task_data_dir: Path,
    cb_data: Path,
    rte_data: Path,
    agn_data: Path,
    label_assets_dir: Path | None = None,
) -> dict[str, list[dict]]:
    labels = load_paper_label_assets(label_assets_dir)
    return {
        "sst2": _load_sst2(task_data_dir / "sst2" / "dev.tsv", labels),
        "mr": _load_sentiment_csv(
            task_data_dir / "mr" / "test.csv", labels, "The sentence has a tone that is"
        ),
        "cr": _load_sentiment_csv(task_data_dir / "cr" / "test.csv", labels, " It was"),
        "rt": _load_rt(task_data_dir / "rotten_tomatoes" / "test.jsonl", labels),
        "hyp": _load_hyp(task_data_dir / "hyp" / "test.csv", labels),
        "cb": _load_cb(cb_data, labels),
        "rte": _load_rte(rte_data),
        "agn": _load_agn(agn_data, labels),
        "yahoo": _load_yahoo(labels),
    }


def _release(wrapper) -> None:
    cleanup = getattr(wrapper, "cleanup", None)
    if cleanup is not None:
        cleanup()
    gc.collect()


def _synonym_token_ids(tokenizer, synonyms: list[list[str]]) -> list[list[int]]:
    """Reproduce kNN-Prompt's first-token synonym mapping for Mistral."""
    token_ids = []
    for label_synonyms in synonyms:
        ids_for_label = []
        for synonym in label_synonyms:
            encoded = tokenizer(synonym, add_special_tokens=True)["input_ids"]
            if not encoded:
                continue
            bos_id = getattr(tokenizer, "bos_token_id", None)
            token_index = 1 if bos_id is not None and encoded[0] == bos_id else 0
            if token_index < len(encoded):
                ids_for_label.append(int(encoded[token_index]))
        if not ids_for_label:
            raise ValueError(f"synonym set produced no token IDs: {label_synonyms!r}")
        ids_for_label = list(dict.fromkeys(ids_for_label))
        token_ids.append(ids_for_label)
    return token_ids


def _batch_next_token_log_probs(
    wrapper,
    tokenizer,
    contexts: list[str],
    device: torch.device,
    set_canon_fn,
    max_context_length: int,
    routing_collector=None,
) -> torch.Tensor:
    """Return next-token log probabilities using the original dCPMI layout."""
    old_padding = tokenizer.padding_side
    old_truncation = tokenizer.truncation_side
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    try:
        batch = tokenizer(
            contexts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context_length,
        )
    finally:
        tokenizer.padding_side = old_padding
        tokenizer.truncation_side = old_truncation

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if set_canon_fn is not None:
        set_canon_fn(input_ids)
    with torch.no_grad():
        outputs = wrapper(input_ids=input_ids, attention_mask=attention_mask)
        if attention_mask is None:
            last_positions = torch.full(
                (input_ids.shape[0],), input_ids.shape[1] - 1,
                dtype=torch.long, device=input_ids.device
            )
        else:
            last_positions = attention_mask.sum(dim=1).long() - 1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        logits = outputs.logits[rows, last_positions, :].float()
        if routing_collector is not None:
            routing_collector.add_last_positions(wrapper, last_positions)
        return F.log_softmax(logits, dim=-1)


def evaluate_paper_task(
    wrapper,
    set_canon_fn,
    examples: list[dict],
    device: torch.device,
    max_context_length: int,
    *,
    batch_size: int,
    routing_collector=None,
) -> dict:
    """Exact dCPMI evaluation used by the public kNN-Prompt evaluator.

    The original evaluator scores only the next token, sums the log scores of
    the synonym tokens belonging to each label, and subtracts a single
    task-level domain prompt distribution.  This is intentionally different
    from full-choice sequence likelihood.
    """
    if not examples:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "protocol": "dcpmi_next_token"}
    tokenizer = wrapper.tokenizer
    label_token_ids = _synonym_token_ids(tokenizer, examples[0]["label_synonyms"])
    domain_log_probs = _batch_next_token_log_probs(
        wrapper,
        tokenizer,
        [examples[0]["domain_context"]],
        device,
        set_canon_fn,
        max_context_length,
        routing_collector=None,
    )[0]
    predictions = []
    correct = 0
    total = len(examples)
    for start in range(0, total, batch_size):
        batch_examples = examples[start:start + batch_size]
        conditional = _batch_next_token_log_probs(
            wrapper,
            tokenizer,
            [example["context"] for example in batch_examples],
            device,
            set_canon_fn,
            max_context_length,
            routing_collector=routing_collector,
        )
        for row, example in zip(conditional, batch_examples):
            scores = []
            for ids_for_label in label_token_ids:
                ids = torch.tensor(ids_for_label, dtype=torch.long, device=device)
                scores.append(float((row[ids] - domain_log_probs[ids]).sum().item()))
            prediction = int(np.argmax(scores))
            predictions.append(prediction)
            correct += prediction == int(example["label"])
        if (start // batch_size) % 25 == 0:
            print(f"  progress {min(start + len(batch_examples), total)}/{total}", flush=True)
    return {
        "accuracy": correct / total,
        "correct": int(correct),
        "total": total,
        "protocol": "dcpmi_next_token_synonym_sum",
        "predictions": predictions,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _condition_args(args, *, reader_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        target_model=args.target_model,
        adaptor_dir=args.adaptor_dir,
        adaptor_checkpoint=args.adaptor_checkpoint,
        source_memory=args.source_memory,
        memory_config=args.memory_config,
        dual_reader_mode=reader_mode,
        seed=args.seed,
        canon_mode=args.canon_mode,
    )


def _adaptors(wrapper):
    if isinstance(wrapper.adaptor, torch.nn.ModuleList):
        return list(wrapper.adaptor)
    return [wrapper.adaptor]


def _configure_random_router(wrapper, seed: int) -> None:
    """Keep the trained readers/memory fixed and permute route weights only."""
    for adaptor in _adaptors(wrapper):
        if not isinstance(adaptor, TriMemoryAdaptor):
            delegate = getattr(adaptor, "_tri_delegate", None)
            if not isinstance(delegate, TriMemoryAdaptor):
                raise TypeError("Random router requires a tri-reader adaptor")
            adaptor = delegate
        adaptor.configure_random_router(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--task-data-dir", required=True)
    parser.add_argument("--cb-data", required=True)
    parser.add_argument("--rte-data", required=True)
    parser.add_argument("--agn-data", required=True)
    parser.add_argument("--label-assets-dir", default=None)
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reader-mode", default="tri_advantage_routed")
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="Comma-separated task names to evaluate (default: all nine paper tasks).",
    )
    parser.add_argument(
        "--conditions",
        default="vanilla_mistral,engram_only,tri_reader_advantage",
        help=(
            "Comma-separated conditions: vanilla_mistral, tri_e (E-only), "
            "engram_only, tri_reader_advantage, random_router."
        ),
    )
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-forward-sequences", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_forward_sequences <= 0:
        parser.error("batch sizes must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None

    selected_tasks = tuple(name.strip() for name in args.tasks.split(",") if name.strip())
    unknown_tasks = sorted(set(selected_tasks) - set(TASKS))
    if not selected_tasks or unknown_tasks:
        parser.error(f"--tasks must be a non-empty subset of {TASKS}; unknown={unknown_tasks}")

    condition_specs = []
    for condition in (name.strip() for name in args.conditions.split(",") if name.strip()):
        if condition == "vanilla_mistral":
            condition_specs.append((condition, None))
        elif condition == "tri_e":
            condition_specs.append((condition, "engram_only"))
        elif condition in {"engram_only", "tri_reader_advantage"}:
            condition_specs.append((condition, condition))
        elif condition == "random_router":
            condition_specs.append((condition, "tri_random_advantage_routed"))
        else:
            parser.error(
                "--conditions entries must be vanilla_mistral, tri_e, "
                "engram_only, tri_reader_advantage, or random_router"
            )
    if not condition_specs:
        parser.error("--conditions must contain at least one condition")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "results.json"
    if result_file.exists():
        raise FileExistsError(f"Refusing to overwrite {result_file}")

    torch.manual_seed(args.seed)
    tasks = load_tasks(
        Path(args.task_data_dir),
        Path(args.cb_data),
        Path(args.rte_data),
        Path(args.agn_data),
        Path(args.label_assets_dir) if args.label_assets_dir else None,
    )
    tasks = {name: tasks[name] for name in selected_tasks}
    if args.max_examples is not None:
        tasks = {name: rows[: args.max_examples] for name, rows in tasks.items()}
    print("Loaded task sizes:", {name: len(rows) for name, rows in tasks.items()}, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    metadata = {
        "source_url": SOURCE_URL,
        "protocol": "domain_conditional_pmi",
        "score_reduction": "next_token_log_probability_sum_over_label_synonyms",
        "paper_targets": PAPER_TARGETS,
        "task_sizes": {name: len(rows) for name, rows in tasks.items()},
        "labels_used_only_for_accuracy": True,
        "selected_tasks": list(selected_tasks),
        "selected_conditions": [name for name, _ in condition_specs],
    }
    if any(name == "random_router" for name, _ in condition_specs):
        metadata["random_router"] = {
            "seed": args.seed,
            "construction": "permute trained three-source E/GE/GH weights across batch-time positions",
            "memory_and_readers_fixed": True,
        }
    evaluation = {}
    status_file = output_dir / "status.json"

    if any(mode is None for _, mode in condition_specs):
        baseline_args = SimpleNamespace(target_model=args.target_model, dual_reader_mode="auto")
        print("Stage baseline: Vanilla Mistral", flush=True)
        baseline, baseline_canon = setup_condition(baseline_args, "baseline", device, dtype)
        max_context = get_model_max_context(baseline, None)
        try:
            evaluation["vanilla_mistral"] = {}
            for task in selected_tasks:
                started = time.time()
                result = evaluate_paper_task(
                    baseline,
                    baseline_canon,
                    tasks[task],
                    device,
                    max_context,
                    batch_size=args.batch_size,
                )
                result["elapsed_s"] = time.time() - started
                evaluation["vanilla_mistral"][task] = result
                print(
                    f"vanilla/{task}: {result['correct']}/{result['total']} = {result['accuracy']:.6f}",
                    flush=True,
                )
        finally:
            _release(baseline)
        status_file.write_text(json.dumps({"stage": "baseline_complete", "evaluation": evaluation}, indent=2) + "\n")

    from scripts.eval_dual_reader_openqa_paired import set_reader_mode

    reader_specs = [(name, mode) for name, mode in condition_specs if mode is not None]
    if reader_specs:
        tri_args = _condition_args(args, reader_mode=args.reader_mode)
        print("Stage transferred readers: requested Engram/router conditions", flush=True)
        tri, tri_canon = setup_condition(tri_args, "transferred", device, dtype)
        max_context = get_model_max_context(tri, None)
        try:
            for mode_name, mode in reader_specs:
                evaluation[mode_name] = {}
                actual_mode = "engram_only" if mode == "engram_only" else args.reader_mode
                if mode == "tri_random_advantage_routed":
                    _configure_random_router(tri, args.seed)
                    actual_mode = "tri_random_advantage_routed"
                set_reader_mode(tri, actual_mode)
                for task in selected_tasks:
                    started = time.time()
                    result = evaluate_paper_task(
                        tri,
                        tri_canon,
                        tasks[task],
                        device,
                        max_context,
                        batch_size=args.batch_size,
                    )
                    result["elapsed_s"] = time.time() - started
                    evaluation[mode_name][task] = result
                    print(
                        f"{mode_name}/{task}: {result['correct']}/{result['total']} = {result['accuracy']:.6f}",
                        flush=True,
                    )
        finally:
            _release(tri)

    summary = {}
    for condition, task_results in evaluation.items():
        values = [task_results[task]["accuracy"] for task in selected_tasks]
        summary[condition] = {
            task: task_results[task]["accuracy"] for task in selected_tasks
        }
        summary[condition]["macro_average"] = float(np.mean(values))

    payload = {
        "target_model": args.target_model,
        "tasks": list(selected_tasks),
        "conditions": [name for name, _ in condition_specs],
        "reader_mode": args.reader_mode,
        "metadata": metadata,
        "evaluation": evaluation,
        "summary": summary,
    }
    result_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    status_file.write_text(json.dumps({"stage": "complete", "summary": summary}, indent=2) + "\n")
    print(f"Wrote {result_file}", flush=True)


if __name__ == "__main__":
    main()
