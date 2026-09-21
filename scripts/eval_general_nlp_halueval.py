"""Evaluate the ATHENA router on general NLP and HaluEval tasks.

The general-NLP tasks follow the domain-conditional PMI protocol used by
MLP Memory (arXiv:2508.01832).  HaluEval is evaluated as binary factuality
classification on its ``*_samples`` configurations and reports accuracy.

This script intentionally keeps the evaluator separate from the existing
open-domain QA runner.  It makes the task formatting, dataset revision, and
reader-mode contract explicit in the output JSON so that results are not
silently mixed with the five-task QA tables.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_openqa import get_model_max_context, setup_condition


GENERAL_TASKS = (
    "sst2",
    "cr",
    "rt",
    "cb",
    "rte",
    "agn",
    "yahoo",
)
HALU_TASKS = (
    "halueval_dialogue",
    "halueval_qa",
    "halueval_summarization",
)
TASKS = (*GENERAL_TASKS, *HALU_TASKS)


def _example(context: str, choices: list[str], label: int, domain_context: str) -> dict:
    if not context.endswith((" ", "\n")):
        context += " "
    if not domain_context.endswith((" ", "\n")):
        domain_context += " "
    return {
        "context": context,
        "domain_context": domain_context,
        "choices": choices,
        "label": int(label),
    }


def _load_dataset(name: str, *args, split: str, **kwargs):
    from datasets import load_dataset

    return load_dataset(name, *args, split=split, **kwargs)


def load_sst2() -> list[dict]:
    ds = _load_dataset("glue", "sst2", split="validation")
    return [
        _example(
            f"Review: {row['sentence']}\nSentiment:",
            ["negative", "positive"],
            int(row["label"]),
            "Review:\nSentiment:",
        )
        for row in ds
    ]


def load_cr() -> list[dict]:
    ds = _load_dataset("SetFit/CR", split="test")
    return [
        _example(
            f"Review: {row['text']}\nSentiment:",
            ["negative", "positive"],
            int(row["label"]),
            "Review:\nSentiment:",
        )
        for row in ds
    ]


def load_rt() -> list[dict]:
    ds = _load_dataset("rotten_tomatoes", split="test")
    return [
        _example(
            f"Review: {row['text']}\nSentiment:",
            ["negative", "positive"],
            int(row["label"]),
            "Review:\nSentiment:",
        )
        for row in ds
    ]


def load_cb() -> list[dict]:
    ds = _load_dataset("super_glue", "cb", split="validation")
    choices = ["entailment", "contradiction", "neutral"]
    return [
        _example(
            f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nRelation:",
            choices,
            int(row["label"]),
            "Premise:\nHypothesis:\nRelation:",
        )
        for row in ds
    ]


def load_rte() -> list[dict]:
    ds = _load_dataset("super_glue", "rte", split="validation")
    choices = ["entailment", "not entailment"]
    return [
        _example(
            f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nRelation:",
            choices,
            int(row["label"]),
            "Premise:\nHypothesis:\nRelation:",
        )
        for row in ds
    ]


def load_agn() -> list[dict]:
    ds = _load_dataset("ag_news", split="test")
    choices = ["world", "sports", "business", "science and technology"]
    return [
        _example(
            f"News article: {row['text']}\nTopic:",
            choices,
            int(row["label"]),
            "News article:\nTopic:",
        )
        for row in ds
    ]


def load_yahoo() -> list[dict]:
    ds = _load_dataset("yahoo_answers_topics", split="test")
    choices = [
        "society and culture",
        "science and mathematics",
        "health",
        "education and reference",
        "computers and internet",
        "sports",
        "business and finance",
        "entertainment and music",
        "family and relationships",
        "politics and government",
    ]
    examples = []
    for row in ds:
        title = str(row.get("question_title") or "").strip()
        content = str(row.get("question_content") or "").strip()
        best = str(row.get("best_answer") or "").strip()
        text = "\n".join(part for part in (title, content, best) if part)
        examples.append(
            _example(
                f"Question: {text}\nTopic:",
                choices,
                int(row["topic"] if "topic" in row else row["label"]),
                "Question:\nTopic:",
            )
        )
    return examples


def _hallucination_label(value) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1", "hallucinated"}:
        return 1
    if normalized in {"no", "false", "0", "not hallucinated"}:
        return 0
    raise ValueError(f"Unrecognised HaluEval label: {value!r}")


def load_halueval(kind: str) -> list[dict]:
    config = f"{kind}_samples"
    ds = _load_dataset("pminervini/HaluEval", config, split="data")
    examples = []
    for row in ds:
        if kind == "dialogue":
            context = (
                f"Knowledge: {row['knowledge']}\n"
                f"Dialogue history: {row['dialogue_history']}\n"
                f"Response: {row['response']}\n"
                "Is the response hallucinated?"
            )
            domain = "Knowledge:\nDialogue history:\nResponse:\nIs the response hallucinated?"
        elif kind == "qa":
            context = (
                f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}\n"
                f"Answer: {row['answer']}\nIs the answer hallucinated?"
            )
            domain = "Knowledge:\nQuestion:\nAnswer:\nIs the answer hallucinated?"
        elif kind == "summarization":
            context = (
                f"Document: {row['document']}\nSummary: {row['summary']}\n"
                "Is the summary hallucinated?"
            )
            domain = "Document:\nSummary:\nIs the summary hallucinated?"
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(kind)
        examples.append(
            _example(context, ["no", "yes"], _hallucination_label(row["hallucination"]), domain)
        )
    return examples


LOADERS: dict[str, Callable[[], list[dict]]] = {
    "sst2": load_sst2,
    "cr": load_cr,
    "rt": load_rt,
    "cb": load_cb,
    "rte": load_rte,
    "agn": load_agn,
    "yahoo": load_yahoo,
    "halueval_dialogue": lambda: load_halueval("dialogue"),
    "halueval_qa": lambda: load_halueval("qa"),
    "halueval_summarization": lambda: load_halueval("summarization"),
}


def _normalise_choice(choice: str) -> str:
    return choice if choice.startswith(" ") else f" {choice}"


def _batch_choice_logprobs(
    wrapper,
    tokenizer,
    contexts: list[str],
    choices: list[str],
    device: torch.device,
    set_canon_fn,
    max_context_length: int,
    max_forward_sequences: int = 2,
    reduction: str = "mean",
) -> list[float]:
    """Score one choice per context in a single padded forward pass."""
    # A Mistral forward pass returns a full ``[batch, seq, vocab]`` tensor.
    # Converting that tensor to float32 for log-softmax can exceed one MI250
    # GCD's HBM when a task expands one example into many answer choices
    # (Yahoo has ten).  Keep the public scoring protocol unchanged while
    # bounding the temporary logits tensor.
    if max_forward_sequences <= 0:
        raise ValueError("max_forward_sequences must be positive")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    if len(contexts) > max_forward_sequences:
        scores = []
        for start in range(0, len(contexts), max_forward_sequences):
            scores.extend(
                _batch_choice_logprobs(
                    wrapper,
                    tokenizer,
                    contexts[start:start + max_forward_sequences],
                    choices[start:start + max_forward_sequences],
                    device,
                    set_canon_fn,
                    max_context_length,
                    max_forward_sequences,
                    reduction,
                )
            )
        return scores

    full_texts = [context + _normalise_choice(choice) for context, choice in zip(contexts, choices)]
    choice_texts = [_normalise_choice(choice) for choice in choices]
    choice_lengths = [
        int(tokenizer(choice, add_special_tokens=False)["input_ids"].__len__())
        for choice in choice_texts
    ]

    old_padding = tokenizer.padding_side
    old_truncation = tokenizer.truncation_side
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    try:
        batch = tokenizer(
            full_texts,
            add_special_tokens=False,
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
        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

    results = []
    for index, choice_len in enumerate(choice_lengths):
        real_len = int(attention_mask[index].sum().item()) if attention_mask is not None else input_ids.shape[1]
        if choice_len <= 0 or choice_len >= real_len:
            results.append(float("-inf"))
            continue
        target_start = real_len - choice_len
        # Right padding keeps the real sequence at the beginning of the row.
        targets = input_ids[index, target_start:real_len]
        logits = log_probs[index, max(0, target_start - 1):real_len - 1]
        if logits.shape[0] != targets.shape[0]:
            results.append(float("-inf"))
            continue
        token_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        aggregate = token_scores.sum() if reduction == "sum" else token_scores.mean()
        results.append(float(aggregate.item()))
    return results


def evaluate_task(
    wrapper,
    set_canon_fn,
    examples: list[dict],
    device,
    max_context_length: int,
    *,
    pmi: bool,
    batch_size: int,
    max_forward_sequences: int = 2,
    score_reduction: str = "mean",
) -> dict:
    def score(contexts, choices):
        # Keep the default call shape compatible with existing lightweight
        # test doubles while allowing Yahoo to opt into larger choice chunks.
        common = (
            wrapper,
            wrapper.tokenizer,
            contexts,
            choices,
            device,
            set_canon_fn,
            max_context_length,
        )
        if max_forward_sequences == 2 and score_reduction == "mean":
            return _batch_choice_logprobs(*common)
        return _batch_choice_logprobs(
            *common, max_forward_sequences, score_reduction
        )

    correct = 0
    total = len(examples)
    predictions = []
    # ``batch_size`` counts task examples, but each example expands into one
    # sequence per answer choice.  Yahoo Answers has ten choices, so using the
    # same example batch as binary tasks creates a 40-row forward pass at the
    # default batch size of four and exceeds a single MI250 GCD's HBM.  Keep
    # the effective number of choice sequences bounded while preserving the
    # larger batch for binary/three-way tasks.
    max_choices = max((len(example["choices"]) for example in examples), default=1)
    effective_batch_size = max(1, min(batch_size, 16 // max_choices))
    for start in range(0, total, effective_batch_size):
        batch_examples = examples[start:start + effective_batch_size]
        contexts = []
        choices = []
        domain_contexts = []
        domain_choices = []
        for example in batch_examples:
            for choice in example["choices"]:
                contexts.append(example["context"])
                choices.append(choice)
                domain_contexts.append(example["domain_context"])
                domain_choices.append(choice)

        conditional = score(contexts, choices)
        if pmi:
            domain = score(domain_contexts, domain_choices)
            scores = [left - right for left, right in zip(conditional, domain)]
        else:
            scores = conditional

        cursor = 0
        for example in batch_examples:
            width = len(example["choices"])
            prediction = int(np.argmax(scores[cursor:cursor + width]))
            cursor += width
            predictions.append(prediction)
            correct += prediction == int(example["label"])
        if (start // effective_batch_size) % 25 == 0:
            print(f"  progress {min(start + len(batch_examples), total)}/{total}", flush=True)

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": int(correct),
        "total": total,
        "protocol": "domain_conditional_pmi" if pmi else "choice_logprob_accuracy",
        "predictions": predictions,
    }


def _args_for_condition(args, adaptor_dir: str | None, checkpoint: str | None):
    return argparse.Namespace(
        target_model=args.target_model,
        adaptor_dir=adaptor_dir,
        adaptor_checkpoint=checkpoint,
        source_memory=args.source_memory,
        memory_config=args.memory_config,
        dual_reader_mode=args.reader_mode,
        seed=args.seed,
        canon_mode=args.canon_mode,
    )


def _release(wrapper) -> None:
    cleanup = getattr(wrapper, "cleanup", None)
    if cleanup is not None:
        cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--reader-mode", default="tri_advantage_routed")
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "general_nlp_halueval_results.json"
    if result_file.exists():
        print(f"Results already exist at {result_file}; refusing to overwrite", flush=True)
        return

    task_meta = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    evaluation = {"tri_E": {}, "tri_reader_advantage": {}}
    tri_args = _args_for_condition(args, args.adaptor_dir, args.adaptor_checkpoint)
    print("Setting up transferred tri-reader checkpoint", flush=True)
    tri, tri_canon = setup_condition(tri_args, "transferred", device, dtype)
    max_context = get_model_max_context(tri, args.max_context_length)
    print("Transferred tri-reader checkpoint ready", flush=True)
    setter = getattr(__import__("scripts.eval_dual_reader_openqa_paired", fromlist=["set_reader_mode"]), "set_reader_mode")
    for task in args.tasks:
        print(f"Loading {task}...", flush=True)
        examples = LOADERS[task]()
        if args.max_examples is not None:
            examples = examples[:args.max_examples]
        task_meta[task] = {
            "n_examples": len(examples),
            "protocol": "domain_conditional_pmi" if task in GENERAL_TASKS else "choice_logprob_accuracy",
        }
        print(f"  {len(examples)} examples", flush=True)
        for mode_name, mode in (("tri_E", "engram_only"), ("tri_reader_advantage", args.reader_mode)):
            setter(tri, mode)
            started = time.time()
            evaluation[mode_name][task] = evaluate_task(
                tri, tri_canon, examples, device, max_context,
                pmi=task in GENERAL_TASKS, batch_size=args.batch_size,
            )
            evaluation[mode_name][task]["elapsed_s"] = time.time() - started
            print(f"{mode_name}/{task}: {evaluation[mode_name][task]['accuracy']:.4f}", flush=True)
        del examples
    _release(tri)

    summary = {}
    for mode, task_results in evaluation.items():
        values = [task_results[task]["accuracy"] for task in args.tasks]
        summary[mode] = {
            task: task_results[task]["accuracy"] for task in args.tasks
        }
        summary[mode]["macro_average"] = float(np.mean(values)) if values else 0.0

    payload = {
        "target_model": args.target_model,
        "tasks": list(args.tasks),
        "task_metadata": task_meta,
        "reader_mode": args.reader_mode,
        "general_protocol": "domain_conditional_pmi",
        "halueval_protocol": "choice_logprob_accuracy",
        "evaluation": evaluation,
        "summary": summary,
    }
    result_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {result_file}", flush=True)


if __name__ == "__main__":
    main()
