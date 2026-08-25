"""Jointly supervise the generated branch on the available training splits.

The frozen Mistral backbone, Engram table, and direct Engram reader are reused
from the existing dual-reader checkpoint.  Only the generator, generated-memory
reader, and generated gate are optimized.  After one joint epoch the checkpoint
is evaluated on all five QA benchmarks using the same task-specific metrics as
``scripts/eval_openqa.py``.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_openqa import (
    TASK_LOADERS,
    TASK_SCALAR_METRICS,
    build_openqa_prompt,
    dedupe_answers,
    evaluate_openqa,
    evaluate_truthfulqa,
    flatten_answers,
    get_model_max_context,
    setup_condition,
    task_scalar_score,
)
from scripts.train_adaptor import get_cosine_schedule_with_warmup
from scripts.train_generative_memory_qa import configure_generated_branch


TRAIN_TASKS = (
    "nq",
    "webqa",
    "triviaqa",
    "hotpotqa",
    "gsm8k",
    "math",
    "kodcode",
)
EVAL_TASKS = ("nq", "webqa", "triviaqa", "truthfulqa", "hotpotqa")

TRAIN_SPECS = {
    "nq": ("google-research-datasets/nq_open", None, "train"),
    "webqa": ("Stanford/web_questions", None, "train"),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "train"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "train"),
    "gsm8k": ("openai/gsm8k", "main", "train"),
    "math": ("DigitalLearningGmbH/MATH-lighteval", None, "train"),
}

TRAINING_EXCLUSIONS = {
    "popqa": "official dataset exposes test only",
    "gpqa": "448-question corpus has no independent train split; avoid leakage into Diamond evaluation",
    "bigcodebench": "official dataset has no splits; all 1,140 rows are evaluation rows",
    "alfworld": "interactive valid_unseen evaluation has no supervised split in this text-QA trainer",
}


def _last_boxed_content(text: str) -> str:
    """Return the content of the last balanced ``\\boxed{...}`` expression."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""
    index = start + len(marker)
    depth = 1
    chars = []
    while index < len(text) and depth:
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
        index += 1
    return "".join(chars).strip() if depth == 0 else text[start:].strip()


def _training_prompt(task: str, question: str) -> str:
    if task in {"gsm8k", "math"}:
        return "Give the final answer in \\boxed{} without unnecessary text.\nProblem: " + question
    if task == "kodcode":
        return (
            "Write a correct Python solution. Return only Python code without Markdown fences.\n"
            + question
        )
    return build_openqa_prompt(question)


def extract_training_example(task: str, raw: dict) -> dict | None:
    """Normalize one raw record into question/answer supervision."""
    if task == "gsm8k":
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).rsplit("####", 1)[-1].strip()
        if not question or not answer:
            return None
        return {
            "task": task,
            "question": question,
            "prompt": _training_prompt(task, question),
            "answer": "\\boxed{" + answer + "}",
            "answers": [answer],
        }

    if task == "math":
        question = str(raw.get("problem", "")).strip()
        answer = _last_boxed_content(str(raw.get("solution", "")))
        if not question or not answer:
            return None
        return {
            "task": task,
            "question": question,
            "prompt": _training_prompt(task, question),
            "answer": "\\boxed{" + answer + "}",
            "answers": [answer],
        }

    if task == "kodcode":
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("solution", "")).strip()
        if not question or not answer:
            return None
        return {
            "task": task,
            "question": question,
            "prompt": _training_prompt(task, question),
            "answer": answer,
            "answers": [answer],
        }

    answer_value = raw.get("answers", raw.get("answer"))
    answers = dedupe_answers(flatten_answers(answer_value))
    if task == "nq" and ")" in answers:
        return None
    answers = [answer for answer in answers if answer and answer != "<unk>"]
    if not answers:
        return None

    preferred = None
    if task == "triviaqa" and isinstance(raw.get("answer"), dict):
        preferred = raw["answer"].get("value")
    answer = str(preferred).strip() if preferred else answers[0]
    question = str(raw.get("question", "")).strip()
    if not question or not answer:
        return None
    return {
        "task": task,
        "question": question,
        "prompt": _training_prompt(task, question),
        "answer": answer,
        "answers": answers,
    }


def load_training_task(task: str, max_examples: int | None = None):
    from datasets import load_dataset

    if task == "kodcode":
        dataset_name = "KodCode/KodCode-Light-RL-10K"
        config_name = None
        source_split = "train"
        full_dataset = load_dataset(dataset_name, split=source_split)
        dataset = full_dataset.train_test_split(
            test_size=0.2, seed=42, shuffle=True
        )["train"]
        split = "seeded_80pct_train"
    else:
        dataset_name, config_name, split = TRAIN_SPECS[task]
        dataset = load_dataset(dataset_name, config_name, split=split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    examples = []
    for raw in dataset:
        example = extract_training_example(task, raw)
        if example is not None:
            examples.append(example)
    source = {
        "dataset_name": dataset_name,
        "config_name": config_name,
        "split": split,
        "raw_count": len(dataset),
        "usable_count": len(examples),
    }
    if task == "kodcode":
        source.update({
            "source_split": source_split,
            "evaluation_split": "seeded_20pct_test",
            "seed": 42,
        })
    return examples, source


class JointQADataset(Dataset):
    def __init__(self, tasks=TRAIN_TASKS, max_examples_per_task: int | None = None):
        self.examples = []
        self.sources = {}
        for task in tasks:
            examples, source = load_training_task(task, max_examples_per_task)
            self.examples.extend(examples)
            self.sources[task] = source

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class AnswerOnlyCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, question: str, answer: str, prompt: str | None = None):
        prompt = prompt or build_openqa_prompt(question)
        answer_text = " " + answer.strip()
        if self.tokenizer.eos_token:
            answer_text += self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(prompt + answer_text, add_special_tokens=True)["input_ids"]

        prefix_len = 0
        for prompt_id, full_id in zip(prompt_ids, full_ids):
            if prompt_id != full_id:
                break
            prefix_len += 1
        if prefix_len == 0 or prefix_len >= len(full_ids):
            raise ValueError("Unable to identify the supervised answer span")
        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            full_ids = full_ids[overflow:]
            prefix_len = max(0, prefix_len - overflow)
        labels = [-100] * prefix_len + full_ids[prefix_len:]
        return full_ids, labels

    def __call__(self, examples):
        encoded = [
            self.encode(ex["question"], ex["answer"], ex.get("prompt"))
            for ex in examples
        ]
        max_len = max(len(ids) for ids, _ in encoded)
        pad_id = self.tokenizer.pad_token_id
        input_batch, label_batch, mask_batch = [], [], []
        for input_ids, labels in encoded:
            pad_len = max_len - len(input_ids)
            input_batch.append(input_ids + [pad_id] * pad_len)
            label_batch.append(labels + [-100] * pad_len)
            mask_batch.append([1] * len(input_ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
            "attention_mask": torch.tensor(mask_batch, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument(
        "--adaptor-checkpoint",
        default=None,
        help="Optional checkpoint path when the adaptor checkpoint is not in adaptor-dir.",
    )
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dual-reader-mode",
        choices=["both", "engram_only", "generated_only"],
        default="generated_only",
        help="Reader contribution used during generated-branch training and evaluation.",
    )
    parser.add_argument("--canon-mode", default="word_boundary", choices=["vocab", "word_boundary"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-train-examples-per-task", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument(
        "--eval-tasks",
        nargs="+",
        choices=EVAL_TASKS,
        default=list(EVAL_TASKS),
        help="Generation-evaluate only these tasks after training.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_all_tasks(wrapper, set_canon_fn, device, args):
    tokenizer = wrapper.tokenizer
    max_context = get_model_max_context(wrapper, None)
    all_results = {}
    for task in args.eval_tasks:
        examples, dataset_meta = TASK_LOADERS[task]()
        if args.max_eval_examples is not None:
            examples = examples[: args.max_eval_examples]
        started = time.time()
        if task == "truthfulqa":
            metrics = evaluate_truthfulqa(
                wrapper=wrapper,
                set_canon_fn=set_canon_fn,
                tokenizer=tokenizer,
                examples=examples,
                device=device,
                max_context_length=max_context,
            )
        else:
            metrics = evaluate_openqa(
                wrapper=wrapper,
                set_canon_fn=set_canon_fn,
                tokenizer=tokenizer,
                task_name=task,
                examples=examples,
                device=device,
                max_new_tokens=args.max_new_tokens,
                max_context_length=max_context,
            )
        metrics["elapsed_s"] = time.time() - started
        all_results[task] = {
            "dataset": dataset_meta,
            "n_examples": len(examples),
            "scalar_metric": TASK_SCALAR_METRICS[task],
            "metrics": metrics,
            "scalar_score": task_scalar_score(task, metrics),
        }
        print(f"EVAL {task}: {json.dumps(all_results[task], default=str)}")
    return all_results


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    if results_file.exists():
        raise FileExistsError(f"Refusing to overwrite {results_file}")
    # Keep the runtime architecture metadata next to the trained checkpoint.
    # The input dual-reader checkpoint predates this task-supervision script,
    # so its config is the authoritative source for injection and reader
    # construction parameters.  Merge CLI values on top without dropping
    # those fields; downstream evaluators can then load this directory
    # directly with the same generated-only reader mode.
    input_runtime_config = {}
    input_config_path = Path(args.adaptor_dir) / "config.json"
    if input_config_path.exists():
        with open(input_config_path) as handle:
            input_runtime_config = json.load(handle)
    runtime_config = dict(input_runtime_config)
    runtime_config.update(vars(args))
    with open(output_dir / "config.json", "w") as handle:
        json.dump(runtime_config, handle, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    wrapper, set_canon_fn = setup_condition(args, "transferred", device, dtype)
    tokenizer = wrapper.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable_names = configure_generated_branch(wrapper)
    trainable = wrapper.get_trainable_params()
    print(f"Trainable generated-branch parameters: {sum(p.numel() for p in trainable):,}")
    print("Trainable tensors:\n  " + "\n  ".join(trainable_names))
    assert trainable and all(not p.requires_grad for p in wrapper.backbone.parameters())
    assert wrapper.memory is None or all(not p.requires_grad for p in wrapper.memory.parameters())

    train_dataset = JointQADataset(max_examples_per_task=args.max_train_examples_per_task)
    print(f"Training sources: {json.dumps(train_dataset.sources)}")
    print(f"Joint train size: {len(train_dataset):,}")
    collator = AnswerOnlyCollator(tokenizer, args.max_length)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collator,
    )

    updates_per_epoch = math.ceil(len(loader) / args.grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    optimizer.zero_grad(set_to_none=True)

    started = time.time()
    first_loss = final_loss = None
    global_step = 0
    with open(output_dir / "train_log.jsonl", "w") as log_handle:
        wrapper.backbone.eval()
        if wrapper.memory is not None:
            wrapper.memory.eval()
        for epoch in range(args.epochs):
            wrapper.adaptor.train()
            running_loss = 0.0
            for micro_step, batch in enumerate(tqdm(loader, desc=f"Joint QA epoch {epoch + 1}"), 1):
                input_ids = batch["input_ids"].to(device)
                set_canon_fn(input_ids)
                outputs = wrapper(
                    input_ids=input_ids,
                    labels=batch["labels"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                raw_loss = outputs.loss
                if first_loss is None:
                    first_loss = float(raw_loss.item())
                final_loss = float(raw_loss.item())
                (raw_loss / args.grad_accum_steps).backward()
                running_loss += final_loss
                should_step = micro_step % args.grad_accum_steps == 0 or micro_step == len(loader)
                if not should_step:
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or global_step % args.log_every == 0:
                    record = {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": running_loss / min(args.grad_accum_steps, micro_step),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    log_handle.write(json.dumps(record) + "\n")
                    log_handle.flush()
                    print(record)
                running_loss = 0.0

    torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor.pt")
    torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor_best.pt")
    wrapper.eval()
    evaluation = evaluate_all_tasks(wrapper, set_canon_fn, device, args)
    results = {
        "objective": "joint_answer_and_code_only_cross_entropy",
        "train_tasks": list(TRAIN_TASKS),
        "eval_tasks": list(args.eval_tasks),
        "frozen": ["backbone", "engram_table", "engram_reader"],
        "trainable": ["generator", "generated_reader", "generated_gate"],
        "dual_reader_mode": args.dual_reader_mode,
        "training_sources": train_dataset.sources,
        "training_exclusions": TRAINING_EXCLUSIONS,
        "train_size": len(train_dataset),
        "first_loss": first_loss,
        "final_batch_loss": final_loss,
        "optimizer_steps": global_step,
        "evaluation": evaluation,
        "elapsed_hours": (time.time() - started) / 3600,
        "completed": True,
    }
    with open(results_file, "w") as handle:
        json.dump(results, handle, indent=2)
    print("MULTITASK_QA_TRAINING_AND_EVAL_COMPLETE")
    wrapper.cleanup()


if __name__ == "__main__":
    main()
