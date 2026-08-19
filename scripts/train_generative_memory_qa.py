"""Task-supervise only the generated branch of a dual-reader checkpoint.

The architecture and direct Engram path are kept fixed.  Question/answer
cross-entropy updates only the generator, generated-memory reader, and its
gate.  The prompt format and TriviaQA metrics match ``eval_openqa.py``.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.generative_memory import GenerativeMemoryAdaptor
from scripts.eval_openqa import (
    build_openqa_prompt,
    dedupe_answers,
    exact_match,
    f1_score,
    flatten_answers,
    get_model_max_context,
    greedy_generate,
    setup_condition,
)
from scripts.train_adaptor import get_cosine_schedule_with_warmup


class TriviaQASFTDataset(Dataset):
    def __init__(self, split: str, max_examples: int | None = None):
        from datasets import load_dataset

        errors = []
        dataset = None
        self.source = None
        for config_name in ("rc.nocontext", "unfiltered.nocontext"):
            try:
                dataset = load_dataset(
                    "mandarjoshi/trivia_qa",
                    config_name,
                    split=split,
                    trust_remote_code=True,
                )
                self.source = f"mandarjoshi/trivia_qa/{config_name}:{split}"
                break
            except Exception as exc:
                errors.append(f"{config_name}: {type(exc).__name__}: {exc}")
        if dataset is None:
            raise RuntimeError("Unable to load TriviaQA:\n" + "\n".join(errors))
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        self.examples = []
        for example in dataset:
            answers = dedupe_answers(flatten_answers(example.get("answer")))
            if not answers:
                continue
            preferred = example.get("answer", {}).get("value")
            answer = str(preferred).strip() if preferred else answers[0]
            if answer:
                self.examples.append(
                    {
                        "question": example["question"],
                        "answer": answer,
                        "answers": answers,
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class AnswerOnlyCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, question: str, answer: str) -> tuple[list[int], list[int]]:
        prompt = build_openqa_prompt(question)
        answer_text = " " + answer.strip()
        if self.tokenizer.eos_token:
            answer_text += self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(
            prompt + answer_text,
            add_special_tokens=True,
        )["input_ids"]

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
        encoded = [self.encode(ex["question"], ex["answer"]) for ex in examples]
        max_len = max(len(input_ids) for input_ids, _ in encoded)
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
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canon-mode", default="word_boundary", choices=["vocab", "word_boundary"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-generation-eval", action="store_true")
    return parser.parse_args()


def configure_generated_branch(wrapper) -> list[str]:
    wrapper.freeze_backbone()
    wrapper.freeze_memory()
    adaptors = list(wrapper.adaptor) if isinstance(wrapper.adaptor, torch.nn.ModuleList) else [wrapper.adaptor]
    trainable_names = []
    for index, adaptor in enumerate(adaptors):
        if not isinstance(adaptor, GenerativeMemoryAdaptor):
            raise TypeError("Expected a GenerativeMemoryAdaptor checkpoint")
        for name in adaptor.train_generated_branch_only():
            trainable_names.append(f"{index}.{name}" if len(adaptors) > 1 else name)
    return trainable_names


def evaluate_loss(wrapper, loader, set_canon_fn, device) -> float:
    wrapper.adaptor.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="TriviaQA validation loss", leave=False):
            input_ids = batch["input_ids"].to(device)
            set_canon_fn(input_ids)
            outputs = wrapper(
                input_ids=input_ids,
                labels=batch["labels"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            losses.append(float(outputs.loss.item()))
    return sum(losses) / max(len(losses), 1)


def evaluate_generation(wrapper, dataset, set_canon_fn, device, max_new_tokens):
    wrapper.adaptor.eval()
    tokenizer = wrapper.tokenizer
    max_context = get_model_max_context(wrapper, None)
    exact, f1_values = 0, []
    for example in tqdm(dataset.examples, desc="TriviaQA generation"):
        prediction = greedy_generate(
            wrapper,
            tokenizer,
            build_openqa_prompt(example["question"]),
            device,
            set_canon_fn,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context,
            official_tokenization=True,
        )
        exact += int(any(exact_match(prediction, answer) for answer in example["answers"]))
        f1_values.append(max(f1_score(prediction, answer)[0] for answer in example["answers"]))
    count = len(dataset)
    return {
        "em": exact / count if count else 0.0,
        "f1": sum(f1_values) / count if count else 0.0,
        "count": count,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)

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

    train_dataset = TriviaQASFTDataset("train", args.max_train_examples)
    eval_dataset = TriviaQASFTDataset("validation", args.max_eval_examples)
    print(f"Dataset: {train_dataset.source}; train={len(train_dataset):,}")
    print(f"Dataset: {eval_dataset.source}; validation={len(eval_dataset):,}")
    collator = AnswerOnlyCollator(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    optimizer.zero_grad(set_to_none=True)

    start = time.time()
    first_loss, final_loss = None, None
    global_step = 0
    log_handle = open(output_dir / "train_log.jsonl", "w")
    wrapper.backbone.eval()
    if wrapper.memory is not None:
        wrapper.memory.eval()

    for epoch in range(args.epochs):
        wrapper.adaptor.train()
        running_loss = 0.0
        for micro_step, batch in enumerate(tqdm(train_loader, desc=f"TriviaQA SFT epoch {epoch + 1}"), 1):
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

            should_step = micro_step % args.grad_accum_steps == 0 or micro_step == len(train_loader)
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.log_every == 0 or global_step == 1:
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

        validation_loss = evaluate_loss(wrapper, eval_loader, set_canon_fn, device)
        torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor_best.pt")
        print(f"Epoch {epoch + 1}: validation answer loss={validation_loss:.6f}")

    log_handle.close()
    metrics = None
    if not args.skip_generation_eval:
        metrics = evaluate_generation(
            wrapper, eval_dataset, set_canon_fn, device, args.max_new_tokens
        )
        print(f"TriviaQA validation EM={metrics['em']:.4f}; F1={metrics['f1']:.4f}")

    elapsed = time.time() - start
    results = {
        "dataset": "TriviaQA",
        "objective": "answer_only_cross_entropy",
        "frozen": ["backbone", "engram_table", "engram_reader"],
        "trainable": ["generator", "generated_reader", "generated_gate"],
        "train_size": len(train_dataset),
        "validation_size": len(eval_dataset),
        "first_loss": first_loss,
        "final_batch_loss": final_loss,
        "validation_loss": validation_loss,
        "generation": metrics,
        "elapsed_hours": elapsed / 3600,
        "completed": True,
    }
    with open(output_dir / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor.pt")
    print("TRAINING_COMPLETED")
    wrapper.cleanup()


if __name__ == "__main__":
    main()
