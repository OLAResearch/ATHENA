#!/usr/bin/env python3
"""Generate the public-protocol vanilla baseline for BigCodeBench.

Functional scoring is intentionally a separate CPU step using the official
``bigcodebench.evaluate`` package.  This keeps generated code execution out of
the GPU job while preserving the official instruct/full/pass@1 protocol.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--official-chat-template", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def strip_generation_wrappers(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip()


def load_official_dataset():
    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    if "instruct_prompt" not in dataset.column_names:
        raise KeyError(f"BigCodeBench dataset lacks instruct_prompt: {dataset.column_names}")
    return dataset


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    dataset = load_official_dataset()
    full_count = len(dataset)
    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, full_count)))
    print(
        f"OFFICIAL_BIGCODEBENCH_DATASET version=v0.1.4 full={full_count} "
        f"evaluated={len(dataset)} mode=instruct subset=full",
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": args.model,
        "dataset": "bigcode/bigcodebench",
        "dataset_version": "v0.1.4",
        "evaluation_count": len(dataset),
        "mode": "instruct",
        "subset": "full",
        "pass_k": "1",
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "official_chat_template": args.official_chat_template,
        "protocol": "official BigCodeBench instruct prompts + greedy generation + official local evaluator",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.dry_run:
        print("OFFICIAL_BIGCODEBENCH_DRY_RUN_COMPLETE", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.official_chat_template:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "official" / "MemGen"))
        from memgen.utils import CONVERSATION_TEMPLATE

        tokenizer.chat_template = CONVERSATION_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    generation_config = GenerationConfig(
        do_sample=False,
        use_cache=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    samples_path = args.output / "bigcodebench_samples.jsonl"
    started = time.time()
    with samples_path.open("x") as writer, torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            # A Hugging Face Dataset slice returns a column dictionary.  Keep
            # a Dataset object so iteration yields row dictionaries.
            rows = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
            messages = [[{"role": "user", "content": row["instruct_prompt"]}] for row in rows]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                return_dict=True,
            )
            inputs = {key: value.cuda() for key, value in inputs.items()}
            generated = model.generate(**inputs, generation_config=generation_config)
            prompt_len = inputs["input_ids"].shape[1]
            completions = tokenizer.batch_decode(
                generated[:, prompt_len:], skip_special_tokens=True
            )
            for row, raw in zip(rows, completions):
                writer.write(json.dumps({
                    "task_id": row["task_id"],
                    "solution": strip_generation_wrappers(raw),
                    "raw_generation": raw,
                }, ensure_ascii=False) + "\n")
            writer.flush()
            done = min(start + len(rows), len(dataset))
            print(f"OFFICIAL_BIGCODEBENCH_PROGRESS {done}/{len(dataset)}", flush=True)

    summary = {
        **metadata,
        "completed_generation": True,
        "elapsed_s": time.time() - started,
        "samples_file": str(samples_path),
        "functional_accuracy": None,
        "functional_evaluator": "pending official bigcodebench.evaluate CPU step",
    }
    (args.output / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print("OFFICIAL_BIGCODEBENCH_GENERATION_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
