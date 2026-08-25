#!/usr/bin/env python3
"""Vanilla KodCode evaluation using MemGen's public builder and reward.

This deliberately does not import ATHENA's task loader.  The dataset split,
prompt, code extraction, function renaming, and reward all come from the
official MemGen checkout supplied with ``--official-repo``.  For SmolLM3 the
official MemGen ChatML template is selected; other backbones keep their native
tokenizer template and are reported as benchmark-protocol comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--official-chat-template", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_official_test(repo: Path):
    sys.path.insert(0, str(repo.resolve()))
    from data.kodcode.builder import KodCodeBuilder

    builder = KodCodeBuilder(
        {
            "mode": "sft",
            "sft": {"train_ratio": 0.7, "valid_ratio": 0.1, "test_ratio": 0.2},
        }
    )
    return builder.get_dataset_dict()["test"]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    dataset = load_official_test(args.official_repo)
    full_count = len(dataset)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, full_count)))
    print(f"OFFICIAL_KODCODE_DATASET full={full_count} evaluated={len(dataset)}", flush=True)
    print(f"OFFICIAL_KODCODE_PROMPT {dataset[0]['prompt'][0]['content'][:240]}", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": args.model,
        "dataset": "KodCode/KodCode-Light-RL-10K",
        "source_split": "train",
        "split_protocol": "official two-stage train_test_split, ratios 0.7/0.1/0.2, no explicit builder seed",
        "evaluation_count": len(dataset),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "official_chat_template": args.official_chat_template,
        "protocol": "MemGen official KodCode builder + KodCodeEnv reward",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.dry_run:
        print("OFFICIAL_KODCODE_DRY_RUN_COMPLETE", flush=True)
        return

    from data.kodcode.env import KodCodeEnv

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.official_chat_template:
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
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
        max_new_tokens=args.max_new_tokens,
    )

    answer_path = args.output / "answer.jsonl"
    correct = 0.0
    started = time.time()
    with answer_path.open("x") as writer, torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            # A Hugging Face Dataset slice returns a column dictionary.  Keep
            # a Dataset object here so iteration yields row dictionaries and
            # column access below remains available to KodCodeEnv.
            batch = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
            prompts = [row["prompt"] for row in batch]
            inputs = tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=True,
                return_dict=True,
            )
            inputs = {key: value.cuda() for key, value in inputs.items()}
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
            )
            prompt_len = inputs["input_ids"].size(1)
            completions = tokenizer.batch_decode(
                generated[:, prompt_len:], skip_special_tokens=True
            )
            scores = KodCodeEnv.compute_reward(
                completions=completions,
                test=batch["test"],
                test_info=batch["test_info"],
            )
            for row, completion, score in zip(batch, completions, scores):
                correct += float(score)
                writer.write(
                    json.dumps(
                        {
                            "question_id": row.get("question_id"),
                            "prompt": row["prompt"],
                            "completion": completion,
                            "score": score,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            done = min(start + len(completions), len(dataset))
            print(f"OFFICIAL_KODCODE_PROGRESS {done}/{len(dataset)}", flush=True)

    summary = {
        **metadata,
        "accuracy": correct / len(dataset) if dataset else 0.0,
        "correct_score_sum": correct,
        "elapsed_s": time.time() - started,
        "completed": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("OFFICIAL_KODCODE_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
