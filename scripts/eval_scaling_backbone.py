#!/usr/bin/env python3
"""Evaluate an untouched pretrained backbone for scaling-law baselines.

This is deliberately eval-only: it never creates an optimizer and never
performs a backward pass.  The output is paired with the ATHENA memory sweep
on the same corpus validation protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram.data import get_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an untouched pretrained backbone for scaling-law comparisons"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--corpus", required=True, choices=["wikitext", "general-mixed"]
    )
    parser.add_argument("--validation-max-tokens", type=int, default=2_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device: torch.device, use_amp: bool) -> tuple[float, int]:
    model.eval()
    losses: list[float] = []
    token_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        else:
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        losses.append(float(outputs.loss.detach().float().cpu()))
        token_count += int(labels.numel())
    if not losses:
        raise RuntimeError("Validation loader produced no batches")
    return math.exp(sum(losses) / len(losses)), token_count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        use_amp = True
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        use_amp = False

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"torch_dtype": dtype} if use_amp else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)
    model.config.use_cache = False

    loader = get_dataloader(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        corpus=args.corpus,
        split="validation",
        max_tokens=args.validation_max_tokens,
        shuffle=False,
    )
    started = time.time()
    ppl, evaluated_tokens = evaluate(model, loader, device, use_amp)
    results = {
        "completed": True,
        "method": "PretrainedBackbone-EvalOnly",
        "no_memory": True,
        "eval_only": True,
        "model": args.model,
        "corpus": args.corpus,
        "training_tokens": 0,
        "validation_max_tokens": args.validation_max_tokens,
        "evaluated_tokens": evaluated_tokens,
        "validation": {"ppl": ppl},
        "wall_time_seconds": time.time() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "COMPLETED").write_text("completed\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
