#!/usr/bin/env python3
"""Train the no-memory GPT-2 continuation baseline for the scaling sweep.

This is the paper-style ``GPT2-ContTrain`` reference: the GPT-2 backbone is
trained end-to-end with ordinary causal-language-model loss on the same corpus
and token budget as its paired ATHENA memory point.  No Engram, generator,
reader, or router parameters are instantiated.

The script intentionally writes only metrics/configuration artifacts by
default.  The scaling curves need the validation perplexity, not a second copy
of every multi-billion-parameter backbone checkpoint.
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
        description="Train a GPT-2 continued-training baseline for scaling-law comparisons"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--corpus",
        required=True,
        choices=["wikitext", "general-mixed"],
    )
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--validation-max-tokens", type=int, default=2_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        else:
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        losses.append(float(outputs.loss.detach().float().cpu()))
    if not losses:
        raise RuntimeError("Validation loader produced no batches")
    return math.exp(sum(losses) / len(losses))


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
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        use_amp = False
    print(f"Device: {device}; dtype: {dtype}; no_memory_baseline=True")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"torch_dtype": dtype} if use_amp else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    loader_kwargs = dict(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        corpus=args.corpus,
    )
    train_loader = get_dataloader(
        split="train",
        max_tokens=args.max_tokens,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = get_dataloader(
        split="validation",
        max_tokens=args.validation_max_tokens,
        shuffle=False,
        **loader_kwargs,
    )
    train_tokens_available = len(train_loader.dataset) * args.seq_len
    print(
        json.dumps(
            {
                "dataset": args.corpus,
                "requested_train_tokens": args.max_tokens,
                "loaded_train_tokens": train_tokens_available,
                "validation_tokens": len(val_loader.dataset) * args.seq_len,
            },
            sort_keys=True,
        )
    )

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    total_steps = args.max_tokens // tokens_per_step
    if total_steps < 1:
        raise ValueError("max_tokens is too small for one optimizer step")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_schedule(optimizer, args.warmup_steps, total_steps)
    optimizer.zero_grad(set_to_none=True)

    log_handle = (output_dir / "train_log.jsonl").open("w")
    started = time.time()
    step = 0
    micro_step = 0
    epoch = 0
    running_loss = 0.0
    first_loss = None
    last_loss = None
    best_val_ppl = float("inf")
    best_step = 0
    no_improvement_evals = 0
    early_stopped = False

    def run_validation(current_step: int) -> float:
        nonlocal best_val_ppl, best_step, no_improvement_evals, early_stopped
        val_ppl = evaluate(model, val_loader, device, use_amp)
        improved = val_ppl < best_val_ppl - 1e-6
        if improved:
            best_val_ppl = val_ppl
            best_step = current_step
            no_improvement_evals = 0
        else:
            no_improvement_evals += 1
        entry = {
            "type": "eval",
            "step": current_step,
            "validation_ppl": val_ppl,
            "best_validation_ppl": best_val_ppl,
            "no_improvement_evals": no_improvement_evals,
        }
        log_handle.write(json.dumps(entry) + "\n")
        log_handle.flush()
        print(json.dumps(entry, sort_keys=True))
        if (
            args.early_stopping_patience > 0
            and no_improvement_evals >= args.early_stopping_patience
        ):
            early_stopped = True
        return val_ppl

    print(f"Training GPT-2 continuation for at most {total_steps:,} optimizer steps")
    while step < total_steps and not early_stopped:
        epoch += 1
        model.train()
        for batch in train_loader:
            if step >= total_steps or early_stopped:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            else:
                outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss
            if first_loss is None:
                first_loss = float(loss.detach().float().cpu())
            last_loss = float(loss.detach().float().cpu())
            (loss / args.grad_accum_steps).backward()
            running_loss += last_loss
            micro_step += 1
            if micro_step % args.grad_accum_steps:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == 1:
                entry = {
                    "type": "train",
                    "step": step,
                    "epoch": epoch,
                    "loss": running_loss / args.grad_accum_steps,
                    "lr": scheduler.get_last_lr()[0],
                }
                log_handle.write(json.dumps(entry) + "\n")
                log_handle.flush()
                print(json.dumps(entry, sort_keys=True))
                running_loss = 0.0
            if step % args.eval_every == 0:
                run_validation(step)
                model.train()

    final_val_ppl = run_validation(step)
    elapsed = time.time() - started
    results = {
        "completed": True,
        "method": "GPT2-ContTrain",
        "no_memory": True,
        "model": args.model,
        "corpus": args.corpus,
        "max_tokens": args.max_tokens,
        "training_tokens": step * tokens_per_step,
        "actual_steps": step,
        "best_step": best_step,
        "best_validation_ppl": best_val_ppl,
        "final_validation_ppl": final_val_ppl,
        "validation": {"ppl": best_val_ppl},
        "early_stopped": early_stopped,
        "first_train_loss": first_loss,
        "last_train_loss": last_loss,
        "wall_time_seconds": elapsed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "COMPLETED").write_text("completed\n")
    log_handle.close()
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
