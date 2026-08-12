"""Phase 1: Train Engram memory on source model (Pythia-160M).

Trains the memory tables and a source adaptor end-to-end while keeping
the Pythia-160M backbone frozen. The trained memory is then extracted
and frozen for transfer to target models.

Usage:
    uv run python scripts/train_source_memory.py --config configs/pilot.yaml
    uv run python scripts/train_source_memory.py \
        --model EleutherAI/pythia-160m \
        --max-tokens 50000000 \
        --output-dir results/source_memory
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.memory import EngramMemory, MemoryConfig
from engram.adaptor import EngramAdaptor
from engram.backbone_wrapper import BackboneWrapper
from engram.canonicalization import VocabCanonicalizer, WordBoundaryCanonicalizer, build_canonicalizer
from engram.hashing import HashConfig, WordNgramHasher
from engram.data import get_dataloader
from engram.metrics import compute_perplexity
from engram.gate_analyzer import GateAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Engram source memory")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--condition", type=str, default="transferred",
                        choices=["transferred", "memory_only"],
                        help="Source-side injection type")
    parser.add_argument("--max-tokens", type=int, default=50_000_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/source_memory")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--injection-layers", type=str, default=None,
                        help="Comma-separated layer indices for memory injection")
    parser.add_argument("--adaptor-branches", type=int, default=1,
                        help="Number of branch-specific key/gating paths in the adaptor")
    # Memory config
    parser.add_argument("--table-size", type=int, default=65536)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--heads-per-order", type=int, default=4)
    parser.add_argument("--canon-mode", type=str, default="vocab",
                        choices=["vocab", "word_boundary"])
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="Freeze backbone during source training (default: train end-to-end)")
    parser.add_argument("--gate-bias-init", type=float, default=0.0,
                        help="Initial gate bias (e.g., -5.0 for conservative start)")
    parser.add_argument("--memory-lr", type=float, default=None,
                        help="Override learning rate for memory params (default: same as --lr)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to reduce activation memory (for larger models)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)")
    parser.add_argument("--optimizer-8bit", action="store_true", default=False,
                        help="Use bitsandbytes 8-bit AdamW to reduce optimizer memory (~4x savings)")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after N evals without val PPL improvement (0=disabled)")
    parser.add_argument("--corpus", type=str, default="wikitext",
                        choices=["wikitext", "wikipedia-2021", "fineweb-edu", "nemotron-cc"],
                        help="Training corpus (default: wikitext)")
    parser.add_argument("--corpus-subset", type=str, default="hq-dqa",
                        help="Subset for nemotron-cc: hq-dqa, hq, mhq, all (default: hq-dqa)")
    parser.add_argument("--wikipedia2021-dataset", type=str, default=None,
                        help="Override HF dataset repo for corpus=wikipedia-2021")
    parser.add_argument("--wikipedia2021-source-tokenizer", type=str, default=None,
                        help="Tokenizer that produced the pre-tokenized wikipedia-2021 corpus")
    parser.add_argument("--wikipedia2021-require-tokenizer-match", action="store_true",
                        help="Disable decode+retokenize fallback for wikipedia-2021 and require a tokenizer-matched dataset")
    return parser.parse_args()


def load_config(args):
    """Override args with YAML config if provided."""
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        source_cfg = cfg.get("source_training", {})
        for key, val in source_cfg.items():
            key_underscore = key.replace("-", "_")
            if hasattr(args, key_underscore):
                setattr(args, key_underscore, val)
    return args


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine decay learning rate schedule with linear warmup."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_injection_layers(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in re.split(r"[\s,:;]+", raw) if part.strip()]


def main():
    args = parse_args()
    args = load_config(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        dtype = torch.float32
    elif args.freeze_backbone:
        # The frozen backbone still feeds gradients through the hook into the
        # trainable Engram/adaptor.  FP16 can therefore overflow/underflow in
        # that path (and did on GH200).  Use BF16 on modern accelerators for
        # its FP32-sized exponent range; retain FP16 only as a legacy fallback.
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        # BF16 has same 8-bit exponent range as FP32 (no gradient underflow),
        # unlike FP16 which has 5-bit exponent. Safe for end-to-end training.
        dtype = torch.bfloat16
    print(f"Device: {device}, dtype: {dtype}, freeze_backbone: {args.freeze_backbone}")

    # Set seed
    torch.manual_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build memory
    mem_cfg = MemoryConfig(
        max_ngram=args.max_ngram,
        heads_per_order=args.heads_per_order,
        table_size=args.table_size,
        d_head=args.d_head,
        hash_seed=args.seed,
    )
    memory = EngramMemory(mem_cfg)
    print(f"Memory config: d_mem={mem_cfg.d_mem}, total_params={mem_cfg.total_params:,}")

    # Build canonicalizer
    canonicalizer = build_canonicalizer(tokenizer, mode=args.canon_mode, max_ngram=mem_cfg.max_ngram)

    # For vocab mode, build ID map; for word_boundary mode, build hasher
    canon_id_map = None
    word_boundary_canon = None
    word_ngram_hasher = None
    if args.canon_mode == "vocab" and hasattr(canonicalizer, "build_id_map"):
        canon_id_map = canonicalizer.build_id_map(tokenizer).to(device)
    elif args.canon_mode == "word_boundary" and isinstance(canonicalizer, WordBoundaryCanonicalizer):
        word_boundary_canon = canonicalizer
        word_ngram_hasher = WordNgramHasher(mem_cfg.hash_config)

    def set_memory_context(wrapper, input_ids):
        """Set canonical IDs or hash indices for the current batch."""
        if canon_id_map is not None:
            canon_ids = canon_id_map[input_ids]
            wrapper.set_canon_ids(canon_ids)
        elif word_boundary_canon is not None and word_ngram_hasher is not None:
            word_ngrams = word_boundary_canon.compute_word_ngrams(input_ids)
            indices = word_ngram_hasher.hash_word_ngrams(word_ngrams, device=input_ids.device)
            wrapper.set_hash_indices(indices)

    # Build wrapper.
    wrapper = BackboneWrapper(
        model_name=args.model,
        memory=memory,
        condition=args.condition,
        device=device,
        dtype=dtype,
        gate_bias_init=args.gate_bias_init,
        injection_layers=parse_injection_layers(args.injection_layers),
        adaptor_branches=args.adaptor_branches,
    )

    # Enable gradient checkpointing (reduces activation memory for larger models)
    if args.gradient_checkpointing:
        wrapper.backbone.gradient_checkpointing_enable()
        print("Gradient checkpointing ENABLED")

    # Configure freezing
    if args.freeze_backbone:
        print("Backbone FROZEN (memory + adaptor only)")
        wrapper.freeze_backbone()
    else:
        print("Backbone UNFROZEN (end-to-end training)")
    wrapper.unfreeze_memory()

    # Data
    corpus_label = f"{args.corpus}[{args.corpus_subset}]" if args.corpus == "nemotron-cc" else args.corpus
    print(f"Loading {corpus_label} train split...")
    if args.corpus == "wikipedia-2021" and args.wikipedia2021_require_tokenizer_match:
        print("Wikipedia-2021 tokenizer alignment: STRICT")
    train_loader = get_dataloader(
        split="train",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        shuffle=True,
        seed=args.seed,
        corpus=args.corpus,
        corpus_subset=args.corpus_subset,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )
    val_loader = get_dataloader(
        split="validation",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=2_000_000,  # 2M tokens for val
        shuffle=False,
        corpus=args.corpus,
        corpus_subset=args.corpus_subset,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )

    accum = args.grad_accum_steps
    tokens_per_step = args.batch_size * args.seq_len * accum
    total_steps = args.max_tokens // tokens_per_step
    print(f"Tokens/step: {tokens_per_step:,} (batch={args.batch_size} x accum={accum}), Total steps: {total_steps:,}")

    # Optimizer with separate param groups
    backbone_params = [p for p in wrapper.backbone.parameters() if p.requires_grad]
    adaptor_params = [p for p in wrapper.adaptor.parameters() if p.requires_grad] if wrapper.adaptor else []
    memory_params = [p for p in wrapper.memory.parameters() if p.requires_grad] if wrapper.memory else []

    trainable_params = backbone_params + adaptor_params + memory_params
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_trainable:,}")
    print(f"  Backbone: {sum(p.numel() for p in backbone_params):,}")
    print(f"  Adaptor: {sum(p.numel() for p in adaptor_params):,}")
    print(f"  Memory: {sum(p.numel() for p in memory_params):,}")

    memory_lr = args.memory_lr if args.memory_lr is not None else args.lr
    param_groups = []
    if backbone_params:
        # Lower lr for backbone (1/10 of adaptor lr) to fine-tune gently
        param_groups.append({"params": backbone_params, "lr": args.lr * 0.1})
    if adaptor_params:
        param_groups.append({"params": adaptor_params, "lr": args.lr})
    if memory_params:
        param_groups.append({"params": memory_params, "lr": memory_lr})

    if args.optimizer_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups, lr=args.lr, weight_decay=0.01)
        print("Using 8-bit AdamW (bitsandbytes)")
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # Training loop
    gate_analyzer = GateAnalyzer()
    log_file = open(output_dir / "train_log.jsonl", "w")
    step = 0
    epoch = 0
    start_time = time.time()
    best_val_ppl = float("inf")
    best_step = 0
    patience_counter = 0
    patience = args.early_stopping_patience
    best_memory_ckpt = output_dir / "memory_best.pt"
    best_adaptor_ckpt = output_dir / "source_adaptor_best.pt"

    print(f"\nTraining source memory for {total_steps} steps...")

    while step < total_steps:
        epoch += 1
        micro_step = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Set canonical IDs / hash indices for memory lookup
            set_memory_context(wrapper, input_ids)

            # Forward
            outputs = wrapper(input_ids=input_ids, labels=labels)
            loss = outputs.loss / accum  # Scale loss for accumulation

            # Backward (accumulate gradients)
            loss.backward()
            micro_step += 1

            if micro_step < accum:
                continue

            # Optimizer step after accumulating gradients
            # Clip each param group separately to prevent backbone from
            # suppressing memory/adaptor gradients
            if backbone_params:
                torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)
            if adaptor_params:
                torch.nn.utils.clip_grad_norm_(adaptor_params, 1.0)
            if memory_params:
                torch.nn.utils.clip_grad_norm_(memory_params, 1.0)

            # Capture grad norms before zero_grad clears them
            step += 1
            grad_norms_snapshot = wrapper.get_grad_norms() if step % args.log_every == 0 else None

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0

            # Use unscaled loss for logging
            loss = loss * accum

            # Record gate values
            gate_vals = wrapper.get_last_gate_values()
            if gate_vals is not None and step % args.log_every == 0:
                gate_analyzer.record(gate_vals)

            # Log
            if step % args.log_every == 0:
                grad_norms = grad_norms_snapshot
                elapsed = time.time() - start_time
                tokens_seen = step * tokens_per_step

                log_entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "ppl": float(torch.exp(loss).item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm_backbone": grad_norms["backbone"],
                    "grad_norm_adaptor": grad_norms["adaptor"],
                    "grad_norm_memory": grad_norms["memory"],
                    "tokens_seen": tokens_seen,
                    "elapsed_s": elapsed,
                    "tokens_per_sec": tokens_seen / elapsed,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

                print(
                    f"  Step {step}/{total_steps} | "
                    f"Loss {loss.item():.4f} | "
                    f"PPL {torch.exp(loss).item():.2f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e} | "
                    f"Backbone grad: {grad_norms['backbone']:.6f}"
                )

            # Eval
            if step % args.eval_every == 0:
                wrapper.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_ids = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        set_memory_context(wrapper, val_ids)
                        val_out = wrapper(input_ids=val_ids, labels=val_labels)
                        val_losses.append(val_out.loss.item())

                mean_val_loss = sum(val_losses) / len(val_losses)
                val_ppl = float(torch.exp(torch.tensor(mean_val_loss)).item())
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_step = step
                    patience_counter = 0
                    torch.save(memory.state_dict(), best_memory_ckpt)
                    torch.save(wrapper.adaptor.state_dict(), best_adaptor_ckpt)
                    print(f"  >> Val PPL: {val_ppl:.2f} (loss: {mean_val_loss:.4f}, new best)")
                else:
                    patience_counter += 1
                    print(
                        f"  >> Val PPL: {val_ppl:.2f} (loss: {mean_val_loss:.4f}, "
                        f"no improvement, patience {patience_counter}/{patience})"
                    )

                val_entry = {
                    "step": step,
                    "val_loss": mean_val_loss,
                    "val_ppl": val_ppl,
                    "type": "eval",
                }
                log_file.write(json.dumps(val_entry) + "\n")
                log_file.flush()
                wrapper.train()

                if patience > 0 and patience_counter >= patience:
                    print(f"\nEarly stopping at step {step} (best val PPL {best_val_ppl:.2f} at step {best_step})")
                    break

        if patience > 0 and patience_counter >= patience:
            break

    log_file.close()

    if patience > 0 and best_memory_ckpt.exists() and best_adaptor_ckpt.exists():
        memory.load_state_dict(torch.load(best_memory_ckpt, map_location=device, weights_only=True))
        wrapper.adaptor.load_state_dict(torch.load(best_adaptor_ckpt, map_location=device, weights_only=True))
        print(f"Restored best source memory/adaptor from step {best_step}")

    # Save memory checkpoint
    print("\nSaving memory checkpoint...")
    torch.save(memory.state_dict(), output_dir / "memory.pt")
    torch.save(wrapper.adaptor.state_dict(), output_dir / "source_adaptor.pt")

    # Save memory config
    with open(output_dir / "memory_config.json", "w") as f:
        json.dump({
            "max_ngram": mem_cfg.max_ngram,
            "heads_per_order": mem_cfg.heads_per_order,
            "table_size": mem_cfg.table_size,
            "d_head": mem_cfg.d_head,
            "hash_seed": mem_cfg.hash_seed,
            "d_mem": mem_cfg.d_mem,
            "total_params": mem_cfg.total_params,
            "canon_mode": args.canon_mode,
        }, f, indent=2)

    # Save gate stats
    gate_stats = gate_analyzer.compute_stats()
    with open(output_dir / "gate_stats.json", "w") as f:
        json.dump(gate_stats, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed/3600:.2f}h")
    print(f"Memory saved to: {output_dir / 'memory.pt'}")

    wrapper.cleanup()


if __name__ == "__main__":
    main()
