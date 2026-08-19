"""Phase 2: Train adaptor on target model with transferred/random/etc. memory.

Supports all 8 experimental conditions via the --condition flag:
  baseline, transferred, random_memory, permuted_keys, no_gate,
  train_from_scratch, ffn_only, affine_stitch

Usage:
    uv run python scripts/train_adaptor.py \
        --config configs/tier1_same_tokenizer.yaml \
        --condition transferred \
        --seed 42 \
        --source-memory results/source_memory/memory.pt \
        --target-model EleutherAI/pythia-410m \
        --output-dir results/tier1/transferred_seed42
"""

import argparse
import json
import math
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
from engram.adaptor import build_adaptor
from engram.backbone_wrapper import BackboneWrapper
from engram.canonicalization import build_canonicalizer, WordBoundaryCanonicalizer
from engram.hf_utils import resolve_pretrained_source
from engram.hashing import HashConfig, WordNgramHasher
from engram.data import get_dataloader
from engram.metrics import compute_perplexity, compute_batch_perplexities, bootstrap_ci
from engram.gate_analyzer import GateAnalyzer


VALID_CONDITIONS = [
    "baseline",
    "transferred",
    "random_memory",
    "permuted_keys",
    "no_gate",
    "train_from_scratch",
    "ffn_only",
    "affine_stitch",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Engram adaptor")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--condition", type=str, required=True, choices=VALID_CONDITIONS)
    parser.add_argument("--target-model", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument("--init-adaptor", type=str, default=None,
                        help="Optional adaptor checkpoint to warm-start the target adaptor")
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--validation-max-tokens",
        type=int,
        default=2_000_000,
        help="Validation-token cap (keep the default for full runs; lower only for smoke tests)",
    )
    parser.add_argument("--injection-layers", type=str, default=None,
                        help="Comma-separated layer indices for memory injection")
    parser.add_argument("--adaptor-branches", type=int, default=1,
                        help="Number of branch-specific key/gating paths in the adaptor")
    parser.add_argument("--architecture", choices=["legacy", "generative"], default="legacy")
    parser.add_argument("--reader-type", choices=["cross_attention", "mean"], default="cross_attention")
    parser.add_argument("--generator-cue-source", choices=["engram", "learned"], default="engram")
    parser.add_argument("--generator-num-latents", type=int, default=4)
    parser.add_argument("--generator-hidden-size", type=int, default=256)
    parser.add_argument("--generator-layers", type=int, default=2)
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--generator-cue-window", type=int, default=3)
    parser.add_argument(
        "--generator-fusion-type",
        choices=["generated_only", "engram_residual", "dual_reader"],
        default="generated_only",
        help="Fuse generated latent value alone or retain an Engram residual.",
    )
    parser.add_argument("--memory-dim", type=int, default=None,
                        help="Memory/cue dimension for generative learned controls")
    # Cross-tokenizer mode
    parser.add_argument("--canon-mode", type=str, default="vocab",
                        choices=["vocab", "word_boundary"])
    parser.add_argument("--corpus", type=str, default="wikitext",
                        choices=["wikitext", "wikipedia-2021", "fineweb-edu", "nemotron-cc"],
                        help="Training/eval corpus for adaptor fitting (default: wikitext)")
    parser.add_argument("--corpus-subset", type=str, default="hq-dqa",
                        help="Subset for nemotron-cc: hq-dqa, hq, mhq, all (default: hq-dqa)")
    parser.add_argument("--wikipedia2021-dataset", type=str, default=None,
                        help="Override HF dataset repo for corpus=wikipedia-2021")
    parser.add_argument("--wikipedia2021-source-tokenizer", type=str, default=None,
                        help="Tokenizer that produced the pre-tokenized wikipedia-2021 corpus")
    parser.add_argument("--wikipedia2021-require-tokenizer-match", action="store_true",
                        help="Disable decode+retokenize fallback for wikipedia-2021 and require a tokenizer-matched dataset")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after N evals without val PPL improvement (0=disabled)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing on backbone (saves memory)")
    parser.add_argument("--skip-final-test-eval", action="store_true",
                        help="Skip the final full-test PPL sweep and save adaptor immediately")
    return parser.parse_args()


def load_config(args):
    """Override args with YAML config if provided."""
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        adaptor_cfg = cfg.get("adaptor_training", {})
        for key, val in adaptor_cfg.items():
            key_underscore = key.replace("-", "_")
            if hasattr(args, key_underscore):
                setattr(args, key_underscore, val)
    return args


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_injection_layers(raw) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    raw = str(raw).strip()
    if not raw:
        return None
    return [
        int(part.strip())
        for part in re.split(r"[\s,:;]+", raw)
        if part.strip()
    ]


def enable_frozen_backbone_gradient_checkpointing(wrapper: BackboneWrapper) -> None:
    """Enable checkpointing without severing gradients to injected adaptors.

    Hugging Face's default re-entrant checkpointing requires at least one input
    tensor to require gradients.  ATHENA freezes every backbone parameter and
    injects the trainable adaptor through layer hooks, so the embedding output
    would otherwise be non-differentiable.  This only asks autograd to track the
    embedding output; it does not unfreeze or optimize any backbone parameter.
    """
    wrapper.backbone.gradient_checkpointing_enable()
    enable_input_grads = getattr(wrapper.backbone, "enable_input_require_grads", None)
    if enable_input_grads is None:
        raise RuntimeError(
            "The selected backbone does not expose enable_input_require_grads(), "
            "which is required when gradient checkpointing a frozen backbone"
        )
    enable_input_grads()


def setup_memory(args, device) -> tuple:
    """Set up memory based on condition."""
    # Learned generative cues are an explicit capacity control.  Keep this
    # branch independent of all Engram files and lookup machinery.
    if args.architecture == "generative" and args.generator_cue_source == "learned":
        if args.memory_dim is None:
            raise ValueError("--memory-dim is required for generative learned cue source")
        return None, None, int(args.memory_dim)

    # Load memory config
    with open(args.memory_config) as f:
        mem_cfg_dict = json.load(f)

    # Validate canon_mode matches the source memory pipeline
    source_canon_mode = mem_cfg_dict.get("canon_mode")
    if source_canon_mode is not None and source_canon_mode != args.canon_mode:
        raise ValueError(
            f"Canon mode mismatch: source memory was trained with "
            f"canon_mode='{source_canon_mode}' but --canon-mode='{args.canon_mode}' "
            f"was specified. Using mismatched canonicalization will produce "
            f"invalid hash indices and silently corrupt transfer results."
        )

    mem_cfg = MemoryConfig(
        max_ngram=mem_cfg_dict["max_ngram"],
        heads_per_order=mem_cfg_dict["heads_per_order"],
        table_size=mem_cfg_dict["table_size"],
        d_head=mem_cfg_dict["d_head"],
        hash_seed=mem_cfg_dict["hash_seed"],
    )

    condition = args.condition

    if condition == "baseline":
        return None, mem_cfg, mem_cfg.d_mem

    if condition == "ffn_only":
        return None, mem_cfg, mem_cfg.d_mem

    # Create memory module
    memory = EngramMemory(mem_cfg)

    if condition == "transferred":
        # Load trained source memory, freeze
        memory.load_state_dict(torch.load(args.source_memory, map_location="cpu", weights_only=True))
        for p in memory.parameters():
            p.requires_grad = False

    elif condition == "random_memory":
        # Fresh random init, freeze
        for p in memory.parameters():
            p.requires_grad = False

    elif condition == "permuted_keys":
        # Load trained source memory, permute rows, freeze
        memory.load_state_dict(torch.load(args.source_memory, map_location="cpu", weights_only=True))
        memory.permute_keys(seed=args.seed)
        for p in memory.parameters():
            p.requires_grad = False

    elif condition == "train_from_scratch":
        # Fresh random init, trainable
        pass  # defaults are fine

    elif condition in ("no_gate", "affine_stitch"):
        # Load trained source memory, freeze
        memory.load_state_dict(torch.load(args.source_memory, map_location="cpu", weights_only=True))
        for p in memory.parameters():
            p.requires_grad = False

    return memory, mem_cfg, mem_cfg.d_mem


def main():
    args = parse_args()
    args = load_config(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Use bf16 on GPUs with compute capability >= 8 (A100, H100), fp16 otherwise
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Device: {device}, dtype: {dtype}, Condition: {args.condition}, Seed: {args.seed}")

    torch.manual_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolve_pretrained_source(args.target_model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup memory
    memory, mem_cfg, memory_dim = setup_memory(args, device)
    args.memory_dim = memory_dim

    # Save the complete runtime configuration after resolving memory_dim.
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build wrapper
    wrapper = BackboneWrapper(
        model_name=args.target_model,
        memory=memory,
        condition=args.condition,
        device=device,
        dtype=dtype,
        injection_layers=parse_injection_layers(args.injection_layers),
        adaptor_branches=args.adaptor_branches,
        memory_dim=memory_dim,
        architecture=args.architecture,
        reader_type=args.reader_type,
        generator_cue_source=args.generator_cue_source,
        generator_num_latents=args.generator_num_latents,
        generator_hidden_size=args.generator_hidden_size,
        generator_layers=args.generator_layers,
        generator_heads=args.generator_heads,
        generator_cue_window=args.generator_cue_window,
        generator_fusion_type=args.generator_fusion_type,
    )

    if args.init_adaptor is not None:
        if wrapper.adaptor is None:
            raise ValueError("--init-adaptor was provided but this condition does not build an adaptor")
        init_state = torch.load(args.init_adaptor, map_location="cpu", weights_only=True)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList):
            for adaptor in wrapper.adaptor:
                adaptor.load_state_dict(init_state, strict=True)
            init_params = sum(v.numel() for v in init_state.values()) * len(wrapper.adaptor)
        else:
            wrapper.adaptor.load_state_dict(init_state, strict=True)
            init_params = sum(v.numel() for v in init_state.values())
        print(f"Warm-started adaptor from {args.init_adaptor} ({init_params:,} params)")

    # Freeze backbone always
    wrapper.freeze_backbone()

    # Enable gradient checkpointing if requested (saves memory for large models)
    if args.gradient_checkpointing:
        enable_frozen_backbone_gradient_checkpointing(wrapper)
        print("Gradient checkpointing enabled with differentiable frozen inputs")

    # For train_from_scratch, memory is trainable
    if args.condition == "train_from_scratch":
        wrapper.unfreeze_memory()
    elif memory is not None:
        wrapper.freeze_memory()

    # Build canonicalization only when an Engram memory is actually present.
    canonicalizer = None
    if memory is not None:
        canonicalizer = build_canonicalizer(tokenizer, mode=args.canon_mode, max_ngram=mem_cfg.max_ngram)

    # For vocab mode, build ID map; for word_boundary mode, build hasher
    canon_id_map = None
    word_boundary_canon = None
    word_ngram_hasher = None
    if canonicalizer is not None and args.canon_mode == "vocab" and hasattr(canonicalizer, "build_id_map"):
        canon_id_map = canonicalizer.build_id_map(tokenizer).to(device)
    elif canonicalizer is not None and args.canon_mode == "word_boundary" and isinstance(canonicalizer, WordBoundaryCanonicalizer):
        word_boundary_canon = canonicalizer
        word_ngram_hasher = WordNgramHasher(mem_cfg.hash_config)

    def set_memory_context(wrapper, input_ids):
        """Set canonical IDs or hash indices for the current batch."""
        if memory is None:
            return
        if canon_id_map is not None:
            canon_ids = canon_id_map[input_ids]
            wrapper.set_canon_ids(canon_ids)
        elif word_boundary_canon is not None and word_ngram_hasher is not None:
            word_ngrams = word_boundary_canon.compute_word_ngrams(input_ids)
            indices = word_ngram_hasher.hash_word_ngrams(word_ngrams, device=input_ids.device)
            wrapper.set_hash_indices(indices)

    # Data
    corpus_label = (
        f"{args.corpus}[{args.corpus_subset}]"
        if args.corpus == "nemotron-cc"
        else args.corpus
    )
    print(f"Loading {corpus_label}...")
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
        max_tokens=args.validation_max_tokens,
        shuffle=False,
        corpus=args.corpus,
        corpus_subset=args.corpus_subset,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )
    test_loader = get_dataloader(
        split="test",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=2_000_000,  # Cap test set: nemotron-cc streaming has no None-handling
        shuffle=False,
        corpus=args.corpus,
        corpus_subset=args.corpus_subset,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    total_steps = args.max_tokens // tokens_per_step
    print(f"Total steps: {total_steps:,} (effective batch = {args.batch_size * args.grad_accum_steps})")

    # Optimizer
    trainable_params = wrapper.get_trainable_params()
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_trainable:,}")

    if total_trainable == 0 and args.condition == "baseline":
        # Baseline: just evaluate
        print("\nBaseline condition - evaluating without training...")
        wrapper.eval()
        test_ppls = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test eval"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = wrapper(input_ids=input_ids, labels=labels)
                ppl = float(torch.exp(outputs.loss).item())
                test_ppls.append(ppl)

        mean_ppl, ci_lower, ci_upper = bootstrap_ci(test_ppls)
        results = {
            "condition": args.condition,
            "seed": args.seed,
            "test_ppl_mean": mean_ppl,
            "test_ppl_ci_lower": ci_lower,
            "test_ppl_ci_upper": ci_upper,
            "test_ppl_std": float(torch.tensor(test_ppls).std().item()),
            "n_test_batches": len(test_ppls),
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test PPL: {mean_ppl:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")
        wrapper.cleanup()
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # Training loop
    gate_analyzer = GateAnalyzer()
    log_file = open(output_dir / "train_log.jsonl", "w")
    val_ppls_over_time = []
    step = 0
    epoch = 0
    start_time = time.time()

    # Early stopping state
    best_val_ppl = float("inf")
    best_step = 0
    patience_counter = 0
    patience = args.early_stopping_patience  # 0 = disabled

    print(f"\nTraining adaptor for {total_steps} steps...")

    accum_steps = args.grad_accum_steps
    micro_step = 0
    running_loss = 0.0

    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Set canonical IDs / hash indices for memory lookup
            set_memory_context(wrapper, input_ids)

            # Forward + backward (accumulate gradients)
            outputs = wrapper(input_ids=input_ids, labels=labels)
            loss = outputs.loss / accum_steps
            loss.backward()

            running_loss += loss.item()
            micro_step += 1

            if micro_step % accum_steps != 0:
                continue  # Accumulate more before optimizer step

            # Optimizer step (every accum_steps micro-batches)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # Record gate values
            gate_vals = wrapper.get_last_gate_values()
            if gate_vals is not None and step % args.log_every == 0:
                gate_analyzer.record(gate_vals)

            # Log
            if step % args.log_every == 0:
                grad_norms = wrapper.get_grad_norms()
                elapsed = time.time() - start_time

                log_entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(running_loss),
                    "ppl": float(math.exp(min(running_loss, 20))),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm_backbone": grad_norms["backbone"],
                    "grad_norm_adaptor": grad_norms["adaptor"],
                    "grad_norm_memory": grad_norms["memory"],
                    "elapsed_s": elapsed,
                    "tokens_seen": step * tokens_per_step,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

                print(
                    f"  Step {step}/{total_steps} | "
                    f"Loss {running_loss:.4f} | "
                    f"PPL {math.exp(min(running_loss, 20)):.2f} | "
                    f"Backbone grad: {grad_norms['backbone']:.6f}"
                )

            running_loss = 0.0

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
                val_ppls_over_time.append({"step": step, "val_ppl": val_ppl})
                improved = val_ppl < best_val_ppl

                # Track best and early stopping
                if improved:
                    best_val_ppl = val_ppl
                    best_step = step
                    patience_counter = 0
                    # Save best adaptor checkpoint
                    if wrapper.adaptor is not None:
                        torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor_best.pt")
                    print(f"  >> Val PPL: {val_ppl:.2f} (new best)")
                else:
                    patience_counter += 1
                    print(f"  >> Val PPL: {val_ppl:.2f} (no improvement, patience {patience_counter}/{patience})")

                val_entry = {"step": step, "val_loss": mean_val_loss, "val_ppl": val_ppl, "type": "eval"}
                log_file.write(json.dumps(val_entry) + "\n")
                log_file.flush()
                wrapper.train()

                if patience > 0 and patience_counter >= patience:
                    print(f"\nEarly stopping at step {step} (best val PPL {best_val_ppl:.2f} at step {best_step})")
                    break

        if patience > 0 and patience_counter >= patience:
            break

    log_file.close()

    # Restore best checkpoint if early stopping was used
    best_ckpt = output_dir / "adaptor_best.pt"
    if patience > 0 and best_ckpt.exists() and wrapper.adaptor is not None:
        wrapper.adaptor.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print(f"Restored best adaptor from step {best_step}")

    actual_steps = step
    if args.skip_final_test_eval:
        gate_stats = gate_analyzer.compute_stats()
        elapsed = time.time() - start_time

        results = {
            "condition": args.condition,
            "seed": args.seed,
            "target_model": args.target_model,
            "source_memory": args.source_memory,
            "memory_config": args.memory_config,
            "actual_steps": actual_steps,
            "best_step": best_step,
            "best_val_ppl": None if best_val_ppl == float("inf") else best_val_ppl,
            "test_ppl_mean": None,
            "test_ppl_ci_lower": None,
            "test_ppl_ci_upper": None,
            "test_ppl_std": None,
            "n_test_batches": 0,
            "gate_stats": gate_stats,
            "elapsed_hours": elapsed / 3600,
            "skipped_final_test_eval": True,
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        if wrapper.adaptor is not None:
            torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor.pt")
        if args.condition == "train_from_scratch" and wrapper.memory is not None:
            torch.save(wrapper.memory.state_dict(), output_dir / "memory.pt")

        print("\nSkipped final test evaluation; saved adaptor checkpoint for downstream eval.")
        wrapper.cleanup()
        return

    # Final test evaluation
    print("\nRunning final test evaluation...")
    wrapper.eval()
    test_ppls = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test eval"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            set_memory_context(wrapper, input_ids)
            outputs = wrapper(input_ids=input_ids, labels=labels)
            ppl = float(torch.exp(outputs.loss).item())
            test_ppls.append(ppl)

    mean_ppl, ci_lower, ci_upper = bootstrap_ci(test_ppls)
    print(f"Test PPL: {mean_ppl:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Save results
    gate_stats = gate_analyzer.compute_stats()
    elapsed = time.time() - start_time

    results = {
        "condition": args.condition,
        "seed": args.seed,
        "target_model": args.target_model,
        "corpus": args.corpus,
        "corpus_subset": args.corpus_subset,
        "wikipedia2021_dataset": args.wikipedia2021_dataset,
        "wikipedia2021_source_tokenizer": args.wikipedia2021_source_tokenizer,
        "wikipedia2021_require_tokenizer_match": args.wikipedia2021_require_tokenizer_match,
        "max_tokens": args.max_tokens,
        "total_steps": actual_steps,
        "best_step": best_step if patience > 0 else actual_steps,
        "early_stopped": patience > 0 and patience_counter >= patience,
        "trainable_params": total_trainable,
        "test_ppl_mean": mean_ppl,
        "test_ppl_ci_lower": ci_lower,
        "test_ppl_ci_upper": ci_upper,
        "test_ppl_std": float(torch.tensor(test_ppls).std().item()),
        "n_test_batches": len(test_ppls),
        "gate_stats": gate_stats,
        "val_ppls": val_ppls_over_time,
        "elapsed_hours": elapsed / 3600,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save adaptor checkpoint
    if wrapper.adaptor is not None:
        torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor.pt")
    if args.condition == "train_from_scratch" and wrapper.memory is not None:
        torch.save(wrapper.memory.state_dict(), output_dir / "memory.pt")

    # Save per-batch test PPLs for bootstrap analysis
    with open(output_dir / "test_ppls.json", "w") as f:
        json.dump(test_ppls, f)

    print(f"\nDone! Total time: {elapsed/3600:.2f}h")
    print(f"Results saved to: {output_dir}")

    wrapper.cleanup()


if __name__ == "__main__":
    main()
