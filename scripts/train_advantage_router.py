"""Train a conservative task-agnostic residual router on Wikipedia only.

The direct Engram reader and generated-memory residual remain frozen.  Each
injection layer independently learns how much of the generated residual to add
through the real frozen downstream backbone.  The only objective is causal
next-token loss plus a small residual-usage penalty; no downstream QA examples
or globally shared counterfactual labels are loaded by this script.
"""

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.backbone_wrapper import BackboneWrapper
from engram.data import get_dataloader
from engram.generative_memory import GenerativeMemoryAdaptor
from engram.memory import EngramMemory, MemoryConfig
from scripts.eval_openqa import build_canon_fn
from scripts.train_adaptor import (
    enable_frozen_backbone_gradient_checkpointing,
    parse_injection_layers,
)


ROUTER_MODES = ("engram_only", "both")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
    parser.add_argument("--validation-max-tokens", type=int, default=200_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--router-hidden-size", type=int, default=16)
    parser.add_argument("--router-semantic-size", type=int, default=0)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--residual-usage-penalty", type=float, default=1e-3)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", choices=["wikipedia-2021"], default="wikipedia-2021")
    parser.add_argument("--wikipedia2021-dataset", default=None)
    parser.add_argument("--wikipedia2021-source-tokenizer", default=None)
    parser.add_argument("--wikipedia2021-require-tokenizer-match", action="store_true")
    return parser.parse_args()


def _adaptors(wrapper: BackboneWrapper) -> list[GenerativeMemoryAdaptor]:
    modules = (
        list(wrapper.adaptor)
        if isinstance(wrapper.adaptor, nn.ModuleList)
        else [wrapper.adaptor]
    )
    if not modules or not all(isinstance(module, GenerativeMemoryAdaptor) for module in modules):
        raise TypeError("Advantage routing requires generative-memory adaptors")
    return modules


def set_reader_mode(wrapper: BackboneWrapper, mode: str) -> None:
    for adaptor in _adaptors(wrapper):
        adaptor.set_dual_reader_mode(mode)


def configure_router_training(
    wrapper: BackboneWrapper, temperature: float
) -> list[str]:
    names = []
    modules = _adaptors(wrapper)
    for index, adaptor in enumerate(modules):
        adaptor.configure_router(temperature=temperature, hard=False)
        local_names = adaptor.train_router_only()
        prefix = f"{index}." if len(modules) > 1 else ""
        names.extend(prefix + name for name in local_names)
    return names


def load_source_config(adaptor_dir: str) -> dict:
    path = Path(adaptor_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source adaptor config: {path}")
    with open(path) as handle:
        return json.load(handle)


def load_experts_allowing_new_router(
    adaptor: nn.Module, checkpoint: str
) -> tuple[list[str], list[str]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = adaptor.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected expert checkpoint tensors: {unexpected}")
    if not missing or not all("router" in name for name in missing):
        raise ValueError(
            "Expert checkpoint may omit only newly initialized router tensors; "
            f"missing={missing}"
        )
    return missing, unexpected


def build_wrapper(args, device: torch.device, dtype: torch.dtype):
    source_config = load_source_config(args.adaptor_dir)
    with open(args.memory_config) as handle:
        memory_config_dict = json.load(handle)
    memory_config = MemoryConfig(
        max_ngram=memory_config_dict["max_ngram"],
        heads_per_order=memory_config_dict["heads_per_order"],
        table_size=memory_config_dict["table_size"],
        d_head=memory_config_dict["d_head"],
        hash_seed=memory_config_dict["hash_seed"],
    )
    memory = EngramMemory(memory_config)
    memory.load_state_dict(
        torch.load(args.source_memory, map_location="cpu", weights_only=True)
    )
    for parameter in memory.parameters():
        parameter.requires_grad = False

    wrapper = BackboneWrapper(
        model_name=args.target_model,
        memory=memory,
        condition="transferred",
        device=device,
        dtype=dtype,
        injection_layers=parse_injection_layers(source_config.get("injection_layers")),
        adaptor_branches=int(source_config.get("adaptor_branches", 1)),
        memory_dim=memory_config.d_mem,
        architecture="generative",
        reader_type=source_config.get("reader_type", "cross_attention"),
        generator_cue_source=source_config.get("generator_cue_source", "engram"),
        generator_num_latents=int(source_config.get("generator_num_latents", 4)),
        generator_hidden_size=int(source_config.get("generator_hidden_size", 256)),
        generator_layers=int(source_config.get("generator_layers", 2)),
        generator_heads=int(source_config.get("generator_heads", 4)),
        generator_cue_window=int(source_config.get("generator_cue_window", 3)),
        generator_fusion_type="dual_reader",
        generator_adaptive_router=True,
        generator_router_hidden_size=args.router_hidden_size,
        generator_router_semantic_size=getattr(args, "router_semantic_size", 0),
        generator_loop_rounds=int(source_config.get("generator_loop_rounds", 1)),
        generator_loop_workspace_size=int(
            source_config.get("generator_loop_workspace_size", 0)
        ),
        generator_loop_gate_max=float(
            source_config.get("generator_loop_gate_max", 0.25)
        ),
    )
    checkpoint = args.adaptor_checkpoint or str(Path(args.adaptor_dir) / "adaptor_best.pt")
    missing, _ = load_experts_allowing_new_router(wrapper.adaptor, checkpoint)
    wrapper.freeze_backbone()
    wrapper.freeze_memory()
    set_canon_fn = build_canon_fn(
        wrapper, memory_config_dict, args.canon_mode, device
    )
    return wrapper, set_canon_fn, source_config, checkpoint, missing


def router_weight_stats(wrapper: BackboneWrapper) -> list[float]:
    values = []
    for adaptor in _adaptors(wrapper):
        weights = adaptor.get_last_router_weights()
        if weights is not None:
            values.append(weights.detach().float().mean(dim=(0, 1)))
    if not values:
        return [0.0, 0.0]
    return [float(value) for value in torch.stack(values).mean(dim=0).cpu()]


def residual_usage(wrapper: BackboneWrapper) -> torch.Tensor:
    """Differentiable mean probability of adding the generated residual."""
    values = []
    for adaptor in _adaptors(wrapper):
        weights = adaptor.get_last_router_weights()
        if weights is None:
            raise RuntimeError("Routed forward did not expose router weights")
        values.append(weights[..., 1].mean())
    return torch.stack(values).mean()


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def scale(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def evaluate_ppl(wrapper, set_canon_fn, loader, device, mode: str) -> tuple[float, list[float]]:
    set_reader_mode(wrapper, mode)
    wrapper.eval()
    losses = []
    weight_sums = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            set_canon_fn(input_ids)
            outputs = wrapper(input_ids=input_ids, labels=labels, use_cache=False)
            losses.append(float(outputs.loss.item()))
            if mode == "routed":
                weight_sums.append(router_weight_stats(wrapper))
    if not losses:
        raise RuntimeError("Wikipedia validation loader produced no batches")
    mean_loss = sum(losses) / len(losses)
    mean_weights = (
        [sum(row[i] for row in weight_sums) / len(weight_sums) for i in range(2)]
        if weight_sums else [0.0, 0.0]
    )
    return math.exp(mean_loss), mean_weights


def save_checkpoint(wrapper, output_dir: Path, name: str) -> None:
    torch.save(wrapper.adaptor.state_dict(), output_dir / name)


def main():
    args = parse_args()
    if args.max_tokens < args.batch_size * args.seq_len * args.grad_accum_steps:
        raise ValueError("max_tokens is too small for one optimizer step")
    if args.residual_usage_penalty < 0:
        raise ValueError("residual_usage_penalty must be non-negative")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    wrapper, set_canon_fn, source_config, checkpoint, missing = build_wrapper(
        args, device, dtype
    )
    if args.gradient_checkpointing:
        enable_frozen_backbone_gradient_checkpointing(wrapper)
        wrapper.backbone.config.use_cache = False
        print("Gradient checkpointing enabled with differentiable frozen inputs")
    tokenizer = wrapper.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable_names = configure_router_training(wrapper, args.router_temperature)
    trainable = wrapper.get_trainable_params()
    if not trainable or not all(
        ("router" in name or "loop_workspace" in name) for name in trainable_names
    ):
        raise RuntimeError(f"Unexpected router trainable boundary: {trainable_names}")
    if any(parameter.requires_grad for parameter in wrapper.backbone.parameters()):
        raise RuntimeError("Backbone must remain frozen")
    if any(parameter.requires_grad for parameter in wrapper.memory.parameters()):
        raise RuntimeError("Engram memory must remain frozen")
    print(f"Loaded frozen experts from {checkpoint}")
    print(f"Initialized router tensors: {len(missing)}")
    print(f"Trainable router parameters: {sum(p.numel() for p in trainable):,}")
    print("Trainable tensors:\n  " + "\n  ".join(trainable_names))

    train_loader = get_dataloader(
        split="train",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        shuffle=True,
        seed=args.seed,
        corpus=args.corpus,
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
        seed=args.seed,
        corpus=args.corpus,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    total_steps = args.max_tokens // tokens_per_step
    print(
        f"Wikipedia-only router training: {args.max_tokens:,} tokens, "
        f"{total_steps:,} steps, residual penalty {args.residual_usage_penalty:g}"
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps)

    runtime_config = dict(source_config)
    runtime_config.update(vars(args))
    runtime_config.update({
        "architecture": "generative",
        "generator_fusion_type": "dual_reader",
        "generator_adaptive_router": True,
        "generator_router_hidden_size": args.router_hidden_size,
        "generator_router_semantic_size": args.router_semantic_size,
        "router_experts": ["engram_only", "engram_plus_generated_residual"],
        "router_strategy": "layer_local_safe_residual_end_to_end",
        "router_training_data": "wikipedia-2021-causal-next-token-only",
        "downstream_training_examples": 0,
        "source_expert_checkpoint": checkpoint,
    })
    with open(output_dir / "config.json", "w") as handle:
        json.dump(runtime_config, handle, indent=2)

    log_handle = open(output_dir / "train_log.jsonl", "w")
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    epoch = 0
    best_val_ppl = float("inf")
    best_step = 0
    started = time.time()
    running = {"lm": 0.0, "penalty": 0.0, "total": 0.0, "steps": 0}

    wrapper.train()
    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            set_reader_mode(wrapper, "routed")
            set_canon_fn(input_ids)
            outputs = wrapper(input_ids=input_ids, labels=labels, use_cache=False)
            lm_loss = outputs.loss
            usage = residual_usage(wrapper)
            usage_penalty = args.residual_usage_penalty * usage
            total_loss = lm_loss + usage_penalty
            (total_loss / args.grad_accum_steps).backward()

            running["lm"] += float(lm_loss.item())
            running["penalty"] += float(usage_penalty.item())
            running["total"] += float(total_loss.item())
            running["steps"] += 1
            micro_step += 1
            if micro_step % args.grad_accum_steps:
                continue

            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == 1:
                denominator = max(1, running["steps"])
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "lm_loss": running["lm"] / denominator,
                    "residual_usage_penalty": running["penalty"] / denominator,
                    "total_loss": running["total"] / denominator,
                    "router_weights": router_weight_stats(wrapper),
                    "router_grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": step * tokens_per_step,
                    "elapsed_s": time.time() - started,
                }
                log_handle.write(json.dumps(entry) + "\n")
                log_handle.flush()
                print(
                    f"Step {step}/{total_steps} | LM {entry['lm_loss']:.4f} | "
                    f"penalty {entry['residual_usage_penalty']:.6f} | "
                    f"grad {grad_norm:.4f} | "
                    f"weights {entry['router_weights']}"
                )
                running = {"lm": 0.0, "penalty": 0.0, "total": 0.0, "steps": 0}

            if step % args.eval_every == 0 or step == total_steps:
                val_ppl, val_weights = evaluate_ppl(
                    wrapper, set_canon_fn, val_loader, device, "routed"
                )
                print(
                    f">> Wikipedia validation routed PPL {val_ppl:.4f}; "
                    f"weights {val_weights}"
                )
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_step = step
                    save_checkpoint(wrapper, output_dir, "adaptor_best.pt")
                    print(f">> New best router at step {step}")
                wrapper.train()

    log_handle.close()
    if best_step == 0:
        best_step = step
        save_checkpoint(wrapper, output_dir, "adaptor_best.pt")
    wrapper.adaptor.load_state_dict(
        torch.load(output_dir / "adaptor_best.pt", map_location=device, weights_only=True)
    )

    validation = {}
    for mode in (*ROUTER_MODES, "routed"):
        ppl, weights = evaluate_ppl(wrapper, set_canon_fn, val_loader, device, mode)
        validation[mode] = {"ppl": ppl, "router_weights": weights if mode == "routed" else None}
        print(f"FINAL Wikipedia validation {mode}: PPL={ppl:.4f}")

    results = {
        "completed": True,
        "training_data": "wikipedia-2021-causal-next-token-only",
        "downstream_training_examples": 0,
        "max_tokens": args.max_tokens,
        "actual_steps": step,
        "best_step": best_step,
        "best_val_ppl": best_val_ppl,
        "validation": validation,
        "trainable_names": trainable_names,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    with open(output_dir / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    save_checkpoint(wrapper, output_dir, "adaptor.pt")
    print("ATHENA_WIKIPEDIA_ADVANTAGE_ROUTER_TRAINING_COMPLETE")
    wrapper.cleanup()


if __name__ == "__main__":
    main()
