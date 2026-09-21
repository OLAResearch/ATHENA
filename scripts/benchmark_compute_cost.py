#!/usr/bin/env python3
"""Measure inference cost for the scaling-law backbone and augmented model.

The benchmark is deliberately inference-only.  It compares the untouched
backbone with the final corpus-aligned Engram + generated-memory + tri-reader
router deployment at each A-protocol scale.  It records device-side peak
memory and synchronized wall time; Slurm allocation time is collected
separately from ``sacct`` after the job finishes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.eval_dual_reader_openqa_paired import set_reader_mode
from scripts.eval_openqa import configure_advantage_reader, setup_condition


SCALES = {
    "small": {"point": "panel-a-small", "model": "gpt2"},
    "medium": {"point": "panel-a-medium", "model": "gpt2-medium"},
    "large": {"point": "panel-a-large", "model": "gpt2-large"},
    "xl": {"point": "panel-a-xl", "model": "gpt2-xl"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--router-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--max-context-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.new_tokens <= 0:
        parser.error("warmup must be non-negative; iterations and new-tokens must be positive")
    return args


def _prompt() -> str:
    # Fixed synthetic text avoids dataset I/O and labels while keeping a
    # stable causal context for all scales and both conditions.
    return (
        "This is a fixed inference-cost benchmark context. The model should "
        "continue the sequence deterministically and ignore the benchmark "
        "instruction as a task. "
    ) * 12 + "Answer:"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _generate_once(wrapper, tokenizer, prompt: str, device, set_canon_fn, *, max_new_tokens: int, max_context_length: int) -> dict:
    tokenized = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    generated = tokenized["input_ids"].to(device)
    input_tokens = int(generated.shape[1])
    past_key_values = None
    output_tokens = 0

    for _ in range(max_new_tokens):
        if generated.shape[1] > max_context_length:
            generated = generated[:, -max_context_length:]
            past_key_values = None
        if set_canon_fn is not None:
            set_canon_fn(generated)
        model_input = generated if past_key_values is None else generated[:, -1:]
        attention_mask = torch.ones_like(generated, device=device)
        with torch.no_grad():
            outputs = wrapper(
                input_ids=model_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        token_id = int(next_token.item())
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        generated = torch.cat([generated, next_token], dim=1)
        output_tokens += 1

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _args_for_scale(model: str, *, adaptor_dir=None, adaptor_checkpoint=None, source_memory=None, memory_config=None):
    return SimpleNamespace(
        target_model=model,
        adaptor_dir=adaptor_dir,
        adaptor_checkpoint=adaptor_checkpoint,
        source_memory=source_memory,
        memory_config=memory_config,
        dual_reader_mode="auto",
        canon_mode="vocab",
        seed=42,
    )


def _load_condition(spec: dict, source_root: Path, router_root: Path, condition: str, device, dtype):
    point = spec["point"]
    model = spec["model"]
    source_point = source_root / point
    expert_dir = source_point / "tri_experts"
    memory_path = source_point / "source_memory" / "memory.pt"
    memory_config = source_point / "source_memory" / "memory_config.json"
    if condition == "baseline":
        args = _args_for_scale(model)
        return setup_condition(args, "baseline", device, dtype)

    repair_dir = router_root / point / "advantage_router"
    router_checkpoint = repair_dir / "adaptor_best.pt"
    expert_checkpoint = expert_dir / "adaptor_best.pt"
    required = [memory_path, memory_config, expert_dir / "config.json", expert_checkpoint, repair_dir / "config.json", router_checkpoint]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark artifact(s): " + ", ".join(missing))

    repair_cfg = json.loads((repair_dir / "config.json").read_text())
    advantage_reader = repair_cfg.get("advantage_reader")
    if not advantage_reader:
        raise RuntimeError(f"Router config has no advantage_reader metadata: {repair_dir}")

    # The repair checkpoint intentionally contains only the advantage-router
    # weights.  Build the full frozen tri-expert deployment from the original
    # checkpoint, then add and replace only the advantage-router state.
    # Using the original expert directory for setup is important: loading the
    # base expert checkpoint with a newly-created lazy router would fail a
    # strict state-dict check before the router-only repair is applied.
    args = _args_for_scale(
        model,
        adaptor_dir=str(expert_dir),
        adaptor_checkpoint=str(expert_checkpoint),
        source_memory=str(memory_path),
        memory_config=str(memory_config),
    )
    wrapper, set_canon_fn = setup_condition(args, "transferred", device, dtype)
    configure_advantage_reader(wrapper, advantage_reader)
    router_state = torch.load(router_checkpoint, map_location="cpu", weights_only=True)
    incompatible = wrapper.adaptor.load_state_dict(router_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected router checkpoint keys for {point}: {incompatible.unexpected_keys}"
        )
    set_reader_mode(wrapper, "tri_advantage_routed")
    wrapper.eval()
    return wrapper, set_canon_fn


def _measure(wrapper, set_canon_fn, *, prompt: str, warmup: int, iterations: int, new_tokens: int, max_context_length: int):
    tokenizer = wrapper.tokenizer
    for _ in range(warmup):
        _generate_once(
            wrapper, tokenizer, prompt, wrapper.device, set_canon_fn,
            max_new_tokens=new_tokens, max_context_length=max_context_length,
        )
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    samples = []
    started = time.perf_counter()
    for _ in range(iterations):
        samples.append(
            _generate_once(
                wrapper, tokenizer, prompt, wrapper.device, set_canon_fn,
                max_new_tokens=new_tokens, max_context_length=max_context_length,
            )
        )
    _sync()
    elapsed = time.perf_counter() - started
    total_tokens = sum(item["total_tokens"] for item in samples)
    output_tokens = sum(item["output_tokens"] for item in samples)
    input_tokens = sum(item["input_tokens"] for item in samples)
    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    return {
        "iterations": iterations,
        "warmup": warmup,
        "input_tokens_mean": input_tokens / iterations,
        "output_tokens_mean": output_tokens / iterations,
        "total_tokens_mean": total_tokens / iterations,
        "elapsed_seconds": elapsed,
        "seconds_per_sequence": elapsed / iterations,
        "sequences_per_second": iterations / elapsed,
        "input_tokens_per_second": input_tokens / elapsed,
        "output_tokens_per_second": output_tokens / elapsed,
        "total_tokens_per_second": total_tokens / elapsed,
        "peak_memory_allocated_bytes": int(peak_allocated),
        "peak_memory_reserved_bytes": int(peak_reserved),
    }


def _release(wrapper) -> None:
    wrapper.cleanup()
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _sync()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        manifest = {
            "scales": args.scales,
            "conditions": ["baseline", "engram_generated_reader_router"],
            "warmup": args.warmup,
            "iterations": args.iterations,
            "new_tokens": args.new_tokens,
            "max_context_length": args.max_context_length,
        }
        (output_dir / "dry_run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, sort_keys=True))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("The compute-cost benchmark requires a LUMI GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    source_root = Path(args.source_root)
    router_root = Path(args.router_root)
    prompt = _prompt()
    results = {
        "completed": False,
        "protocol": {
            "inference_only": True,
            "conditions": ["baseline", "engram_generated_reader_router"],
            "warmup": args.warmup,
            "iterations": args.iterations,
            "new_tokens": args.new_tokens,
            "max_context_length": args.max_context_length,
            "prompt": "fixed synthetic causal context",
            "dtype": str(dtype),
            "device": torch.cuda.get_device_name(0),
            "hip_version": getattr(torch.version, "hip", None),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "scales": [],
    }
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    for scale in args.scales:
        spec = SCALES[scale]
        print(f"=== {scale} / {spec['model']} ===", flush=True)
        scale_result = {"scale": scale, "model": spec["model"], "conditions": {}}
        for condition in ("baseline", "augmented"):
            started = time.perf_counter()
            wrapper, set_canon_fn = _load_condition(
                spec, source_root, router_root, "baseline" if condition == "baseline" else "augmented", device, dtype
            )
            load_seconds = time.perf_counter() - started
            measurement = _measure(
                wrapper, set_canon_fn, prompt=prompt, warmup=args.warmup,
                iterations=args.iterations, new_tokens=args.new_tokens,
                max_context_length=args.max_context_length,
            )
            measurement["load_seconds"] = load_seconds
            scale_result["conditions"][condition] = measurement
            print(json.dumps({"scale": scale, "condition": condition, **measurement}, sort_keys=True), flush=True)
            _release(wrapper)
        results["scales"].append(scale_result)
        (output_dir / f"{scale}.json").write_text(json.dumps(scale_result, indent=2) + "\n")
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    results["completed"] = True
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "COMPLETED").write_text("completed\n")
    print(json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
