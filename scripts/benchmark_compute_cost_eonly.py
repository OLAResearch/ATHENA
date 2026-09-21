#!/usr/bin/env python3
"""Measure actual inference cost for E-only and full MemoryAthena.

Both conditions use the same frozen backbone, Engram memory, tri-reader
adaptor, and advantage-router checkpoint.  E-only changes only the runtime
reader mode to ``engram_only``; MemoryAthena uses the trained
``tri_advantage_routed`` deployment.  The benchmark is inference-only and
records synchronized device time, token rates, and peak accelerator memory.
Slurm allocation time and GPU-hours are added from ``sacct`` after completion.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

torch = None
_load_condition = None
_measure = None
_prompt = None
_release = None
set_reader_mode = None

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
    parser.add_argument("--router-root-xl", required=True)
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


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _router_root(args: argparse.Namespace, scale: str) -> Path:
    return Path(args.router_root_xl if scale == "xl" else args.router_root)


def _load_memoryathena(spec: dict, source_root: Path, router_root: Path, device, dtype, mode: str):
    wrapper, set_canon_fn = _load_condition(
        spec, source_root, router_root, "augmented", device, dtype
    )
    if mode == "e_only":
        set_reader_mode(wrapper, "engram_only")
    elif mode == "memoryathena":
        set_reader_mode(wrapper, "tri_advantage_routed")
    else:
        raise ValueError(f"unknown benchmark mode: {mode}")
    wrapper.eval()
    return wrapper, set_canon_fn


def main() -> None:
    global torch, _load_condition, _measure, _prompt, _release, set_reader_mode
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        manifest = {
            "scales": args.scales,
            "conditions": ["e_only", "memoryathena"],
            "warmup": args.warmup,
            "iterations": args.iterations,
            "new_tokens": args.new_tokens,
            "max_context_length": args.max_context_length,
            "router_root_xl": args.router_root_xl,
        }
        (output_dir / "dry_run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, sort_keys=True))
        return

    import torch as torch_module
    from scripts.benchmark_compute_cost import (
        _load_condition as load_condition,
        _measure as measure,
        _prompt as prompt_fn,
        _release as release,
    )
    from scripts.eval_dual_reader_openqa_paired import set_reader_mode as set_mode

    torch = torch_module
    _load_condition = load_condition
    _measure = measure
    _prompt = prompt_fn
    _release = release
    set_reader_mode = set_mode

    if not torch.cuda.is_available():
        raise RuntimeError("The E-only/MemoryAthena benchmark requires a LUMI GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    source_root = Path(args.source_root)
    prompt = _prompt()
    results = {
        "completed": False,
        "protocol": {
            "inference_only": True,
            "conditions": ["e_only", "memoryathena"],
            "condition_definition": {
                "e_only": "same frozen Engram memory/adaptor/router deployment with runtime reader mode engram_only",
                "memoryathena": "same deployment with tri_advantage_routed E/GE/GH routing",
            },
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
        router_root = _router_root(args, scale)
        print(f"=== {scale} / {spec['model']} ===", flush=True)
        scale_result = {
            "scale": scale,
            "model": spec["model"],
            "router_root": str(router_root),
            "conditions": {},
        }
        for mode in ("e_only", "memoryathena"):
            started = time.perf_counter()
            wrapper, set_canon_fn = _load_memoryathena(
                spec, source_root, router_root, device, dtype, mode
            )
            _sync()
            load_seconds = time.perf_counter() - started
            measurement = _measure(
                wrapper,
                set_canon_fn,
                prompt=prompt,
                warmup=args.warmup,
                iterations=args.iterations,
                new_tokens=args.new_tokens,
                max_context_length=args.max_context_length,
            )
            measurement["load_seconds"] = load_seconds
            measurement["runtime_reader_mode"] = mode
            scale_result["conditions"][mode] = measurement
            print(json.dumps({"scale": scale, "condition": mode, **measurement}, sort_keys=True), flush=True)
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
