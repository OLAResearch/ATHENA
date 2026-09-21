"""Evaluate matched inference-only router and memory controls on CR.

The evaluator keeps the exact paper-aligned CR prompt and dCPMI scorer used by
``eval_cr_mlp_aligned_router.py``. It compares the normal E-anchored router
with two controls while keeping the trained checkpoint fixed:

* ``random_router`` permutes realised E/GE/GH weights across token positions;
* ``random_memory_inference`` replaces only the Engram table with a fresh,
  same-configuration random initialization.

No CR labels are used for routing or control construction. They are consumed
only by the final accuracy scorer.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from engram.tri_memory import TriMemoryAdaptor
from scripts.eval_cr_mlp_aligned_router import load_cr
from scripts.eval_dual_reader_openqa_paired import set_reader_mode
from scripts.eval_openqa import get_model_max_context, setup_condition
from scripts.eval_tri_inference_random_memory import (
    load_memory_config,
    replace_memory_with_random,
)
from scripts.eval_general_nlp_halueval import evaluate_task


def _adaptors(wrapper):
    if isinstance(wrapper.adaptor, torch.nn.ModuleList):
        return list(wrapper.adaptor)
    return [wrapper.adaptor]


def _configure_random_router(wrapper, seed: int) -> None:
    for adaptor in _adaptors(wrapper):
        if not isinstance(adaptor, TriMemoryAdaptor):
            delegate = getattr(adaptor, "_tri_delegate", None)
            if not isinstance(delegate, TriMemoryAdaptor):
                raise TypeError("Random router requires a tri-reader adaptor")
            adaptor = delegate
        adaptor.configure_random_router(seed)


def _restore_memory(wrapper, state) -> None:
    if wrapper.memory is None:
        raise RuntimeError("Memory control requires an Engram memory")
    wrapper.memory.load_state_dict(state, strict=True)
    for parameter in wrapper.memory.parameters():
        parameter.requires_grad = False


def _release(wrapper) -> None:
    cleanup = getattr(wrapper, "cleanup", None)
    if cleanup is not None:
        cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--cr-data", required=True)
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", required=True)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--random-router-seed", type=int, default=42)
    parser.add_argument("--random-memory-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "results.json"
    if result_file.exists():
        raise FileExistsError(f"Refusing to overwrite {result_file}")

    torch.manual_seed(args.seed)
    examples = load_cr(Path(args.cr_data))
    if not examples:
        raise ValueError("CR task data is empty")
    print(f"Loaded CR examples: {len(examples)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    condition_args = SimpleNamespace(
        target_model=args.target_model,
        adaptor_dir=args.adaptor_dir,
        adaptor_checkpoint=args.adaptor_checkpoint,
        source_memory=args.source_memory,
        memory_config=args.memory_config,
        dual_reader_mode="tri_advantage_routed",
        seed=args.seed,
        canon_mode=args.canon_mode,
    )
    wrapper, set_canon_fn = setup_condition(condition_args, "transferred", device, dtype)
    max_context = get_model_max_context(wrapper, None)
    if wrapper.memory is None:
        _release(wrapper)
        raise RuntimeError("Transferred checkpoint did not expose Engram memory")
    original_memory = {
        name: value.detach().cpu().clone()
        for name, value in wrapper.memory.state_dict().items()
    }
    memory_config = load_memory_config(args.memory_config)
    evaluation = {}
    random_memory_stats = None
    conditions = (
        ("engram_only", "engram_only"),
        ("tri_reader_advantage", "tri_advantage_routed"),
        ("random_router", "tri_random_advantage_routed"),
        ("random_memory_inference", "tri_advantage_routed"),
    )
    try:
        for condition, mode in conditions:
            _restore_memory(wrapper, original_memory)
            if condition == "random_router":
                _configure_random_router(wrapper, args.random_router_seed)
            elif condition == "random_memory_inference":
                random_memory_stats = replace_memory_with_random(
                    wrapper, memory_config, args.random_memory_seed
                )
            set_reader_mode(wrapper, mode)
            started = time.time()
            result = evaluate_task(
                wrapper,
                set_canon_fn,
                examples,
                device,
                max_context,
                pmi=True,
                batch_size=args.batch_size,
            )
            result["elapsed_s"] = time.time() - started
            evaluation[condition] = result
            print(
                f"{condition}/cr: {result['correct']}/{result['total']} "
                f"= {result['accuracy']:.6f}",
                flush=True,
            )
    finally:
        _release(wrapper)

    payload = {
        "completed": True,
        "target_model": args.target_model,
        "task": "cr",
        "conditions": [name for name, _ in conditions],
        "protocol": "domain_conditional_pmi",
        "scoring": "original_kNN_prompt_choice_sequence_logprob_mean",
        "source_url": "https://github.com/swj0419/kNN_prompt/tree/main/task_data",
        "evaluation_split": "kNN_prompt/task_data/cr/test.csv",
        "task_data": str(Path(args.cr_data)),
        "prompt_protocol": {
            "prompt": "<input> It was",
            "choices": ["negative", "positive"],
            "domain_context": "It was",
        },
        "task_size": len(examples),
        "labels_used_only_for_accuracy": True,
        "random_router": {
            "seed": args.random_router_seed,
            "construction": "permute trained three-source E/GE/GH weights across batch-time positions",
            "memory_and_readers_fixed": True,
        },
        "random_memory_inference": {
            "seed": args.random_memory_seed,
            "memory_config": args.memory_config,
            "adaptor_and_router_fixed": True,
            "stats": random_memory_stats,
        },
        "memory": {
            "source_memory": args.source_memory,
            "memory_config": args.memory_config,
        },
        "router": {
            "adaptor_dir": args.adaptor_dir,
            "adaptor_checkpoint": args.adaptor_checkpoint,
        },
        "evaluation": evaluation,
        "summary": {name: result["accuracy"] for name, result in evaluation.items()},
    }
    result_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "status.json").write_text(
        json.dumps({"stage": "complete", "summary": payload["summary"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {result_file}", flush=True)


if __name__ == "__main__":
    main()
