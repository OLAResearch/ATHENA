"""Inference-only random-memory ablation for a trained tri-reader checkpoint.

This deliberately differs from the ``random_memory`` training ablation: the
adaptor/router checkpoint is loaded from the transferred run first, then only
the Engram memory tensor is replaced with a deterministic fresh initialization.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.memory import EngramMemory, MemoryConfig
from scripts.eval_fair_joint_openqa_paired import (
    _evaluate_tasks,
    load_task,
    set_reader_mode,
)
from scripts.eval_openqa import TASK_SCALAR_METRICS, setup_condition
from scripts.validate_tri_advantage_eval import TASK_COUNTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_COUNTS), required=True)
    parser.add_argument("--triviaqa-config", choices=["rc.nocontext"], default="rc.nocontext")
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--reasoning-mode", choices=["vanilla", "cot"], default="vanilla")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--random-memory-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")
    return args


def load_memory_config(path: str | Path) -> MemoryConfig:
    config = json.loads(Path(path).read_text())
    return MemoryConfig(
        max_ngram=config["max_ngram"],
        heads_per_order=config["heads_per_order"],
        table_size=config["table_size"],
        d_head=config["d_head"],
        hash_seed=config["hash_seed"],
    )


def replace_memory_with_random(wrapper, memory_config: MemoryConfig, seed: int) -> dict:
    """Replace only wrapper.memory and return auditable initialization stats."""
    if wrapper.memory is None:
        raise RuntimeError("Inference-only random-memory ablation requires Engram memory")

    # Keep model/adaptor initialization and random-memory initialization
    # independent. The generated tensor is CPU-created, then copied into the
    # already-loaded memory module on the evaluation device.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        random_memory = EngramMemory(memory_config)
    wrapper.memory.load_state_dict(random_memory.state_dict())
    for parameter in wrapper.memory.parameters():
        parameter.requires_grad = False

    values = torch.cat(
        [parameter.detach().float().cpu().reshape(-1) for parameter in wrapper.memory.parameters()]
    )
    return {
        "seed": int(seed),
        "num_parameters": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _condition_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "adaptor_checkpoint": args.adaptor_checkpoint,
            # Load the transferred checkpoint and force its deployment mode;
            # the memory replacement happens only after checkpoint loading.
            "dual_reader_mode": "tri_advantage_routed",
        }
    )
    return argparse.Namespace(**values)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    task_data = {}
    task_meta = {}
    for task in args.tasks:
        examples, dataset_meta = load_task(task, args.triviaqa_config)
        expected = TASK_COUNTS[task]
        if len(examples) != expected:
            raise RuntimeError(f"{task}: expected {expected}, found {len(examples)}")
        task_data[task] = examples
        task_meta[task] = {
            "dataset": dataset_meta,
            "n_examples": len(examples),
            "scalar_metric": TASK_SCALAR_METRICS[task],
        }
        print(f"Loaded {task}: {len(examples)} examples from {dataset_meta}", flush=True)

    condition_args = _condition_args(args)
    wrapper, canon = setup_condition(condition_args, "transferred", device, dtype)
    try:
        set_reader_mode(wrapper, "tri_advantage_routed")
        memory_config = load_memory_config(args.memory_config)
        random_stats = replace_memory_with_random(
            wrapper, memory_config, args.random_memory_seed
        )
        print("Replaced only Engram memory after loading transferred checkpoint", flush=True)
        print("Random memory stats: " + json.dumps(random_stats), flush=True)

        evaluation = _evaluate_tasks(wrapper, canon, task_data, args, device)
    finally:
        wrapper.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        "completed": True,
        "ablation": "inference_only_random_memory",
        "training_condition": "transferred",
        "runtime_mode": "tri_advantage_routed",
        "adaptor_dir": args.adaptor_dir,
        "adaptor_checkpoint": args.adaptor_checkpoint,
        "source_memory": args.source_memory,
        "memory_config": args.memory_config,
        "random_memory": random_stats,
        "seed": args.seed,
        "triviaqa_config": args.triviaqa_config,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_mode": args.reasoning_mode,
        "tasks": task_meta,
        "evaluation": {"tri_reader_advantage": evaluation},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    for task in args.tasks:
        (output_dir / f"{task}.json").write_text(
            json.dumps({"dataset": task_meta[task], **evaluation[task]}, indent=2)
        )
        metrics = evaluation[task]["metrics"]
        print(
            f"{task}: "
            + json.dumps(
                {
                    key: metrics[key]
                    for key in (("mc1", "mc2", "mc3", "mc_avg")
                                if task == "truthfulqa" else ("em", "f1", "correct", "total"))
                }
            ),
            flush=True,
        )
    print("ATHENA_TRI_INFERENCE_RANDOM_MEMORY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
