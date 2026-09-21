"""Inference-only random-router evaluation on the five QA benchmarks.

The transferred tri-reader checkpoint, Engram memory, and reader parameters
remain fixed.  The only intervention is the token-wise permutation of the
trained E/GE/GH route weights implemented by ``tri_random_advantage_routed``.
Benchmark labels are consumed only by the final accuracy/EM/F1 scorer.
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.tri_memory import TriMemoryAdaptor
from scripts.eval_dual_reader_openqa_paired import (
    TASKS,
    compact_metrics,
    load_task,
    set_reader_mode,
)
from scripts.eval_openqa import (
    TASK_SCALAR_METRICS,
    evaluate_openqa,
    evaluate_truthfulqa,
    get_model_max_context,
    setup_condition,
    task_scalar_score,
)


EXPECTED_COUNTS = {
    "nq": 3609,
    "webqa": 2032,
    "triviaqa": 17944,
    "truthfulqa": 817,
    "hotpotqa": 7405,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", required=True)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument(
        "--triviaqa-config", choices=["rc.nocontext"], default="rc.nocontext"
    )
    parser.add_argument(
        "--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary"
    )
    parser.add_argument("--reasoning-mode", choices=["vanilla", "cot"], default="vanilla")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--random-router-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")
    return args


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


def _release(wrapper) -> None:
    cleanup = getattr(wrapper, "cleanup", None)
    if cleanup is not None:
        cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_tasks(wrapper, set_canon_fn, task_data, args, device):
    tokenizer = wrapper.tokenizer
    max_context = get_model_max_context(wrapper, args.max_context_length)
    results = {}
    for task in args.tasks:
        started = time.time()
        if task == "truthfulqa":
            metrics = evaluate_truthfulqa(
                wrapper=wrapper,
                set_canon_fn=set_canon_fn,
                tokenizer=tokenizer,
                examples=task_data[task],
                device=device,
                max_context_length=max_context,
                retain_all_examples=True,
            )
        else:
            metrics = evaluate_openqa(
                wrapper=wrapper,
                set_canon_fn=set_canon_fn,
                tokenizer=tokenizer,
                task_name=task,
                examples=task_data[task],
                device=device,
                max_new_tokens=args.max_new_tokens,
                max_context_length=max_context,
                reasoning_mode=args.reasoning_mode,
                retain_all_predictions=True,
            )
        metrics["elapsed_s"] = time.time() - started
        results[task] = {"metrics": metrics}
        print(f"{task}: {json.dumps(compact_metrics(task, metrics))}", flush=True)
    return results


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
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    task_data = {}
    task_meta = {}
    for task in args.tasks:
        examples, dataset_meta = load_task(task, args.triviaqa_config)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        task_data[task] = examples
        task_meta[task] = {
            "dataset": dataset_meta,
            "n_examples": len(examples),
            "expected_full_count": EXPECTED_COUNTS[task],
            "full_data": args.max_examples is None,
            "scalar_metric": TASK_SCALAR_METRICS[task],
        }
        print(f"Loaded {task}: {len(examples)} examples from {dataset_meta}", flush=True)

    wrapper = None
    try:
        condition_args = argparse.Namespace(
            target_model=args.target_model,
            adaptor_dir=args.adaptor_dir,
            adaptor_checkpoint=args.adaptor_checkpoint,
            source_memory=args.source_memory,
            memory_config=args.memory_config,
            dual_reader_mode="tri_advantage_routed",
            seed=args.seed,
            canon_mode=args.canon_mode,
        )
        wrapper, set_canon_fn = setup_condition(
            condition_args, "transferred", device, dtype
        )
        _configure_random_router(wrapper, args.random_router_seed)
        set_reader_mode(wrapper, "tri_random_advantage_routed")
        started = time.time()
        evaluation = _evaluate_tasks(
            wrapper, set_canon_fn, task_data, args, device
        )
        elapsed_s = time.time() - started
    finally:
        if wrapper is not None:
            _release(wrapper)

    scalar_summary = {
        task: task_scalar_score(task, evaluation[task]["metrics"])
        for task in args.tasks
    }
    scalar_summary["macro_average"] = float(np.mean(list(scalar_summary.values())))
    payload = {
        "completed": True,
        "target_model": args.target_model,
        "protocol": "vanilla_qa_five_dataset",
        "tasks": task_meta,
        "conditions": ["random_router"],
        "evaluation": {"random_router": evaluation},
        "scalar_summary": {"random_router": scalar_summary},
        "random_router": {
            "seed": args.random_router_seed,
            "construction": "permute trained three-source E/GE/GH weights across batch-time positions",
            "memory_and_readers_fixed": True,
            "router_checkpoint_fixed": True,
            "labels_used_only_for_accuracy": True,
        },
        "inference_only": True,
        "adaptor_dir": args.adaptor_dir,
        "adaptor_checkpoint": args.adaptor_checkpoint,
        "source_memory": args.source_memory,
        "memory_config": args.memory_config,
        "canon_mode": args.canon_mode,
        "reasoning_mode": args.reasoning_mode,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "elapsed_s": elapsed_s,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "status.json").write_text(
        json.dumps(
            {
                "stage": "complete",
                "tasks": list(args.tasks),
                "n_examples": {task: task_meta[task]["n_examples"] for task in args.tasks},
                "macro_average": scalar_summary["macro_average"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("ATHENA_RANDOM_ROUTER_QA5_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
