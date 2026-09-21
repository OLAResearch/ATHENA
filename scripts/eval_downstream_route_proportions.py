"""Measure realised E/GE/GH routing on one downstream dataset.

This is inference-only.  It reuses the exact QA and DCPMI evaluators, records
the route at positions that produce a downstream score or generated token, and
never uses labels to configure the router.  One invocation handles one task so
Slurm can run datasets independently in parallel.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.eval_dual_reader_openqa_paired import set_reader_mode
from scripts.eval_general_paper_aligned import (
    TASKS as NLP_TASKS,
    evaluate_paper_task,
    load_tasks,
)
from scripts.eval_openqa import (
    TASKS as QA_TASKS,
    TASK_LOADERS,
    evaluate_openqa,
    evaluate_truthfulqa,
    get_model_max_context,
    load_triviaqa_examples,
    setup_condition,
    task_scalar_metric,
)
from scripts.routing_proportion import RoutingProportionCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-kind", choices=("qa", "nlp"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canon-mode", choices=("vocab", "word_boundary"), default="word_boundary")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--triviaqa-config", choices=("rc.nocontext",), default="rc.nocontext")
    parser.add_argument("--reasoning-mode", choices=("vanilla", "cot"), default="vanilla")
    parser.add_argument("--task-data-dir", default=None)
    parser.add_argument("--cb-data", default=None)
    parser.add_argument("--rte-data", default=None)
    parser.add_argument("--agn-data", default=None)
    parser.add_argument("--label-assets-dir", default=None)
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None
    if args.task_kind == "qa" and args.task not in QA_TASKS:
        parser.error(f"unknown QA task: {args.task}")
    if args.task_kind == "nlp" and args.task not in NLP_TASKS:
        parser.error(f"unknown NLP task: {args.task}")
    if args.task_kind == "nlp":
        required = ("task_data_dir", "cb_data", "rte_data", "agn_data")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error("NLP route evaluation requires: " + ", ".join(missing))
    return args


def _write_status(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _condition_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        target_model=args.target_model,
        adaptor_dir=args.adaptor_dir,
        adaptor_checkpoint=args.adaptor_checkpoint,
        source_memory=args.source_memory,
        memory_config=args.memory_config,
        dual_reader_mode="tri_advantage_routed",
        seed=args.seed,
        canon_mode=args.canon_mode,
    )


def _load_nlp_task(args: argparse.Namespace) -> list[dict]:
    tasks = load_tasks(
        Path(args.task_data_dir),
        Path(args.cb_data),
        Path(args.rte_data),
        Path(args.agn_data),
        Path(args.label_assets_dir) if args.label_assets_dir else None,
    )
    examples = tasks[args.task]
    return examples if args.max_examples is None else examples[: args.max_examples]


def _load_qa_task(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.task == "triviaqa":
        return load_triviaqa_examples(args.triviaqa_config)
    return TASK_LOADERS[args.task]()


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    status_path = output_dir / "status.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    _write_status(
        status_path,
        stage="starting",
        task_kind=args.task_kind,
        task=args.task,
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    )

    if args.task_kind == "qa":
        examples, dataset_meta = _load_qa_task(args)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
    else:
        examples = _load_nlp_task(args)
        dataset_meta = {"source": "paper-aligned local task assets", "task": args.task}
    _write_status(
        status_path,
        stage="dataset_loaded",
        task_kind=args.task_kind,
        task=args.task,
        n_examples=len(examples),
        dataset=dataset_meta,
    )

    wrapper, set_canon_fn = setup_condition(
        _condition_args(args), "transferred", device, dtype
    )
    collector = RoutingProportionCollector()
    max_context = get_model_max_context(wrapper, None)
    try:
        set_reader_mode(wrapper, "tri_advantage_routed")
        started = time.time()
        if args.task_kind == "qa":
            if args.task == "truthfulqa":
                metrics = evaluate_truthfulqa(
                    wrapper=wrapper,
                    set_canon_fn=set_canon_fn,
                    tokenizer=wrapper.tokenizer,
                    examples=examples,
                    device=device,
                    max_context_length=max_context,
                    retain_all_examples=True,
                    routing_collector=collector,
                )
            else:
                metrics = evaluate_openqa(
                    wrapper=wrapper,
                    set_canon_fn=set_canon_fn,
                    tokenizer=wrapper.tokenizer,
                    task_name=args.task,
                    examples=examples,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    max_context_length=max_context,
                    reasoning_mode=args.reasoning_mode,
                    retain_all_predictions=True,
                    routing_collector=collector,
                )
            scalar_metric = task_scalar_metric(args.task)
        else:
            metrics = evaluate_paper_task(
                wrapper,
                set_canon_fn,
                examples,
                device,
                max_context,
                batch_size=args.batch_size,
                routing_collector=collector,
            )
            scalar_metric = "accuracy"
        route_report = collector.report()
        if route_report["positions"] <= 0:
            raise RuntimeError("The tri-router emitted no downstream route positions")
        payload = {
            "completed": True,
            "task_kind": args.task_kind,
            "task": args.task,
            "target_model": args.target_model,
            "dataset": dataset_meta,
            "n_examples": len(examples),
            "scalar_metric": scalar_metric,
            "metrics": metrics,
            "routing": route_report,
            "routing_protocol": {
                "reader_mode": "tri_advantage_routed",
                "checkpoint": args.adaptor_checkpoint or args.adaptor_dir,
                "labels_used_only_for_accuracy": args.task_kind == "nlp",
                "inference_only": True,
                "position_scope": route_report["definition"]["positions"],
            },
            "elapsed_s": time.time() - started,
        }
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_status(
            status_path,
            stage="complete",
            task_kind=args.task_kind,
            task=args.task,
            n_examples=len(examples),
            positions=route_report["positions"],
            elapsed_s=payload["elapsed_s"],
        )
        return payload
    finally:
        wrapper.cleanup()


def main() -> None:
    args = parse_args()
    try:
        payload = run(args)
        print(json.dumps({"task": payload["task"], "routing": payload["routing"]}, indent=2))
    except Exception as exc:
        output_dir = Path(args.output_dir)
        _write_status(
            output_dir / "status.json",
            stage="failed",
            task_kind=args.task_kind,
            task=args.task,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
