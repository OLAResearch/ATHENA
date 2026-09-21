"""Evaluate Engram-only and the three-model router on the MLP-aligned CR.

This is intentionally the CR-only counterpart of
``vanilla_general_protocol_20260918r2``.  It keeps the exact public
kNN-Prompt formatting and scoring path used by that run:

* context: ``<review> It was``;
* choices: ``negative`` and ``positive``;
* domain-conditional PMI with the original choice-sequence evaluator.

The downstream labels are read only for the final accuracy calculation; they
are never used to configure or train the memory/router.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.eval_dual_reader_openqa_paired import set_reader_mode
from scripts.eval_general_nlp_halueval import _example, evaluate_task
from scripts.eval_openqa import get_model_max_context, setup_condition


def load_cr(path: Path) -> list[dict]:
    examples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            examples.append(
                _example(
                    f"{row['text']} It was",
                    ["negative", "positive"],
                    int(row["label"]),
                    "It was",
                )
            )
    return examples


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
    parser.add_argument("--reader-mode", default="tri_advantage_routed")
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--batch-size", type=int, default=2)
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
        assert torch.cuda.is_available()

    condition_args = SimpleNamespace(
        target_model=args.target_model,
        adaptor_dir=args.adaptor_dir,
        adaptor_checkpoint=args.adaptor_checkpoint,
        source_memory=args.source_memory,
        memory_config=args.memory_config,
        dual_reader_mode=args.reader_mode,
        seed=args.seed,
        canon_mode=args.canon_mode,
    )
    wrapper, set_canon_fn = setup_condition(condition_args, "transferred", device, dtype)
    max_context = get_model_max_context(wrapper, None)
    evaluation = {}
    try:
        for condition, mode in (
            ("engram_only", "engram_only"),
            ("tri_reader_advantage", args.reader_mode),
        ):
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
        "target_model": args.target_model,
        "task": "cr",
        "conditions": ["engram_only", "tri_reader_advantage"],
        "reader_mode": args.reader_mode,
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
