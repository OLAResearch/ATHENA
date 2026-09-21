"""Evaluate the bare Mistral baseline on the CB validation split."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.eval_general_nlp_halueval import evaluate_task, load_cb
from scripts.eval_openqa import get_model_max_context, setup_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    baseline_args = SimpleNamespace(
        target_model=args.target_model,
        dual_reader_mode="auto",
    )
    wrapper, canon = setup_condition(baseline_args, "baseline", device, dtype)
    max_context = get_model_max_context(wrapper, None)
    examples = load_cb()
    result = evaluate_task(
        wrapper,
        canon,
        examples,
        device,
        max_context,
        pmi=True,
        batch_size=1,
    )
    payload = {
        "method": "vanilla_mistral_baseline",
        "target_model": args.target_model,
        "dataset": "super_glue/cb",
        "evaluation_split": "validation",
        "n_examples": len(examples),
        "memory_loaded": False,
        "adaptor_loaded": False,
        "protocol": result["protocol"],
        "accuracy": result["accuracy"],
        "correct": result["correct"],
        "total": result["total"],
        "predictions": result["predictions"],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    cleanup = getattr(wrapper, "cleanup", None)
    if cleanup is not None:
        cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
