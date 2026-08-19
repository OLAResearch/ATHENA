"""Evaluate a task-supervised dual-reader checkpoint on a held-out QA split."""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_openqa import setup_condition
from scripts.train_generative_memory_qa import (
    TriviaQASFTDataset,
    evaluate_generation,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--base-adaptor-dir", required=True)
    parser.add_argument("--qa-adaptor", required=True)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--canon-mode", default="word_boundary", choices=["vocab", "word_boundary"])
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # setup_condition reads architecture metadata and the frozen Engram table
    # from the original Wiki-trained dual-reader directory.
    args.adaptor_dir = args.base_adaptor_dir
    wrapper, set_canon_fn = setup_condition(args, "transferred", device, dtype)
    qa_state = torch.load(args.qa_adaptor, map_location="cpu", weights_only=True)
    wrapper.adaptor.load_state_dict(qa_state, strict=True)
    wrapper.adaptor.to(device)
    wrapper.freeze_backbone()
    wrapper.freeze_memory()
    for parameter in wrapper.adaptor.parameters():
        parameter.requires_grad = False
    wrapper.eval()

    dataset = TriviaQASFTDataset(args.split, args.max_eval_examples)
    print(f"Dataset: {dataset.source}; {args.split}={len(dataset):,}")
    metrics = evaluate_generation(
        wrapper,
        dataset,
        set_canon_fn,
        device,
        args.max_new_tokens,
    )
    results = {
        "dataset": "TriviaQA",
        "split": args.split,
        "checkpoint": args.qa_adaptor,
        "em": metrics["em"],
        "f1": metrics["f1"],
        "count": metrics["count"],
        "completed": True,
    }
    with open(output_dir / f"{args.split}_results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    print(
        f"TriviaQA {args.split} EM={metrics['em']:.4f}; "
        f"F1={metrics['f1']:.4f}; count={metrics['count']}"
    )
    print("TEST_EVALUATION_COMPLETED")
    wrapper.cleanup()


if __name__ == "__main__":
    main()
