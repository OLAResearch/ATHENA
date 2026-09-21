"""Small CB pilot for inference-only and label-using router variants.

Methods A and B are inference-only sweeps. Methods C and D are optional
label-using baselines: C calibrates on the CB training split and D fine-tunes
the existing advantage head on that split. The selected methods are controlled
with ``--methods`` so an A/B run never reads the downstream training labels.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_general_nlp_halueval import _example, _load_dataset, evaluate_task, _normalise_choice
from scripts.eval_dual_reader_openqa_paired import set_reader_mode
from scripts.eval_openqa import get_model_max_context, setup_condition


CB_CHOICES = ["entailment", "contradiction", "neutral"]


def load_cb_split(split: str) -> list[dict]:
    rows = _load_dataset("super_glue", "cb", split=split)
    return [
        _example(
            f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nRelation:",
            CB_CHOICES,
            int(row["label"]),
            "Premise:\nHypothesis:\nRelation:",
        )
        for row in rows
    ]


def _adaptors(wrapper):
    if isinstance(wrapper.adaptor, torch.nn.ModuleList):
        return list(wrapper.adaptor)
    return [wrapper.adaptor]


def configure_advantage(wrapper, *, threshold: float, confidence: float, max_scale: float) -> None:
    for adaptor in _adaptors(wrapper):
        adaptor.configure_advantage_reader(
            candidates="sources",
            threshold=threshold,
            confidence_threshold=confidence,
            temperature=0.15,
            max_scale=max_scale,
        )


def evaluate_cb(wrapper, canon, examples, device, max_context, *, mode: str) -> dict:
    set_reader_mode(wrapper, mode)
    return evaluate_task(
        wrapper,
        canon,
        examples,
        device,
        max_context,
        pmi=True,
        batch_size=1,
    )


def _choice_logprob(wrapper, tokenizer, context: str, choice: str, device, canon, max_context):
    choice_text = _normalise_choice(choice)
    full_text = context + choice_text
    old_padding, old_truncation = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    try:
        batch = tokenizer(
            [full_text],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context,
        )
        choice_ids = tokenizer(choice_text, add_special_tokens=False)["input_ids"]
    finally:
        tokenizer.padding_side, tokenizer.truncation_side = old_padding, old_truncation

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    canon(input_ids)
    outputs = wrapper(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    real_len = int(attention_mask[0].sum().item()) if attention_mask is not None else input_ids.shape[1]
    choice_len = len(choice_ids)
    if choice_len <= 0 or choice_len >= real_len:
        return log_probs.sum() * 0.0
    target_start = real_len - choice_len
    targets = input_ids[0, target_start:real_len]
    token_log_probs = log_probs[0, target_start - 1:real_len - 1]
    return token_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).mean()


def _pmi_scores(wrapper, examples, device, canon, max_context) -> torch.Tensor:
    scores = []
    tokenizer = wrapper.tokenizer
    for example in examples:
        values = []
        for choice in example["choices"]:
            conditional = _choice_logprob(
                wrapper, tokenizer, example["context"], choice, device, canon, max_context
            )
            domain = _choice_logprob(
                wrapper, tokenizer, example["domain_context"], choice, device, canon, max_context
            )
            values.append(conditional - domain)
        scores.append(torch.stack(values))
    return torch.stack(scores)


def train_pairwise_router(wrapper, examples, device, canon, max_context, *, epochs: int, lr: float, seed: int) -> list[dict]:
    for parameter in wrapper.parameters():
        parameter.requires_grad = False
    router_parameters = []
    for adaptor in _adaptors(wrapper):
        if adaptor.advantage_router is None:
            raise RuntimeError("Method D requires an existing advantage router head")
        for parameter in adaptor.advantage_router.parameters():
            parameter.requires_grad = True
            router_parameters.append(parameter)
    if not router_parameters:
        raise RuntimeError("No trainable advantage-router parameters found")

    set_reader_mode(wrapper, "tri_advantage_routed")
    configure_advantage(wrapper, threshold=0.0, confidence=0.5, max_scale=0.2)
    optimizer = torch.optim.AdamW(router_parameters, lr=lr)
    rng = random.Random(seed)
    history = []
    wrapper.train()
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total_loss = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            scores = _pmi_scores(wrapper, [examples[index]], device, canon, max_context)[0]
            correct = scores[examples[index]["label"]]
            negatives = torch.cat((scores[: examples[index]["label"]], scores[examples[index]["label"] + 1:]))
            loss = F.softplus(0.05 - correct + negatives).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
        history.append({"epoch": epoch + 1, "mean_pairwise_loss": total_loss / max(1, len(order))})
    wrapper.eval()
    return history


def _config_result(name: str, config: dict, train_result: dict | None, valid_result: dict) -> dict:
    return {
        "method": name,
        "config": config,
        "train_accuracy": None if train_result is None else train_result["accuracy"],
        "validation_accuracy": valid_result["accuracy"],
        "validation_correct": valid_result["correct"],
        "validation_total": valid_result["total"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        default="abcd",
        help="Methods to run, as a subset of abcd. A/B are inference-only; C/D use CB train labels.",
    )
    parser.add_argument("--max-train-examples", type=int, default=64)
    parser.add_argument("--d-epochs", type=int, default=1)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    methods = set(args.methods.lower())
    if not methods or not methods.issubset(set("abcd")):
        parser.error("--methods must be a non-empty subset of abcd")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_examples = load_cb_split("validation")
    train_examples = (
        load_cb_split("train")[: args.max_train_examples]
        if {"c", "d"} & methods
        else []
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    condition_args = SimpleNamespace(
        target_model=args.target_model,
        adaptor_dir=args.adaptor_dir,
        adaptor_checkpoint=args.adaptor_checkpoint,
        source_memory=args.source_memory,
        memory_config=args.memory_config,
        reader_mode="tri_advantage_routed",
        seed=args.seed,
        canon_mode="word_boundary",
    )
    wrapper, canon = setup_condition(condition_args, "transferred", device, dtype)
    max_context = get_model_max_context(wrapper, None)

    results = {
        "dataset": "super_glue/cb",
        "calibration_split": "train" if {"c", "d"} & methods else None,
        "evaluation_split": "validation",
        "train_examples": len(train_examples),
        "validation_examples": len(valid_examples),
        "test_labels_used": bool({"c", "d"} & methods),
        "selected_methods": "".join(sorted(methods)),
        "methods": [],
    }

    baseline = evaluate_cb(wrapper, canon, valid_examples, device, max_context, mode="engram_only")
    results["baseline_E"] = {"accuracy": baseline["accuracy"], "correct": baseline["correct"], "total": baseline["total"]}

    if "a" in methods:
        # A: inference-only E-anchored residual scale sweep.
        for alpha in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
            configure_advantage(wrapper, threshold=0.0, confidence=0.5, max_scale=alpha)
            valid = evaluate_cb(wrapper, canon, valid_examples, device, max_context, mode="tri_advantage_routed")
            results["methods"].append(_config_result("A_alpha_sweep", {"alpha_max_scale": alpha}, None, valid))

    if "b" in methods:
        # B: inference-only conservative gate sweep.
        for threshold in (0.05, 0.10, 0.20, 0.30):
            configure_advantage(wrapper, threshold=threshold, confidence=0.7, max_scale=1.0)
            valid = evaluate_cb(wrapper, canon, valid_examples, device, max_context, mode="tri_advantage_routed")
            results["methods"].append(_config_result("B_threshold_gate", {"threshold": threshold, "confidence": 0.7}, None, valid))

    if "c" in methods:
        # C: choose one conservative configuration using only CB train labels.
        calibration_candidates = [
            {"alpha_max_scale": alpha, "threshold": threshold, "confidence": 0.7}
            for alpha in (0.10, 0.20, 0.30)
            for threshold in (0.05, 0.10, 0.20)
        ]
        calibrated = []
        for config in calibration_candidates:
            configure_advantage(wrapper, threshold=config["threshold"], confidence=config["confidence"], max_scale=config["alpha_max_scale"])
            train_result = evaluate_cb(wrapper, canon, train_examples, device, max_context, mode="tri_advantage_routed")
            calibrated.append((train_result["accuracy"], config, train_result))
        _, best_config, best_train = max(calibrated, key=lambda item: item[0])
        configure_advantage(wrapper, threshold=best_config["threshold"], confidence=best_config["confidence"], max_scale=best_config["alpha_max_scale"])
        best_valid = evaluate_cb(wrapper, canon, valid_examples, device, max_context, mode="tri_advantage_routed")
        results["methods"].append(_config_result("C_train_calibrated", best_config, best_train, best_valid))

    if "d" in methods:
        # D: fine-tune only the advantage head with pairwise PMI on CB train.
        history = train_pairwise_router(
            wrapper, train_examples, device, canon, max_context,
            epochs=args.d_epochs, lr=args.d_lr, seed=args.seed,
        )
        configure_advantage(wrapper, threshold=0.0, confidence=0.5, max_scale=0.2)
        d_valid = evaluate_cb(wrapper, canon, valid_examples, device, max_context, mode="tri_advantage_routed")
        results["methods"].append(_config_result("D_pairwise_PMI_router", {"epochs": args.d_epochs, "lr": args.d_lr, "alpha_max_scale": 0.2}, None, d_valid))
        results["D_training"] = history

    if "d" in methods:
        torch.save(
            {
                name: parameter.detach().cpu()
                for name, parameter in wrapper.named_parameters()
                if "advantage_router" in name
            },
            output_dir / "d_pairwise_pmi_router.pt",
        )
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    cleanup = getattr(wrapper, "cleanup", None)
    if cleanup is not None:
        cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
