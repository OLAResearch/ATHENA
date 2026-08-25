"""GRPO MemGen prompt-Weaver baseline for the five OpenQA tasks.

This is the GRPO counterpart of ``memgen_mistral_qa.py``.  It keeps the
Mistral reasoner frozen and optimizes the Weaver LoRA/latent parameters with
group-relative sequence rewards.  Each prompt samples a group of answers,
uses the existing OpenQA F1 evaluator as the reward, and applies the clipped
GRPO objective to the sampled continuations.

The public MemGen repository does not currently provide a complete joint
NQ/WebQA/TriviaQA/HotpotQA GRPO entry point, so this script is intentionally
labelled as a project-local GRPO reproduction.  TruthfulQA remains
evaluation-only because it has no compatible supervised answer-training split
in this project.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_openqa import (  # noqa: E402
    TASK_LOADERS,
    TASK_SCALAR_METRICS,
    build_openqa_prompt,
    build_truthfulqa_prompt,
    compute_truthfulqa_mc_scores,
    dedupe_answers,
    exact_match,
    flatten_answers,
    f1_score,
    use_official_tokenization,
)
from scripts.memgen_mistral_qa import (  # noqa: E402
    EVAL_TASKS,
    TRAIN_SPECS,
    MemGenMistralQA,
    _position_ids,
    build_model,
    generate_text,
    load_training_examples,
)


def grpo_objective(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Compute the clipped sequence-level GRPO objective for one group."""
    if new_logprobs.shape != old_logprobs.shape or new_logprobs.shape != advantages.shape:
        raise ValueError("GRPO inputs must have the same shape")
    ratios = torch.exp((new_logprobs - old_logprobs).clamp(-20.0, 20.0))
    unclipped = ratios * advantages
    clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def normalize_group_rewards(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Return group-relative advantages; constant-reward groups get zero."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("A GRPO group needs at least two rewards")
    std = rewards.std(unbiased=False)
    if float(std.detach().cpu()) < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def _tokenize(tokenizer, text: str, add_special_tokens: bool, device: torch.device):
    return tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )["input_ids"].to(device)


def _extract_completion(tokenizer, prompt_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    """Handle both HF generation return conventions (prompt+completion or completion-only)."""
    output_ids = output_ids.detach().to("cpu")
    prompt_ids = prompt_ids[0].detach().to("cpu")
    if output_ids.numel() >= prompt_ids.numel() and torch.equal(
        output_ids[: prompt_ids.numel()], prompt_ids
    ):
        output_ids = output_ids[prompt_ids.numel() :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def sample_group(
    model: MemGenMistralQA,
    tokenizer,
    prompt: str,
    device: torch.device,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Sample a GRPO group from the current prompt-Weaver policy."""
    prompt_ids = _tokenize(tokenizer, prompt, True, device)
    attention_mask = torch.ones_like(prompt_ids)
    embeds, augmented_mask = model._augment_one(
        prompt_ids, attention_mask, prompt_len=prompt_ids.shape[1]
    )
    generated = model.reasoner.generate(
        inputs_embeds=embeds,
        attention_mask=augmented_mask,
        position_ids=_position_ids(augmented_mask),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=group_size,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [_extract_completion(tokenizer, prompt_ids, row) for row in generated]


def _completion_logprob_tensor(
    model: MemGenMistralQA,
    tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
) -> torch.Tensor:
    """Differentiable log p(completion | prompt, Weaver(prompt))."""
    continuation = " " + completion.strip()
    prompt_ids = _tokenize(tokenizer, prompt, True, device)
    full_ids = _tokenize(tokenizer, prompt + continuation, True, device)
    prompt_len = min(prompt_ids.shape[1], full_ids.shape[1] - 1)
    if full_ids.shape[1] <= prompt_len:
        return full_ids.new_zeros((), dtype=torch.float32, requires_grad=True)

    attention_mask = torch.ones_like(full_ids)
    embeds, augmented_mask = model._augment_one(full_ids, attention_mask, prompt_len)
    outputs = model.reasoner(
        inputs_embeds=embeds,
        attention_mask=augmented_mask,
        position_ids=_position_ids(augmented_mask),
        use_cache=False,
    )
    logits = outputs.logits[:, :-1].float().contiguous()
    target_ids = full_ids[:, prompt_len:]
    target_start = prompt_len + model.weaver.latents_len
    target_logits = logits[:, target_start - 1 : -1]
    if target_logits.shape[1] != target_ids.shape[1]:
        length = min(target_logits.shape[1], target_ids.shape[1])
        target_logits = target_logits[:, :length]
        target_ids = target_ids[:, :length]
    token_log_probs = F.log_softmax(target_logits, dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    )
    return token_log_probs.sum()


def reward_for_example(task: str, prediction: str, answers: list[str]) -> float:
    """Use the same answer normalization as the existing OpenQA evaluator."""
    del task  # task-specific tokenization is already applied by f1_score.
    return max(float(f1_score(prediction, answer)[0]) for answer in answers)


def train_grpo(
    model: MemGenMistralQA,
    tokenizer,
    examples: list[dict],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    clip_epsilon: float,
    max_steps: int | None,
    log_every: int,
) -> tuple[int, dict]:
    model.reasoner.eval()
    step = 0
    skipped_constant_groups = 0
    reward_values: list[float] = []
    for example in tqdm(examples, desc="MemGen GRPO"):
        model.weaver.eval()
        completions = sample_group(
            model,
            tokenizer,
            example["prompt"],
            device,
            group_size,
            max_new_tokens,
            temperature,
            top_p,
        )
        with torch.no_grad():
            old_logprobs = torch.stack(
                [
                    _completion_logprob_tensor(
                        model, tokenizer, example["prompt"], completion, device
                    ).detach()
                    for completion in completions
                ]
            )
        rewards = torch.tensor(
            [reward_for_example(example["task"], completion, example["answers"]) for completion in completions],
            dtype=torch.float32,
            device=device,
        )
        advantages = normalize_group_rewards(rewards)
        reward_values.extend(float(value) for value in rewards.detach().cpu())
        if not bool(torch.any(advantages != 0)):
            skipped_constant_groups += 1
            continue

        model.weaver.train()
        optimizer.zero_grad(set_to_none=True)
        new_logprobs = torch.stack(
            [
                _completion_logprob_tensor(
                    model, tokenizer, example["prompt"], completion, device
                )
                for completion in completions
            ]
        )
        loss = grpo_objective(new_logprobs, old_logprobs, advantages, clip_epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % log_every == 0 or step == 1:
            print(
                f"step={step} loss={loss.item():.6f} "
                f"reward_mean={rewards.mean().item():.4f} "
                f"reward_std={rewards.std(unbiased=False).item():.4f}"
            )
        if max_steps is not None and step >= max_steps:
            break
    stats = {
        "mean_reward": float(np.mean(reward_values)) if reward_values else 0.0,
        "reward_samples": len(reward_values),
        "skipped_constant_groups": skipped_constant_groups,
    }
    return step, stats


def evaluate(model, tokenizer, tasks: list[str], device: torch.device, max_new_tokens: int, max_examples: int | None) -> dict:
    """Reuse the existing deterministic five-task evaluator protocol."""
    results = {}
    for task in tasks:
        examples, dataset_meta = TASK_LOADERS[task]()
        if max_examples is not None:
            examples = examples[:max_examples]
        if task == "truthfulqa":
            totals = {"MC1": 0.0, "MC2": 0.0, "MC3": 0.0}
            sample = []
            for ex in tqdm(examples, desc=f"MemGen GRPO {task}"):
                prompt = build_truthfulqa_prompt(ex["question"])
                true_scores = [
                    _completion_logprob_tensor(model, tokenizer, prompt, answer, device).item()
                    for answer in ex["correct_answers"]
                ]
                false_scores = [
                    _completion_logprob_tensor(model, tokenizer, prompt, answer, device).item()
                    for answer in ex["incorrect_answers"]
                ]
                metrics = compute_truthfulqa_mc_scores(
                    true_scores, false_scores, ex["correct_answers"], ex["best_answer"]
                )
                for key in totals:
                    totals[key] += metrics[key]
                if len(sample) < 10:
                    sample.append({"question": ex["question"], "metrics": metrics})
            total = len(examples)
            mc1 = totals["MC1"] / total if total else 0.0
            mc2 = totals["MC2"] / total if total else 0.0
            mc3 = totals["MC3"] / total if total else 0.0
            metrics = {
                "acc": mc1,
                "mc1": mc1,
                "mc2": mc2,
                "mc3": mc3,
                "mc_avg": (mc1 + mc2 + mc3) / 3.0,
                "total": total,
                "sample_examples": sample,
            }
        else:
            correct = 0
            f1_values = []
            sample = []
            official = use_official_tokenization(task)
            for ex in tqdm(examples, desc=f"MemGen GRPO {task}"):
                prompt = build_openqa_prompt(ex["question"])
                prediction = generate_text(model, tokenizer, prompt, device, max_new_tokens, official)
                is_correct = any(exact_match(prediction, answer) for answer in ex["answers"])
                best_f1 = max(f1_score(prediction, answer)[0] for answer in ex["answers"])
                correct += int(is_correct)
                f1_values.append(best_f1)
                if len(sample) < 10:
                    sample.append({"question": ex["question"], "prediction": prediction, "answers": ex["answers"]})
            total = len(examples)
            em = correct / total if total else 0.0
            metrics = {
                "acc": em,
                "em": em,
                "f1": float(np.mean(f1_values)) if f1_values else 0.0,
                "correct": correct,
                "total": total,
                "sample_predictions": sample,
            }
        results[task] = {
            "dataset": dataset_meta,
            "n_examples": len(examples),
            "metrics": metrics,
            "scalar_metric": TASK_SCALAR_METRICS[task],
            "scalar_score": metrics[TASK_SCALAR_METRICS[task]],
        }
        print(task, json.dumps(metrics, ensure_ascii=False))
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--weaver-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--train-tasks", nargs="+", default=["nq", "webqa", "triviaqa", "hotpotqa"], choices=list(TRAIN_SPECS))
    parser.add_argument("--eval-tasks", nargs="+", default=list(EVAL_TASKS), choices=list(EVAL_TASKS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-examples-per-task", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--prompt-latents-len", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("GRPO requires --group-size >= 2")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_model(args, device, dtype)

    if args.mode == "train":
        examples, sources = load_training_examples(args.train_tasks, args.max_train_examples_per_task)
        if not examples:
            raise RuntimeError("No usable training examples were loaded")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
        )
        all_steps = 0
        aggregate_stats = {"mean_reward": 0.0, "reward_samples": 0, "skipped_constant_groups": 0}
        for epoch in range(args.epochs):
            random.shuffle(examples)
            remaining_steps = None if args.max_steps is None else max(args.max_steps - all_steps, 0)
            if remaining_steps == 0:
                break
            steps, stats = train_grpo(
                model,
                tokenizer,
                examples,
                device,
                optimizer,
                args.group_size,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.clip_epsilon,
                remaining_steps,
                args.log_every,
            )
            all_steps += steps
            aggregate_stats["reward_samples"] += stats["reward_samples"]
            aggregate_stats["skipped_constant_groups"] += stats["skipped_constant_groups"]
            if stats["reward_samples"]:
                aggregate_stats["mean_reward"] = stats["mean_reward"]
            if args.max_steps is not None and all_steps >= args.max_steps:
                break
        metadata = {
            "method": "memgen_prompt_only_weaver_grpo_project_reproduction",
            "paper": "arXiv:2509.24704",
            "target_model": args.target_model,
            "weaver_model": args.weaver_model or args.target_model,
            "prompt_latents_len": args.prompt_latents_len,
            "trigger_active": False,
            "max_inference_aug_num": 0,
            "train_tasks": list(args.train_tasks),
            "train_size": len(examples),
            "train_sources": sources,
            "epochs": args.epochs,
            "optimizer_steps": all_steps,
            "group_size": args.group_size,
            "reward": "max_openqa_f1",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "clip_epsilon": args.clip_epsilon,
            "grpo_stats": aggregate_stats,
            "seed": args.seed,
        }
        model.save_checkpoint(str(output_dir), metadata)
        if args.skip_eval:
            print("MEMGEN_MISTRAL_QA_GRPO_TRAIN_COMPLETE", output_dir)
            return
    else:
        checkpoint_dir = args.checkpoint_dir or str(output_dir)
        model.load_checkpoint(checkpoint_dir)

    model.eval()
    model.reasoner.eval()
    results = evaluate(model, tokenizer, args.eval_tasks, device, args.max_new_tokens, args.max_examples)
    payload = {
        "completed": True,
        "method": "memgen_prompt_only_weaver_grpo_project_reproduction",
        "target_model": args.target_model,
        "checkpoint_dir": str(args.checkpoint_dir or output_dir),
        "eval_tasks": list(args.eval_tasks),
        "results": results,
    }
    (output_dir / "openqa_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print("MEMGEN_MISTRAL_QA_GRPO_COMPLETE", output_dir)


if __name__ == "__main__":
    main()
