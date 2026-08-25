"""Prompt-only MemGen latent-memory baseline for the five OpenQA tasks.

This is a small, self-contained adaptation of the official MemGen prompt
Weaver path.  It keeps a frozen Mistral reasoner, trains a LoRA-equipped
Weaver plus the two embedding projections, and inserts the Weaver's latent
states between the question and the answer.  The official MemGen TriviaQA
configuration uses this prompt-only setting (the trigger and inference-time
augmentation are disabled), so this is the appropriate first MemGen baseline
for the user's generated-memory-only comparison.

The five-task evaluator and prompt strings are imported from eval_openqa.py so
that NQ/WebQA/TriviaQA/TruthfulQA/HotpotQA use exactly the existing protocol.
The training split contains the available supervised train splits for NQ,
WebQuestions, TriviaQA, and HotpotQA; TruthfulQA is evaluation-only because it
does not provide the same kind of supervised train split in this project.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_openqa import (
    TASK_LOADERS,
    TASK_SCALAR_METRICS,
    build_openqa_prompt,
    build_truthfulqa_prompt,
    compute_truthfulqa_mc_scores,
    dedupe_answers,
    flatten_answers,
    f1_score,
    exact_match,
    use_official_tokenization,
)


TRAIN_TASKS = ("nq", "webqa", "triviaqa", "hotpotqa")
EVAL_TASKS = ("nq", "webqa", "triviaqa", "truthfulqa", "hotpotqa")
TRAIN_SPECS = {
    "nq": ("google-research-datasets/nq_open", None, "train"),
    "webqa": ("Stanford/web_questions", None, "train"),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "train"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "train"),
}


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.clamp_min(0).masked_fill(attention_mask == 0, 0)


class PromptMemoryWeaver(nn.Module):
    """Official MemGen-style learnable query latents and prompt Weaver."""

    def __init__(self, model: nn.Module, latents_len: int, lora_config: LoraConfig):
        super().__init__()
        self.model = get_peft_model(model, lora_config, adapter_name="weaver")
        hidden_size = int(self.model.base_model.config.hidden_size)
        self.query_latents = nn.Parameter(torch.randn(latents_len, hidden_size))
        self.latent_ln = nn.LayerNorm(hidden_size)
        self.latent_scale = nn.Parameter(torch.ones(1))

    @property
    def latents_len(self) -> int:
        return int(self.query_latents.shape[0])

    def augment(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = inputs_embeds.shape[0]
        latents = self.latent_ln(self.query_latents) * self.latent_scale
        latents = latents.unsqueeze(0).expand(batch_size, -1, -1)
        combined_embeds = torch.cat([inputs_embeds, latents], dim=1)
        latent_mask = torch.ones(
            (batch_size, self.latents_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_mask = torch.cat([attention_mask, latent_mask], dim=1)
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            position_ids=_position_ids(combined_mask),
            output_hidden_states=True,
            use_cache=False,
        )
        return outputs.hidden_states[-1][:, -self.latents_len :, :]


class MemGenMistralQA(nn.Module):
    """Mistral reasoner with prompt-only MemGen latent memory."""

    def __init__(
        self,
        reasoner: nn.Module,
        weaver: nn.Module,
        prompt_latents_len: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.reasoner = reasoner
        self.weaver = PromptMemoryWeaver(
            weaver,
            latents_len=prompt_latents_len,
            lora_config=LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
        reasoner_hidden = int(reasoner.config.hidden_size)
        weaver_hidden = int(weaver.config.hidden_size)
        self.reasoner_to_weaver = nn.Linear(reasoner_hidden, weaver_hidden)
        self.weaver_to_reasoner = nn.Linear(weaver_hidden, reasoner_hidden)
        self.reasoner.requires_grad_(False)
        self.reasoner.eval()

    @property
    def device(self):
        return next(self.reasoner.parameters()).device

    def _augment_one(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert latent memory after the prompt and return embeds/mask."""
        token_embeds = self.reasoner.get_input_embeddings()(input_ids)
        prefix = token_embeds[:, :prompt_len]
        suffix = token_embeds[:, prompt_len:]
        prefix_mask = attention_mask[:, :prompt_len]
        suffix_mask = attention_mask[:, prompt_len:]
        weaver_prefix = self.reasoner_to_weaver(prefix)
        weaver_latents = self.weaver.augment(weaver_prefix, prefix_mask)
        reasoner_latents = self.weaver_to_reasoner(weaver_latents)
        inputs_embeds = torch.cat([prefix, reasoner_latents, suffix], dim=1)
        latent_mask = torch.ones(
            (attention_mask.shape[0], self.weaver.latents_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        augmented_mask = torch.cat([prefix_mask, latent_mask, suffix_mask], dim=1)
        return inputs_embeds, augmented_mask

    def training_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_lengths: list[int],
    ) -> torch.Tensor:
        """Compute answer-only causal loss, one example at a time."""
        losses = []
        for index, prompt_len in enumerate(prompt_lengths):
            valid_len = int(attention_mask[index].sum().item())
            ids = input_ids[index : index + 1, :valid_len]
            mask = attention_mask[index : index + 1, :valid_len]
            target = labels[index : index + 1, :valid_len]
            embeds, augmented_mask = self._augment_one(ids, mask, prompt_len)
            latent_labels = torch.full(
                (1, self.weaver.latents_len),
                -100,
                dtype=target.dtype,
                device=target.device,
            )
            extended_labels = torch.cat(
                [target[:, :prompt_len], latent_labels, target[:, prompt_len:]],
                dim=1,
            )
            outputs = self.reasoner(
                inputs_embeds=embeds,
                attention_mask=augmented_mask,
                position_ids=_position_ids(augmented_mask),
                use_cache=False,
            )
            logits = outputs.logits[:, :-1].contiguous()
            shifted_labels = extended_labels[:, 1:].contiguous()
            valid = shifted_labels != -100
            if not valid.any():
                raise ValueError("Training example has no supervised answer tokens")
            losses.append(
                F.cross_entropy(logits[valid], shifted_labels[valid])
            )
        return torch.stack(losses).mean()

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        embeds, augmented_mask = self._augment_one(
            input_ids, attention_mask, prompt_len=input_ids.shape[1]
        )
        return self.reasoner.generate(
            inputs_embeds=embeds,
            attention_mask=augmented_mask,
            position_ids=_position_ids(augmented_mask),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    @torch.no_grad()
    def continuation_logprob(
        self,
        prompt_ids: torch.Tensor,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> float:
        del prompt_ids  # retained in the signature to make the scoring boundary explicit
        attention_mask = torch.ones_like(full_ids)
        embeds, augmented_mask = self._augment_one(
            full_ids, attention_mask, prompt_len=prompt_len
        )
        outputs = self.reasoner(
            inputs_embeds=embeds,
            attention_mask=augmented_mask,
            position_ids=_position_ids(augmented_mask),
            use_cache=False,
        )
        logits = outputs.logits.float()
        latent_offset = self.weaver.latents_len
        target_start = prompt_len + latent_offset
        target_ids = full_ids[:, prompt_len:]
        target_logits = logits[:, target_start - 1 : -1]
        if target_ids.shape[1] == 0 or target_logits.shape[1] != target_ids.shape[1]:
            return float("-inf")
        token_log_probs = F.log_softmax(target_logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        )
        return float(token_log_probs.sum().item())

    def save_checkpoint(self, output_dir: str, metadata: dict) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "reasoner_to_weaver": self.reasoner_to_weaver.state_dict(),
                "weaver_to_reasoner": self.weaver_to_reasoner.state_dict(),
                "query_latents": self.weaver.query_latents.detach().cpu(),
                "latent_ln": self.weaver.latent_ln.state_dict(),
                "latent_scale": self.weaver.latent_scale.detach().cpu(),
            },
            output / "memgen_state.pt",
        )
        torch.save(
            get_peft_model_state_dict(self.weaver.model, adapter_name="weaver"),
            output / "weaver_lora.pt",
        )
        (output / "config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint = Path(checkpoint_dir)
        state = torch.load(checkpoint / "memgen_state.pt", map_location="cpu", weights_only=True)
        self.reasoner_to_weaver.load_state_dict(state["reasoner_to_weaver"])
        self.weaver_to_reasoner.load_state_dict(state["weaver_to_reasoner"])
        self.weaver.query_latents.data.copy_(state["query_latents"])
        self.weaver.latent_ln.load_state_dict(state["latent_ln"])
        self.weaver.latent_scale.data.copy_(state["latent_scale"])
        lora_state = torch.load(checkpoint / "weaver_lora.pt", map_location="cpu", weights_only=True)
        set_peft_model_state_dict(self.weaver.model, lora_state, adapter_name="weaver")


def _last_boxed_content(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""
    index = start + len(marker)
    depth = 1
    chars = []
    while index < len(text) and depth:
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
        index += 1
    return "".join(chars).strip() if depth == 0 else text[start:].strip()


def _training_example(task: str, raw: dict) -> dict | None:
    if task == "nq":
        question = str(raw.get("question", "")).strip()
        answers = dedupe_answers(flatten_answers(raw.get("answer", raw.get("answers"))))
    elif task == "webqa":
        question = str(raw.get("question", "")).strip()
        answers = dedupe_answers(flatten_answers(raw.get("answers", raw.get("answer"))))
    elif task == "triviaqa":
        question = str(raw.get("question", "")).strip()
        answer = raw.get("answer", {})
        preferred = answer.get("value") if isinstance(answer, dict) else None
        answers = dedupe_answers(flatten_answers(answer))
        if preferred:
            answers.insert(0, str(preferred))
        answers = dedupe_answers(answers)
    elif task == "hotpotqa":
        question = str(raw.get("question", "")).strip()
        answers = dedupe_answers(flatten_answers(raw.get("answer")))
    else:
        raise ValueError(f"Unsupported training task: {task}")
    answers = [answer for answer in answers if answer and answer != "<unk>"]
    if not question or not answers or (task == "nq" and ")" in answers):
        return None
    answer = answers[0]
    return {
        "task": task,
        "question": question,
        "prompt": build_openqa_prompt(question),
        "answer": answer,
        "answers": answers,
    }


def load_training_examples(
    tasks: Iterable[str], max_examples_per_task: int | None = None
) -> tuple[list[dict], dict]:
    examples = []
    sources = {}
    for task in tasks:
        dataset_name, config_name, split = TRAIN_SPECS[task]
        dataset = load_dataset(dataset_name, config_name, split=split)
        raw_count = len(dataset)
        if max_examples_per_task is not None:
            dataset = dataset.select(range(min(max_examples_per_task, raw_count)))
        task_examples = []
        for raw in dataset:
            item = _training_example(task, raw)
            if item is not None:
                task_examples.append(item)
        examples.extend(task_examples)
        sources[task] = {
            "dataset_name": dataset_name,
            "config_name": config_name,
            "split": split,
            "raw_count": raw_count,
            "usable_count": len(task_examples),
        }
    return examples, sources


def encode_training_example(tokenizer, example: dict, max_length: int) -> tuple[list[int], list[int], int]:
    prompt = example["prompt"]
    answer = " " + example["answer"].strip()
    if tokenizer.eos_token:
        answer += tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + answer, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_ids)
    if prompt_len >= len(full_ids):
        raise ValueError("Unable to identify answer span")
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    labels = [-100] * min(prompt_len, len(full_ids))
    labels.extend(full_ids[prompt_len:])
    labels = labels[: len(full_ids)]
    return full_ids, labels, min(prompt_len, len(full_ids) - 1)


def collate_training(tokenizer, examples: list[dict], max_length: int):
    encoded = [encode_training_example(tokenizer, example, max_length) for example in examples]
    max_len = max(len(item[0]) for item in encoded)
    pad_id = tokenizer.pad_token_id
    input_ids, labels, masks, prompt_lengths = [], [], [], []
    for ids, target, prompt_len in encoded:
        padding = max_len - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        labels.append(target + [-100] * padding)
        masks.append([1] * len(ids) + [0] * padding)
        prompt_lengths.append(prompt_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "prompt_lengths": prompt_lengths,
    }


def load_hf_model(model_name: str, dtype: torch.dtype, attn_implementation: str):
    kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def build_model(args, device: torch.device, dtype: torch.dtype) -> tuple[MemGenMistralQA, object]:
    weaver_name = args.weaver_model or args.target_model
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    reasoner = load_hf_model(args.target_model, dtype, args.attn_implementation)
    weaver = load_hf_model(weaver_name, dtype, args.attn_implementation)
    model = MemGenMistralQA(
        reasoner=reasoner,
        weaver=weaver,
        prompt_latents_len=args.prompt_latents_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    # The pretrained Mistral modules are loaded in bf16 on LUMI.  Cast the
    # newly-created projections and latent parameters to the same dtype before
    # the first embedding projection; otherwise PyTorch rejects bf16 x fp32.
    model.to(device=device, dtype=dtype)
    model.reasoner.eval()
    return model, tokenizer


def _tokenize(tokenizer, text: str, add_special_tokens: bool, device: torch.device):
    return tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )["input_ids"].to(device)


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int, official: bool) -> str:
    input_ids = _tokenize(tokenizer, prompt, official, device)
    attention_mask = torch.ones_like(input_ids)
    generated = model.generate_ids(
        input_ids,
        attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


@torch.no_grad()
def score_continuation(model, tokenizer, prompt: str, continuation: str, device: torch.device, official: bool) -> float:
    prompt_ids = _tokenize(tokenizer, prompt, official, device)
    full_ids = _tokenize(tokenizer, prompt + continuation, official, device)
    prompt_len = min(prompt_ids.shape[1], full_ids.shape[1] - 1)
    return model.continuation_logprob(prompt_ids, full_ids, prompt_len)


def evaluate(model, tokenizer, tasks: list[str], device: torch.device, max_new_tokens: int, max_examples: int | None) -> dict:
    results = {}
    for task in tasks:
        examples, dataset_meta = TASK_LOADERS[task]()
        if max_examples is not None:
            examples = examples[:max_examples]
        if task == "truthfulqa":
            totals = {"MC1": 0.0, "MC2": 0.0, "MC3": 0.0}
            sample = []
            for ex in tqdm(examples, desc=f"MemGen {task}"):
                prompt = build_truthfulqa_prompt(ex["question"])
                true_scores = [
                    score_continuation(model, tokenizer, prompt, " " + answer, device, False)
                    for answer in ex["correct_answers"]
                ]
                false_scores = [
                    score_continuation(model, tokenizer, prompt, " " + answer, device, False)
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
            for ex in tqdm(examples, desc=f"MemGen {task}"):
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
    parser.add_argument("--train-tasks", nargs="+", default=list(TRAIN_TASKS), choices=list(TRAIN_TASKS))
    parser.add_argument("--eval-tasks", nargs="+", default=list(EVAL_TASKS), choices=list(EVAL_TASKS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-train-examples-per-task", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--prompt-latents-len", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Save the trained checkpoint without running the downstream evaluator",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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
        model.weaver.train()
        model.reasoner.eval()
        optimizer.zero_grad(set_to_none=True)
        step = 0
        for epoch in range(args.epochs):
            random.shuffle(examples)
            for start in tqdm(range(0, len(examples), args.batch_size), desc=f"MemGen train epoch {epoch + 1}"):
                batch_examples = examples[start : start + args.batch_size]
                batch = collate_training(tokenizer, batch_examples, args.max_length)
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                loss = model.training_loss(**batch) / args.grad_accum_steps
                loss.backward()
                micro_step = (start // args.batch_size) + 1
                is_last_batch = start + args.batch_size >= len(examples)
                if micro_step % args.grad_accum_steps == 0 or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if step % 10 == 0:
                        print(f"epoch={epoch + 1} step={step} loss={loss.item() * args.grad_accum_steps:.6f}")
        metadata = {
            "method": "memgen_prompt_only_weaver",
            "paper": "arXiv:2509.24704",
            "target_model": args.target_model,
            "weaver_model": args.weaver_model or args.target_model,
            "prompt_latents_len": args.prompt_latents_len,
            "trigger_active": False,
            "max_inference_aug_num": 0,
            "train_tasks": list(args.train_tasks),
            "train_size": len(examples),
            "train_sources": sources,
            "optimizer_steps": step,
            "seed": args.seed,
        }
        model.save_checkpoint(str(output_dir), metadata)
        if args.skip_eval:
            print("MEMGEN_MISTRAL_QA_TRAIN_COMPLETE", output_dir)
            return
    else:
        checkpoint_dir = args.checkpoint_dir or str(output_dir)
        model.load_checkpoint(checkpoint_dir)

    model.eval()
    model.reasoner.eval()
    results = evaluate(model, tokenizer, args.eval_tasks, device, args.max_new_tokens, args.max_examples)
    payload = {
        "completed": True,
        "method": "memgen_prompt_only_weaver",
        "target_model": args.target_model,
        "checkpoint_dir": str(args.checkpoint_dir or output_dir),
        "eval_tasks": list(args.eval_tasks),
        "results": results,
    }
    (output_dir / "openqa_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print("MEMGEN_MISTRAL_QA_COMPLETE", output_dir)


if __name__ == "__main__":
    main()
