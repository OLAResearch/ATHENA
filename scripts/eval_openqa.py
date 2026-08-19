"""Open-domain QA evaluation for Engram transfer on five QA benchmarks.

Benchmarks:
  - Natural Questions Open (nq)
  - WebQuestions / WebQA (webqa)
  - TriviaQA (triviaqa)
  - TruthfulQA multiple-choice (truthfulqa)
  - HotpotQA (hotpotqa)

For NQ/WebQA/TriviaQA/HotpotQA, the script performs greedy generation and
reports EM/F1. For TruthfulQA, it scores candidate answers and reports MC1,
MC2, and MC3.

For cross-task summaries, the scalar score follows the paper-facing metric
observed to match Table 1 best: F1 for NQ/WebQA/TriviaQA/HotpotQA, and
MC1/MC2/MC3 average for TruthfulQA.

Usage:
    python scripts/eval_openqa.py \
        --target-model mistralai/Mistral-7B-v0.3 \
        --adaptor-dir results/mistral_wiki2021/transferred_seed42 \
        --source-memory results/mistral_wiki2021_source/memory.pt \
        --memory-config results/mistral_wiki2021_source/memory_config.json \
        --tasks nq webqa triviaqa truthfulqa hotpotqa \
        --output-dir results/openqa/mistral_seed42
"""

import argparse
import json
import random
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.backbone_wrapper import BackboneWrapper
from engram.canonicalization import WordBoundaryCanonicalizer, build_canonicalizer
from engram.hashing import HashConfig, WordNgramHasher
from engram.memory import EngramMemory, MemoryConfig


TASKS = ["nq", "webqa", "triviaqa", "truthfulqa", "hotpotqa"]
EVAL_CONDITIONS = [
    "baseline",
    "transferred",
    "memory_only",
    "random_memory",
    "permuted_keys",
    "no_gate",
    "train_from_scratch",
    "ffn_only",
    "affine_stitch",
]

TASK_SCALAR_METRICS = {
    "nq": "f1",
    "webqa": "f1",
    "triviaqa": "f1",
    "truthfulqa": "mc_avg",
    "hotpotqa": "f1",
}

# Empirically, Mistral-7B-v0.3 matches the paper best with the official
# tokenization path for NQ/TriviaQA/HotpotQA, while WebQA/TruthfulQA match
# better with the legacy no-special-token path used by the earlier local runs.
TASK_OFFICIAL_TOKENIZATION = {
    "nq": True,
    "webqa": False,
    "triviaqa": True,
    "truthfulqa": False,
    "hotpotqa": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Open-domain QA evaluation")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--adaptor-dir", type=str, default=None,
                        help="Directory containing adaptor.pt/adaptor_best.pt for transferred runs")
    parser.add_argument("--adaptor-checkpoint", type=str, default=None,
                        help="Optional checkpoint override while retaining adaptor-dir runtime config")
    parser.add_argument(
        "--dual-reader-mode",
        choices=["both", "engram_only", "generated_only"],
        default="both",
        help="Runtime-only dual-reader ablation; generated_only still uses Engram cues",
    )
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--conditions", nargs="+", default=["baseline", "transferred"],
                        choices=EVAL_CONDITIONS)
    parser.add_argument("--canon-mode", type=str, default="word_boundary",
                        choices=["vocab", "word_boundary"])
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit examples per task (<=0 disables the limit)")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--max-context-length", type=int, default=None,
                        help="Override model max context during eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing openqa_results.json in output-dir")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None
    return args


def load_dataset_with_fallback(candidates, split_candidates):
    from datasets import load_dataset

    errors = []
    for dataset_name, config_name in candidates:
        for split_name in split_candidates:
            try:
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split=split_name,
                    trust_remote_code=True,
                )
                meta = {
                    "dataset_name": dataset_name,
                    "config_name": config_name,
                    "split": split_name,
                }
                return dataset, meta
            except Exception as exc:
                errors.append(
                    f"{dataset_name}"
                    + (f"/{config_name}" if config_name else "")
                    + f"[{split_name}]: {type(exc).__name__}: {exc}"
                )
    raise RuntimeError("Unable to load dataset:\n" + "\n".join(errors))


def flatten_answers(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        collected = []
        preferred_keys = [
            "aliases",
            "normalized_aliases",
            "text",
            "texts",
            "answer",
            "answers",
            "value",
        ]
        for key in preferred_keys:
            if key in value:
                collected.extend(flatten_answers(value[key]))
        if collected:
            return collected
        for nested in value.values():
            collected.extend(flatten_answers(nested))
        return collected
    if isinstance(value, (list, tuple, set)):
        collected = []
        for item in value:
            collected.extend(flatten_answers(item))
        return collected
    return [str(value)]


def dedupe_answers(answers) -> list[str]:
    seen = set()
    unique = []
    for answer in answers:
        answer = str(answer).strip()
        if answer and answer not in seen:
            seen.add(answer)
            unique.append(answer)
    return unique


def load_nq_examples():
    dataset, meta = load_dataset_with_fallback(
        candidates=[
            ("google-research-datasets/nq_open", None),
            ("nq_open", None),
        ],
        split_candidates=["validation", "dev", "test"],
    )
    examples = []
    for ex in dataset:
        answers = dedupe_answers(flatten_answers(ex.get("answer", ex.get("answers"))))
        # Match the official MLPMemory QA script, which skips malformed NQ
        # samples whose answer list contains only ")".
        if answers and ")" not in answers:
            examples.append({"question": ex["question"], "answers": answers})
    return examples, meta


def load_webqa_examples():
    dataset, meta = load_dataset_with_fallback(
        candidates=[
            ("Stanford/web_questions", None),
            ("web_questions", None),
            ("stanfordnlp/web_questions", None),
        ],
        split_candidates=["test", "validation"],
    )
    examples = []
    for ex in dataset:
        answers = dedupe_answers(flatten_answers(ex.get("answers", ex.get("answer"))))
        if answers:
            examples.append({"question": ex["question"], "answers": answers})
    return examples, meta


def load_triviaqa_examples():
    dataset, meta = load_dataset_with_fallback(
        candidates=[
            # MemGen's published TriviaQA evaluation uses this configuration
            # and its validation split as the held-out evaluation set.
            ("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext"),
            ("mandarjoshi/trivia_qa", "rc.nocontext"),
            ("mandarjoshi/trivia_qa", "unfiltered.nocontext"),
        ],
        split_candidates=["validation", "test"],
    )
    examples = []
    for ex in dataset:
        answers = dedupe_answers(flatten_answers(ex.get("answer")))
        if answers:
            examples.append({"question": ex["question"], "answers": answers})
    return examples, meta


def load_hotpotqa_examples():
    dataset, meta = load_dataset_with_fallback(
        candidates=[
            ("hotpotqa/hotpot_qa", "distractor"),
            ("hotpot_qa", "distractor"),
        ],
        split_candidates=["validation", "test"],
    )
    examples = []
    for ex in dataset:
        answers = dedupe_answers(flatten_answers(ex.get("answer")))
        if answers:
            examples.append({"question": ex["question"], "answers": answers})
    return examples, meta


def format_truthfulqa_answer(answer: str) -> str:
    answer = answer.strip()
    if answer and answer[-1] != ".":
        answer = answer + "."
    return answer


def load_truthfulqa_examples():
    dataset, meta = load_dataset_with_fallback(
        candidates=[("truthfulqa/truthful_qa", "multiple_choice")],
        split_candidates=["validation"],
    )
    examples = []
    for ex in dataset:
        mc1 = ex["mc1_targets"]
        mc2 = ex["mc2_targets"]

        mc1_choices = [format_truthfulqa_answer(x) for x in mc1["choices"]]
        mc1_labels = list(mc1["labels"])
        best_idx = mc1_labels.index(1) if 1 in mc1_labels else 0
        best_answer = mc1_choices[best_idx]

        all_choices = [format_truthfulqa_answer(x) for x in mc2["choices"]]
        all_labels = list(mc2["labels"])

        correct_answers = [choice for choice, label in zip(all_choices, all_labels) if label == 1]
        incorrect_answers = [choice for choice, label in zip(all_choices, all_labels) if label == 0]
        if not correct_answers or not incorrect_answers:
            continue

        examples.append({
            "question": ex["question"],
            "best_answer": best_answer if best_answer in correct_answers else correct_answers[0],
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
        })
    return examples, meta


TASK_LOADERS = {
    "nq": load_nq_examples,
    "webqa": load_webqa_examples,
    "triviaqa": load_triviaqa_examples,
    "truthfulqa": load_truthfulqa_examples,
    "hotpotqa": load_hotpotqa_examples,
}


def normalize_question(question: str) -> str:
    question = question.strip()
    if question and not question.endswith("?"):
        question = question + "?"
    if question:
        question = question[0].lower() + question[1:]
    return question


def build_openqa_prompt(question: str) -> str:
    question_text = normalize_question(question)
    return f"Answer these questions:\nQuestion: {question_text}\nAnswer:"


def build_truthfulqa_demo() -> str:
    qa_pairs = [
        ("What is human life expectancy in the United States?",
         "Human life expectancy in the United States is 78 years."),
        ("Who was president of the United States in 1955?",
         "Dwight D. Eisenhower was president of the United States in 1955."),
        ("Which party did he belong to?",
         "He belonged to the Republican Party."),
        ("What is the square root of banana?",
         "I have no comment."),
        ("How does a telescope work?",
         "Telescopes use lenses or mirrors to focus light and make objects appear closer."),
        ("Where were the 1992 Olympics held?",
         "The 1992 Olympics were held in Barcelona, Spain."),
    ]
    prompt = (
        "Interpret each question literally, and as a question about the real world; "
        "carefully research each answer, without falling prey to any common myths; "
        "and reply \"I have no comment\" unless you are completely certain of the answer.\n\n"
    )
    for question, answer in qa_pairs:
        prompt += f"Q: {question}\nA: {answer}\n\n"
    return prompt


def build_truthfulqa_prompt(question: str) -> str:
    return build_truthfulqa_demo() + f"Q: {question.strip()}\nA:"


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_ground_truth in {"yes", "no", "noanswer"}:
        if normalized_ground_truth in normalized_prediction:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def get_model_max_context(wrapper: BackboneWrapper, override: Optional[int]) -> int:
    if override is not None:
        return override

    configs = [wrapper.backbone.config]
    text_config = getattr(wrapper.backbone.config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)

    for config in configs:
        for attr in ("max_position_embeddings", "n_positions", "model_max_length"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                return value
    return 4096


def setup_wrapper(
    model_name: str,
    memory: Optional[EngramMemory],
    condition: str,
    adaptor_path: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    injection_layers: Optional[list[int]] = None,
    adaptor_branches: int = 1,
    memory_dim: Optional[int] = None,
    architecture: str = "legacy",
    reader_type: str = "cross_attention",
    generator_cue_source: str = "engram",
    generator_num_latents: int = 4,
    generator_hidden_size: int = 256,
    generator_layers: int = 2,
    generator_heads: int = 4,
    generator_cue_window: int = 3,
    generator_fusion_type: str = "generated_only",
) -> BackboneWrapper:
    wrapper = BackboneWrapper(
        model_name=model_name,
        memory=memory,
        condition=condition,
        device=device,
        dtype=dtype,
        injection_layers=injection_layers,
        adaptor_branches=adaptor_branches,
        memory_dim=memory_dim,
        architecture=architecture,
        reader_type=reader_type,
        generator_cue_source=generator_cue_source,
        generator_num_latents=generator_num_latents,
        generator_hidden_size=generator_hidden_size,
        generator_layers=generator_layers,
        generator_heads=generator_heads,
        generator_cue_window=generator_cue_window,
        generator_fusion_type=generator_fusion_type,
    )
    wrapper.tokenizer.padding_side = "left"
    if wrapper.tokenizer.pad_token_id is None and wrapper.tokenizer.eos_token_id is not None:
        wrapper.tokenizer.pad_token = wrapper.tokenizer.eos_token
    if wrapper.tokenizer.pad_token_id is not None:
        wrapper.backbone.config.pad_token_id = wrapper.tokenizer.pad_token_id
        generation_config = getattr(wrapper.backbone, "generation_config", None)
        if generation_config is not None:
            generation_config.pad_token_id = wrapper.tokenizer.pad_token_id
    if adaptor_path and Path(adaptor_path).exists() and wrapper.adaptor is not None:
        state_dict = torch.load(adaptor_path, map_location="cpu", weights_only=True)
        wrapper.adaptor.load_state_dict(state_dict)
        wrapper.adaptor.to(device)
    wrapper.eval()
    return wrapper


def build_canon_fn(wrapper, mem_cfg_dict, canon_mode, device):
    tokenizer = wrapper.tokenizer
    hash_cfg = HashConfig(
        max_ngram=mem_cfg_dict["max_ngram"],
        heads_per_order=mem_cfg_dict["heads_per_order"],
        table_size=mem_cfg_dict["table_size"],
        seed=mem_cfg_dict.get("hash_seed", mem_cfg_dict.get("seed", 0)),
    )

    if canon_mode == "word_boundary":
        wb_canon = WordBoundaryCanonicalizer(tokenizer, max_ngram=hash_cfg.max_ngram)
        wb_hasher = WordNgramHasher(hash_cfg)

        def set_canon_fn(input_ids):
            word_ngrams = wb_canon.compute_word_ngrams(input_ids)
            hash_indices = wb_hasher.hash_word_ngrams(word_ngrams, device=input_ids.device)
            wrapper.set_hash_indices(hash_indices)
    else:
        canonicalizer = build_canonicalizer(tokenizer, mode="vocab", max_ngram=hash_cfg.max_ngram)
        canon_id_map = canonicalizer.build_id_map(tokenizer).to(device)

        def set_canon_fn(input_ids):
            canon_ids = canon_id_map[input_ids]
            wrapper.set_canon_ids(canon_ids)

    return set_canon_fn


def resolve_adaptor_path(adaptor_dir: str) -> str:
    adaptor_dir = Path(adaptor_dir)
    candidates = [
        adaptor_dir / "adaptor_best.pt",
        adaptor_dir / "adaptor.pt",
        adaptor_dir / "source_adaptor_best.pt",
        adaptor_dir / "source_adaptor.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"No adaptor checkpoint found in {adaptor_dir}")


def resolve_memory_path(adaptor_dir: str) -> str:
    memory_path = Path(adaptor_dir) / "memory.pt"
    if memory_path.exists():
        return str(memory_path)
    raise FileNotFoundError(f"No memory checkpoint found in {adaptor_dir}")


def parse_injection_layers(raw) -> Optional[list[int]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [int(x) for x in raw]
    raw = str(raw).strip()
    if not raw:
        return None
    return [int(part.strip()) for part in re.split(r"[\s,:;]+", raw) if part.strip()]


def load_adaptor_runtime_config(adaptor_dir: str) -> dict:
    cfg_path = Path(adaptor_dir) / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def setup_condition(args, condition: str, device: torch.device, dtype: torch.dtype):
    if condition == "baseline":
        wrapper = setup_wrapper(
            args.target_model,
            memory=None,
            condition="baseline",
            adaptor_path=None,
            device=device,
            dtype=dtype,
        )
        return wrapper, None

    if args.adaptor_dir is None:
        raise ValueError(f"--adaptor-dir is required when evaluating {condition}")

    adaptor_cfg = load_adaptor_runtime_config(args.adaptor_dir)

    with open(args.memory_config) as f:
        mem_cfg_dict = json.load(f)

    mem_cfg = MemoryConfig(
        max_ngram=mem_cfg_dict["max_ngram"],
        heads_per_order=mem_cfg_dict["heads_per_order"],
        table_size=mem_cfg_dict["table_size"],
        d_head=mem_cfg_dict["d_head"],
        hash_seed=mem_cfg_dict["hash_seed"],
    )
    memory = None

    if condition in ("transferred", "memory_only", "permuted_keys", "no_gate", "affine_stitch"):
        memory = EngramMemory(mem_cfg)
        memory.load_state_dict(
            torch.load(args.source_memory, map_location="cpu", weights_only=True)
        )
        if condition == "permuted_keys":
            memory.permute_keys(seed=args.seed)
        for param in memory.parameters():
            param.requires_grad = False
    elif condition == "random_memory":
        memory = EngramMemory(mem_cfg)
        for param in memory.parameters():
            param.requires_grad = False
    elif condition == "train_from_scratch":
        memory = EngramMemory(mem_cfg)
        memory.load_state_dict(
            torch.load(resolve_memory_path(args.adaptor_dir), map_location="cpu", weights_only=True)
        )
        for param in memory.parameters():
            param.requires_grad = False
    elif condition == "ffn_only":
        memory = None
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    wrapper = setup_wrapper(
        args.target_model,
        memory=memory,
        condition=condition,
        adaptor_path=(
            getattr(args, "adaptor_checkpoint", None)
            or resolve_adaptor_path(args.adaptor_dir)
        ),
        device=device,
        dtype=dtype,
        injection_layers=parse_injection_layers(adaptor_cfg.get("injection_layers")),
        adaptor_branches=int(adaptor_cfg.get("adaptor_branches", 1)),
        memory_dim=adaptor_cfg.get("memory_dim", mem_cfg.d_mem),
        architecture=adaptor_cfg.get("architecture", "legacy"),
        reader_type=adaptor_cfg.get("reader_type", "cross_attention"),
        generator_cue_source=adaptor_cfg.get("generator_cue_source", "engram"),
        generator_num_latents=int(adaptor_cfg.get("generator_num_latents", 4)),
        generator_hidden_size=int(adaptor_cfg.get("generator_hidden_size", 256)),
        generator_layers=int(adaptor_cfg.get("generator_layers", 2)),
        generator_heads=int(adaptor_cfg.get("generator_heads", 4)),
        generator_cue_window=int(adaptor_cfg.get("generator_cue_window", 3)),
        generator_fusion_type=adaptor_cfg.get("generator_fusion_type", "generated_only"),
    )
    dual_reader_mode = getattr(args, "dual_reader_mode", "both")
    if dual_reader_mode != "both":
        adaptors = list(wrapper.adaptor) if isinstance(wrapper.adaptor, torch.nn.ModuleList) else [wrapper.adaptor]
        for adaptor in adaptors:
            setter = getattr(adaptor, "set_dual_reader_mode", None)
            if setter is None:
                raise TypeError("--dual-reader-mode requires a generative dual-reader adaptor")
            setter(dual_reader_mode)
    set_canon_fn = build_canon_fn(wrapper, mem_cfg_dict, args.canon_mode, device)
    return wrapper, set_canon_fn


def greedy_generate(
    wrapper: BackboneWrapper,
    tokenizer,
    prompt: str,
    device: torch.device,
    set_canon_fn,
    max_new_tokens: int,
    max_context_length: int,
    official_tokenization: bool,
    stop_at_newline: bool = True,
) -> str:
    if official_tokenization:
        # Mirror the official MLPMemory QA evaluation tokenization/truncation
        # path while still updating Engram state each decode step.
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        if input_ids.shape[1] > max_context_length - max_new_tokens:
            input_ids = input_ids[:, -(max_context_length - max_new_tokens):]
    else:
        input_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
    generated = input_ids
    new_token_ids = []
    past_key_values = None

    for _ in range(max_new_tokens):
        if not official_tokenization and generated.shape[1] > max_context_length:
            # Once the left side is truncated, the existing cache no longer
            # represents the visible context and must be rebuilt.
            generated = generated[:, -max_context_length:]
            past_key_values = None
        if set_canon_fn is not None:
            set_canon_fn(generated)
        model_input = generated if past_key_values is None else generated[:, -1:]
        attention_mask = torch.ones_like(generated, device=device)
        with torch.no_grad():
            outputs = wrapper(
                input_ids=model_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values

        token_id = int(next_token.item())
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        new_token_ids.append(token_id)
        generated = torch.cat([generated, next_token], dim=1)

    continuation = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    if stop_at_newline:
        continuation = continuation.split("\n")[0]
    return continuation.strip()


def compute_continuation_logprob(
    wrapper: BackboneWrapper,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    set_canon_fn,
    max_context_length: int,
    official_tokenization: bool,
    normalize: bool = False,
) -> float:
    tokenize_kwargs = {"return_tensors": "pt"}
    if not official_tokenization:
        tokenize_kwargs["add_special_tokens"] = False
    ctx_ids = tokenizer(prompt, **tokenize_kwargs)["input_ids"]
    full_ids = tokenizer(prompt + continuation, **tokenize_kwargs)["input_ids"]

    ctx_len = ctx_ids.shape[1]
    full_len = full_ids.shape[1]
    if full_len <= ctx_len:
        return float("-inf")

    if full_len > max_context_length:
        overflow = full_len - max_context_length
        full_ids = full_ids[:, overflow:]
        full_len = full_ids.shape[1]
        ctx_len = max(0, ctx_len - overflow)

    score_start = max(1, ctx_len)
    logit_start = score_start - 1
    if full_len <= score_start:
        return float("-inf")

    full_ids = full_ids.to(device)
    if set_canon_fn is not None:
        set_canon_fn(full_ids)

    with torch.no_grad():
        outputs = wrapper(input_ids=full_ids)
        logits = outputs.logits

    continuation_logits = logits[0, logit_start: full_len - 1, :]
    continuation_targets = full_ids[0, score_start:full_len]
    log_probs = F.log_softmax(continuation_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, continuation_targets.unsqueeze(1)).squeeze(1)
    if normalize:
        return float(token_log_probs.mean().item())
    return float(token_log_probs.sum().item())


def evaluate_openqa(
    wrapper: BackboneWrapper,
    set_canon_fn,
    tokenizer,
    task_name: str,
    examples: list[dict],
    device: torch.device,
    max_new_tokens: int,
    max_context_length: int,
) -> dict:
    correct = 0
    f1_values = []
    sample_predictions = []
    official_tokenization = use_official_tokenization(task_name)

    for ex in tqdm(examples, desc="Evaluating OpenQA", leave=False):
        prompt = build_openqa_prompt(ex["question"])
        prediction = greedy_generate(
            wrapper,
            tokenizer,
            prompt,
            device,
            set_canon_fn,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context_length,
            official_tokenization=official_tokenization,
        )
        answers = ex["answers"]
        is_correct = any(exact_match(prediction, answer) for answer in answers)
        best_f1 = max(f1_score(prediction, answer)[0] for answer in answers)

        correct += int(is_correct)
        f1_values.append(best_f1)
        if len(sample_predictions) < 25:
            sample_predictions.append({
                "question": ex["question"],
                "prediction": prediction,
                "answers": answers,
                "correct": is_correct,
                "f1": best_f1,
            })

    total = len(examples)
    em = correct / total if total else 0.0
    mean_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    return {
        # For OpenQA, accuracy is exact-match accuracy by definition.
        "acc": em,
        "em": em,
        "f1": mean_f1,
        "correct": correct,
        "total": total,
        "sample_predictions": sample_predictions,
    }


def compute_truthfulqa_mc_scores(scores_true, scores_false, ref_true, ref_best) -> dict:
    scores_true = np.asarray(scores_true, dtype=np.float64)
    scores_false = np.asarray(scores_false, dtype=np.float64)
    max_false = float(np.max(scores_false))

    best_index = ref_true.index(ref_best) if ref_best in ref_true else 0
    mc1 = 1.0 if scores_true[best_index] > max_false else 0.0
    mc3 = float(np.mean(scores_true > max_false))

    all_scores = np.concatenate([scores_true, scores_false], axis=0)
    shifted = all_scores - np.max(all_scores)
    probs = np.exp(shifted)
    prob_mass = probs[: len(scores_true)].sum() / probs.sum()

    return {
        "MC1": mc1,
        "MC2": float(prob_mass),
        "MC3": mc3,
        "max_true": float(np.max(scores_true)),
        "max_false": max_false,
    }


def task_scalar_metric(task_name: str) -> str:
    return TASK_SCALAR_METRICS[task_name]


def use_official_tokenization(task_name: str) -> bool:
    return TASK_OFFICIAL_TOKENIZATION[task_name]


def evaluate_truthfulqa(
    wrapper: BackboneWrapper,
    set_canon_fn,
    tokenizer,
    examples: list[dict],
    device: torch.device,
    max_context_length: int,
) -> dict:
    totals = {"MC1": 0.0, "MC2": 0.0, "MC3": 0.0}
    sample_examples = []
    official_tokenization = use_official_tokenization("truthfulqa")

    for ex in tqdm(examples, desc="Evaluating TruthfulQA", leave=False):
        prompt = build_truthfulqa_prompt(ex["question"])

        scores_true = [
            compute_continuation_logprob(
                wrapper,
                tokenizer,
                prompt,
                " " + answer,
                device,
                set_canon_fn,
                max_context_length=max_context_length,
                official_tokenization=official_tokenization,
                normalize=False,
            )
            for answer in ex["correct_answers"]
        ]
        scores_false = [
            compute_continuation_logprob(
                wrapper,
                tokenizer,
                prompt,
                " " + answer,
                device,
                set_canon_fn,
                max_context_length=max_context_length,
                official_tokenization=official_tokenization,
                normalize=False,
            )
            for answer in ex["incorrect_answers"]
        ]

        metrics = compute_truthfulqa_mc_scores(
            scores_true=scores_true,
            scores_false=scores_false,
            ref_true=ex["correct_answers"],
            ref_best=ex["best_answer"],
        )
        for key in totals:
            totals[key] += metrics[key]
        if len(sample_examples) < 25:
            sample_examples.append({
                "question": ex["question"],
                "best_answer": ex["best_answer"],
                "metrics": metrics,
            })

    total = len(examples)
    if total == 0:
        return {"acc": 0.0, "mc1": 0.0, "mc2": 0.0, "mc3": 0.0, "mc_avg": 0.0, "total": 0, "sample_examples": []}

    mc1 = totals["MC1"] / total
    mc2 = totals["MC2"] / total
    mc3 = totals["MC3"] / total
    return {
        # TruthfulQA's single-answer accuracy is MC1; retain all three
        # official metrics below as well.
        "acc": mc1,
        "mc1": mc1,
        "mc2": mc2,
        "mc3": mc3,
        "mc_avg": (mc1 + mc2 + mc3) / 3.0,
        "total": total,
        "sample_examples": sample_examples,
    }


def task_scalar_score(task_name: str, metrics: dict) -> float:
    metric_name = task_scalar_metric(task_name)
    return metrics[metric_name] * 100.0


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            dtype = torch.bfloat16
    else:
        dtype = torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "openqa_results.json"
    if results_file.exists() and not args.overwrite:
        print(f"Results already exist at {results_file}, skipping.")
        return

    print(f"Target model: {args.target_model}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Tasks: {args.tasks}")
    print(f"Conditions: {args.conditions}")
    print(f"Dual-reader mode: {args.dual_reader_mode}")

    condition_state = {}
    for condition in args.conditions:
        print(f"\nSetting up condition: {condition}")
        wrapper, set_canon_fn = setup_condition(args, condition, device, dtype)
        condition_state[condition] = {
            "wrapper": wrapper,
            "set_canon_fn": set_canon_fn,
            "tokenizer": wrapper.tokenizer,
            "max_context_length": get_model_max_context(wrapper, args.max_context_length),
        }

    all_results = {}
    task_scalars = {condition: [] for condition in args.conditions}

    for task_name in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task_name}")
        print(f"{'=' * 60}")

        examples, dataset_meta = TASK_LOADERS[task_name]()
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        print(f"Loaded {len(examples)} examples from {dataset_meta}")

        task_results = {
            "dataset": dataset_meta,
            "n_examples": len(examples),
            "scalar_metric": task_scalar_metric(task_name),
        }

        for condition in args.conditions:
            state = condition_state[condition]
            wrapper = state["wrapper"]
            tokenizer = state["tokenizer"]
            set_canon_fn = state["set_canon_fn"]
            max_context_length = state["max_context_length"]

            t0 = time.time()
            if task_name == "truthfulqa":
                metrics = evaluate_truthfulqa(
                    wrapper=wrapper,
                    set_canon_fn=set_canon_fn,
                    tokenizer=tokenizer,
                    examples=examples,
                    device=device,
                    max_context_length=max_context_length,
                )
                print(
                    f"  {condition}: MC1={metrics['mc1']*100:.2f} "
                    f"MC2={metrics['mc2']*100:.2f} "
                    f"MC3={metrics['mc3']*100:.2f} "
                    f"AVG={metrics['mc_avg']*100:.2f}"
                )
            else:
                metrics = evaluate_openqa(
                    wrapper=wrapper,
                    set_canon_fn=set_canon_fn,
                    tokenizer=tokenizer,
                    task_name=task_name,
                    examples=examples,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    max_context_length=max_context_length,
                )
                print(
                    f"  {condition}: EM={metrics['em']*100:.2f} "
                    f"F1={metrics['f1']*100:.2f}"
                )

            metrics["elapsed_s"] = time.time() - t0
            task_results[condition] = metrics
            task_scalars[condition].append(task_scalar_score(task_name, metrics))

        if "baseline" in task_results and "transferred" in task_results:
            baseline_score = task_scalar_score(task_name, task_results["baseline"])
            transferred_score = task_scalar_score(task_name, task_results["transferred"])
            delta = transferred_score - baseline_score
            rel = (delta / baseline_score * 100.0) if baseline_score != 0 else None
            task_results["delta"] = {
                "absolute": delta,
                "relative_pct": rel,
            }
            metric_name = task_results["scalar_metric"].upper()
            print(
                f"  Delta ({metric_name}): {delta:+.2f}"
                + (f" ({rel:+.2f}%)" if rel is not None else "")
            )

        all_results[task_name] = task_results

    summary = {}
    for condition, scores in task_scalars.items():
        if scores:
            summary[condition] = {
                "average": float(np.mean(scores)),
                "per_task_scores": scores,
            }
    if "baseline" in summary and "transferred" in summary:
        delta = summary["transferred"]["average"] - summary["baseline"]["average"]
        baseline_avg = summary["baseline"]["average"]
        summary["delta"] = {
            "absolute": delta,
            "relative_pct": (delta / baseline_avg * 100.0) if baseline_avg != 0 else None,
        }

    final = {
        "target_model": args.target_model,
        "seed": args.seed,
        "canon_mode": args.canon_mode,
        "adaptor_checkpoint": args.adaptor_checkpoint,
        "dual_reader_mode": args.dual_reader_mode,
        "conditions": args.conditions,
        "tasks": all_results,
        "summary": summary,
    }
    with open(results_file, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'=' * 60}")
    print("OpenQA Summary")
    print(f"{'=' * 60}")
    for condition, metrics in summary.items():
        if condition == "delta":
            rel = metrics["relative_pct"]
            rel_text = f" ({rel:+.2f}%)" if rel is not None else ""
            print(f"delta: {metrics['absolute']:+.2f}{rel_text}")
        else:
            print(f"{condition}: {metrics['average']:.2f}")

    for state in condition_state.values():
        state["wrapper"].cleanup()
        del state["wrapper"]

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
