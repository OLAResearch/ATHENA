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
import concurrent.futures
import json
import os
import random
import re
import subprocess
import sys
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
        choices=[
            "auto",
            "both",
            "engram_only",
            "generated_only",
            "routed",
            "generated_from_engram_only",
            "generated_from_context_only",
            "e_ge",
            "e_gh",
            "ge_gh",
            "tri_routed",
            "tri_soft_fused",
            "tri_safe_routed",
            "tri_subset_routed",
            "tri_subset_soft_fused",
            "tri_advantage_routed",
        ],
        default="auto",
        help=(
            "Runtime reader mode. auto uses checkpoint deployment metadata, "
            "then known training flags; legacy checkpoints keep the historical "
            "default. Explicit modes are never overridden."
        ),
    )
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument(
        "--triviaqa-config",
        type=str,
        choices=["rc.nocontext"],
        default="rc.nocontext",
        help="Fixed TriviaQA validation configuration used by the five-task comparison",
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--conditions", nargs="+", default=["baseline", "transferred"],
                        choices=EVAL_CONDITIONS)
    parser.add_argument("--canon-mode", type=str, default="word_boundary",
                        choices=["vocab", "word_boundary"])
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit examples per task (<=0 disables the limit)")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument(
        "--reasoning-mode",
        choices=["vanilla", "cot"],
        default="vanilla",
        help=(
            "Prompt mode for generated-answer tasks. Vanilla preserves the "
            "historical eval_openqa prompt; cot extracts the final answer "
            "from an optional <answer> block before scoring."
        ),
    )
    parser.add_argument("--max-context-length", type=int, default=None,
                        help="Override model max context during eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parallel-tasks",
        type=int,
        default=1,
        help=(
            "Evaluate tasks in isolated subprocesses (one model per worker). "
            "Values <=1 preserve the historical sequential path."
        ),
    )
    parser.add_argument(
        "--task-devices",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device ids for --parallel-tasks. Devices are "
            "assigned round-robin; defaults to the current CUDA visibility."
        ),
    )
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


def load_triviaqa_examples(config_name: str = "rc.nocontext"):
    if config_name != "rc.nocontext":
        raise ValueError(
            "The comparable five-task evaluation requires TriviaQA config rc.nocontext"
        )
    dataset, meta = load_dataset_with_fallback(
        candidates=[
            # Do not fall back to a different TriviaQA configuration: the
            # smaller rc.wikipedia.nocontext split makes method comparisons
            # depend on cache availability.
            ("mandarjoshi/trivia_qa", config_name),
        ],
        split_candidates=["validation"],
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


def build_openqa_prompt(question: str, reasoning_mode: str = "vanilla") -> str:
    question_text = normalize_question(question)
    if reasoning_mode == "cot":
        return (
            "Answer these questions step by step, then put only the concise final "
            "answer inside <answer></answer>.\n"
            f"Question: {question_text}\nAnswer:"
        )
    return f"Answer these questions:\nQuestion: {question_text}\nAnswer:"


def truncate_openqa_prediction(prediction: str) -> str:
    """Stop before a continuation begins another QA item."""
    first_line = prediction.split("\n", 1)[0]
    return re.split(
        r"\s*(?:;\s*)?(?:Question|Q)\s*:",
        first_line,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def _adjacent_repeated_ngram_count(tokens: list[str], n: int) -> int:
    if n < 1 or len(tokens) < 2 * n:
        return 0
    return sum(
        1
        for index in range(len(tokens) - 2 * n + 1)
        if tokens[index : index + n] == tokens[index + n : index + 2 * n]
    )


def prediction_diagnostics(raw_prediction: str, extracted_prediction: str) -> dict:
    """Describe generated text without participating in any QA score."""
    raw_tokens = raw_prediction.split()
    extracted_tokens = extracted_prediction.split()

    def describe(tokens: list[str], text: str) -> dict:
        repeated = {
            f"adjacent_repeated_{n}gram_count": _adjacent_repeated_ngram_count(
                tokens, n
            )
            for n in (1, 2, 3)
        }
        repeated_total = sum(repeated.values())
        return {
            "length_chars": len(text),
            "length_tokens": len(tokens),
            **repeated,
            "repetition_count": repeated_total,
            "has_repetition": bool(repeated_total),
        }

    return {
        "raw": describe(raw_tokens, raw_prediction),
        "extracted": describe(extracted_tokens, extracted_prediction),
        "raw_equals_extracted": raw_prediction == extracted_prediction,
        "extraction_changed": raw_prediction != extracted_prediction,
    }


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


def _reader_adaptors(wrapper):
    """Return adaptor modules without assuming singleton or ModuleList layout."""
    if wrapper.adaptor is None:
        return []
    if isinstance(wrapper.adaptor, torch.nn.ModuleList):
        return list(wrapper.adaptor)
    return [wrapper.adaptor]


def configure_advantage_reader(wrapper, advantage_reader: Optional[dict]) -> bool:
    """Materialize a lazy advantage head before loading a strict checkpoint.

    The advantage reader is deliberately configured per adaptor because a
    checkpoint may contain either a singleton adaptor or a layer-indexed
    ``ModuleList``.  Only the public runtime controls are forwarded; metadata
    added by a trainer is retained in ``config.json`` but cannot accidentally
    become an unsupported keyword argument.
    """
    if advantage_reader is None:
        return False
    if not isinstance(advantage_reader, dict):
        raise TypeError("config['advantage_reader'] must be a dictionary")
    if not advantage_reader.get("enabled", True):
        return False
    adaptors = _reader_adaptors(wrapper)
    if not adaptors:
        raise ValueError(
            "config['advantage_reader'] requires a generative adaptor checkpoint"
        )

    values = {
        "candidates": advantage_reader.get("candidates", "sources"),
        "threshold": float(advantage_reader.get("threshold", 0.0)),
        "confidence_threshold": float(
            advantage_reader.get("confidence_threshold", 0.5)
        ),
        "temperature": float(advantage_reader.get("temperature", 0.15)),
        "max_scale": float(advantage_reader.get("max_scale", 1.0)),
    }
    if values["candidates"] not in {"sources", "subsets"}:
        raise ValueError(
            "advantage_reader.candidates must be 'sources' or 'subsets'"
        )
    for adaptor in adaptors:
        configure = getattr(adaptor, "configure_advantage_reader", None)
        if configure is None:
            raise AttributeError(
                f"{type(adaptor).__name__} does not support advantage_reader"
            )
        # This call is intentionally before the checkpoint load in
        # ``setup_wrapper``.  The core implementation lazily registers the
        # head, so its tensors are then part of the strict expected state.
        configure(**values)
    return True


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
    generator_adaptive_router: bool = False,
    generator_router_hidden_size: int = 16,
    generator_router_semantic_size: int = 0,
    generator_router_expert_mode: str = "residual",
    generator_source_adapter_rank: int = 16,
    generator_loop_rounds: int = 1,
    generator_loop_workspace_size: int = 0,
    generator_loop_gate_max: float = 0.25,
    advantage_reader: Optional[dict] = None,
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
        generator_adaptive_router=generator_adaptive_router,
        generator_router_hidden_size=generator_router_hidden_size,
        generator_router_semantic_size=generator_router_semantic_size,
        generator_router_expert_mode=generator_router_expert_mode,
        generator_source_adapter_rank=generator_source_adapter_rank,
        generator_loop_rounds=generator_loop_rounds,
        generator_loop_workspace_size=generator_loop_workspace_size,
        generator_loop_gate_max=generator_loop_gate_max,
    )
    wrapper.tokenizer.padding_side = "left"
    if wrapper.tokenizer.pad_token_id is None and wrapper.tokenizer.eos_token_id is not None:
        wrapper.tokenizer.pad_token = wrapper.tokenizer.eos_token
    if wrapper.tokenizer.pad_token_id is not None:
        wrapper.backbone.config.pad_token_id = wrapper.tokenizer.pad_token_id
        generation_config = getattr(wrapper.backbone, "generation_config", None)
        if generation_config is not None:
            generation_config.pad_token_id = wrapper.tokenizer.pad_token_id
    # The advantage head is lazy by design.  It must exist before a strict
    # state-dict load, otherwise a valid new checkpoint appears incomplete.
    configure_advantage_reader(wrapper, advantage_reader)
    if adaptor_path and Path(adaptor_path).exists() and wrapper.adaptor is not None:
        state_dict = torch.load(adaptor_path, map_location="cpu", weights_only=True)
        wrapper.adaptor.load_state_dict(state_dict, strict=True)
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


def _is_tri_reader_config(adaptor_cfg: dict) -> bool:
    return (
        adaptor_cfg.get("generator_fusion_type") == "tri_reader"
        or adaptor_cfg.get("fusion_type") == "tri_reader"
        or any(
            bool(adaptor_cfg.get(key))
            for key in (
                "joint_tri_reader",
                "joint_tri_subset_reader",
                "joint_tri_route_only",
            )
        )
    )


def resolve_reader_mode_contract(
    requested_mode: str,
    adaptor_cfg: Optional[dict] = None,
    *,
    condition: str = "transferred",
) -> dict:
    """Resolve an evaluator reader mode and record why it was selected.

    ``auto`` is intentionally resolved from deployment metadata first.  The
    training flags are only a compatibility fallback for checkpoints written
    before ``deployment_reader_mode`` was persisted.  A checkpoint without
    either form of metadata uses the historical mode: ``both`` for legacy
    dual readers and hard ``tri_routed`` for legacy tri readers.
    """
    requested_mode = requested_mode or "auto"
    adaptor_cfg = adaptor_cfg or {}
    if condition == "baseline":
        return {
            "requested_reader_mode": requested_mode,
            "effective_reader_mode": None,
            "reader_mode_provenance": "baseline_bare_backbone",
        }
    if requested_mode != "auto":
        return {
            "requested_reader_mode": requested_mode,
            "effective_reader_mode": requested_mode,
            "reader_mode_provenance": "explicit_cli",
        }

    # Advantage-reader checkpoints carry a stronger deployment contract than
    # legacy tri routing metadata.  Prefer it when older configs still retain
    # ``tri_routed``/``tri_soft_fused`` flags from the source expert run.
    advantage_reader = adaptor_cfg.get("advantage_reader")
    if isinstance(advantage_reader, dict) and advantage_reader.get("enabled"):
        return {
            "requested_reader_mode": requested_mode,
            "effective_reader_mode": "tri_advantage_routed",
            "reader_mode_provenance": "checkpoint.advantage_reader",
        }

    deployment = adaptor_cfg.get("deployment_reader_mode")
    if deployment and deployment != "auto":
        return {
            "requested_reader_mode": requested_mode,
            "effective_reader_mode": str(deployment),
            "reader_mode_provenance": "checkpoint.deployment_reader_mode",
        }

    if bool(adaptor_cfg.get("joint_tri_route_only")):
        return {
            "requested_reader_mode": requested_mode,
            "effective_reader_mode": "tri_soft_fused",
            "reader_mode_provenance": "checkpoint.training_flags.joint_tri_route_only",
        }
    if bool(adaptor_cfg.get("joint_tri_subset_reader")):
        return {
            "requested_reader_mode": requested_mode,
            "effective_reader_mode": "tri_subset_soft_fused",
            "reader_mode_provenance": "checkpoint.training_flags.joint_tri_subset_reader",
        }
    if bool(adaptor_cfg.get("joint_tri_reader")):
        return {
            "requested_reader_mode": requested_mode,
            "effective_reader_mode": "tri_routed",
            "reader_mode_provenance": "checkpoint.training_flags.joint_tri_reader",
        }

    training = adaptor_cfg.get("training_reader_mode")
    if training and training != "auto":
        return {
            "requested_reader_mode": requested_mode,
            "effective_reader_mode": str(training),
            "reader_mode_provenance": "checkpoint.training_reader_mode",
        }

    return {
        "requested_reader_mode": requested_mode,
        "effective_reader_mode": "tri_routed" if _is_tri_reader_config(adaptor_cfg) else "both",
        "reader_mode_provenance": "legacy_default",
    }


def resolve_reader_mode(
    requested_mode: str,
    adaptor_cfg: Optional[dict] = None,
    *,
    condition: str = "transferred",
) -> Optional[str]:
    """Return only the effective mode for callers that do not need provenance."""
    return resolve_reader_mode_contract(
        requested_mode, adaptor_cfg, condition=condition
    )["effective_reader_mode"]


def _apply_reader_mode(wrapper, mode: Optional[str]) -> Optional[str]:
    """Apply a resolved mode and return the actual runtime mode."""
    if mode is None:
        return None
    applied = mode
    for adaptor in _reader_adaptors(wrapper):
        tri_setter = getattr(adaptor, "set_tri_reader_mode", None)
        is_tri = tri_setter is not None and getattr(
            adaptor, "fusion_type", None
        ) == "tri_reader"
        if is_tri:
            # ``both`` was the historical dual-reader spelling.  A tri reader
            # has no literal ``both`` path; retain the explicit alias by using
            # its soft all-source endpoint.
            applied = "tri_soft_fused" if mode == "both" else mode
            tri_setter(applied)
        else:
            setter = getattr(adaptor, "set_dual_reader_mode", None)
            if mode == "both" and setter is None:
                # A plain Engram adaptor has only its historical E path.  The
                # legacy ``both`` fallback is intentionally a no-op here;
                # callers asking for a different mode still get an error.
                continue
            if setter is None:
                raise TypeError(
                    "Reader mode requires a compatible generative reader adaptor"
                )
            setter(applied)
    return applied


def reader_mode_result_metadata(wrapper) -> dict:
    """Return serializable mode metadata attached by ``setup_condition``."""
    metadata = getattr(wrapper, "reader_mode_contract", None)
    if metadata is None:
        return {
            "requested_reader_mode": None,
            "effective_reader_mode": None,
            "reader_mode_provenance": "unspecified",
        }
    return dict(metadata)


def configure_adaptive_routers(wrapper, adaptor_cfg: dict) -> None:
    """Restore runtime router controls that are not stored in a state dict."""
    adaptors = (
        list(wrapper.adaptor)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList)
        else [wrapper.adaptor]
    )
    for adaptor in adaptors:
        if getattr(adaptor, "router", None) is not None:
            router_kwargs = {
                "temperature": float(adaptor_cfg.get("router_temperature", 1.0)),
                "hard": bool(adaptor_cfg.get("router_hard", False)),
                "min_generated_probability": float(
                    adaptor_cfg.get("router_min_generated_probability", 0.5)
                ),
            }
            if getattr(adaptor, "fusion_type", None) == "tri_reader":
                router_kwargs.update(
                    safe_residual_threshold=float(
                        adaptor_cfg.get("safe_residual_threshold", 1.0)
                    ),
                    safe_residual_scale=float(
                        adaptor_cfg.get("safe_residual_scale", 1.0)
                    ),
                )
            adaptor.configure_router(**router_kwargs)


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
        # Test doubles and plain baseline wrappers may intentionally expose no
        # mutable attributes; baseline has no reader mode to configure.
        if hasattr(wrapper, "__dict__"):
            wrapper.reader_mode_contract = resolve_reader_mode_contract(
                getattr(args, "dual_reader_mode", "auto"),
                {},
                condition="baseline",
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

    if condition in ("random_memory", "permuted_keys"):
        saved_memory = Path(args.adaptor_dir) / "memory.pt"
        if not saved_memory.exists():
            raise FileNotFoundError("Content ablations require the exact training memory.pt")
        memory.load_state_dict(torch.load(saved_memory, map_location="cpu", weights_only=True))

    mode_contract = resolve_reader_mode_contract(
        getattr(args, "dual_reader_mode", "auto"), adaptor_cfg, condition=condition
    )
    advantage_reader = adaptor_cfg.get("advantage_reader")
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
        generator_adaptive_router=bool(
            adaptor_cfg.get("generator_adaptive_router", False)
            or (
                isinstance(advantage_reader, dict)
                and advantage_reader.get("enabled", True)
            )
        ),
        generator_router_hidden_size=int(adaptor_cfg.get("generator_router_hidden_size", 16)),
        generator_router_semantic_size=int(
            adaptor_cfg.get("generator_router_semantic_size", 0)
        ),
        generator_router_expert_mode=adaptor_cfg.get(
            "generator_router_expert_mode", "residual"
        ),
        generator_source_adapter_rank=int(
            adaptor_cfg.get("generator_source_adapter_rank", 16)
        ),
        generator_loop_rounds=int(adaptor_cfg.get("generator_loop_rounds", 1)),
        generator_loop_workspace_size=int(
            adaptor_cfg.get("generator_loop_workspace_size", 0)
        ),
        generator_loop_gate_max=float(
            adaptor_cfg.get("generator_loop_gate_max", 0.25)
        ),
        advantage_reader=advantage_reader,
    )
    configure_adaptive_routers(wrapper, adaptor_cfg)
    applied_mode = _apply_reader_mode(
        wrapper, mode_contract["effective_reader_mode"]
    )
    mode_contract["effective_reader_mode"] = applied_mode
    mode_contract["applied_reader_mode"] = applied_mode
    wrapper.reader_mode_contract = mode_contract
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
    temperature: float = 0.0,
    top_p: float = 1.0,
    routing_collector=None,
) -> str:
    """Generate one continuation, optionally using nucleus sampling.

    The default remains the historical greedy path.  Sampling is used only by
    the code Pass@k evaluator and still runs through the same wrapper/canon
    state updates as greedy generation.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
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
            if routing_collector is not None:
                routing_collector.add_last_token(wrapper)
            logits = outputs.logits[:, -1, :]
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative_probs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, sorted_indices, sorted_logits)
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
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
    routing_collector=None,
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
    if routing_collector is not None:
        routing_collector.add_slice(wrapper, logit_start, full_len - 1)

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
    reasoning_mode: str = "vanilla",
    retain_all_predictions: bool = False,
    routing_collector=None,
) -> dict:
    correct = 0
    f1_values = []
    sample_predictions = []
    diagnostic_totals = {
        "raw_length_chars": 0,
        "raw_length_tokens": 0,
        "extracted_length_chars": 0,
        "extracted_length_tokens": 0,
        "raw_repetition_count": 0,
        "extracted_repetition_count": 0,
        "extraction_changed": 0,
    }
    official_tokenization = use_official_tokenization(task_name)

    for ex in tqdm(examples, desc="Evaluating OpenQA", leave=False):
        prompt = build_openqa_prompt(ex["question"], reasoning_mode=reasoning_mode)
        prediction = greedy_generate(
            wrapper,
            tokenizer,
            prompt,
            device,
            set_canon_fn,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context_length,
            official_tokenization=official_tokenization,
            stop_at_newline=(reasoning_mode == "vanilla"),
            routing_collector=routing_collector,
        )
        raw_prediction = prediction
        if reasoning_mode == "cot":
            answer_matches = re.findall(
                r"<answer>(.*?)</answer>", prediction, flags=re.DOTALL | re.IGNORECASE
            )
            if answer_matches:
                prediction = answer_matches[-1].strip()
            else:
                lines = [line.strip() for line in prediction.splitlines() if line.strip()]
                prediction = lines[-1] if lines else ""
                prediction = re.sub(
                    r"^(?:final answer|answer)\s*:\s*",
                    "",
                    prediction,
                    flags=re.IGNORECASE,
                ).strip()
        else:
            prediction = truncate_openqa_prediction(prediction)
        prediction_info = prediction_diagnostics(raw_prediction, prediction)
        diagnostic_totals["raw_length_chars"] += prediction_info["raw"]["length_chars"]
        diagnostic_totals["raw_length_tokens"] += prediction_info["raw"]["length_tokens"]
        diagnostic_totals["extracted_length_chars"] += prediction_info["extracted"]["length_chars"]
        diagnostic_totals["extracted_length_tokens"] += prediction_info["extracted"]["length_tokens"]
        diagnostic_totals["raw_repetition_count"] += prediction_info["raw"]["repetition_count"]
        diagnostic_totals["extracted_repetition_count"] += prediction_info["extracted"]["repetition_count"]
        diagnostic_totals["extraction_changed"] += int(
            prediction_info["extraction_changed"]
        )
        answers = ex["answers"]
        is_correct = any(exact_match(prediction, answer) for answer in answers)
        best_f1 = max(f1_score(prediction, answer)[0] for answer in answers)

        correct += int(is_correct)
        f1_values.append(best_f1)
        if retain_all_predictions or len(sample_predictions) < 25:
            sample_predictions.append({
                "question": ex["question"],
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "extracted_prediction": prediction,
                "prediction_diagnostics": prediction_info,
                "answers": answers,
                "correct": is_correct,
                "f1": best_f1,
            })

    total = len(examples)
    em = correct / total if total else 0.0
    mean_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    denominator = max(total, 1)
    return {
        # For OpenQA, accuracy is exact-match accuracy by definition.
        "acc": em,
        "em": em,
        "f1": mean_f1,
        "correct": correct,
        "total": total,
        "sample_predictions": sample_predictions,
        "prediction_diagnostics": {
            "n_examples": total,
            "raw_length_chars_mean": diagnostic_totals["raw_length_chars"] / denominator,
            "raw_length_tokens_mean": diagnostic_totals["raw_length_tokens"] / denominator,
            "extracted_length_chars_mean": diagnostic_totals["extracted_length_chars"] / denominator,
            "extracted_length_tokens_mean": diagnostic_totals["extracted_length_tokens"] / denominator,
            "raw_repetition_count": diagnostic_totals["raw_repetition_count"],
            "extracted_repetition_count": diagnostic_totals["extracted_repetition_count"],
            "raw_repetition_rate": diagnostic_totals["raw_repetition_count"] / denominator,
            "extracted_repetition_rate": diagnostic_totals["extracted_repetition_count"] / denominator,
            "extraction_changed": diagnostic_totals["extraction_changed"],
            "extraction_changed_rate": diagnostic_totals["extraction_changed"] / denominator,
        },
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
    retain_all_examples: bool = False,
    routing_collector=None,
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
                routing_collector=routing_collector,
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
                routing_collector=routing_collector,
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
        if retain_all_examples or len(sample_examples) < 25:
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


def _drop_cli_option(argv: list[str], option: str, multiple: bool = False) -> list[str]:
    """Remove one option from an argv list without interpreting other options.

    ``--tasks`` is the only option with a variable number of values. Keeping
    this helper deliberately small makes the parallel launcher robust to new
    evaluator flags: all arguments other than the ones it owns are passed
    through unchanged to each isolated child process.
    """
    result = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            index += 1
            if multiple:
                while index < len(argv) and not argv[index].startswith("--"):
                    index += 1
            elif index < len(argv):
                index += 1
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def _parallel_child_argv(task: str, output_dir: Path) -> list[str]:
    """Build argv for one task while preserving every evaluator setting."""
    child = list(sys.argv[1:])
    child = _drop_cli_option(child, "--tasks", multiple=True)
    child = _drop_cli_option(child, "--output-dir")
    child = _drop_cli_option(child, "--parallel-tasks")
    child = _drop_cli_option(child, "--task-devices")
    child.append("--tasks")
    child.append(task)
    child.extend(["--output-dir", str(output_dir), "--parallel-tasks", "1"])
    return child


def _parse_task_devices(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _aggregate_parallel_results(args, task_dirs: dict[str, Path]) -> dict:
    """Merge one-task child artifacts into the historical result schema."""
    child_payloads = {}
    for task in args.tasks:
        result_path = task_dirs[task] / "openqa_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing child result for {task}: {result_path}")
        with result_path.open() as handle:
            payload = json.load(handle)
        task_results = payload.get("tasks", {})
        if set(task_results) != {task}:
            raise ValueError(
                f"Child result for {task} contains tasks {sorted(task_results)}"
            )
        child_payloads[task] = payload

    all_results = {task: child_payloads[task]["tasks"][task] for task in args.tasks}
    summary = {}
    for condition in args.conditions:
        scores = [
            task_scalar_score(task, all_results[task][condition])
            for task in args.tasks
        ]
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

    return {
        "target_model": args.target_model,
        "seed": args.seed,
        "canon_mode": args.canon_mode,
        "triviaqa_config": args.triviaqa_config,
        "reasoning_mode": args.reasoning_mode,
        "adaptor_checkpoint": args.adaptor_checkpoint,
        "dual_reader_mode": args.dual_reader_mode,
        "reader_mode_resolution": child_payloads[args.tasks[0]].get(
            "reader_mode_resolution", {}
        ) if args.tasks else {},
        "conditions": args.conditions,
        "tasks": all_results,
        "summary": summary,
    }


def _run_parallel_tasks(args, output_dir: Path) -> None:
    """Evaluate tasks in isolated processes and aggregate their artifacts.

    Each process loads exactly one model/adaptor pair and evaluates exactly one
    task. This avoids sharing mutable hash/adaptor state across tasks and lets
    callers map workers to separate GPUs. The child command uses
    ``--parallel-tasks 1``, so this path cannot recursively fan out.
    """
    requested_workers = max(1, int(args.parallel_tasks))
    devices = _parse_task_devices(args.task_devices)
    if not devices and torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            devices = [part.strip() for part in visible.split(",") if part.strip()]
        else:
            devices = [str(index) for index in range(torch.cuda.device_count())]
    if devices:
        workers = min(requested_workers, len(devices), len(args.tasks))
    else:
        workers = min(requested_workers, len(args.tasks))

    task_dirs = {}
    for task in args.tasks:
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_dirs[task] = task_dir

    print(
        f"Parallel evaluation: {len(args.tasks)} tasks, {workers} workers"
        + (f", devices={devices}" if devices else ", CPU workers")
    )

    def launch(item):
        task_index, task = item
        task_dir = task_dirs[task]
        stdout_path = task_dir / "stdout.log"
        stderr_path = task_dir / "stderr.log"
        command = [sys.executable, str(Path(__file__).resolve())] + _parallel_child_argv(
            task, task_dir
        )
        env = os.environ.copy()
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[task_index % len(devices)]
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
        return task, completed.returncode, stdout_path, stderr_path

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, item) for item in enumerate(args.tasks)]
        for future in concurrent.futures.as_completed(futures):
            task, returncode, stdout_path, stderr_path = future.result()
            print(f"  {task}: {'ok' if returncode == 0 else f'failed ({returncode})'}")
            if returncode != 0:
                failures.append((task, returncode, stderr_path))

    if failures:
        details = []
        for task, returncode, stderr_path in failures:
            try:
                tail = stderr_path.read_text(errors="replace")[-2000:]
            except OSError:
                tail = "<stderr unavailable>"
            details.append(f"{task} (exit {returncode})\n{tail}")
        raise RuntimeError("Parallel evaluation failed:\n" + "\n".join(details))

    final = _aggregate_parallel_results(args, task_dirs)
    final["parallel"] = {
        "workers": workers,
        "devices": devices,
        "task_dirs": {task: str(path) for task, path in task_dirs.items()},
    }
    results_file = output_dir / "openqa_results.json"
    with results_file.open("w") as handle:
        json.dump(final, handle, indent=2)
    print(f"Results saved to {results_file}")


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "openqa_results.json"
    if results_file.exists() and not args.overwrite:
        print(f"Results already exist at {results_file}, skipping.")
        return

    if args.parallel_tasks > 1 and len(args.tasks) > 1:
        _run_parallel_tasks(args, output_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"Target model: {args.target_model}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Tasks: {args.tasks}")
    print(f"Conditions: {args.conditions}")
    print(f"Reasoning mode: {args.reasoning_mode}")
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
            "reader_mode": reader_mode_result_metadata(wrapper),
        }

    all_results = {}
    task_scalars = {condition: [] for condition in args.conditions}

    for task_name in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task_name}")
        print(f"{'=' * 60}")

        if task_name == "triviaqa":
            examples, dataset_meta = load_triviaqa_examples(args.triviaqa_config)
        else:
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
                    reasoning_mode=args.reasoning_mode,
                )
                print(
                    f"  {condition}: EM={metrics['em']*100:.2f} "
                    f"F1={metrics['f1']*100:.2f}"
                )

            metrics["elapsed_s"] = time.time() - t0
            mode_metadata = state["reader_mode"]
            metrics.update(mode_metadata)
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
        "triviaqa_config": args.triviaqa_config,
        "reasoning_mode": args.reasoning_mode,
        "adaptor_checkpoint": args.adaptor_checkpoint,
        "dual_reader_mode": args.dual_reader_mode,
        "reader_mode_resolution": {
            condition: state["reader_mode"]
            for condition, state in condition_state.items()
        },
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
