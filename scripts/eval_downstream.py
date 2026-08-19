"""Zero-shot downstream task evaluation for Engram memory transfer.

Evaluates baseline (no memory) vs transferred memory on standard NLP benchmarks
using log-likelihood scoring (same protocol as GPT-3 / lm-evaluation-harness):
  - HellaSwag (4-choice commonsense reasoning)
  - PIQA (2-choice physical intuition)
  - ARC-Easy (4-choice science QA)
  - WinoGrande (2-choice coreference resolution)
  - LAMBADA (last-word prediction accuracy)
  - BoolQ (2-choice yes/no reading comprehension)

Usage:
    uv run python scripts/eval_downstream.py \
        --target-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adaptor-dir results/cross_arch2/transferred_seed42 \
        --source-memory results/qwen35_source_memory/memory.pt \
        --memory-config results/qwen35_source_memory/memory_config.json \
        --tasks hellaswag piqa arc_easy winogrande lambada boolq \
        --output-dir results/downstream/tinyllama_seed42
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.memory import EngramMemory, MemoryConfig
from engram.adaptor import build_adaptor
from engram.backbone_wrapper import BackboneWrapper
from engram.canonicalization import build_canonicalizer, WordBoundaryCanonicalizer
from engram.hashing import HashConfig, WordNgramHasher
from engram.metrics import bootstrap_ci


# These values describe the legacy transfer setup.  At evaluation time they
# are overridden by the saved training config, then by explicit CLI arguments.
RUNTIME_CONFIG_DEFAULTS = {
    "condition": "transferred",
    "architecture": "legacy",
    "reader_type": "cross_attention",
    "generator_cue_source": "engram",
    "generator_num_latents": 4,
    "generator_hidden_size": 256,
    "generator_layers": 2,
    "generator_heads": 4,
    "generator_cue_window": 3,
    "generator_fusion_type": "generated_only",
    "memory_dim": None,
    "injection_layers": None,
    "adaptor_branches": 1,
    "canon_mode": "vocab",
    "source_memory": "results/source_memory/memory.pt",
    "memory_config": "results/source_memory/memory_config.json",
}


# ── Task Loaders ─────────────────────────────────────────────────────


def load_hellaswag():
    """HellaSwag: 4-choice commonsense sentence completion."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)

    examples = []
    for ex in ds:
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])
        examples.append({
            "context": ctx,
            "choices": endings,
            "label": label,
        })
    return examples


def load_piqa():
    """PIQA: 2-choice physical intuition QA."""
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", revision="refs/convert/parquet",
                      split="validation", trust_remote_code=True)

    examples = []
    for ex in ds:
        goal = ex["goal"]
        choices = [ex["sol1"], ex["sol2"]]
        label = int(ex["label"])
        examples.append({
            "context": goal,
            "choices": choices,
            "label": label,
        })
    return examples


def load_arc_easy():
    """ARC-Easy: multi-choice science QA."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test",
                      trust_remote_code=True)

    examples = []
    for ex in ds:
        question = ex["question"]
        choices_text = ex["choices"]["text"]
        choices_labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]

        # Find correct answer index
        label = choices_labels.index(answer_key) if answer_key in choices_labels else 0

        examples.append({
            "context": question,
            "choices": choices_text,
            "label": label,
        })
    return examples


def load_winogrande():
    """WinoGrande: 2-choice coreference resolution."""
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl",
                      split="validation", trust_remote_code=True)

    examples = []
    for ex in ds:
        sentence = ex["sentence"]
        option1 = ex["option1"]
        option2 = ex["option2"]
        label = int(ex["answer"]) - 1  # 1-indexed → 0-indexed

        # Replace _ with each option to form full sentences
        choice1 = sentence.replace("_", option1)
        choice2 = sentence.replace("_", option2)

        examples.append({
            "context": "",
            "choices": [choice1, choice2],
            "label": label,
        })
    return examples


def load_lambada():
    """LAMBADA: last-word prediction accuracy."""
    from datasets import load_dataset
    ds = load_dataset("lambada", split="test")

    examples = []
    for ex in ds:
        text = ex["text"]
        words = text.rsplit(" ", 1)
        if len(words) == 2:
            context = words[0]
            last_word = words[1]
            examples.append({
                "context": context,
                "last_word": last_word,
                "full_text": text,
            })
    return examples


def load_boolq():
    """BoolQ: yes/no reading comprehension."""
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation", trust_remote_code=True)

    examples = []
    for ex in ds:
        passage = ex["passage"]
        question = ex["question"]
        label = 0 if ex["answer"] else 1  # 0 = yes, 1 = no

        context = f"{passage}\nQuestion: {question}\nAnswer:"
        examples.append({
            "context": context,
            "choices": [" yes", " no"],
            "label": label,
        })
    return examples


# ── Knowledge-Intensive Task Loaders ─────────────────────────────


def load_rte():
    """RTE (SuperGLUE): 2-class textual entailment."""
    from datasets import load_dataset
    ds = load_dataset("super_glue", "rte", split="validation", trust_remote_code=True)

    examples = []
    for ex in ds:
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        label = ex["label"]  # 0=entailment, 1=not_entailment

        context = f"{premise}\nQuestion: {hypothesis} True or False?\nAnswer:"
        examples.append({
            "context": context,
            "choices": [" True", " False"],
            "label": label,
        })
    return examples


def load_openbookqa():
    """OpenBookQA: 4-choice science QA requiring fact retrieval."""
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split="test",
                      trust_remote_code=True)

    examples = []
    for ex in ds:
        question = ex["question_stem"]
        choices_text = ex["choices"]["text"]
        choices_labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]
        label = choices_labels.index(answer_key) if answer_key in choices_labels else 0

        examples.append({
            "context": question,
            "choices": choices_text,
            "label": label,
        })
    return examples


def load_arc_challenge():
    """ARC-Challenge: 4-choice science QA (harder subset of ARC)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test",
                      trust_remote_code=True)

    examples = []
    for ex in ds:
        question = ex["question"]
        choices_text = ex["choices"]["text"]
        choices_labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]
        label = choices_labels.index(answer_key) if answer_key in choices_labels else 0

        examples.append({
            "context": question,
            "choices": choices_text,
            "label": label,
        })
    return examples


def load_sciq():
    """SciQ: 4-choice science QA with supporting passage."""
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="test", trust_remote_code=True)

    examples = []
    for ex in ds:
        support = ex["support"].strip()
        question = ex["question"]
        correct = ex["correct_answer"]
        distractors = [ex["distractor1"], ex["distractor2"], ex["distractor3"]]

        # Correct answer at index 0; position doesn't affect log-likelihood scoring
        choices = [correct] + distractors
        context = f"{support}\nQuestion: {question}" if support else question

        examples.append({
            "context": context,
            "choices": choices,
            "label": 0,
        })
    return examples


def load_truthfulqa():
    """TruthfulQA MC1: factual correctness (variable # choices, 1 correct)."""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation", trust_remote_code=True)

    examples = []
    for ex in ds:
        question = ex["question"]
        mc1_targets = ex["mc1_targets"]
        choices = mc1_targets["choices"]
        labels = mc1_targets["labels"]  # list of 0/1, exactly one 1
        correct_idx = labels.index(1)

        examples.append({
            "context": f"Question: {question}\nAnswer:",
            "choices": choices,
            "label": correct_idx,
        })
    return examples


def load_race():
    """RACE-High: 4-choice reading comprehension from English exams."""
    from datasets import load_dataset
    ds = load_dataset("ehovy/race", "high", split="test", trust_remote_code=True)

    examples = []
    for ex in ds:
        article = ex["article"]
        question = ex["question"]
        options = ex["options"]
        answer = ex["answer"]  # 'A', 'B', 'C', or 'D'
        label = ord(answer) - ord("A")

        context = f"{article}\nQuestion: {question}"
        examples.append({
            "context": context,
            "choices": options,
            "label": label,
        })
    return examples


TASK_LOADERS = {
    "hellaswag": load_hellaswag,
    "piqa": load_piqa,
    "arc_easy": load_arc_easy,
    "winogrande": load_winogrande,
    "lambada": load_lambada,
    "boolq": load_boolq,
    # Knowledge-intensive tasks
    "rte": load_rte,
    "openbookqa": load_openbookqa,
    "arc_challenge": load_arc_challenge,
    "sciq": load_sciq,
    "truthfulqa": load_truthfulqa,
    "race": load_race,
}


# ── Evaluation Logic ─────────────────────────────────────────────────


def compute_choice_logprob(
    wrapper: BackboneWrapper,
    tokenizer,
    context: str,
    choice: str,
    device: torch.device,
    set_canon_fn=None,
) -> float:
    """Compute average log-probability of choice tokens given context.

    Tokenizes context + choice, runs forward pass, and computes the mean
    log-prob of only the choice tokens (not the context).
    """
    # Tokenize context and full sequence separately to find boundary
    ctx_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_text = context + choice
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"]

    ctx_len = ctx_ids.shape[1]
    full_len = full_ids.shape[1]

    # If choice tokenizes to 0 additional tokens, skip
    if full_len <= ctx_len:
        return float("-inf")

    # When context is empty (ctx_len=0), score from token index 1 onward.
    # Python negative indexing makes logits[0, -1:N, :] empty — guard here.
    score_start = max(1, ctx_len)
    logit_start = score_start - 1  # = max(0, ctx_len - 1)

    if full_len <= score_start:
        return float("-inf")

    full_ids = full_ids.to(device)

    # Set canonicalization context
    if set_canon_fn is not None:
        set_canon_fn(full_ids)

    with torch.no_grad():
        outputs = wrapper(input_ids=full_ids)
        logits = outputs.logits  # (1, T, V)

    # Compute log-probs for choice tokens only
    # logits[t] predicts token[t+1]; for choice starting at score_start,
    # use logits[logit_start : full_len-1]
    choice_logits = logits[0, logit_start : full_len - 1, :]  # (n_choice_tokens, V)
    choice_targets = full_ids[0, score_start : full_len]       # (n_choice_tokens,)

    log_probs = F.log_softmax(choice_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, choice_targets.unsqueeze(1)).squeeze(1)

    # Length-normalized average log-probability
    return float(token_log_probs.mean().item())


def eval_multiple_choice(
    wrapper: BackboneWrapper,
    tokenizer,
    examples: list[dict],
    device: torch.device,
    set_canon_fn=None,
) -> dict:
    """Evaluate multiple-choice task by log-likelihood scoring."""
    correct = 0
    total = 0

    for ex in tqdm(examples, desc="Evaluating", leave=False):
        context = ex["context"]
        choices = ex["choices"]
        label = ex["label"]

        # Score each choice
        scores = []
        for choice in choices:
            # Prepend space to choice for proper tokenization
            choice_text = choice if choice.startswith(" ") else f" {choice}"
            score = compute_choice_logprob(
                wrapper, tokenizer, context, choice_text, device, set_canon_fn,
            )
            scores.append(score)

        predicted = int(np.argmax(scores))
        if predicted == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def eval_lambada(
    wrapper: BackboneWrapper,
    tokenizer,
    examples: list[dict],
    device: torch.device,
    set_canon_fn=None,
) -> dict:
    """Evaluate LAMBADA: predict the last word of each passage."""
    correct = 0
    total = 0

    for ex in tqdm(examples, desc="Evaluating LAMBADA", leave=False):
        context = ex["context"]
        last_word = ex["last_word"]

        # Tokenize context
        ctx_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ctx_ids = ctx_ids.to(device)

        if set_canon_fn is not None:
            set_canon_fn(ctx_ids)

        with torch.no_grad():
            outputs = wrapper(input_ids=ctx_ids)
            logits = outputs.logits  # (1, T, V)

        # Predict next token (greedy)
        next_token_logits = logits[0, -1, :]
        predicted_token_id = int(next_token_logits.argmax().item())
        predicted_text = tokenizer.decode([predicted_token_id]).strip()

        # Compare (case-insensitive, strip whitespace)
        target_text = last_word.strip()
        if predicted_text.lower() == target_text.lower():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


# ── Model Setup ──────────────────────────────────────────────────────


def parse_injection_layers(raw) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    raw = str(raw).strip()
    if not raw:
        return None
    return [int(part.strip()) for part in re.split(r"[\s,:;]+", raw) if part.strip()]


def setup_wrapper(
    model_name: str,
    memory: Optional[EngramMemory],
    condition: str,
    adaptor_path: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    injection_layers: Optional[list] = None,
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
    """Create and configure a BackboneWrapper."""
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

    if adaptor_path and Path(adaptor_path).exists() and wrapper.adaptor is not None:
        state_dict = torch.load(adaptor_path, map_location="cpu", weights_only=True)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList):
            # A multi-layer checkpoint contains 0.*, 1.*, ... keys.  Older
            # one-adaptor checkpoints are intentionally replicated here.
            if any(key.split(".", 1)[0].isdigit() for key in state_dict):
                wrapper.adaptor.load_state_dict(state_dict)
            else:
                for adaptor in wrapper.adaptor:
                    adaptor.load_state_dict(state_dict)
        else:
            wrapper.adaptor.load_state_dict(state_dict)
        wrapper.adaptor.to(device)

    wrapper.eval()
    return wrapper


def build_canon_fn(wrapper, mem_cfg_dict, canon_mode, device):
    """Build canonicalization function for memory lookup."""
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


# ── Main ─────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot downstream task evaluation")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--adaptor-dir", type=str, required=True,
                        help="Directory containing adaptor.pt or adaptor_best.pt")
    # Runtime-recoverable settings default to None so a saved training config
    # is not accidentally overwritten by parser defaults.
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--source-memory", type=str, default=None)
    parser.add_argument("--memory-config", type=str, default=None)
    parser.add_argument("--tasks", nargs="+",
                        default=["hellaswag", "piqa", "arc_easy", "winogrande", "lambada", "boolq"],
                        choices=list(TASK_LOADERS.keys()))
    parser.add_argument("--canon-mode", type=str, default=None,
                        choices=["vocab", "word_boundary"])
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit examples per task (for quick testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--injection-layers", type=str, default=None,
                        help="Comma-separated layer indices (e.g. '2,10'). Must match training.")
    parser.add_argument("--adaptor-branches", type=int, default=None,
                        help="Number of adaptor branches. Must match training.")
    parser.add_argument("--architecture", choices=["legacy", "generative"], default=None)
    parser.add_argument("--reader-type", choices=["cross_attention", "mean"], default=None)
    parser.add_argument("--generator-cue-source", choices=["engram", "learned"], default=None)
    parser.add_argument("--generator-num-latents", type=int, default=None)
    parser.add_argument("--generator-hidden-size", type=int, default=None)
    parser.add_argument("--generator-layers", type=int, default=None)
    parser.add_argument("--generator-heads", type=int, default=None)
    parser.add_argument("--generator-cue-window", type=int, default=None)
    parser.add_argument("--generator-fusion-type", choices=["generated_only", "engram_residual", "dual_reader"], default=None)
    parser.add_argument("--memory-dim", type=int, default=None)
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> dict:
    """Recover architecture details saved next to the trained adaptor."""
    runtime_config = dict(RUNTIME_CONFIG_DEFAULTS)
    saved_path = Path(args.adaptor_dir) / "config.json"
    if saved_path.exists():
        with open(saved_path) as f:
            saved_config = json.load(f)
        runtime_config.update({
            key: value for key, value in saved_config.items()
            if key in RUNTIME_CONFIG_DEFAULTS and value is not None
        })
    for key in RUNTIME_CONFIG_DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            runtime_config[key] = value
    return runtime_config


def main():
    args = parse_args()
    runtime_config = resolve_runtime_config(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            dtype = torch.bfloat16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "downstream_results.json"
    if results_file.exists():
        print(f"Results already exist at {results_file}, skipping.")
        return

    print(f"Target model: {args.target_model}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Tasks: {args.tasks}")
    print(f"Runtime config: {runtime_config}")

    learned_generative = (
        runtime_config["architecture"] == "generative"
        and runtime_config["generator_cue_source"] == "learned"
    )
    mem_cfg_dict = None
    mem_cfg = None
    if not learned_generative:
        with open(runtime_config["memory_config"]) as f:
            mem_cfg_dict = json.load(f)

        mem_cfg = MemoryConfig(
            max_ngram=mem_cfg_dict["max_ngram"],
            heads_per_order=mem_cfg_dict["heads_per_order"],
            table_size=mem_cfg_dict["table_size"],
            d_head=mem_cfg_dict["d_head"],
            hash_seed=mem_cfg_dict["hash_seed"],
        )
        memory_dim = mem_cfg.d_mem
    else:
        if runtime_config["memory_dim"] is None:
            raise ValueError(
                "Saved/CLI memory_dim is required for generative learned cues"
            )
        memory_dim = int(runtime_config["memory_dim"])

    injection_layers = parse_injection_layers(runtime_config["injection_layers"])
    wrapper_kwargs = dict(
        injection_layers=injection_layers,
        adaptor_branches=int(runtime_config["adaptor_branches"]),
        memory_dim=memory_dim,
        architecture=runtime_config["architecture"],
        reader_type=runtime_config["reader_type"],
        generator_cue_source=runtime_config["generator_cue_source"],
        generator_num_latents=int(runtime_config["generator_num_latents"]),
        generator_hidden_size=int(runtime_config["generator_hidden_size"]),
        generator_layers=int(runtime_config["generator_layers"]),
        generator_heads=int(runtime_config["generator_heads"]),
        generator_cue_window=int(runtime_config["generator_cue_window"]),
        generator_fusion_type=runtime_config["generator_fusion_type"],
    )
    treatment_name = "generative_learned" if learned_generative else runtime_config["condition"]

    # Find adaptor checkpoint
    adaptor_dir = Path(args.adaptor_dir)
    adaptor_path = adaptor_dir / "adaptor_best.pt"
    if not adaptor_path.exists():
        adaptor_path = adaptor_dir / "adaptor.pt"
    if not adaptor_path.exists():
        raise FileNotFoundError(f"No adaptor checkpoint found in {adaptor_dir}")
    print(f"Adaptor: {adaptor_path}")

    # Tokenizer for task formatting
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for task_name in args.tasks:
        print(f"\n{'='*60}")
        print(f"  Task: {task_name}")
        print(f"{'='*60}")

        # Load task data
        print(f"Loading {task_name}...")
        examples = TASK_LOADERS[task_name]()
        if args.max_examples:
            random.shuffle(examples)
            examples = examples[:args.max_examples]
        print(f"  {len(examples)} examples")

        task_results = {}
        eval_fn = eval_lambada if task_name == "lambada" else eval_multiple_choice

        # ── Baseline (no memory) ──
        print(f"\n  [1/2] Baseline (no memory)...")
        t0 = time.time()
        baseline_wrapper = setup_wrapper(
            args.target_model, None, "baseline", None, device, dtype,
            **wrapper_kwargs,
        )

        baseline_res = eval_fn(
            baseline_wrapper, tokenizer, examples, device, set_canon_fn=None,
        )
        baseline_res["elapsed_s"] = time.time() - t0
        task_results["baseline"] = baseline_res
        print(f"    Accuracy: {baseline_res['accuracy']:.4f} "
              f"({baseline_res['correct']}/{baseline_res['total']})")

        # Clean up baseline wrapper
        baseline_wrapper.cleanup()
        del baseline_wrapper
        torch.cuda.empty_cache()

        # ── Memory treatment ──
        print(f"\n  [2/2] {treatment_name}...")
        memory = None
        if not learned_generative:
            memory = EngramMemory(mem_cfg)
            memory.load_state_dict(
                torch.load(
                    runtime_config["source_memory"],
                    map_location="cpu",
                    weights_only=True,
                )
            )
            for p in memory.parameters():
                p.requires_grad = False

        t0 = time.time()
        treatment_wrapper = setup_wrapper(
            args.target_model, memory, runtime_config["condition"],
            str(adaptor_path), device, dtype,
            **wrapper_kwargs,
        )

        set_canon_fn = None
        if memory is not None:
            set_canon_fn = build_canon_fn(
                treatment_wrapper, mem_cfg_dict, runtime_config["canon_mode"], device,
            )

        treatment_res = eval_fn(
            treatment_wrapper, tokenizer, examples, device, set_canon_fn=set_canon_fn,
        )
        treatment_res["elapsed_s"] = time.time() - t0
        task_results[treatment_name] = treatment_res
        print(f"    Accuracy: {treatment_res['accuracy']:.4f} "
              f"({treatment_res['correct']}/{treatment_res['total']})")

        # Compute delta
        delta_acc = treatment_res["accuracy"] - baseline_res["accuracy"]
        task_results["delta_accuracy"] = delta_acc
        task_results["delta_accuracy_pct"] = delta_acc * 100
        print(f"\n  Delta: {delta_acc:+.4f} ({delta_acc*100:+.2f}%)")

        all_results[task_name] = task_results

        # Cleanup
        treatment_wrapper.cleanup()
        del treatment_wrapper, memory
        torch.cuda.empty_cache()

    # Save results
    final = {
        "target_model": args.target_model,
        "adaptor_dir": args.adaptor_dir,
        "source_memory": runtime_config["source_memory"],
        "seed": args.seed,
        "canon_mode": runtime_config["canon_mode"],
        "treatment": treatment_name,
        "runtime_config": runtime_config,
        "tasks": all_results,
    }
    with open(results_file, "w") as f:
        json.dump(final, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("  Downstream Task Evaluation Summary")
    print(f"{'='*60}")
    print(f"{'Task':<14} {'Baseline':>10} {treatment_name:>20} {'Delta':>8}")
    print("-" * 54)
    for task_name, res in all_results.items():
        bl = res["baseline"]["accuracy"]
        tr = res[treatment_name]["accuracy"]
        d = res["delta_accuracy_pct"]
        print(f"{task_name:<14} {bl:>10.4f} {tr:>20.4f} {d:>+7.2f}%")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
