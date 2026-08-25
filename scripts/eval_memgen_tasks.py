"""Evaluate vanilla or ATHENA memory-augmented models on MemGen benchmarks.

The script uses official public benchmark splits where labels are available and
records a common ``acc``, ``em`` and ``f1`` schema.  ``acc`` is always the
task-facing correctness metric.  For multiple-choice and numeric-answer tasks,
EM is exact task correctness and F1 is a lexical diagnostic.  Code benchmarks
also save generated samples for isolated functional evaluation; lexical EM/F1
against the canonical solution are explicitly marked as diagnostics.

ALFWorld is handled by ``scripts/eval_alfworld_memory.py`` because it requires
an interactive environment rather than a static Hugging Face dataset.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_openqa import (
    dedupe_answers,
    exact_match,
    f1_score,
    get_model_max_context,
    greedy_generate,
    load_triviaqa_examples,
    setup_condition,
)


STATIC_TASKS = (
    "triviaqa",
    "popqa",
    "kodcode",
    "bigcodebench",
    "gpqa",
    "gsm8k",
    "math",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--condition",
        choices=["baseline", "transferred"],
        default="transferred",
        help="baseline loads only the target backbone; transferred loads ATHENA memory readers",
    )
    parser.add_argument("--adaptor-dir", default=None)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", default=None)
    parser.add_argument("--memory-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=STATIC_TASKS, default=list(STATIC_TASKS))
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--dual-reader-mode", choices=["both", "engram_only", "generated_only"], default="both")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset into each task's official split; useful for full-data sharding.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default="chat")
    parser.add_argument(
        "--reasoning-mode",
        choices=["vanilla", "cot"],
        default="vanilla",
        help="Prompt condition; vanilla is the unchanged backbone and cot requests explicit reasoning.",
    )
    parser.add_argument(
        "--code-num-samples",
        type=int,
        default=1,
        help="Number of code solutions sampled per problem; use 10 for Pass@1/5/10.",
    )
    parser.add_argument(
        "--code-temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for code solutions when --code-num-samples > 1.",
    )
    parser.add_argument(
        "--code-top-p",
        type=float,
        default=0.95,
        help="Nucleus threshold for sampled code solutions.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return [value]
        return _json_list(parsed)
    return [str(value)]


def load_popqa_examples():
    from datasets import load_dataset

    dataset = load_dataset("akariasai/PopQA", split="test")
    examples = []
    for row in dataset:
        answers = dedupe_answers(_json_list(row.get("possible_answers")) + [row.get("obj", "")])
        examples.append({"question": row["question"], "answers": answers})
    return examples, {"dataset_name": "akariasai/PopQA", "split": "test"}


def load_gpqa_examples(seed: int):
    from datasets import load_dataset

    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        dataset_meta = {
            "dataset_name": "Idavidrein/gpqa",
            "config_name": "gpqa_diamond",
            "split": "train",
        }
    except Exception as exc:
        # The canonical repository is gated. A batch job without an injected
        # HF_TOKEN must still be able to evaluate the public benchmark without
        # persisting a credential in a Slurm script or log. Reconstruct the
        # exact 198-row Diamond subset by joining a pinned public membership
        # list against a pinned public copy of the original 448 GPQA records.
        message = str(exc).lower()
        if not any(marker in message for marker in ("gated", "authenticated", "access")):
            raise
        raw_revision = "8c23b9cbb7871e81172ef3b36d8903fa3a9b84a1"
        membership_revision = "68be7564497676e07a77a042fdb587deb88c51c3"
        raw_dataset = load_dataset(
            "ankner/gpqa", split="train", revision=raw_revision
        )
        membership_dataset = load_dataset(
            "fingertap/GPQA-Diamond", split="test", revision=membership_revision
        )
        dataset = _select_gpqa_diamond_rows(raw_dataset, membership_dataset)
        dataset_meta = {
            "dataset_name": "Idavidrein/gpqa",
            "config_name": "gpqa_diamond",
            "split": "train",
            "source": "public_mirror_reconstruction",
            "raw_mirror": "ankner/gpqa",
            "raw_revision": raw_revision,
            "membership_mirror": "fingertap/GPQA-Diamond",
            "membership_revision": membership_revision,
        }
    examples = []
    rng = random.Random(seed)
    for row in dataset:
        correct = row["Correct Answer"]
        choices = [correct] + [row[f"Incorrect Answer {i}"] for i in range(1, 4)]
        rng.shuffle(choices)
        examples.append({
            "question": row["Question"],
            "choices": choices,
            "label": choices.index(correct),
            "answer": correct,
            "paper_solution": rf"\boxed{{{'ABCD'[choices.index(correct)]}}}",
        })
    if len(examples) != 198:
        raise RuntimeError(f"Expected 198 GPQA Diamond rows, found {len(examples)}")
    return examples, dataset_meta


def _select_gpqa_diamond_rows(raw_dataset, membership_dataset):
    """Recover canonical GPQA rows selected by a public Diamond membership list.

    The membership mirror appends a shuffled A-D choice block to each original
    question. Matching the unchanged original question prefix lets us retain
    the canonical answer strings and perform our own deterministic shuffle.
    """
    raw_rows = list(raw_dataset)
    selected = []
    seen_record_ids = set()
    for member in membership_dataset:
        member_question = str(member["question"]).strip()
        matches = []
        for row in raw_rows:
            raw_question = str(row["Question"]).strip()
            if member_question == raw_question or member_question.startswith(raw_question + "\n"):
                matches.append(row)
        if len(matches) != 1:
            raise RuntimeError(
                "Expected one raw GPQA match for Diamond member, "
                f"found {len(matches)}: {member_question[:120]!r}"
            )
        row = matches[0]
        record_id = row.get("Record ID", row["Question"])
        if record_id in seen_record_ids:
            raise RuntimeError(f"Duplicate GPQA Diamond record: {record_id}")
        seen_record_ids.add(record_id)
        selected.append(row)
    if len(selected) != 198:
        raise RuntimeError(f"Expected 198 GPQA Diamond members, found {len(selected)}")
    return selected


def load_gsm8k_examples():
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for row in dataset:
        answer = row["answer"].rsplit("####", 1)[-1].strip()
        examples.append({
            "question": row["question"],
            "answers": [answer],
            "paper_solution": rf"\boxed{{{answer}}}",
        })
    return examples, {"dataset_name": "openai/gsm8k", "config_name": "main", "split": "test"}


def _last_boxed(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text.strip().splitlines()[-1] if text.strip() else ""
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
    return "".join(chars).strip()


def load_math_examples():
    from datasets import load_dataset

    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    examples = [{
        "question": row["problem"],
        "answers": [_last_boxed(row["solution"])],
        "paper_solution": row["solution"],
    } for row in dataset]
    return examples, {"dataset_name": "DigitalLearningGmbH/MATH-lighteval", "split": "test"}


def load_kodcode_examples(seed: int):
    from datasets import load_dataset

    dataset = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    split = dataset.train_test_split(test_size=0.2, seed=seed, shuffle=True)
    examples = []
    for row in split["test"]:
        examples.append({
            "task_id": row["question_id"],
            "question": row["question"],
            "reference": row["solution"],
            "tests": row["test"],
            "test_info": row["test_info"],
        })
    return examples, {
        "dataset_name": "KodCode/KodCode-Light-RL-10K",
        "source_split": "train",
        "evaluation_split": "seeded_20pct_test",
        "seed": seed,
    }


def load_bigcodebench_examples():
    from datasets import load_dataset

    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    examples = []
    for row in dataset:
        examples.append({
            "task_id": row["task_id"],
            "question": row["instruct_prompt"],
            "code_prompt": row["code_prompt"],
            "reference": row["canonical_solution"],
            "tests": row["test"],
            "entry_point": row["entry_point"],
            "libs": row["libs"],
        })
    return examples, {"dataset_name": "bigcode/bigcodebench", "split": "v0.1.4", "mode": "instruct"}


def build_prompt(tokenizer, instruction: str, prompt_style: str) -> str:
    if prompt_style == "chat" and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return instruction.rstrip() + "\nAnswer:"


def _best_text_metrics(prediction: str, answers: list[str]) -> tuple[float, float]:
    em = float(any(exact_match(prediction, answer) for answer in answers))
    f1 = max((f1_score(prediction, answer)[0] for answer in answers), default=0.0)
    return em, f1


def _paper_alias_accuracy(prediction: str, answers: list[str]) -> float:
    """MemGen TriviaQA/PopQA correctness: case-insensitive alias containment."""
    prediction = prediction.lower()
    return float(any(str(answer).lower() in prediction for answer in answers))


def _boxed_string(text: str, *, first: bool) -> str | None:
    positions = [pos for marker in ("\\boxed", "\\fbox") if (pos := text.find(marker)) >= 0]
    if not positions:
        return None
    start = min(positions) if first else max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if text.startswith("\\boxed ", start):
        return "\\boxed " + text[start + len("\\boxed "):].split("$")[0]
    index = start
    depth = 0
    saw_left = False
    while index < len(text):
        if text[index] == "{":
            depth += 1
            saw_left = True
        elif text[index] == "}":
            depth -= 1
            if saw_left and depth == 0:
                return text[start:index + 1]
        index += 1
    return None


def _remove_boxed(text: str) -> str:
    if text.startswith("\\boxed "):
        return text[len("\\boxed "):]
    for marker in ("\\boxed{", "\\fbox{"):
        if text.startswith(marker) and text.endswith("}"):
            return text[len(marker):-1]
    raise ValueError(f"Not a boxed expression: {text!r}")


def _fix_math_frac(text: str) -> str:
    pieces = text.split("\\frac")
    rebuilt = pieces[0]
    for piece in pieces[1:]:
        rebuilt += "\\frac"
        if not piece or piece[0] == "{":
            rebuilt += piece
        elif len(piece) >= 2:
            first, second, rest = piece[0], piece[1], piece[2:]
            rebuilt += "{" + first + "}"
            rebuilt += second + rest if second == "{" else "{" + second + "}" + rest
        else:
            return text
    return rebuilt


def _paper_strip_math_string(text: str) -> str:
    """Normalization used by MemGen's published math reward."""
    text = text.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    text = text.replace("tfrac", "frac").replace("dfrac", "frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\$", "").replace("\\%", "").replace("%", "")
    if "\\text{ " in text:
        text = text.split("\\text{ ", 1)[0]
    text = text.replace(" .", " 0.").replace("{.", "{0.")
    if text.startswith("."):
        text = "0" + text
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=", 1)[1]
    if "\\sqrt" in text:
        pieces = text.split("\\sqrt")
        rebuilt = pieces[0]
        for piece in pieces[1:]:
            rebuilt += "\\sqrt" + (piece if piece.startswith("{") else "{" + piece[:1] + "}" + piece[1:])
        text = rebuilt
    text = _fix_math_frac(text.replace(" ", ""))
    if text == "0.5":
        text = "\\frac{1}{2}"
    parts = text.split("/")
    if len(parts) == 2:
        try:
            numerator, denominator = int(parts[0]), int(parts[1])
            if text == f"{numerator}/{denominator}":
                text = f"\\frac{{{numerator}}}{{{denominator}}}"
        except ValueError:
            pass
    return text


def _paper_math_accuracy(completion: str, ground_truth: str) -> float:
    """Match MemGen: first boxed prediction vs last boxed reference."""
    try:
        predicted = _boxed_string(completion, first=True)
        expected = _boxed_string(ground_truth, first=False)
        if predicted is None or expected is None:
            return 0.0
        return float(
            _paper_strip_math_string(_remove_boxed(predicted))
            == _paper_strip_math_string(_remove_boxed(expected))
        )
    except (AssertionError, IndexError, ValueError):
        return 0.0


def _extract_short_answer(task: str, text: str, reasoning_mode: str = "vanilla") -> str:
    text = (text or "").strip()
    # A decoder can legally return an empty completion (for example after an
    # immediate EOS).  Keep the benchmark loop alive and let the metric
    # functions score it as incorrect instead of indexing an empty list.
    if not text:
        return ""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_matches:
        text = answer_matches[-1].strip()
        # The model can emit an empty answer tag even when the surrounding
        # completion is non-empty.  Treat it exactly like an empty completion.
        if not text:
            return ""
    boxed = _last_boxed(text)
    if task == "math" and boxed:
        return boxed
    if task == "gsm8k":
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", boxed or text)
        if numbers:
            return numbers[-1].replace(",", "")
    if reasoning_mode == "cot" and task in {"triviaqa", "popqa"}:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return re.sub(r"^(?:final answer|answer)\s*:\s*", "", lines[-1], flags=re.I).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _build_instruction(task: str, example: dict, reasoning_mode: str) -> str:
    if task in {"triviaqa", "popqa"}:
        if reasoning_mode == "cot":
            return (
                "Solve the question step by step, then put only the concise final answer "
                "inside <answer></answer>.\n"
                f"Question: {example['question']}"
            )
        return f"Answer the following question concisely.\nQuestion: {example['question']}"
    if reasoning_mode == "cot":
        prefix = "Solve the problem step by step and put the final answer in \\boxed{}."
    else:
        prefix = "Give the final answer in \\boxed{} without unnecessary text."
    if task in {"gsm8k", "math"}:
        return f"{prefix}\nProblem: {example['question']}"
    if task == "gpqa":
        labels = "ABCD"
        choices_text = "\n".join(f"{labels[i]}. {choice}" for i, choice in enumerate(example["choices"]))
        return f"{prefix}\nQuestion: {example['question']}\n{choices_text}"
    if task in {"kodcode", "bigcodebench"}:
        return (
            ("Reason carefully before writing the solution. " if reasoning_mode == "cot" else "")
            + "Write a correct Python solution. Return only Python code without Markdown fences.\n"
            + example["question"]
        )
    raise ValueError(task)


def evaluate_generation_task(task, examples, wrapper, set_canon_fn, device, args):
    tokenizer = wrapper.tokenizer
    max_context = get_model_max_context(wrapper, args.max_context_length)
    default_tokens = {"triviaqa": 32, "popqa": 32, "gsm8k": 256, "math": 384}[task]
    max_new_tokens = args.max_new_tokens or default_tokens
    acc_values, em_values, f1_values, samples = [], [], [], []
    task_started_at = time.monotonic()
    for example_index, example in enumerate(examples, 1):
        if example_index == 1 or example_index % 10 == 0:
            print(
                f"TASK_ITEM_START {task} {example_index}/{len(examples)} "
                f"elapsed_s={time.monotonic() - task_started_at:.1f}",
                flush=True,
            )
        instruction = _build_instruction(task, example, args.reasoning_mode)
        prompt = build_prompt(tokenizer, instruction, args.prompt_style)
        raw = greedy_generate(
            wrapper, tokenizer, prompt, device, set_canon_fn,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context,
            official_tokenization=True,
            stop_at_newline=False,
        )
        prediction = _extract_short_answer(task, raw, args.reasoning_mode)
        em, f1 = _best_text_metrics(prediction, example["answers"])
        if task in {"triviaqa", "popqa"}:
            acc = _paper_alias_accuracy(prediction, example["answers"])
        else:
            acc = _paper_math_accuracy(raw, example["paper_solution"])
        acc_values.append(acc)
        em_values.append(em)
        f1_values.append(f1)
        if len(samples) < 10:
            samples.append({"question": example["question"], "prediction": prediction, "raw": raw, "answers": example["answers"]})
        if example_index % 100 == 0 or example_index == len(examples):
            print(f"TASK_PROGRESS {task} {example_index}/{len(examples)}", flush=True)
    acc = float(np.mean(acc_values)) if acc_values else 0.0
    em = float(np.mean(em_values)) if em_values else 0.0
    return {
        "acc": acc,
        "em": em,
        "f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "metric_note": (
            "acc uses case-insensitive answer-alias containment"
            if task in {"triviaqa", "popqa"}
            else "acc uses first-boxed prediction vs last-boxed reference normalization"
        ),
        "total": len(examples),
        "samples": samples,
    }


def evaluate_gpqa(examples, wrapper, set_canon_fn, device, args):
    tokenizer = wrapper.tokenizer
    max_context = get_model_max_context(wrapper, args.max_context_length)
    correct, f1_values, samples = 0, [], []
    labels = "ABCD"
    task_started_at = time.monotonic()
    for example_index, example in enumerate(examples, 1):
        if example_index == 1 or example_index % 10 == 0:
            print(
                f"TASK_ITEM_START gpqa {example_index}/{len(examples)} "
                f"elapsed_s={time.monotonic() - task_started_at:.1f}",
                flush=True,
            )
        instruction = _build_instruction("gpqa", example, args.reasoning_mode)
        prompt = build_prompt(tokenizer, instruction, args.prompt_style)
        raw = greedy_generate(
            wrapper, tokenizer, prompt, device, set_canon_fn,
            max_new_tokens=args.max_new_tokens or 384,
            max_context_length=max_context,
            official_tokenization=True,
            stop_at_newline=False,
        )
        boxed = _boxed_string(raw, first=True)
        predicted_label = _remove_boxed(boxed).strip().upper() if boxed else ""
        prediction_index = labels.find(predicted_label)
        prediction = example["choices"][prediction_index] if prediction_index >= 0 else predicted_label
        is_correct = bool(_paper_math_accuracy(raw, example["paper_solution"]))
        correct += int(is_correct)
        f1_values.append(f1_score(prediction, example["answer"])[0])
        if len(samples) < 10:
            samples.append({"question": example["question"], "prediction": prediction, "raw": raw, "answer": example["answer"]})
        if example_index % 25 == 0 or example_index == len(examples):
            print(f"TASK_PROGRESS gpqa {example_index}/{len(examples)}", flush=True)
    acc = correct / len(examples) if examples else 0.0
    return {
        "acc": acc,
        "em": acc,
        "f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "metric_note": "acc uses boxed-choice correctness on GPQA Diamond",
        "total": len(examples),
        "samples": samples,
    }


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def evaluate_code_generation(task, examples, wrapper, set_canon_fn, device, args, output_dir: Path):
    tokenizer = wrapper.tokenizer
    max_context = get_model_max_context(wrapper, args.max_context_length)
    max_new_tokens = args.max_new_tokens or 768
    num_samples = int(args.code_num_samples)
    if num_samples < 1:
        raise ValueError("--code-num-samples must be at least 1")
    if num_samples > 1 and args.code_temperature <= 0:
        raise ValueError("--code-temperature must be positive when sampling multiple code solutions")
    lexical_em, lexical_f1, records = [], [], []
    task_started_at = time.monotonic()
    for example_index, example in enumerate(examples, 1):
        if example_index == 1 or example_index % 10 == 0:
            print(
                f"TASK_ITEM_START {task} {example_index}/{len(examples)} "
                f"elapsed_s={time.monotonic() - task_started_at:.1f}",
                flush=True,
            )
        instruction = _build_instruction(task, example, args.reasoning_mode)
        prompt = build_prompt(tokenizer, instruction, args.prompt_style)
        raw_generations = []
        predictions = []
        for _ in range(num_samples):
            raw = greedy_generate(
                wrapper, tokenizer, prompt, device, set_canon_fn,
                max_new_tokens=max_new_tokens,
                max_context_length=max_context,
                official_tokenization=True,
                stop_at_newline=False,
                temperature=(args.code_temperature if num_samples > 1 else 0.0),
                top_p=(args.code_top_p if num_samples > 1 else 1.0),
            )
            raw_generations.append(raw)
            predictions.append(_strip_code_fence(raw))
        prediction = predictions[0]
        reference = example["reference"]
        em, f1 = _best_text_metrics(prediction, [reference])
        lexical_em.append(em)
        lexical_f1.append(f1)
        record = dict(example)
        record.update({
            "prediction": prediction,
            "raw_generation": raw_generations[0],
            "predictions": predictions,
            "raw_generations": raw_generations,
            "num_samples": num_samples,
        })
        if task == "bigcodebench":
            # Official BigCodeBench local evaluation accepts one full
            # self-contained solution per task under the ``solution`` key.
            record["solution"] = prediction
        records.append(record)
        if example_index % 50 == 0 or example_index == len(examples):
            print(f"TASK_PROGRESS {task} {example_index}/{len(examples)}", flush=True)
    samples_path = output_dir / f"{task}_samples.jsonl"
    with open(samples_path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    # Functional accuracy is intentionally computed in a separate isolated
    # execution step.  A missing value must never be mistaken for zero pass@1.
    return {
        "acc": None,
        "em": float(np.mean(lexical_em)) if lexical_em else 0.0,
        "f1": float(np.mean(lexical_f1)) if lexical_f1 else 0.0,
        "metric_note": "EM/F1 are lexical diagnostics; acc awaits isolated functional execution",
        "total": len(examples),
        "num_samples": num_samples,
        "pass_k": [k for k in (1, 5, 10) if num_samples >= k],
        "samples_file": str(samples_path),
    }


def load_task(task: str, seed: int):
    if task == "triviaqa":
        return load_triviaqa_examples()
    if task == "popqa":
        return load_popqa_examples()
    if task == "gpqa":
        return load_gpqa_examples(seed)
    if task == "gsm8k":
        return load_gsm8k_examples()
    if task == "math":
        return load_math_examples()
    if task == "kodcode":
        return load_kodcode_examples(seed)
    if task == "bigcodebench":
        return load_bigcodebench_examples()
    raise ValueError(task)


def main():
    args = parse_args()
    if args.code_num_samples < 1:
        raise ValueError("--code-num-samples must be at least 1")
    if not 0 < args.code_top_p <= 1:
        raise ValueError("--code-top-p must be in (0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {result_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}; dtype: {dtype}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    wrapper, set_canon_fn = setup_condition(args, args.condition, device, dtype)
    results = {}
    for task in args.tasks:
        examples, dataset_meta = load_task(task, args.seed)
        full_count = len(examples)
        if args.start_index < 0 or args.start_index > full_count:
            raise ValueError(
                f"--start-index {args.start_index} is outside {task} split of {full_count} examples"
            )
        shard_start = args.start_index
        examples = examples[shard_start:]
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        shard_end = shard_start + len(examples)
        print(
            f"TASK_START {task} examples={len(examples)}/{full_count} "
            f"slice=[{shard_start}:{shard_end}] source={dataset_meta}"
        )
        started = time.time()
        if task == "gpqa":
            metrics = evaluate_gpqa(examples, wrapper, set_canon_fn, device, args)
        elif task in {"kodcode", "bigcodebench"}:
            metrics = evaluate_code_generation(task, examples, wrapper, set_canon_fn, device, args, output_dir)
        else:
            metrics = evaluate_generation_task(task, examples, wrapper, set_canon_fn, device, args)
        metrics["elapsed_s"] = time.time() - started
        results[task] = {
            "dataset": dataset_meta,
            "full_count": full_count,
            "evaluated_count": len(examples),
            "slice_start": shard_start,
            "slice_end": shard_end,
            "metrics": metrics,
        }
        print(f"TASK_COMPLETE {task} {json.dumps(metrics, default=str)}")

    final = {
        "target_model": args.target_model,
        "condition": args.condition,
        "architecture": (
            "vanilla_backbone"
            if args.condition == "baseline"
            else "generated_memory+engram+dual_reader"
        ),
        "dual_reader_mode": (
            None if args.condition == "baseline" else args.dual_reader_mode
        ),
        "reasoning_mode": args.reasoning_mode,
        "tasks": results,
        "completed": True,
    }
    with open(result_path, "w") as handle:
        json.dump(final, handle, indent=2)
    wrapper.cleanup()
    print("STATIC_TASK_EVAL_COMPLETE")


if __name__ == "__main__":
    main()
