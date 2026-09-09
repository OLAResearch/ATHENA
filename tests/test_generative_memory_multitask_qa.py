import sys

import pytest

from scripts.train_generative_memory_multitask_qa import (
    AnswerOnlyCollator,
    EVAL_TASKS,
    TRAIN_TASKS,
    configure_generated_branch,
    compare_reader_results,
    extract_training_example,
    parse_args,
)

import torch.nn as nn


class TinyTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + [ord(ch) + 2 for ch in text]}


class TinyWrapper:
    def __init__(self, adaptor):
        self.backbone = nn.Linear(2, 2)
        self.memory = None
        self.adaptor = adaptor

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def freeze_memory(self):
        return None


def test_joint_task_sets_include_available_training_splits():
    assert TRAIN_TASKS == (
        "nq", "webqa", "triviaqa", "hotpotqa", "gsm8k", "math", "kodcode"
    )
    assert EVAL_TASKS == ("nq", "webqa", "triviaqa", "truthfulqa", "hotpotqa")


def test_multitask_defaults_to_residual_reader_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_generative_memory_multitask_qa.py",
            "--adaptor-dir",
            "adaptor",
            "--source-memory",
            "memory.pt",
            "--memory-config",
            "memory.json",
            "--output-dir",
            "output",
        ],
    )
    args = parse_args()
    assert args.dual_reader_mode == "both"
    assert args.train_tasks == list(TRAIN_TASKS)


def test_multitask_configurator_selects_residual_or_legacy_mode():
    from engram.generative_memory import GenerativeMemoryAdaptor

    adaptor = GenerativeMemoryAdaptor(8, 4, hidden_size=8, num_heads=2, fusion_type="dual_reader")
    wrapper = TinyWrapper(adaptor)
    configure_generated_branch(wrapper, "both")
    assert adaptor.dual_reader_mode == "both"
    assert not wrapper.backbone.weight.requires_grad

    configure_generated_branch(wrapper, "generated_only")
    assert adaptor.dual_reader_mode == "generated_only"

    with pytest.raises(ValueError, match="eval_openqa.py"):
        configure_generated_branch(wrapper, "engram_only")


def test_extract_training_examples_for_all_four_schemas():
    assert extract_training_example("nq", {"question": "Q", "answer": ["A"]})["answer"] == "A"
    assert extract_training_example("webqa", {"question": "Q", "answers": ["A", "B"]})["answers"] == ["A", "B"]
    trivia = {"question": "Q", "answer": {"value": "Best", "aliases": ["Alias"]}}
    assert extract_training_example("triviaqa", trivia)["answer"] == "Best"
    assert extract_training_example("hotpotqa", {"question": "Q", "answer": "yes"})["answer"] == "yes"
    assert extract_training_example("nq", {"question": "bad", "answer": [")"]}) is None


def test_extract_training_examples_for_math_and_code_schemas():
    gsm = extract_training_example(
        "gsm8k", {"question": "Q", "answer": "work\n#### 72"}
    )
    assert gsm["answer"] == r"\boxed{72}"
    assert "Problem: Q" in gsm["prompt"]

    math = extract_training_example(
        "math", {"problem": "P", "solution": r"reasoning, so \boxed{4}"}
    )
    assert math["answer"] == r"\boxed{4}"

    code = extract_training_example(
        "kodcode", {"question": "write f", "solution": "def f():\n    return 1"}
    )
    assert code["answer"].startswith("def f")
    assert "Python code" in code["prompt"]


def test_answer_only_collator_masks_prompt_and_pads():
    batch = AnswerOnlyCollator(TinyTokenizer(), max_length=256)([
        {"question": "Short?", "answer": "A"},
        {"question": "Longer question?", "answer": "Answer"},
    ])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert (batch["labels"] != -100).any(dim=1).all()
    assert (batch["labels"] == -100).any(dim=1).all()


def test_paired_reader_comparison_reports_direction_and_counts():
    def result(rows):
        return {"nq": {"metrics": {"sample_predictions": rows}}}

    baseline = result([
        {"question": "q1", "correct": False, "f1": 0.0},
        {"question": "q2", "correct": True, "f1": 1.0},
        {"question": "q3", "correct": False, "f1": 0.5},
    ])
    candidate = result([
        {"question": "q1", "correct": True, "f1": 1.0},
        {"question": "q2", "correct": False, "f1": 0.0},
        {"question": "q3", "correct": False, "f1": 0.75},
    ])
    comparison = compare_reader_results(baseline, candidate, seed=42)
    assert comparison["em_delta"] == 0.0
    assert comparison["f1_delta"] > 0
    assert comparison["em_improved"] == 1
    assert comparison["em_regressed"] == 1
