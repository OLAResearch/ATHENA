import json
from argparse import Namespace

import torch

import scripts.eval_openqa as eval_openqa
from scripts.eval_memgen_tasks import (
    _extract_short_answer,
    _json_list,
    _last_boxed,
    _strip_code_fence,
    _paper_alias_accuracy,
    _paper_math_accuracy,
    _select_gpqa_diamond_rows,
)


def test_baseline_setup_does_not_read_memory_or_adaptor(monkeypatch):
    sentinel = object()

    def fake_setup_wrapper(model_name, **kwargs):
        assert model_name == "tiny-backbone"
        assert kwargs["memory"] is None
        assert kwargs["condition"] == "baseline"
        assert kwargs["adaptor_path"] is None
        return sentinel

    monkeypatch.setattr(eval_openqa, "setup_wrapper", fake_setup_wrapper)
    args = Namespace(
        target_model="tiny-backbone",
        adaptor_dir=None,
        source_memory=None,
        memory_config=None,
    )

    wrapper, set_canon_fn = eval_openqa.setup_condition(
        args, "baseline", torch.device("cpu"), torch.float32
    )

    assert wrapper is sentinel
    assert set_canon_fn is None


def test_json_list_accepts_popqa_serialized_aliases():
    value = json.dumps(["politician", "political leader"])
    assert _json_list(value) == ["politician", "political leader"]


def test_last_boxed_supports_nested_latex():
    assert _last_boxed(r"work therefore \\boxed{\\frac{1}{2}}") == r"\\frac{1}{2}"


def test_gsm8k_uses_last_number():
    text = "We compute 16 - 7 = 9, then 9 * 2 = 18."
    assert _extract_short_answer("gsm8k", text) == "18"


def test_code_fence_is_removed():
    assert _strip_code_fence("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"


def test_paper_openqa_accuracy_uses_alias_containment_not_strict_em():
    assert _paper_alias_accuracy("The answer is Mount Kilimanjaro.", ["Kilimanjaro"]) == 1.0
    assert _paper_alias_accuracy("Mount Kenya", ["Kilimanjaro"]) == 0.0


def test_paper_math_accuracy_uses_first_prediction_and_last_reference_box():
    prediction = r"work \boxed{1/2} later \boxed{9}"
    reference = r"intermediate \boxed{7}, final \boxed{\frac{1}{2}}"
    assert _paper_math_accuracy(prediction, reference) == 1.0


def test_gpqa_public_membership_reconstructs_exact_raw_rows():
    raw_rows = []
    members = []
    for index in range(198):
        question = f"Question {index}?"
        raw_rows.append({
            "Question": question,
            "Correct Answer": "correct",
            "Incorrect Answer 1": "wrong 1",
            "Incorrect Answer 2": "wrong 2",
            "Incorrect Answer 3": "wrong 3",
            "Record ID": f"record-{index}",
        })
        members.append({
            "question": question + "\n\nA. wrong 1\nB. correct\nC. wrong 2\nD. wrong 3",
            "answer": "B",
        })
    raw_rows.append({
        "Question": "Not in Diamond?",
        "Correct Answer": "unused",
        "Incorrect Answer 1": "unused 1",
        "Incorrect Answer 2": "unused 2",
        "Incorrect Answer 3": "unused 3",
        "Record ID": "not-diamond",
    })

    selected = _select_gpqa_diamond_rows(raw_rows, members)

    assert len(selected) == 198
    assert [row["Record ID"] for row in selected] == [
        f"record-{index}" for index in range(198)
    ]
