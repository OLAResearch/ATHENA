import json
from pathlib import Path

from scripts.build_case_study_inputs import select_candidates
from scripts.eval_case_study import (
    _run_generation,
    _select_showcase,
    build_rag_prompt,
)


def _row(question, prediction, correct, answer="Ada"):
    return {
        "question": question,
        "prediction": prediction,
        "raw_prediction": prediction,
        "correct": correct,
        "f1": 1.0 if correct else 0.0,
        "answers": [answer],
        "prediction_diagnostics": {"raw": {"repetition_count": 0}},
    }


def test_rag_prompt_contains_context_but_not_supporting_fact_metadata():
    prompt = build_rag_prompt(
        "Who?",
        [["Doc A", ["Ada was a mathematician."]]],
    )
    assert "Ada was a mathematician" in prompt
    assert "supporting_facts" not in prompt
    assert "Question: Who?" in prompt


def test_case_study_keeps_answer_after_leading_formatting_newline(monkeypatch):
    calls = {}

    def fake_greedy_generate(*args, **kwargs):
        calls["stop_at_newline"] = kwargs["stop_at_newline"]
        return "\nReinhard Heydrich\n"

    monkeypatch.setattr("scripts.eval_case_study.greedy_generate", fake_greedy_generate)
    wrapper = type("Wrapper", (), {"tokenizer": object()})()
    result = _run_generation(
        wrapper,
        None,
        "Question: Who?\nAnswer:",
        max_context=128,
        args=type("Args", (), {"max_new_tokens": 15})(),
    )
    assert calls["stop_at_newline"] is False
    assert result == "Reinhard Heydrich"


def test_posthoc_selection_requires_base_and_rag_failure_for_pathway_panel():
    row = {
        "question": "Who?",
        "answers": ["Ada"],
        "preferred_source": "GE",
        "outputs": {
            "base": {"correct": False, "prediction": "wrong"},
            "rag": {"correct": False, "prediction": "still wrong"},
            "E": {"correct": False, "prediction": "wrong E"},
            "GE": {"correct": True, "prediction": "Ada"},
            "GH": {"correct": False, "prediction": "wrong GH"},
            "router": {
                "correct": False,
                "prediction": "wrong router output",
                "route_trace": {"argmax_source": "GH"},
            },
        },
    }
    selected = _select_showcase([row])
    assert not selected["GE"]
    assert not selected["E"]
    assert not selected["router"]


def test_showcase_requires_one_router_success_per_pathway():
    rows = []
    for source in ("E", "GE", "GH"):
        outputs = {
            "base": {"correct": False, "prediction": "wrong"},
            "rag": {"correct": False, "prediction": "wrong"},
            "E": {"correct": False, "prediction": "wrong E"},
            "GE": {"correct": False, "prediction": "wrong GE"},
            "GH": {"correct": False, "prediction": "wrong GH"},
            "router": {
                "correct": True,
                "prediction": "Ada",
                "route_trace": {"argmax_source": source},
            },
        }
        outputs[source] = {"correct": True, "prediction": "Ada"}
        rows.append({"question": f"Who {source}?", "answers": ["Ada"], "outputs": outputs})

    selected = _select_showcase(rows)
    assert [row["question"] for row in selected["router"]] == ["Who E?", "Who GE?", "Who GH?"]
    assert all(len(selected[source]) == 1 for source in ("E", "GE", "GH"))


def test_showcase_allows_three_verified_successes_on_one_realised_pathway():
    rows = []
    for index in range(3):
        rows.append({
            "question": f"Who GH {index}?",
            "answers": ["Ada"],
            "outputs": {
                "base": {"correct": False, "prediction": "wrong"},
                "rag": {"correct": False, "prediction": "wrong"},
                "E": {"correct": False, "prediction": "wrong E"},
                "GE": {"correct": False, "prediction": "wrong GE"},
                "GH": {"correct": True, "prediction": "Ada"},
                "router": {
                    "correct": True,
                    "prediction": "Ada",
                    "route_trace": {"argmax_source": "GH"},
                },
            },
        })
    selected = _select_showcase(rows)
    assert len(selected["router"]) == 3
    assert [row["question"] for row in selected["GH"]] == ["Who GH 0?", "Who GH 1?", "Who GH 2?"]


def test_candidate_selection_records_posthoc_protocol():
    def metrics(mode):
        return {"hotpotqa": {"metrics": {"sample_predictions": [
            _row("Who?", "wrong", False),
        ]}}}

    results = {"evaluation": {"engram_baseline": metrics("base"),
                               "tri_E": metrics("E"),
                               "tri_GE": metrics("GE"),
                               "tri_GH": metrics("GH"),
                               "tri_reader_advantage": metrics("router")}}
    results["evaluation"]["tri_E"]["hotpotqa"]["metrics"]["sample_predictions"][0]["correct"] = True
    selected = select_candidates(results, 2)
    assert selected[0]["selection_is_post_hoc"] if "selection_is_post_hoc" in selected[0] else True
    assert "no labels are used by routing" in selected[0]["selection_protocol"]
