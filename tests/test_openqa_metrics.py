from scripts.eval_openqa import (
    compute_truthfulqa_mc_scores,
    exact_match,
    f1_score,
    load_triviaqa_examples,
    normalize_answer,
    task_scalar_metric,
    task_scalar_score,
    use_official_tokenization,
)


def test_triviaqa_loader_is_pinned_to_full_validation_config(monkeypatch):
    calls = {}

    def fake_loader(*, candidates, split_candidates):
        calls["candidates"] = candidates
        calls["splits"] = split_candidates
        return ([{"question": "q", "answer": "answer"}], {"split": "validation"})

    monkeypatch.setattr("scripts.eval_openqa.load_dataset_with_fallback", fake_loader)
    examples, meta = load_triviaqa_examples()

    assert examples == [{"question": "q", "answers": ["answer"]}]
    assert calls == {
        "candidates": [("mandarjoshi/trivia_qa", "rc.nocontext")],
        "splits": ["validation"],
    }
    assert meta == {"split": "validation"}


def test_normalize_answer_removes_articles_punctuation_and_case():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_exact_match_uses_normalized_answers():
    assert exact_match("The Eiffel Tower", "eiffel tower!")


def test_f1_score_handles_partial_overlap():
    f1, precision, recall = f1_score("Barack Obama", "Obama")
    assert round(f1, 4) == round(2 / 3, 4)
    assert precision == 0.5
    assert recall == 1.0


def test_truthfulqa_mc_scores():
    scores = compute_truthfulqa_mc_scores(
        scores_true=[-0.1, -0.3],
        scores_false=[-1.0, -1.5],
        ref_true=["A.", "B."],
        ref_best="A.",
    )
    assert scores["MC1"] == 1.0
    assert scores["MC3"] == 1.0
    assert 0.0 < scores["MC2"] < 1.0


def test_task_scalar_score_uses_task_specific_summary_metric():
    assert task_scalar_metric("nq") == "f1"
    assert task_scalar_score("nq", {"f1": 0.25}) == 25.0
    assert task_scalar_metric("webqa") == "f1"
    assert task_scalar_score("webqa", {"f1": 0.3}) == 30.0
    assert task_scalar_score("truthfulqa", {"mc_avg": 0.4}) == 40.0


def test_task_specific_tokenization_profile_matches_empirical_reproduction():
    assert use_official_tokenization("nq") is True
    assert use_official_tokenization("triviaqa") is True
    assert use_official_tokenization("hotpotqa") is True
    assert use_official_tokenization("webqa") is False
    assert use_official_tokenization("truthfulqa") is False
