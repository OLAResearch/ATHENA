from scripts.eval_general_nlp_halueval import (
    _hallucination_label,
    _normalise_choice,
    evaluate_task,
    load_cb,
)


def test_halueval_labels_are_binary_and_explicit():
    assert _hallucination_label("yes") == 1
    assert _hallucination_label("no") == 0
    assert _normalise_choice("positive") == " positive"
    assert _normalise_choice(" yes") == " yes"


def test_cb_loader_can_be_replaced_by_a_small_protocol_fixture(monkeypatch):
    class FakeDataset(list):
        pass

    def fake_load_dataset(name, *args, split, **kwargs):
        assert name == "super_glue"
        assert args == ("cb",)
        assert split == "validation"
        return FakeDataset([
            {"premise": "A", "hypothesis": "B", "label": 2},
        ])

    monkeypatch.setattr(
        "scripts.eval_general_nlp_halueval._load_dataset", fake_load_dataset
    )
    examples = load_cb()
    assert examples[0]["choices"] == ["entailment", "contradiction", "neutral"]
    assert examples[0]["label"] == 2


def test_evaluate_task_uses_pmi_protocol_for_general_fixture():
    class Tokenizer:
        pass

    class Wrapper:
        tokenizer = Tokenizer()

    examples = [
        {
            "context": "x",
            "domain_context": "d",
            "choices": ["a", "b"],
            "label": 1,
        }
    ]

    calls = []

    def fake_batch(wrapper, tokenizer, contexts, choices, device, set_canon_fn, max_context_length):
        calls.append((contexts, choices))
        # conditional: b wins; domain: a has the larger prior, so PMI still b
        return [0.1, 0.8] if contexts[0] == "x" else [0.7, 0.2]

    monkey = __import__("pytest").MonkeyPatch()
    monkey.setattr("scripts.eval_general_nlp_halueval._batch_choice_logprobs", fake_batch)
    try:
        result = evaluate_task(
            Wrapper(), None, examples, device="cpu", max_context_length=128,
            pmi=True, batch_size=1,
        )
    finally:
        monkey.undo()
    assert result["accuracy"] == 1.0
    assert result["protocol"] == "domain_conditional_pmi"
    assert len(calls) == 2
