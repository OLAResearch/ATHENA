from scripts.pilot_cb_router import CB_CHOICES, load_cb_split


def test_cb_pilot_uses_three_fixed_choices():
    assert CB_CHOICES == ["entailment", "contradiction", "neutral"]


def test_cb_loader_preserves_split_and_label_contract(monkeypatch):
    seen = []

    def fake_load_dataset(name, *args, split, **kwargs):
        seen.append((name, args, split))
        return [{"premise": "A", "hypothesis": "B", "label": 2}]

    monkeypatch.setattr("scripts.pilot_cb_router._load_dataset", fake_load_dataset)
    examples = load_cb_split("train")
    assert seen == [("super_glue", ("cb",), "train")]
    assert examples[0]["choices"] == CB_CHOICES
    assert examples[0]["label"] == 2
    assert examples[0]["context"].startswith("Premise: A")
