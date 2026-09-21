from engram.data import CORPUS_REGISTRY, GENERAL_MIXTURE


def test_general_mixture_has_requested_corpora_and_configs():
    assert [item[0] for item in GENERAL_MIXTURE] == [
        "wikitext",
        "amazon_reviews",
        "cc_news",
        "imdb",
    ]
    assert GENERAL_MIXTURE[0][1:] == (
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
    )


def test_requested_training_corpora_are_registered():
    assert "general-mixed" in CORPUS_REGISTRY
    assert "nemotron-cc-code" in CORPUS_REGISTRY


def test_nemotron_code_uses_the_available_data_config(monkeypatch):
    import sys
    import types

    calls = {}

    def fake_load_dataset(dataset_id, **kwargs):
        calls["dataset_id"] = dataset_id
        calls["kwargs"] = kwargs
        return [{"text": "synthetic code sample"}]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    from engram.data import NemotronCCCodeDataset

    class TinyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2]}

    dataset = NemotronCCCodeDataset(
        split="train",
        tokenizer=TinyTokenizer(),
        seq_len=2,
        max_tokens=2,
    )

    assert len(dataset) == 1
    assert calls["dataset_id"] == "nvidia/Nemotron-CC-Code-v1"
    assert calls["kwargs"]["name"] == "data"
    assert calls["kwargs"]["split"] == "train"
    assert calls["kwargs"]["streaming"] is True
    assert calls["kwargs"]["token"] is True
