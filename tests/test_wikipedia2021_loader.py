import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.data import Wikipedia2021Dataset


class _DummySourceTokenizer:
    def __init__(self):
        self.id_to_word = {
            1: "alpha",
            2: "beta",
            3: "gamma",
            4: "delta",
        }

    def decode(self, input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(self.id_to_word[token_id] for token_id in input_ids)


class _DummyTargetTokenizer:
    name_or_path = "dummy-target"

    def __call__(self, text, add_special_tokens=False):
        vocab = {
            "alpha": 10,
            "beta": 11,
            "gamma": 12,
            "delta": 13,
        }
        return {"input_ids": [vocab[word] for word in text.split()]}


class _NamedTokenizer:
    def __init__(self, name_or_path):
        self.name_or_path = name_or_path


def test_tokenizer_match_is_based_on_repo_id():
    tokenizer = _NamedTokenizer("mistralai/Mistral-7B-v0.3")
    assert Wikipedia2021Dataset._tokenizer_matches_source(
        tokenizer,
        "mistralai/Mistral-7B-v0.3",
    )


def test_tokenizer_match_recognizes_huggingface_snapshot_cache_path():
    tokenizer = _NamedTokenizer(
        "/scratch/cache/huggingface/hub/"
        "models--mistralai--Mistral-7B-v0.3/snapshots/deadbeef"
    )
    assert Wikipedia2021Dataset._tokenizer_matches_source(
        tokenizer,
        "mistralai/Mistral-7B-v0.3",
    )


def test_chunk_token_ids_drops_incomplete_tail():
    chunks = Wikipedia2021Dataset._chunk_token_ids([1, 2, 3, 4, 5], seq_len=2)
    assert chunks == [[1, 2], [3, 4]]


def test_retokenize_window_uses_target_tokenizer_and_rechunks():
    chunks = Wikipedia2021Dataset._retokenize_window(
        input_ids=[1, 2, 3, 4],
        seq_len=2,
        source_tokenizer=_DummySourceTokenizer(),
        target_tokenizer=_DummyTargetTokenizer(),
    )
    assert chunks == [[10, 11], [12, 13]]


def test_extract_token_stream_prefers_non_padding_labels():
    ids = Wikipedia2021Dataset._extract_token_stream(
        {
            "input_ids": [101, 102, 103, 104],
            "labels": [-100, -100, 103, 104],
        }
    )
    assert ids == [103, 104]


def test_read_preprocessing_config_returns_saved_tokenizer_path(tmp_path):
    dataset_dir = tmp_path / "wiki2021-llama"
    dataset_dir.mkdir()
    with open(dataset_dir / "preprocessing_config.json", "w") as f:
        json.dump({"tokenizer_path": "NousResearch/Llama-2-7b-hf"}, f)

    config = Wikipedia2021Dataset._read_preprocessing_config(dataset_dir)
    assert config == {"tokenizer_path": "NousResearch/Llama-2-7b-hf"}


def test_require_tokenizer_match_rejects_mismatched_tokenizer():
    tokenizer = _NamedTokenizer("NousResearch/Llama-2-7b-hf")
    with pytest.raises(ValueError, match="tokenizer alignment is required"):
        Wikipedia2021Dataset(
            split="train",
            tokenizer=tokenizer,
            seq_len=2,
            max_tokens=4,
            dataset_name="dummy/wiki2021",
            source_tokenizer_name="mistralai/Mistral-7B-v0.3",
            require_tokenizer_match=True,
        )
