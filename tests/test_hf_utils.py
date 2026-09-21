from pathlib import Path

import huggingface_hub
import pytest

from engram.hf_utils import resolve_pretrained_source


def test_resolve_pretrained_source_preserves_existing_local_path(tmp_path):
    assert resolve_pretrained_source(str(tmp_path)) == str(tmp_path)


def test_resolve_pretrained_source_preserves_model_id_online(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cache miss")),
    )
    assert resolve_pretrained_source("org/model") == "org/model"


def test_resolve_pretrained_source_prefers_cached_snapshot_online(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *, repo_id, local_files_only: str(snapshot),
    )

    assert resolve_pretrained_source("org/model") == str(snapshot)


def test_resolve_pretrained_source_returns_cached_snapshot_offline(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *, repo_id, local_files_only: str(snapshot),
    )

    assert resolve_pretrained_source("org/model") == str(snapshot)


def test_resolve_pretrained_source_finds_complete_hub_snapshot_directly(
    monkeypatch, tmp_path
):
    snapshot = (
        tmp_path
        / "models--org--model"
        / "snapshots"
        / "commit"
    )
    snapshot.mkdir(parents=True)
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / filename).write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cache metadata miss")),
    )

    assert resolve_pretrained_source("org/model") == str(snapshot)


def test_resolve_pretrained_source_fails_clearly_when_cache_missing(monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "true")

    def fail(**_kwargs):
        raise RuntimeError("cache miss")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(FileNotFoundError, match="complete local snapshot"):
        resolve_pretrained_source("org/missing")
