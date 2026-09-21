"""Helpers for loading Hugging Face artifacts from an offline cache."""

import os
from pathlib import Path


_TRUE_VALUES = {"1", "on", "true", "yes"}


def _find_cached_snapshot(name_or_path: str) -> str | None:
    """Find a complete local Hub snapshot without consulting Hub metadata.

    ``snapshot_download`` normally handles this lookup, but can miss a valid
    snapshot when its constants were initialized before the container cache
    environment was propagated.  A concrete snapshot path is safe for both
    Transformers and the offline resolver and avoids that initialization race.
    """
    if "/" not in name_or_path:
        return None
    cache_root = os.environ.get("HF_HUB_CACHE") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE"
    )
    if not cache_root:
        return None
    repo_dir = Path(cache_root) / f"models--{name_or_path.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in candidates:
        has_config = (snapshot / "config.json").is_file()
        has_model = any(
            path.is_file()
            for path in (
                snapshot / "model.safetensors.index.json",
                snapshot / "pytorch_model.bin.index.json",
            )
        ) or any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin"))
        has_tokenizer = any(
            (snapshot / filename).is_file()
            for filename in (
                "tokenizer.json",
                "tokenizer.model",
                "spiece.model",
                "vocab.json",
            )
        )
        if has_config and has_model and has_tokenizer:
            return str(snapshot)
    return None


def _offline_mode_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in _TRUE_VALUES
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def resolve_pretrained_source(name_or_path: str) -> str:
    """Prefer a complete local snapshot and otherwise retain an online ID.

    Recent Transformers tokenizer loading may call the Hub API for a model ID
    even when all files are cached and ``local_files_only`` is requested.  A
    concrete snapshot directory is unambiguously local and avoids that call.
    """
    expanded = Path(name_or_path).expanduser()
    if expanded.exists():
        return str(expanded)
    cached_snapshot = _find_cached_snapshot(name_or_path)
    if cached_snapshot is not None:
        return cached_snapshot
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=name_or_path, local_files_only=True)
    except Exception as exc:
        cached_snapshot = _find_cached_snapshot(name_or_path)
        if cached_snapshot is not None:
            return cached_snapshot
        if not _offline_mode_enabled():
            return name_or_path
        raise FileNotFoundError(
            f"Hugging Face offline mode is enabled, but {name_or_path!r} "
            "is not available as a complete local snapshot"
        ) from exc
