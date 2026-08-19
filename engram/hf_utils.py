"""Helpers for loading Hugging Face artifacts from an offline cache."""

import os
from pathlib import Path


_TRUE_VALUES = {"1", "on", "true", "yes"}


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
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=name_or_path, local_files_only=True)
    except Exception as exc:
        if not _offline_mode_enabled():
            return name_or_path
        raise FileNotFoundError(
            f"Hugging Face offline mode is enabled, but {name_or_path!r} "
            "is not available as a complete local snapshot"
        ) from exc
