#!/usr/bin/env python3
"""Download the official Search-R1 HNSW64 index and wiki-18 corpus.

The index is downloaded in two Hugging Face shards and appended into the
single FAISS file expected by the local retriever. Completed shards are
removed after they are appended so the host does not need two full copies of
the index at once.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


INDEX_REPO = "PeterJinGo/wiki-18-e5-index-HNSW64"
CORPUS_REPO = "PeterJinGo/wiki-18-corpus"
CORPUS_MEMBER = "data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
INDEX_SHARDS = ("part_aa", "part_ab")
FIRST_SHARD_BYTES = 42_949_672_960


def download(repo_id: str, filename: str, data_dir: Path) -> Path:
    token = os.environ.get("HF_TOKEN") or None
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir,
            token=token,
        )
    )


def append_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("ab") as dst:
        shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)


def download_index(data_dir: Path) -> None:
    target = data_dir / "e5_HNSW64.index"
    marker = data_dir / ".e5_HNSW64.index.complete"
    if marker.is_file() and target.is_file():
        print(f"INDEX_ALREADY_COMPLETE {target} {target.stat().st_size}", flush=True)
        return
    if target.exists():
        second_shard = data_dir / "part_ab"
        if not second_shard.is_file():
            raise RuntimeError(
                f"Refusing to resume incomplete index {target}; part_ab is missing"
            )
        offset = target.stat().st_size - FIRST_SHARD_BYTES
        if offset < 0 or offset > second_shard.stat().st_size:
            raise RuntimeError(
                f"Unexpected partial index size={target.stat().st_size}; "
                f"expected at least {FIRST_SHARD_BYTES}"
            )
        print(f"INDEX_RESUME part_ab offset={offset}", flush=True)
        with second_shard.open("rb") as src, target.open("ab") as dst:
            src.seek(offset)
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        second_shard.unlink()
        marker.write_text("complete\n")
        print(f"INDEX_COMPLETE {target} {target.stat().st_size}", flush=True)
        return

    for shard in INDEX_SHARDS:
        shard_path = download(INDEX_REPO, shard, data_dir)
        print(f"INDEX_SHARD_DOWNLOADED {shard} {shard_path.stat().st_size}", flush=True)
        append_file(shard_path, target)
        shard_path.unlink()
        print(f"INDEX_SHARD_CONSUMED {shard} total={target.stat().st_size}", flush=True)
    marker.write_text("complete\n")
    print(f"INDEX_COMPLETE {target} {target.stat().st_size}", flush=True)


def download_corpus(data_dir: Path) -> None:
    target = data_dir / "wiki-18.jsonl"
    marker = data_dir / ".wiki-18.jsonl.complete"
    if marker.is_file() and target.is_file():
        print(f"CORPUS_ALREADY_COMPLETE {target} {target.stat().st_size}", flush=True)
        return
    if target.exists():
        raise RuntimeError(
            f"Refusing to overwrite incomplete corpus {target}; remove it only after inspection"
        )

    compressed = download(CORPUS_REPO, "wiki-18.jsonl.gz", data_dir)
    print(f"CORPUS_COMPRESSED_DOWNLOADED {compressed.stat().st_size}", flush=True)
    with tarfile.open(compressed, mode="r:gz") as archive:
        member = archive.getmember(CORPUS_MEMBER)
        src = archive.extractfile(member)
        if src is None:
            raise RuntimeError(f"Could not extract corpus member {CORPUS_MEMBER}")
        with src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    compressed.unlink()
    marker.write_text("complete\n")
    print(f"CORPUS_COMPLETE {target} {target.stat().st_size}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    download_index(args.data_dir)
    download_corpus(args.data_dir)


if __name__ == "__main__":
    main()
