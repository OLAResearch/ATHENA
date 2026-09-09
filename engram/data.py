"""Data loading for Engram memory training.

Supports:
  - WikiText-103 (default, for backward compatibility)
  - Wikipedia-2021 (pre-tokenized Dec 2021 dump, configurable tokenizer)
  - FineWeb-Edu (educational web text)
  - Nemotron-CC (high-quality web subsets)

Provides fixed-length token sequences with configurable token budgets.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader


def _parse_env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name}={value!r} is not a valid boolean. "
        "Use one of: 1/0, true/false, yes/no, on/off."
    )


class WikiTextDataset(Dataset):
    """WikiText-103 dataset that produces fixed-length token sequences.

    Concatenates all tokens into a single stream and slices into
    non-overlapping sequences of `seq_len` tokens.
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        seq_len: int = 512,
        max_tokens: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.split = split

        # Load dataset
        from datasets import load_dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)

        # Tokenize and concatenate all text
        all_ids = []
        for example in dataset:
            text = example["text"]
            if text.strip():
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                all_ids.extend(ids)

            if max_tokens and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

        # Store as single tensor
        n_tokens = len(all_ids)
        # Truncate to multiple of seq_len
        n_seqs = n_tokens // seq_len
        self.tokens = torch.tensor(all_ids[:n_seqs * seq_len], dtype=torch.long)
        self.tokens = self.tokens.view(n_seqs, seq_len)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.tokens[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class Wikipedia2021Dataset(Dataset):
    """Wikipedia-2021 dataset backed by a pre-tokenized corpus.

    The default source is the public Hugging Face dataset from the MLP Memory
    release, preprocessed with the Mistral tokenizer over the Dec 2021 English
    Wikipedia dump. The dataset exposes `train` and `test`; for adaptor/source
    training we map both validation and test to the published `test` split.

    Override the source dataset by setting `ENGRAM_WIKIPEDIA2021_DATASET` and
    `ENGRAM_WIKIPEDIA2021_SOURCE_TOKENIZER`. When the run tokenizer does not
    match the source tokenizer, each source window is decoded back to text and
    re-tokenized on the fly unless strict tokenizer matching is requested.
    """

    DEFAULT_DATASET = "Rubin-Wei/enwiki-dec2021-preprocessed-mistral"
    DEFAULT_SOURCE_TOKENIZER = "mistralai/Mistral-7B-v0.3"
    DEFAULT_LABEL_PADDING_INDEX = -100

    @staticmethod
    def _read_preprocessing_config(dataset_path: Path) -> Optional[dict]:
        config_path = dataset_path / "preprocessing_config.json"
        if not config_path.is_file():
            return None

        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def _tokenizer_matches_source(tokenizer, source_tokenizer_name: str) -> bool:
        def canonical_name(value: str) -> str:
            normalized = str(value or "").lower().rstrip("/")
            # Offline Hugging Face resolution changes ``name_or_path`` from a
            # repo id into e.g. ``.../models--org--repo/snapshots/<sha>``.
            # Recover the repo id so strict pre-tokenized-corpus alignment does
            # not reject the exact same tokenizer merely because it is cached.
            for part in Path(normalized).parts:
                if not part.startswith("models--"):
                    continue
                pieces = part[len("models--"):].split("--", 1)
                if len(pieces) == 2 and all(pieces):
                    return "/".join(pieces)
            return normalized

        name = getattr(tokenizer, "name_or_path", "") or ""
        return canonical_name(name) == canonical_name(source_tokenizer_name)

    @staticmethod
    def _chunk_token_ids(token_ids: list[int], seq_len: int) -> list[list[int]]:
        n_full_chunks = len(token_ids) // seq_len
        return [
            token_ids[chunk_idx * seq_len: (chunk_idx + 1) * seq_len]
            for chunk_idx in range(n_full_chunks)
        ]

    @classmethod
    def _retokenize_ids(
        cls,
        input_ids: list[int],
        source_tokenizer,
        target_tokenizer,
    ) -> list[int]:
        text = source_tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return target_tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

    @classmethod
    def _retokenize_window(
        cls,
        input_ids: list[int],
        seq_len: int,
        source_tokenizer,
        target_tokenizer,
    ) -> list[list[int]]:
        target_ids = cls._retokenize_ids(
            input_ids=input_ids,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        return cls._chunk_token_ids(target_ids, seq_len)

    @classmethod
    def _extract_token_stream(
        cls,
        example: dict,
        label_padding_index: int = DEFAULT_LABEL_PADDING_INDEX,
    ) -> list[int]:
        input_ids = example.get("input_ids")
        if input_ids is None:
            raise KeyError("Example does not expose an `input_ids` field.")

        labels = example.get("labels")
        if labels is not None:
            novel_tokens = [
                token_id
                for token_id, label in zip(input_ids, labels)
                if label != label_padding_index
            ]
            if novel_tokens:
                return novel_tokens

        return list(input_ids)

    def __init__(
        self,
        split: str,
        tokenizer,
        seq_len: int = 512,
        max_tokens: Optional[int] = None,
        dataset_name: Optional[str] = None,
        source_tokenizer_name: Optional[str] = None,
        require_tokenizer_match: Optional[bool] = None,
    ):
        self.seq_len = seq_len
        self.split = split

        dataset_name = (
            dataset_name
            if dataset_name is not None
            else os.environ.get(
                "ENGRAM_WIKIPEDIA2021_DATASET",
                self.DEFAULT_DATASET,
            )
        )
        source_tokenizer_name = (
            source_tokenizer_name
            if source_tokenizer_name is not None
            else os.environ.get("ENGRAM_WIKIPEDIA2021_SOURCE_TOKENIZER")
        )
        if require_tokenizer_match is None:
            require_tokenizer_match = _parse_env_bool(
                "ENGRAM_WIKIPEDIA2021_REQUIRE_TOKENIZER_MATCH",
                default=False,
            )
        dataset_split = "train" if split == "train" else "test"

        dataset_path = Path(dataset_name)
        preprocessing_config = None
        if dataset_path.exists():
            preprocessing_config = self._read_preprocessing_config(dataset_path)

        if source_tokenizer_name is None and preprocessing_config is not None:
            source_tokenizer_name = (
                preprocessing_config.get("tokenizer_path")
                or preprocessing_config.get("tokenizer_name_or_path")
            )
        if source_tokenizer_name is None:
            source_tokenizer_name = self.DEFAULT_SOURCE_TOKENIZER

        direct_ids_compatible = self._tokenizer_matches_source(
            tokenizer,
            source_tokenizer_name,
        )
        if require_tokenizer_match and not direct_ids_compatible:
            raise ValueError(
                "Wikipedia-2021 tokenizer alignment is required, but the "
                f"configured dataset {dataset_name!r} uses "
                f"{source_tokenizer_name!r} while the active tokenizer is "
                f"{getattr(tokenizer, 'name_or_path', '<unknown>')!r}. "
                "Provide a tokenizer-matched preprocessed dataset via "
                "--wikipedia2021-dataset/--wikipedia2021-source-tokenizer or "
                "the ENGRAM_WIKIPEDIA2021_DATASET / "
                "ENGRAM_WIKIPEDIA2021_SOURCE_TOKENIZER environment variables."
            )

        from datasets import DatasetDict
        from datasets import load_dataset
        from datasets import load_from_disk

        dataset = None
        selected_split = dataset_split
        if dataset_path.exists():
            try:
                local_dataset = load_from_disk(str(dataset_path))
                if isinstance(local_dataset, DatasetDict):
                    split_candidates = [dataset_split]
                    if dataset_split != "test":
                        split_candidates.append("test")
                    if dataset_split != "train":
                        split_candidates.append("train")
                    for candidate in split_candidates:
                        if candidate in local_dataset:
                            selected_split = candidate
                            dataset = local_dataset[candidate]
                            break
                    if dataset is None:
                        selected_split = next(iter(local_dataset.keys()))
                        dataset = local_dataset[selected_split]
                else:
                    dataset = local_dataset
                print(
                    f"  Wikipedia-2021: loaded tokenizer-aligned local dataset "
                    f"from {dataset_name} ({selected_split})"
                )
            except Exception:
                dataset = None

        if dataset is None:
            dataset = load_dataset(
                dataset_name,
                split=dataset_split,
                trust_remote_code=True,
            )
        source_tokenizer = None
        if not direct_ids_compatible:
            from transformers import AutoTokenizer

            from .hf_utils import resolve_pretrained_source

            source_tokenizer = AutoTokenizer.from_pretrained(
                resolve_pretrained_source(source_tokenizer_name)
            )
            print(
                "  Wikipedia-2021: source tokenizer mismatch detected; "
                f"re-tokenizing from {source_tokenizer_name} to "
                f"{getattr(tokenizer, 'name_or_path', '<unknown>')}"
            )

        all_ids = []
        n_examples = 0

        for example in dataset:
            ids = self._extract_token_stream(
                example,
                label_padding_index=self.DEFAULT_LABEL_PADDING_INDEX,
            )

            if not direct_ids_compatible:
                ids = self._retokenize_ids(
                    input_ids=ids,
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=tokenizer,
                )

            all_ids.extend(ids)
            n_examples += 1
            if n_examples % 10_000 == 0:
                print(
                    f"  Wikipedia-2021: {len(all_ids):,}"
                    + (
                        f"/{max_tokens:,}" if max_tokens is not None else ""
                    )
                    + f" tokens from {n_examples:,} windows"
                )

            if max_tokens and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

        print(
            f"  Wikipedia-2021 loaded: {len(all_ids):,} tokens "
            f"from {n_examples:,} windows ({dataset_name}:{selected_split})"
        )

        n_seqs = len(all_ids) // seq_len
        self.tokens = torch.tensor(all_ids[: n_seqs * seq_len], dtype=torch.long)
        self.tokens = self.tokens.view(n_seqs, seq_len)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.tokens[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class FineWebEduDataset(Dataset):
    """FineWeb-Edu dataset via streaming.

    Streams from HuggingFace's FineWeb-Edu (educationally filtered web text).
    Higher quality than SlimPajama for downstream NLP benchmarks (MMLU, ARC,
    HellaSwag, etc.) due to educational content filtering.
    Uses the sample-10BT subset (10B tokens) for efficient streaming.
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        seq_len: int = 512,
        max_tokens: int = 200_000_000,
        seed: int = 42,
    ):
        self.seq_len = seq_len

        from datasets import load_dataset

        # FineWeb-Edu only has a 'train' split; we carve out validation
        # by skipping the first `max_tokens` examples (different seed region)
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        if split == "train":
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        elif split == "validation":
            # Use a different seed region for validation data
            ds = ds.shuffle(seed=seed + 9999, buffer_size=10_000)

        # Tokenize and concatenate
        all_ids = []
        n_docs = 0
        for example in ds:
            text = example["text"]
            if text.strip():
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                all_ids.extend(ids)
                n_docs += 1

            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

            # Progress reporting every 10K docs
            if n_docs % 10_000 == 0 and n_docs > 0:
                print(f"  FineWeb-Edu: {len(all_ids):,}/{max_tokens:,} tokens "
                      f"({len(all_ids)/max_tokens*100:.1f}%) from {n_docs:,} docs")

        print(f"  FineWeb-Edu loaded: {len(all_ids):,} tokens from {n_docs:,} documents")

        # Store as single tensor
        n_seqs = len(all_ids) // seq_len
        self.tokens = torch.tensor(all_ids[:n_seqs * seq_len], dtype=torch.long)
        self.tokens = self.tokens.view(n_seqs, seq_len)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.tokens[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class NemotronCCDataset(Dataset):
    """Nemotron-CC v2.1 dataset via streaming.

    NVIDIA's high-quality Common Crawl dataset with explicit quality tiers
    and STEM Q&A content. Subsets are HuggingFace dataset configs:
      - "hq-dqa" -> "High-Quality-DQA": 8B tokens of STEM question-answer pairs
      - "hq"     -> "High-Quality": 26B tokens of highest-quality organic web
      - "mhq"    -> "Medium-High-Quality": 16.9B tokens
    """

    SUBSET_MAP = {
        "hq-dqa": "High-Quality-DQA",
        "hq": "High-Quality",
        "mhq": "Medium-High-Quality",
    }

    def __init__(
        self,
        split: str,
        tokenizer,
        seq_len: int = 512,
        max_tokens: int = 200_000_000,
        seed: int = 42,
        subset: str = "hq-dqa",
    ):
        self.seq_len = seq_len

        from datasets import load_dataset

        config_name = self.SUBSET_MAP.get(subset, subset)

        ds = load_dataset(
            "nvidia/Nemotron-CC-v2.1",
            name=config_name,
            split="train",
            streaming=True,
            trust_remote_code=True,
            token=True,
        )

        if split == "train":
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        elif split == "validation":
            ds = ds.shuffle(seed=seed + 9999, buffer_size=10_000)

        # Tokenize and concatenate
        all_ids = []
        n_docs = 0
        for example in ds:
            text = example["text"]
            if text.strip():
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                all_ids.extend(ids)
                n_docs += 1

            if len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

            if n_docs % 10_000 == 0 and n_docs > 0:
                print(f"  Nemotron-CC [{subset}]: {len(all_ids):,}/{max_tokens:,} tokens "
                      f"({len(all_ids)/max_tokens*100:.1f}%) from {n_docs:,} docs")

        print(f"  Nemotron-CC [{subset}] loaded: {len(all_ids):,} tokens "
              f"from {n_docs:,} documents")

        n_seqs = len(all_ids) // seq_len
        self.tokens = torch.tensor(all_ids[:n_seqs * seq_len], dtype=torch.long)
        self.tokens = self.tokens.view(n_seqs, seq_len)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.tokens[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


# Registry of supported corpora
CORPUS_REGISTRY = {
    "wikitext": WikiTextDataset,
    "wikipedia-2021": Wikipedia2021Dataset,
    "fineweb-edu": FineWebEduDataset,
    "nemotron-cc": NemotronCCDataset,
}


def get_dataloader(
    split: str,
    tokenizer,
    seq_len: int = 512,
    batch_size: int = 16,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    corpus: str = "wikitext",
    corpus_subset: str = "hq-dqa",
    wikipedia2021_dataset: Optional[str] = None,
    wikipedia2021_source_tokenizer: Optional[str] = None,
    wikipedia2021_require_tokenizer_match: bool = False,
) -> DataLoader:
    """Create a DataLoader for training or evaluation.

    Args:
        split: "train", "validation", or "test"
        tokenizer: HF tokenizer
        seq_len: Sequence length
        batch_size: Batch size
        max_tokens: Maximum tokens to load (None = full split)
        shuffle: Whether to shuffle
        num_workers: DataLoader workers
        seed: Random seed for shuffling
        corpus: "wikitext", "wikipedia-2021", "fineweb-edu", or "nemotron-cc"
        corpus_subset: Subset for nemotron-cc (hq-dqa, hq, mhq, all)
        wikipedia2021_dataset: Optional override for the pre-tokenized
            Wikipedia-2021 dataset repo.
        wikipedia2021_source_tokenizer: Optional override for the tokenizer
            that produced the pre-tokenized Wikipedia-2021 corpus.
        wikipedia2021_require_tokenizer_match: If True, refuse to fall back to
            decode-and-retokenize for Wikipedia-2021.

    Returns:
        DataLoader yielding dicts with "input_ids" and "labels"
    """
    dataset_cls = CORPUS_REGISTRY.get(corpus)
    if dataset_cls is None:
        raise ValueError(f"Unknown corpus: {corpus}. Available: {list(CORPUS_REGISTRY.keys())}")

    kwargs = dict(
        split=split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_tokens=max_tokens,
    )
    if corpus in ("fineweb-edu", "nemotron-cc"):
        kwargs["seed"] = seed
    if corpus == "nemotron-cc":
        kwargs["subset"] = corpus_subset
    if corpus == "wikipedia-2021":
        kwargs["dataset_name"] = wikipedia2021_dataset
        kwargs["source_tokenizer_name"] = wikipedia2021_source_tokenizer
        kwargs["require_tokenizer_match"] = wikipedia2021_require_tokenizer_match

    dataset = dataset_cls(**kwargs)

    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        generator=generator if shuffle else None,
    )


def verify_split_disjointness(tokenizer, seq_len: int = 512) -> dict:
    """Verify that train/val/test splits are disjoint by content hash.

    Returns dict with hash sets and overlap counts.
    """
    results = {}

    for split in ("train", "validation", "test"):
        from datasets import load_dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        texts = set()
        for example in dataset:
            text = example["text"].strip()
            if text:
                h = hashlib.sha256(text.encode()).hexdigest()
                texts.add(h)
        results[split] = texts

    train_val_overlap = len(results["train"] & results["validation"])
    train_test_overlap = len(results["train"] & results["test"])
    val_test_overlap = len(results["validation"] & results["test"])

    return {
        "train_size": len(results["train"]),
        "val_size": len(results["validation"]),
        "test_size": len(results["test"]),
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "all_disjoint": (
            train_val_overlap == 0
            and train_test_overlap == 0
            and val_test_overlap == 0
        ),
    }
