#!/usr/bin/env python3
"""Run the official MemGen main.py with a bounded TriviaQA smoke test."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


# The wrapper lives outside the official repository.  Put the current MemGen
# source checkout ahead of /workspace/scripts so its top-level ``data`` package
# cannot be shadowed by an unrelated installed module.
source_root = Path.cwd().resolve()
sys.path.insert(0, str(source_root))

from data.triviaqa.builder import TriviaQABuilder


MAX_TEST_EXAMPLES = 1
_original_build = TriviaQABuilder._build_sft_datasets


def _bounded_build(self):
    dataset_dict = _original_build(self)
    dataset_dict["test"] = dataset_dict["test"].select(
        range(min(MAX_TEST_EXAMPLES, len(dataset_dict["test"])))
    )
    print(f"OFFICIAL_SMOKE_TEST_SIZE {len(dataset_dict['test'])}", flush=True)
    return dataset_dict


TriviaQABuilder._build_sft_datasets = _bounded_build
runpy.run_path("main.py", run_name="__main__")
