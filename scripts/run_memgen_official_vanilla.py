#!/usr/bin/env python3
"""Run the official MemGen runner in the repository's vanilla mode.

The public MemGen README says vanilla evaluation is obtained by replacing the
active ``MemGenModel.generate`` with the commented vanilla implementation.
This wrapper applies that switch in memory, leaving the official checkout
unchanged, and then executes its unmodified ``main.py``/runner/dataset/env.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch
from transformers import GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("main_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.official_repo.resolve()
    if not (repo / "main.py").is_file():
        raise FileNotFoundError(f"official MemGen main.py not found under {repo}")

    # Ensure imports resolve to the official checkout, not ATHENA modules.
    sys.path.insert(0, str(repo))
    from memgen.model.modeling_memgen import MemGenModel

    # This is the vanilla implementation from the official repository's
    # commented block, activated exactly as instructed by its README.
    @torch.no_grad()
    def official_vanilla_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig | None = None,
        return_augmentation_mask: bool = False,
        **kwargs,
    ):
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)

        inputs_embeds = reasoner.get_input_embeddings()(input_ids.to(self.device))
        attention_mask = attention_mask.to(self.device)
        batch_size, _, _ = inputs_embeds.shape
        augmentation_pos = torch.full(
            (batch_size, max_new_tokens),
            fill_value=-100,
            device=self.device,
        )

        generation_config = GenerationConfig(
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=False,
            max_new_tokens=max_new_tokens,
        )
        generated = reasoner.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        current_input_ids = torch.cat([input_ids.to(self.device), generated], dim=1)

        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
        self._check_generate(current_input_ids[:, prompt_len:], augmentation_pos)

        if return_augmentation_mask:
            return current_input_ids, augmentation_pos
        return current_input_ids

    MemGenModel.generate = official_vanilla_generate

    # Optional bounded smoke mode: patch only the selected official builder's
    # test split. Full evaluation leaves the official dataset construction
    # untouched.
    if args.max_test_examples:
        dataset_name = _dataset_name_from_args(args.main_args)
        if dataset_name != "kodcode":
            raise ValueError("--max-test-examples currently supports --dataset kodcode only")
        from data.kodcode.builder import KodCodeBuilder

        original = KodCodeBuilder._build_sft_datasets

        def bounded_build(self):
            dataset_dict = original(self)
            limit = min(args.max_test_examples, len(dataset_dict["test"]))
            dataset_dict["test"] = dataset_dict["test"].select(range(limit))
            print(f"OFFICIAL_VANILLA_SMOKE_TEST_SIZE {limit}", flush=True)
            return dataset_dict

        KodCodeBuilder._build_sft_datasets = bounded_build

    sys.argv = [str(repo / "main.py"), *args.main_args]
    runpy.run_path(str(repo / "main.py"), run_name="__main__")


def _dataset_name_from_args(values: list[str]) -> str | None:
    for index, value in enumerate(values):
        if value == "--cfg-path" and index + 1 < len(values):
            return Path(values[index + 1]).stem
    return None


if __name__ == "__main__":
    main()
