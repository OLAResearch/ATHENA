"""Cluster smoke test for a target backbone with ATHENA memory readers.

This is intentionally synthetic: it validates model loading, frozen Engram
lookup, Generated Memory + direct Engram dual readers, backward propagation,
and cached decoding without consuming a benchmark example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.backbone_wrapper import BackboneWrapper
from engram.canonicalization import WordBoundaryCanonicalizer, build_canonicalizer
from engram.hashing import WordNgramHasher
from engram.memory import EngramMemory, MemoryConfig
from scripts.eval_openqa import greedy_generate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    return parser.parse_args()


def load_memory(config_path: str, checkpoint_path: str, device):
    config = json.loads(Path(config_path).read_text())
    memory_config = MemoryConfig(
        max_ngram=config["max_ngram"],
        heads_per_order=config["heads_per_order"],
        table_size=config["table_size"],
        d_head=config["d_head"],
        hash_seed=config["hash_seed"],
    )
    memory = EngramMemory(memory_config)
    memory.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    memory.to(device)
    for parameter in memory.parameters():
        parameter.requires_grad = False
    return memory, memory_config


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a visible GPU")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"GPU={torch.cuda.get_device_name(0)}")
    print(f"TORCH={torch.__version__}")

    memory, memory_config = load_memory(
        args.memory_config, args.source_memory, device
    )
    wrapper = BackboneWrapper(
        model_name=args.target_model,
        memory=memory,
        condition="transferred",
        device=device,
        dtype=dtype,
        injection_layers=[2, 10],
        adaptor_branches=4,
        architecture="generative",
        reader_type="cross_attention",
        generator_cue_source="engram",
        generator_num_latents=4,
        generator_hidden_size=256,
        generator_layers=2,
        generator_heads=4,
        generator_cue_window=3,
        generator_fusion_type="dual_reader",
    )
    wrapper.freeze_backbone()
    wrapper.freeze_memory()
    tokenizer = wrapper.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    canonicalizer = build_canonicalizer(
        tokenizer, mode="word_boundary", max_ngram=memory_config.max_ngram
    )
    if not isinstance(canonicalizer, WordBoundaryCanonicalizer):
        raise TypeError("Expected WordBoundaryCanonicalizer")
    hasher = WordNgramHasher(memory_config.hash_config)

    def set_context(input_ids):
        word_ngrams = canonicalizer.compute_word_ngrams(input_ids)
        wrapper.set_hash_indices(
            hasher.hash_word_ngrams(word_ngrams, device=input_ids.device)
        )

    encoded = tokenizer(
        "Question: What is two plus two?\nAnswer:",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    set_context(input_ids)
    outputs = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        use_cache=False,
    )
    loss = outputs.loss
    loss.backward()
    adaptor_grad = sum(
        float(parameter.grad.float().norm().item())
        for parameter in wrapper.adaptor.parameters()
        if parameter.grad is not None
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")
    if adaptor_grad <= 0:
        raise RuntimeError("No adaptor gradient observed")

    wrapper.eval()
    with torch.no_grad():
        generated = greedy_generate(
            wrapper,
            tokenizer,
            "Question: What is two plus two?\nAnswer:",
            device,
            set_context,
            max_new_tokens=args.max_new_tokens,
            max_context_length=256,
            official_tokenization=True,
            stop_at_newline=False,
        )

    result = {
        "completed": True,
        "target_model": args.target_model,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hidden_size": wrapper.d_model,
        "num_layers": wrapper.num_layers,
        "injection_layers": wrapper.injection_layers,
        "adaptor_branches": wrapper.adaptor_branches,
        "loss": float(loss.item()),
        "adaptor_grad_norm_sum": adaptor_grad,
        "generated": generated,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print("TARGET_MEMGEN_SMOKE_COMPLETE " + json.dumps(result))
    wrapper.cleanup()


if __name__ == "__main__":
    main()
