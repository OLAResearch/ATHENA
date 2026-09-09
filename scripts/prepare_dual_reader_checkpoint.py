"""Convert a frozen legacy Engram adaptor into a dual-reader checkpoint.

The direct reader is copied exactly.  Only the generated residual branch is
newly initialized, so ``engram_only`` remains a bit-for-bit architectural
match to the source Engram adaptor.
"""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.adaptor import EngramAdaptor, MultiBranchEngramAdaptor
from engram.generative_memory import GenerativeMemoryAdaptor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--legacy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--d-mem", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reader-type", choices=["cross_attention", "mean"], default="cross_attention")
    parser.add_argument("--generator-num-latents", type=int, default=4)
    parser.add_argument("--generator-hidden-size", type=int, default=256)
    parser.add_argument("--generator-layers", type=int, default=2)
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--generator-cue-window", type=int, default=3)
    return parser.parse_args()


def _has_module_list_prefix(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.split(".", 1)[0].isdigit() for key in state_dict)


def convert_state_dict(
    legacy_state: dict[str, torch.Tensor],
    d_model: int,
    d_mem: int,
    num_branches: int,
    num_injection_layers: int,
    generator_kwargs: dict,
) -> dict[str, torch.Tensor]:
    legacy_modules = []
    generated_modules = []
    for _ in range(num_injection_layers):
        if num_branches == 1:
            legacy_modules.append(EngramAdaptor(d_model, d_mem))
        else:
            legacy_modules.append(MultiBranchEngramAdaptor(d_model, d_mem, num_branches))
        generated_modules.append(
            GenerativeMemoryAdaptor(
                d_model=d_model,
                d_mem=d_mem,
                num_branches=num_branches,
                fusion_type="dual_reader",
                cue_source="engram",
                **generator_kwargs,
            )
        )

    legacy_container = legacy_modules[0] if num_injection_layers == 1 else nn.ModuleList(legacy_modules)
    legacy_container.load_state_dict(legacy_state)
    for generated, legacy in zip(generated_modules, legacy_modules):
        generated.initialize_engram_reader_from_legacy(legacy)
    generated_container = (
        generated_modules[0]
        if num_injection_layers == 1
        else nn.ModuleList(generated_modules)
    )
    return generated_container.state_dict()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.legacy_config) as handle:
        legacy_config = json.load(handle)
    injection_layers = legacy_config.get("injection_layers") or [10]
    if isinstance(injection_layers, str):
        injection_layers = [int(value) for value in injection_layers.replace(":", ",").split(",")]
    num_branches = int(legacy_config.get("adaptor_branches", 1))
    torch.manual_seed(args.seed)
    legacy_state = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=True)
    uses_module_list = _has_module_list_prefix(legacy_state)
    if uses_module_list != (len(injection_layers) > 1):
        raise ValueError("Checkpoint prefixes do not match configured injection layers")

    generator_kwargs = {
        "reader_type": args.reader_type,
        "num_latents": args.generator_num_latents,
        "hidden_size": args.generator_hidden_size,
        "num_layers": args.generator_layers,
        "num_heads": args.generator_heads,
        "cue_window": args.generator_cue_window,
    }
    converted = convert_state_dict(
        legacy_state,
        d_model=args.d_model,
        d_mem=args.d_mem,
        num_branches=num_branches,
        num_injection_layers=len(injection_layers),
        generator_kwargs=generator_kwargs,
    )
    torch.save(converted, output_dir / "adaptor_best.pt")

    runtime_config = dict(legacy_config)
    runtime_config.update({
        "architecture": "generative",
        "reader_type": args.reader_type,
        "generator_cue_source": "engram",
        "generator_num_latents": args.generator_num_latents,
        "generator_hidden_size": args.generator_hidden_size,
        "generator_layers": args.generator_layers,
        "generator_heads": args.generator_heads,
        "generator_cue_window": args.generator_cue_window,
        "generator_fusion_type": "dual_reader",
        "legacy_engram_checkpoint": str(Path(args.legacy_checkpoint).resolve()),
        "legacy_engram_exact_import": True,
        "initialization_seed": args.seed,
    })
    with open(output_dir / "config.json", "w") as handle:
        json.dump(runtime_config, handle, indent=2)
    print(f"Converted {len(injection_layers)} adaptor(s); direct Engram reader preserved exactly")


if __name__ == "__main__":
    main()
