"""Phase 2: Train adaptor on target model with transferred/random/etc. memory.

Supports all 8 experimental conditions via the --condition flag:
  baseline, transferred, random_memory, permuted_keys, no_gate,
  train_from_scratch, ffn_only, affine_stitch

Usage:
    uv run python scripts/train_adaptor.py \
        --config configs/tier1_same_tokenizer.yaml \
        --condition transferred \
        --seed 42 \
        --source-memory results/source_memory/memory.pt \
        --target-model EleutherAI/pythia-410m \
        --output-dir results/tier1/transferred_seed42
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.memory import EngramMemory, MemoryConfig
from engram.adaptor import EngramAdaptor, MultiBranchEngramAdaptor, build_adaptor
from engram.backbone_wrapper import BackboneWrapper
from engram.generative_memory import GenerativeMemoryAdaptor
from engram.tri_memory import TRI_READER_SUBSETS, TriMemoryAdaptor
from engram.canonicalization import build_canonicalizer, WordBoundaryCanonicalizer
from engram.hf_utils import resolve_pretrained_source
from engram.hashing import HashConfig, WordNgramHasher
from engram.data import get_dataloader
from engram.metrics import compute_perplexity, compute_batch_perplexities, bootstrap_ci
from engram.gate_analyzer import GateAnalyzer


VALID_CONDITIONS = [
    "baseline",
    "transferred",
    "random_memory",
    "permuted_keys",
    "no_gate",
    "train_from_scratch",
    "ffn_only",
    "affine_stitch",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Engram adaptor")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--condition", type=str, required=True, choices=VALID_CONDITIONS)
    parser.add_argument("--target-model", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--source-memory", type=str, default="results/source_memory/memory.pt")
    parser.add_argument("--memory-config", type=str, default="results/source_memory/memory_config.json")
    parser.add_argument("--init-adaptor", type=str, default=None,
                        help="Optional adaptor checkpoint to warm-start the target adaptor")
    parser.add_argument(
        "--train-generated-branch-only",
        action="store_true",
        help=(
            "Freeze an imported dual-reader Engram path and optimize only the "
            "generated residual branch. Requires architecture=generative, "
            "generator_fusion_type=dual_reader, and --init-adaptor."
        ),
    )
    parser.add_argument(
        "--dual-reader-mode",
        choices=["both", "generated_only"],
        default="both",
        help="Reader contributions active while fitting a generated branch.",
    )
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--validation-max-tokens",
        type=int,
        default=2_000_000,
        help="Validation-token cap (keep the default for full runs; lower only for smoke tests)",
    )
    parser.add_argument("--injection-layers", type=str, default=None,
                        help="Comma-separated layer indices for memory injection")
    parser.add_argument("--adaptor-branches", type=int, default=1,
                        help="Number of branch-specific key/gating paths in the adaptor")
    parser.add_argument("--architecture", choices=["legacy", "generative"], default="legacy")
    parser.add_argument("--reader-type", choices=["cross_attention", "mean"], default="cross_attention")
    parser.add_argument(
        "--generator-cue-source",
        choices=["engram", "context", "learned", "hybrid"],
        default="engram",
    )
    parser.add_argument("--generator-num-latents", type=int, default=4)
    parser.add_argument("--generator-hidden-size", type=int, default=256)
    parser.add_argument("--generator-layers", type=int, default=2)
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--generator-cue-window", type=int, default=3)
    parser.add_argument("--generator-adaptive-router", action="store_true")
    parser.add_argument("--generator-router-hidden-size", type=int, default=16)
    parser.add_argument("--generator-router-semantic-size", type=int, default=0)
    parser.add_argument(
        "--generator-source-adapter-rank",
        type=int,
        default=16,
        help="Rank of the source-specific GE/GH output adapters",
    )
    parser.add_argument(
        "--generator-loop-rounds",
        type=int,
        default=1,
        help="Shared Reader workspace passes; 1 keeps the legacy path",
    )
    parser.add_argument(
        "--generator-loop-workspace-size",
        type=int,
        default=0,
        help="Per-token workspace width (0 uses generator hidden size)",
    )
    parser.add_argument(
        "--generator-loop-gate-max",
        type=float,
        default=0.25,
        help="Maximum residual coefficient per loop pass",
    )
    parser.add_argument(
        "--generator-router-expert-mode",
        choices=["residual", "source"],
        default="residual",
        help="Route between [Engram, Engram+Generated] or [Engram, Generated].",
    )
    parser.add_argument(
        "--generator-fusion-type",
        choices=["generated_only", "engram_residual", "dual_reader", "tri_reader"],
        default="generated_only",
        help="Fuse generated latent value alone or retain an Engram residual.",
    )
    parser.add_argument("--memory-dim", type=int, default=None,
                        help="Memory/cue dimension for generative learned controls")
    # Cross-tokenizer mode
    parser.add_argument("--canon-mode", type=str, default="vocab",
                        choices=["vocab", "word_boundary"])
    parser.add_argument("--corpus", type=str, default="wikitext",
                        choices=["wikitext", "wikipedia-2021", "fineweb-edu", "nemotron-cc"],
                        help="Training/eval corpus for adaptor fitting (default: wikitext)")
    parser.add_argument("--corpus-subset", type=str, default="hq-dqa",
                        help="Subset for nemotron-cc: hq-dqa, hq, mhq, all (default: hq-dqa)")
    parser.add_argument("--wikipedia2021-dataset", type=str, default=None,
                        help="Override HF dataset repo for corpus=wikipedia-2021")
    parser.add_argument("--wikipedia2021-source-tokenizer", type=str, default=None,
                        help="Tokenizer that produced the pre-tokenized wikipedia-2021 corpus")
    parser.add_argument("--wikipedia2021-require-tokenizer-match", action="store_true",
                        help="Disable decode+retokenize fallback for wikipedia-2021 and require a tokenizer-matched dataset")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after N evals without val PPL improvement (0=disabled)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing on backbone (saves memory)")
    parser.add_argument(
        "--joint-engram-generated-router",
        action="store_true",
        help=(
            "Jointly fit a freshly initialized direct Engram reader, generated "
            "residual, and adaptive router in one target-fitting stage."
        ),
    )
    parser.add_argument(
        "--joint-source-reader",
        action="store_true",
        help=(
            "Train Engram-only, Generated-only, and an E/G source reader from "
            "the first token using independent causal-LM losses."
        ),
    )
    parser.add_argument(
        "--joint-tri-reader",
        action="store_true",
        help=(
            "Train direct Engram (E), Engram-conditioned generation (GE), "
            "context-only generation (GH), and a three-way Reader jointly "
            "from token zero with an independent causal-LM loss per path."
        ),
    )
    parser.add_argument(
        "--joint-tri-subset-reader",
        action="store_true",
        help=(
            "Train E, GE, GH and the unified seven-way subset Reader jointly "
            "from token zero. The Reader chooses E, GE, GH, any pair, or all "
            "three independently for each token."
        ),
    )
    parser.add_argument(
        "--joint-tri-route-only",
        action="store_true",
        help=(
            "Train a three-expert MoE-style Reader with one routed forward per "
            "batch. E, GE, and GH are mixed per token; forced endpoint paths "
            "and seven-way subset paths are skipped for speed."
        ),
    )
    parser.add_argument(
        "--deployment-reader-mode",
        choices=[
            "tri_routed",
            "tri_soft_fused",
            "tri_subset_routed",
            "tri_subset_soft_fused",
        ],
        default=None,
        help=(
            "Optional explicit deployment mode for a tri-reader checkpoint. "
            "Defaults to the same soft mode used for validation/selection; "
            "choose a hard mode only for an intentional ablation."
        ),
    )
    parser.add_argument(
        "--advantage-reader",
        type=json.loads,
        default=None,
        help=(
            "JSON object passed to configure_advantage_reader before a strict "
            "warm-start load, e.g. '{\"candidates\":\"sources\"}'."
        ),
    )
    parser.add_argument("--engram-source-loss-weight", type=float, default=1.0)
    parser.add_argument("--generated-source-loss-weight", type=float, default=1.0)
    parser.add_argument("--ge-source-loss-weight", type=float, default=1.0)
    parser.add_argument("--gh-source-loss-weight", type=float, default=1.0)
    parser.add_argument("--routed-source-loss-weight", type=float, default=1.0)
    parser.add_argument("--subset-routed-source-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--tri-router-distillation-weight",
        type=float,
        default=0.2,
        help=(
            "Weight for Wikipedia next-token counterfactual supervision: the "
            "Reader predicts which of E/GE/GH has the lowest per-token NLL."
        ),
    )
    parser.add_argument(
        "--tri-router-load-balance-weight",
        type=float,
        default=0.01,
        help="KL-to-uniform weight on mean E/GE/GH usage for route-only training.",
    )
    parser.add_argument(
        "--deterministic-engram-init-seed",
        type=int,
        default=None,
        help="Reset direct Engram readers from a standalone deterministic legacy initialization.",
    )
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument(
        "--router-start-tokens",
        type=int,
        default=2_000_000,
        help="Use fixed alpha=1 before this many tokens, then enable the learned router.",
    )
    parser.add_argument("--router-init-alpha", type=float, default=0.95)
    parser.add_argument("--generated-residual-penalty", type=float, default=1e-4)
    parser.add_argument("--skip-final-test-eval", action="store_true",
                        help="Skip the final full-test PPL sweep and save adaptor immediately")
    return parser.parse_args()


def load_config(args):
    """Override args with YAML config if provided."""
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        adaptor_cfg = cfg.get("adaptor_training", {})
        for key, val in adaptor_cfg.items():
            key_underscore = key.replace("-", "_")
            if hasattr(args, key_underscore):
                setattr(args, key_underscore, val)
    return args


def tri_reader_mode_contract(args) -> dict:
    """Describe the train/selection/deployment modes for a tri run.

    Route-only training performs its single Reader forward in soft mode.  The
    same mode must drive every validation-based checkpoint decision; hard
    ``tri_routed`` remains an explicitly named downstream ablation.  The
    subset Reader follows the analogous soft contract.
    """
    if getattr(args, "joint_tri_route_only", False):
        training_mode = "tri_soft_fused"
        hard_ablation = "tri_routed"
    elif getattr(args, "joint_tri_subset_reader", False):
        training_mode = "tri_subset_soft_fused"
        hard_ablation = "tri_subset_routed"
    elif getattr(args, "joint_tri_reader", False):
        # The historical endpoint-supervised tri design intentionally trains
        # its hard source Reader path.  Preserve that mode as its contract.
        training_mode = "tri_routed"
        hard_ablation = None
    else:
        training_mode = None
        hard_ablation = None

    deployment_mode = getattr(args, "deployment_reader_mode", None)
    if deployment_mode is None:
        deployment_mode = training_mode
    return {
        "training_reader_mode": training_mode,
        "deployment_reader_mode": deployment_mode,
        "checkpoint_selection_reader_mode": training_mode,
        "hard_ablation_reader_mode": hard_ablation,
    }


def get_tri_training_reader_mode(args) -> str | None:
    """Return the exact mode used for validation checkpoint selection."""
    return tri_reader_mode_contract(args)["training_reader_mode"]


def get_checkpoint_selection_reader_mode(args) -> str | None:
    """Compatibility-named helper used by training smoke/audit tests."""
    return tri_reader_mode_contract(args)["checkpoint_selection_reader_mode"]


def configure_advantage_reader(wrapper, advantage_reader: dict | None) -> bool:
    """Materialize a lazy advantage head before a strict adaptor load."""
    if advantage_reader is None:
        return False
    if not isinstance(advantage_reader, dict):
        raise TypeError("advantage_reader must be a dictionary")
    adaptors = _wrapper_adaptors(wrapper)
    if not adaptors:
        raise ValueError("advantage_reader requires an adaptor")
    values = {
        "candidates": advantage_reader.get("candidates", "sources"),
        "threshold": float(advantage_reader.get("threshold", 0.0)),
        "confidence_threshold": float(
            advantage_reader.get("confidence_threshold", 0.5)
        ),
        "temperature": float(advantage_reader.get("temperature", 0.15)),
        "max_scale": float(advantage_reader.get("max_scale", 1.0)),
    }
    if values["candidates"] not in {"sources", "subsets"}:
        raise ValueError("advantage_reader.candidates must be 'sources' or 'subsets'")
    for adaptor in adaptors:
        configure = getattr(adaptor, "configure_advantage_reader", None)
        if configure is None:
            raise AttributeError(
                f"{type(adaptor).__name__} does not support advantage_reader"
            )
        configure(**values)
    return True


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_injection_layers(raw) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    raw = str(raw).strip()
    if not raw:
        return None
    return [
        int(part.strip())
        for part in re.split(r"[\s,:;]+", raw)
        if part.strip()
    ]


def enable_frozen_backbone_gradient_checkpointing(wrapper: BackboneWrapper) -> None:
    """Enable checkpointing without severing gradients to injected adaptors.

    Hugging Face's default re-entrant checkpointing requires at least one input
    tensor to require gradients.  ATHENA freezes every backbone parameter and
    injects the trainable adaptor through layer hooks, so the embedding output
    would otherwise be non-differentiable.  This only asks autograd to track the
    embedding output; it does not unfreeze or optimize any backbone parameter.
    """
    wrapper.backbone.gradient_checkpointing_enable()
    enable_input_grads = getattr(wrapper.backbone, "enable_input_require_grads", None)
    if enable_input_grads is None:
        raise RuntimeError(
            "The selected backbone does not expose enable_input_require_grads(), "
            "which is required when gradient checkpointing a frozen backbone"
        )
    enable_input_grads()


def load_initial_adaptor(
    wrapper: BackboneWrapper,
    checkpoint_path: str,
    advantage_reader: dict | None = None,
) -> int:
    """Load either a complete multi-layer checkpoint or a shared adaptor.

    Current checkpoints save the full ``ModuleList`` with keys such as
    ``0.w_v.weight``.  Older warm starts sometimes contain one unprefixed
    adaptor that should be copied into every injection layer.
    """
    if wrapper.adaptor is None:
        raise ValueError("--init-adaptor was provided but this condition does not build an adaptor")
    configure_advantage_reader(wrapper, advantage_reader)
    init_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(wrapper.adaptor, torch.nn.ModuleList):
        has_layer_prefix = any(key.split(".", 1)[0].isdigit() for key in init_state)
        if has_layer_prefix:
            wrapper.adaptor.load_state_dict(init_state, strict=True)
            init_params = sum(value.numel() for value in init_state.values())
        else:
            for adaptor in wrapper.adaptor:
                adaptor.load_state_dict(init_state, strict=True)
            init_params = sum(value.numel() for value in init_state.values()) * len(wrapper.adaptor)
    else:
        wrapper.adaptor.load_state_dict(init_state, strict=True)
        init_params = sum(value.numel() for value in init_state.values())
    return init_params


def configure_generated_residual_training(
    wrapper: BackboneWrapper, mode: str = "both"
) -> list[str]:
    """Freeze the imported Engram reader and expose generated parameters."""
    if mode not in {"both", "generated_only"}:
        raise ValueError(f"Unsupported generated-branch training mode: {mode}")
    adaptors = (
        list(wrapper.adaptor)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList)
        else [wrapper.adaptor]
    )
    trainable_names = []
    for index, adaptor in enumerate(adaptors):
        if not isinstance(adaptor, GenerativeMemoryAdaptor):
            raise TypeError("Generated-branch-only training requires GenerativeMemoryAdaptor")
        names = adaptor.train_generated_branch_only()
        adaptor.set_dual_reader_mode(mode)
        prefix = f"{index}." if len(adaptors) > 1 else ""
        trainable_names.extend(prefix + name for name in names)
    return trainable_names


def _wrapper_adaptors(wrapper: BackboneWrapper) -> list[torch.nn.Module]:
    return (
        list(wrapper.adaptor)
        if isinstance(wrapper.adaptor, torch.nn.ModuleList)
        else [wrapper.adaptor]
    )


def initialize_direct_engram_readers(
    wrapper: BackboneWrapper, seed: int
) -> list[str]:
    """Give legacy and joint variants exactly the same direct-reader start.

    The private RNG scope makes the direct Engram initialization independent of
    how many generated-memory modules were constructed around it.
    """
    targets = _wrapper_adaptors(wrapper)
    initialized = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for index, target in enumerate(targets):
            if wrapper.adaptor_branches > 1:
                legacy = MultiBranchEngramAdaptor(
                    wrapper.d_model,
                    wrapper.memory.d_mem,
                    num_branches=wrapper.adaptor_branches,
                )
            else:
                legacy = EngramAdaptor(wrapper.d_model, wrapper.memory.d_mem)
            if isinstance(target, (GenerativeMemoryAdaptor, TriMemoryAdaptor)):
                target.initialize_engram_reader_from_legacy(legacy)
            elif isinstance(target, (EngramAdaptor, MultiBranchEngramAdaptor)):
                target.load_state_dict(legacy.state_dict(), strict=True)
            else:
                raise TypeError(
                    "Deterministic Engram initialization requires a legacy or "
                    f"dual-reader adaptor, got {type(target).__name__}"
                )
            initialized.append(f"reader_{index}")
    return initialized


def _set_joint_reader_mode(wrapper: BackboneWrapper, mode: str) -> None:
    for adaptor in _wrapper_adaptors(wrapper):
        if isinstance(adaptor, TriMemoryAdaptor):
            adaptor.set_tri_reader_mode(mode)
        elif isinstance(adaptor, GenerativeMemoryAdaptor):
            adaptor.set_dual_reader_mode(mode)
        else:
            raise TypeError("Joint reader mode requires a generative adaptor")


def configure_joint_tri_reader_training(
    wrapper: BackboneWrapper,
    *,
    unified_subset_reader: bool = False,
    route_only: bool = False,
) -> tuple[list[str], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Expose E/GE/GH and a Reader from the first token.

    ``unified_subset_reader`` selects the new seven-way subset Reader.  With
    ``route_only`` the caller uses one soft three-way forward per batch; the
    historical endpoint-supervised path remains the default for reproduction.
    """
    adaptors = _wrapper_adaptors(wrapper)
    if not adaptors or not all(isinstance(adaptor, TriMemoryAdaptor) for adaptor in adaptors):
        raise TypeError("Three-way training requires TriMemoryAdaptor instances")
    for adaptor in adaptors:
        if adaptor.router is None:
            raise ValueError("Three-way training requires an adaptive router")
        for parameter in adaptor.parameters():
            parameter.requires_grad = True
        adaptor.configure_router(temperature=1.0, hard=False)
        # Train the Reader with a differentiable soft mixture.  A
        # hard straight-through argmax is still exposed for final evaluation,
        # but using it during the zero-initialized warm-up makes the E tie win
        # the first updates and can starve all six pair/triple candidates.
        adaptor.set_tri_reader_mode(
            "tri_subset_soft_fused"
            if unified_subset_reader
            else "tri_soft_fused"
            if route_only
            else "tri_routed"
        )

    named = list(wrapper.adaptor.named_parameters())
    router_params = [
        parameter
        for name, parameter in named
        if ".router." in f".{name}" or ".subset_router." in f".{name}"
    ]
    expert_params = [
        parameter
        for name, parameter in named
        if ".router." not in f".{name}"
        and ".subset_router." not in f".{name}"
    ]
    names = [name for name, parameter in named if parameter.requires_grad]
    if not router_params or not expert_params:
        raise RuntimeError("Tri-reader optimizer groups must contain experts and router")
    return names, expert_params, router_params


def configure_joint_engram_generated_router_training(
    wrapper: BackboneWrapper,
    *,
    router_init_alpha: float,
) -> tuple[list[str], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Expose both readers and router while preserving the frozen backbone/table."""
    if not 0.0 < router_init_alpha < 1.0:
        raise ValueError("--router-init-alpha must be strictly between 0 and 1")
    adaptors = _wrapper_adaptors(wrapper)
    if not adaptors or not all(
        isinstance(adaptor, GenerativeMemoryAdaptor) for adaptor in adaptors
    ):
        raise TypeError("Joint training requires generative-memory adaptors")

    alpha_logit = math.log(router_init_alpha / (1.0 - router_init_alpha))
    for adaptor in adaptors:
        if adaptor.fusion_type != "dual_reader" or adaptor.router is None:
            raise ValueError(
                "Joint training requires dual_reader fusion with an adaptive router"
            )
        for parameter in adaptor.parameters():
            parameter.requires_grad = True
        with torch.no_grad():
            # Step zero is exactly the direct Engram model.  The output layer
            # learns first; gradients then reach the deeper generator.
            adaptor.output_projection.weight.zero_()
            adaptor.router[-1].weight.zero_()
            adaptor.router[-1].bias.copy_(
                torch.tensor(
                    [-0.5 * alpha_logit, 0.5 * alpha_logit],
                    device=adaptor.router[-1].bias.device,
                    dtype=adaptor.router[-1].bias.dtype,
                )
            )
        adaptor.configure_router(temperature=1.0, hard=False)
        adaptor.set_dual_reader_mode("both")

    named = list(wrapper.adaptor.named_parameters())
    router_params = [parameter for name, parameter in named if ".router." in f".{name}"]
    reader_params = [parameter for name, parameter in named if ".router." not in f".{name}"]
    names = [name for name, parameter in named if parameter.requires_grad]
    if not router_params or not reader_params:
        raise RuntimeError("Joint optimizer groups must contain reader and router parameters")
    return names, reader_params, router_params


def generated_residual_energy(wrapper: BackboneWrapper) -> torch.Tensor:
    values = []
    for adaptor in _wrapper_adaptors(wrapper):
        if not isinstance(adaptor, GenerativeMemoryAdaptor):
            continue
        residual = adaptor.get_last_generated_residual()
        if residual is None:
            raise RuntimeError("Joint forward did not expose a generated residual")
        values.append(residual.float().square().mean())
    if not values:
        raise RuntimeError("No generated residuals were observed")
    return torch.stack(values).mean()


def joint_router_weights(wrapper: BackboneWrapper) -> list[float] | None:
    values = []
    for adaptor in _wrapper_adaptors(wrapper):
        getter = getattr(adaptor, "get_last_router_weights", None)
        if getter is None:
            continue
        weights = getter()
        if weights is not None:
            values.append(weights.detach().float().mean(dim=(0, 1)))
    if not values:
        return None
    return [float(value) for value in torch.stack(values).mean(dim=0).cpu()]


def joint_subset_router_weights(wrapper: BackboneWrapper) -> list[float] | None:
    """Return the seven-way subset probabilities from unified Readers."""
    values = []
    for adaptor in _wrapper_adaptors(wrapper):
        getter = getattr(adaptor, "get_last_subset_router_weights", None)
        if getter is None:
            continue
        weights = getter()
        if weights is not None:
            values.append(weights.detach().float().mean(dim=(0, 1)))
    if not values:
        return None
    return [float(value) for value in torch.stack(values).mean(dim=0).cpu()]


def tri_reader_diagnostics(wrapper: BackboneWrapper) -> dict | None:
    """Return source usage, entropy, and a simple collapse alarm.

    The diagnostics are computed only from the Reader's prediction weights,
    never from benchmark labels.  ``collapse`` is intentionally an alarm, not
    a training decision: a source receiving at least 95% of the average mass
    is reported so the run can be audited before changing the objective.  The
    seven-way Reader additionally reports singleton/pair/triple mass.  This
    catches a distinct degeneracy where usage is spread over E, GE, and GH so
    the ordinary max-mass alarm is false, but no multi-expert subset is ever
    selected.
    """
    values = []
    subset_values = []
    for adaptor in _wrapper_adaptors(wrapper):
        if not isinstance(adaptor, TriMemoryAdaptor):
            continue
        weights = adaptor.get_last_router_weights()
        if weights is not None:
            values.append(weights.detach().float())
        subset_weights = adaptor.get_last_subset_router_weights()
        if subset_weights is not None:
            subset_values.append(subset_weights.detach().float())
    if not values:
        return None
    weights = torch.stack(values).mean(dim=0)
    usage = weights.mean(dim=(0, 1))
    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
    diagnostics = {
        "usage": [float(value) for value in usage.cpu()],
        "entropy": float(entropy.mean().item()),
        "normalized_entropy": float(
            (entropy.mean() / math.log(weights.shape[-1])).item()
        ),
        "collapse": bool(usage.max().item() >= 0.95),
    }
    if subset_values:
        subset_weights = torch.stack(subset_values).mean(dim=0)
        subset_usage = subset_weights.mean(dim=(0, 1))
        subset_entropy = -(
            subset_weights.clamp_min(1e-8)
            * subset_weights.clamp_min(1e-8).log()
        ).sum(dim=-1)
        diagnostics.update(
            {
                "subset_usage": [
                    float(value) for value in subset_usage.cpu()
                ],
                "subset_entropy": float(subset_entropy.mean().item()),
                "subset_normalized_entropy": float(
                    (subset_entropy.mean() / math.log(subset_weights.shape[-1])).item()
                ),
                "subset_collapse": bool(subset_usage.max().item() >= 0.95),
                "subset_singleton_mass": float(subset_usage[:3].sum().item()),
                "subset_pair_mass": float(subset_usage[3:6].sum().item()),
                "subset_triple_mass": float(subset_usage[6].item()),
                "subset_multi_expert_mass": float(subset_usage[3:].sum().item()),
                "subset_active_classes": int((subset_usage >= 1e-4).sum().item()),
                "subset_combination_collapse": bool(
                    subset_usage[3:].sum().item() < 0.01
                ),
            }
        )
    return diagnostics


def tri_router_load_balance_loss(wrapper: BackboneWrapper) -> torch.Tensor:
    """Keep route-only training from assigning every token to one expert.

    The penalty is a KL divergence between the batch-average source usage and
    the uniform three-way prior.  It does not force every token to use all
    experts; it only prevents a global E/GE/GH collapse while the soft router
    learns.  Hard argmax routing is used later for evaluation.
    """
    values = []
    for adaptor in _wrapper_adaptors(wrapper):
        if not isinstance(adaptor, TriMemoryAdaptor):
            continue
        weights = adaptor.get_last_router_weights()
        if weights is not None:
            values.append(weights.float().mean(dim=(0, 1)))
    if not values:
        raise RuntimeError("No three-way router weights were produced")
    usage = torch.stack(values).mean(dim=0).clamp_min(1e-8)
    return (usage * (usage * usage.new_tensor(3.0)).log()).sum()


def tri_reader_gradient_norms(wrapper: BackboneWrapper) -> dict[str, float] | None:
    """Expose source-specific gradient evidence for tri-reader smoke audits."""
    if not any(isinstance(adaptor, TriMemoryAdaptor) for adaptor in _wrapper_adaptors(wrapper)):
        return None
    squared = {"E": 0.0, "GE": 0.0, "GH": 0.0, "shared_generator": 0.0, "reader": 0.0}
    for name, parameter in wrapper.adaptor.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.float().norm().item()) ** 2
        if ".router." in f".{name}" or ".subset_router." in f".{name}":
            squared["reader"] += value
        elif "engram_cue_projection" in name or "source_output_down.0" in name or "source_output_up.0" in name:
            squared["GE"] += value
        elif "context_cue_projection" in name or "source_output_down.1" in name or "source_output_up.1" in name:
            squared["GH"] += value
        elif name.startswith(("engram_value_projection", "engram_key_projection", "engram_reader_norm", "engram_gate_bias")) or ".engram_value_projection" in name or ".engram_key_projection" in name:
            squared["E"] += value
        else:
            squared["shared_generator"] += value
    return {name: math.sqrt(value) for name, value in squared.items()}


def per_token_next_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return detached causal next-token NLLs with shape ``(B, T - 1)``."""
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shifted_labels)
    return losses.detach()


def tri_joint_loss_terms(
    source_lm_loss: torch.Tensor,
    source_distillation: torch.Tensor,
    source_weight: float,
    normalizer: float,
    distillation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine one path loss with an independently weighted Reader target.

    The path losses form a weighted average, while counterfactual Reader
    supervision is intentionally not divided by that path normalizer.  This
    keeps ``distillation_weight`` interpretable when the number of endpoint
    paths changes (for example from 4 to 8 in the seven-way design).
    """
    if normalizer <= 0:
        raise ValueError("Tri-reader path-loss normalizer must be positive")
    path_scale = source_weight / normalizer
    weighted_lm = path_scale * source_lm_loss
    weighted_distillation = distillation_weight * source_distillation
    return weighted_lm + weighted_distillation, weighted_lm, weighted_distillation


def tri_reader_distillation_loss(
    wrapper: BackboneWrapper,
    endpoint_token_nlls: list[torch.Tensor],
    labels: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    """Train the Reader toward the best E/GE/GH next-token endpoint."""
    if len(endpoint_token_nlls) != 3:
        raise ValueError("Tri-reader distillation requires E, GE, and GH losses")
    targets = torch.stack(endpoint_token_nlls, dim=-1).argmin(dim=-1)
    valid = labels[:, 1:] != -100
    if not bool(valid.any()):
        raise ValueError("Tri-reader distillation batch contains no target tokens")
    losses = []
    for adaptor in _wrapper_adaptors(wrapper):
        if not isinstance(adaptor, TriMemoryAdaptor):
            continue
        logits = adaptor.get_last_router_logits()
        if logits is None:
            raise RuntimeError("tri_routed forward did not expose Reader logits")
        token_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 3),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        losses.append(token_loss[valid].mean())
    if not losses:
        raise RuntimeError("No tri-reader logits were available for distillation")
    usage = [
        float(((targets == index) & valid).sum().item() / valid.sum().item())
        for index in range(3)
    ]
    return torch.stack(losses).mean(), usage


def tri_subset_reader_distillation_loss(
    wrapper: BackboneWrapper,
    endpoint_token_nlls: list[torch.Tensor],
    labels: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    """Train the unified Reader toward the best of seven subset endpoints."""
    if len(endpoint_token_nlls) != len(TRI_READER_SUBSETS):
        raise ValueError(
            "Unified subset distillation requires the seven subset endpoint losses"
        )
    targets = torch.stack(endpoint_token_nlls, dim=-1).argmin(dim=-1)
    valid = labels[:, 1:] != -100
    if not bool(valid.any()):
        raise ValueError("Unified subset distillation batch contains no target tokens")
    losses = []
    for adaptor in _wrapper_adaptors(wrapper):
        if not isinstance(adaptor, TriMemoryAdaptor):
            continue
        logits = adaptor.get_last_subset_router_logits()
        if logits is None:
            raise RuntimeError(
                "tri_subset_routed forward did not expose seven-way Reader logits"
            )
        token_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, len(TRI_READER_SUBSETS)),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        losses.append(token_loss[valid].mean())
    if not losses:
        raise RuntimeError("No unified Reader logits were available for distillation")
    usage = [
        float(((targets == index) & valid).sum().item() / valid.sum().item())
        for index in range(len(TRI_READER_SUBSETS))
    ]
    return torch.stack(losses).mean(), usage


def setup_memory(args, device) -> tuple:
    """Set up memory based on condition."""
    # Learned generative cues are an explicit capacity control.  Keep this
    # branch independent of all Engram files and lookup machinery.
    if args.architecture == "generative" and args.generator_cue_source == "learned":
        if args.memory_dim is None:
            raise ValueError("--memory-dim is required for generative learned cue source")
        return None, None, int(args.memory_dim)

    # Load memory config
    with open(args.memory_config) as f:
        mem_cfg_dict = json.load(f)

    # Validate canon_mode matches the source memory pipeline
    source_canon_mode = mem_cfg_dict.get("canon_mode")
    if source_canon_mode is not None and source_canon_mode != args.canon_mode:
        raise ValueError(
            f"Canon mode mismatch: source memory was trained with "
            f"canon_mode='{source_canon_mode}' but --canon-mode='{args.canon_mode}' "
            f"was specified. Using mismatched canonicalization will produce "
            f"invalid hash indices and silently corrupt transfer results."
        )

    mem_cfg = MemoryConfig(
        max_ngram=mem_cfg_dict["max_ngram"],
        heads_per_order=mem_cfg_dict["heads_per_order"],
        table_size=mem_cfg_dict["table_size"],
        d_head=mem_cfg_dict["d_head"],
        hash_seed=mem_cfg_dict["hash_seed"],
    )

    condition = args.condition

    if condition == "baseline":
        return None, mem_cfg, mem_cfg.d_mem

    if condition == "ffn_only":
        return None, mem_cfg, mem_cfg.d_mem

    # Create memory module
    memory = EngramMemory(mem_cfg)

    if condition == "transferred":
        # Load trained source memory, freeze
        memory.load_state_dict(torch.load(args.source_memory, map_location="cpu", weights_only=True))
        for p in memory.parameters():
            p.requires_grad = False

    elif condition == "random_memory":
        # Fresh random init, freeze
        for p in memory.parameters():
            p.requires_grad = False

    elif condition == "permuted_keys":
        # Load trained source memory, permute rows, freeze
        memory.load_state_dict(torch.load(args.source_memory, map_location="cpu", weights_only=True))
        memory.permute_keys(seed=args.seed)
        for p in memory.parameters():
            p.requires_grad = False

    elif condition == "train_from_scratch":
        # Fresh random init, trainable
        pass  # defaults are fine

    elif condition in ("no_gate", "affine_stitch"):
        # Load trained source memory, freeze
        memory.load_state_dict(torch.load(args.source_memory, map_location="cpu", weights_only=True))
        for p in memory.parameters():
            p.requires_grad = False

    return memory, mem_cfg, mem_cfg.d_mem


def main():
    args = parse_args()
    args = load_config(args)

    joint_flags = (
        args.joint_engram_generated_router,
        args.joint_source_reader,
        args.joint_tri_reader,
        args.joint_tri_subset_reader,
        args.joint_tri_route_only,
    )
    if sum(bool(flag) for flag in joint_flags) > 1:
        raise ValueError(
            "Choose only one joint reader training design"
        )

    if args.joint_engram_generated_router or args.joint_source_reader:
        if args.architecture != "generative" or args.generator_fusion_type != "dual_reader":
            raise ValueError(
                "Joint reader training requires architecture=generative "
                "and generator_fusion_type=dual_reader"
            )
        if args.train_generated_branch_only or args.init_adaptor is not None:
            raise ValueError(
                "Fair joint training must start fresh; do not combine it with "
                "--train-generated-branch-only or --init-adaptor"
            )
        if args.deterministic_engram_init_seed is None:
            raise ValueError(
                "Fair joint training requires --deterministic-engram-init-seed"
            )
        if args.router_start_tokens < 0 or args.router_start_tokens >= args.max_tokens:
            raise ValueError("--router-start-tokens must be in [0, max_tokens)")
        if args.generated_residual_penalty < 0:
            raise ValueError("--generated-residual-penalty must be non-negative")
        if args.joint_source_reader:
            weights = (
                args.engram_source_loss_weight,
                args.generated_source_loss_weight,
                args.routed_source_loss_weight,
            )
            if any(weight <= 0 for weight in weights):
                raise ValueError("All source-reader loss weights must be positive")
            if args.generator_router_expert_mode != "source":
                raise ValueError(
                    "--joint-source-reader requires --generator-router-expert-mode source"
                )
            if args.router_start_tokens != 0:
                raise ValueError(
                    "--joint-source-reader starts at token zero; set --router-start-tokens 0"
                )
        args.generator_adaptive_router = True

    if (
        args.joint_tri_reader
        or args.joint_tri_subset_reader
        or args.joint_tri_route_only
    ):
        if (
            args.architecture != "generative"
            or args.generator_fusion_type != "tri_reader"
            or args.generator_cue_source != "hybrid"
        ):
            raise ValueError(
                "Tri-memory joint training requires architecture=generative, "
                "generator_fusion_type=tri_reader, and generator_cue_source=hybrid"
            )
        if args.train_generated_branch_only or args.init_adaptor is not None:
            raise ValueError("Tri-memory joint training must start from a fresh adaptor")
        if args.deterministic_engram_init_seed is None:
            raise ValueError(
                "Tri-memory joint training requires --deterministic-engram-init-seed"
            )
        if not args.joint_tri_route_only:
            weights = (
                args.engram_source_loss_weight,
                args.ge_source_loss_weight,
                args.gh_source_loss_weight,
                args.routed_source_loss_weight,
            )
            if args.joint_tri_subset_reader:
                weights = weights + (args.subset_routed_source_loss_weight,)
            if any(weight <= 0 for weight in weights):
                raise ValueError("All tri-reader path loss weights must be positive")
        if args.tri_router_distillation_weight < 0:
            raise ValueError("--tri-router-distillation-weight must be non-negative")
        if args.tri_router_load_balance_weight < 0:
            raise ValueError("--tri-router-load-balance-weight must be non-negative")
        if args.generated_residual_penalty != 0:
            raise ValueError(
                "Tri-memory source competition does not use a residual penalty; "
                "set --generated-residual-penalty 0"
            )
        args.generator_adaptive_router = True

    reader_contract = tri_reader_mode_contract(args)
    args.training_reader_mode = reader_contract["training_reader_mode"]
    args.deployment_reader_mode = reader_contract["deployment_reader_mode"]
    args.checkpoint_selection_reader_mode = reader_contract[
        "checkpoint_selection_reader_mode"
    ]
    args.hard_ablation_reader_mode = reader_contract["hard_ablation_reader_mode"]
    checkpoint_selection_mode = reader_contract["checkpoint_selection_reader_mode"]
    if args.advantage_reader is not None:
        # The lazy head is a reader-side module and requires the adaptive
        # construction path before it can be materialized.
        args.generator_adaptive_router = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Use bf16 on GPUs with compute capability >= 8 (A100, H100), fp16 otherwise
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Device: {device}, dtype: {dtype}, Condition: {args.condition}, Seed: {args.seed}")

    torch.manual_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolve_pretrained_source(args.target_model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup memory
    memory, mem_cfg, memory_dim = setup_memory(args, device)
    args.memory_dim = memory_dim

    # Save the complete runtime configuration after resolving memory_dim.
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build wrapper
    wrapper = BackboneWrapper(
        model_name=args.target_model,
        memory=memory,
        condition=args.condition,
        device=device,
        dtype=dtype,
        injection_layers=parse_injection_layers(args.injection_layers),
        adaptor_branches=args.adaptor_branches,
        memory_dim=memory_dim,
        architecture=args.architecture,
        reader_type=args.reader_type,
        generator_cue_source=args.generator_cue_source,
        generator_num_latents=args.generator_num_latents,
        generator_hidden_size=args.generator_hidden_size,
        generator_layers=args.generator_layers,
        generator_heads=args.generator_heads,
        generator_cue_window=args.generator_cue_window,
        generator_fusion_type=args.generator_fusion_type,
        generator_adaptive_router=args.generator_adaptive_router,
        generator_router_hidden_size=args.generator_router_hidden_size,
        generator_router_semantic_size=args.generator_router_semantic_size,
        generator_router_expert_mode=args.generator_router_expert_mode,
        generator_source_adapter_rank=args.generator_source_adapter_rank,
        generator_loop_rounds=args.generator_loop_rounds,
        generator_loop_workspace_size=args.generator_loop_workspace_size,
        generator_loop_gate_max=args.generator_loop_gate_max,
    )

    if args.advantage_reader is not None:
        configure_advantage_reader(wrapper, args.advantage_reader)

    if args.init_adaptor is not None:
        init_params = load_initial_adaptor(wrapper, args.init_adaptor)
        print(f"Warm-started adaptor from {args.init_adaptor} ({init_params:,} params)")

    if args.deterministic_engram_init_seed is not None:
        initialized = initialize_direct_engram_readers(
            wrapper, args.deterministic_engram_init_seed
        )
        print(
            "Deterministically initialized direct Engram readers: "
            + ", ".join(initialized)
        )

    joint_reader_params = None
    joint_router_params = None
    if args.joint_tri_route_only:
        trainable_names, joint_reader_params, joint_router_params = (
            configure_joint_tri_reader_training(wrapper, route_only=True)
        )
        print(
            "Jointly trainable E/GE/GH route-only MoE tensors:\n  "
            + "\n  ".join(trainable_names)
        )
    elif args.joint_tri_reader:
        trainable_names, joint_reader_params, joint_router_params = (
            configure_joint_tri_reader_training(wrapper)
        )
        print(
            "Jointly trainable E/GE/GH/Reader tensors:\n  "
            + "\n  ".join(trainable_names)
        )
    elif args.joint_tri_subset_reader:
        trainable_names, joint_reader_params, joint_router_params = (
            configure_joint_tri_reader_training(wrapper, unified_subset_reader=True)
        )
        print(
            "Jointly trainable E/GE/GH/unified-subset-Reader tensors:\n  "
            + "\n  ".join(trainable_names)
        )
    elif args.joint_engram_generated_router or args.joint_source_reader:
        trainable_names, joint_reader_params, joint_router_params = (
            configure_joint_engram_generated_router_training(
                wrapper,
                router_init_alpha=args.router_init_alpha,
            )
        )
        print(
            "Jointly trainable direct Engram/generated/router tensors:\n  "
            + "\n  ".join(trainable_names)
        )
    elif args.train_generated_branch_only:
        if args.init_adaptor is None:
            raise ValueError("--train-generated-branch-only requires --init-adaptor")
        if args.architecture != "generative" or args.generator_fusion_type != "dual_reader":
            raise ValueError(
                "--train-generated-branch-only requires architecture=generative "
                "and generator_fusion_type=dual_reader"
            )
        trainable_names = configure_generated_residual_training(
            wrapper, args.dual_reader_mode
        )
        print("Frozen imported Engram reader; generated tensors:\n  " + "\n  ".join(trainable_names))

    # Freeze backbone always
    wrapper.freeze_backbone()

    # Enable gradient checkpointing if requested (saves memory for large models)
    if args.gradient_checkpointing:
        enable_frozen_backbone_gradient_checkpointing(wrapper)
        print("Gradient checkpointing enabled with differentiable frozen inputs")

    # For train_from_scratch, memory is trainable
    if args.condition == "train_from_scratch":
        wrapper.unfreeze_memory()
    elif memory is not None:
        wrapper.freeze_memory()

    # Build canonicalization only when an Engram memory is actually present.
    canonicalizer = None
    if memory is not None:
        canonicalizer = build_canonicalizer(tokenizer, mode=args.canon_mode, max_ngram=mem_cfg.max_ngram)

    # For vocab mode, build ID map; for word_boundary mode, build hasher
    canon_id_map = None
    word_boundary_canon = None
    word_ngram_hasher = None
    if canonicalizer is not None and args.canon_mode == "vocab" and hasattr(canonicalizer, "build_id_map"):
        canon_id_map = canonicalizer.build_id_map(tokenizer).to(device)
    elif canonicalizer is not None and args.canon_mode == "word_boundary" and isinstance(canonicalizer, WordBoundaryCanonicalizer):
        word_boundary_canon = canonicalizer
        word_ngram_hasher = WordNgramHasher(mem_cfg.hash_config)

    def set_memory_context(wrapper, input_ids):
        """Set canonical IDs or hash indices for the current batch."""
        if memory is None:
            return
        if canon_id_map is not None:
            canon_ids = canon_id_map[input_ids]
            wrapper.set_canon_ids(canon_ids)
        elif word_boundary_canon is not None and word_ngram_hasher is not None:
            word_ngrams = word_boundary_canon.compute_word_ngrams(input_ids)
            indices = word_ngram_hasher.hash_word_ngrams(word_ngrams, device=input_ids.device)
            wrapper.set_hash_indices(indices)

    # Data
    corpus_label = (
        f"{args.corpus}[{args.corpus_subset}]"
        if args.corpus == "nemotron-cc"
        else args.corpus
    )
    print(f"Loading {corpus_label}...")
    if args.corpus == "wikipedia-2021" and args.wikipedia2021_require_tokenizer_match:
        print("Wikipedia-2021 tokenizer alignment: STRICT")
    train_loader = get_dataloader(
        split="train",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        shuffle=True,
        seed=args.seed,
        corpus=args.corpus,
        corpus_subset=args.corpus_subset,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )
    val_loader = get_dataloader(
        split="validation",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=args.validation_max_tokens,
        shuffle=False,
        corpus=args.corpus,
        corpus_subset=args.corpus_subset,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )

    def evaluate_validation_ppl(mode: str | None = None) -> float:
        if mode is not None:
            _set_joint_reader_mode(wrapper, mode)
        wrapper.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_ids = val_batch["input_ids"].to(device)
                val_labels = val_batch["labels"].to(device)
                set_memory_context(wrapper, val_ids)
                val_out = wrapper(
                    input_ids=val_ids,
                    labels=val_labels,
                    use_cache=False,
                )
                val_losses.append(float(val_out.loss.item()))
        if not val_losses:
            raise RuntimeError("Validation loader produced no batches")
        return math.exp(sum(val_losses) / len(val_losses))
    test_loader = None
    if not args.skip_final_test_eval or args.condition == "baseline":
        test_loader = get_dataloader(
            split="test",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_tokens=2_000_000,  # Cap test set: nemotron-cc streaming has no None-handling
            shuffle=False,
            corpus=args.corpus,
            corpus_subset=args.corpus_subset,
            wikipedia2021_dataset=args.wikipedia2021_dataset,
            wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
            wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
        )

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    total_steps = args.max_tokens // tokens_per_step
    print(f"Total steps: {total_steps:,} (effective batch = {args.batch_size * args.grad_accum_steps})")

    # Optimizer
    trainable_params = wrapper.get_trainable_params()
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_trainable:,}")

    if total_trainable == 0 and args.condition == "baseline":
        # Baseline: just evaluate
        print("\nBaseline condition - evaluating without training...")
        wrapper.eval()
        test_ppls = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test eval"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = wrapper(input_ids=input_ids, labels=labels)
                ppl = float(torch.exp(outputs.loss).item())
                test_ppls.append(ppl)

        mean_ppl, ci_lower, ci_upper = bootstrap_ci(test_ppls)
        results = {
            "completed": True,
            "condition": args.condition,
            "seed": args.seed,
            "test_ppl_mean": mean_ppl,
            "test_ppl_ci_lower": ci_lower,
            "test_ppl_ci_upper": ci_upper,
            "test_ppl_std": float(torch.tensor(test_ppls).std().item()),
            "n_test_batches": len(test_ppls),
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test PPL: {mean_ppl:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")
        wrapper.cleanup()
        return

    if (
        args.joint_engram_generated_router
        or args.joint_source_reader
        or args.joint_tri_reader
        or args.joint_tri_subset_reader
        or args.joint_tri_route_only
    ):
        optimizer = torch.optim.AdamW(
            [
                {"params": joint_reader_params, "lr": args.lr},
                {"params": joint_router_params, "lr": args.router_lr},
            ],
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # Training loop
    gate_analyzer = GateAnalyzer()
    log_file = open(output_dir / "train_log.jsonl", "w")
    val_ppls_over_time = []
    step = 0
    epoch = 0
    start_time = time.time()

    # Early stopping state
    best_val_ppl = float("inf")
    best_step = 0
    patience_counter = 0
    patience = args.early_stopping_patience  # 0 = disabled

    print(f"\nTraining adaptor for {total_steps} steps...")

    accum_steps = args.grad_accum_steps
    micro_step = 0
    running_loss = 0.0
    running_lm_loss = 0.0
    running_residual_penalty = 0.0
    running_router_distillation = 0.0

    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Set canonical IDs / hash indices for memory lookup
            set_memory_context(wrapper, input_ids)

            reader_mode = None
            if args.joint_engram_generated_router:
                reader_mode = (
                    "both"
                    if step * tokens_per_step < args.router_start_tokens
                    else "routed"
                )
                _set_joint_reader_mode(wrapper, reader_mode)
            path_losses = None
            oracle_target_usage = None
            if args.joint_tri_route_only:
                # Fast MoE-style training: the adaptor computes E, GE, and GH
                # once inside a single routed forward.  The soft mixture lets
                # all experts receive gradients; final validation switches to
                # hard per-token argmax routing.
                _set_joint_reader_mode(wrapper, "tri_soft_fused")
                routed_outputs = wrapper(
                    input_ids=input_ids,
                    labels=labels,
                    use_cache=False,
                )
                routed_lm_loss = routed_outputs.loss
                load_balance = tri_router_load_balance_loss(wrapper)
                objective = (
                    routed_lm_loss
                    + args.tri_router_load_balance_weight * load_balance
                )
                (objective / accum_steps).backward()
                path_losses = {"tri_soft_fused": float(routed_lm_loss.item())}
                reader_mode = "tri_route_only"
                running_loss += float(objective.item()) / accum_steps
                running_lm_loss += float(routed_lm_loss.item()) / accum_steps
                running_router_distillation += (
                    float(load_balance.item()) / accum_steps
                )
            elif args.joint_tri_reader or args.joint_tri_subset_reader:
                # All three experts and the Reader are active from token zero.
                # Forced-path losses prevent a weak source from being starved
                # by the learned Reader during early training.  The unified
                # variant additionally trains all seven subset endpoints and
                # then distills the best per-token endpoint into its subset
                # head.
                if args.joint_tri_subset_reader:
                    source_weights = {
                        "engram_only": args.engram_source_loss_weight,
                        "generated_from_engram_only": args.ge_source_loss_weight,
                        "generated_from_context_only": args.gh_source_loss_weight,
                        "e_ge": args.routed_source_loss_weight,
                        "e_gh": args.routed_source_loss_weight,
                        "ge_gh": args.routed_source_loss_weight,
                        "tri_soft_fused": args.routed_source_loss_weight,
                        "tri_subset_soft_fused": args.subset_routed_source_loss_weight,
                    }
                else:
                    source_weights = {
                        "engram_only": args.engram_source_loss_weight,
                        "generated_from_engram_only": args.ge_source_loss_weight,
                        "generated_from_context_only": args.gh_source_loss_weight,
                        "tri_routed": args.routed_source_loss_weight,
                    }
                normalizer = sum(source_weights.values())
                path_losses = {}
                objective_value = 0.0
                lm_value = 0.0
                distillation_value = 0.0
                endpoint_token_nlls = []
                oracle_target_usage = None
                for source_mode, source_weight in source_weights.items():
                    _set_joint_reader_mode(wrapper, source_mode)
                    source_outputs = wrapper(
                        input_ids=input_ids,
                        labels=labels,
                        use_cache=False,
                    )
                    source_lm_loss = source_outputs.loss
                    source_distillation = source_lm_loss.new_zeros(())
                    if source_mode not in {
                        "tri_routed",
                        "tri_subset_routed",
                        "tri_subset_soft_fused",
                    }:
                        endpoint_token_nlls.append(
                            per_token_next_nll(source_outputs.logits, labels)
                        )
                    elif (
                        source_mode == "tri_routed"
                        and args.tri_router_distillation_weight
                    ):
                        source_distillation, oracle_target_usage = (
                            tri_reader_distillation_loss(
                                wrapper, endpoint_token_nlls, labels
                            )
                        )
                    elif (
                        source_mode in {
                            "tri_subset_routed",
                            "tri_subset_soft_fused",
                        }
                        and args.tri_router_distillation_weight
                    ):
                        source_distillation, oracle_target_usage = (
                            tri_subset_reader_distillation_loss(
                                wrapper, endpoint_token_nlls, labels
                            )
                        )
                    (
                        source_objective,
                        weighted_lm,
                        distillation_objective,
                    ) = tri_joint_loss_terms(
                        source_lm_loss,
                        source_distillation,
                        source_weight,
                        normalizer,
                        args.tri_router_distillation_weight,
                    )
                    # The Reader target is deliberately independent of the
                    # endpoint-path averaging scale.  In the seven-way setup,
                    # the old expression silently changed weight 0.2 into
                    # 0.025 and allowed the hard Reader to stick to E.
                    (source_objective / accum_steps).backward()
                    path_losses[source_mode] = float(source_lm_loss.item())
                    objective_value += float(source_objective.item())
                    lm_value += float(weighted_lm.item())
                    distillation_value += float(distillation_objective.item())
                reader_mode = (
                    "tri_subset_joint"
                    if args.joint_tri_subset_reader
                    else "tri_source_joint"
                )
                running_loss += objective_value / accum_steps
                running_lm_loss += lm_value / accum_steps
                running_router_distillation += distillation_value / accum_steps
            elif args.joint_source_reader:
                # Every source receives its own causal-LM loss on every batch,
                # so a weak initial router cannot starve either expert.  The
                # routed path is active from token zero and learns jointly.
                source_weights = {
                    "engram_only": args.engram_source_loss_weight,
                    "generated_only": args.generated_source_loss_weight,
                    "routed": args.routed_source_loss_weight,
                }
                normalizer = sum(source_weights.values())
                path_losses = {}
                objective_value = 0.0
                lm_value = 0.0
                penalty_value = 0.0
                for source_mode, source_weight in source_weights.items():
                    _set_joint_reader_mode(wrapper, source_mode)
                    source_outputs = wrapper(
                        input_ids=input_ids,
                        labels=labels,
                        use_cache=False,
                    )
                    source_lm_loss = source_outputs.loss
                    source_penalty = source_lm_loss.new_zeros(())
                    if source_mode in {"generated_only", "routed"}:
                        source_penalty = (
                            args.generated_residual_penalty
                            * generated_residual_energy(wrapper)
                        )
                    source_objective = source_lm_loss + source_penalty
                    scale = source_weight / normalizer
                    (scale * source_objective / accum_steps).backward()
                    path_losses[source_mode] = float(source_lm_loss.item())
                    objective_value += scale * float(source_objective.item())
                    lm_value += scale * float(source_lm_loss.item())
                    penalty_value += scale * float(source_penalty.item())
                reader_mode = "source_joint"
                running_loss += objective_value / accum_steps
                running_lm_loss += lm_value / accum_steps
                running_residual_penalty += penalty_value / accum_steps
            else:
                # Forward + backward (accumulate gradients)
                outputs = wrapper(input_ids=input_ids, labels=labels, use_cache=False)
                lm_loss = outputs.loss
                residual_penalty = lm_loss.new_zeros(())
                if args.joint_engram_generated_router:
                    residual_penalty = (
                        args.generated_residual_penalty
                        * generated_residual_energy(wrapper)
                    )
                objective = lm_loss + residual_penalty
                loss = objective / accum_steps
                loss.backward()

                running_loss += loss.item()
                running_lm_loss += float(lm_loss.item()) / accum_steps
                running_residual_penalty += float(residual_penalty.item()) / accum_steps
            micro_step += 1

            if micro_step % accum_steps != 0:
                continue  # Accumulate more before optimizer step

            # Optimizer step (every accum_steps micro-batches).  Gradient
            # diagnostics are only emitted on log steps; computing them on
            # every update performs many small GPU-to-CPU synchronizations
            # without affecting training.
            collect_grad_diagnostics = (step + 1) % args.log_every == 0
            pre_step_grad_norms = (
                wrapper.get_grad_norms() if collect_grad_diagnostics else None
            )
            pre_step_tri_grad_norms = (
                tri_reader_gradient_norms(wrapper)
                if collect_grad_diagnostics
                else None
            )
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # Record gate values
            gate_vals = wrapper.get_last_gate_values()
            if gate_vals is not None and step % args.log_every == 0:
                gate_analyzer.record(gate_vals)

            # Log
            if step % args.log_every == 0:
                grad_norms = pre_step_grad_norms
                elapsed = time.time() - start_time
                tri_diagnostics = (
                    tri_reader_diagnostics(wrapper)
                    if (
                        args.joint_tri_reader
                        or args.joint_tri_subset_reader
                        or args.joint_tri_route_only
                    )
                    else None
                )

                log_entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(running_loss),
                    "lm_loss": float(running_lm_loss),
                    "generated_residual_penalty": float(running_residual_penalty),
                    "router_distillation_loss": float(running_router_distillation),
                    "oracle_target_usage": (
                        oracle_target_usage
                        if (
                            args.joint_tri_reader
                            or args.joint_tri_subset_reader
                            or args.joint_tri_route_only
                        )
                        else None
                    ),
                    "ppl": float(math.exp(min(running_loss, 20))),
                    "lr": [float(value) for value in scheduler.get_last_lr()],
                    "reader_mode": reader_mode,
                    "training_reader_mode": args.training_reader_mode,
                    "deployment_reader_mode": args.deployment_reader_mode,
                    "checkpoint_selection_reader_mode": checkpoint_selection_mode,
                    "effective_reader_mode": (
                        "tri_soft_fused"
                        if args.joint_tri_route_only
                        else args.training_reader_mode
                    ),
                    "path_lm_losses": path_losses,
                    "routed_lm_loss": (
                        (
                            path_losses.get(
                                "tri_subset_soft_fused"
                                if args.joint_tri_subset_reader
                                else "tri_soft_fused"
                                if args.joint_tri_route_only
                                else "tri_routed"
                            )
                        )
                        if (
                            args.joint_tri_reader
                            or args.joint_tri_subset_reader
                            or args.joint_tri_route_only
                        )
                        and path_losses
                        else None
                    ),
                    "router_weights": (
                        joint_router_weights(wrapper)
                        if (
                            args.joint_engram_generated_router
                            or args.joint_tri_reader
                            or args.joint_tri_subset_reader
                            or args.joint_tri_route_only
                        )
                        else None
                    ),
                    "router_subset_weights": (
                        joint_subset_router_weights(wrapper)
                        if args.joint_tri_subset_reader
                        else None
                    ),
                    "router_source_usage": (
                        tri_diagnostics["usage"] if tri_diagnostics else None
                    ),
                    "router_entropy": (
                        tri_diagnostics["entropy"] if tri_diagnostics else None
                    ),
                    "router_normalized_entropy": (
                        tri_diagnostics["normalized_entropy"]
                        if tri_diagnostics
                        else None
                    ),
                    "router_collapse": (
                        tri_diagnostics["collapse"] if tri_diagnostics else None
                    ),
                    "router_subset_usage": (
                        tri_diagnostics.get("subset_usage")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_entropy": (
                        tri_diagnostics.get("subset_entropy")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_normalized_entropy": (
                        tri_diagnostics.get("subset_normalized_entropy")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_collapse": (
                        tri_diagnostics.get("subset_collapse")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_singleton_mass": (
                        tri_diagnostics.get("subset_singleton_mass")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_pair_mass": (
                        tri_diagnostics.get("subset_pair_mass")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_triple_mass": (
                        tri_diagnostics.get("subset_triple_mass")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_multi_expert_mass": (
                        tri_diagnostics.get("subset_multi_expert_mass")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_active_classes": (
                        tri_diagnostics.get("subset_active_classes")
                        if tri_diagnostics
                        else None
                    ),
                    "router_subset_combination_collapse": (
                        tri_diagnostics.get("subset_combination_collapse")
                        if tri_diagnostics
                        else None
                    ),
                    "grad_norm_backbone": grad_norms["backbone"],
                    "grad_norm_adaptor": grad_norms["adaptor"],
                    "grad_norm_memory": grad_norms["memory"],
                    "tri_grad_norms": pre_step_tri_grad_norms,
                    "elapsed_s": elapsed,
                    "tokens_seen": step * tokens_per_step,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

                print(
                    f"  Step {step}/{total_steps} | "
                    f"Loss {running_loss:.4f} | "
                    f"PPL {math.exp(min(running_loss, 20)):.2f} | "
                    f"Mode {reader_mode or 'engram_only'} | "
                    f"Backbone grad: {grad_norms['backbone']:.6f}"
                )

            running_loss = 0.0
            running_lm_loss = 0.0
            running_residual_penalty = 0.0
            running_router_distillation = 0.0

            # Eval
            if step % args.eval_every == 0:
                val_mode = (
                    checkpoint_selection_mode
                    if checkpoint_selection_mode is not None
                    else (
                        "routed"
                        if args.joint_source_reader
                        else reader_mode if args.joint_engram_generated_router else None
                    )
                )
                val_ppl = evaluate_validation_ppl(val_mode)
                mean_val_loss = math.log(val_ppl)
                val_ppls_over_time.append({"step": step, "val_ppl": val_ppl})
                improved = val_ppl < best_val_ppl

                # Track best and early stopping
                if improved:
                    best_val_ppl = val_ppl
                    best_step = step
                    patience_counter = 0
                    # Save best adaptor checkpoint
                    if wrapper.adaptor is not None:
                        torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor_best.pt")
                    print(f"  >> Val PPL: {val_ppl:.2f} (new best)")
                else:
                    patience_counter += 1
                    print(f"  >> Val PPL: {val_ppl:.2f} (no improvement, patience {patience_counter}/{patience})")

                val_entry = {
                    "step": step,
                    "val_loss": mean_val_loss,
                    "val_ppl": val_ppl,
                    "type": "eval",
                    "reader_mode": val_mode,
                    "reader_mode_provenance": "checkpoint_selection_trained_mode",
                }
                log_file.write(json.dumps(val_entry) + "\n")
                log_file.flush()
                wrapper.train()

                if patience > 0 and patience_counter >= patience:
                    print(f"\nEarly stopping at step {step} (best val PPL {best_val_ppl:.2f} at step {best_step})")
                    break

        if patience > 0 and patience_counter >= patience:
            break

    log_file.close()

    # Always restore the validation-selected checkpoint for downstream tasks.
    best_ckpt = output_dir / "adaptor_best.pt"
    if best_ckpt.exists() and wrapper.adaptor is not None:
        wrapper.adaptor.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print(f"Restored best adaptor from step {best_step}")
    if args.deployment_reader_mode is not None:
        _set_joint_reader_mode(wrapper, args.deployment_reader_mode)

    actual_steps = step
    if args.skip_final_test_eval:
        gate_stats = gate_analyzer.compute_stats()
        elapsed = time.time() - start_time
        validation = {}
        if args.joint_tri_subset_reader:
            validation_modes = (
                "engram_only",
                "generated_from_engram_only",
                "generated_from_context_only",
                "e_ge",
                "e_gh",
                "ge_gh",
                "tri_soft_fused",
                "tri_subset_routed",
                "tri_subset_soft_fused",
            )
        elif args.joint_tri_route_only:
            validation_modes = (
                "engram_only",
                "generated_from_engram_only",
                "generated_from_context_only",
                "tri_routed",
            )
        elif args.joint_tri_reader:
            validation_modes = (
                "engram_only",
                "generated_from_engram_only",
                "generated_from_context_only",
                "tri_routed",
            )
        elif args.joint_source_reader:
            validation_modes = ("engram_only", "generated_only", "both", "routed")
        elif args.joint_engram_generated_router:
            validation_modes = ("engram_only", "both", "routed")
        else:
            validation_modes = ("engram_only",)
        for mode in validation_modes:
            validation[mode] = {"ppl": evaluate_validation_ppl(
                mode
                if (
                    args.joint_engram_generated_router
                    or args.joint_source_reader
                    or args.joint_tri_reader
                    or args.joint_tri_subset_reader
                    or args.joint_tri_route_only
                )
                else None
            )}
            print(
                f"FINAL Wikipedia validation {mode}: "
                f"PPL={validation[mode]['ppl']:.4f}"
            )

        results = {
            "completed": True,
            "condition": args.condition,
            "seed": args.seed,
            "target_model": args.target_model,
            "source_memory": args.source_memory,
            "memory_config": args.memory_config,
            "actual_steps": actual_steps,
            "best_step": best_step,
            "best_val_ppl": None if best_val_ppl == float("inf") else best_val_ppl,
            "test_ppl_mean": None,
            "test_ppl_ci_lower": None,
            "test_ppl_ci_upper": None,
            "test_ppl_std": None,
            "n_test_batches": 0,
            "gate_stats": gate_stats,
            "elapsed_hours": elapsed / 3600,
            "skipped_final_test_eval": True,
            "training_design": (
                "token_zero_joint_E_GE_GH_route_only_moe"
                if args.joint_tri_route_only
                else (
                    "token_zero_joint_E_GE_GH_tri_reader"
                    if args.joint_tri_reader
                    else (
                    "token_zero_joint_E_GE_GH_unified_seven_way_subset_reader"
                    if args.joint_tri_subset_reader
                    else (
                    "token_zero_joint_engram_generated_source_reader"
                    if args.joint_source_reader
                    else (
                        "single_stage_joint_engram_generated_router"
                        if args.joint_engram_generated_router
                        else "matched_single_stage_engram_only"
                    )
                    )
                    )
                )
            ),
            "training_data": "wikipedia-2021-causal-next-token-only",
            "max_tokens": args.max_tokens,
            "validation": validation,
            "trainable_params": total_trainable,
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        if wrapper.adaptor is not None:
            torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor.pt")
        if args.condition == "train_from_scratch" and wrapper.memory is not None:
            torch.save(wrapper.memory.state_dict(), output_dir / "memory.pt")

        print("\nSkipped final test evaluation; saved adaptor checkpoint for downstream eval.")
        print("ATHENA_FAIR_TARGET_FITTING_COMPLETE")
        wrapper.cleanup()
        return

    # Final test evaluation
    print("\nRunning final test evaluation...")
    wrapper.eval()
    test_ppls = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test eval"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            set_memory_context(wrapper, input_ids)
            outputs = wrapper(input_ids=input_ids, labels=labels)
            ppl = float(torch.exp(outputs.loss).item())
            test_ppls.append(ppl)

    mean_ppl, ci_lower, ci_upper = bootstrap_ci(test_ppls)
    print(f"Test PPL: {mean_ppl:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Save results
    gate_stats = gate_analyzer.compute_stats()
    elapsed = time.time() - start_time

    results = {
        "condition": args.condition,
        "seed": args.seed,
        "target_model": args.target_model,
        "corpus": args.corpus,
        "corpus_subset": args.corpus_subset,
        "wikipedia2021_dataset": args.wikipedia2021_dataset,
        "wikipedia2021_source_tokenizer": args.wikipedia2021_source_tokenizer,
        "wikipedia2021_require_tokenizer_match": args.wikipedia2021_require_tokenizer_match,
        "max_tokens": args.max_tokens,
        "total_steps": actual_steps,
        "best_step": best_step if patience > 0 else actual_steps,
        "early_stopped": patience > 0 and patience_counter >= patience,
        "trainable_params": total_trainable,
        "test_ppl_mean": mean_ppl,
        "test_ppl_ci_lower": ci_lower,
        "test_ppl_ci_upper": ci_upper,
        "test_ppl_std": float(torch.tensor(test_ppls).std().item()),
        "n_test_batches": len(test_ppls),
        "gate_stats": gate_stats,
        "val_ppls": val_ppls_over_time,
        "elapsed_hours": elapsed / 3600,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save adaptor checkpoint
    if wrapper.adaptor is not None:
        torch.save(wrapper.adaptor.state_dict(), output_dir / "adaptor.pt")
    if args.condition == "train_from_scratch" and wrapper.memory is not None:
        torch.save(wrapper.memory.state_dict(), output_dir / "memory.pt")

    # Save per-batch test PPLs for bootstrap analysis
    with open(output_dir / "test_ppls.json", "w") as f:
        json.dump(test_ppls, f)

    print(f"\nDone! Total time: {elapsed/3600:.2f}h")
    print(f"Results saved to: {output_dir}")

    wrapper.cleanup()


if __name__ == "__main__":
    main()
