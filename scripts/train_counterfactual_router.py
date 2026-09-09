"""Distill a task-agnostic Engram/Both selector from raw Wikipedia spans.

The two memory experts stay frozen.  For every Wikipedia sequence, the script
measures which expert assigns higher likelihood to the real future tokens and
uses that counterfactual advantage as a soft teacher for a semantic router.
No QA examples, prompts, answers, or downstream validation labels are used.
"""

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.backbone_wrapper import BackboneWrapper
from engram.data import get_dataloader
from engram.memory import EngramMemory, MemoryConfig
from scripts.eval_openqa import build_canon_fn
from scripts.train_adaptor import parse_injection_layers
from scripts.train_advantage_router import (
    ROUTER_MODES,
    build_wrapper as build_dual_wrapper,
    configure_router_training,
    evaluate_ppl,
    get_cosine_schedule,
    load_source_config,
    save_checkpoint,
    set_reader_mode as set_dual_reader_mode,
)
from engram.tri_memory import TriMemoryAdaptor


def _adaptors(wrapper: BackboneWrapper) -> list[torch.nn.Module]:
    """Return adaptor modules for both legacy dual and tri wrappers."""
    if wrapper.adaptor is None:
        return []
    modules = list(wrapper.adaptor) if isinstance(wrapper.adaptor, torch.nn.ModuleList) else [wrapper.adaptor]
    if not modules:
        raise TypeError("Wrapper has no adaptor modules")
    return modules


def router_weight_stats(wrapper: BackboneWrapper) -> list[float]:
    """Summarize the latest selector weights for dual or tri readers."""
    values = []
    for adaptor in _adaptors(wrapper):
        if isinstance(adaptor, TriMemoryAdaptor):
            weights = adaptor.get_last_advantage_weights()
            if weights is None:
                weights = adaptor.get_last_router_weights()
            if weights is not None:
                values.append(
                    weights.detach().float().reshape(-1, weights.shape[-1]).mean(dim=0)
                )
        else:
            weights = adaptor.get_last_router_weights()
            if weights is not None:
                values.append(
                    weights.detach().float().reshape(-1, weights.shape[-1]).mean(dim=0)
                )
    if not values:
        return [0.0, 0.0]
    return [float(value) for value in torch.stack(values).mean(dim=0).cpu()]


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("span lengths must be positive integers")
    return values


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(not 0.5 <= value < 1.0 for value in values):
        raise argparse.ArgumentTypeError(
            "router thresholds must be comma-separated values in [0.5, 1.0)"
        )
    return values


def parse_advantage_threshold_list(raw: str) -> tuple[float, ...]:
    """Parse finite loss-gain thresholds for the E-anchored tri Reader."""
    try:
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "advantage thresholds must be comma-separated finite numbers"
        ) from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(
            "advantage thresholds must be comma-separated finite numbers"
        )
    return values


def get_calibration_thresholds(args, *, tri: bool) -> tuple[float, ...]:
    """Return the validation grid for the selected reader contract."""
    name = "advantage_thresholds" if tri else "router_thresholds"
    thresholds = getattr(args, name, None)
    if thresholds is None:
        raise AttributeError(f"Missing calibration argument: {name}")
    values = tuple(float(value) for value in thresholds)
    if not values:
        raise ValueError(f"{name} must contain at least one threshold")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--adaptor-dir", required=True)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary"
    )
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
    parser.add_argument("--validation-max-tokens", type=int, default=200_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--router-hidden-size", type=int, default=64)
    parser.add_argument("--router-semantic-size", type=int, default=64)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument(
        "--span-lengths", type=parse_int_list, default=parse_int_list("1,4,8,16,32")
    )
    parser.add_argument("--teacher-temperature", type=float, default=0.15)
    parser.add_argument("--advantage-weight-floor", type=float, default=0.01)
    parser.add_argument("--advantage-weight-cap", type=float, default=2.0)
    parser.add_argument(
        "--router-thresholds",
        type=parse_float_list,
        default=parse_float_list("0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"),
    )
    parser.add_argument(
        "--advantage-thresholds",
        type=parse_advantage_threshold_list,
        default=parse_advantage_threshold_list(
            "0.00,0.01,0.02,0.05,0.10,0.20,0.30,0.50"
        ),
        help=(
            "Wikipedia validation grid for predicted loss advantage. "
            "This is separate from the legacy binary-router probability grid."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", choices=["wikipedia-2021"], default="wikipedia-2021")
    parser.add_argument("--wikipedia2021-dataset", default=None)
    parser.add_argument("--wikipedia2021-source-tokenizer", default=None)
    parser.add_argument("--wikipedia2021-require-tokenizer-match", action="store_true")
    parser.add_argument(
        "--candidate-space",
        choices=["auto", "sources", "subsets"],
        default="auto",
        help="Tri-reader candidates: E/GE/GH or all seven non-empty subsets.",
    )
    parser.add_argument("--advantage-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--advantage-max-scale", type=float, default=1.0)
    parser.add_argument("--advantage-regression-weight", type=float, default=1.0)
    parser.add_argument("--advantage-confidence-weight", type=float, default=0.25)
    return parser.parse_args()


TRI_SOURCE_MODES = (
    "engram_only",
    "generated_from_engram_only",
    "generated_from_context_only",
)
TRI_SUBSET_MODES = TRI_SOURCE_MODES + (
    "e_ge",
    "e_gh",
    "ge_gh",
    "tri_soft_fused",
)


def is_tri_wrapper(wrapper: BackboneWrapper) -> bool:
    return any(isinstance(adaptor, TriMemoryAdaptor) for adaptor in _adaptors(wrapper))


def tri_candidate_modes(candidate_space: str) -> tuple[str, ...]:
    if candidate_space == "sources":
        return TRI_SOURCE_MODES
    if candidate_space == "subsets":
        return TRI_SUBSET_MODES
    raise ValueError(f"Unknown tri candidate space: {candidate_space}")


def _load_tri_experts(adaptor: torch.nn.Module, checkpoint: str) -> list[str]:
    """Load a frozen tri expert checkpoint, allowing only the lazy head keys."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = adaptor.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected tri expert checkpoint tensors: {unexpected}")
    # A multi-injection wrapper stores adaptor keys under a numeric ModuleList
    # prefix (for example ``0.advantage_router.0.weight``).  The advantage
    # head is lazy, so a source expert checkpoint legitimately omits it in
    # either the single-adaptor or prefixed form.
    def is_lazy_advantage_key(name: str) -> bool:
        return name == "advantage_router" or ".advantage_router." in name or name.startswith(
            "advantage_router."
        )

    if missing and not all(is_lazy_advantage_key(name) for name in missing):
        raise ValueError(
            "Tri expert checkpoint may omit only newly initialized advantage head "
            f"tensors; missing={missing}"
        )
    return missing


def _is_tri_config(source_config: dict) -> bool:
    return (
        source_config.get("generator_fusion_type") == "tri_reader"
        or source_config.get("fusion_type") == "tri_reader"
        or any(
            bool(source_config.get(key))
            for key in ("joint_tri_reader", "joint_tri_subset_reader", "joint_tri_route_only")
        )
    )


def _build_tri_wrapper(args, device: torch.device, dtype: torch.dtype):
    source_config = load_source_config(args.adaptor_dir)
    with open(args.memory_config) as handle:
        memory_config_dict = json.load(handle)
    memory_config = MemoryConfig(
        max_ngram=memory_config_dict["max_ngram"],
        heads_per_order=memory_config_dict["heads_per_order"],
        table_size=memory_config_dict["table_size"],
        d_head=memory_config_dict["d_head"],
        hash_seed=memory_config_dict.get("hash_seed", 42),
    )
    memory = EngramMemory(memory_config)
    memory.load_state_dict(torch.load(args.source_memory, map_location="cpu", weights_only=True))
    for parameter in memory.parameters():
        parameter.requires_grad = False

    def cfg(name, default):
        value = source_config.get(name, default)
        return default if value is None else value

    wrapper = BackboneWrapper(
        model_name=args.target_model,
        memory=memory,
        condition="transferred",
        device=device,
        dtype=dtype,
        injection_layers=parse_injection_layers(source_config.get("injection_layers")),
        adaptor_branches=int(cfg("adaptor_branches", 1)),
        memory_dim=memory_config.d_mem,
        architecture="generative",
        reader_type=cfg("reader_type", "cross_attention"),
        generator_cue_source="hybrid",
        generator_num_latents=int(cfg("generator_num_latents", 4)),
        generator_hidden_size=int(cfg("generator_hidden_size", 256)),
        generator_layers=int(cfg("generator_layers", 2)),
        generator_heads=int(cfg("generator_heads", 4)),
        generator_cue_window=int(cfg("generator_cue_window", 3)),
        generator_fusion_type="tri_reader",
        generator_adaptive_router=True,
        generator_router_hidden_size=int(cfg("generator_router_hidden_size", args.router_hidden_size)),
        generator_router_semantic_size=int(cfg("generator_router_semantic_size", args.router_semantic_size)),
        generator_source_adapter_rank=int(cfg("generator_source_adapter_rank", 16)),
        generator_loop_rounds=int(cfg("generator_loop_rounds", 1)),
        generator_loop_workspace_size=int(cfg("generator_loop_workspace_size", 0)),
        generator_loop_gate_max=float(cfg("generator_loop_gate_max", 0.25)),
    )
    adaptors = _adaptors(wrapper)
    candidate_space = args.candidate_space
    if candidate_space == "auto":
        candidate_space = (
            source_config.get("advantage_reader", {}).get("candidates", "sources")
            if isinstance(source_config.get("advantage_reader"), dict)
            else "sources"
        )
    checkpoint = args.adaptor_checkpoint or str(Path(args.adaptor_dir) / "adaptor_best.pt")
    for adaptor in adaptors:
        # Pair/triple candidates must use the same endpoint contract as the
        # source checkpoint at deployment.  In particular, ``router_hard``
        # changes pair candidates while the named triple endpoint remains
        # soft; restoring it here keeps the Wikipedia teacher aligned with
        # inference rather than silently using constructor defaults.
        adaptor.configure_router(
            temperature=float(source_config.get("router_temperature", 1.0)),
            hard=bool(source_config.get("router_hard", False)),
            min_generated_probability=float(
                source_config.get("router_min_generated_probability", 0.5)
            ),
            safe_residual_threshold=float(
                source_config.get("safe_residual_threshold", 1.0)
            ),
            safe_residual_scale=float(
                source_config.get("safe_residual_scale", 1.0)
            ),
        )
        adaptor.configure_advantage_reader(
            candidates=candidate_space,
            threshold=args.advantage_threshold,
            confidence_threshold=args.confidence_threshold,
            temperature=args.teacher_temperature,
            max_scale=args.advantage_max_scale,
        )
    missing = _load_tri_experts(wrapper.adaptor, checkpoint)
    wrapper.freeze_backbone()
    wrapper.freeze_memory()
    set_canon_fn = build_canon_fn(wrapper, memory_config_dict, args.canon_mode, device)
    return wrapper, set_canon_fn, source_config, checkpoint, missing, candidate_space


def build_wrapper(args, device: torch.device, dtype: torch.dtype):
    """Dispatch to the legacy dual builder or the tri-aware builder."""
    source_config = load_source_config(args.adaptor_dir)
    if _is_tri_config(source_config):
        return _build_tri_wrapper(args, device, dtype)
    result = build_dual_wrapper(args, device, dtype)
    return result


def set_reader_mode(wrapper: BackboneWrapper, mode: str) -> None:
    """Set a reader mode for either the legacy dual or tri adaptor."""
    if is_tri_wrapper(wrapper):
        for adaptor in _adaptors(wrapper):
            adaptor.set_tri_reader_mode(mode)
    else:
        set_dual_reader_mode(wrapper, mode)


def gold_token_log_probs(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shifted gold-token log probabilities and their valid mask."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected logits (B,T,V) and labels (B,T), got {logits.shape}, {labels.shape}"
        )
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    log_probs = shifted_logits.log_softmax(dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return log_probs.masked_fill(~valid, 0.0), valid


def future_span_advantage(
    token_advantage: torch.Tensor,
    valid: torch.Tensor,
    span_lengths: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average future-token advantages over several causal span horizons."""
    if valid.ndim != 2 or token_advantage.ndim not in (2, 3):
        raise ValueError("expected token_advantage with shape (B,T) or (B,T,C)")
    if token_advantage.shape[:2] != valid.shape:
        raise ValueError(
            "token_advantage and valid must share their batch/time shape"
        )
    if not span_lengths or any(length < 1 for length in span_lengths):
        raise ValueError("span_lengths must contain positive integers")

    # The tri-reader teacher carries one advantage per candidate in a trailing
    # dimension.  Broadcast the causal validity mask across that dimension
    # while keeping the public returned mask at (B,T), as both dual and tri
    # callers use it to mask token positions.
    valid_float = valid.to(token_advantage.dtype)
    if token_advantage.ndim == 3:
        valid_float = valid_float.unsqueeze(-1)
    pad_time = (1, 0) if token_advantage.ndim == 2 else (0, 0, 1, 0)
    advantage_cumsum = F.pad(
        (token_advantage * valid_float).cumsum(dim=1), pad_time
    )
    count_cumsum = F.pad(valid_float.cumsum(dim=1), pad_time)
    positions = torch.arange(token_advantage.shape[1], device=token_advantage.device)
    span_sum = torch.zeros_like(token_advantage)
    span_count = torch.zeros_like(token_advantage)
    for length in span_lengths:
        ends = (positions + length).clamp(max=token_advantage.shape[1])
        sums = advantage_cumsum[:, ends] - advantage_cumsum[:, positions]
        counts = count_cumsum[:, ends] - count_cumsum[:, positions]
        horizon_valid = counts.gt(0)
        span_sum += (sums / counts.clamp_min(1.0)) * horizon_valid
        span_count += horizon_valid.to(span_count.dtype)
    if token_advantage.ndim == 3:
        span_valid = span_count[..., 0].gt(0) & valid
        fill_mask = ~span_valid.unsqueeze(-1)
    else:
        span_valid = span_count.gt(0) & valid
        fill_mask = ~span_valid
    span_advantage = span_sum / span_count.clamp_min(1.0)
    return span_advantage.masked_fill(fill_mask, 0.0), span_valid


@torch.no_grad()
def expert_token_advantage(
    wrapper: BackboneWrapper,
    set_canon_fn,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log p_Both(gold) - log p_Engram(gold) without gradients."""
    log_probs = {}
    valid_masks = {}
    for mode in ROUTER_MODES:
        set_reader_mode(wrapper, mode)
        set_canon_fn(input_ids)
        outputs = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        log_probs[mode], valid_masks[mode] = gold_token_log_probs(outputs.logits, labels)
        del outputs
    if not torch.equal(valid_masks["engram_only"], valid_masks["both"]):
        raise RuntimeError("Engram and Both teacher masks differ")
    return log_probs["both"] - log_probs["engram_only"], valid_masks["engram_only"]


@torch.no_grad()
def tri_expert_token_advantage(
    wrapper: BackboneWrapper,
    set_canon_fn,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    candidate_space: str = "sources",
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure every tri-reader candidate against the frozen E endpoint.

    The returned tensor is ``(B, T-1, C)`` and uses the same causal shift as
    :func:`gold_token_log_probs`.  Only the Wikipedia sequence is used as the
    source of labels; candidate selection is never exposed to the router.
    """
    if not is_tri_wrapper(wrapper):
        raise TypeError("tri_expert_token_advantage requires a TriMemoryAdaptor")
    modes = tri_candidate_modes(candidate_space)
    log_probs = []
    valid_masks = []
    for mode in modes:
        for adaptor in _adaptors(wrapper):
            adaptor.set_tri_reader_mode(mode)
        set_canon_fn(input_ids)
        outputs = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        candidate_log_probs, candidate_valid = gold_token_log_probs(
            outputs.logits, labels
        )
        log_probs.append(candidate_log_probs)
        valid_masks.append(candidate_valid)
        del outputs
    reference_mask = valid_masks[0]
    if any(not torch.equal(reference_mask, mask) for mask in valid_masks[1:]):
        raise RuntimeError("Tri-reader candidate teacher masks differ")
    stacked = torch.stack(log_probs, dim=-1)
    return stacked - stacked[..., :1], reference_mask


def set_tri_advantage_mode(wrapper: BackboneWrapper, supervision_only: bool) -> None:
    for adaptor in _adaptors(wrapper):
        if not isinstance(adaptor, TriMemoryAdaptor):
            raise TypeError("Advantage mode requires TriMemoryAdaptor instances")
        adaptor.set_tri_reader_mode("tri_advantage_routed")
        adaptor.set_advantage_supervision_only(supervision_only)


def tri_advantage_predictions(wrapper: BackboneWrapper) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    predictions = []
    confidence = []
    for adaptor in _adaptors(wrapper):
        values = adaptor.get_last_advantage_predictions()
        scores = adaptor.get_last_advantage_confidence_logits()
        if values is None or scores is None:
            raise RuntimeError("Advantage probe forward did not expose predictions")
        predictions.append(values)
        confidence.append(scores)
    return predictions, confidence


def tri_counterfactual_distillation_loss(
    wrapper: BackboneWrapper,
    span_advantage: torch.Tensor,
    valid: torch.Tensor,
    *,
    teacher_temperature: float,
    weight_floor: float,
    weight_cap: float,
    regression_weight: float = 1.0,
    confidence_weight: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    """Train the E-anchored gain and confidence heads.

    Candidate zero is the fixed E anchor.  Small measured gains contribute
    less to the regression, while confidence learns whether a candidate is a
    useful positive replacement.  This keeps ``a=0`` an exact deployment
    fallback and does not train the frozen experts.
    """
    if span_advantage.ndim != 3 or valid.ndim != 2:
        raise ValueError("Expected span advantage (B,T,C) and valid (B,T)")
    if span_advantage.shape[:2] != valid.shape or span_advantage.shape[-1] < 2:
        raise ValueError("Tri advantage candidate shape mismatch")
    if teacher_temperature <= 0 or regression_weight < 0 or confidence_weight < 0:
        raise ValueError("Invalid tri advantage loss parameters")
    weights = span_advantage[..., 1:].abs().clamp(
        min=weight_floor, max=weight_cap
    ) * valid.unsqueeze(-1).to(span_advantage.dtype)
    denominator = weights.sum().clamp_min(torch.finfo(span_advantage.dtype).eps)
    target_advantage = span_advantage[..., 1:].clamp(-weight_cap, weight_cap)
    target_confidence = torch.sigmoid(
        (span_advantage[..., 1:] - span_advantage[..., :1]) / teacher_temperature
    )
    losses = []
    predicted = []
    for predictions, confidence in zip(*tri_advantage_predictions(wrapper)):
        predictions = predictions[:, :-1]
        confidence = confidence[:, :-1]
        if predictions.shape != span_advantage.shape:
            raise RuntimeError(
                f"Tri advantage/head alignment mismatch: {predictions.shape} vs {span_advantage.shape}"
            )
        predicted_advantage = predictions[..., 1:]
        predicted_confidence = confidence[..., 1:]
        regression = F.smooth_l1_loss(
            predicted_advantage.float(), target_advantage.float(), reduction="none"
        )
        confidence_loss = F.binary_cross_entropy_with_logits(
            predicted_confidence.float(), target_confidence.float(), reduction="none"
        )
        losses.append(
            regression_weight * (regression * weights).sum() / denominator
            + confidence_weight * (confidence_loss * weights).sum() / denominator
        )
        predicted.append(predicted_advantage.detach())
    mean_prediction = torch.stack(predicted).mean(dim=0)
    candidate_valid = valid.unsqueeze(-1).expand_as(mean_prediction)
    predicted_useful = mean_prediction.gt(0)
    teacher_useful = span_advantage[..., 1:].gt(0)
    valid_count = candidate_valid.sum().clamp_min(1)
    metrics = {
        "teacher_useful_rate": float((teacher_useful & candidate_valid).sum() / valid_count),
        "predicted_useful_rate": float((predicted_useful & candidate_valid).sum() / valid_count),
        "sign_accuracy": float(
            ((predicted_useful == teacher_useful) & candidate_valid).sum() / valid_count
        ),
        "mean_predicted_advantage": float(mean_prediction.masked_select(candidate_valid).mean()),
        "mean_teacher_advantage": float(span_advantage[..., 1:].masked_select(candidate_valid).mean()),
    }
    return torch.stack(losses).mean(), metrics


def set_router_supervision_only(wrapper: BackboneWrapper, enabled: bool) -> None:
    for adaptor in _adaptors(wrapper):
        adaptor.set_router_supervision_only(enabled)


def initialize_distillation_router(wrapper: BackboneWrapper) -> None:
    """Start the selector uncommitted; deployment safety comes from thresholding."""
    with torch.no_grad():
        for adaptor in _adaptors(wrapper):
            adaptor.router[-1].weight.zero_()
            adaptor.router[-1].bias.zero_()


def router_logits(wrapper: BackboneWrapper) -> list[torch.Tensor]:
    values = []
    for adaptor in _adaptors(wrapper):
        logits = adaptor.get_last_router_logits()
        if logits is None:
            raise RuntimeError("Router probe forward did not expose logits")
        values.append(logits)
    return values


def run_router_probe(
    wrapper: BackboneWrapper,
    set_canon_fn,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> None:
    set_reader_mode(wrapper, "routed")
    set_router_supervision_only(wrapper, True)
    set_canon_fn(input_ids)
    outputs = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    del outputs


def counterfactual_distillation_loss(
    wrapper: BackboneWrapper,
    span_advantage: torch.Tensor,
    valid: torch.Tensor,
    *,
    router_temperature: float,
    teacher_temperature: float,
    weight_floor: float,
    weight_cap: float,
) -> tuple[torch.Tensor, dict]:
    """Supervise every injection-layer router with the same final-LM teacher."""
    if teacher_temperature <= 0 or router_temperature <= 0:
        raise ValueError("router and teacher temperatures must be positive")
    if weight_floor < 0 or weight_cap <= 0 or weight_floor > weight_cap:
        raise ValueError("invalid advantage weight bounds")

    target_probability = torch.sigmoid(span_advantage / teacher_temperature)
    weights = span_advantage.abs().clamp(min=weight_floor, max=weight_cap)
    weights = weights * valid.to(weights.dtype)
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    losses = []
    probabilities = []
    for logits in router_logits(wrapper):
        score = (logits[..., 1] - logits[..., 0])[:, :-1].float()
        if score.shape != span_advantage.shape:
            raise RuntimeError(
                f"Router/teacher alignment mismatch: {score.shape} vs {span_advantage.shape}"
            )
        token_loss = F.binary_cross_entropy_with_logits(
            score / router_temperature,
            target_probability,
            reduction="none",
        )
        losses.append((token_loss * weights).sum() / denominator)
        probabilities.append(torch.sigmoid(score / router_temperature).detach())

    probability = torch.stack(probabilities).mean(dim=0)
    valid_count = valid.sum().clamp_min(1)
    predicted_both = probability.ge(0.5)
    teacher_both = span_advantage.gt(0)
    metrics = {
        "teacher_both_rate": float((teacher_both & valid).sum() / valid_count),
        "predicted_both_rate": float((predicted_both & valid).sum() / valid_count),
        "mean_both_probability": float(
            probability.masked_select(valid).mean().item()
        ),
        "sign_accuracy": float(
            ((predicted_both == teacher_both) & valid).sum() / valid_count
        ),
        "mean_span_advantage": float(
            span_advantage.masked_select(valid).mean().item()
        ),
    }
    return torch.stack(losses).mean(), metrics


def configure_inference_router(
    wrapper: BackboneWrapper,
    *,
    temperature: float,
    threshold: float,
) -> None:
    set_router_supervision_only(wrapper, False)
    for adaptor in _adaptors(wrapper):
        adaptor.configure_router(
            temperature=temperature,
            hard=True,
            min_generated_probability=threshold,
        )


@torch.no_grad()
def calibrate_counterfactual_router(
    wrapper: BackboneWrapper,
    set_canon_fn,
    loader,
    device: torch.device,
    *,
    span_lengths: tuple[int, ...],
    router_temperature: float,
    teacher_temperature: float,
    weight_floor: float,
    weight_cap: float,
    thresholds: tuple[float, ...],
) -> dict:
    """Select a conservative threshold using held-out Wikipedia only."""
    wrapper.eval()
    totals = {
        threshold: {"selected": 0, "helpful": 0, "advantage_sum": 0.0}
        for threshold in thresholds
    }
    loss_sum = 0.0
    batches = 0
    valid_predictions = 0
    teacher_both = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        token_advantage, token_valid = expert_token_advantage(
            wrapper, set_canon_fn, input_ids, labels, attention_mask
        )
        span_advantage, valid = future_span_advantage(
            token_advantage, token_valid, span_lengths
        )
        run_router_probe(wrapper, set_canon_fn, input_ids, attention_mask)
        loss, _ = counterfactual_distillation_loss(
            wrapper,
            span_advantage,
            valid,
            router_temperature=router_temperature,
            teacher_temperature=teacher_temperature,
            weight_floor=weight_floor,
            weight_cap=weight_cap,
        )
        loss_sum += float(loss.item())
        batches += 1
        valid_predictions += int(valid.sum().item()) * len(router_logits(wrapper))
        teacher_both += int((span_advantage.gt(0) & valid).sum().item()) * len(
            router_logits(wrapper)
        )
        for logits in router_logits(wrapper):
            probability = torch.softmax(
                logits[:, :-1].float() / router_temperature, dim=-1
            )[..., 1]
            for threshold in thresholds:
                selected = probability.ge(threshold) & valid
                totals[threshold]["selected"] += int(selected.sum().item())
                totals[threshold]["helpful"] += int(
                    (selected & span_advantage.gt(0)).sum().item()
                )
                totals[threshold]["advantage_sum"] += float(
                    span_advantage.masked_select(selected).sum().item()
                )

    if not batches or not valid_predictions:
        raise RuntimeError("Wikipedia validation produced no valid predictions")
    candidates = []
    for threshold in thresholds:
        selected = totals[threshold]["selected"]
        candidates.append({
            "threshold": threshold,
            "selection_rate": selected / valid_predictions,
            "helpful_precision": (
                totals[threshold]["helpful"] / selected if selected else 1.0
            ),
            "estimated_advantage_per_token": (
                totals[threshold]["advantage_sum"] / valid_predictions
            ),
        })
    best = max(
        candidates,
        key=lambda row: (row["estimated_advantage_per_token"], row["helpful_precision"]),
    )
    return {
        "distillation_loss": loss_sum / batches,
        "teacher_both_rate": teacher_both / valid_predictions,
        "selected_threshold": best["threshold"],
        "estimated_advantage_per_token": best["estimated_advantage_per_token"],
        "thresholds": candidates,
    }


@torch.no_grad()
def calibrate_tri_advantage_router(
    wrapper: BackboneWrapper,
    set_canon_fn,
    loader,
    device: torch.device,
    *,
    candidate_space: str,
    span_lengths: tuple[int, ...],
    teacher_temperature: float,
    weight_floor: float,
    weight_cap: float,
    regression_weight: float,
    confidence_weight: float,
    thresholds: tuple[float, ...],
    confidence_threshold: float,
) -> dict:
    """Calibrate hard E-anchored selection against actual candidate gains."""
    wrapper.eval()
    totals = {
        threshold: {"selected": 0, "helpful": 0, "advantage_sum": 0.0}
        for threshold in thresholds
    }
    loss_sum = 0.0
    batches = 0
    valid_predictions = 0
    candidate_count = len(tri_candidate_modes(candidate_space))
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        token_advantage, token_valid = tri_expert_token_advantage(
            wrapper, set_canon_fn, input_ids, labels, candidate_space, attention_mask
        )
        span_advantage, valid = future_span_advantage(
            token_advantage, token_valid, span_lengths
        )
        set_tri_advantage_mode(wrapper, True)
        set_canon_fn(input_ids)
        wrapper(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        loss, _ = tri_counterfactual_distillation_loss(
            wrapper,
            span_advantage,
            valid,
            teacher_temperature=teacher_temperature,
            weight_floor=weight_floor,
            weight_cap=weight_cap,
            regression_weight=regression_weight,
            confidence_weight=confidence_weight,
        )
        loss_sum += float(loss.item())
        batches += 1
        predictions, confidence_logits = tri_advantage_predictions(wrapper)
        for prediction, confidence_logit in zip(predictions, confidence_logits):
            prediction = prediction[:, :-1]
            confidence = torch.sigmoid(confidence_logit[:, :-1])
            valid_predictions += int(valid.sum().item())
            for threshold in thresholds:
                eligible = (
                    prediction[..., 1:].gt(threshold)
                    & confidence[..., 1:].ge(confidence_threshold)
                    & valid.unsqueeze(-1)
                )
                best = prediction[..., 1:].masked_fill(~eligible, -torch.inf).argmax(dim=-1)
                selected_mask = eligible.any(dim=-1)
                selected_advantage = span_advantage[..., 1:].gather(
                    -1, best.unsqueeze(-1)
                ).squeeze(-1)
                selected = selected_mask & valid
                totals[threshold]["selected"] += int(selected.sum().item())
                totals[threshold]["helpful"] += int(
                    (selected & selected_advantage.gt(0)).sum().item()
                )
                totals[threshold]["advantage_sum"] += float(
                    selected_advantage.masked_select(selected).sum().item()
                )
    if not batches or not valid_predictions:
        raise RuntimeError("Wikipedia validation produced no valid tri predictions")
    candidates = []
    for threshold in thresholds:
        selected = totals[threshold]["selected"]
        candidates.append({
            "threshold": threshold,
            "selection_rate": selected / valid_predictions,
            "helpful_precision": totals[threshold]["helpful"] / selected if selected else 1.0,
            "estimated_advantage_per_token": totals[threshold]["advantage_sum"] / valid_predictions,
        })
    best = max(candidates, key=lambda row: (row["estimated_advantage_per_token"], row["helpful_precision"]))
    return {
        "distillation_loss": loss_sum / batches,
        "selected_threshold": best["threshold"],
        "candidate_count": candidate_count,
        "thresholds": candidates,
    }


@torch.no_grad()
def evaluate_tri_ppl(wrapper, set_canon_fn, loader, device, modes):
    results = {}
    for mode in modes:
        set_reader_mode(wrapper, mode)
        wrapper.eval()
        losses = []
        weight_sums = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            set_canon_fn(input_ids)
            outputs = wrapper(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                use_cache=False,
            )
            losses.append(float(outputs.loss.item()))
            adaptor_weights = [
                adaptor.get_last_router_weights()
                for adaptor in _adaptors(wrapper)
                if adaptor.get_last_router_weights() is not None
            ]
            if adaptor_weights:
                weight_sums.append(
                    torch.stack(
                        [
                            weight.detach()
                            .float()
                            .reshape(-1, weight.shape[-1])
                            .mean(dim=0)
                            for weight in adaptor_weights
                        ]
                    ).mean(dim=0).cpu().tolist()
                )
        if not losses:
            raise RuntimeError("Wikipedia validation loader produced no batches")
        results[mode] = {
            "ppl": math.exp(sum(losses) / len(losses)),
            "router_weights": (
                [sum(row[i] for row in weight_sums) / len(weight_sums) for i in range(3)]
                if weight_sums else None
            ),
        }
    return results


def main_tri(args) -> None:
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    if args.max_tokens < tokens_per_step:
        raise ValueError("max_tokens is too small for one optimizer step")
    if args.teacher_temperature <= 0 or args.router_semantic_size < 1:
        raise ValueError("tri advantage training requires positive temperature and semantic size")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    wrapper, set_canon_fn, source_config, checkpoint, missing, candidate_space = build_wrapper(
        args, device, dtype
    )
    tokenizer = wrapper.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    trainable_names = []
    for index, adaptor in enumerate(_adaptors(wrapper)):
        local = adaptor.train_advantage_reader_only()
        prefix = f"{index}." if len(_adaptors(wrapper)) > 1 else ""
        trainable_names.extend(prefix + name for name in local)
    set_tri_advantage_mode(wrapper, True)
    trainable = wrapper.get_trainable_params()
    if not trainable or not all("advantage_router" in name for name in trainable_names):
        raise RuntimeError(f"Unexpected advantage-reader trainable boundary: {trainable_names}")
    if any(parameter.requires_grad for parameter in wrapper.backbone.parameters()):
        raise RuntimeError("Backbone must remain frozen")
    if any(parameter.requires_grad for parameter in wrapper.memory.parameters()):
        raise RuntimeError("Engram memory must remain frozen")
    print(f"Loaded frozen tri experts from {checkpoint}")
    print(f"Initialized advantage tensors: {len(missing)}")
    print(f"Trainable advantage parameters: {sum(p.numel() for p in trainable):,}")

    loader_kwargs = dict(
        tokenizer=tokenizer, seq_len=args.seq_len, batch_size=args.batch_size,
        seed=args.seed, corpus=args.corpus,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )
    train_loader = get_dataloader(split="train", max_tokens=args.max_tokens, shuffle=True, **loader_kwargs)
    val_loader = get_dataloader(split="validation", max_tokens=args.validation_max_tokens, shuffle=False, **loader_kwargs)
    total_steps = args.max_tokens // tokens_per_step
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps)
    runtime_config = dict(source_config)
    runtime_config.update(vars(args))
    runtime_config.update({
        "architecture": "generative",
        "generator_fusion_type": "tri_reader",
        "generator_cue_source": "hybrid",
        "advantage_reader": {
            "enabled": True, "candidates": candidate_space,
            "threshold": args.advantage_threshold,
            "confidence_threshold": args.confidence_threshold,
            "temperature": args.teacher_temperature,
            "max_scale": args.advantage_max_scale,
        },
        "deployment_reader_mode": "tri_advantage_routed",
        "router_strategy": "wikipedia_counterfactual_tri_advantage_distillation",
        "router_training_data": "wikipedia-2021-causal-next-token-only",
        "downstream_training_examples": 0,
        "source_expert_checkpoint": checkpoint,
        "experts_frozen": True,
    })
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(runtime_config, indent=2))
    log_handle = open(output_dir / "train_log.jsonl", "w")
    optimizer.zero_grad(set_to_none=True)
    step = micro_step = epoch = 0
    best_estimated_advantage = -float("inf")
    best_step = 0
    best_calibration = None
    started = time.time()
    running = {"loss": 0.0, "steps": 0}
    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            token_advantage, token_valid = tri_expert_token_advantage(
                wrapper, set_canon_fn, input_ids, labels, candidate_space, attention_mask
            )
            span_advantage, valid = future_span_advantage(token_advantage, token_valid, args.span_lengths)
            set_tri_advantage_mode(wrapper, True)
            set_canon_fn(input_ids)
            wrapper(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            loss, metrics = tri_counterfactual_distillation_loss(
                wrapper, span_advantage, valid,
                teacher_temperature=args.teacher_temperature,
                weight_floor=args.advantage_weight_floor,
                weight_cap=args.advantage_weight_cap,
                regression_weight=args.advantage_regression_weight,
                confidence_weight=args.advantage_confidence_weight,
            )
            (loss / args.grad_accum_steps).backward()
            running["loss"] += float(loss.item())
            running["steps"] += 1
            micro_step += 1
            if micro_step % args.grad_accum_steps:
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == 1:
                entry = {
                    "step": step, "epoch": epoch,
                    "distillation_loss": running["loss"] / max(1, running["steps"]),
                    "advantage_grad_norm": grad_norm, **metrics,
                    "router_weights": router_weight_stats(wrapper),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": step * tokens_per_step,
                    "teacher_forward_tokens": step * tokens_per_step * len(tri_candidate_modes(candidate_space)),
                    "elapsed_s": time.time() - started,
                }
                log_handle.write(json.dumps(entry) + "\n"); log_handle.flush()
                print(f"Step {step}/{total_steps} | distill {entry['distillation_loss']:.4f} | sign acc {entry['sign_accuracy']:.3f}")
                running = {"loss": 0.0, "steps": 0}
            if step % args.eval_every == 0 or step == total_steps:
                calibration = calibrate_tri_advantage_router(
                    wrapper, set_canon_fn, val_loader, device,
                    candidate_space=candidate_space, span_lengths=args.span_lengths,
                    teacher_temperature=args.teacher_temperature,
                    weight_floor=args.advantage_weight_floor, weight_cap=args.advantage_weight_cap,
                    regression_weight=args.advantage_regression_weight,
                    confidence_weight=args.advantage_confidence_weight,
                    thresholds=get_calibration_thresholds(args, tri=True),
                    confidence_threshold=args.confidence_threshold,
                )
                estimated = max(row["estimated_advantage_per_token"] for row in calibration["thresholds"])
                if estimated > best_estimated_advantage:
                    best_estimated_advantage = estimated; best_step = step; best_calibration = calibration
                    save_checkpoint(wrapper, output_dir, "adaptor_best.pt")
                print(f">> Wikipedia tri validation threshold={calibration['selected_threshold']:.2f}; advantage/token={estimated:.6f}")
    log_handle.close()
    if best_step == 0:
        raise RuntimeError("Tri advantage reader never produced a validation checkpoint")
    wrapper.adaptor.load_state_dict(torch.load(output_dir / "adaptor_best.pt", map_location=device, weights_only=True), strict=True)
    final_calibration = calibrate_tri_advantage_router(
        wrapper, set_canon_fn, val_loader, device,
        candidate_space=candidate_space, span_lengths=args.span_lengths,
        teacher_temperature=args.teacher_temperature,
        weight_floor=args.advantage_weight_floor, weight_cap=args.advantage_weight_cap,
        regression_weight=args.advantage_regression_weight,
        confidence_weight=args.advantage_confidence_weight,
        thresholds=get_calibration_thresholds(args, tri=True), confidence_threshold=args.confidence_threshold,
    )
    selected_threshold = float(final_calibration["selected_threshold"])
    for adaptor in _adaptors(wrapper):
        adaptor.configure_advantage_reader(
            candidates=candidate_space, threshold=selected_threshold,
            confidence_threshold=args.confidence_threshold,
            temperature=args.teacher_temperature, max_scale=args.advantage_max_scale,
        )
        adaptor.set_advantage_supervision_only(False)
    runtime_config["advantage_reader"]["threshold"] = selected_threshold
    runtime_config["best_step"] = best_step
    config_path.write_text(json.dumps(runtime_config, indent=2))
    validation = evaluate_tri_ppl(
        wrapper, set_canon_fn, val_loader, device,
        ("engram_only", "tri_soft_fused", "tri_routed", "tri_advantage_routed"),
    )
    results = {
        "completed": True, "training_design": "wikipedia_counterfactual_tri_advantage_distillation",
        "training_data": "wikipedia-2021-causal-next-token-only", "downstream_training_examples": 0,
        "max_tokens": args.max_tokens, "actual_steps": step, "best_step": best_step,
        "best_estimated_advantage_per_token": best_estimated_advantage,
        "selected_threshold": selected_threshold, "calibration": final_calibration,
        "validation": validation, "trainable_names": trainable_names,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    save_checkpoint(wrapper, output_dir, "adaptor.pt")
    print("ATHENA_WIKIPEDIA_COUNTERFACTUAL_TRI_ADVANTAGE_TRAINING_COMPLETE")
    wrapper.cleanup()


def main():
    args = parse_args()
    source_config = load_source_config(args.adaptor_dir)
    if _is_tri_config(source_config):
        return main_tri(args)
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    if args.max_tokens < tokens_per_step:
        raise ValueError("max_tokens is too small for one optimizer step")
    if args.teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")
    if args.router_semantic_size < 1:
        raise ValueError("counterfactual routing requires router_semantic_size >= 1")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}; dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    wrapper, set_canon_fn, source_config, checkpoint, missing = build_wrapper(args, device, dtype)
    tokenizer = wrapper.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable_names = configure_router_training(wrapper, args.router_temperature)
    initialize_distillation_router(wrapper)
    set_router_supervision_only(wrapper, True)
    trainable = wrapper.get_trainable_params()
    if not trainable or not all(
        ("router" in name or "loop_workspace" in name) for name in trainable_names
    ):
        raise RuntimeError(f"Unexpected router trainable boundary: {trainable_names}")
    if any(parameter.requires_grad for parameter in wrapper.backbone.parameters()):
        raise RuntimeError("Backbone must remain frozen")
    if any(parameter.requires_grad for parameter in wrapper.memory.parameters()):
        raise RuntimeError("Engram memory must remain frozen")
    print(f"Loaded frozen experts from {checkpoint}")
    print(f"Initialized router tensors: {len(missing)}")
    print(f"Trainable router parameters: {sum(p.numel() for p in trainable):,}")
    print("Trainable tensors:\n  " + "\n  ".join(trainable_names))

    loader_kwargs = dict(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        corpus=args.corpus,
        wikipedia2021_dataset=args.wikipedia2021_dataset,
        wikipedia2021_source_tokenizer=args.wikipedia2021_source_tokenizer,
        wikipedia2021_require_tokenizer_match=args.wikipedia2021_require_tokenizer_match,
    )
    train_loader = get_dataloader(
        split="train", max_tokens=args.max_tokens, shuffle=True, **loader_kwargs
    )
    val_loader = get_dataloader(
        split="validation",
        max_tokens=args.validation_max_tokens,
        shuffle=False,
        **loader_kwargs,
    )

    total_steps = args.max_tokens // tokens_per_step
    print(
        f"Wikipedia counterfactual span distillation: {args.max_tokens:,} tokens, "
        f"{total_steps:,} steps, spans={args.span_lengths}"
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps)

    runtime_config = dict(source_config)
    runtime_config.update(vars(args))
    runtime_config.update({
        "span_lengths": list(args.span_lengths),
        "router_thresholds": list(args.router_thresholds),
        "architecture": "generative",
        "generator_fusion_type": "dual_reader",
        "generator_adaptive_router": True,
        "generator_router_hidden_size": args.router_hidden_size,
        "generator_router_semantic_size": args.router_semantic_size,
        "router_experts": ["engram_only", "engram_plus_generated_residual"],
        "router_strategy": "wikipedia_counterfactual_span_advantage_distillation",
        "router_training_data": "wikipedia-2021-causal-next-token-only",
        "router_hard": True,
        "router_min_generated_probability": 0.5,
        "downstream_training_examples": 0,
        "source_expert_checkpoint": checkpoint,
    })
    config_path = output_dir / "config.json"
    with open(config_path, "w") as handle:
        json.dump(runtime_config, handle, indent=2)

    log_handle = open(output_dir / "train_log.jsonl", "w")
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    epoch = 0
    best_estimated_advantage = -float("inf")
    best_step = 0
    best_calibration = None
    started = time.time()
    running = {"loss": 0.0, "grad": 0.0, "steps": 0}
    last_metrics = {}

    # The backbone remains in deterministic eval mode.  Trainability is
    # controlled by requires_grad, and the selector itself has no dropout.
    wrapper.eval()
    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            token_advantage, token_valid = expert_token_advantage(
                wrapper, set_canon_fn, input_ids, labels, attention_mask
            )
            span_advantage, valid = future_span_advantage(
                token_advantage, token_valid, args.span_lengths
            )
            run_router_probe(wrapper, set_canon_fn, input_ids, attention_mask)
            loss, last_metrics = counterfactual_distillation_loss(
                wrapper,
                span_advantage,
                valid,
                router_temperature=args.router_temperature,
                teacher_temperature=args.teacher_temperature,
                weight_floor=args.advantage_weight_floor,
                weight_cap=args.advantage_weight_cap,
            )
            (loss / args.grad_accum_steps).backward()
            running["loss"] += float(loss.item())
            running["steps"] += 1
            micro_step += 1
            if micro_step % args.grad_accum_steps:
                continue

            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running["grad"] += grad_norm

            if step % args.log_every == 0 or step == 1:
                denominator = max(1, running["steps"])
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "distillation_loss": running["loss"] / denominator,
                    "router_grad_norm": running["grad"] / max(1, denominator),
                    **last_metrics,
                    "router_weights": router_weight_stats(wrapper),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": step * tokens_per_step,
                    "teacher_forward_tokens": step * tokens_per_step * 2,
                    "elapsed_s": time.time() - started,
                }
                log_handle.write(json.dumps(entry) + "\n")
                log_handle.flush()
                print(
                    f"Step {step}/{total_steps} | distill {entry['distillation_loss']:.4f} | "
                    f"grad {grad_norm:.4f} | teacher Both {entry['teacher_both_rate']:.3f} | "
                    f"pred Both {entry['predicted_both_rate']:.3f} | "
                    f"sign acc {entry['sign_accuracy']:.3f}"
                )
                running = {"loss": 0.0, "grad": 0.0, "steps": 0}

            if step % args.eval_every == 0 or step == total_steps:
                calibration = calibrate_counterfactual_router(
                    wrapper,
                    set_canon_fn,
                    val_loader,
                    device,
                    span_lengths=args.span_lengths,
                    router_temperature=args.router_temperature,
                    teacher_temperature=args.teacher_temperature,
                    weight_floor=args.advantage_weight_floor,
                    weight_cap=args.advantage_weight_cap,
                    thresholds=get_calibration_thresholds(args, tri=False),
                )
                estimated_advantage = calibration["estimated_advantage_per_token"]
                print(
                    f">> Wikipedia validation distill={calibration['distillation_loss']:.4f}; "
                    f"threshold={calibration['selected_threshold']:.2f}; "
                    f"estimated advantage/token={estimated_advantage:.6f}"
                )
                if estimated_advantage > best_estimated_advantage:
                    best_estimated_advantage = estimated_advantage
                    best_step = step
                    best_calibration = calibration
                    save_checkpoint(wrapper, output_dir, "adaptor_best.pt")
                    print(f">> New best counterfactual router at step {step}")
                set_router_supervision_only(wrapper, True)

    log_handle.close()
    if best_step == 0:
        raise RuntimeError("Counterfactual router never produced a validation checkpoint")
    wrapper.adaptor.load_state_dict(
        torch.load(output_dir / "adaptor_best.pt", map_location=device, weights_only=True)
    )
    best_calibration = calibrate_counterfactual_router(
        wrapper,
        set_canon_fn,
        val_loader,
        device,
        span_lengths=args.span_lengths,
        router_temperature=args.router_temperature,
        teacher_temperature=args.teacher_temperature,
        weight_floor=args.advantage_weight_floor,
        weight_cap=args.advantage_weight_cap,
        thresholds=get_calibration_thresholds(args, tri=False),
    )
    selected_threshold = float(best_calibration["selected_threshold"])
    configure_inference_router(
        wrapper,
        temperature=args.router_temperature,
        threshold=selected_threshold,
    )

    runtime_config["router_min_generated_probability"] = selected_threshold
    runtime_config["best_step"] = best_step
    with open(config_path, "w") as handle:
        json.dump(runtime_config, handle, indent=2)

    validation = {}
    for mode in (*ROUTER_MODES, "routed"):
        ppl, weights = evaluate_ppl(wrapper, set_canon_fn, val_loader, device, mode)
        validation[mode] = {
            "ppl": ppl,
            "router_weights": weights if mode == "routed" else None,
        }
        print(f"FINAL Wikipedia validation {mode}: PPL={ppl:.4f}")

    results = {
        "completed": True,
        "training_design": "wikipedia_counterfactual_span_advantage_distillation",
        "training_data": "wikipedia-2021-causal-next-token-only",
        "downstream_training_examples": 0,
        "max_tokens": args.max_tokens,
        "teacher_forward_tokens": args.max_tokens * 2,
        "actual_steps": step,
        "best_step": best_step,
        "best_estimated_advantage_per_token": best_estimated_advantage,
        "selected_threshold": selected_threshold,
        "calibration": best_calibration,
        "validation": validation,
        "trainable_names": trainable_names,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    with open(output_dir / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    save_checkpoint(wrapper, output_dir, "adaptor.pt")
    print("ATHENA_WIKIPEDIA_COUNTERFACTUAL_ROUTER_TRAINING_COMPLETE")
    wrapper.cleanup()


if __name__ == "__main__":
    main()
