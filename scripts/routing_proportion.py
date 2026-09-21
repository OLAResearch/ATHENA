"""Utilities for auditing the realised E/GE/GH route at inference time.

The training-time JSON files contain corpus-level mean router weights.  This
module records the same weights at the actual downstream decision positions so
that a report can distinguish soft source mass, hard winner counts, and exact
E-only fallback.  It deliberately does not inspect labels or alter routing.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


SOURCE_NAMES = ("E", "GE", "GH")


def _adaptors(wrapper) -> list[object]:
    adaptor = getattr(wrapper, "adaptor", None)
    if adaptor is None:
        return []
    if isinstance(adaptor, torch.nn.ModuleList):
        return list(adaptor)
    return [adaptor]


def _latest_weights(wrapper) -> torch.Tensor | None:
    """Return the latest (B,T,3) source weights, averaging injection sites."""
    values = []
    for adaptor in _adaptors(wrapper):
        getter = getattr(adaptor, "get_last_router_weights", None)
        if getter is None:
            continue
        weights = getter()
        if weights is None:
            continue
        if weights.ndim != 3 or weights.shape[-1] != 3:
            raise ValueError(
                "Expected E/GE/GH router weights with shape (B,T,3), got "
                f"{tuple(weights.shape)}"
            )
        values.append(weights.detach().float())
    if not values:
        return None
    reference = values[0].shape
    if any(value.shape != reference for value in values[1:]):
        raise ValueError("All injection sites must expose equally shaped route weights")
    return torch.stack(values, dim=0).mean(dim=0)


class RoutingProportionCollector:
    """Accumulate realised source routing statistics over selected positions."""

    def __init__(self, *, exact_e_tolerance: float = 1e-6) -> None:
        self.exact_e_tolerance = float(exact_e_tolerance)
        self._weighted_sum = torch.zeros(3, dtype=torch.float64)
        self._winner_counts = torch.zeros(3, dtype=torch.int64)
        self._generated_winner_counts = torch.zeros(2, dtype=torch.int64)
        self._positions = 0
        self._admitted = 0
        self._exact_e = 0
        self._alpha_sum = 0.0
        self._alpha_sq_sum = 0.0

    @property
    def positions(self) -> int:
        return self._positions

    def add_weights(self, weights: torch.Tensor | None, mask: torch.Tensor | None = None) -> None:
        if weights is None:
            return
        values = weights.detach().float().cpu()
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError(f"Expected (B,T,3) route weights, got {tuple(values.shape)}")
        values = values.reshape(-1, 3)
        if mask is not None:
            mask = mask.detach().bool().cpu().reshape(-1)
            if mask.numel() != values.shape[0]:
                raise ValueError("Routing mask has a different number of positions")
            values = values[mask]
        if values.numel() == 0:
            return
        values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        # Numerical noise in mixed precision should not change the reported
        # proportions.  Renormalise only when a row has positive mass.
        row_sum = values.sum(dim=-1, keepdim=True)
        values = torch.where(row_sum.gt(0), values / row_sum.clamp_min(1e-12), values)
        winner = values.argmax(dim=-1)
        winner_counts = torch.bincount(winner, minlength=3).to(torch.int64)
        alpha = (1.0 - values[:, 0]).clamp(0.0, 1.0)
        generated = alpha.gt(self.exact_e_tolerance)
        generated_winner = values[:, 1:].argmax(dim=-1)
        self._weighted_sum += values.double().sum(dim=0)
        self._winner_counts += winner_counts
        self._generated_winner_counts += torch.bincount(
            generated_winner[generated], minlength=2
        ).to(torch.int64)
        self._positions += int(values.shape[0])
        self._admitted += int(generated.sum().item())
        self._exact_e += int(alpha.le(self.exact_e_tolerance).sum().item())
        self._alpha_sum += float(alpha.double().sum().item())
        self._alpha_sq_sum += float(alpha.double().square().sum().item())

    def add_last_positions(self, wrapper, positions: torch.Tensor) -> None:
        weights = _latest_weights(wrapper)
        if weights is None:
            return
        if positions.ndim != 1 or positions.shape[0] != weights.shape[0]:
            raise ValueError("last-position index must have one entry per batch item")
        positions = positions.to(weights.device).long().clamp(0, weights.shape[1] - 1)
        rows = torch.arange(weights.shape[0], device=weights.device)
        self.add_weights(weights[rows, positions].unsqueeze(1))

    def add_last_token(self, wrapper) -> None:
        weights = _latest_weights(wrapper)
        if weights is not None:
            self.add_weights(weights[:, -1:, :])

    def add_slice(self, wrapper, start: int, end: int) -> None:
        weights = _latest_weights(wrapper)
        if weights is not None:
            self.add_weights(weights[:, start:end, :])

    def report(self) -> dict:
        count = max(self._positions, 1)
        weighted = (self._weighted_sum / count).tolist()
        winners = (self._winner_counts / count).tolist()
        generated_count = max(self._admitted, 1)
        generated_winners = (self._generated_winner_counts / generated_count).tolist()
        alpha_mean = self._alpha_sum / count
        alpha_variance = max(self._alpha_sq_sum / count - alpha_mean * alpha_mean, 0.0)
        return {
            "source_order": list(SOURCE_NAMES),
            "positions": self._positions,
            "weighted_source_mass": {
                name: float(weighted[index]) for index, name in enumerate(SOURCE_NAMES)
            },
            "argmax_source_fraction": {
                name: float(winners[index]) for index, name in enumerate(SOURCE_NAMES)
            },
            "argmax_source_counts": {
                name: int(self._winner_counts[index].item())
                for index, name in enumerate(SOURCE_NAMES)
            },
            "generated_admission_rate": self._admitted / count,
            "exact_e_fallback_rate": self._exact_e / count,
            "mean_alpha": alpha_mean,
            "std_alpha": alpha_variance**0.5,
            "generated_argmax_fraction": {
                "GE": float(generated_winners[0]),
                "GH": float(generated_winners[1]),
            },
            "generated_argmax_counts": {
                "GE": int(self._generated_winner_counts[0].item()),
                "GH": int(self._generated_winner_counts[1].item()),
            },
            "definition": {
                "positions": "model positions used to produce a downstream score or generated token",
                "weighted_source_mass": "mean E/GE/GH interpolation mass",
                "argmax_source_fraction": "fraction of positions whose largest realised mass is E, GE, or GH",
                "generated_admission_rate": "fraction with non-zero GE/GH interpolation mass",
                "exact_e_fallback_rate": "fraction with alpha <= exact_e_tolerance",
                "labels_used_for_routing": False,
                "exact_e_tolerance": self.exact_e_tolerance,
            },
        }


def collect_last_positions(wrapper, collector: RoutingProportionCollector, positions: torch.Tensor) -> None:
    """Small callback helper used by the existing evaluators."""
    collector.add_last_positions(wrapper, positions)
