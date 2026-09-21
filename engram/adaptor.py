"""Engram adaptor variants for all experimental conditions.

Supports:
  - linear (full): W_K + W_V + sigmoid gate (main claim)
  - no_gate: W_V only, alpha=1 always
  - affine_stitch: W_aff + bias, no gate (Chen et al. baseline)
  - ffn_only: 2-layer MLP residual, no memory (extra-params control)
  - memory_only: zero-parameter direct memory injection (strict Engram-only)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class EngramAdaptor(nn.Module):
    """Full adaptor: W_K + W_V projections with sigmoid gating.

    Gate: alpha_t = sigmoid(RMSNorm(h_t)^T . RMSNorm(k_t) / sqrt(d_model) + bias)
    Output: alpha_t * v_t added to residual stream

    The gate bias is initialized to a negative value (default -2.0) so that
    the gate starts near sigmoid(-2)=0.12, preventing large memory contributions
    from destabilizing training before the adaptor has learned useful projections.
    """

    def __init__(self, d_model: int, d_mem: int, gate_bias_init: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.w_k = nn.Linear(d_mem, d_model, bias=False)
        self.w_v = nn.Linear(d_mem, d_model, bias=False)
        self.norm_h = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.gate_bias = nn.Parameter(torch.tensor(gate_bias_init))

        nn.init.normal_(self.w_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)

    def forward(
        self, h: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, T, d_model) hidden states from backbone
            mem: (B, T, d_mem) memory vectors

        Returns:
            (contribution, gate_values): contribution to add to residual,
                gate values for logging
        """
        k = self.w_k(mem)    # (B, T, d_model)
        v = self.w_v(mem)    # (B, T, d_model)
        gate_logit = (self.norm_h(h) * self.norm_k(k)).sum(dim=-1) / math.sqrt(self.d_model)
        gate = torch.sigmoid(gate_logit + self.gate_bias)  # (B, T)
        contribution = gate.unsqueeze(-1) * v
        return contribution, gate

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MultiBranchEngramAdaptor(nn.Module):
    """Shared-value Engram adaptor with branch-specific key/gating paths."""

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        num_branches: int = 4,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_branches = num_branches
        self.w_v = nn.Linear(d_mem, d_model, bias=False)
        self.w_k = nn.ModuleList(
            [nn.Linear(d_mem, d_model, bias=False) for _ in range(num_branches)]
        )
        self.norm_h = RMSNorm(d_model)
        self.norm_k = nn.ModuleList([RMSNorm(d_model) for _ in range(num_branches)])
        self.gate_bias = nn.Parameter(torch.full((num_branches,), gate_bias_init))

        nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)
        for proj in self.w_k:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    def forward(
        self, h: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.w_v(mem)
        norm_h = self.norm_h(h)
        contributions = []
        gates = []

        for branch_idx, (proj_k, norm_k) in enumerate(zip(self.w_k, self.norm_k)):
            k = proj_k(mem)
            gate_logit = (norm_h * norm_k(k)).sum(dim=-1) / math.sqrt(self.d_model)
            gate = torch.sigmoid(gate_logit + self.gate_bias[branch_idx])
            contributions.append(gate.unsqueeze(-1) * v)
            gates.append(gate)

        contribution = torch.stack(contributions, dim=0).mean(dim=0)
        gate_values = torch.stack(gates, dim=0)
        return contribution, gate_values

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NoGateAdaptor(nn.Module):
    """Adaptor without gating: W_V only, alpha=1 always.

    Tests whether the sigmoid gate adds value.
    """

    def __init__(self, d_model: int, d_mem: int):
        super().__init__()
        self.w_v = nn.Linear(d_mem, d_model, bias=False)
        nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)

    def forward(
        self, h: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.w_v(mem)
        gate = torch.ones(h.shape[0], h.shape[1], device=h.device)
        return v, gate

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AffineStitchAdaptor(nn.Module):
    """Affine map baseline (Chen et al. arXiv:2506.06609).

    W_aff @ mem + bias, added directly to residual. No gating.
    """

    def __init__(self, d_model: int, d_mem: int):
        super().__init__()
        self.linear = nn.Linear(d_mem, d_model, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, h: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contribution = self.linear(mem)
        gate = torch.ones(h.shape[0], h.shape[1], device=h.device)
        return contribution, gate

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FFNOnlyAdaptor(nn.Module):
    """2-layer MLP residual (no memory). Extra-params attribution control.

    MLP: Linear(d_model, d_hidden) -> SiLU -> Linear(d_hidden, d_model)
    d_hidden chosen so total params ~= full EngramAdaptor params.
    """

    def __init__(self, d_model: int, target_param_count: int, match_biases: bool = False):
        super().__init__()
        # d_hidden = floor(target_params / (2 * d_model))
        d_hidden = target_param_count // (2 * d_model)
        d_hidden = max(d_hidden, 16)  # minimum hidden dim
        if match_biases:
            d_hidden = max(1, round((target_param_count - d_model) / (2 * d_model + 1)))

        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model),
        )

        # Small init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, h: torch.Tensor, mem: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mem is ignored — this adaptor doesn't use memory."""
        contribution = self.net(h)
        gate = torch.ones(h.shape[0], h.shape[1], device=h.device)
        return contribution, gate

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MemoryOnlyAdaptor(nn.Module):
    """Zero-parameter direct memory injection.

    Requires d_mem == d_model. The memory vector is RMS-normalized with no
    learned affine parameters and added directly to the residual stream.
    """

    def __init__(self, d_model: int, d_mem: int, eps: float = 1e-6):
        super().__init__()
        if d_mem != d_model:
            raise ValueError(
                f"memory_only requires d_mem == d_model, got d_mem={d_mem}, d_model={d_model}"
            )
        self.d_model = d_model
        self.eps = eps

    def forward(
        self, h: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rms = mem.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        contribution = mem / rms
        gate = torch.ones(h.shape[0], h.shape[1], device=h.device)
        return contribution, gate

    @property
    def trainable_param_count(self) -> int:
        return 0


def build_adaptor(
    condition: str,
    d_model: int,
    d_mem: int,
    gate_bias_init: float = 0.0,
    num_branches: int = 1,
    architecture: str = "legacy",
    reader_type: str = "cross_attention",
    generator_cue_source: str = "engram",
    generator_num_latents: int = 4,
    generator_hidden_size: int = 256,
    generator_layers: int = 2,
    generator_heads: int = 4,
    generator_cue_window: int = 3,
    generator_fusion_type: str = "generated_only",
    generator_adaptive_router: bool = False,
    generator_router_hidden_size: int = 16,
    generator_router_semantic_size: int = 0,
    generator_router_expert_mode: str = "residual",
    generator_source_adapter_rank: int = 16,
    generator_loop_rounds: int = 1,
    generator_loop_workspace_size: int = 0,
    generator_loop_gate_max: float = 0.25,
) -> nn.Module:
    """Factory function for adaptor variants.

    Args:
        condition: One of the experimental conditions
        d_model: Target model hidden dimension
        d_mem: Memory vector dimension
        gate_bias_init: Initial value for gate bias parameter

    Returns:
        Adaptor module
    """
    if architecture not in ("legacy", "generative"):
        raise ValueError(f"Unknown architecture: {architecture}")
    if condition == "baseline":
        return None
    if condition == "ffn_only" and architecture == "generative" and generator_fusion_type == "tri_reader":
        # Match all parameters of the full three-source adaptor plus advantage
        # head; construct on meta so this does not allocate a second model.
        import inspect
        values = locals().copy()
        kwargs = {k: values[k] for k in inspect.signature(build_adaptor).parameters}
        kwargs["condition"] = "transferred"
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            reference = build_adaptor(**kwargs)
            reference.configure_advantage_reader(candidates="sources")
            target = sum(p.numel() for p in reference.parameters())
        ffn = FFNOnlyAdaptor(d_model=d_model, target_param_count=target, match_biases=True)
        ffn.matched_tri_parameter_count = target
        return ffn
    if architecture == "generative" and condition not in ("ffn_only", "memory_only"):
        if generator_fusion_type == "tri_reader":
            from .tri_memory import TriMemoryAdaptor

            if generator_cue_source != "hybrid":
                raise ValueError(
                    "tri_reader fusion requires generator_cue_source='hybrid'"
                )
            tri_class = TriMemoryAdaptor
            extra = {}
            if condition in ("no_gate", "affine_stitch"):
                from .tri_ablation import AblatedTriMemoryAdaptor
                tri_class = AblatedTriMemoryAdaptor
                extra["condition"] = condition
            return tri_class(
                **extra,
                d_model=d_model,
                d_mem=d_mem,
                reader_type=reader_type,
                num_latents=generator_num_latents,
                hidden_size=generator_hidden_size,
                num_layers=generator_layers,
                num_heads=generator_heads,
                cue_window=generator_cue_window,
                gate_bias_init=gate_bias_init,
                num_branches=num_branches,
                adaptive_router=generator_adaptive_router,
                router_hidden_size=generator_router_hidden_size,
                router_semantic_size=generator_router_semantic_size,
                source_adapter_rank=generator_source_adapter_rank,
                generator_loop_rounds=generator_loop_rounds,
                generator_loop_workspace_size=generator_loop_workspace_size,
                generator_loop_gate_max=generator_loop_gate_max,
            )
        # Local import avoids a module-level cycle because generative_memory
        # reuses RMSNorm from this module.
        from .generative_memory import GenerativeMemoryAdaptor

        return GenerativeMemoryAdaptor(
            d_model=d_model,
            d_mem=d_mem,
            reader_type=reader_type,
            cue_source=generator_cue_source,
            num_latents=generator_num_latents,
            hidden_size=generator_hidden_size,
            num_layers=generator_layers,
            num_heads=generator_heads,
            cue_window=generator_cue_window,
            gate_bias_init=gate_bias_init,
            num_branches=num_branches,
            fusion_type=generator_fusion_type,
            adaptive_router=generator_adaptive_router,
            router_hidden_size=generator_router_hidden_size,
            router_semantic_size=generator_router_semantic_size,
            router_expert_mode=generator_router_expert_mode,
            source_adapter_rank=generator_source_adapter_rank,
            generator_loop_rounds=generator_loop_rounds,
            generator_loop_workspace_size=generator_loop_workspace_size,
            generator_loop_gate_max=generator_loop_gate_max,
        )
    if condition in ("transferred", "random_memory", "permuted_keys", "train_from_scratch"):
        if num_branches > 1:
            return MultiBranchEngramAdaptor(
                d_model=d_model,
                d_mem=d_mem,
                num_branches=num_branches,
                gate_bias_init=gate_bias_init,
            )
        return EngramAdaptor(d_model=d_model, d_mem=d_mem, gate_bias_init=gate_bias_init)
    elif condition == "memory_only":
        return MemoryOnlyAdaptor(d_model=d_model, d_mem=d_mem)
    elif condition == "no_gate":
        return NoGateAdaptor(d_model=d_model, d_mem=d_mem)
    elif condition == "affine_stitch":
        return AffineStitchAdaptor(d_model=d_model, d_mem=d_mem)
    elif condition == "ffn_only":
        # Target params: ~same as full EngramAdaptor
        target = 2 * d_mem * d_model + 2 * d_model  # W_K + W_V + norms
        return FFNOnlyAdaptor(d_model=d_model, target_param_count=target)
    else:
        raise ValueError(f"Unknown condition: {condition}")
