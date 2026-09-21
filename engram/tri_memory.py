"""Three-source Engram/generative-memory adaptor.

The three experts are deliberately separable at runtime:

* ``engram_only`` (E): a direct Engram reader.
* ``generated_from_engram_only`` (GE): a shared generator conditioned on
  causal Engram cue windows, followed by its own generated-memory reader. GE
  is a replacement expert, not the direct E reader plus a correction.
* ``generated_from_context_only`` (GH): the same generator conditioned only
  on the causal backbone state, without looking up Engram memory.

The forced modes expose any requested non-empty subset of the three experts to
a task-agnostic Reader. ``tri_routed`` and ``tri_soft_fused`` route within all
three sources, while ``tri_subset_routed`` and ``tri_subset_soft_fused`` add a
unified seven-way subset Reader that can choose the subset itself per token.
``tri_safe_routed`` keeps the direct Engram contribution as a residual base
and only adds generated-source corrections selected by the three-way router.
``tri_random_advantage_routed`` keeps the trained E-anchored route weights but
permutes them across token positions, providing an inference-only
random-router control with matched route mass.
GE and GH share the expensive generator and reader parameters; cue
projections, source embeddings, and small low-rank output adapters preserve
source identity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptor import RMSNorm
from .generative_memory import _GeneratorLayer


# Public runtime modes.  The first three are strict single-expert paths; the
# next three are deterministic two-expert fusions; ``tri_routed`` and
# ``tri_soft_fused`` expose the learned three-way source Reader.  The final
# two expose the unified seven-way subset Reader.
TRI_READER_MODES = frozenset(
    {
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
        "e_ge",
        "e_gh",
        "ge_gh",
        "tri_routed",
        "tri_soft_fused",
        "tri_safe_routed",
        "tri_subset_routed",
        "tri_subset_soft_fused",
        "tri_advantage_routed",
        "tri_random_advantage_routed",
    }
)

TRI_READER_MODE_ALIASES = {
    # Human-facing spellings used by evaluation manifests and paper tables.
    "E": "engram_only",
    "GE": "generated_from_engram_only",
    "GH": "generated_from_context_only",
    "E+GE": "e_ge",
    "E+GH": "e_gh",
    "GE+GH": "ge_gh",
    "E+GE+GH": "tri_soft_fused",
    "e+ge": "e_ge",
    "e+gh": "e_gh",
    "ge+gh": "ge_gh",
    "e+ge+gh": "tri_soft_fused",
    "e_ge_gh": "tri_soft_fused",
    "tri_fused": "tri_soft_fused",
    "tri_soft": "tri_soft_fused",
    "tri_safe": "tri_safe_routed",
    "safe_routed": "tri_safe_routed",
    # The unified Reader chooses one of all seven non-empty subsets per token.
    "tri_subset_hard": "tri_subset_routed",
    "tri_subset_soft": "tri_subset_soft_fused",
    "tri_subset": "tri_subset_soft_fused",
    "tri_auto": "tri_subset_routed",
    "tri_auto_soft": "tri_subset_soft_fused",
    "tri_advantage": "tri_advantage_routed",
    "advantage_routed": "tri_advantage_routed",
    "random_router": "tri_random_advantage_routed",
    "tri_random_router": "tri_random_advantage_routed",
}

TRI_MODE_EXPERTS = {
    "engram_only": (0,),
    "generated_from_engram_only": (1,),
    "generated_from_context_only": (2,),
    "e_ge": (0, 1),
    "e_gh": (0, 2),
    "ge_gh": (1, 2),
    "tri_routed": (0, 1, 2),
    "tri_soft_fused": (0, 1, 2),
    "tri_safe_routed": (0, 1, 2),
    # These modes compute all three streams; the subset Reader chooses the
    # active subset independently at every token.
    "tri_subset_routed": (0, 1, 2),
    "tri_subset_soft_fused": (0, 1, 2),
    # The E-anchored advantage reader always evaluates all three standalone
    # experts before selecting a source or subset candidate.
    "tri_advantage_routed": (0, 1, 2),
    "tri_random_advantage_routed": (0, 1, 2),
}

# Candidate subsets for the unified Reader.  The order is part of the
# checkpoint/evaluation contract and follows the paper notation.
TRI_READER_SUBSETS = (
    (0,),
    (1,),
    (2,),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1, 2),
)
TRI_READER_SUBSET_NAMES = (
    "E",
    "GE",
    "GH",
    "E+GE",
    "E+GH",
    "GE+GH",
    "E+GE+GH",
)


def normalize_tri_reader_mode(mode: str) -> str:
    """Return the canonical name for a tri-reader runtime mode."""
    if not isinstance(mode, str):
        raise TypeError("Tri-reader mode must be a string")
    canonical = TRI_READER_MODE_ALIASES.get(mode, mode)
    if canonical not in TRI_READER_MODES:
        valid = ", ".join(sorted(TRI_READER_MODES))
        raise ValueError(f"Unknown tri-reader mode {mode!r}; expected one of: {valid}")
    return canonical


class TriMemoryAdaptor(nn.Module):
    """Direct Engram, Engram-conditioned, and context-conditioned experts."""

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        reader_type: str = "cross_attention",
        num_latents: int = 4,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        cue_window: int = 3,
        gate_bias_init: float = 0.0,
        num_branches: int = 1,
        adaptive_router: bool = True,
        router_hidden_size: int = 16,
        router_semantic_size: int = 0,
        source_adapter_rank: int = 16,
        generator_loop_rounds: int = 1,
        generator_loop_workspace_size: int = 0,
        generator_loop_gate_max: float = 0.25,
    ) -> None:
        super().__init__()
        if reader_type not in ("cross_attention", "mean"):
            raise ValueError(f"Unknown reader_type: {reader_type}")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if min(d_mem, num_latents, num_layers, num_heads, cue_window, num_branches) < 1:
            raise ValueError("Memory and generator dimensions must be positive")
        if router_hidden_size < 1 or router_semantic_size < 0:
            raise ValueError("Invalid router dimensions")
        if source_adapter_rank < 1:
            raise ValueError("source_adapter_rank must be positive")
        if generator_loop_rounds < 1:
            raise ValueError("generator_loop_rounds must be positive")
        if generator_loop_workspace_size < 0:
            raise ValueError("generator_loop_workspace_size must be non-negative")
        if not 0.0 <= generator_loop_gate_max <= 1.0:
            raise ValueError("generator_loop_gate_max must be in [0, 1]")

        self.d_model = d_model
        self.d_mem = d_mem
        self.reader_type = reader_type
        self.cue_source = "hybrid"
        self.cue_window = cue_window
        self.num_latents = num_latents
        self.hidden_size = hidden_size
        self.num_branches = num_branches
        self.fusion_type = "tri_reader"
        self.adaptive_router = adaptive_router
        self.router_hidden_size = router_hidden_size
        self.router_semantic_size = router_semantic_size
        self.source_adapter_rank = min(source_adapter_rank, hidden_size)
        self.generator_loop_rounds = int(generator_loop_rounds)
        self.generator_loop_workspace_size = int(
            generator_loop_workspace_size or hidden_size
        )
        self.generator_loop_gate_max = float(generator_loop_gate_max)

        self.tri_reader_mode = "tri_routed" if adaptive_router else "engram_only"
        self.router_temperature = 1.0
        self.router_hard = False
        # ``tri_safe_routed`` always keeps E and adds bounded GE/GH residuals.
        # The margin makes the zero-initialized router conservative at startup.
        self.safe_router_threshold = 1.0
        self.safe_router_scale = 1.0
        self.subset_router_temperature = 1.0
        self.subset_router_hard = False
        self._last_router_logits: torch.Tensor | None = None
        self._last_router_weights: torch.Tensor | None = None
        self._last_subset_router_logits: torch.Tensor | None = None
        self._last_subset_router_weights: torch.Tensor | None = None
        self._last_source_outputs: torch.Tensor | None = None
        self._last_loop_workspace: torch.Tensor | None = None
        self._last_loop_gate: torch.Tensor | None = None

        # The span-advantage reader is deliberately lazy.  Keeping this as
        # ``None`` until explicitly configured means legacy checkpoints,
        # state-dict keys, and constructor RNG consumption are unchanged.
        self.advantage_router: nn.Module | None = None
        self._advantage_reader_candidates: str | None = None
        self._advantage_candidate_subsets: tuple[tuple[int, ...], ...] | None = None
        self._advantage_candidate_names: tuple[str, ...] | None = None
        self._advantage_threshold = 0.0
        self._advantage_confidence_threshold = 0.5
        self._advantage_temperature = 0.15
        self._advantage_max_scale = 1.0
        self._advantage_supervision_only = False
        self._last_advantage_predictions: torch.Tensor | None = None
        self._last_advantage_confidence_logits: torch.Tensor | None = None
        self._last_advantage_weights: torch.Tensor | None = None
        self._random_router_seed = 0
        self._random_router_calls = 0

        # Direct Engram reader (E). Names intentionally mirror the legacy
        # dual reader so deterministic initialization can be copied exactly.
        self.norm_h = RMSNorm(d_model)
        self.engram_value_projection = nn.Linear(d_mem, d_model, bias=False)
        if num_branches == 1:
            self.engram_key_projection = nn.Linear(d_mem, d_model, bias=False)
            self.engram_reader_norm = RMSNorm(d_model)
            self.engram_gate_bias = nn.Parameter(torch.tensor(gate_bias_init))
        else:
            self.engram_key_projection = nn.ModuleList(
                [nn.Linear(d_mem, d_model, bias=False) for _ in range(num_branches)]
            )
            self.engram_reader_norm = nn.ModuleList(
                [RMSNorm(d_model) for _ in range(num_branches)]
            )
            self.engram_gate_bias = nn.Parameter(
                torch.full((num_branches,), gate_bias_init)
            )

        # GE and GH share the generator/reader core. Only their cue encoders,
        # source embeddings, and compact output adapters are source-specific.
        self.engram_cue_projection = nn.Linear(d_mem, hidden_size)
        self.context_cue_projection = nn.Linear(d_model, hidden_size)
        self.source_embeddings = nn.Parameter(torch.empty(2, hidden_size))
        self.latent_queries = nn.Parameter(torch.empty(num_latents, hidden_size))
        self.generator_layers = nn.ModuleList(
            [_GeneratorLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        if reader_type == "cross_attention":
            self.reader_query = nn.Linear(d_model, hidden_size, bias=False)
            self.reader_attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=0.0, batch_first=True
            )
        else:
            self.reader_query = None
            self.reader_attention = None
        self.reader_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, d_model, bias=False)
        self.source_output_down = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.source_adapter_rank, bias=False)
                for _ in range(2)
            ]
        )
        self.source_output_up = nn.ModuleList(
            [
                nn.Linear(self.source_adapter_rank, d_model, bias=False)
                for _ in range(2)
            ]
        )

        if num_branches == 1:
            self.generated_key_projection = None
            self.generated_reader_norm = RMSNorm(d_model)
            self.generated_gate_bias = nn.Parameter(
                torch.full((2,), gate_bias_init)
            )
        else:
            self.generated_key_projection = nn.ModuleList(
                [nn.Linear(d_model, d_model, bias=False) for _ in range(num_branches)]
            )
            self.generated_reader_norm = nn.ModuleList(
                [RMSNorm(d_model) for _ in range(num_branches)]
            )
            self.generated_gate_bias = nn.Parameter(
                torch.full((2, num_branches), gate_bias_init)
            )

        if adaptive_router:
            if router_semantic_size:
                self.router_semantic_projection = nn.Linear(
                    d_model, router_semantic_size, bias=False
                )
            else:
                self.router_semantic_projection = None
            # Three gate means, three log RMS values, three pairwise cosine
            # similarities, and three active-expert indicators, optionally
            # plus semantic context. The indicators make a pairwise Reader an
            # explicit masked subproblem.
            self.router = nn.Sequential(
                nn.Linear(12 + router_semantic_size, router_hidden_size),
                nn.SiLU(),
                nn.Linear(router_hidden_size, 3),
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
            # A separate head selects among the seven non-empty subsets.  Keep
            # the final layer exactly zero-initialized: hard argmax routing
            # deterministically breaks the tie in favour of the first (E)
            # subset, preserving the Engram fallback without imposing a large
            # logit margin that the six generated/pair subsets must spend most
            # of training undoing.
            self.subset_router = nn.Sequential(
                nn.Linear(12 + router_semantic_size, router_hidden_size),
                nn.SiLU(),
                nn.Linear(router_hidden_size, len(TRI_READER_SUBSETS)),
            )
            nn.init.zeros_(self.subset_router[-1].weight)
            nn.init.zeros_(self.subset_router[-1].bias)
        else:
            self.router = None
            self.subset_router = None
            self.router_semantic_projection = None

        # A small shared recurrent workspace lets the Reader refine generated
        # candidates before routing.  The modules are only materialized for
        # looped checkpoints, so legacy state dicts remain unchanged when the
        # default (one pass) path is used.
        if self.generator_loop_rounds > 1:
            workspace = self.generator_loop_workspace_size
            self.loop_workspace_update = nn.Linear(
                2 * d_model + workspace, workspace
            )
            self.loop_workspace_to_output = nn.Linear(workspace, d_model, bias=False)
            self.loop_workspace_gate = nn.Linear(workspace, 1)
            nn.init.zeros_(self.loop_workspace_to_output.weight)
            nn.init.constant_(self.loop_workspace_gate.bias, -2.0)
        else:
            self.loop_workspace_update = None
            self.loop_workspace_to_output = None
            self.loop_workspace_gate = None

        self._reset_parameters()

    def _apply_looped_reader(
        self,
        h: torch.Tensor,
        outputs: list[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        """Refine generated candidates through a shared bounded workspace.

        The direct Engram expert stays an exact fallback.  Each recurrent pass
        updates a per-token workspace from the clean hidden state and current
        candidate aggregate, then adds a bounded, zero-initialized correction
        to GE/GH.  Reusing the same modules across rounds gives true parameter
        sharing while keeping the first forward numerically identical.
        """
        if self.generator_loop_rounds <= 1 or self.loop_workspace_update is None:
            return outputs
        generated = [value for value in outputs[1:] if value is not None]
        if not generated:
            return outputs
        aggregate = torch.stack(generated, dim=0).mean(dim=0)
        batch_size, seq_len, _ = aggregate.shape
        workspace = aggregate.new_zeros(
            batch_size, seq_len, self.generator_loop_workspace_size
        )
        clean_h = self.norm_h(h)
        for _ in range(self.generator_loop_rounds):
            update_input = torch.cat((clean_h, aggregate, workspace), dim=-1)
            proposal = torch.tanh(self.loop_workspace_update(update_input))
            strength = torch.sigmoid(self.loop_workspace_gate(workspace))
            strength = strength * self.generator_loop_gate_max
            workspace = workspace + strength * proposal
            correction = self.loop_workspace_to_output(workspace)
            # Apply the shared refinement only to generated experts; E remains
            # the baseline path used by safe routing and fallback modes.
            refined = []
            for value in outputs[1:]:
                if value is not None:
                    refined.append(value + strength * correction)
                else:
                    refined.append(None)
            outputs = [outputs[0], *refined]
            aggregate = torch.stack(
                [value for value in outputs[1:] if value is not None], dim=0
            ).mean(dim=0)
        self._last_loop_workspace = workspace
        self._last_loop_gate = strength
        return outputs

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.source_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        for projection in self.source_output_up:
            nn.init.zeros_(projection.weight)
        nn.init.normal_(self.engram_value_projection.weight, mean=0.0, std=0.02)
        direct_keys = (
            self.engram_key_projection
            if isinstance(self.engram_key_projection, nn.ModuleList)
            else [self.engram_key_projection]
        )
        for projection in direct_keys:
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)
        if self.generated_key_projection is not None:
            for projection in self.generated_key_projection:
                nn.init.eye_(projection.weight)

    def active_expert_indices(self, mode: str | None = None) -> tuple[int, ...]:
        """Return active experts in ``(E, GE, GH)`` order.

        This is deliberately a public, side-effect-free description of the
        runtime path.  The backbone wrapper uses it to decide whether a table
        lookup or an Engram cue history is needed before calling the adaptor.
        """
        canonical = normalize_tri_reader_mode(mode or self.tri_reader_mode)
        return TRI_MODE_EXPERTS[canonical]

    def needs_engram_memory(self) -> bool:
        """Whether the current path is allowed to access the Engram table."""
        return bool(set(self.active_expert_indices()) & {0, 1})

    def needs_engram_cue_history(self) -> bool:
        return 1 in self.active_expert_indices()

    def _causal_windows(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build left-padded windows containing values up to the current step."""
        if values.ndim != 3:
            raise ValueError(
                f"Expected a (B, T, D) causal stream, got {tuple(values.shape)}"
            )
        batch_size, seq_len, value_size = values.shape
        left_pad = values.new_zeros(batch_size, self.cue_window - 1, value_size)
        padded = torch.cat((left_pad, values), dim=1)
        windows = padded.unfold(1, self.cue_window, 1).permute(0, 1, 3, 2)
        time = torch.arange(seq_len, device=values.device).unsqueeze(1)
        offsets = torch.arange(self.cue_window, device=values.device).unsqueeze(0)
        valid = offsets >= (self.cue_window - 1 - time)
        mask = ~valid.unsqueeze(0).expand(batch_size, -1, -1)
        return windows, mask

    def _generate(
        self,
        source_index: int,
        h: torch.Tensor,
        mem: torch.Tensor | None,
        cue_mem: torch.Tensor | None,
        context_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape
        if source_index == 0:
            if mem is None:
                raise ValueError("GE requires Engram memory cues")
            cue_context = mem if cue_mem is None else cue_mem
            if cue_context.shape[0] != batch_size or cue_context.shape[1] < seq_len:
                raise ValueError("Engram cue history does not cover the hidden sequence")
            windows, padding_mask = self._causal_windows(cue_context)
            windows = windows[:, -seq_len:]
            padding_mask = padding_mask[:, -seq_len:].reshape(
                batch_size * seq_len, self.cue_window
            )
            cues = self.engram_cue_projection(windows).reshape(
                batch_size * seq_len, self.cue_window, self.hidden_size
            )
        elif source_index == 1:
            context = h if context_h is None else context_h
            if context.shape != h.shape:
                raise ValueError(
                    "GH causal context must have the same shape as the reader "
                    f"stream: context={tuple(context.shape)}, h={tuple(h.shape)}"
                )
            windows, padding_mask = self._causal_windows(context)
            windows = windows[:, -seq_len:]
            padding_mask = padding_mask[:, -seq_len:].reshape(
                batch_size * seq_len, self.cue_window
            )
            cues = self.context_cue_projection(windows).reshape(
                batch_size * seq_len, self.cue_window, self.hidden_size
            )
        else:
            raise ValueError(f"Unknown generated source index: {source_index}")

        source_embedding = self.source_embeddings[source_index].view(1, 1, -1)
        cues = cues + source_embedding
        latents = (
            self.latent_queries.view(1, self.num_latents, self.hidden_size)
            .expand(batch_size * seq_len, -1, -1)
            + source_embedding
        )
        for layer in self.generator_layers:
            latents = layer(latents, cues, padding_mask)
        return latents

    def _read_generated(
        self, source_index: int, h: torch.Tensor, latents: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape
        if self.reader_type == "cross_attention":
            query = self.reader_query(h).reshape(
                batch_size * seq_len, 1, self.hidden_size
            )
            summary, _ = self.reader_attention(
                query, latents, latents, need_weights=False
            )
            summary = summary.squeeze(1)
        else:
            summary = latents.mean(dim=1)
        summary = self.reader_norm(summary).reshape(
            batch_size, seq_len, self.hidden_size
        )
        shared = self.output_projection(summary)
        source_delta = self.source_output_up[source_index](
            self.source_output_down[source_index](summary)
        )
        return shared + source_delta

    def _direct_engram(
        self, h: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_h = self.norm_h(h)
        value = self.engram_value_projection(mem)
        if self.num_branches == 1:
            key = self.engram_reader_norm(self.engram_key_projection(mem))
            gate = torch.sigmoid(
                (norm_h * key).sum(dim=-1) / math.sqrt(self.d_model)
                + self.engram_gate_bias
            )
            return gate.unsqueeze(-1) * value, gate.unsqueeze(0)

        contributions = []
        gates = []
        for branch_index, (projection, norm) in enumerate(
            zip(self.engram_key_projection, self.engram_reader_norm)
        ):
            key = norm(projection(mem))
            gate = torch.sigmoid(
                (norm_h * key).sum(dim=-1) / math.sqrt(self.d_model)
                + self.engram_gate_bias[branch_index]
            )
            contributions.append(gate.unsqueeze(-1) * value)
            gates.append(gate)
        return (
            torch.stack(contributions, dim=0).mean(dim=0),
            torch.stack(gates, dim=0),
        )

    def _generated_expert(
        self,
        source_index: int,
        h: torch.Tensor,
        mem: torch.Tensor | None,
        cue_mem: torch.Tensor | None,
        context_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self._generate(source_index, h, mem, cue_mem, context_h=context_h)
        # If the wrapper supplies a clean context stream, use it for both the
        # GH generator and its Reader query. This prevents an earlier E/GE
        # injection from leaking into the GH expert at a later injection layer.
        reader_h = context_h if source_index == 1 and context_h is not None else h
        value = self._read_generated(source_index, reader_h, latents)
        norm_h = self.norm_h(reader_h)
        if self.num_branches == 1:
            key = self.generated_reader_norm(value)
            gate = torch.sigmoid(
                (norm_h * key).sum(dim=-1) / math.sqrt(self.d_model)
                + self.generated_gate_bias[source_index]
            )
            return gate.unsqueeze(-1) * value, gate.unsqueeze(0)

        contributions = []
        gates = []
        for branch_index, (projection, norm) in enumerate(
            zip(self.generated_key_projection, self.generated_reader_norm)
        ):
            key = norm(projection(value))
            gate = torch.sigmoid(
                (norm_h * key).sum(dim=-1) / math.sqrt(self.d_model)
                + self.generated_gate_bias[source_index, branch_index]
            )
            contributions.append(gate.unsqueeze(-1) * value)
            gates.append(gate)
        return (
            torch.stack(contributions, dim=0).mean(dim=0),
            torch.stack(gates, dim=0),
        )

    @staticmethod
    def _mean_gate(gates: torch.Tensor) -> torch.Tensor:
        return gates.mean(dim=0)

    def _route(
        self,
        h: torch.Tensor,
        outputs: list[torch.Tensor | None],
        gates: list[torch.Tensor | None],
        active: tuple[int, ...],
        *,
        hard: bool,
    ) -> torch.Tensor:
        if self.router is None:
            raise RuntimeError("Reader routing requires adaptive_router=True")
        if len(outputs) != 3 or len(gates) != 3:
            raise ValueError("The tri-reader must receive exactly three expert slots")
        if not active:
            raise ValueError("At least one expert must be active")

        features, reference = self._router_features(h, outputs, gates, active)
        logits = self.router(features)
        active_mask = torch.zeros(3, dtype=torch.bool, device=logits.device)
        active_mask[list(active)] = True
        masked_logits = logits.masked_fill(~active_mask.view(1, 1, 3), -torch.inf)
        weights = torch.softmax(masked_logits / self.router_temperature, dim=-1)
        if hard:
            hard = F.one_hot(weights.argmax(dim=-1), num_classes=3).to(weights.dtype)
            if self.training:
                weights = hard + weights - weights.detach()
            else:
                weights = hard
        self._last_router_logits = logits
        self._last_router_weights = weights
        return sum(
            weights[..., source_index : source_index + 1] * output
            for source_index, output in enumerate(outputs)
            if output is not None
        )

    def _safe_route(
        self,
        h: torch.Tensor,
        outputs: list[torch.Tensor | None],
        gates: list[torch.Tensor | None],
        *,
        hard: bool,
    ) -> torch.Tensor:
        """Keep E and add conservative GE/GH residuals.

        The regular tri-reader uses a probability simplex and can therefore
        replace E with a generated expert.  Safe routing instead compares each
        generated logit with E's logit and uses that advantage as a bounded
        residual coefficient:

        ``E + a_GE * GE + a_GH * GH``.

        A positive threshold gives the zero-initialized router a small residual
        prior while preserving E as an exact fallback.  ``hard`` turns each
        residual into a binary on/off decision (with a straight-through
        estimator during training).
        """
        if self.router is None:
            raise RuntimeError("Safe routing requires adaptive_router=True")
        if len(outputs) != 3 or len(gates) != 3 or outputs[0] is None:
            raise ValueError("Safe routing requires E, GE and GH outputs")
        features, reference = self._router_features(h, outputs, gates, (0, 1, 2))
        logits = self.router(features)
        advantage = logits[..., 1:] - logits[..., :1] - self.safe_router_threshold
        coefficients = torch.sigmoid(advantage / self.router_temperature)
        if hard:
            hard_coefficients = (advantage >= 0).to(coefficients.dtype)
            if self.training:
                coefficients = hard_coefficients + coefficients - coefficients.detach()
            else:
                coefficients = hard_coefficients
        coefficients = coefficients * self.safe_router_scale
        weights = torch.cat(
            (reference.new_ones(*reference.shape[:2], 1), coefficients), dim=-1
        )
        self._last_router_logits = logits
        # This is intentionally not a probability simplex: the first column
        # documents the unconditional E residual and the other columns are
        # generated residual coefficients.
        self._last_router_weights = weights
        return (
            outputs[0]
            + coefficients[..., 0:1] * outputs[1]
            + coefficients[..., 1:2] * outputs[2]
        )

    def _router_features(
        self,
        h: torch.Tensor,
        outputs: list[torch.Tensor | None],
        gates: list[torch.Tensor | None],
        active: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the per-token evidence shared by source and subset Readers."""
        if len(outputs) != 3 or len(gates) != 3:
            raise ValueError("The tri-reader must receive exactly three expert slots")
        if not active:
            raise ValueError("At least one expert must be active")

        reference = next(value for value in outputs if value is not None)
        eps = torch.finfo(reference.dtype).eps
        batch_size, seq_len = reference.shape[:2]
        zero_scalar = reference.new_zeros(batch_size, seq_len)
        gate_features = [
            self._mean_gate(value) if value is not None else zero_scalar
            for value in gates
        ]
        rms = [
            value.square().mean(dim=-1).add(eps).sqrt()
            if value is not None
            else reference.new_full((batch_size, seq_len), eps**0.5)
            for value in outputs
        ]
        cosine_features = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            if left not in active or right not in active:
                cosine_features.append(zero_scalar)
            else:
                cosine_features.append(
                    (
                        (outputs[left] * outputs[right]).mean(dim=-1)
                        / (rms[left] * rms[right]).clamp_min(eps)
                    ).clamp(-1.0, 1.0)
                )
        active_features = reference.new_zeros(batch_size, seq_len, 3)
        active_features[..., list(active)] = 1.0
        scalar_features = torch.stack(
            gate_features
            + [value.clamp_min(eps).log() for value in rms]
            + cosine_features,
            dim=-1,
        )
        features = torch.cat((scalar_features, active_features), dim=-1)
        if self.router_semantic_projection is not None:
            semantic = torch.tanh(self.router_semantic_projection(self.norm_h(h)))
            features = torch.cat((features, semantic), dim=-1)
        return features, reference

    def configure_advantage_reader(
        self,
        candidates: str = "sources",
        threshold: float = 0.0,
        confidence_threshold: float = 0.5,
        temperature: float = 0.15,
        max_scale: float = 1.0,
    ) -> dict:
        """Lazily enable the causal E-anchored advantage Reader.

        The new head is intentionally not constructed in ``__init__``.  This
        keeps an unconfigured adaptor byte-for-byte compatible with old
        checkpoints and leaves its constructor RNG stream untouched.  Its
        output layout is ``[advantages, confidence_logits]`` with one column
        per configured candidate.  Candidate zero is always E and its
        predicted advantage is pinned to zero at read time.

        ``temperature`` is persisted as part of the reader contract for the
        span-teacher/calibration loss and converts an admitted advantage into
        a bounded interpolation scale at runtime.  The explicit advantage and
        confidence thresholds still decide whether a candidate is admitted;
        an admitted candidate can then be blended continuously from the E
        anchor.
        """
        if candidates not in {"sources", "subsets"}:
            raise ValueError("candidates must be either 'sources' or 'subsets'")
        values = {
            "threshold": threshold,
            "confidence_threshold": confidence_threshold,
            "temperature": temperature,
            "max_scale": max_scale,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Advantage reader configuration values must be finite")
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")

        candidate_subsets = (
            ((0,), (1,), (2,))
            if candidates == "sources"
            else TRI_READER_SUBSETS
        )
        candidate_names = (
            ("E", "GE", "GH")
            if candidates == "sources"
            else TRI_READER_SUBSET_NAMES
        )
        feature_size = (
            12
            + (
                self.router_semantic_size
                if self.router_semantic_projection is not None
                else 0
            )
            # ``router_semantic_size`` is allowed to be zero on existing
            # checkpoints.  The advantage head still receives the full
            # detached causal hidden state so it can make semantic decisions.
            + self.d_model
        )
        output_size = 2 * len(candidate_subsets)

        if self.advantage_router is None:
            head = nn.Sequential(
                nn.Linear(feature_size, self.router_hidden_size),
                nn.SiLU(),
                nn.Linear(self.router_hidden_size, output_size),
            )
            # Match the already-materialized adaptor's placement and dtype.
            head = head.to(
                device=self.norm_h.weight.device,
                dtype=self.norm_h.weight.dtype,
            )
            # Start uncommitted.  In particular, a hard route with zero
            # outputs cannot displace the E fallback before supervision.
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.advantage_router = head
            self._advantage_feature_size = feature_size
            self._advantage_output_size = output_size
        elif (
            self._advantage_feature_size != feature_size
            or self._advantage_output_size != output_size
        ):
            raise ValueError(
                "Advantage reader is already configured with a different "
                "candidate set or feature shape"
            )

        self._advantage_reader_candidates = candidates
        self._advantage_candidate_subsets = tuple(candidate_subsets)
        self._advantage_candidate_names = tuple(candidate_names)
        self._advantage_threshold = float(threshold)
        self._advantage_confidence_threshold = float(confidence_threshold)
        self._advantage_temperature = float(temperature)
        # ``max_scale`` is a safety coefficient, so clamp rather than allow a
        # caller to amplify a generated candidate beyond the E anchor.
        self._advantage_max_scale = min(max(float(max_scale), 0.0), 1.0)
        return self.get_advantage_reader_config()

    def get_advantage_reader_config(self) -> dict:
        """Return JSON-serializable advantage-reader runtime metadata."""
        enabled = self.advantage_router is not None
        return {
            "enabled": enabled,
            "candidates": self._advantage_reader_candidates,
            "candidate_names": (
                list(self._advantage_candidate_names)
                if self._advantage_candidate_names is not None
                else None
            ),
            "candidate_subsets": (
                [list(subset) for subset in self._advantage_candidate_subsets]
                if self._advantage_candidate_subsets is not None
                else None
            ),
            "threshold": float(self._advantage_threshold),
            "confidence_threshold": float(self._advantage_confidence_threshold),
            "temperature": float(self._advantage_temperature),
            "max_scale": float(self._advantage_max_scale),
            "feature_size": (
                int(self._advantage_feature_size)
                if hasattr(self, "_advantage_feature_size")
                else None
            ),
            "semantic_context_size": self.d_model,
        }

    def configure_random_router(self, seed: int = 42) -> None:
        """Configure deterministic token-wise permutation for random routing.

        The trained advantage head is still evaluated. Only its realised
        source-weight vectors are permuted across token positions, preserving
        the empirical route/alpha distribution while removing its input
        alignment. This is intentionally an inference-only control.
        """
        self._random_router_seed = int(seed)
        self._random_router_calls = 0

    def train_advantage_reader_only(self) -> list[str]:
        """Freeze every existing parameter and expose only the new head."""
        if self.advantage_router is None:
            raise ValueError(
                "Configure the advantage reader before advantage-only training"
            )
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.advantage_router.parameters():
            parameter.requires_grad = True
        self.set_tri_reader_mode("tri_advantage_routed")
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def set_advantage_supervision_only(self, enabled: bool) -> None:
        """Run the advantage head while returning the exact frozen E output."""
        if self.advantage_router is None:
            raise ValueError(
                "Configure the advantage reader before setting supervision mode"
            )
        self._advantage_supervision_only = bool(enabled)

    def get_last_advantage_predictions(self) -> torch.Tensor | None:
        """Return latest per-candidate predicted LM advantages."""
        return self._last_advantage_predictions

    def get_last_advantage_confidence_logits(self) -> torch.Tensor | None:
        """Return latest per-candidate confidence logits."""
        return self._last_advantage_confidence_logits

    def get_last_advantage_weights(self) -> torch.Tensor | None:
        """Return latest E-anchored candidate interpolation weights."""
        return self._last_advantage_weights

    @staticmethod
    def _finite_candidate(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a finite tensor and a per-token validity mask."""
        finite = torch.isfinite(value)
        valid = finite.all(dim=-1)
        safe = torch.where(finite, value, torch.zeros_like(value))
        return safe, valid

    @staticmethod
    def _finite_gate(value: torch.Tensor) -> torch.Tensor:
        """Normalize and sanitize gate diagnostics without changing values.

        Runtime experts expose gates as ``(branches, batch, time)``.  A
        single-branch test double may omit the leading branch dimension, so
        accept ``(batch, time)`` and restore the canonical shape here.
        """
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(
                "Expected gate diagnostics with shape (branches, batch, time), "
                f"got {tuple(value.shape)}"
            )
        return torch.where(torch.isfinite(value), value, torch.zeros_like(value))

    def _advantage_features(
        self,
        h: torch.Tensor,
        outputs: list[torch.Tensor | None],
        gates: list[torch.Tensor | None],
    ) -> torch.Tensor:
        """Build detached causal evidence plus semantic hidden context."""
        # No gradients should reach an expert, backbone, or an existing router
        # while a span teacher probes the new head.  The detached hidden state
        # is still causal because it is the current adaptor input.
        with torch.no_grad():
            evidence, _ = self._router_features(h, outputs, gates, (0, 1, 2))
            semantic = self.norm_h(h)
            semantic = torch.where(
                torch.isfinite(semantic), semantic, torch.zeros_like(semantic)
            )
            features = torch.cat((evidence, semantic), dim=-1)
        return features.detach()

    def _fuse_advantage_subset(
        self,
        h: torch.Tensor,
        outputs: list[torch.Tensor | None],
        gates: list[torch.Tensor | None],
        subset: tuple[int, ...],
    ) -> torch.Tensor:
        """Match the existing masked source Reader for one candidate subset."""
        if len(subset) == 1:
            return outputs[subset[0]]
        if self.router is None:
            return self._fixed_fuse(outputs, subset)
        # Existing pair modes honor router_hard; the named all-three soft
        # endpoint remains soft.  Advantage selection itself is hard in both
        # train and eval, so this only defines the candidate endpoint.
        return self._route(
            h,
            outputs,
            gates,
            subset,
            hard=len(subset) == 2 and self.router_hard,
        )

    def _compute_advantage_candidates(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        cue_mem: torch.Tensor | None,
        context_h: torch.Tensor | None,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
    ]:
        """Compute standalone source/subset endpoints and safety masks.

        Each subset is loop-refined independently.  In particular, the GE/GH
        stream used for ``(GE, GH)`` or ``(E, GE, GH)`` is not copied from a
        jointly-refined stream when its single-endpoint loop would differ.
        """
        direct_output, direct_gate = self._direct_engram(h, mem)
        ge_output, ge_gate = self._generated_expert(0, h, mem, cue_mem)
        gh_h = h if context_h is None else context_h
        gh_output, gh_gate = self._generated_expert(
            1, h, None, None, context_h=gh_h
        )

        raw_outputs = [direct_output, ge_output, gh_output]
        raw_gates = [direct_gate, ge_gate, gh_gate]
        safe_sources: list[torch.Tensor] = []
        source_validity: list[torch.Tensor] = []
        safe_gates: list[torch.Tensor] = []
        for output, gate in zip(raw_outputs, raw_gates):
            safe_output, valid = self._finite_candidate(output)
            safe_sources.append(safe_output)
            source_validity.append(valid)
            safe_gates.append(self._finite_gate(gate))

        if self._advantage_candidate_subsets is None:
            raise RuntimeError("Advantage reader candidate set is not configured")

        candidates: list[torch.Tensor] = []
        candidate_validity: list[torch.Tensor] = []
        for subset in self._advantage_candidate_subsets:
            subset_outputs: list[torch.Tensor | None] = [None, None, None]
            subset_gates: list[torch.Tensor | None] = [None, None, None]
            for source_index in subset:
                subset_outputs[source_index] = safe_sources[source_index]
                subset_gates[source_index] = safe_gates[source_index]
            subset_outputs = self._apply_looped_reader(h, subset_outputs)
            fused = self._fuse_advantage_subset(
                h, subset_outputs, subset_gates, subset
            )
            safe_fused, finite = self._finite_candidate(fused)
            source_valid = torch.ones_like(finite)
            for source_index in subset:
                source_valid = source_valid & source_validity[source_index]
            candidates.append(safe_fused)
            candidate_validity.append(finite & source_valid)

        source_endpoints = candidates[:3]
        feature_outputs = torch.stack(source_endpoints, dim=0)
        feature_gates = safe_gates
        return (
            candidates,
            torch.stack(candidate_validity, dim=-1),
            safe_gates,
            source_endpoints,
            feature_outputs,
        )

    def _advantage_route(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        cue_mem: torch.Tensor | None,
        context_h: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a valid candidate and blend it continuously from E.

        The selected candidate is admitted only when its predicted advantage
        and confidence clear their configured thresholds.  Its E-anchored
        interpolation coefficient is then

            ``a = max_scale * clamp((advantage - threshold) / temperature, 0, 1)
                 * sigmoid(confidence_logit)``.

        Rejected candidates use ``a=0`` and therefore return the exact E
        output.  Candidate tensors are sanitized before any interpolation so
        invalid generated values cannot contaminate that fallback.
        """
        if self.advantage_router is None:
            raise RuntimeError(
                "tri_advantage_routed requires configure_advantage_reader()"
            )
        if self._advantage_candidate_subsets is None:
            raise RuntimeError("Advantage reader candidate set is not configured")

        if self._advantage_supervision_only:
            with torch.no_grad():
                (
                    candidates,
                    candidate_validity,
                    safe_gates,
                    source_endpoints,
                    feature_outputs,
                ) = self._compute_advantage_candidates(h, mem, cue_mem, context_h)
        else:
            (
                candidates,
                candidate_validity,
                safe_gates,
                source_endpoints,
                feature_outputs,
            ) = self._compute_advantage_candidates(h, mem, cue_mem, context_h)

        # The first three candidates are always E/GE/GH in canonical order,
        # including for the seven-subset Reader.  Use those standalone
        # endpoints as scalar evidence and add detached semantic context.
        feature_gates = safe_gates
        router_features = self._advantage_features(
            h,
            [value for value in feature_outputs],
            feature_gates,
        )
        head_output = self.advantage_router(router_features)
        candidate_count = len(self._advantage_candidate_subsets)
        raw_predictions = head_output[..., :candidate_count]
        raw_confidence = head_output[..., candidate_count:]
        finite_predictions = torch.isfinite(raw_predictions)
        finite_confidence = torch.isfinite(raw_confidence)
        predictions = torch.where(
            finite_predictions, raw_predictions, torch.zeros_like(raw_predictions)
        )
        confidence_logits = torch.where(
            finite_confidence, raw_confidence, torch.zeros_like(raw_confidence)
        )
        # E is a true baseline, not a learned class whose scale can drift.
        predictions = predictions.clone()
        predictions[..., 0] = 0.0
        self._last_advantage_predictions = predictions
        self._last_advantage_confidence_logits = confidence_logits

        if candidate_count == 1:
            selected = torch.zeros(
                h.shape[0], h.shape[1], dtype=torch.long, device=h.device
            )
        else:
            generated_advantages = predictions[..., 1:]
            confidence = torch.sigmoid(confidence_logits[..., 1:])
            eligible = (
                generated_advantages.gt(self._advantage_threshold)
                & confidence.ge(self._advantage_confidence_threshold)
                & candidate_validity[..., 1:]
                & finite_predictions[..., 1:]
                & finite_confidence[..., 1:]
            )
            masked_advantages = generated_advantages.masked_fill(~eligible, -torch.inf)
            best = masked_advantages.argmax(dim=-1) + 1
            has_candidate = eligible.any(dim=-1)
            selected = torch.where(
                has_candidate,
                best,
                torch.zeros_like(best),
            )

        # Convert the selected candidate's evidence into a continuous,
        # E-anchored interpolation coefficient.  Admission remains discrete,
        # so a rejected or invalid candidate always has exactly ``a=0``.
        scale = self._advantage_max_scale
        selected_generated = selected.ne(0)
        interpolation = h.new_zeros(h.shape[:2])
        if candidate_count > 1:
            selected_index = selected.unsqueeze(-1)
            selected_advantage = predictions.gather(-1, selected_index).squeeze(-1)
            selected_confidence = torch.sigmoid(
                confidence_logits.gather(-1, selected_index).squeeze(-1)
            )
            advantage_scale = (
                (selected_advantage - self._advantage_threshold)
                / self._advantage_temperature
            ).clamp(0.0, 1.0)
            interpolation = scale * advantage_scale * selected_confidence
            interpolation = torch.where(
                selected_generated,
                interpolation,
                torch.zeros_like(interpolation),
            )
            interpolation = torch.where(
                torch.isfinite(interpolation),
                interpolation,
                torch.zeros_like(interpolation),
            )

        # Candidate weights describe the actual E-anchor interpolation.  The
        # selected candidate receives ``a`` and E receives ``1-a``.
        weights = h.new_zeros(h.shape[0], h.shape[1], candidate_count)
        weights[..., 0] = 1.0 - interpolation
        for candidate_index in range(1, candidate_count):
            weights[..., candidate_index] = (
                selected.eq(candidate_index).to(weights.dtype) * interpolation
            )
        self._last_advantage_weights = weights

        e_output = candidates[0]
        contribution = e_output
        for candidate_index in range(1, candidate_count):
            selected_mask = selected.eq(candidate_index)
            selected_mask_3d = selected_mask.unsqueeze(-1)
            # Replace unselected values before taking the difference.  This
            # keeps an invalid candidate from producing a NaN in an E-only
            # token, even if a future endpoint implementation is not finite.
            candidate = torch.where(selected_mask_3d, candidates[candidate_index], e_output)
            delta = candidate - e_output
            blended = e_output + interpolation.unsqueeze(-1) * delta
            # Preserve an exact endpoint when the coefficient reaches one;
            # this makes ``max_scale=1`` a true complete switch.
            blended = torch.where(
                interpolation.eq(1.0).unsqueeze(-1),
                candidate,
                blended,
            )
            contribution = torch.where(selected_mask_3d, blended, contribution)

        # Preserve the existing three-source diagnostics API while exposing
        # the richer candidate-space weights through the new getter.
        source_weights = h.new_zeros(h.shape[0], h.shape[1], 3)
        source_weights[..., 0] = 1.0
        for candidate_index, subset in enumerate(self._advantage_candidate_subsets):
            if candidate_index == 0:
                continue
            selected_mask = selected.eq(candidate_index)
            candidate_scale = selected_mask.to(source_weights.dtype) * interpolation
            for source_index in subset:
                source_weights[..., source_index] += candidate_scale / len(subset)
            source_weights[..., 0] -= candidate_scale
        self._last_router_weights = source_weights
        self._last_source_outputs = torch.stack(source_endpoints, dim=0)
        visible_gates = torch.stack(safe_gates, dim=0)

        if self._advantage_supervision_only:
            # The head has already run above, but a teacher probe must not
            # inject a selected/generated residual into the clean backbone.
            return e_output, visible_gates
        return contribution, visible_gates

    def _random_advantage_route(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        cue_mem: torch.Tensor | None,
        context_h: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply trained route weights after token-wise randomisation.

        The current ATHENA checkpoint uses the three standalone candidates
        E/GE/GH. We first run the normal E-anchored route, then permute its
        realised three-source weights over batch/time positions and apply
        those weights to the same source endpoints. This preserves the
        empirical source and interpolation-strength distribution while
        removing its input-conditioned alignment.
        """
        contribution, visible_gates = self._advantage_route(
            h, mem, cue_mem, context_h
        )
        if self._advantage_candidate_subsets != ((0,), (1,), (2,)):
            raise ValueError(
                "Random-router control requires standalone E/GE/GH candidates"
            )
        if self._last_source_outputs is None or self._last_router_weights is None:
            raise RuntimeError("Advantage route did not expose source diagnostics")

        del contribution
        weights = self._last_router_weights
        if weights.shape[-1] != 3:
            raise RuntimeError(
                f"Expected three-source route weights, got {tuple(weights.shape)}"
            )
        flat_weights = weights.reshape(-1, 3)
        generator = torch.Generator(device=flat_weights.device)
        generator.manual_seed(self._random_router_seed + self._random_router_calls)
        self._random_router_calls += 1
        permutation = torch.randperm(
            flat_weights.shape[0],
            generator=generator,
            device=flat_weights.device,
        )
        shuffled = flat_weights[permutation].reshape_as(weights)
        self._last_router_weights = shuffled
        self._last_advantage_weights = shuffled

        source_outputs = self._last_source_outputs
        randomised = torch.zeros_like(source_outputs[0])
        for source_index in range(3):
            randomised = randomised + (
                shuffled[..., source_index : source_index + 1]
                * source_outputs[source_index]
            )
        return randomised, visible_gates

    def _subset_route(
        self,
        h: torch.Tensor,
        outputs: list[torch.Tensor | None],
        gates: list[torch.Tensor | None],
        *,
        hard: bool,
    ) -> torch.Tensor:
        """Hierarchical unified Reader over all seven non-empty subsets.

        First predict ``p(S | h, expert evidence)`` for every non-empty
        subset.  For each candidate subset, use the source Reader logits with
        a subset mask to obtain ``p(i | S, h)``.  The final residual is

            sum_S p(S) sum_{i in S} p(i | S) expert_i.

        The source marginal weights are retained in ``_last_router_weights``
        for existing monitoring APIs; the seven-way probabilities are exposed
        by ``get_last_subset_router_weights``.
        """
        if self.router is None or self.subset_router is None:
            raise RuntimeError(
                "Unified subset routing requires adaptive_router=True"
            )
        active = (0, 1, 2)
        features, reference = self._router_features(h, outputs, gates, active)
        source_logits = self.router(features)
        subset_logits = self.subset_router(features)
        subset_weights = torch.softmax(
            subset_logits / self.subset_router_temperature,
            dim=-1,
        )
        if hard:
            hard_subset = F.one_hot(
                subset_weights.argmax(dim=-1), num_classes=len(TRI_READER_SUBSETS)
            ).to(subset_weights.dtype)
            if self.training:
                subset_weights = (
                    hard_subset + subset_weights - subset_weights.detach()
                )
            else:
                subset_weights = hard_subset

        source_marginals = reference.new_zeros(*reference.shape[:2], 3)
        contribution = reference.new_zeros(reference.shape)
        for subset_index, subset in enumerate(TRI_READER_SUBSETS):
            subset_mask = torch.zeros(
                3, dtype=torch.bool, device=source_logits.device
            )
            subset_mask[list(subset)] = True
            masked_source_logits = source_logits.masked_fill(
                ~subset_mask.view(1, 1, 3), -torch.inf
            )
            source_weights = torch.softmax(
                masked_source_logits / self.router_temperature,
                dim=-1,
            )
            candidate = sum(
                source_weights[..., source_index : source_index + 1] * outputs[source_index]
                for source_index in subset
            )
            subset_probability = subset_weights[..., subset_index : subset_index + 1]
            contribution = contribution + subset_probability * candidate
            source_marginals = source_marginals + subset_probability * source_weights

        self._last_router_logits = source_logits
        self._last_router_weights = source_marginals
        self._last_subset_router_logits = subset_logits
        self._last_subset_router_weights = subset_weights
        return contribution

    def _fixed_fuse(
        self,
        outputs: list[torch.Tensor | None],
        active: tuple[int, ...],
    ) -> torch.Tensor:
        """Fallback fusion for a tri adaptor built without a Router.

        Normal tri-reader runs use the masked Reader for pair/triple modes.
        This deterministic sum exists only so an adaptor explicitly created
        with ``adaptive_router=False`` still has usable subset ablations.
        """
        if not active:
            raise ValueError("At least one tri-reader expert must be active")
        present = [outputs[index] for index in active]
        if any(value is None for value in present):
            raise RuntimeError("A requested tri-reader expert was not computed")
        return sum(present[1:], start=present[0])

    def forward(
        self,
        h: torch.Tensor,
        mem: torch.Tensor | None,
        cue_mem: torch.Tensor | None = None,
        context_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 3 or h.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected h shape (B, T, {self.d_model}), got {tuple(h.shape)}"
            )
        self._last_router_logits = None
        self._last_router_weights = None
        self._last_subset_router_logits = None
        self._last_subset_router_weights = None
        self._last_source_outputs = None
        self._last_loop_workspace = None
        self._last_loop_gate = None

        mode = normalize_tri_reader_mode(self.tri_reader_mode)
        active = self.active_expert_indices(mode)
        if 0 in active or 1 in active:
            if mem is None:
                raise ValueError(f"{mode} requires Engram memory for E or GE")

        expert_outputs: list[torch.Tensor | None] = [None, None, None]
        expert_gates: list[torch.Tensor | None] = [None, None, None]
        if mode == "tri_advantage_routed":
            contribution, gate_values = self._advantage_route(
                h,
                mem,
                cue_mem,
                context_h,
            )
            return contribution, gate_values
        if mode == "tri_random_advantage_routed":
            contribution, gate_values = self._random_advantage_route(
                h,
                mem,
                cue_mem,
                context_h,
            )
            return contribution, gate_values

        direct_output = None
        direct_gate = None
        if 0 in active:
            # E is the original direct Engram lookup plus Reader path.
            direct_output, direct_gate = self._direct_engram(h, mem)
        if 0 in active:
            expert_outputs[0], expert_gates[0] = direct_output, direct_gate
        if 1 in active:
            generated_output, generated_gate = self._generated_expert(
                0, h, mem, cue_mem
            )
            # GE is a complete, independent expert: Engram lookup vectors are
            # used only as causal generator cues. The direct E output is not
            # added here; E and GE must remain separately measurable.
            expert_outputs[1] = generated_output
            expert_gates[1] = generated_gate
        if 2 in active:
            # GH deliberately receives no ``mem`` or ``cue_mem`` argument.
            # This makes it impossible for the context expert to accidentally
            # fall back to an Engram lookup when it is used in a combination.
            gh_h = h if context_h is None else context_h
            if gh_h.shape != h.shape:
                raise ValueError(
                    "GH context hidden states must have the same shape as the "
                    f"current hidden states: context={tuple(gh_h.shape)}, "
                    f"current={tuple(h.shape)}"
                )
            expert_outputs[2], expert_gates[2] = self._generated_expert(
                1, h, None, None, context_h=gh_h
            )

        outputs = [value for value in expert_outputs]
        gates = [value for value in expert_gates]
        outputs = self._apply_looped_reader(h, outputs)
        first_gate = next(value for value in gates if value is not None)
        zero_gate = first_gate.new_zeros(first_gate.shape)
        visible_gates = [value if value is not None else zero_gate for value in gates]

        if active == (0,):
            contribution = outputs[0]
            reader_index = torch.zeros_like(visible_gates[0], dtype=torch.long)
            self._last_router_weights = F.one_hot(reader_index, num_classes=3).to(h.dtype)
        elif active == (1,):
            contribution = outputs[1]
            reader_index = torch.ones_like(visible_gates[1], dtype=torch.long)
            self._last_router_weights = F.one_hot(reader_index, num_classes=3).to(h.dtype)
        elif active == (2,):
            contribution = outputs[2]
            reader_index = torch.full_like(visible_gates[2], 2, dtype=torch.long)
            self._last_router_weights = F.one_hot(reader_index, num_classes=3).to(h.dtype)
        elif mode in {"tri_subset_routed", "tri_subset_soft_fused"}:
            self._last_source_outputs = torch.stack(
                [value if value is not None else h.new_zeros(h.shape) for value in outputs],
                dim=0,
            )
            contribution = self._subset_route(
                h,
                outputs,
                visible_gates,
                hard=(mode == "tri_subset_routed"),
            )
        elif mode == "tri_safe_routed":
            self._last_source_outputs = torch.stack(
                [value if value is not None else h.new_zeros(h.shape) for value in outputs],
                dim=0,
            )
            contribution = self._safe_route(
                h,
                outputs,
                visible_gates,
                hard=self.router_hard,
            )
        elif self.router is None:
            # A non-adaptive tri adaptor can still expose deterministic subset
            # ablations, but learned pair/triple fusion requires the Reader.
            contribution = self._fixed_fuse(outputs, active)
        else:
            self._last_source_outputs = torch.stack(
                [value if value is not None else h.new_zeros(h.shape) for value in outputs],
                dim=0,
            )
            contribution = self._route(
                h,
                outputs,
                visible_gates,
                active,
                # Pair modes are masked soft fusion by default.  They can be
                # made hard through configure_router(hard=True); tri_routed
                # is always hard and tri_soft_fused is always soft by name.
                hard=(
                    mode == "tri_routed"
                    or (mode in {"e_ge", "e_gh", "ge_gh"} and self.router_hard)
                ),
            )

        # Keep a fixed (E, GE, GH) shape for monitoring and ablation tests.
        return contribution, torch.stack(visible_gates, dim=0)

    def set_tri_reader_mode(self, mode: str) -> None:
        mode = normalize_tri_reader_mode(mode)
        if mode in {"tri_advantage_routed", "tri_random_advantage_routed"} and self.advantage_router is None:
            raise ValueError(
                "Advantage routing modes require configure_advantage_reader()"
            )
        if mode in {"tri_routed", "tri_soft_fused", "tri_safe_routed"} and self.router is None:
            raise ValueError("tri routed modes require adaptive_router=True")
        if mode in {"tri_subset_routed", "tri_subset_soft_fused"} and (
            self.router is None or self.subset_router is None
        ):
            raise ValueError(
                "Unified subset routing requires adaptive_router=True"
            )
        self.tri_reader_mode = mode

    def configure_router(
        self,
        temperature: float = 1.0,
        hard: bool = False,
        min_generated_probability: float = 0.5,
        safe_residual_threshold: float = 1.0,
        safe_residual_scale: float = 1.0,
    ) -> None:
        # Kept for compatibility with the shared evaluation loader. A
        # three-way hard router uses argmax rather than a binary threshold.
        del min_generated_probability
        if self.router is None:
            raise ValueError("Router configuration requires adaptive_router=True")
        if temperature <= 0:
            raise ValueError("Router temperature must be positive")
        if safe_residual_threshold < 0:
            raise ValueError("safe_residual_threshold must be non-negative")
        if safe_residual_scale < 0:
            raise ValueError("safe_residual_scale must be non-negative")
        self.router_temperature = float(temperature)
        self.router_hard = bool(hard)
        self.safe_router_threshold = float(safe_residual_threshold)
        self.safe_router_scale = float(safe_residual_scale)
        self.subset_router_temperature = float(temperature)
        self.subset_router_hard = bool(hard)

    def get_last_router_logits(self) -> torch.Tensor | None:
        return self._last_router_logits

    def get_last_router_weights(self) -> torch.Tensor | None:
        return self._last_router_weights

    def get_last_source_outputs(self) -> torch.Tensor | None:
        return self._last_source_outputs

    def get_last_loop_workspace(self) -> torch.Tensor | None:
        """Return the latest recurrent workspace, if looped reading is active."""
        return self._last_loop_workspace

    def get_last_loop_gate(self) -> torch.Tensor | None:
        """Return bounded per-token recurrent update gates."""
        return self._last_loop_gate

    def get_last_subset_router_logits(self) -> torch.Tensor | None:
        """Return seven-way subset logits from the latest unified forward."""
        return self._last_subset_router_logits

    def get_last_subset_router_weights(self) -> torch.Tensor | None:
        """Return ``(E, GE, GH, E+GE, E+GH, GE+GH, E+GE+GH)`` weights."""
        return self._last_subset_router_weights

    def initialize_engram_reader_from_legacy(self, legacy_adaptor: nn.Module) -> None:
        from .adaptor import EngramAdaptor, MultiBranchEngramAdaptor

        expected = EngramAdaptor if self.num_branches == 1 else MultiBranchEngramAdaptor
        if not isinstance(legacy_adaptor, expected):
            raise TypeError(
                f"Expected {expected.__name__}, got {type(legacy_adaptor).__name__}"
            )
        if legacy_adaptor.d_model != self.d_model:
            raise ValueError("Legacy and tri-reader d_model do not match")
        if self.engram_value_projection.weight.shape != legacy_adaptor.w_v.weight.shape:
            raise ValueError("Legacy and tri-reader memory dimensions do not match")
        with torch.no_grad():
            self.engram_value_projection.weight.copy_(legacy_adaptor.w_v.weight)
            self.norm_h.weight.copy_(legacy_adaptor.norm_h.weight)
            if self.num_branches == 1:
                self.engram_key_projection.weight.copy_(legacy_adaptor.w_k.weight)
                self.engram_reader_norm.weight.copy_(legacy_adaptor.norm_k.weight)
                self.engram_gate_bias.copy_(legacy_adaptor.gate_bias)
            else:
                for target, source in zip(
                    self.engram_key_projection, legacy_adaptor.w_k
                ):
                    target.weight.copy_(source.weight)
                for target, source in zip(
                    self.engram_reader_norm, legacy_adaptor.norm_k
                ):
                    target.weight.copy_(source.weight)
                self.engram_gate_bias.copy_(legacy_adaptor.gate_bias)

    @property
    def trainable_param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
