"""Latent generative memory and reader adaptor.

The generator reconstructs a small set of latent memory vectors exclusively
from causal Engram cues.  A separate reader uses the frozen backbone hidden
state as a query and injects the resulting summary through a gated residual.
"""

import math

import torch
import torch.nn as nn

from .adaptor import RMSNorm


class _GeneratorLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.self_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        latents: torch.Tensor,
        cues: torch.Tensor,
        cue_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        update, _ = self.cross_attention(
            latents, cues, cues, key_padding_mask=cue_padding_mask, need_weights=False
        )
        latents = self.cross_norm(latents + update)
        update, _ = self.self_attention(latents, latents, latents, need_weights=False)
        latents = self.self_norm(latents + update)
        return self.ffn_norm(latents + self.ffn(latents))


class GenerativeMemoryAdaptor(nn.Module):
    """Generate latent memory from cues, then read it with a backbone query.

    Information-flow boundary:

        causal Engram cues or the current causal backbone state -> generator latents
        backbone hidden state + generator latents -> reader -> gated residual

    ``num_branches`` keeps the historical multi-branch reader semantics: each
    branch has its own key projection and gate, while all branches share the
    generated memory value and their contributions are averaged.

    With ``cue_source='context'`` the generator receives only the current
    causal backbone state.  ``cue_window`` must be one in that mode so cached
    autoregressive decoding has exactly the same information boundary as
    full-sequence training.  With ``cue_source='learned'``, it receives only
    learned cue tokens, providing a parameter-capacity control.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        reader_type: str = "cross_attention",
        cue_source: str = "engram",
        num_latents: int = 4,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        cue_window: int = 3,
        gate_bias_init: float = 0.0,
        num_branches: int = 1,
        fusion_type: str = "generated_only",
        adaptive_router: bool = False,
        router_hidden_size: int = 16,
        router_semantic_size: int = 0,
        router_expert_mode: str = "residual",
        source_adapter_rank: int = 16,
        generator_loop_rounds: int = 1,
        generator_loop_workspace_size: int = 0,
        generator_loop_gate_max: float = 0.25,
    ):
        super().__init__()
        if reader_type not in ("cross_attention", "mean"):
            raise ValueError(f"Unknown reader_type: {reader_type}")
        if cue_source not in ("engram", "context", "learned", "hybrid"):
            raise ValueError(f"Unknown cue_source: {cue_source}")
        if fusion_type not in (
            "generated_only",
            "engram_residual",
            "dual_reader",
            "tri_reader",
        ):
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        if fusion_type == "tri_reader" or cue_source == "hybrid":
            if fusion_type != "tri_reader" or cue_source != "hybrid":
                raise ValueError(
                    "Three-source generation requires cue_source='hybrid' and "
                    "fusion_type='tri_reader'"
                )
            # Keep the original module as the single-source implementation and
            # expose the three-source implementation from the same public
            # module.  This compatibility delegate lets callers that already
            # instantiate GenerativeMemoryAdaptor use the new strict Reader;
            # build_adaptor returns TriMemoryAdaptor directly so checkpoints
            # retain the compact, stable state-dict layout.
            from .tri_memory import TriMemoryAdaptor

            self._tri_delegate = TriMemoryAdaptor(
                d_model=d_model,
                d_mem=d_mem,
                reader_type=reader_type,
                num_latents=num_latents,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                cue_window=cue_window,
                gate_bias_init=gate_bias_init,
                num_branches=num_branches,
                adaptive_router=adaptive_router,
                router_hidden_size=router_hidden_size,
                router_semantic_size=router_semantic_size,
                source_adapter_rank=source_adapter_rank,
                generator_loop_rounds=generator_loop_rounds,
                generator_loop_workspace_size=generator_loop_workspace_size,
                generator_loop_gate_max=generator_loop_gate_max,
            )
            self.d_model = d_model
            self.d_mem = d_mem
            self.fusion_type = "tri_reader"
            self.cue_source = "hybrid"
            self.dual_reader_mode = "both"
            self.router = self._tri_delegate.router
            self.subset_router = self._tri_delegate.subset_router
            return
        if fusion_type == "engram_residual" and cue_source != "engram":
            raise ValueError("engram_residual fusion requires Engram cues")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if min(num_latents, num_layers, num_heads, cue_window, num_branches) < 1:
            raise ValueError("Generator sizes, layers, heads, and cue_window must be positive")
        if router_hidden_size < 1:
            raise ValueError("router_hidden_size must be positive")
        if router_semantic_size < 0:
            raise ValueError("router_semantic_size must be non-negative")
        if generator_loop_rounds < 1:
            raise ValueError("generator_loop_rounds must be positive")
        if generator_loop_workspace_size < 0:
            raise ValueError("generator_loop_workspace_size must be non-negative")
        if not 0.0 <= generator_loop_gate_max <= 1.0:
            raise ValueError("generator_loop_gate_max must be in [0, 1]")
        if router_expert_mode not in ("residual", "source"):
            raise ValueError(f"Unknown router_expert_mode: {router_expert_mode}")
        if adaptive_router and fusion_type != "dual_reader":
            raise ValueError("Adaptive routing requires fusion_type='dual_reader'")
        if cue_source == "engram" and d_mem < 1:
            raise ValueError("Engram cue source requires d_mem > 0")
        if cue_source == "context" and cue_window != 1:
            raise ValueError("Context cue source requires cue_window=1 for cached decoding")

        self.d_model = d_model
        self.d_mem = d_mem
        self.reader_type = reader_type
        self.cue_source = cue_source
        self.num_latents = num_latents
        self.hidden_size = hidden_size
        self.cue_window = cue_window
        self.num_branches = num_branches
        self.fusion_type = fusion_type
        self.adaptive_router = adaptive_router
        self.router_hidden_size = router_hidden_size
        self.router_semantic_size = router_semantic_size
        self.router_expert_mode = router_expert_mode
        self.generator_loop_rounds = int(generator_loop_rounds)
        self.generator_loop_workspace_size = int(
            generator_loop_workspace_size or hidden_size
        )
        self.generator_loop_gate_max = float(generator_loop_gate_max)
        # Runtime-only evaluation control.  It is intentionally absent from
        # the state dict so the exact same checkpoint can be evaluated with
        # both readers, only the direct Engram reader, or only the generated
        # reader.
        self.dual_reader_mode = "both"
        self.router_temperature = 1.0
        self.router_hard = False
        self.router_min_generated_probability = 0.5
        self.router_supervision_only = False
        self._last_router_logits = None
        self._last_router_weights = None
        self._last_generated_residual = None
        self._last_loop_workspace = None
        self._last_loop_gate = None

        if cue_source == "engram":
            self.cue_projection = nn.Linear(d_mem, hidden_size)
            self.register_parameter("learned_cues", None)
        elif cue_source == "context":
            self.cue_projection = nn.Linear(d_model, hidden_size)
            self.register_parameter("learned_cues", None)
        else:
            self.cue_projection = None
            self.learned_cues = nn.Parameter(torch.empty(cue_window, hidden_size))

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
        if fusion_type in ("engram_residual", "dual_reader"):
            # Preserve a direct Engram signal and learn a context-dependent
            # correction from the generated latent memory.
            self.engram_value_projection = nn.Linear(d_mem, d_model, bias=False)
            self.delta_norm = RMSNorm(d_model)
            self.delta_gate_bias = nn.Parameter(torch.tensor(-2.0))
        else:
            self.engram_value_projection = None
            self.delta_norm = None
            self.register_parameter("delta_gate_bias", None)
        self.norm_h = RMSNorm(d_model)
        if num_branches == 1:
            # Keep historical single-branch checkpoints loadable.
            self.norm_memory = RMSNorm(d_model)
            self.memory_key_projection = None
            self.gate_bias = nn.Parameter(torch.tensor(gate_bias_init))
            if fusion_type == "dual_reader":
                self.engram_key_projection = nn.Linear(d_mem, d_model, bias=False)
                self.engram_reader_norm = RMSNorm(d_model)
                self.engram_gate_bias = nn.Parameter(torch.tensor(gate_bias_init))
            else:
                self.engram_key_projection = None
                self.engram_reader_norm = None
                self.register_parameter("engram_gate_bias", None)
        else:
            self.norm_memory = nn.ModuleList(
                [RMSNorm(d_model) for _ in range(num_branches)]
            )
            self.memory_key_projection = nn.ModuleList(
                [nn.Linear(d_model, d_model, bias=False) for _ in range(num_branches)]
            )
            self.gate_bias = nn.Parameter(torch.full((num_branches,), gate_bias_init))
            if fusion_type == "dual_reader":
                self.engram_key_projection = nn.ModuleList(
                    [nn.Linear(d_mem, d_model, bias=False) for _ in range(num_branches)]
                )
                self.engram_reader_norm = nn.ModuleList(
                    [RMSNorm(d_model) for _ in range(num_branches)]
                )
                self.engram_gate_bias = nn.Parameter(torch.full((num_branches,), gate_bias_init))
            else:
                self.engram_key_projection = None
                self.engram_reader_norm = None
                self.register_parameter("engram_gate_bias", None)

        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        if self.learned_cues is not None:
            nn.init.normal_(self.learned_cues, mean=0.0, std=0.02)
        # A small non-zero initialization preserves initial stability while
        # allowing generator and reader gradients on the first backward pass.
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        if self.memory_key_projection is not None:
            for projection in self.memory_key_projection:
                nn.init.eye_(projection.weight)
        if self.engram_value_projection is not None:
            nn.init.normal_(self.engram_value_projection.weight, mean=0.0, std=0.02)
        if self.engram_key_projection is not None:
            key_projections = (
                self.engram_key_projection
                if isinstance(self.engram_key_projection, nn.ModuleList)
                else [self.engram_key_projection]
            )
            for projection in key_projections:
                nn.init.normal_(projection.weight, mean=0.0, std=0.02)

        if adaptive_router:
            # The five task-agnostic signals are both gates, both contribution
            # magnitudes, and their agreement. ``residual`` preserves the safe
            # historical experts [E, E+G]; ``source`` exposes the independent
            # source competition [E, G].
            if router_semantic_size:
                self.router_semantic_projection = nn.Linear(
                    d_model, router_semantic_size, bias=False
                )
            else:
                self.router_semantic_projection = None
            self.router = nn.Sequential(
                nn.Linear(5 + router_semantic_size, router_hidden_size),
                nn.SiLU(),
                nn.Linear(router_hidden_size, 2),
            )
            nn.init.zeros_(self.router[-1].weight)
            with torch.no_grad():
                # Start close to exact Engram while keeping a finite gradient
                # for Wikipedia causal-LM training.
                self.router[-1].bias.copy_(torch.tensor([4.0, -4.0]))
        else:
            self.router = None
            self.router_semantic_projection = None

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

    def _causal_windows(
        self, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left-padded causal cue windows and their padding mask."""
        if mem.ndim != 3 or mem.shape[-1] != self.d_mem:
            raise ValueError(
                f"Expected mem shape (B, T, {self.d_mem}), got {tuple(mem.shape)}"
            )
        batch_size, seq_len, _ = mem.shape
        left_pad = mem.new_zeros(batch_size, self.cue_window - 1, self.d_mem)
        padded = torch.cat((left_pad, mem), dim=1)
        # Tensor.unfold appends the window dimension after the untouched dims.
        windows = padded.unfold(1, self.cue_window, 1).permute(0, 1, 3, 2)

        time = torch.arange(seq_len, device=mem.device).unsqueeze(1)
        offsets = torch.arange(self.cue_window, device=mem.device).unsqueeze(0)
        valid = offsets >= (self.cue_window - 1 - time)
        padding_mask = ~valid.unsqueeze(0).expand(batch_size, -1, -1)
        return windows, padding_mask

    def _generate(
        self,
        mem: torch.Tensor | None,
        h: torch.Tensor,
        batch_size: int,
        seq_len: int,
        cue_mem: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cue_source == "engram":
            if mem is None:
                raise ValueError("Engram cue source requires memory vectors")
            cue_context = mem if cue_mem is None else cue_mem
            if cue_context.shape[0] != batch_size or cue_context.shape[1] < seq_len:
                raise ValueError(
                    "Engram cue context must cover every hidden-state position: "
                    f"cue={tuple(cue_context.shape)}, hidden=({batch_size}, {seq_len})"
                )
            windows, padding_mask = self._causal_windows(cue_context)
            # During cached decoding ``cue_context`` includes the recent
            # history while ``h`` contains only the new token(s).  Generate
            # latents only for the hidden states in the current model call.
            windows = windows[:, -seq_len:]
            padding_mask = padding_mask[:, -seq_len:]
            cues = self.cue_projection(windows)
            cues = cues.reshape(batch_size * seq_len, self.cue_window, self.hidden_size)
            padding_mask = padding_mask.reshape(batch_size * seq_len, self.cue_window)
        elif self.cue_source == "context":
            if h.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    "Context cues must align with hidden states: "
                    f"hidden={tuple(h.shape)}, expected=({batch_size}, {seq_len}, *)"
                )
            cues = self.cue_projection(h).reshape(
                batch_size * seq_len, 1, self.hidden_size
            )
            padding_mask = None
        else:
            cues = self.learned_cues.view(1, self.cue_window, self.hidden_size).expand(
                batch_size * seq_len, -1, -1
            )
            padding_mask = None

        latents = self.latent_queries.view(1, self.num_latents, self.hidden_size).expand(
            batch_size * seq_len, -1, -1
        )
        for layer in self.generator_layers:
            latents = layer(latents, cues, padding_mask)
        return latents

    def forward(
        self,
        h: torch.Tensor,
        mem: torch.Tensor | None,
        cue_mem: torch.Tensor | None = None,
        context_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate(
                h, mem, cue_mem=cue_mem, context_h=context_h
            )
        if h.ndim != 3 or h.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected h shape (B, T, {self.d_model}), got {tuple(h.shape)}"
            )
        batch_size, seq_len, _ = h.shape
        latents = self._generate(mem, h, batch_size, seq_len, cue_mem=cue_mem)

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

        summary = self.reader_norm(summary).reshape(batch_size, seq_len, self.hidden_size)
        generated_value = self.output_projection(summary)
        if self.fusion_type == "engram_residual":
            if mem is None:
                raise ValueError("Engram residual fusion requires memory vectors")
            raw_value = self.engram_value_projection(mem)
            delta_logit = (
                self.norm_h(h) * self.delta_norm(generated_value)
            ).sum(dim=-1) / math.sqrt(self.d_model)
            delta_gate = torch.sigmoid(delta_logit + self.delta_gate_bias)
            memory_value = raw_value + delta_gate.unsqueeze(-1) * generated_value
        elif self.fusion_type == "dual_reader":
            if mem is None:
                raise ValueError("Dual-reader fusion requires memory vectors")
            # Independent Engram reader: the raw Engram value has its own
            # projection/norm/gate and is not folded into the generated path.
            engram_value = self.engram_value_projection(mem)
            # In residual-corrector mode the generated reader represents a
            # correction, not a replacement for the direct Engram output.
            generated_delta = generated_value
        else:
            memory_value = generated_value

        if self.fusion_type == "dual_reader":
            self._last_router_logits = None
            self._last_router_weights = None
            self._last_generated_residual = None
            norm_h = self.norm_h(h)
            generated_contributions = []
            engram_contributions = []
            generated_gates = []
            engram_gates = []
            if self.num_branches == 1:
                generated_key = self.norm_memory(generated_delta)
                engram_key = self.engram_reader_norm(
                    self.engram_key_projection(mem)
                )
                generated_gate = torch.sigmoid(
                    (norm_h * generated_key).sum(dim=-1) / math.sqrt(self.d_model)
                    + self.gate_bias
                )
                engram_gate = torch.sigmoid(
                    (norm_h * engram_key).sum(dim=-1) / math.sqrt(self.d_model)
                    + self.engram_gate_bias
                )
                engram_output = engram_gate.unsqueeze(-1) * engram_value
                generated_correction = generated_gate.unsqueeze(-1) * generated_delta
                return self._fuse_dual_reader_outputs(
                    h,
                    engram_output,
                    generated_correction,
                    generated_gate,
                    engram_gate,
                )
            for branch_idx, (
                key_projection,
                norm_memory,
                engram_key_projection,
                engram_norm,
            ) in enumerate(
                zip(
                    self.memory_key_projection,
                    self.norm_memory,
                    self.engram_key_projection,
                    self.engram_reader_norm,
                )
            ):
                generated_key = norm_memory(key_projection(generated_delta))
                engram_key = engram_norm(engram_key_projection(mem))
                generated_gate = torch.sigmoid(
                    (norm_h * generated_key).sum(dim=-1) / math.sqrt(self.d_model)
                    + self.gate_bias[branch_idx]
                )
                engram_gate = torch.sigmoid(
                    (norm_h * engram_key).sum(dim=-1) / math.sqrt(self.d_model)
                    + self.engram_gate_bias[branch_idx]
                )
                generated_contributions.append(generated_gate.unsqueeze(-1) * generated_delta)
                engram_contributions.append(engram_gate.unsqueeze(-1) * engram_value)
                generated_gates.append(generated_gate)
                engram_gates.append(engram_gate)
            return self._fuse_dual_reader_outputs(
                h,
                torch.stack(engram_contributions, dim=0).mean(dim=0),
                torch.stack(generated_contributions, dim=0).mean(dim=0),
                torch.stack(generated_gates, dim=0),
                torch.stack(engram_gates, dim=0),
            )

        norm_h = self.norm_h(h)
        if self.num_branches == 1:
            gate_logit = (
                norm_h * self.norm_memory(memory_value)
            ).sum(dim=-1) / math.sqrt(self.d_model)
            gate = torch.sigmoid(gate_logit + self.gate_bias)
            return gate.unsqueeze(-1) * memory_value, gate

        contributions = []
        gates = []
        for branch_idx, (key_projection, norm_memory) in enumerate(
            zip(self.memory_key_projection, self.norm_memory)
        ):
            memory_key = key_projection(memory_value)
            gate_logit = (
                norm_h * norm_memory(memory_key)
            ).sum(dim=-1) / math.sqrt(self.d_model)
            gate = torch.sigmoid(gate_logit + self.gate_bias[branch_idx])
            contributions.append(gate.unsqueeze(-1) * memory_value)
            gates.append(gate)

        contribution = torch.stack(contributions, dim=0).mean(dim=0)
        return contribution, torch.stack(gates, dim=0)

    def _fuse_dual_reader_outputs(
        self,
        h: torch.Tensor,
        engram_output: torch.Tensor,
        generated_output: torch.Tensor,
        generated_gates: torch.Tensor,
        engram_gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse exact ablations or safely route the generated residual."""
        if self.generator_loop_rounds > 1 and self.loop_workspace_update is not None:
            batch_size, seq_len, _ = generated_output.shape
            workspace = generated_output.new_zeros(
                batch_size, seq_len, self.generator_loop_workspace_size
            )
            aggregate = generated_output
            clean_h = self.norm_h(h)
            for _ in range(self.generator_loop_rounds):
                update_input = torch.cat((clean_h, aggregate, workspace), dim=-1)
                proposal = torch.tanh(self.loop_workspace_update(update_input))
                strength = torch.sigmoid(self.loop_workspace_gate(workspace))
                strength = strength * self.generator_loop_gate_max
                workspace = workspace + strength * proposal
                generated_output = generated_output + strength * self.loop_workspace_to_output(
                    workspace
                )
                aggregate = generated_output
            self._last_loop_workspace = workspace
            self._last_loop_gate = strength
        else:
            self._last_loop_workspace = None
            self._last_loop_gate = None
        # Keep the differentiable residual available to the joint-training
        # objective.  This is intentionally the post-gate correction that is
        # actually added to the residual stream, not an internal latent.
        self._last_generated_residual = generated_output
        if self.dual_reader_mode == "engram_only":
            visible_generated_gates = torch.zeros_like(generated_gates)
            contribution = engram_output
        elif self.dual_reader_mode == "generated_only":
            visible_generated_gates = generated_gates
            engram_gates = torch.zeros_like(engram_gates)
            contribution = generated_output
        elif self.dual_reader_mode == "both":
            visible_generated_gates = generated_gates
            contribution = engram_output + generated_output
        else:
            if self.router is None:
                raise RuntimeError("Routed mode requires an adaptive router")
            visible_generated_gates = generated_gates
            generated_gate_mean = (
                generated_gates.mean(dim=0)
                if generated_gates.ndim == 3
                else generated_gates
            )
            engram_gate_mean = (
                engram_gates.mean(dim=0)
                if engram_gates.ndim == 3
                else engram_gates
            )
            eps = torch.finfo(engram_output.dtype).eps
            engram_rms = engram_output.square().mean(dim=-1).add(eps).sqrt()
            generated_rms = generated_output.square().mean(dim=-1).add(eps).sqrt()
            cosine = (
                (engram_output * generated_output).mean(dim=-1)
                / (engram_rms * generated_rms).clamp_min(eps)
            ).clamp(-1.0, 1.0)
            scalar_features = torch.stack(
                (
                    generated_gate_mean,
                    engram_gate_mean,
                    generated_rms.clamp_min(eps).log(),
                    engram_rms.clamp_min(eps).log(),
                    cosine,
                ),
                dim=-1,
            )
            if self.router_semantic_projection is None:
                features = scalar_features
            else:
                semantic_features = torch.tanh(
                    self.router_semantic_projection(self.norm_h(h))
                )
                features = torch.cat((scalar_features, semantic_features), dim=-1)
            router_logits = self.router(features)
            router_weights = torch.softmax(
                router_logits / self.router_temperature, dim=-1
            )
            if self.router_hard:
                hard_indices = (
                    router_weights[..., 1]
                    >= self.router_min_generated_probability
                ).long()
                hard_weights = torch.nn.functional.one_hot(
                    hard_indices, num_classes=2
                ).to(router_weights.dtype)
                if self.training:
                    router_weights = hard_weights + router_weights - router_weights.detach()
                else:
                    router_weights = hard_weights
            self._last_router_logits = router_logits
            self._last_router_weights = router_weights
            # The historical safe router uses [Engram, Engram+generated].  A
            # source reader instead competes directly between [Engram,
            # Generated] and is trained with explicit losses for both paths.
            if self.router_supervision_only:
                # Counterfactual distillation supervises ``router_logits``
                # directly.  Keep the actual residual stream on the frozen
                # Engram path so autograd does not retain the remaining 7B
                # backbone graph merely to train this small selector.
                contribution = engram_output
            elif self.router_expert_mode == "residual":
                contribution = (
                    router_weights[..., 0:1] * engram_output
                    + router_weights[..., 1:2] * (engram_output + generated_output)
                )
            else:
                contribution = (
                    router_weights[..., 0:1] * engram_output
                    + router_weights[..., 1:2] * generated_output
                )

        return contribution, torch.stack(
            (visible_generated_gates, engram_gates), dim=0
        )

    def set_dual_reader_mode(self, mode: str) -> None:
        """Select active reader contributions for a dual-reader checkpoint."""
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            self._tri_delegate.set_tri_reader_mode(mode)
            return
        if self.fusion_type != "dual_reader":
            raise ValueError("Reader ablation requires fusion_type='dual_reader'")
        if mode not in {"both", "engram_only", "generated_only", "routed"}:
            raise ValueError(f"Unknown dual-reader mode: {mode}")
        if mode == "routed" and self.router is None:
            raise ValueError("Routed mode requires adaptive_router=True")
        self.dual_reader_mode = mode

    def set_tri_reader_mode(self, mode: str) -> None:
        """Select an E/GE/GH expert or combination on the compatibility path."""
        if not hasattr(self, "_tri_delegate") or self._tri_delegate is None:
            raise ValueError("Tri-reader mode requires fusion_type='tri_reader'")
        self._tri_delegate.set_tri_reader_mode(mode)

    def active_expert_indices(self, mode: str | None = None) -> tuple[int, ...]:
        if not hasattr(self, "_tri_delegate") or self._tri_delegate is None:
            raise ValueError("Active tri-reader experts require fusion_type='tri_reader'")
        return self._tri_delegate.active_expert_indices(mode)

    def needs_engram_memory(self) -> bool:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate.needs_engram_memory()
        return self.cue_source != "learned"

    def needs_engram_cue_history(self) -> bool:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate.needs_engram_cue_history()
        return self.cue_source == "engram"

    def configure_router(
        self,
        temperature: float = 1.0,
        hard: bool = False,
        min_generated_probability: float = 0.5,
    ) -> None:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            self._tri_delegate.configure_router(
                temperature=temperature,
                hard=hard,
                min_generated_probability=min_generated_probability,
            )
            return
        if self.router is None:
            raise ValueError("Router configuration requires adaptive_router=True")
        if temperature <= 0:
            raise ValueError("Router temperature must be positive")
        if not 0.5 <= min_generated_probability < 1.0:
            raise ValueError(
                "min_generated_probability must be in the interval [0.5, 1.0)"
            )
        self.router_temperature = float(temperature)
        self.router_hard = bool(hard)
        self.router_min_generated_probability = float(min_generated_probability)

    def set_router_supervision_only(self, enabled: bool) -> None:
        """Record router logits while preserving the exact Engram path."""
        if self.router is None:
            raise ValueError("Router supervision requires adaptive_router=True")
        self.router_supervision_only = bool(enabled)

    def get_last_router_logits(self) -> torch.Tensor | None:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate.get_last_router_logits()
        return self._last_router_logits

    def get_last_router_weights(self) -> torch.Tensor | None:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate.get_last_router_weights()
        return self._last_router_weights

    def get_last_subset_router_logits(self) -> torch.Tensor | None:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate.get_last_subset_router_logits()
        return None

    def get_last_subset_router_weights(self) -> torch.Tensor | None:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate.get_last_subset_router_weights()
        return None

    def get_last_loop_workspace(self) -> torch.Tensor | None:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate.get_last_loop_workspace()
        return self._last_loop_workspace

    def get_last_loop_gate(self) -> torch.Tensor | None:
        if hasattr(self, "_tri_delegate") and self._tri_delegate is not None:
            return self._tri_delegate.get_last_loop_gate()
        return self._last_loop_gate

    def get_last_generated_residual(self) -> torch.Tensor | None:
        return self._last_generated_residual

    def train_router_only(self) -> list[str]:
        """Freeze experts and expose the router plus loop workspace (if any).

        Loop workspace parameters are part of the task-agnostic Wikipedia
        reader, so a loop-enabled checkpoint must train them alongside the
        source router rather than leaving the zero-initialized correction
        permanently disabled.
        """
        if self.fusion_type != "dual_reader" or self.router is None:
            raise ValueError(
                "Router-only training requires dual_reader with adaptive_router=True"
            )
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.router.parameters():
            parameter.requires_grad = True
        if self.router_semantic_projection is not None:
            for parameter in self.router_semantic_projection.parameters():
                parameter.requires_grad = True
        for module in (
            self.loop_workspace_update,
            self.loop_workspace_to_output,
            self.loop_workspace_gate,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        self.set_dual_reader_mode("routed")
        return [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def initialize_engram_reader_from_legacy(self, legacy_adaptor: nn.Module) -> None:
        """Copy a legacy Engram reader without changing its computation.

        The generated branch is left at its independent initialization.  In
        ``engram_only`` mode, the converted adaptor therefore reproduces the
        legacy ``EngramAdaptor`` or ``MultiBranchEngramAdaptor`` exactly.
        """
        if self.fusion_type != "dual_reader":
            raise ValueError("Legacy Engram import requires fusion_type='dual_reader'")

        from .adaptor import EngramAdaptor, MultiBranchEngramAdaptor

        expected_type = EngramAdaptor if self.num_branches == 1 else MultiBranchEngramAdaptor
        if not isinstance(legacy_adaptor, expected_type):
            raise TypeError(
                f"Expected {expected_type.__name__} for {self.num_branches} branch(es), "
                f"got {type(legacy_adaptor).__name__}"
            )
        if legacy_adaptor.d_model != self.d_model:
            raise ValueError(
                f"d_model mismatch: legacy={legacy_adaptor.d_model}, new={self.d_model}"
            )
        if self.engram_value_projection.weight.shape != legacy_adaptor.w_v.weight.shape:
            raise ValueError("Legacy and generated adaptor memory dimensions do not match")

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

    def train_generated_branch_only(self) -> list[str]:
        """Freeze the Engram reader and expose only generated parameters.

        This is intended for downstream task supervision of ``dual_reader``
        checkpoints.  The direct Engram value/reader/gate, and the shared
        backbone-query normalization, remain fixed.  The generator, generated
        cross-attention reader, output projection, and generated gate stay
        trainable.  This parameter boundary is independent of runtime reader
        mode: use ``both`` to learn a correction on top of the frozen Engram
        output, or ``generated_only`` for the legacy ablation.

        Returns:
            Names of the parameters left trainable, for logging and tests.
        """
        if self.fusion_type != "dual_reader":
            raise ValueError(
                "Generated-branch-only training requires fusion_type='dual_reader'"
            )

        for parameter in self.parameters():
            parameter.requires_grad = False

        generated_modules = [
            self.cue_projection,
            self.generator_layers,
            self.reader_query,
            self.reader_attention,
            self.reader_norm,
            self.output_projection,
            self.norm_memory,
            self.memory_key_projection,
        ]
        for module in generated_modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

        self.latent_queries.requires_grad = True
        if self.learned_cues is not None:
            self.learned_cues.requires_grad = True
        self.gate_bias.requires_grad = True

        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]
