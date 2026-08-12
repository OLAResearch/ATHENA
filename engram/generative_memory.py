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

        causal Engram cues -> generator latents
        backbone hidden state + generator latents -> reader -> gated residual

    The generator never receives ``h``.  With ``cue_source='learned'``, it
    receives only learned cue tokens, providing a parameter-capacity control.
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
    ):
        super().__init__()
        if reader_type not in ("cross_attention", "mean"):
            raise ValueError(f"Unknown reader_type: {reader_type}")
        if cue_source not in ("engram", "learned"):
            raise ValueError(f"Unknown cue_source: {cue_source}")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if min(num_latents, num_layers, num_heads, cue_window) < 1:
            raise ValueError("Generator sizes, layers, heads, and cue_window must be positive")
        if cue_source == "engram" and d_mem < 1:
            raise ValueError("Engram cue source requires d_mem > 0")

        self.d_model = d_model
        self.d_mem = d_mem
        self.reader_type = reader_type
        self.cue_source = cue_source
        self.num_latents = num_latents
        self.hidden_size = hidden_size
        self.cue_window = cue_window

        if cue_source == "engram":
            self.cue_projection = nn.Linear(d_mem, hidden_size)
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
        self.norm_h = RMSNorm(d_model)
        self.norm_memory = RMSNorm(d_model)
        self.gate_bias = nn.Parameter(torch.tensor(gate_bias_init))

        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        if self.learned_cues is not None:
            nn.init.normal_(self.learned_cues, mean=0.0, std=0.02)
        # A small non-zero initialization preserves initial stability while
        # allowing generator and reader gradients on the first backward pass.
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)

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
        self, mem: torch.Tensor | None, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        if self.cue_source == "engram":
            if mem is None:
                raise ValueError("Engram cue source requires memory vectors")
            windows, padding_mask = self._causal_windows(mem)
            cues = self.cue_projection(windows)
            cues = cues.reshape(batch_size * seq_len, self.cue_window, self.hidden_size)
            padding_mask = padding_mask.reshape(batch_size * seq_len, self.cue_window)
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
        self, h: torch.Tensor, mem: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 3 or h.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected h shape (B, T, {self.d_model}), got {tuple(h.shape)}"
            )
        batch_size, seq_len, _ = h.shape
        latents = self._generate(mem, batch_size, seq_len)

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
        memory_value = self.output_projection(summary)
        gate_logit = (
            self.norm_h(h) * self.norm_memory(memory_value)
        ).sum(dim=-1) / math.sqrt(self.d_model)
        gate = torch.sigmoid(gate_logit + self.gate_bias)
        return gate.unsqueeze(-1) * memory_value, gate

    @property
    def trainable_param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
