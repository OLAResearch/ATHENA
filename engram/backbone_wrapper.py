"""HuggingFace model wrapper with forward hooks for memory injection."""

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adaptor import build_adaptor
from .hf_utils import resolve_pretrained_source
from .memory import EngramMemory


def move_backbone_to_device_staged(
    backbone: nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Move a decoder-only HF backbone one block at a time.

    LUMI's ROCm stack can SIGSEGV while copying a fully materialized Mistral
    model with one top-level ``model.to(cuda)`` call.  Moving the embedding,
    each transformer block, normalization, and output head separately keeps
    the exact same model placement while avoiding that large monolithic copy.
    Models without the usual ``model.layers`` layout retain the normal path.
    """
    if device.type != "cuda":
        return backbone.to(device)
    transformer = getattr(backbone, "model", None)
    layers = getattr(transformer, "layers", None)
    if transformer is None or layers is None:
        return backbone.to(device)
    for name, module in transformer.named_children():
        if name == "layers":
            for layer in module:
                layer.to(device=device, dtype=dtype)
        else:
            module.to(device=device, dtype=dtype)
    for name, module in backbone.named_children():
        if name != "model":
            module.to(device=device, dtype=dtype)
    return backbone


class BackboneWrapper(nn.Module):
    """Wraps a HF causal LM with Engram memory injection.

    Memory is injected at layer (num_layers // 3) via a forward hook
    on the corresponding transformer layer. The backbone is always frozen;
    only the adaptor (and optionally memory) are trainable.
    """

    def __init__(
        self,
        model_name: str,
        memory: Optional[EngramMemory],
        condition: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        gate_bias_init: float = 0.0,
        injection_layers: Optional[list[int]] = None,
        adaptor_branches: int = 1,
        memory_dim: Optional[int] = None,
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
    ):
        super().__init__()
        self.model_name = model_name
        self.condition = condition
        self.device = device
        self.adaptor_branches = adaptor_branches
        self.architecture = architecture
        self.generator_cue_source = generator_cue_source
        self.generator_fusion_type = generator_fusion_type

        # Load backbone
        # Phi-4-mini's custom modeling code is incompatible with transformers 5.x;
        # use built-in phi3 support instead. Only enable trust_remote_code for
        # models that actually need it (e.g., Qwen).
        needs_remote_code = "Phi" not in model_name
        pretrained_source = resolve_pretrained_source(model_name)
        cpu_load_dtype = (
            torch.float32
            if device.type == "cuda" and os.environ.get("ATHENA_CPU_FP32_LOAD") == "1"
            else dtype
        )
        model_kwargs = dict(
            torch_dtype=cpu_load_dtype,
            trust_remote_code=needs_remote_code,
        )
        if device.type == "cuda" and os.environ.get("ATHENA_STAGED_DEVICE_TRANSFER") == "1":
            self.backbone = AutoModelForCausalLM.from_pretrained(
                pretrained_source, **model_kwargs
            )
            self.backbone = move_backbone_to_device_staged(self.backbone, device, dtype=dtype)
        elif device.type == "cuda" and os.environ.get("ATHENA_DIRECT_DEVICE_MAP") == "1":
            # On LUMI ROCm, moving a fully materialized Mistral checkpoint via
            # model.to(cuda) can SIGSEGV before any adaptor/checkpoint code
            # runs. Accelerate's direct placement avoids that whole-model copy.
            model_kwargs.update(device_map={"": str(device)}, low_cpu_mem_usage=True)
            self.backbone = AutoModelForCausalLM.from_pretrained(pretrained_source, **model_kwargs)
        else:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                pretrained_source, **model_kwargs
            ).to(device=device, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_source, trust_remote_code=needs_remote_code
        )

        # Determine model architecture
        self.d_model = self._get_d_model()
        self.num_layers = self._get_num_layers()
        if injection_layers is None:
            injection_layers = [self.num_layers // 3]
        self.injection_layers = self._normalize_injection_layers(injection_layers)

        # Memory and adaptor
        self.memory = memory
        if memory is not None:
            self.memory = memory.to(device)

        d_mem = memory.d_mem if memory is not None else (memory_dim or 0)
        adaptor_kwargs = dict(
            gate_bias_init=gate_bias_init,
            num_branches=adaptor_branches,
            architecture=architecture,
            reader_type=reader_type,
            generator_cue_source=generator_cue_source,
            generator_num_latents=generator_num_latents,
            generator_hidden_size=generator_hidden_size,
            generator_layers=generator_layers,
            generator_heads=generator_heads,
            generator_cue_window=generator_cue_window,
            generator_fusion_type=generator_fusion_type,
            generator_adaptive_router=generator_adaptive_router,
            generator_router_hidden_size=generator_router_hidden_size,
            generator_router_semantic_size=generator_router_semantic_size,
            generator_router_expert_mode=generator_router_expert_mode,
            generator_source_adapter_rank=generator_source_adapter_rank,
            generator_loop_rounds=generator_loop_rounds,
            generator_loop_workspace_size=generator_loop_workspace_size,
            generator_loop_gate_max=generator_loop_gate_max,
        )
        if len(self.injection_layers) == 1:
            self.adaptor = build_adaptor(
                condition,
                self.d_model,
                d_mem,
                **adaptor_kwargs,
            )
        else:
            self.adaptor = nn.ModuleList([
                build_adaptor(
                    condition,
                    self.d_model,
                    d_mem,
                    **adaptor_kwargs,
                )
                for _ in self.injection_layers
            ])
        # FFN-only warm starts are loaded by train_adaptor before the large
        # parameter block is moved to ROCm.  Keeping this adaptor on CPU here
        # avoids a GPU->CPU transition that can SIGSEGV on LUMI's ROCm stack.
        if self.adaptor is not None and condition != "ffn_only":
            self.adaptor = self.adaptor.to(device)

        # Storage for gate activations (populated by hook)
        self._last_gate_values: Optional[torch.Tensor] = None
        self._current_forward_gate_values: list[torch.Tensor] = []
        self._capture_context_hidden = False
        self._context_hidden_by_slot: dict[int, torch.Tensor] = {}
        # Joint training evaluates several reader modes on the same batch.
        # GH's clean frozen-backbone stream is identical for all of them, so
        # keep it for the current batch and invalidate it in set_*_ids().
        self._shared_clean_context_hidden_by_slot: dict[int, torch.Tensor] = {}
        self._shared_clean_context_batch_key = None
        # A multi-layer tri-reader needs a clean backbone stream for GH.  When
        # decoding with a KV cache, that stream has its own cache: reusing the
        # main cache would append the clean pass to the cache that already
        # contains E/GE residuals and silently shift all subsequent positions.
        self._clean_context_past_key_values = None

        # Register injection hook
        if self.adaptor is not None:
            self._hook_handles = self._register_injection_hooks()
        else:
            self._hook_handles = []

        # Storage for canon_ids (set before forward pass)
        self._current_canon_ids: Optional[torch.LongTensor] = None
        self._current_hash_indices: Optional[torch.LongTensor] = None

    def _get_text_config(self):
        """Get the text config, handling nested configs (e.g., Qwen3.5 multimodal)."""
        config = self.backbone.config
        if hasattr(config, "text_config"):
            return config.text_config
        return config

    def _get_d_model(self) -> int:
        config = self._get_text_config()
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError(f"Cannot determine d_model for {self.model_name}")

    def _get_num_layers(self) -> int:
        config = self._get_text_config()
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError(f"Cannot determine num_layers for {self.model_name}")

    def _get_layers(self) -> nn.ModuleList:
        """Get the transformer layer list from the backbone."""
        model = self.backbone
        # Pythia / GPT-NeoX
        if hasattr(model, "gpt_neox"):
            return model.gpt_neox.layers
        # OpenAI GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        # Llama / TinyLlama
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        raise ValueError(f"Cannot find layer list for {self.model_name}")

    def _normalize_injection_layers(self, injection_layers: list[int]) -> list[int]:
        normalized = []
        for layer_idx in injection_layers:
            idx = int(layer_idx)
            if idx < 0:
                idx = self.num_layers + idx
            if idx < 0 or idx >= self.num_layers:
                raise ValueError(
                    f"Injection layer {layer_idx} is out of range for {self.model_name} "
                    f"(num_layers={self.num_layers})"
                )
            normalized.append(idx)
        if not normalized:
            raise ValueError("At least one injection layer must be provided")
        return normalized

    def _get_adaptor_for_layer(self, layer_slot: int):
        if isinstance(self.adaptor, nn.ModuleList):
            return self.adaptor[layer_slot]
        return self.adaptor

    def _register_injection_hooks(self):
        """Register forward hooks on the configured injection layers."""
        layers = self._get_layers()
        handles = []

        for layer_slot, injection_layer in enumerate(self.injection_layers):
            target_layer = layers[injection_layer]
            adaptor = self._get_adaptor_for_layer(layer_slot)

            def hook_fn(module, input, output, adaptor=adaptor, layer_slot=layer_slot):
                if adaptor is None:
                    return output

                # Extract hidden states from the layer output
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # A tri-reader GH stream must not inherit an E/GE residual
                # injected at an earlier layer.  During the optional clean
                # context pass, capture the frozen backbone representation at
                # each injection point before any adaptor is applied.
                if (
                    self._capture_context_hidden
                    and getattr(adaptor, "fusion_type", None) == "tri_reader"
                ):
                    self._context_hidden_by_slot[layer_slot] = hidden_states.detach()
                    return output

                # Get direct-reader vectors plus enough recent memory history
                # for the generated cue window.  Cached decoding normally has
                # one hidden token but the generator still needs its preceding
                # Engram cues.
                needs_memory_fn = getattr(adaptor, "needs_engram_memory", None)
                needs_memory = (
                    bool(needs_memory_fn())
                    if needs_memory_fn is not None
                    else not (
                        self.architecture == "generative"
                        and self.generator_cue_source == "learned"
                    )
                )
                needs_history_fn = getattr(
                    adaptor, "needs_engram_cue_history", None
                )
                needs_history = (
                    bool(needs_history_fn())
                    if needs_history_fn is not None
                    else (
                        self.architecture == "generative"
                        and self.generator_cue_source == "engram"
                    )
                )
                cue_window = (
                    int(getattr(adaptor, "cue_window", 1))
                    if needs_history
                    else 1
                )
                if needs_memory:
                    mem_vectors, cue_mem_vectors = self._get_memory_context(
                        hidden_states, cue_window=cue_window
                    )
                else:
                    mem_vectors, cue_mem_vectors = None, None

                # For conditions that don't use memory (ffn_only), mem_vectors
                # will be None. The FFNOnlyAdaptor ignores the mem argument,
                # so we pass None through and let the adaptor handle it.
                # For memory-based conditions, mem_vectors must exist.
                allow_no_memory = self.condition == "ffn_only" or not needs_memory
                if mem_vectors is None and not allow_no_memory:
                    return output

                # Compute adaptor contribution
                h_float = hidden_states.float()
                mem_float = mem_vectors.float() if mem_vectors is not None else None
                if self.architecture == "generative" and self.condition != "ffn_only":
                    cue_mem_float = (
                        cue_mem_vectors.float()
                        if cue_mem_vectors is not None
                        else None
                    )
                    if getattr(adaptor, "fusion_type", None) == "tri_reader":
                        context_hidden = self._context_hidden_by_slot.get(layer_slot)
                        contribution, gate_values = adaptor(
                            h_float,
                            mem_float,
                            cue_mem=cue_mem_float,
                            context_h=(
                                context_hidden.float()
                                if context_hidden is not None
                                else None
                            ),
                        )
                    else:
                        contribution, gate_values = adaptor(
                            h_float, mem_float, cue_mem=cue_mem_float
                        )
                else:
                    contribution, gate_values = adaptor(h_float, mem_float)

                self._current_forward_gate_values.append(gate_values.detach())

                hidden_states = hidden_states + contribution.to(hidden_states.dtype)

                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                return hidden_states

            handles.append(target_layer.register_forward_hook(hook_fn))

        return handles

    def _get_memory_vectors(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Get memory vectors, using pre-computed indices or canon_ids."""
        direct, _ = self._get_memory_context(hidden_states, cue_window=1)
        return direct

    def _get_memory_context(
        self, hidden_states: torch.Tensor, cue_window: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return direct memory plus recent cue history for current hidden states.

        For a full-sequence forward both tensors span the current sequence.
        During KV-cache decoding, direct memory spans only the new token(s),
        while cue memory additionally contains up to ``cue_window - 1`` recent
        positions.  Hashing is performed before truncation so vocabulary-mode
        n-grams retain their true left context as well.
        """
        if self.memory is None:
            return None, None
        if cue_window < 1:
            raise ValueError("cue_window must be positive")

        context = (
            self._current_hash_indices
            if self._current_hash_indices is not None
            else self._current_canon_ids
        )
        if context is None:
            return None, None

        hidden_length = hidden_states.shape[1]
        context_length = context.shape[1]
        if context_length < hidden_length:
            raise ValueError(
                "Memory context is shorter than the transformer hidden sequence: "
                f"context={context_length}, hidden={hidden_length}"
            )
        cue_length = min(context_length, hidden_length + cue_window - 1)
        if self._current_hash_indices is not None:
            cue_vectors = self.memory.forward_from_indices(context[:, -cue_length:])
        else:
            # Compute n-gram hashes on the complete canonical prefix before
            # selecting the recent cue positions.
            cue_vectors = self.memory(context)[:, -cue_length:]
        return cue_vectors[:, -hidden_length:], cue_vectors

    @staticmethod
    def _tail_context(context: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Align precomputed memory context with cached-decoding hidden states.

        During the initial forward pass the transformer sees the complete prompt,
        so the canonical/hash context has the same sequence length.  With KV-cache
        decoding, later forward passes normally contain only the newest token while
        canonicalization still needs the complete generated prefix to construct the
        correct n-gram.  In that case only the final context positions belong to the
        hidden states handled by the current transformer call.
        """
        hidden_length = hidden_states.shape[1]
        context_length = context.shape[1]
        if context_length < hidden_length:
            raise ValueError(
                "Memory context is shorter than the transformer hidden sequence: "
                f"context={context_length}, hidden={hidden_length}"
            )
        if context_length == hidden_length:
            return context
        return context[:, -hidden_length:]

    def set_canon_ids(self, canon_ids: torch.LongTensor) -> None:
        """Set canonical IDs for the next forward pass."""
        self._current_canon_ids = canon_ids
        self._current_hash_indices = None
        self._shared_clean_context_hidden_by_slot = {}
        self._shared_clean_context_batch_key = None
        # Do not invalidate the clean GH KV stream here.  Cached generation
        # calls this setter once per newly generated token; the clean stream
        # must therefore survive across those calls and be advanced in
        # ``forward`` alongside the injected main stream.  A new sequence is
        # detected by ``past_key_values is None`` and resets the stream before
        # its full-prefix clean pass.

    def set_hash_indices(self, indices: torch.LongTensor) -> None:
        """Set pre-computed hash indices for cross-tokenizer mode."""
        self._current_hash_indices = indices
        self._current_canon_ids = None
        self._shared_clean_context_hidden_by_slot = {}
        self._shared_clean_context_batch_key = None
        # Keep the clean GH KV stream alive during cached decoding.  The
        # forward path resets it when a new sequence starts (no incoming
        # ``past_key_values``), so clearing it on every token would make the
        # next cached call fail before it can advance the clean stream.

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **backbone_kwargs,
    ):
        """Forward pass through backbone with memory injection via hook.

        Memory injection happens automatically in the registered hook.
        Call set_canon_ids() or set_hash_indices() before this.
        """
        self._current_forward_gate_values = []
        self._context_hidden_by_slot = {}

        # With one injection point the hook's hidden state is already a clean
        # causal backbone state.  With multiple points, run one frozen,
        # no-injection pass to provide GH with a context stream independent of
        # E and GE residuals from earlier points.  The pass is skipped for
        # legacy/dual adaptors and therefore does not change their cost.
        if self._needs_clean_tri_context():
            incoming_past_key_values = backbone_kwargs.get("past_key_values")
            if incoming_past_key_values is None:
                batch_key = (
                    input_ids.data_ptr(),
                    tuple(input_ids.shape),
                    tuple(input_ids.stride()),
                    input_ids.device.type,
                    input_ids.device.index,
                )
                if (
                    self._shared_clean_context_hidden_by_slot
                    and self._shared_clean_context_batch_key == batch_key
                ):
                    # Joint mode paths share this exact input batch.  The
                    # clean stream is frozen and detached, so reusing it is
                    # numerically equivalent to another no-grad pass.
                    self._context_hidden_by_slot = dict(
                        self._shared_clean_context_hidden_by_slot
                    )
                else:
                    # The first GH path for a batch computes and stores the
                    # clean context.  set_*_ids() invalidates this cache when
                    # the next batch is installed.
                    self._clean_context_past_key_values = None
                    clean_backbone_kwargs = dict(backbone_kwargs)
                    clean_backbone_kwargs["past_key_values"] = None
                    self._capture_context_hidden = True
                    try:
                        with torch.no_grad():
                            clean_outputs = self.backbone(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **clean_backbone_kwargs,
                            )
                            self._clean_context_past_key_values = getattr(
                                clean_outputs, "past_key_values", None
                            )
                    finally:
                        self._capture_context_hidden = False
                    self._shared_clean_context_hidden_by_slot = dict(
                        self._context_hidden_by_slot
                    )
                    self._shared_clean_context_batch_key = batch_key
            elif self._clean_context_past_key_values is None:
                raise ValueError(
                    "GH clean context KV cache is unavailable for a cached "
                    "forward; restart decoding with past_key_values=None"
                )
            if incoming_past_key_values is not None:
                # Cached decoding needs a separate clean KV stream.  It must
                # continue from the clean pass, never from the injected main
                # stream.
                clean_backbone_kwargs = dict(backbone_kwargs)
                clean_backbone_kwargs["past_key_values"] = (
                    self._clean_context_past_key_values
                )
                self._capture_context_hidden = True
                try:
                    with torch.no_grad():
                        clean_outputs = self.backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **clean_backbone_kwargs,
                        )
                        self._clean_context_past_key_values = getattr(
                            clean_outputs, "past_key_values", None
                        )
                finally:
                    self._capture_context_hidden = False
        else:
            # Do not let a clean cache from an earlier GH run be reused if the
            # caller switches to a non-GH runtime mode and later starts a new
            # cached sequence.
            self._clean_context_past_key_values = None

        outputs = self.backbone(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            **backbone_kwargs,
        )
        if not self._current_forward_gate_values:
            self._last_gate_values = None
        elif len(self._current_forward_gate_values) == 1:
            self._last_gate_values = self._current_forward_gate_values[0]
        else:
            self._last_gate_values = torch.stack(self._current_forward_gate_values, dim=0)
        return outputs

    def _needs_clean_tri_context(self) -> bool:
        if len(self.injection_layers) < 2:
            return False
        adaptors = (
            list(self.adaptor)
            if isinstance(self.adaptor, nn.ModuleList)
            else [self.adaptor]
        )
        for adaptor in adaptors:
            if (
                adaptor is None
                or getattr(adaptor, "fusion_type", None) != "tri_reader"
                or not hasattr(adaptor, "active_expert_indices")
            ):
                continue
            if 2 in adaptor.active_expert_indices():
                return True
        return False

    def train(self, mode: bool = True):
        """Train adaptors while keeping the frozen backbone deterministic."""
        super().train(mode)
        self.backbone.eval()
        return self

    def get_last_gate_values(self) -> Optional[torch.Tensor]:
        return self._last_gate_values

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_memory(self) -> None:
        """Freeze memory table parameters."""
        if self.memory is not None:
            for param in self.memory.parameters():
                param.requires_grad = False

    def unfreeze_memory(self) -> None:
        """Unfreeze memory (for train_from_scratch condition)."""
        if self.memory is not None:
            for param in self.memory.parameters():
                param.requires_grad = True

    def get_trainable_params(self) -> list:
        """Get list of trainable parameters."""
        params = []
        # Backbone params (only if unfrozen)
        params.extend(
            p for p in self.backbone.parameters() if p.requires_grad
        )
        if self.adaptor is not None:
            params.extend(
                p for p in self.adaptor.parameters() if p.requires_grad
            )
        if self.memory is not None:
            params.extend(
                p for p in self.memory.parameters() if p.requires_grad
            )
        return params

    def get_grad_norms(self) -> dict:
        """Get gradient norms for monitoring (frozen params should be zero)."""
        norms = {
            "backbone": 0.0,
            "adaptor": 0.0,
            "memory": 0.0,
        }

        for name, param in self.backbone.named_parameters():
            if param.grad is not None:
                norms["backbone"] += param.grad.norm().item() ** 2

        if self.adaptor is not None:
            for name, param in self.adaptor.named_parameters():
                if param.grad is not None:
                    norms["adaptor"] += param.grad.norm().item() ** 2

        if self.memory is not None:
            for name, param in self.memory.named_parameters():
                if param.grad is not None:
                    norms["memory"] += param.grad.norm().item() ** 2

        return {k: v**0.5 for k, v in norms.items()}

    def cleanup(self) -> None:
        """Remove hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._clean_context_past_key_values = None
