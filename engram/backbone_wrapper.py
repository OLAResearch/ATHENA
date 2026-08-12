"""HuggingFace model wrapper with forward hooks for memory injection."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adaptor import build_adaptor
from .memory import EngramMemory


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
    ):
        super().__init__()
        self.model_name = model_name
        self.condition = condition
        self.device = device
        self.adaptor_branches = adaptor_branches
        self.architecture = architecture
        self.generator_cue_source = generator_cue_source

        # Load backbone
        # Phi-4-mini's custom modeling code is incompatible with transformers 5.x;
        # use built-in phi3 support instead. Only enable trust_remote_code for
        # models that actually need it (e.g., Qwen).
        needs_remote_code = "Phi" not in model_name
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=needs_remote_code,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=needs_remote_code)

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
        if self.adaptor is not None:
            self.adaptor = self.adaptor.to(device)

        # Storage for gate activations (populated by hook)
        self._last_gate_values: Optional[torch.Tensor] = None
        self._current_forward_gate_values: list[torch.Tensor] = []

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

            def hook_fn(module, input, output, adaptor=adaptor):
                if adaptor is None:
                    return output

                # Extract hidden states from the layer output
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Get memory vectors (may be None for ffn_only condition)
                mem_vectors = self._get_memory_vectors(hidden_states)

                # For conditions that don't use memory (ffn_only), mem_vectors
                # will be None. The FFNOnlyAdaptor ignores the mem argument,
                # so we pass None through and let the adaptor handle it.
                # For memory-based conditions, mem_vectors must exist.
                allow_no_memory = (
                    self.condition == "ffn_only"
                    or (
                        self.architecture == "generative"
                        and self.generator_cue_source == "learned"
                    )
                )
                if mem_vectors is None and not allow_no_memory:
                    return output

                # Compute adaptor contribution
                h_float = hidden_states.float()
                mem_float = mem_vectors.float() if mem_vectors is not None else None
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
        if self.memory is None:
            return None

        if self._current_hash_indices is not None:
            return self.memory.forward_from_indices(self._current_hash_indices)
        elif self._current_canon_ids is not None:
            return self.memory(self._current_canon_ids)
        return None

    def set_canon_ids(self, canon_ids: torch.LongTensor) -> None:
        """Set canonical IDs for the next forward pass."""
        self._current_canon_ids = canon_ids
        self._current_hash_indices = None

    def set_hash_indices(self, indices: torch.LongTensor) -> None:
        """Set pre-computed hash indices for cross-tokenizer mode."""
        self._current_hash_indices = indices
        self._current_canon_ids = None

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass through backbone with memory injection via hook.

        Memory injection happens automatically in the registered hook.
        Call set_canon_ids() or set_hash_indices() before this.
        """
        self._current_forward_gate_values = []
        outputs = self.backbone(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        if not self._current_forward_gate_values:
            self._last_gate_values = None
        elif len(self._current_forward_gate_values) == 1:
            self._last_gate_values = self._current_forward_gate_values[0]
        else:
            self._last_gate_values = torch.stack(self._current_forward_gate_values, dim=0)
        return outputs

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
