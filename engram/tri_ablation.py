"""Structural controls for the three-source reader, preserving router contracts."""
import torch
from torch import nn
from .tri_memory import TriMemoryAdaptor


class AblatedTriMemoryAdaptor(TriMemoryAdaptor):
    def __init__(self, *, condition, **kwargs):
        super().__init__(**kwargs)
        self.ablation_condition = condition
        if condition == 'affine_stitch':
            # E reads memory directly; GE/GH retain their generators but replace
            # query-dependent attention with affine reading of mean latents.
            self.affine_e_bias = nn.Parameter(torch.zeros(self.d_model))
            self.affine_generated_bias = nn.Parameter(torch.zeros(2, self.d_model))
            self.reader_type = 'mean'
            self.reader_query = None
            self.reader_attention = None
            self.reader_norm = nn.Identity()
        self.freeze_unused_ablation_parameters()

    def freeze_unused_ablation_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(('engram_key_projection.', 'engram_reader_norm.',
                                'engram_gate_bias', 'generated_key_projection.',
                                'generated_reader_norm.', 'generated_gate_bias')):
                parameter.requires_grad = False

    def _direct_engram(self, h, mem):
        value = self.engram_value_projection(mem)
        if self.ablation_condition == 'affine_stitch':
            value = value + self.affine_e_bias
        gates = h.new_ones(self.num_branches, *h.shape[:2])
        return value, gates

    def _generated_expert(self, source_index, h, mem, cue_mem, context_h=None):
        latents = self._generate(source_index, h, mem, cue_mem, context_h=context_h)
        reader_h = context_h if source_index == 1 and context_h is not None else h
        value = self._read_generated(source_index, reader_h, latents)
        if self.ablation_condition == 'affine_stitch':
            value = value + self.affine_generated_bias[source_index]
        return value, h.new_ones(self.num_branches, *h.shape[:2])
