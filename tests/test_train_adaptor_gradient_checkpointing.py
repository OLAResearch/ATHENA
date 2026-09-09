from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from engram.adaptor import MultiBranchEngramAdaptor
from engram.generative_memory import GenerativeMemoryAdaptor
from scripts.train_adaptor import (
    configure_joint_engram_generated_router_training,
    configure_generated_residual_training,
    enable_frozen_backbone_gradient_checkpointing,
    initialize_direct_engram_readers,
    load_initial_adaptor,
)


class _CheckpointableBackbone:
    def __init__(self):
        self.checkpointing_enabled = False
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


def test_frozen_backbone_checkpointing_keeps_inputs_differentiable():
    backbone = _CheckpointableBackbone()
    wrapper = SimpleNamespace(backbone=backbone)

    enable_frozen_backbone_gradient_checkpointing(wrapper)

    assert backbone.checkpointing_enabled is True
    assert backbone.input_grads_enabled is True


def test_frozen_backbone_checkpointing_rejects_unsupported_backbone():
    class UnsupportedBackbone:
        def gradient_checkpointing_enable(self):
            pass

    wrapper = SimpleNamespace(backbone=UnsupportedBackbone())

    with pytest.raises(RuntimeError, match="enable_input_require_grads"):
        enable_frozen_backbone_gradient_checkpointing(wrapper)


def _dual_reader():
    return GenerativeMemoryAdaptor(
        8,
        4,
        hidden_size=8,
        num_heads=2,
        num_branches=2,
        fusion_type="dual_reader",
    )


def test_load_initial_adaptor_accepts_complete_multilayer_checkpoint(tmp_path):
    torch.manual_seed(3)
    source = nn.ModuleList([_dual_reader(), _dual_reader()])
    checkpoint = tmp_path / "adaptor.pt"
    torch.save(source.state_dict(), checkpoint)

    torch.manual_seed(4)
    target = nn.ModuleList([_dual_reader(), _dual_reader()])
    wrapper = SimpleNamespace(adaptor=target)
    loaded = load_initial_adaptor(wrapper, str(checkpoint))

    assert loaded == sum(value.numel() for value in source.state_dict().values())
    for name, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[name]), name


def test_generated_residual_training_freezes_direct_engram_reader():
    adaptor = _dual_reader()
    wrapper = SimpleNamespace(adaptor=adaptor)
    names = configure_generated_residual_training(wrapper, "both")

    assert names
    assert adaptor.dual_reader_mode == "both"
    assert adaptor.output_projection.weight.requires_grad
    assert adaptor.gate_bias.requires_grad
    assert not adaptor.engram_value_projection.weight.requires_grad
    assert not adaptor.engram_gate_bias.requires_grad
    assert all(not parameter.requires_grad for parameter in adaptor.engram_key_projection.parameters())


def test_fair_joint_and_engram_baseline_share_exact_step_zero_reader():
    baseline = nn.ModuleList([
        MultiBranchEngramAdaptor(8, 4, num_branches=2),
        MultiBranchEngramAdaptor(8, 4, num_branches=2),
    ])
    joint = nn.ModuleList([
        GenerativeMemoryAdaptor(
            8,
            4,
            hidden_size=8,
            num_heads=2,
            num_branches=2,
            fusion_type="dual_reader",
            adaptive_router=True,
        )
        for _ in range(2)
    ])
    baseline_wrapper = SimpleNamespace(
        adaptor=baseline,
        adaptor_branches=2,
        d_model=8,
        memory=SimpleNamespace(d_mem=4),
    )
    joint_wrapper = SimpleNamespace(
        adaptor=joint,
        adaptor_branches=2,
        d_model=8,
        memory=SimpleNamespace(d_mem=4),
    )
    initialize_direct_engram_readers(baseline_wrapper, seed=42)
    initialize_direct_engram_readers(joint_wrapper, seed=42)
    names, reader_params, router_params = (
        configure_joint_engram_generated_router_training(
            joint_wrapper, router_init_alpha=0.95
        )
    )

    assert names
    assert reader_params and router_params
    assert not set(map(id, reader_params)) & set(map(id, router_params))
    h = torch.randn(2, 3, 8)
    mem = torch.randn(2, 3, 4)
    for baseline_adaptor, joint_adaptor in zip(baseline, joint):
        expected, _ = baseline_adaptor(h, mem)
        joint_adaptor.set_dual_reader_mode("both")
        actual_both, _ = joint_adaptor(h, mem)
        joint_adaptor.set_dual_reader_mode("routed")
        actual_routed, _ = joint_adaptor(h, mem)
        assert torch.equal(expected, actual_both)
        assert torch.allclose(expected, actual_routed, atol=1e-7, rtol=1e-6)
        assert joint_adaptor.get_last_generated_residual() is not None
