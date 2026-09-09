import torch
import pytest

from engram.tri_memory import TRI_READER_SUBSETS, TriMemoryAdaptor
from scripts.train_adaptor import (
    configure_joint_tri_reader_training,
    tri_reader_diagnostics,
    tri_joint_loss_terms,
    tri_subset_reader_distillation_loss,
)


class _Wrapper:
    def __init__(self, adaptor):
        self.adaptor = adaptor


def test_tri_joint_distillation_weight_is_not_path_normalized_twice():
    lm = torch.tensor(2.0, requires_grad=True)
    distillation = torch.tensor(3.0, requires_grad=True)
    objective, weighted_lm, weighted_distillation = tri_joint_loss_terms(
        lm, distillation, source_weight=1.0, normalizer=8.0, distillation_weight=0.2
    )

    assert weighted_lm.item() == pytest.approx(0.25)
    assert weighted_distillation.item() == pytest.approx(0.6)
    assert objective.item() == pytest.approx(0.85)

    objective.backward()
    assert lm.grad.item() == pytest.approx(0.125)
    assert distillation.grad.item() == pytest.approx(0.2)


def test_unified_training_configuration_and_subset_distillation():
    torch.manual_seed(101)
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    wrapper = _Wrapper(adaptor)
    _, expert_params, reader_params = configure_joint_tri_reader_training(
        wrapper, unified_subset_reader=True
    )
    assert adaptor.tri_reader_mode == "tri_subset_soft_fused"
    assert any(
        parameter is adaptor.subset_router[-1].weight for parameter in reader_params
    )
    assert all(parameter.requires_grad for parameter in expert_params + reader_params)

    h = torch.randn(2, 4, 16)
    mem = torch.randn(2, 4, 8)
    adaptor(h, mem)
    endpoint_nlls = [torch.rand(2, 3) for _ in TRI_READER_SUBSETS]
    labels = torch.ones(2, 4, dtype=torch.long)
    loss, usage = tri_subset_reader_distillation_loss(
        wrapper, endpoint_nlls, labels
    )
    assert loss.item() > 0
    assert len(usage) == 7
    assert abs(sum(usage) - 1.0) < 1e-6
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in adaptor.subset_router.parameters()
    )


def test_subset_diagnostics_detect_singleton_only_degeneracy():
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    wrapper = _Wrapper(adaptor)
    adaptor._last_router_weights = torch.eye(3).unsqueeze(0)
    adaptor._last_subset_router_weights = torch.zeros(1, 3, 7)
    adaptor._last_subset_router_weights[0, 0, 0] = 1.0
    adaptor._last_subset_router_weights[0, 1, 1] = 1.0
    adaptor._last_subset_router_weights[0, 2, 2] = 1.0

    diagnostics = tri_reader_diagnostics(wrapper)

    assert diagnostics is not None
    assert diagnostics["subset_collapse"] is False
    assert diagnostics["subset_singleton_mass"] == 1.0
    assert diagnostics["subset_pair_mass"] == 0.0
    assert diagnostics["subset_triple_mass"] == 0.0
    assert diagnostics["subset_multi_expert_mass"] == 0.0
    assert diagnostics["subset_active_classes"] == 3
    assert diagnostics["subset_combination_collapse"] is True


def test_subset_reader_distillation_recovers_a_multi_expert_target():
    """The differentiable training path must escape the E singleton tie."""
    torch.manual_seed(202)
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    wrapper = _Wrapper(adaptor)
    adaptor.set_tri_reader_mode("tri_subset_soft_fused")

    # Isolate the seven-way Reader: the endpoint losses are a fixed teacher
    # signal, while the Reader logits are the only trainable quantities.
    for parameter in adaptor.parameters():
        parameter.requires_grad = False
    for parameter in adaptor.subset_router.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.SGD(adaptor.subset_router.parameters(), lr=1.0)
    h = torch.randn(2, 5, 16)
    mem = torch.randn(2, 5, 8)
    labels = torch.ones(2, 5, dtype=torch.long)
    endpoint_nlls = [torch.full((2, 4), 2.0) for _ in TRI_READER_SUBSETS]
    endpoint_nlls[6] = torch.zeros(2, 4)  # E+GE+GH is teacher-optimal.

    initial_loss = None
    for _ in range(30):
        adaptor(h, mem)
        loss, _ = tri_subset_reader_distillation_loss(
            wrapper, endpoint_nlls, labels
        )
        if initial_loss is None:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    adaptor(h, mem)
    weights = adaptor.get_last_subset_router_weights()
    assert weights is not None
    assert float(loss.item()) < initial_loss
    assert weights[..., 6].mean().item() > 0.6
    assert weights[..., 6].mean().item() > weights[..., 0].mean().item()
