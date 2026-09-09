from types import SimpleNamespace

import torch

from engram.generative_memory import GenerativeMemoryAdaptor
from scripts.eval_openqa import configure_adaptive_routers
from scripts.train_advantage_router import (
    configure_router_training,
    load_experts_allowing_new_router,
    residual_usage,
)
from engram.tri_memory import TriMemoryAdaptor


def _adaptor(adaptive_router: bool):
    return GenerativeMemoryAdaptor(
        8,
        4,
        hidden_size=8,
        num_heads=2,
        fusion_type="dual_reader",
        adaptive_router=adaptive_router,
    )


def test_existing_expert_checkpoint_can_initialize_only_new_router(tmp_path):
    source = _adaptor(adaptive_router=False)
    checkpoint = tmp_path / "experts.pt"
    torch.save(source.state_dict(), checkpoint)
    target = _adaptor(adaptive_router=True)

    missing, unexpected = load_experts_allowing_new_router(target, str(checkpoint))

    assert missing and all("router" in name for name in missing)
    assert not unexpected
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[name]), name


def test_residual_usage_reaches_every_injection_layer_router():
    modules = torch.nn.ModuleList([_adaptor(True), _adaptor(True)])
    wrapper = SimpleNamespace(adaptor=modules)
    names = configure_router_training(wrapper, temperature=1.0)
    h = torch.randn(2, 4, 8)
    mem = torch.randn(2, 4, 4)
    for module in modules:
        module(h, mem)

    loss = residual_usage(wrapper)
    loss.backward()

    assert names and all("router" in name for name in names)
    assert all(module.router[-1].bias.grad is not None for module in modules)
    assert all(module.router[-1].bias.grad.abs().sum() > 0 for module in modules)


def test_router_runtime_temperature_is_restored_from_config():
    modules = torch.nn.ModuleList([_adaptor(True), _adaptor(True)])
    wrapper = SimpleNamespace(adaptor=modules)

    configure_adaptive_routers(
        wrapper, {"router_temperature": 0.35, "router_hard": True}
    )

    assert all(module.router_temperature == 0.35 for module in modules)
    assert all(module.router_hard is True for module in modules)


def test_safe_router_runtime_controls_are_restored_from_config():
    modules = torch.nn.ModuleList(
        [
            TriMemoryAdaptor(8, 4, hidden_size=8, num_heads=2),
            TriMemoryAdaptor(8, 4, hidden_size=8, num_heads=2),
        ]
    )
    wrapper = SimpleNamespace(adaptor=modules)

    configure_adaptive_routers(
        wrapper,
        {
            "router_temperature": 0.7,
            "router_hard": True,
            "safe_residual_threshold": 1.8,
            "safe_residual_scale": 0.4,
        },
    )

    assert all(module.safe_router_threshold == 1.8 for module in modules)
    assert all(module.safe_router_scale == 0.4 for module in modules)
