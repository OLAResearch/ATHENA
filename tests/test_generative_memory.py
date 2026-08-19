import inspect

import torch

from engram.adaptor import build_adaptor
from engram.generative_memory import GenerativeMemoryAdaptor
from scripts.train_adaptor import setup_memory


def make_inputs(batch=2, seq=5, d_mem=8, d_model=16):
    torch.manual_seed(7)
    h = torch.randn(batch, seq, d_model, requires_grad=True)
    mem = torch.randn(batch, seq, d_mem)
    return h, mem


def test_causal_windows_shape_mask_and_values():
    adaptor = GenerativeMemoryAdaptor(16, 2, cue_window=3, hidden_size=8, num_heads=2)
    mem = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    windows, mask = adaptor._causal_windows(mem)
    assert windows.shape == (1, 3, 3, 2)
    assert mask.shape == (1, 3, 3)
    assert torch.equal(windows[0, 0], torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 10.0]]))
    assert torch.equal(windows[0, 2], torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    assert mask[0, 0].tolist() == [True, True, False]
    assert mask[0, 2].tolist() == [False, False, False]


def test_generator_forward_backward_cross_attention():
    h, mem = make_inputs()
    adaptor = GenerativeMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    contribution, gate = adaptor(h, mem)
    assert contribution.shape == h.shape
    assert gate.shape == h.shape[:2]
    loss = contribution.square().mean() + gate.mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name in ("generator_layers", "latent_queries", "cue_projection", "reader_query", "reader_attention", "output_projection"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in adaptor.named_parameters() if name in n), name


def test_mean_reader_and_learned_cues_without_memory():
    h, _ = make_inputs(d_mem=4)
    adaptor = GenerativeMemoryAdaptor(16, 4, reader_type="mean", cue_source="learned", hidden_size=16, num_heads=4)
    contribution, gate = adaptor(h, None)
    assert contribution.shape == h.shape
    (contribution.mean() + gate.mean()).backward()
    assert adaptor.learned_cues.grad is not None


def test_state_dict_round_trip_and_future_memory_causality():
    torch.manual_seed(3)
    adaptor = GenerativeMemoryAdaptor(8, 4, hidden_size=8, num_heads=2, cue_window=3)
    h = torch.randn(1, 4, 8)
    mem = torch.randn(1, 4, 4)
    out_a = adaptor(h, mem)[0][:, :2]
    mem_future = mem.clone()
    mem_future[:, 2:] += 100
    out_b = adaptor(h, mem_future)[0][:, :2]
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-5)
    restored = GenerativeMemoryAdaptor(8, 4, hidden_size=8, num_heads=2, cue_window=3)
    restored.load_state_dict(adaptor.state_dict())
    assert torch.allclose(adaptor(h, mem)[0], restored(h, mem)[0])


def test_build_adaptor_and_generator_boundary():
    adaptor = build_adaptor("transferred", 16, 8, architecture="generative", generator_hidden_size=16, generator_heads=4)
    assert isinstance(adaptor, GenerativeMemoryAdaptor)
    assert "h" not in inspect.signature(adaptor._generate).parameters
    learned = build_adaptor("transferred", 16, 8, architecture="generative", generator_cue_source="learned")
    assert learned.cue_projection is None
    branched = build_adaptor(
        "transferred",
        16,
        8,
        architecture="generative",
        num_branches=4,
        generator_fusion_type="generated_only",
    )
    assert branched.num_branches == 4


def test_four_branch_engram_cues_flow_through_generator_and_reader():
    h, mem = make_inputs()
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        num_branches=4,
        fusion_type="generated_only",
    )
    contribution, gates = adaptor(h, mem)
    assert contribution.shape == h.shape
    assert gates.shape == (4, *h.shape[:2])
    loss = contribution.square().mean() + gates.mean()
    loss.backward()
    assert adaptor.cue_projection.weight.grad is not None
    assert all(projection.weight.grad is not None for projection in adaptor.memory_key_projection)


def test_engram_residual_fusion_keeps_generator_in_chain():
    h, mem = make_inputs()
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        num_branches=4,
        fusion_type="engram_residual",
    )
    contribution, gates = adaptor(h, mem)
    assert contribution.shape == h.shape
    assert gates.shape == (4, *h.shape[:2])
    (contribution.square().mean() + gates.mean()).backward()
    assert adaptor.engram_value_projection.weight.grad is not None
    assert adaptor.delta_gate_bias.grad is not None


def test_dual_reader_has_independent_engram_and_generated_gates():
    h, mem = make_inputs()
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        num_branches=4,
        fusion_type="dual_reader",
    )
    contribution, gates = adaptor(h, mem)
    assert contribution.shape == h.shape
    assert gates.shape == (2, 4, *h.shape[:2])
    (contribution.square().mean() + gates.mean()).backward()
    assert adaptor.engram_value_projection.weight.grad is not None
    assert adaptor.output_projection.weight.grad is not None
    assert adaptor.engram_gate_bias.grad is not None


def test_dual_reader_runtime_ablation_zeroes_only_selected_gate():
    h, mem = make_inputs()
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        num_branches=4,
        fusion_type="dual_reader",
    )
    both, both_gates = adaptor(h, mem)

    adaptor.set_dual_reader_mode("engram_only")
    engram_only, engram_gates = adaptor(h, mem)
    assert torch.count_nonzero(engram_gates[0]) == 0
    assert torch.allclose(engram_gates[1], both_gates[1])

    adaptor.set_dual_reader_mode("generated_only")
    generated_only, generated_gates = adaptor(h, mem)
    assert torch.allclose(generated_gates[0], both_gates[0])
    assert torch.count_nonzero(generated_gates[1]) == 0
    assert torch.allclose(both, engram_only + generated_only, atol=1e-6, rtol=1e-5)


def test_dual_reader_can_train_only_generated_branch():
    h, mem = make_inputs()
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        num_branches=4,
        fusion_type="dual_reader",
    )
    trainable_names = adaptor.train_generated_branch_only()

    assert trainable_names
    assert adaptor.latent_queries.requires_grad
    assert adaptor.output_projection.weight.requires_grad
    assert adaptor.gate_bias.requires_grad
    assert not adaptor.norm_h.weight.requires_grad
    assert not adaptor.engram_value_projection.weight.requires_grad
    assert not adaptor.engram_gate_bias.requires_grad
    assert all(not p.requires_grad for p in adaptor.engram_reader_norm.parameters())

    contribution, gates = adaptor(h, mem)
    (contribution.square().mean() + gates[0].mean()).backward()
    assert adaptor.output_projection.weight.grad is not None
    assert adaptor.gate_bias.grad is not None
    assert adaptor.engram_value_projection.weight.grad is None
    assert adaptor.engram_gate_bias.grad is None


def test_learned_setup_memory_does_not_open_engram_files():
    args = type("Args", (), {
        "architecture": "generative",
        "generator_cue_source": "learned",
        "memory_dim": 8,
        "memory_config": "/definitely/nonexistent/memory_config.json",
    })()
    memory, cfg, dim = setup_memory(args, torch.device("cpu"))
    assert memory is None and cfg is None and dim == 8
