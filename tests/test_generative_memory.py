import torch

from engram.adaptor import EngramAdaptor, MultiBranchEngramAdaptor, build_adaptor
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
    h, mem = make_inputs()
    generated = adaptor._generate(mem, h, h.shape[0], h.shape[1])
    changed_h = adaptor._generate(mem, h + 100, h.shape[0], h.shape[1])
    assert torch.equal(generated, changed_h)
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
    assert all(projection.weight.grad is not None for projection in adaptor.engram_key_projection)
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


def test_adaptive_router_selects_only_engram_or_engram_plus_residual():
    h, mem = make_inputs()
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        num_branches=4,
        fusion_type="dual_reader",
        adaptive_router=True,
    )
    outputs = {}
    for mode in ("engram_only", "both"):
        adaptor.set_dual_reader_mode(mode)
        outputs[mode] = adaptor(h, mem)[0]

    adaptor.eval()
    adaptor.configure_router(hard=True)
    adaptor.set_dual_reader_mode("routed")
    for expert_index, mode in enumerate(("engram_only", "both")):
        with torch.no_grad():
            adaptor.router[-1].weight.zero_()
            adaptor.router[-1].bias.fill_(-10.0)
            adaptor.router[-1].bias[expert_index] = 10.0
        routed, _ = adaptor(h, mem)
        assert torch.equal(routed, outputs[mode]), mode


def test_router_only_training_freezes_both_memory_experts():
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        num_branches=4,
        fusion_type="dual_reader",
        adaptive_router=True,
    )
    names = adaptor.train_router_only()
    assert names
    assert adaptor.dual_reader_mode == "routed"
    assert all("router" in name for name in names)
    assert all(
        parameter.requires_grad == ("router" in name)
        for name, parameter in adaptor.named_parameters()
    )


def test_dual_reader_zero_generated_delta_matches_engram_only():
    h, mem = make_inputs()
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        fusion_type="dual_reader",
    )
    adaptor.set_dual_reader_mode("both")
    with torch.no_grad():
        adaptor.output_projection.weight.zero_()
    residual_with_zero_delta, _ = adaptor(h, mem)

    adaptor.set_dual_reader_mode("engram_only")
    engram_only, _ = adaptor(h, mem)
    assert torch.allclose(
        residual_with_zero_delta, engram_only, atol=1e-6, rtol=1e-5
    )


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
    assert all(not p.requires_grad for p in adaptor.engram_key_projection.parameters())
    assert not adaptor.engram_gate_bias.requires_grad
    assert all(not p.requires_grad for p in adaptor.engram_reader_norm.parameters())

    contribution, gates = adaptor(h, mem)
    (contribution.square().mean() + gates[0].mean()).backward()
    assert adaptor.output_projection.weight.grad is not None
    assert adaptor.gate_bias.grad is not None
    assert adaptor.engram_value_projection.weight.grad is None
    assert adaptor.engram_gate_bias.grad is None


def test_dual_reader_import_reproduces_single_branch_legacy_engram():
    h, mem = make_inputs()
    legacy = EngramAdaptor(16, 8)
    converted = GenerativeMemoryAdaptor(
        16, 8, hidden_size=16, num_heads=4, fusion_type="dual_reader"
    )
    converted.initialize_engram_reader_from_legacy(legacy)
    converted.set_dual_reader_mode("engram_only")
    expected, expected_gate = legacy(h, mem)
    actual, actual_gates = converted(h, mem)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gate, actual_gates[1])


def test_dual_reader_import_reproduces_four_branch_legacy_engram():
    h, mem = make_inputs()
    legacy = MultiBranchEngramAdaptor(16, 8, num_branches=4)
    converted = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        num_branches=4,
        fusion_type="dual_reader",
    )
    converted.initialize_engram_reader_from_legacy(legacy)
    converted.set_dual_reader_mode("engram_only")
    expected, expected_gates = legacy(h, mem)
    actual, actual_gates = converted(h, mem)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gates, actual_gates[1])


def test_dual_reader_single_step_updates_only_generated_branch():
    torch.manual_seed(11)
    h, mem = make_inputs(batch=2, seq=4)
    target = torch.randn_like(h)
    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        fusion_type="dual_reader",
    )
    adaptor.train_generated_branch_only()
    adaptor.set_dual_reader_mode("both")

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in adaptor.named_parameters()
        if not parameter.requires_grad
    }
    generated_before = {
        name: parameter.detach().clone()
        for name, parameter in adaptor.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.SGD(
        [parameter for parameter in adaptor.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    contribution, _ = adaptor(h, mem)
    loss = (contribution - target).square().mean()
    loss.backward()
    assert adaptor.output_projection.weight.grad is not None
    assert adaptor.gate_bias.grad is not None
    optimizer.step()

    for name, before in frozen_before.items():
        assert torch.equal(adaptor.state_dict()[name], before), name
    assert any(
        not torch.equal(adaptor.state_dict()[name], before)
        for name, before in generated_before.items()
    )


def test_learned_setup_memory_does_not_open_engram_files():
    args = type("Args", (), {
        "architecture": "generative",
        "generator_cue_source": "learned",
        "memory_dim": 8,
        "memory_config": "/definitely/nonexistent/memory_config.json",
    })()
    memory, cfg, dim = setup_memory(args, torch.device("cpu"))
    assert memory is None and cfg is None and dim == 8
