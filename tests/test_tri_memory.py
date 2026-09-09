import torch
import pytest

from engram.adaptor import EngramAdaptor, build_adaptor
from engram.tri_memory import TRI_READER_SUBSETS, TriMemoryAdaptor


def make_inputs(batch=2, seq=5, d_mem=8, d_model=16):
    torch.manual_seed(19)
    h = torch.randn(batch, seq, d_model, requires_grad=True)
    mem = torch.randn(batch, seq, d_mem)
    return h, mem


@pytest.mark.parametrize("candidates, expected_count", [("sources", 3), ("subsets", 7)])
def test_advantage_checkpoint_strict_reload_preserves_runtime_contract(
    candidates, expected_count
):
    torch.manual_seed(17)
    source = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    source.configure_advantage_reader(candidates=candidates, threshold=0.1, max_scale=0.7)
    with torch.no_grad():
        source.advantage_router[0].weight.normal_()
        source.advantage_router[-1].weight.normal_()
        source.advantage_router[-1].bias.normal_()

    target = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    target.configure_advantage_reader(candidates=candidates, threshold=0.1, max_scale=0.7)
    target.load_state_dict(source.state_dict(), strict=True)
    target.eval()
    source.eval()

    h, mem = make_inputs()
    with torch.no_grad():
        source.set_tri_reader_mode("tri_advantage_routed")
        target.set_tri_reader_mode("tri_advantage_routed")
        source_output, _ = source(h, mem)
        target_output, _ = target(h, mem)

    assert torch.equal(source_output, target_output)
    weights = target.get_last_advantage_weights()
    predictions = target.get_last_advantage_predictions()
    assert weights is not None and weights.shape[-1] == expected_count
    assert predictions is not None and predictions.shape[-1] == expected_count


def test_advantage_checkpoint_strict_reload_preserves_multilayer_prefixes():
    torch.manual_seed(23)
    source = torch.nn.ModuleList(
        [
            TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4),
            TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4),
        ]
    )
    for adaptor in source:
        adaptor.configure_advantage_reader(candidates="sources")
    checkpoint = source.state_dict()

    target = torch.nn.ModuleList(
        [
            TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4),
            TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4),
        ]
    )
    for adaptor in target:
        adaptor.configure_advantage_reader(candidates="sources")

    result = target.load_state_dict(checkpoint, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    for source_adaptor, target_adaptor in zip(source, target):
        assert torch.equal(
            source_adaptor.advantage_router[-1].weight,
            target_adaptor.advantage_router[-1].weight,
        )


def test_factory_builds_hybrid_tri_reader():
    adaptor = build_adaptor(
        "transferred",
        16,
        8,
        architecture="generative",
        generator_cue_source="hybrid",
        generator_fusion_type="tri_reader",
        generator_adaptive_router=True,
        generator_hidden_size=16,
        generator_heads=4,
    )
    assert isinstance(adaptor, TriMemoryAdaptor)


def test_forced_experts_are_separate_and_gh_needs_no_memory():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(
        16, 8, hidden_size=16, num_heads=4, num_branches=2
    )
    outputs = {}
    for mode in (
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
    ):
        adaptor.set_tri_reader_mode(mode)
        active_mem = None if mode == "generated_from_context_only" else mem
        outputs[mode], gates = adaptor(h, active_mem)
        assert outputs[mode].shape == h.shape
        assert gates.shape == (3, 2, *h.shape[:2])
        active = gates.flatten(1).abs().sum(dim=1) > 0
        assert active.sum() == 1
    assert not torch.equal(
        outputs["generated_from_engram_only"],
        outputs["generated_from_context_only"],
    )
    assert adaptor.needs_engram_memory() is False


def test_ge_is_generated_memory_reader_without_direct_engram_residual():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)

    adaptor.set_tri_reader_mode("generated_from_engram_only")
    ge, _ = adaptor(h, mem)
    generated_only, generated_gate = adaptor._generated_expert(0, h, mem, None)
    assert torch.allclose(ge, generated_only, atol=1e-6, rtol=1e-5)

    adaptor.set_tri_reader_mode("engram_only")
    e, _ = adaptor(h, mem)
    assert not torch.equal(ge, e)
    assert generated_gate.shape[0] == 1


def test_tri_router_can_select_each_forced_expert_exactly():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(
        16, 8, hidden_size=16, num_heads=4, adaptive_router=True
    )
    forced = []
    for mode in (
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
    ):
        adaptor.set_tri_reader_mode(mode)
        forced.append(adaptor(h, None if mode.endswith("context_only") else mem)[0])

    adaptor.eval()
    adaptor.configure_router(hard=True)
    adaptor.set_tri_reader_mode("tri_routed")
    for source_index in range(3):
        with torch.no_grad():
            adaptor.router[-1].weight.zero_()
            adaptor.router[-1].bias.fill_(-10.0)
            adaptor.router[-1].bias[source_index] = 10.0
        routed, _ = adaptor(h, mem)
        assert torch.equal(routed, forced[source_index])


def test_all_three_experts_and_router_receive_gradient_from_token_zero():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(
        16, 8, hidden_size=16, num_heads=4, adaptive_router=True
    )
    loss = h.new_zeros(())
    for mode in (
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
        "tri_routed",
    ):
        adaptor.set_tri_reader_mode(mode)
        active_mem = None if mode == "generated_from_context_only" else mem
        output, _ = adaptor(h, active_mem)
        loss = loss + output.square().mean()
    loss.backward()

    required = (
        "engram_value_projection",
        "engram_cue_projection",
        "context_cue_projection",
        "generator_layers",
        "output_projection",
        "router",
    )
    for fragment in required:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for name, parameter in adaptor.named_parameters()
            if fragment in name
        ), fragment


def test_ge_is_causal_and_legacy_engram_import_is_exact():
    torch.manual_seed(31)
    h = torch.randn(1, 5, 16)
    mem = torch.randn(1, 5, 8)
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4, cue_window=3)
    adaptor.set_tri_reader_mode("generated_from_engram_only")
    early = adaptor(h, mem)[0][:, :2]
    changed = mem.clone()
    changed[:, 2:] += 100
    assert torch.allclose(early, adaptor(h, changed)[0][:, :2], atol=1e-6, rtol=1e-5)

    legacy = EngramAdaptor(16, 8)
    adaptor.initialize_engram_reader_from_legacy(legacy)
    adaptor.set_tri_reader_mode("engram_only")
    expected, expected_gate = legacy(h, mem)
    actual, gates = adaptor(h, mem)
    assert torch.equal(actual, expected)
    assert torch.equal(gates[0, 0], expected_gate)


def test_gh_uses_a_strictly_causal_hidden_window():
    torch.manual_seed(37)
    h = torch.randn(1, 5, 16)
    context = torch.randn(1, 5, 16)
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4, cue_window=3)
    adaptor.set_tri_reader_mode("generated_from_context_only")
    early = adaptor(h, None, context_h=context)[0][:, :2]
    changed = context.clone()
    changed[:, 2:] += 100
    later_changed = adaptor(h, None, context_h=changed)[0][:, :2]
    assert torch.allclose(early, later_changed, atol=1e-6, rtol=1e-5)


def test_ge_only_does_not_run_the_direct_engram_reader():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    adaptor.set_tri_reader_mode("generated_from_engram_only")

    def fail_direct(*args, **kwargs):
        raise AssertionError("GE-only must not execute the direct E reader")

    adaptor._direct_engram = fail_direct
    output, gates = adaptor(h, mem)
    assert output.shape == h.shape
    assert torch.count_nonzero(gates[0]) == 0


def test_reader_supports_every_requested_expert_subset():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    singles = {}
    for mode in (
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
    ):
        adaptor.set_tri_reader_mode(mode)
        singles[mode] = adaptor(h, None if mode.endswith("context_only") else mem)[0]

    combinations = {
        "e_ge": ("engram_only", "generated_from_engram_only"),
        "E+GH": ("engram_only", "generated_from_context_only"),
        "GE+GH": ("generated_from_engram_only", "generated_from_context_only"),
    }
    expert_names = (
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
    )
    for mode, members in combinations.items():
        adaptor.set_tri_reader_mode(mode)
        actual, gates = adaptor(h, mem)
        weights = adaptor.get_last_router_weights()
        assert weights is not None
        expected = sum(
            weights[..., index : index + 1] * singles[member]
            for index, member in enumerate(
                expert_names
            )
            if member in members
        )
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        active_indices = [expert_names.index(member) for member in members]
        inactive_indices = [index for index in range(3) if index not in active_indices]
        assert torch.count_nonzero(weights[..., inactive_indices]) == 0
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
        active = gates.flatten(1).abs().sum(dim=1) > 0
        expected_active = [
            "engram_only" in members,
            "generated_from_engram_only" in members,
            "generated_from_context_only" in members,
        ]
        assert active.tolist() == expected_active


def test_pair_reader_can_hard_select_each_active_expert():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    expert_names = (
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
    )
    forced = []
    for mode in expert_names:
        adaptor.set_tri_reader_mode(mode)
        forced.append(adaptor(h, None if mode.endswith("context_only") else mem)[0])

    adaptor.eval()
    adaptor.configure_router(hard=True)
    adaptor.set_tri_reader_mode("E+GH")
    for selected in (0, 2):
        with torch.no_grad():
            adaptor.router[-1].weight.zero_()
            adaptor.router[-1].bias.fill_(-10.0)
            adaptor.router[-1].bias[selected] = 10.0
        actual, _ = adaptor(h, mem)
        assert torch.equal(actual, forced[selected])
        weights = adaptor.get_last_router_weights()
        assert torch.count_nonzero(weights[..., 1]) == 0


def test_tri_soft_fused_and_tri_routed_have_distinct_reader_contracts():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)

    adaptor.set_tri_reader_mode("tri_soft_fused")
    soft, _ = adaptor(h, mem)
    soft_weights = adaptor.get_last_router_weights()
    assert soft_weights is not None
    assert torch.allclose(soft_weights.sum(dim=-1), torch.ones_like(soft_weights[..., 0]))
    assert torch.all((soft_weights > 0).sum(dim=-1) == 3)

    adaptor.set_tri_reader_mode("tri_routed")
    hard, _ = adaptor(h, mem)
    hard_weights = adaptor.get_last_router_weights()
    assert hard_weights is not None
    assert torch.allclose(hard_weights.sum(dim=-1), torch.ones_like(hard_weights[..., 0]))
    assert torch.all((hard_weights == 1).sum(dim=-1) == 1)
    assert not torch.equal(soft, hard)


def test_e_safe_router_always_retains_e_and_adds_generated_residuals():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    adaptor.eval()
    forced = {}
    for mode in (
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
    ):
        adaptor.set_tri_reader_mode(mode)
        forced[mode], _ = adaptor(h, None if mode.endswith("context_only") else mem)

    # Make the generated-vs-E advantages deterministic.  With threshold=1,
    # GE receives sigmoid(1) and GH receives sigmoid(-3).
    with torch.no_grad():
        adaptor.router[-1].weight.zero_()
        adaptor.router[-1].bias.copy_(torch.tensor([0.0, 2.0, -2.0]))
    adaptor.configure_router(safe_residual_threshold=1.0)
    adaptor.set_tri_reader_mode("tri_safe_routed")
    actual, _ = adaptor(h, mem)
    weights = adaptor.get_last_router_weights()
    assert weights is not None
    assert torch.allclose(weights[..., 0], torch.ones_like(weights[..., 0]))
    expected = (
        forced["engram_only"]
        + torch.sigmoid(torch.tensor(1.0)) * forced["generated_from_engram_only"]
        + torch.sigmoid(torch.tensor(-3.0)) * forced["generated_from_context_only"]
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_e_safe_router_can_be_configured_as_exact_engram_fallback():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    adaptor.eval()
    adaptor.set_tri_reader_mode("engram_only")
    expected, _ = adaptor(h, mem)
    adaptor.configure_router(safe_residual_scale=0.0)
    adaptor.set_tri_reader_mode("safe_routed")
    actual, _ = adaptor(h, mem)
    assert torch.equal(actual, expected)


def test_advantage_reader_falls_back_to_engram_and_can_switch_to_ge():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(
        16, 8, hidden_size=16, num_heads=4, adaptive_router=True
    )
    adaptor.eval()
    adaptor.configure_advantage_reader(candidates="sources")

    adaptor.set_tri_reader_mode("engram_only")
    expected_engram, _ = adaptor(h, mem)
    adaptor.set_tri_reader_mode("tri_advantage_routed")
    fallback, _ = adaptor(h, mem)
    assert torch.equal(fallback, expected_engram)
    weights = adaptor.get_last_advantage_weights()
    assert weights is not None
    assert torch.all(weights[..., 0] == 1)
    assert torch.count_nonzero(weights[..., 1:]) == 0

    # The E anchor remains available, while a confident positive GE advantage
    # is allowed to replace it completely when max_scale is one.
    with torch.no_grad():
        adaptor.advantage_router[-1].weight.zero_()
        adaptor.advantage_router[-1].bias.copy_(
            torch.tensor([0.0, 2.0, -2.0, 0.0, 100.0, -100.0])
        )
    adaptor.set_tri_reader_mode("generated_from_engram_only")
    expected_ge, _ = adaptor(h, mem)
    adaptor.set_tri_reader_mode("tri_advantage_routed")
    selected_ge, _ = adaptor(h, mem)
    assert torch.equal(selected_ge, expected_ge)
    weights = adaptor.get_last_advantage_weights()
    assert weights is not None
    assert torch.all(weights[..., 1] == 1)
    assert torch.all(weights[..., 0] == 0)


def test_advantage_reader_uses_continuous_e_anchored_interpolation():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(
        16, 8, hidden_size=16, num_heads=4, adaptive_router=True
    )
    adaptor.eval()
    adaptor.configure_advantage_reader(
        candidates="sources", threshold=0.0, confidence_threshold=0.5,
        temperature=0.15, max_scale=1.0
    )

    adaptor.set_tri_reader_mode("engram_only")
    engram, _ = adaptor(h, mem)
    adaptor.set_tri_reader_mode("generated_from_engram_only")
    generated, _ = adaptor(h, mem)

    # GE advantage 0.075 gives an advantage scale of 0.5; confidence logit 0
    # gives a confidence scale of 0.5, so the final interpolation is a=0.25.
    with torch.no_grad():
        adaptor.advantage_router[-1].weight.zero_()
        adaptor.advantage_router[-1].bias.copy_(
            torch.tensor([0.0, 0.075, -1.0, 0.0, 0.0, -10.0])
        )
    adaptor.set_tri_reader_mode("tri_advantage_routed")
    actual, _ = adaptor(h, mem)
    expected = engram + 0.25 * (generated - engram)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    weights = adaptor.get_last_advantage_weights()
    assert weights is not None
    assert torch.allclose(weights[..., 0], torch.full_like(weights[..., 0], 0.75))
    assert torch.allclose(weights[..., 1], torch.full_like(weights[..., 1], 0.25))
    assert torch.count_nonzero(weights[..., 2]) == 0


def test_advantage_reader_rejects_nonfinite_candidate_with_exact_engram_fallback(
    monkeypatch,
):
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(
        16, 8, hidden_size=16, num_heads=4, adaptive_router=True
    )
    adaptor.eval()
    adaptor.configure_advantage_reader(candidates="sources")
    adaptor.set_tri_reader_mode("engram_only")
    expected, _ = adaptor(h, mem)

    def nonfinite_generated(index, hidden, memory, cue_mem, context_h=None):
        del index, memory, cue_mem, context_h
        return torch.full_like(hidden, float("nan")), hidden.new_zeros(hidden.shape[:2])

    monkeypatch.setattr(adaptor, "_generated_expert", nonfinite_generated)
    with torch.no_grad():
        adaptor.advantage_router[-1].weight.zero_()
        adaptor.advantage_router[-1].bias.copy_(
            torch.tensor([0.0, 100.0, -1.0, 0.0, 100.0, -100.0])
        )
    adaptor.set_tri_reader_mode("tri_advantage_routed")
    actual, _ = adaptor(h, mem)

    assert torch.equal(actual, expected)
    weights = adaptor.get_last_advantage_weights()
    assert weights is not None
    assert torch.all(weights[..., 0] == 1)
    assert torch.count_nonzero(weights[..., 1:]) == 0


def test_advantage_subset_endpoints_follow_pair_hard_contract():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(
        16, 8, hidden_size=16, num_heads=4, adaptive_router=True
    )
    adaptor.configure_advantage_reader(candidates="subsets")
    with torch.no_grad():
        adaptor.router[-1].weight.zero_()
        adaptor.router[-1].bias.copy_(torch.tensor([0.0, 3.0, -3.0]))

    # The E+GE candidate is the fourth endpoint in the canonical seven-way
    # order. It must use the same pair router contract as direct evaluation.
    adaptor.configure_router(hard=False)
    candidates, _, _, _, _ = adaptor._compute_advantage_candidates(
        h, mem, None, None
    )
    adaptor.set_tri_reader_mode("e_ge")
    expected_soft, _ = adaptor(h, mem)
    assert torch.allclose(candidates[3], expected_soft, atol=1e-6, rtol=1e-5)

    adaptor.configure_router(hard=True)
    candidates, _, _, _, _ = adaptor._compute_advantage_candidates(
        h, mem, None, None
    )
    adaptor.set_tri_reader_mode("e_ge")
    expected_hard, _ = adaptor(h, mem)
    assert torch.equal(candidates[3], expected_hard)


def test_unified_reader_exposes_all_seven_subset_probabilities():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    adaptor.eval()

    adaptor.set_tri_reader_mode("tri_subset_soft_fused")
    output, _ = adaptor(h, mem)
    subset_weights = adaptor.get_last_subset_router_weights()
    source_weights = adaptor.get_last_router_weights()
    assert output.shape == h.shape
    assert subset_weights is not None
    assert subset_weights.shape == (*h.shape[:2], 7)
    assert torch.all(subset_weights > 0)
    assert torch.allclose(
        subset_weights.sum(dim=-1), torch.ones_like(subset_weights[..., 0])
    )
    assert source_weights is not None
    assert source_weights.shape == (*h.shape[:2], 3)
    assert torch.allclose(
        source_weights.sum(dim=-1), torch.ones_like(source_weights[..., 0])
    )


def test_unified_hard_reader_can_select_each_single_pair_or_triple_subset():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    adaptor.eval()
    forced = []
    for mode in (
        "engram_only",
        "generated_from_engram_only",
        "generated_from_context_only",
    ):
        adaptor.set_tri_reader_mode(mode)
        forced.append(adaptor(h, None if mode.endswith("context_only") else mem)[0])

    # Make the inner source Reader uniform, so each selected pair/triple has
    # an unambiguous expected mean while the seven-way head is hard-routed.
    with torch.no_grad():
        adaptor.router[-1].weight.zero_()
        adaptor.router[-1].bias.zero_()
    adaptor.set_tri_reader_mode("tri_subset_routed")
    for subset_index, subset in enumerate(TRI_READER_SUBSETS):
        with torch.no_grad():
            adaptor.subset_router[-1].weight.zero_()
            adaptor.subset_router[-1].bias.fill_(-10.0)
            adaptor.subset_router[-1].bias[subset_index] = 10.0
        actual, _ = adaptor(h, mem)
        expected = sum((forced[index] for index in subset)) / len(subset)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        weights = adaptor.get_last_subset_router_weights()
        assert weights is not None
        assert torch.all((weights == 1).sum(dim=-1) == 1)
        assert torch.all(weights[..., subset_index] == 1)


def test_zero_init_hard_unified_reader_falls_back_to_engram():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    adaptor.eval()
    adaptor.configure_router(hard=True)

    adaptor.set_tri_reader_mode("engram_only")
    expected, _ = adaptor(h, mem)

    # The constructor's zero-init routing prior must select the E singleton
    # before the generated branches have learned useful representations.
    adaptor.set_tri_reader_mode("tri_subset_routed")
    actual, _ = adaptor(h, mem)
    weights = adaptor.get_last_subset_router_weights()

    assert weights is not None
    assert torch.count_nonzero(adaptor.subset_router[-1].weight) == 0
    assert torch.count_nonzero(adaptor.subset_router[-1].bias) == 0
    assert torch.all(weights[..., 0] == 1)
    assert torch.count_nonzero(weights[..., 1:]) == 0
    assert torch.equal(actual, expected)


def test_unified_reader_updates_both_subset_and_source_heads():
    h, mem = make_inputs()
    adaptor = TriMemoryAdaptor(16, 8, hidden_size=16, num_heads=4)
    adaptor.set_tri_reader_mode("tri_subset_soft_fused")
    output, _ = adaptor(h, mem)
    output.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in adaptor.subset_router.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in adaptor.router.parameters()
    )


def test_hybrid_constructor_is_available_from_generative_memory_module():
    from engram.generative_memory import GenerativeMemoryAdaptor

    adaptor = GenerativeMemoryAdaptor(
        16,
        8,
        hidden_size=16,
        num_heads=4,
        cue_source="hybrid",
        fusion_type="tri_reader",
    )
    assert hasattr(adaptor, "set_tri_reader_mode")
    adaptor.set_tri_reader_mode("E+GH")
    output, _ = adaptor(*make_inputs())
    assert output.shape == (2, 5, 16)
