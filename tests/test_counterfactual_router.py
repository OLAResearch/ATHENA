from types import SimpleNamespace

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

from engram.backbone_wrapper import BackboneWrapper
from engram.generative_memory import GenerativeMemoryAdaptor
from engram.memory import EngramMemory, MemoryConfig
from scripts.train_advantage_router import configure_router_training
from scripts.train_counterfactual_router import (
    _load_tri_experts,
    counterfactual_distillation_loss,
    expert_token_advantage,
    future_span_advantage,
    gold_token_log_probs,
    get_calibration_thresholds,
    initialize_distillation_router,
    parse_args,
    router_weight_stats,
    run_router_probe,
    tri_counterfactual_distillation_loss,
)
from engram.tri_memory import TriMemoryAdaptor


def test_counterfactual_router_accepts_wikitext_corpus(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_counterfactual_router.py",
            "--corpus",
            "wikitext",
            "--adaptor-dir",
            "adaptor",
            "--source-memory",
            "memory.pt",
            "--memory-config",
            "memory.json",
            "--output-dir",
            "out",
        ],
    )

    assert parse_args().corpus == "wikitext"


def test_calibration_thresholds_follow_reader_contract():
    args = SimpleNamespace(
        advantage_thresholds=(0.0, 0.1),
        router_thresholds=(0.5, 0.75),
    )

    assert get_calibration_thresholds(args, tri=True) == (0.0, 0.1)
    assert get_calibration_thresholds(args, tri=False) == (0.5, 0.75)


def test_calibration_thresholds_require_the_selected_grid():
    args = SimpleNamespace(advantage_thresholds=(0.0,))

    with pytest.raises(AttributeError, match="router_thresholds"):
        get_calibration_thresholds(args, tri=False)


def test_multi_layer_tri_expert_checkpoint_allows_lazy_advantage_heads(tmp_path):
    torch.manual_seed(7)
    source = torch.nn.ModuleList(
        [
            TriMemoryAdaptor(8, 4, hidden_size=8, num_heads=2),
            TriMemoryAdaptor(8, 4, hidden_size=8, num_heads=2),
        ]
    )
    checkpoint = tmp_path / "experts.pt"
    torch.save(source.state_dict(), checkpoint)

    target = torch.nn.ModuleList(
        [
            TriMemoryAdaptor(8, 4, hidden_size=8, num_heads=2),
            TriMemoryAdaptor(8, 4, hidden_size=8, num_heads=2),
        ]
    )
    for adaptor in target:
        adaptor.configure_advantage_reader(candidates="sources")

    missing = _load_tri_experts(target, str(checkpoint))

    assert len(missing) == 8
    assert all(".advantage_router." in name for name in missing)
    for source_adaptor, target_adaptor in zip(source, target):
        assert torch.equal(
            source_adaptor.norm_h.weight, target_adaptor.norm_h.weight
        )


def test_gold_token_log_probs_use_causal_shift_and_ignore_masked_labels():
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[2, 1, -100, 0]])

    log_probs, valid = gold_token_log_probs(logits, labels)

    assert valid.tolist() == [[True, False, True]]
    assert torch.allclose(
        log_probs,
        torch.tensor([[-torch.log(torch.tensor(3.0)), 0.0, -torch.log(torch.tensor(3.0))]]),
    )


def test_future_span_advantage_averages_multiple_horizons_causally():
    advantage = torch.tensor([[1.0, -1.0, 3.0]])
    valid = torch.ones_like(advantage, dtype=torch.bool)

    span_advantage, span_valid = future_span_advantage(
        advantage, valid, span_lengths=(1, 2)
    )

    assert span_valid.all()
    assert torch.allclose(span_advantage, torch.tensor([[0.5, 0.0, 3.0]]))


def test_future_span_advantage_broadcasts_tri_candidate_dimension():
    advantage = torch.tensor(
        [[[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0], [3.0, 1.0, -2.0]]]
    )
    valid = torch.ones(1, 3, dtype=torch.bool)

    span_advantage, span_valid = future_span_advantage(
        advantage, valid, span_lengths=(1, 2)
    )

    expected = torch.tensor(
        [[[0.5, 1.5, 3.25], [0.0, 0.25, 2.5], [3.0, 1.0, -2.0]]]
    )
    assert span_valid.shape == valid.shape
    assert span_valid.all()
    assert torch.allclose(span_advantage, expected)


def test_tri_router_weight_stats_keeps_candidate_axis_flat():
    adaptor = TriMemoryAdaptor(
        d_model=8,
        d_mem=4,
        hidden_size=8,
        num_heads=2,
        adaptive_router=True,
    )
    adaptor._last_advantage_weights = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )

    class Wrapper:
        pass

    wrapper = Wrapper()
    wrapper.adaptor = adaptor
    assert router_weight_stats(wrapper) == [0.5, 0.5, 0.0]


def test_semantic_router_probe_preserves_exact_engram_contribution():
    adaptor = GenerativeMemoryAdaptor(
        d_model=8,
        d_mem=4,
        hidden_size=8,
        num_heads=2,
        fusion_type="dual_reader",
        adaptive_router=True,
        router_hidden_size=6,
        router_semantic_size=3,
    )
    h = torch.randn(2, 5, 8)
    mem = torch.randn(2, 5, 4)
    adaptor.set_dual_reader_mode("engram_only")
    engram, _ = adaptor(h, mem)

    adaptor.set_dual_reader_mode("routed")
    adaptor.set_router_supervision_only(True)
    probed, _ = adaptor(h, mem)

    assert torch.equal(engram, probed)
    assert adaptor.get_last_router_logits().shape == (2, 5, 2)
    names = adaptor.train_router_only()
    assert names
    assert any("router_semantic_projection" in name for name in names)
    assert all("router" in name for name in names)


def test_hard_router_uses_configured_conservative_threshold():
    adaptor = GenerativeMemoryAdaptor(
        d_model=8,
        d_mem=4,
        hidden_size=8,
        num_heads=2,
        fusion_type="dual_reader",
        adaptive_router=True,
    )
    adaptor.configure_router(
        temperature=1.0, hard=True, min_generated_probability=0.8
    )
    adaptor.eval()
    with torch.no_grad():
        adaptor.router[-1].weight.zero_()
        adaptor.router[-1].bias.copy_(torch.tensor([0.0, 1.0]))
    adaptor.set_dual_reader_mode("routed")
    adaptor(torch.randn(1, 3, 8), torch.randn(1, 3, 4))

    # sigmoid(1) is about 0.73, so a 0.8 threshold must keep Engram.
    weights = adaptor.get_last_router_weights()
    assert torch.equal(weights[..., 0], torch.ones_like(weights[..., 0]))
    assert torch.equal(weights[..., 1], torch.zeros_like(weights[..., 1]))


def test_counterfactual_teacher_backpropagates_only_into_router(monkeypatch):
    config = MistralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = MistralForCausalLM(config)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    monkeypatch.setattr(
        "engram.backbone_wrapper.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "engram.backbone_wrapper.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    memory_config = MemoryConfig(
        max_ngram=2,
        heads_per_order=1,
        table_size=16,
        d_head=4,
    )
    wrapper = BackboneWrapper(
        model_name="local-tiny-mistral",
        memory=EngramMemory(memory_config),
        condition="transferred",
        device=torch.device("cpu"),
        dtype=torch.float32,
        injection_layers=[0],
        adaptor_branches=2,
        architecture="generative",
        generator_hidden_size=8,
        generator_heads=2,
        generator_layers=1,
        generator_fusion_type="dual_reader",
        generator_adaptive_router=True,
        generator_router_hidden_size=8,
        generator_router_semantic_size=4,
    )
    wrapper.freeze_backbone()
    wrapper.freeze_memory()
    names = configure_router_training(wrapper, temperature=1.0)
    initialize_distillation_router(wrapper)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    indices = torch.randint(0, memory_config.table_size, (2, 6, 1))

    def set_indices(_):
        wrapper.set_hash_indices(indices)

    token_advantage, token_valid = expert_token_advantage(
        wrapper, set_indices, input_ids, input_ids
    )
    span_advantage, valid = future_span_advantage(
        token_advantage, token_valid, span_lengths=(1, 2, 4)
    )
    run_router_probe(wrapper, set_indices, input_ids)
    loss, metrics = counterfactual_distillation_loss(
        wrapper,
        span_advantage,
        valid,
        router_temperature=1.0,
        teacher_temperature=0.15,
        weight_floor=0.01,
        weight_cap=2.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert 0.0 <= metrics["sign_accuracy"] <= 1.0
    assert names and all("router" in name for name in names)
    assert all(parameter.grad is not None for parameter in wrapper.get_trainable_params())
    assert all(parameter.grad is None for parameter in wrapper.backbone.parameters())
    assert all(parameter.grad is None for parameter in wrapper.memory.parameters())
    wrapper.cleanup()


def test_tri_advantage_loss_aligns_causal_shift_and_trains_only_new_head():
    adaptor = TriMemoryAdaptor(
        d_model=8,
        d_mem=4,
        hidden_size=8,
        num_heads=2,
        adaptive_router=True,
        router_hidden_size=6,
        router_semantic_size=3,
    )
    adaptor.configure_advantage_reader(candidates="sources")
    names = adaptor.train_advantage_reader_only()
    adaptor.set_advantage_supervision_only(True)
    adaptor.set_tri_reader_mode("tri_advantage_routed")
    hidden = torch.randn(2, 5, 8)
    memory = torch.randn(2, 5, 4)
    adaptor(hidden, memory)

    class Wrapper:
        pass

    wrapper = Wrapper()
    wrapper.adaptor = adaptor
    span_advantage = torch.randn(2, 4, 3)
    valid = torch.ones(2, 4, dtype=torch.bool)
    loss, metrics = tri_counterfactual_distillation_loss(
        wrapper,
        span_advantage,
        valid,
        teacher_temperature=0.15,
        weight_floor=0.01,
        weight_cap=2.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["sign_accuracy"] <= 1.0
    assert names and all("advantage_router" in name for name in names)
    assert all(parameter.grad is not None for parameter in adaptor.advantage_router.parameters())
    assert all(parameter.grad is None for name, parameter in adaptor.named_parameters() if "advantage_router" not in name)
