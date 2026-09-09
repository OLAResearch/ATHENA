import torch
from transformers import MistralConfig, MistralForCausalLM

from engram.backbone_wrapper import BackboneWrapper
from engram.memory import EngramMemory, MemoryConfig


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


def test_gh_mode_skips_engram_table_lookup(monkeypatch):
    config = MistralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    monkeypatch.setattr(
        "engram.backbone_wrapper.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: MistralForCausalLM(config),
    )
    monkeypatch.setattr(
        "engram.backbone_wrapper.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _TinyTokenizer(),
    )
    memory = EngramMemory(
        MemoryConfig(max_ngram=2, heads_per_order=1, table_size=16, d_head=4)
    )
    wrapper = BackboneWrapper(
        model_name="tiny",
        memory=memory,
        condition="transferred",
        device=torch.device("cpu"),
        dtype=torch.float32,
        injection_layers=[0],
        architecture="generative",
        generator_cue_source="hybrid",
        generator_fusion_type="tri_reader",
        generator_adaptive_router=True,
        generator_hidden_size=8,
        generator_layers=1,
        generator_heads=2,
        generator_cue_window=2,
    )
    wrapper.adaptor.set_tri_reader_mode("generated_from_context_only")

    def fail_lookup(*args, **kwargs):
        raise AssertionError("GH must not access Engram memory")

    monkeypatch.setattr(wrapper.memory, "forward", fail_lookup)
    monkeypatch.setattr(wrapper.memory, "forward_from_indices", fail_lookup)
    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    wrapper.set_canon_ids(input_ids)
    output = wrapper(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert torch.isfinite(output.loss)
    wrapper.cleanup()


def test_multilayer_gh_cached_decoding_matches_full_sequence(monkeypatch):
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
    monkeypatch.setattr(
        "engram.backbone_wrapper.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "engram.backbone_wrapper.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _TinyTokenizer(),
    )
    memory_config = MemoryConfig(
        max_ngram=2, heads_per_order=1, table_size=16, d_head=4
    )
    wrapper = BackboneWrapper(
        model_name="tiny",
        memory=EngramMemory(memory_config),
        condition="transferred",
        device=torch.device("cpu"),
        dtype=torch.float32,
        injection_layers=[0, 1],
        architecture="generative",
        generator_cue_source="hybrid",
        generator_fusion_type="tri_reader",
        generator_adaptive_router=True,
        generator_hidden_size=8,
        generator_layers=1,
        generator_heads=2,
        generator_cue_window=3,
    )
    wrapper.eval()
    for adaptor in wrapper.adaptor:
        adaptor.set_tri_reader_mode("generated_from_context_only")

    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
    with torch.no_grad():
        full = wrapper(input_ids=input_ids, use_cache=False).logits

        cached_logits = []
        past_key_values = None
        for end in range(1, input_ids.shape[1] + 1):
            output = wrapper(
                input_ids=input_ids[:, end - 1 : end],
                attention_mask=torch.ones(1, end, dtype=torch.long),
                past_key_values=past_key_values,
                use_cache=True,
            )
            cached_logits.append(output.logits[:, -1:])
            past_key_values = output.past_key_values

    cached = torch.cat(cached_logits, dim=1)
    # Mistral's fused attention has small full-vs-cached floating-point
    # differences on CPU.  The important invariant is that the clean GH pass
    # does not append to the main cache (the pre-fix implementation differed
    # by roughly 3e-2 on this tiny model).
    assert torch.allclose(full, cached, atol=5e-3, rtol=5e-3)
    wrapper.cleanup()


def test_multilayer_gh_reuses_clean_context_for_same_batch(monkeypatch):
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
    monkeypatch.setattr(
        "engram.backbone_wrapper.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "engram.backbone_wrapper.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _TinyTokenizer(),
    )
    wrapper = BackboneWrapper(
        model_name="tiny",
        memory=EngramMemory(
            MemoryConfig(max_ngram=2, heads_per_order=1, table_size=16, d_head=4)
        ),
        condition="transferred",
        device=torch.device("cpu"),
        dtype=torch.float32,
        injection_layers=[0, 1],
        architecture="generative",
        generator_cue_source="hybrid",
        generator_fusion_type="tri_reader",
        generator_adaptive_router=True,
        generator_hidden_size=8,
        generator_layers=1,
        generator_heads=2,
        generator_cue_window=2,
    )
    for adaptor in wrapper.adaptor:
        adaptor.set_tri_reader_mode("generated_from_context_only")

    calls = 0
    original_forward = wrapper.backbone.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(wrapper.backbone, "forward", counted_forward)
    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    wrapper.set_canon_ids(input_ids)
    with torch.no_grad():
        first = wrapper(input_ids=input_ids, use_cache=False).logits
        second = wrapper(input_ids=input_ids, use_cache=False).logits

    # Two injected forwards plus one shared no-grad clean pass.
    assert calls == 3
    assert torch.allclose(first, second)
    wrapper.cleanup()
