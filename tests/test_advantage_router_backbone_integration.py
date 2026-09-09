from types import SimpleNamespace

import torch
from transformers import MistralConfig, MistralForCausalLM

from engram.backbone_wrapper import BackboneWrapper
from engram.memory import EngramMemory, MemoryConfig
from scripts.train_advantage_router import configure_router_training


def test_router_trains_end_to_end_through_frozen_tiny_causal_lm(monkeypatch):
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
    )
    wrapper.freeze_backbone()
    wrapper.freeze_memory()
    names = configure_router_training(wrapper, temperature=1.0)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    indices = torch.randint(0, memory_config.table_size, (2, 6, 1))
    wrapper.set_hash_indices(indices)

    outputs = wrapper(input_ids=input_ids, labels=input_ids, use_cache=False)
    outputs.loss.backward()

    assert names and all("router" in name for name in names)
    assert all(parameter.grad is not None for parameter in wrapper.adaptor.router.parameters())
    assert all(parameter.grad is None for parameter in wrapper.backbone.parameters())
    assert all(parameter.grad is None for parameter in wrapper.memory.parameters())
    wrapper.cleanup()


def test_generated_cue_window_matches_full_and_cached_decoding(monkeypatch):
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
        generator_cue_window=3,
        generator_fusion_type="dual_reader",
    )
    wrapper.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    indices = torch.randint(0, memory_config.table_size, (1, 6, 1))

    with torch.no_grad():
        wrapper.set_hash_indices(indices)
        full_logits = wrapper(input_ids=input_ids, use_cache=False).logits

        cached_logits = []
        past_key_values = None
        for end in range(1, input_ids.shape[1] + 1):
            wrapper.set_hash_indices(indices[:, :end])
            output = wrapper(
                input_ids=input_ids[:, end - 1:end],
                attention_mask=torch.ones(1, end, dtype=torch.long),
                past_key_values=past_key_values,
                use_cache=True,
            )
            cached_logits.append(output.logits[:, -1:])
            past_key_values = output.past_key_values

    cached_logits = torch.cat(cached_logits, dim=1)
    assert torch.allclose(full_logits, cached_logits, atol=2e-5, rtol=2e-5)
    wrapper.cleanup()
