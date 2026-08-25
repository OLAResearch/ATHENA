import torch
from transformers import MistralConfig, MistralForCausalLM

from scripts.memgen_mistral_qa import MemGenMistralQA


def tiny_mistral():
    return MistralForCausalLM(
        MistralConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )


def test_memgen_prompt_memory_loss_and_gradients():
    torch.manual_seed(0)
    model = MemGenMistralQA(
        reasoner=tiny_mistral(),
        weaver=tiny_mistral(),
        prompt_latents_len=3,
        lora_r=2,
        lora_alpha=4,
    )
    input_ids = torch.tensor([[1, 7, 8, 9, 10, 11, 12, 2]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, -100, -100, 10, 11, 12, 2]])

    loss = model.training_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_lengths=[4],
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.weaver.query_latents.grad is not None
    assert model.reasoner_to_weaver.weight.grad is not None
    assert any(
        parameter.grad is not None
        for name, parameter in model.weaver.model.named_parameters()
        if "lora_" in name
    )
    assert all(parameter.grad is None for parameter in model.reasoner.parameters())


def test_memgen_generation_returns_new_tokens():
    torch.manual_seed(1)
    model = MemGenMistralQA(
        reasoner=tiny_mistral(),
        weaver=tiny_mistral(),
        prompt_latents_len=2,
        lora_r=2,
        lora_alpha=4,
    )
    input_ids = torch.tensor([[1, 7, 8, 9]])
    output = model.generate_ids(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=3,
        pad_token_id=0,
        eos_token_id=2,
    )
    assert output.ndim == 2
    assert output.shape[0] == 1
    assert output.shape[1] <= 3
