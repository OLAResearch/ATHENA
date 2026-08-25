import torch

from scripts.grpo_memgen_mistral_qa import (
    grpo_objective,
    normalize_group_rewards,
)


def test_group_rewards_are_centered_and_scaled():
    advantages = normalize_group_rewards(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.isclose(advantages.mean(), torch.tensor(0.0))
    assert torch.isclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-3)


def test_constant_reward_group_has_no_update_signal():
    advantages = normalize_group_rewards(torch.ones(4))
    assert torch.equal(advantages, torch.zeros(4))


def test_clipped_grpo_objective_has_gradient():
    new_logprobs = torch.tensor([1.0, 1.2, 0.8], requires_grad=True)
    old_logprobs = torch.tensor([1.0, 1.0, 1.0])
    advantages = torch.tensor([-1.0, 0.0, 1.0])
    loss = grpo_objective(new_logprobs, old_logprobs, advantages, clip_epsilon=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert new_logprobs.grad is not None
