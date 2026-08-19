import pytest
import torch

from engram.backbone_wrapper import BackboneWrapper


def test_tail_context_keeps_full_prompt_context():
    context = torch.arange(24).reshape(2, 3, 4)
    hidden = torch.zeros(2, 3, 8)
    assert BackboneWrapper._tail_context(context, hidden) is context


def test_tail_context_selects_current_cached_token():
    context = torch.arange(40).reshape(2, 5, 4)
    hidden = torch.zeros(2, 1, 8)
    selected = BackboneWrapper._tail_context(context, hidden)
    assert torch.equal(selected, context[:, -1:])


def test_tail_context_rejects_short_memory_context():
    context = torch.zeros(2, 1, 4)
    hidden = torch.zeros(2, 2, 8)
    with pytest.raises(ValueError, match="shorter"):
        BackboneWrapper._tail_context(context, hidden)
