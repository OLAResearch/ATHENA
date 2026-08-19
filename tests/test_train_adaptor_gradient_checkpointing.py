from types import SimpleNamespace

import pytest

from scripts.train_adaptor import enable_frozen_backbone_gradient_checkpointing


class _CheckpointableBackbone:
    def __init__(self):
        self.checkpointing_enabled = False
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


def test_frozen_backbone_checkpointing_keeps_inputs_differentiable():
    backbone = _CheckpointableBackbone()
    wrapper = SimpleNamespace(backbone=backbone)

    enable_frozen_backbone_gradient_checkpointing(wrapper)

    assert backbone.checkpointing_enabled is True
    assert backbone.input_grads_enabled is True


def test_frozen_backbone_checkpointing_rejects_unsupported_backbone():
    class UnsupportedBackbone:
        def gradient_checkpointing_enable(self):
            pass

    wrapper = SimpleNamespace(backbone=UnsupportedBackbone())

    with pytest.raises(RuntimeError, match="enable_input_require_grads"):
        enable_frozen_backbone_gradient_checkpointing(wrapper)
