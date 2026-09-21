from types import SimpleNamespace

from engram.backbone_wrapper import BackboneWrapper


def test_backbone_wrapper_finds_gpt2_transformer_blocks():
    blocks = [object(), object(), object()]
    wrapper = object.__new__(BackboneWrapper)
    wrapper.backbone = SimpleNamespace(
        transformer=SimpleNamespace(h=blocks),
    )

    assert list(wrapper._get_layers()) == blocks
