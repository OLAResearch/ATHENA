import json

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

from engram.memory import EngramMemory, MemoryConfig
from scripts import train_adaptor


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "</s>"


class _VocabCanonicalizer:
    def build_id_map(self, tokenizer):
        return torch.arange(32, dtype=torch.long)


def test_load_config_coerces_scientific_notation_learning_rate(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"adaptor_training": {"lr": "3e-05"}}))
    monkeypatch.setattr(
        "sys.argv",
        ["train_adaptor.py", "--condition", "random_memory", "--config", str(config), "--output-dir", str(tmp_path / "out")],
    )
    args = train_adaptor.load_config(train_adaptor.parse_args())
    assert args.lr == 3e-5
    assert isinstance(args.lr, float)


@pytest.mark.parametrize("condition", ["random_memory", "permuted_keys", "no_gate", "affine_stitch", "train_from_scratch", "ffn_only"])
def test_tri_memory_trains_every_path_from_first_step(
    monkeypatch, tmp_path, condition
):
    reader_variant = "subset"
    unified_subset_reader = True
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
    monkeypatch.setattr(
        "scripts.train_adaptor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _TinyTokenizer(),
    )
    monkeypatch.setattr(
        train_adaptor,
        "build_canonicalizer",
        lambda *args, **kwargs: _VocabCanonicalizer(),
    )
    batch = {
        "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long),
        "labels": torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long),
    }
    monkeypatch.setattr(train_adaptor, "get_dataloader", lambda **kwargs: [batch])

    memory_config = MemoryConfig(
        max_ngram=2,
        heads_per_order=1,
        table_size=16,
        d_head=4,
    )
    source_memory = tmp_path / "memory.pt"
    memory_config_path = tmp_path / "memory_config.json"
    torch.save(EngramMemory(memory_config).state_dict(), source_memory)
    memory_config_path.write_text(
        json.dumps(
            {
                "max_ngram": 2,
                "heads_per_order": 1,
                "table_size": 16,
                "d_head": 4,
                "hash_seed": 42,
                "canon_mode": "vocab",
            }
        )
    )
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_adaptor.py",
            "--condition", condition,
            "--target-model", "local-tiny-mistral",
            "--source-memory", str(source_memory),
            "--memory-config", str(memory_config_path),
            "--output-dir", str(output_dir),
            "--max-tokens", "6",
            "--validation-max-tokens", "6",
            "--seq-len", "6",
            "--batch-size", "1",
            "--lr", "0.001",
            "--router-lr", "0.01",
            "--warmup-steps", "1",
            "--eval-every", "1",
            "--log-every", "1",
            "--injection-layers", "0",
            "--adaptor-branches", "2",
            "--architecture", "generative",
            "--generator-hidden-size", "8",
            "--generator-layers", "1",
            "--generator-heads", "2",
            "--generator-cue-window", "2",
            "--generator-cue-source", "hybrid",
            "--generator-fusion-type", "tri_reader",
            (
                "--joint-tri-subset-reader"
                if unified_subset_reader
                else "--joint-tri-route-only"
                if reader_variant == "route_only"
                else "--joint-tri-reader"
            ),
            "--deterministic-engram-init-seed", "42",
            "--generated-residual-penalty", "0",
            "--canon-mode", "vocab",
            "--corpus", "wikipedia-2021",
            "--skip-final-test-eval",
        ],
    )

    if condition == "ffn_only":
        import sys
        sys.argv.remove('--joint-tri-subset-reader')
        index = sys.argv.index('--deterministic-engram-init-seed')
        del sys.argv[index:index + 2]
    train_adaptor.main()

    results = json.loads((output_dir / "results.json").read_text())
    assert results["completed"] is True
    assert results["actual_steps"] == 1
    if condition == "ffn_only":
        assert results["trainable_params"] > 0
        _reload_eval(monkeypatch, output_dir, source_memory, memory_config_path, condition)
        return
    if unified_subset_reader:
        assert (
            results["training_design"]
            == "token_zero_joint_E_GE_GH_unified_seven_way_subset_reader"
        )
        expected_validation = {
            "engram_only",
            "generated_from_engram_only",
            "generated_from_context_only",
            "e_ge",
            "e_gh",
            "ge_gh",
            "tri_soft_fused",
            "tri_subset_routed",
            "tri_subset_soft_fused",
        }
    elif reader_variant == "route_only":
        assert results["training_design"] == "token_zero_joint_E_GE_GH_route_only_moe"
        expected_validation = {
            "engram_only",
            "generated_from_engram_only",
            "generated_from_context_only",
            "tri_routed",
        }
    else:
        assert results["training_design"] == "token_zero_joint_E_GE_GH_tri_reader"
        expected_validation = {
            "engram_only",
            "generated_from_engram_only",
            "generated_from_context_only",
            "tri_routed",
        }
    assert set(results["validation"]) == expected_validation
    if condition in ("random_memory", "permuted_keys", "train_from_scratch"):
        assert (output_dir / "memory.pt").exists()
    if condition == "train_from_scratch":
        assert (output_dir / "memory_best.pt").exists()
        best = torch.load(output_dir / "memory_best.pt", weights_only=True)
        final = torch.load(output_dir / "memory.pt", weights_only=True)
        assert all(torch.equal(best[k], final[k]) for k in best)
    # Round-trip through the same builder used by the router training stage.
    from scripts import train_counterfactual_router as router
    import argparse
    args = argparse.Namespace(adaptor_dir=str(output_dir), memory_config=str(memory_config_path),
        source_memory=str(source_memory), target_model="local-tiny-mistral",
        candidate_space="sources", adaptor_checkpoint=None, router_hidden_size=4,
        router_semantic_size=4, advantage_threshold=0.0, confidence_threshold=0.5,
        teacher_temperature=0.15, advantage_max_scale=1.0, canon_mode="vocab")
    monkeypatch.setattr(router, "build_canon_fn", lambda wrapper, *a: lambda ids: wrapper.set_canon_ids(ids))
    wrapper, _, _, _, _, _ = router._build_tri_wrapper(args, torch.device("cpu"), torch.float32)
    adaptor = wrapper.adaptor
    assert adaptor.get_advantage_reader_config()["enabled"]
    if condition in ("no_gate", "affine_stitch"):
        assert adaptor.ablation_condition == condition
    # Exercise teacher generation, router optimization, calibration and save.
    monkeypatch.setattr(router, "get_dataloader", lambda **kw: [batch])
    router_output = tmp_path / 'router'
    monkeypatch.setattr('sys.argv', ['train_counterfactual_router.py',
        '--adaptor-dir', str(output_dir), '--memory-config', str(memory_config_path),
        '--source-memory', str(source_memory), '--target-model', 'local-tiny-mistral',
        '--output-dir', str(router_output), '--candidate-space', 'sources',
        '--max-tokens', '6', '--validation-max-tokens', '6', '--seq-len', '6',
        '--canon-mode', 'vocab', '--eval-every', '1'])
    router.main()
    assert json.loads((router_output / 'results.json').read_text())['completed']
    _reload_eval(monkeypatch, router_output, source_memory, memory_config_path, condition)
    if condition in ('random_memory', 'permuted_keys', 'train_from_scratch'):
        before = torch.load(output_dir / 'memory.pt', weights_only=True)
        after = torch.load(router_output / 'memory.pt', weights_only=True)
        assert all(torch.equal(before[k], after[k]) for k in before)


def test_ffn_matches_tri_and_ignores_memory():
    from engram.adaptor import build_adaptor
    kw = dict(d_model=16, d_mem=4, architecture='generative', generator_fusion_type='tri_reader',
              generator_cue_source='hybrid', generator_hidden_size=8, generator_heads=2,
              generator_layers=1, generator_adaptive_router=True, num_branches=2)
    ffn = build_adaptor('ffn_only', **kw)
    tri = build_adaptor('transferred', **kw)
    tri.configure_advantage_reader(candidates='sources')
    target = sum(p.numel() for p in tri.parameters())
    actual = sum(p.numel() for p in ffn.parameters())
    assert abs(actual-target) / target < .01
    h = torch.randn(1, 4, 16)
    a, _ = ffn(h, torch.randn(1, 4, 4))
    b, _ = ffn(h, None)
    assert torch.equal(a, b)
    a.square().mean().backward()
    assert all(p.grad is not None for p in ffn.parameters())


@pytest.mark.parametrize('condition', ['no_gate', 'affine_stitch'])
def test_structural_control_gates_and_causal_gradients(condition):
    from engram.adaptor import build_adaptor
    a = build_adaptor(condition, 16, 4, architecture='generative', generator_fusion_type='tri_reader',
        generator_cue_source='hybrid', generator_hidden_size=8, generator_heads=2,
        generator_layers=1, generator_adaptive_router=True, num_branches=2)
    h, mem = torch.randn(1, 5, 16), torch.randn(1, 5, 4)
    for mode in ['engram_only', 'generated_from_engram_only', 'generated_from_context_only']:
        a.set_tri_reader_mode(mode)
        out, gates = a(h, mem)
        assert torch.isfinite(out).all()
        # Inactive paths have zero diagnostic gates, active paths are all one.
        assert (gates == 1).any()
        assert ((gates == 0) | (gates == 1)).all()
        out.square().mean().backward()
    if condition == 'affine_stitch':
        assert a.reader_attention is None
        assert a.affine_e_bias.grad is not None


def _reload_eval(monkeypatch, output, source_memory, memory_config, condition):
    import argparse
    from scripts import eval_openqa
    monkeypatch.setattr(eval_openqa, 'build_canon_fn',
        lambda wrapper, *a: lambda ids: wrapper.set_canon_ids(ids))
    args = argparse.Namespace(target_model='local-tiny-mistral', adaptor_dir=str(output),
        source_memory=str(source_memory), memory_config=str(memory_config), seed=999,
        canon_mode='vocab', dual_reader_mode='both' if condition == 'ffn_only' else 'tri_advantage_routed')
    model, canon = eval_openqa.setup_condition(args, condition, torch.device('cpu'), torch.float32)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    canon(ids)
    assert torch.isfinite(model(input_ids=ids).logits).all()
    if condition in ('random_memory', 'permuted_keys', 'train_from_scratch'):
        saved = torch.load(output / 'memory.pt', weights_only=True)
        assert all(torch.equal(saved[k], v) for k, v in model.memory.state_dict().items())
    model.cleanup()
