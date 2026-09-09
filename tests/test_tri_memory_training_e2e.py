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


@pytest.mark.parametrize("reader_variant", ["three_way", "subset", "route_only"])
def test_tri_memory_trains_every_path_from_first_step(
    monkeypatch, tmp_path, reader_variant
):
    unified_subset_reader = reader_variant == "subset"
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
            "--condition", "transferred",
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

    train_adaptor.main()

    results = json.loads((output_dir / "results.json").read_text())
    assert results["completed"] is True
    assert results["actual_steps"] == 1
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
    rows = [json.loads(line) for line in (output_dir / "train_log.jsonl").read_text().splitlines()]
    training_row = next(row for row in rows if row.get("type") != "eval")
    if reader_variant == "route_only":
        assert results["training_design"] == "token_zero_joint_E_GE_GH_route_only_moe"
        assert training_row["reader_mode"] == "tri_route_only"
    else:
        assert training_row["reader_mode"] == (
            "tri_subset_joint" if unified_subset_reader else "tri_source_joint"
        )
    expected_training_paths = (
        {
            "engram_only",
            "generated_from_engram_only",
            "generated_from_context_only",
            "e_ge",
            "e_gh",
            "ge_gh",
            "tri_soft_fused",
            "tri_subset_soft_fused",
        }
        if unified_subset_reader
        else {"tri_soft_fused"}
        if reader_variant == "route_only"
        else set(results["validation"])
    )
    assert set(training_row["path_lm_losses"]) == expected_training_paths
    assert len(training_row["router_weights"]) == 3
    if unified_subset_reader:
        assert len(training_row["router_subset_weights"]) == 7
        assert len(training_row["oracle_target_usage"]) == 7
        assert 0.0 <= training_row["router_subset_singleton_mass"] <= 1.0
        assert 0.0 <= training_row["router_subset_pair_mass"] <= 1.0
        assert 0.0 <= training_row["router_subset_triple_mass"] <= 1.0
        assert 0.0 <= training_row["router_subset_multi_expert_mass"] <= 1.0
        assert 1 <= training_row["router_subset_active_classes"] <= 7
        assert isinstance(training_row["router_subset_combination_collapse"], bool)
    elif reader_variant == "route_only":
        assert training_row["router_subset_weights"] is None
        assert training_row["oracle_target_usage"] is None
        assert training_row["router_distillation_loss"] >= 0
    else:
        assert training_row["router_subset_weights"] is None
        assert len(training_row["oracle_target_usage"]) == 3
    if reader_variant == "route_only":
        assert training_row["router_distillation_loss"] >= 0
    else:
        assert training_row["router_distillation_loss"] > 0
    assert all(value > 0 for value in training_row["tri_grad_norms"].values())
    assert training_row["grad_norm_backbone"] == 0.0
    assert training_row["grad_norm_memory"] == 0.0
