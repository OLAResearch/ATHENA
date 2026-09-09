import json
from types import SimpleNamespace

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


def test_fair_joint_training_runs_end_to_end_and_writes_artifacts(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        train_adaptor,
        "get_dataloader",
        lambda **kwargs: [batch],
    )

    memory_config = MemoryConfig(
        max_ngram=2,
        heads_per_order=1,
        table_size=16,
        d_head=4,
    )
    source_memory = tmp_path / "memory.pt"
    memory_config_path = tmp_path / "memory_config.json"
    torch.save(EngramMemory(memory_config).state_dict(), source_memory)
    memory_config_path.write_text(json.dumps({
        "max_ngram": 2,
        "heads_per_order": 1,
        "table_size": 16,
        "d_head": 4,
        "hash_seed": 42,
        "canon_mode": "vocab",
    }))
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
            "--max-tokens", "12",
            "--validation-max-tokens", "6",
            "--seq-len", "6",
            "--batch-size", "1",
            "--lr", "0.001",
            "--router-lr", "0.01",
            "--warmup-steps", "1",
            "--eval-every", "1",
            "--log-every", "1",
            "--injection-layers", "0:1",
            "--adaptor-branches", "2",
            "--architecture", "generative",
            "--generator-hidden-size", "8",
            "--generator-layers", "1",
            "--generator-heads", "2",
            "--generator-cue-window", "2",
            "--generator-fusion-type", "dual_reader",
            "--joint-engram-generated-router",
            "--deterministic-engram-init-seed", "42",
            "--router-start-tokens", "6",
            "--router-init-alpha", "0.95",
            "--generated-residual-penalty", "0.0001",
            "--canon-mode", "vocab",
            "--corpus", "wikipedia-2021",
            "--skip-final-test-eval",
        ],
    )
    train_adaptor.main()

    results = json.loads((output_dir / "results.json").read_text())
    assert results["completed"] is True
    assert results["actual_steps"] == 2
    assert results["max_tokens"] == 12
    assert results["training_design"] == "single_stage_joint_engram_generated_router"
    assert set(results["validation"]) == {"engram_only", "both", "routed"}
    assert (output_dir / "adaptor_best.pt").is_file()
    assert (output_dir / "adaptor.pt").is_file()
    log_rows = [json.loads(row) for row in (output_dir / "train_log.jsonl").read_text().splitlines()]
    training_rows = [row for row in log_rows if row.get("type") != "eval"]
    assert [row["reader_mode"] for row in training_rows] == ["both", "routed"]
