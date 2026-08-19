# Runner scripts

Cluster submission files are separated from the portable experiment runners:

- `run/lumi/`: LUMI Slurm files and their LUMI-only shell payloads.
- `run/lumi/lumi_target_memgen_eval_gpqa.slurm`: full GPQA Diamond retry from pinned public mirrors when the canonical dataset is gated.
- `run/roihu/`: Roihu Slurm files and their Roihu-only shell payloads.
- `scripts/`: Python training and evaluation programs; no Slurm files belong there.

Each `.sh` file reproduces one paper section end-to-end. Run from the repository root:

```
bash run/<NN_name>.sh
```

| Script | Paper section | Runtime (H100) | Prereq |
|---|---|---|---|
| `00_install.sh` | — | < 5 min | `uv` installed |
| `01_pilot.sh` | Pilot validation | ~30 min | 00 |
| `02_source_memories.sh` | Phase 1 (source memories) | days | 00 |
| `10_tier1_same_tokenizer.sh` | §5.1.1 Pythia-160M → 410M | ~8 h | 02 (Pythia) |
| `11_tier2_cross_tokenizer.sh` | §5.1.2 Pythia → TinyLlama | ~8 h | 02 (Pythia WB) |
| `12_tier3_ablations.sh` | §5.1.3 Source-of-benefit | ~12 h | 02 (Pythia) |
| `13_tier4_scaling.sh` | §5.1.4 QWen scaling | ~24 h | 02 (QWen-0.8B) |
| `14_cka_extended.sh` | §5.1.6 5×5 CKA | ~1 h | 00 |
| `20_downstream_fw_matched.sh` | §5.2 FW downstream | ~48 h | 02 (FW-4B, FW-9B) |
| `21_domain_alignment.sh` | §5.2 Nemo downstream | ~48 h | 02 (Nemo-4B, 9B) |
| `22_corpus_control.sh` | §5.2 HQ 26B vs HQ-DQA | ~12 h | 02 (HQ-26B) |
| `30_baselines.sh` | LoRA / Cross-LoRA / KNN-LM | ~18 h | 00 |
| `40_analysis.sh` | figures, gates, collisions | ~30 min | any runs |

## Notes

- Each script is idempotent: existing checkpoints are detected and the corresponding step is skipped.
- All scripts use `uv run`, so the virtualenv at `.venv/` is implicit.
- Required models download automatically from the HuggingFace Hub on first use (~50 GB total for all experiments).
- The 9B source memory training (`02_source_memories.sh`) requires a single 80 GB GPU (A100/H100); other experiments fit on 24 GB GPUs with the default gradient-accumulation settings.
- To run only a subset, edit the loops at the top of each script.
