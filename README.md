# MemoryAthena

MemoryAthena studies adaptive, multi-path Engram memory: a frozen or jointly
trained external memory is read through three complementary pathways—E, GE,
and GH—and an E-anchored router decides when generated memory should be
admitted and interpolated.

The repository contains the model components, training/evaluation entry
points, tests, and paper-facing evidence notes. The organization of the
artifact documentation is inspired by the reproducibility-oriented structure
of [XMemTransfer](https://github.com/OLAResearch/XMemTransfer), but the code
and experiments here are independent.

## What is in the repository

| Area | Location |
| --- | --- |
| Memory, readers, routing, and HF utilities | [`engram/`](engram/) |
| Training, evaluation, and analysis entry points | [`scripts/`](scripts/) |
| Small tracked configs and probe data | [`configs/`](configs/), [`data/`](data/) |
| Smoke tests and unit tests | [`tests/`](tests/) |
| Reproduction wrappers | [`run/`](run/) |
| Paper-to-artifact ledger | [`ARTIFACTS.md`](ARTIFACTS.md) |
| Offline experiment dashboard | [`web/index.html`](web/index.html) |

The source paper is maintained locally as `paper/ICLR_submit.tex`. The paper
tree and large experiment outputs are intentionally excluded from the public
source checkout; `ARTIFACTS.md` records their run-family names and verification
status without embedding private workstation or scratch paths.

## Method at a glance

1. Learn or import an Engram-style source memory.
2. Adapt generated-memory readers while keeping the configured backbone and
   memory fixed for the adaptation stage.
3. Train a compact E-relative advantage router from causal-text counterfactual
   supervision, not downstream task labels.
4. Freeze the system for downstream evaluation. Labels are used for final
   scoring and explicitly marked post-hoc analyses only.

The main Mistral configuration injects memory at layers 2 and 10 with a
four-branch reader. The appendix reports the exact training budgets, routing
thresholds, architecture, and checkpoint-selection rules.

## Scaling configuration

Scaling is joint model-side scaling, not only “make the memory table larger”:
the backbone, Engram table, generated-memory modules, readers, and router are
scaled together. The memory-side count excludes the backbone.

| Scale | Backbone | Engram | Generator | Readers | Router | Memory-side total |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Small | 124M | 33.554M | 256 / 2 / 4 | 16 | 64 | 37.573M |
| Medium | 345M | 93.716M | 428 / 2 / 6 | 24 | 104 | 104.008M |
| Large | 774M | 209.715M | 640 / 3 / 8 | 40 | 160 | 238.212M |
| XL | 1.5B | 405.537M | 896 / 4 / 12 | 56 | 224 | 472.912M |

The current scaling evidence is deliberately reported as a partial artifact
set until every memory, expert, and final router point has the same verified
corpus, budget, checkpoint, and completion metadata. See the scaling section in
[`ARTIFACTS.md`](ARTIFACTS.md).

## Reproduce locally

The project uses Python 3.11+ and `uv` for dependency management.

```bash
uv sync
uv run python -m py_compile engram/*.py scripts/*.py
uv run pytest -q
```

For a small end-to-end smoke run, inspect the wrappers under [`run/`](run/)
before launching them. Large training and evaluation runs are designed for
the CSC cluster and should be submitted only after local syntax/tests and the
resource-minimization audit pass.

Useful entry points include:

```text
scripts/eval_openqa.py
scripts/eval_general_paper_aligned.py
scripts/eval_general_nlp_halueval.py
scripts/eval_scaling_backbone.py
scripts/eval_case_study.py
scripts/eval_code_functional.py
scripts/benchmark_compute_cost_eonly.py
```

The compute-cost benchmark measures E-only versus MemoryAthena inference
latency, tokens/s, peak CUDA memory, Slurm elapsed time, and actual GCD-hours.
It is an inference-only measurement; values should be quoted only after the
corresponding Slurm result and completion marker exist.

## Evidence and limitations

Read [`ARTIFACTS.md`](ARTIFACTS.md) before using a number in a paper. The
current audit records several important boundaries:

- the historical Vanilla NLP row used a different scorer from the E/router
  rows and is not a matched headline baseline;
- the Yahoo high-threshold result is a post-hoc test-set threshold sweep and
  must be labeled as such;
- the transfer results lack a matched bare target-Llama baseline;
- HaluEval uses a separate choice-log-probability protocol;
- coding functional accuracy and the complete routed downstream comparison are
  not interchangeable with lexical PPL/F1 diagnostics;
- scaling and compute-cost claims remain conditional on the completion and
  validation status recorded in the ledger.

No training token, private cluster path, or credential belongs in a public
README. Raw logs, checkpoints, and job manifests remain outside the source
checkout.

## Offline dashboard

Open [`web/index.html`](web/index.html) directly in a browser. It has no CDN,
build step, or network dependency and presents the paper result map, scaling
configuration, case-study status, and compute-cost job state. The dashboard is
a local viewing aid, not a replacement for the raw JSON artifacts.

## License

Apache 2.0. See [`LICENSE`](LICENSE).
