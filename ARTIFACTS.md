# MemoryAthena artifact ledger

This ledger maps result-bearing objects referenced by `paper/ICLR_submit.tex`
and `paper/appendix.tex` to their source artifacts. It is intentionally safe to
publish: private workstation names, project numbers, scratch usernames, and
credentials are represented by placeholders.

Use these placeholders when resolving a raw path on the experiment machine:

```text
<RUN_ROOT>       = <private scratch>/ATHENA/run
<SOURCE_ROOT>    = <private source-memory bundle>
<LOCAL_ROOT>     = this repository checkout
```

## Verification legend

| Status | Meaning |
| --- | --- |
| `tracked` | The artifact is versioned in this repository. |
| `local-audit` | A local aggregate/audit file exists, but raw outputs are intentionally not versioned. |
| `remote-recorded` | The run family and expected artifact path are recorded by the audit; raw output is on the cluster. |
| `partial` | Some points/tasks are complete, but the paper-level claim is not closed. |
| `pending` | Submitted and not yet complete; do not quote measurements. |
| `disclosure` | Numerically available, but the protocol or baseline caveat must travel with the number. |

## Main-paper tables and figures

| Paper object | Result or figure | Artifact chain | Status / usage note |
| --- | --- | --- | --- |
| `tab:qa_combined` | Five-task QA comparison | `paper_aggregates/qa_five_task.aggregate.json`; `<RUN_ROOT>/tri_advantage_formal_20260908T133625Z/{nq,webqa,triviaqa,truthfulqa,hotpotqa}_reader_eval/results.json`; reader checkpoint `reader_20m_train/adaptor_best.pt` | `local-audit` + `remote-recorded`; `disclosure`: training histories are not all compute-matched |
| `tab:nlp` | Six-task general NLP | `paper_aggregates/dcpmi_nlp_six_task.aggregate.json`; `<RUN_ROOT>/general_paper_aligned_tri_e_router_20260918/results/results.json`; `<RUN_ROOT>/general_paper_aligned_cr_agn_yahoo_20260918/results/results.json` | `local-audit` + `remote-recorded`; historical Vanilla scorer differs from E/router scorer |
| `tab:controls` | QA interface, initialization, and random-router ablations | `<RUN_ROOT>/tri_ablations_20260909/{no_gate,permuted_keys,random_memory,train_from_scratch}/downstream/results.json`; `<RUN_ROOT>/tri_ffn_matched_20260912/ffn_only/downstream/results.json`; `<RUN_ROOT>/tri_affine_stitch_20260912/affine_stitch/downstream/results.json`; random-router QA manifests under `run/random_router_*` | `remote-recorded`; preserve condition names and seed metadata |
| `fig:transfer-overview` | Mistral-to-Llama transfer | `<RUN_ROOT>/mistral_to_llama_tri_20260909/llama_downstream_repair5/results.json`; `<RUN_ROOT>/general_paper_aligned_mistral_to_llama_20260918/results/results.json` | `remote-recorded`; target-side transfer only, no matched bare-Llama baseline |
| `fig:routing-and-oracle` | Downstream routing and gold-label oracle | `paper_aggregates/routing_and_support_audit_20260919.json`; QA result predictions and oracle fields in the five task `results.json` files | `local-audit`; oracle is post-hoc and is not deployed routing frequency |

The paper source also references the following main figures. Their TeX paths
are part of the paper source, but the rendered figure directory is not part of
the public source checkout; keep the rendered files with the paper artifact
bundle rather than claiming they are reproducible from this clone alone:

```text
paper/figures/overview.pdf
paper/figures/memoryathena_downstream_routing.pdf
paper/figures/memoryathena_oracle_headroom.pdf
```

## Appendix tables and figures

| Paper object | Artifact chain | Status / caveat |
| --- | --- | --- |
| `tab:tokenbudget`, `tab:routerconfig`, `tab:architecture` | Configuration in `paper/appendix.tex`; implementation in `engram/`, `scripts/train_source_memory.py`, `scripts/train_adaptor.py`, and `scripts/train_counterfactual_router.py` | `tracked` for code; paper source is local-only |
| `fig:qa-summary`, `fig:qa-emf1`, `fig:qa-paired` | QA five-task result JSONs and local audit aggregate | `local-audit` + `remote-recorded` |
| `tab:truthfulqa-mc` | `truthfulqa_reader_eval/results.json` in the QA run family | `remote-recorded`; report MC1/MC2/MC3/mean, not F1 |
| `fig:ablation-delta` | The six ablation result JSONs listed for `tab:controls` | `remote-recorded` |
| `tab:yahoo-threshold` | `<RUN_ROOT>/yahoo_router_threshold_sweep_20260919_opt1/results/results.json`; `<RUN_ROOT>/yahoo_router_threshold_sweep_20260919_opt2/results/threshold-0p9/results.json`; `<RUN_ROOT>/yahoo_router_threshold_sweep_20260919_opt2/results/threshold-1p0/results.json` | `remote-recorded` + `disclosure`: post-hoc test-set threshold sweep |
| `tab:routing-stats` | `paper_aggregates/routing_and_support_audit_20260919.json`; training-validation result JSONs for QA, NLP, and coding | `local-audit`; interpolation mass is not discrete route frequency |
| `tab:halueval` | `<RUN_ROOT>/tri_general_nlp_halueval_20260915/results/general_nlp_halueval_results.json` | `remote-recorded`; separate choice-log-probability protocol |
| `tab:scaling-config`, `tab:model-scaling-values`, `tab:token-scaling` | `scripts/scaling_law_config.py`, `scripts/eval_scaling_backbone.py`, `<RUN_ROOT>/scaling_law_20260918/manifest.json` | `tracked` config + `partial` run family |
| `fig:model-scaling`, `fig:scaling-wikitext`, `fig:scaling-general`, `fig:scaling-tokens`, `fig:scaling-params` | `<RUN_ROOT>/scaling_law_20260918/points/`; `submitted_jobs_nomemory_eval_20260919.json`; router-corpus repair manifests | `partial`: no-memory points are more complete than the final memory/router curve; verify every point before quoting |
| `fig:case-study` | `run/case_study_20260920/repair8_results/{candidates.json,results.json,status.json,case_study.md}`; source inference family `<RUN_ROOT>/case_study_20260920_repair3/` | `local-audit` for the finalized post-processor; exactly three router-correct rows, all realised argmax `GH` in the current slice (`E=0, GE=0, GH=3`) |

## Other result-bearing experiments

| Experiment | Artifact | Status / caveat |
| --- | --- | --- |
| HYP/CB/RTE extension | `<RUN_ROOT>/general_paper_aligned_hyp_cb_rte_20260919/results/results.json`; `repair2/results/results.json` | `remote-recorded`; not a complete three-row main-table aggregate |
| Mistral-to-Llama NLP | `<RUN_ROOT>/general_paper_aligned_mistral_to_llama_20260918/results/results.json` | `remote-recorded`; missing bare target-Llama baseline |
| Nemotron-CC-Code training | `<RUN_ROOT>/tri_nemotron-cc-code_four_stage_20260917_project493_r2/{experts,advantage_router}/results.json` | `remote-recorded`; downstream functional comparison is incomplete |
| BigCodeBench baseline | `<RUN_ROOT>/tri_code_compare_20260915/04_eval_repair2/code/bigcodebench/baseline/results.json` | lexical diagnostic only; `acc=null` is not functional pass@1 |
| Inference-only random memory | `<RUN_ROOT>/tri_inference_random_memory_20260912/`; `<RUN_ROOT>/tri_inference_random_memory_20260912_repair2/` | `remote-recorded`; memory replacement only, no retraining |
| Downstream route proportions | `run/route_proportion_downstream_20260920/submitted_jobs_route_proportions_repair1_20260920.json` and the TriviaQA retry manifest | `remote-recorded`; require status/results/routing fields before reporting proportions |
| CR control comparison | `run/cr_router_controls_20260920/submitted_jobs_repair1.json` | `remote-recorded`; distinguish random-router from random-memory inference |
| Table 3 router multi-seed | `run/table3_router_seed_multiseed_20260921/submitted_jobs.json` and repair1 manifest | `remote-recorded`/possibly pending; do not replace single-seed values until all seeds are complete |

## Compute-cost benchmark

The new inference-only benchmark is recorded in
`run/compute_cost_benchmark_eonly_20260921/submitted_jobs.json`. It compares
E-only and MemoryAthena at Small/Medium/Large/XL and records:

- synchronized elapsed seconds and seconds per sequence;
- input, output, and total tokens/s;
- peak CUDA allocated/reserved memory;
- Slurm elapsed time and actual GCD-hours from `sacct`.

The job is intentionally represented as `pending` until its dependency and
completion marker succeed. No latency, memory, or cost number is estimated in
this ledger. After completion, add the immutable `results.json`, `status.json`,
and `sacct` summary here or regenerate the dashboard data.

## Reproducibility and privacy checks

- Do not commit Hugging Face tokens, SSH material, raw user directories, or
  cluster account identifiers.
- Treat `submitted_jobs*.json` as provenance, not proof that a result file is
  complete.
- Require a completion marker, finite metrics, expected protocol metadata, and
  a nonzero result file before promoting a run to `complete`.
- Keep post-hoc label selection, oracle analysis, and test-set threshold
  tuning visibly separate from training and deployment routing.
