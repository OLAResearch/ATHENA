#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=athena-safe-router-nq
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_safe_router_20260831T154022Z/nq-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_safe_router_20260831T154022Z/nq-%j.err

set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

ROOT=/scratch/project_462001493/mingyl/ATHENA
RUN_ROOT=$ROOT/run/wiki_safe_router_20260831T154022Z
TRAIN_OUT=$RUN_ROOT/wiki_20m_train
SOURCE_ROOT=/scratch/project_462001050/mingyl/llm-memory-transfer/publish_bundle/models/hf_repos/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
EVAL_OUT=$RUN_ROOT/nq_full_eval

test -f "$TRAIN_OUT/adaptor_best.pt"
test -f "$TRAIN_OUT/config.json"
test -f "$TRAIN_OUT/results.json"
test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test -f "$SIF"
test ! -e "$EVAL_OUT"
mkdir -p "$RUN_ROOT" "$CACHE/miopen/$SLURM_JOB_ID" "$CACHE/tmp/$SLURM_JOB_ID"

export SINGULARITYENV_PYTHONPATH=/workspace
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_HF_HOME=$CACHE/huggingface
export SINGULARITYENV_HF_HUB_CACHE=$CACHE/huggingface/hub
export SINGULARITYENV_HF_DATASETS_CACHE=$CACHE/huggingface/datasets
export SINGULARITYENV_TRANSFORMERS_CACHE=$CACHE/huggingface/transformers
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_HF_DATASETS_OFFLINE=1
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR=$CACHE/miopen/$SLURM_JOB_ID
export SINGULARITYENV_MIOPEN_USER_DB=$CACHE/miopen/$SLURM_JOB_ID/config
export SINGULARITYENV_TMPDIR=$CACHE/tmp/$SLURM_JOB_ID

cd "$ROOT"
srun singularity run \
  -B "$ROOT:/workspace" \
  -B "/scratch/project_462001050:/scratch/project_462001050" \
  "$SIF" bash -lc "
set -euo pipefail
cd /workspace
python - <<'PY'
import torch
print('torch=' + torch.__version__)
print('hip=' + str(torch.version.hip))
print('gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))
assert torch.cuda.is_available()
PY
python scripts/eval_dual_reader_nq_paired.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --adaptor-dir '$TRAIN_OUT' \
  --source-memory '$SOURCE_ROOT/memory.pt' \
  --memory-config '$SOURCE_ROOT/memory_config.json' \
  --output-dir '$EVAL_OUT' \
  --canon-mode word_boundary \
  --max-new-tokens 15 \
  --seed 42
"

python3 - "$EVAL_OUT/results.json" <<'PY'
import json
import math
import sys

results = json.load(open(sys.argv[1]))
assert results["completed"] is True
assert results["n_examples"] == 3_609
assert set(results["evaluation"]) == {"engram_only", "both", "routed"}
assert set(results["paired_comparisons"]) == {"both", "routed"}
for mode in ("engram_only", "both", "routed"):
    metrics = results["evaluation"][mode]["nq"]["metrics"]
    assert metrics["total"] == 3_609
    assert math.isfinite(metrics["em"])
    assert math.isfinite(metrics["f1"])
for comparison in results["paired_comparisons"].values():
    assert comparison["n_examples"] == 3_609
    assert len(comparison["em_delta_95pct_paired_bootstrap"]) == 2
    assert len(comparison["f1_delta_95pct_paired_bootstrap"]) == 2
summary = {
    mode: {
        key: results["evaluation"][mode]["nq"]["metrics"][key]
        for key in ("em", "f1", "correct", "total", "elapsed_s")
    }
    for mode in ("engram_only", "both", "routed")
}
summary["paired_comparisons"] = results["paired_comparisons"]
print(json.dumps(summary))
PY

printf '%s\n' ATHENA_LUMI_SAFE_ROUTER_NQ_COMPLETE
