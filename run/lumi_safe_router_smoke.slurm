#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:08:00
#SBATCH --job-name=athena-safe-router-smoke
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_safe_router_20260831T154022Z/smoke-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_safe_router_20260831T154022Z/smoke-%j.err

set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

ROOT=/scratch/project_462001493/mingyl/ATHENA
RUN_ROOT=$ROOT/run/wiki_safe_router_20260831T154022Z
EXPERT_ROOT=$ROOT/run/wiki_residual_20260830T130138Z/wiki_20m_train
SOURCE_ROOT=/scratch/project_462001050/mingyl/llm-memory-transfer/publish_bundle/models/hf_repos/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
TRAIN_OUT=$RUN_ROOT/smoke

test -f "$EXPERT_ROOT/adaptor_best.pt"
test -f "$EXPERT_ROOT/config.json"
test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test -f "$SIF"
test ! -e "$TRAIN_OUT"
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
python scripts/train_advantage_router.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --adaptor-dir '$EXPERT_ROOT' \
  --source-memory '$SOURCE_ROOT/memory.pt' \
  --memory-config '$SOURCE_ROOT/memory_config.json' \
  --output-dir '$TRAIN_OUT' \
  --max-tokens 512 \
  --validation-max-tokens 512 \
  --seq-len 512 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --warmup-steps 0 \
  --eval-every 1 \
  --log-every 1 \
  --gradient-checkpointing \
  --wikipedia2021-source-tokenizer mistralai/Mistral-7B-v0.3 \
  --wikipedia2021-require-tokenizer-match
"

python3 - "$TRAIN_OUT/results.json" "$TRAIN_OUT/train_log.jsonl" <<'PY'
import json
import math
import sys

results = json.load(open(sys.argv[1]))
log = json.loads(open(sys.argv[2]).readline())
assert results["completed"] is True
assert results["training_data"] == "wikipedia-2021-causal-next-token-only"
assert results["downstream_training_examples"] == 0
assert results["actual_steps"] == 1
assert results["trainable_names"]
assert all("router" in name for name in results["trainable_names"])
assert log["router_grad_norm"] > 0
assert math.isfinite(results["best_val_ppl"])
assert len(log["router_weights"]) == 2
assert abs(sum(log["router_weights"]) - 1.0) < 1e-4
print(json.dumps({
    "step": results["actual_steps"],
    "best_val_ppl": results["best_val_ppl"],
    "router_grad_norm": log["router_grad_norm"],
    "router_weights": log["router_weights"],
}))
PY

printf '%s\n' ATHENA_LUMI_SAFE_ROUTER_SMOKE_COMPLETE
