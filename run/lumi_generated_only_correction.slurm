#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
# NQ evaluates the matched baseline and the standalone GE correction path.
# Keep enough walltime for the complete uncapped validation split.
#SBATCH --time=06:00:00
#SBATCH --job-name=athena-ge-correction
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/ge-correction-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/ge-correction-%j.err

set -euo pipefail

EXPERIMENT_TAG=${EXPERIMENT_TAG:?Submit with --export=ALL,EXPERIMENT_TAG=<tag>}
ROOT=/scratch/project_462001493/mingyl/ATHENA
CODE_ROOT=$ROOT/snapshots/$EXPERIMENT_TAG
OLD_RUN_ROOT=$ROOT/run/wiki_fair_joint_20260901T173940Z
BASELINE_OUT=$OLD_RUN_ROOT/engram_20m_train
JOINT_OUT=$OLD_RUN_ROOT/joint_20m_train
EVAL_OUT=$OLD_RUN_ROOT/joint_generated_only_correction_nq
SOURCE_ROOT=/scratch/project_462001050/mingyl/llm-memory-transfer/publish_bundle/models/hf_repos/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

test -f "$BASELINE_OUT/adaptor_best.pt"
test -f "$BASELINE_OUT/config.json"
test -f "$JOINT_OUT/adaptor_best.pt"
test -f "$JOINT_OUT/config.json"
test -f "$CODE_ROOT/scripts/eval_fair_joint_openqa_paired.py"
test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test -f "$SIF"
# The first attempt may have created this directory before failing during
# model setup.  Reuse it only when no result exists; never overwrite a real
# evaluation artifact.
test ! -e "$EVAL_OUT/results.json"
mkdir -p "$EVAL_OUT" "$ROOT/logs" "$CACHE/miopen/$SLURM_JOB_ID" "$CACHE/tmp/$SLURM_JOB_ID"
test -w "$EVAL_OUT"

export SINGULARITYENV_PYTHONPATH=/workspace
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_XDG_CACHE_HOME=$CACHE/xdg
export SINGULARITYENV_HF_HOME=$CACHE/huggingface
export SINGULARITYENV_HF_HUB_CACHE=$CACHE/huggingface/hub
export SINGULARITYENV_HF_DATASETS_CACHE=$CACHE/huggingface/datasets
export SINGULARITYENV_TRANSFORMERS_CACHE=$CACHE/huggingface/transformers
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR=$CACHE/miopen/$SLURM_JOB_ID
export SINGULARITYENV_MIOPEN_USER_DB=$CACHE/miopen/$SLURM_JOB_ID/config
export SINGULARITYENV_TMPDIR=$CACHE/tmp/$SLURM_JOB_ID

cd "$ROOT"
srun singularity run \
  -B "$CODE_ROOT:/workspace" \
  -B "$ROOT:$ROOT:rw" \
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
python scripts/eval_fair_joint_openqa_paired.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --baseline-adaptor-dir '$BASELINE_OUT' \
  --baseline-adaptor-checkpoint '$BASELINE_OUT/adaptor_best.pt' \
  --joint-adaptor-dir '$JOINT_OUT' \
  --joint-adaptor-checkpoint '$JOINT_OUT/adaptor_best.pt' \
  --source-memory '$SOURCE_ROOT/memory.pt' \
  --memory-config '$SOURCE_ROOT/memory_config.json' \
  --output-dir '$EVAL_OUT' \
  --tasks nq \
  --candidate-modes joint_generated_only \
  --canon-mode word_boundary \
  --reasoning-mode vanilla \
  --max-new-tokens 15 \
  --bootstrap-samples 0 \
  --seed 42
"

python3 - "$EVAL_OUT/results.json" <<'PY'
import json
import math
import sys

results = json.load(open(sys.argv[1]))
assert results["completed"] is True
assert results["tasks"]["nq"]["n_examples"] == 3609
assert set(results["evaluation"]) == {"engram_baseline", "joint_generated_only"}
metrics = results["evaluation"]["joint_generated_only"]["nq"]["metrics"]
assert metrics["total"] == 3609
assert len(metrics["sample_predictions"]) == 3609
assert all(math.isfinite(metrics[key]) for key in ("em", "f1"))
print(json.dumps({key: metrics[key] for key in ("em", "f1", "correct", "total", "elapsed_s")}))
PY

printf '%s\n' ATHENA_GE_CORRECTION_EVAL_COMPLETE
