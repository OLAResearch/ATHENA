#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=1-06:00:00
#SBATCH --job-name=athena-looped-eval
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/looped-eval-%x-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/looped-eval-%x-%j.err

set -euo pipefail
: "${EXPERIMENT_TAG:?}" "${TASK:?}"
MODE=${MODE:-tri_subset_routed}
ROOT=/scratch/project_462001493/mingyl/ATHENA
CODE_ROOT=$ROOT/snapshots/$EXPERIMENT_TAG
RUN_ROOT=$ROOT/run/$EXPERIMENT_TAG
SOURCE_ROOT=/scratch/project_462001050/mingyl/llm-memory-transfer/publish_bundle/models/hf_repos/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
ADAPTOR_DIR=$RUN_ROOT/looped_wiki20m_train
EVAL_OUT=$RUN_ROOT/eval/$MODE/$TASK

test -f "$CODE_ROOT/scripts/eval_openqa.py"
test -f "$ADAPTOR_DIR/adaptor_best.pt" -o -f "$ADAPTOR_DIR/adaptor.pt"
test -f "$SOURCE_ROOT/memory.pt"
test ! -e "$EVAL_OUT/openqa_results.json"
mkdir -p "$EVAL_OUT" "$CACHE/miopen/$SLURM_JOB_ID" "$CACHE/tmp/$SLURM_JOB_ID"

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
print('gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))
assert torch.cuda.is_available()
PY
python scripts/eval_openqa.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --adaptor-dir '$ADAPTOR_DIR' \
  --source-memory '$SOURCE_ROOT/memory.pt' \
  --memory-config '$SOURCE_ROOT/memory_config.json' \
  --tasks '$TASK' \
  --conditions baseline transferred \
  --dual-reader-mode "$MODE" \
  --canon-mode word_boundary \
  --parallel-tasks 1 \
  --output-dir '$EVAL_OUT' \
  --overwrite
"
printf '%s\n' "ATHENA_LOOPED_WIKIPEDIA_${TASK}_EVAL_COMPLETE"
