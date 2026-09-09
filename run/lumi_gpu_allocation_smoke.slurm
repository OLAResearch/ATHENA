#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --exclude=nid005026,nid005027,nid005058,nid007855
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:05:00
#SBATCH --job-name=athena-gpu-alloc-smoke
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/gpu-alloc-smoke-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/gpu-alloc-smoke-%j.err

set -euo pipefail

ROOT=/scratch/project_462001493/mingyl/ATHENA
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

mkdir -p "$ROOT/logs" "$CACHE/miopen/$SLURM_JOB_ID" "$CACHE/tmp/$SLURM_JOB_ID"
export SINGULARITYENV_XDG_CACHE_HOME=$CACHE/xdg
export SINGULARITYENV_HF_HOME=$CACHE/huggingface
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR=$CACHE/miopen/$SLURM_JOB_ID
export SINGULARITYENV_MIOPEN_USER_DB=$CACHE/miopen/$SLURM_JOB_ID/config
export SINGULARITYENV_TMPDIR=$CACHE/tmp/$SLURM_JOB_ID

printf 'job_id=%s host=%s slurm_job_gpus=%s\n' "$SLURM_JOB_ID" "$(hostname)" "${SLURM_JOB_GPUS:-NONE}"
scontrol show job -dd "$SLURM_JOB_ID"
test -n "${SLURM_JOB_GPUS:-}"
test -f "$SIF"

srun singularity run \
  -B "$ROOT:$ROOT:rw" \
  -B "/scratch/project_462001050:/scratch/project_462001050" \
  "$SIF" bash -lc "
set -euo pipefail
python - <<'PY'
import torch

print('torch=' + torch.__version__)
print('hip=' + str(torch.version.hip))
print('gpu_count=' + str(torch.cuda.device_count()))
print('gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
PY
"

printf '%s\n' ATHENA_LUMI_GPU_ALLOCATION_SMOKE_COMPLETE
