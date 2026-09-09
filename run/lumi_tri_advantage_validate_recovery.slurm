#!/usr/bin/env bash
#SBATCH --account=project_462001493
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=athena-adv-eval-recover
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/tri_advantage_formal_20260908T133625Z/validate-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/tri_advantage_formal_20260908T133625Z/validate-%j.err

set -euo pipefail
TASK=${TASK:?Expected task name}
ORIGINAL_JOB_ID=${ORIGINAL_JOB_ID:?Expected original GPU job ID}
ROOT=/scratch/project_462001493/mingyl/ATHENA
FIX_ROOT=$ROOT/run/tri_advantage_validation_fix_20260909
RUN_ROOT=$ROOT/run/tri_advantage_formal_20260908T133625Z
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/scratch/project_462001050/cache/xdg
export TMPDIR=/scratch/project_462001050/cache/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
cd "$ROOT"
printf 'Recovering task=%s original_job=%s recovery_job=%s GPU_required=false\n' "$TASK" "$ORIGINAL_JOB_ID" "$SLURM_JOB_ID"
srun python3 "$FIX_ROOT/validate_tri_advantage_eval.py" "$RUN_ROOT/${TASK}_reader_eval/results.json" "$TASK"
printf 'ATHENA_TRI_ADVANTAGE_VALIDATION_RECOVERY_COMPLETE %s\n' "$TASK"
