#!/bin/bash --login
#SBATCH --job-name=athena-safe-router-smoke
#SBATCH --account=project_2020132
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --export=HOME,CSC_ENV_INIT_NON_INTERACTIVE=yes
#SBATCH --output=/scratch/project_2015042/mingyl/ATHENA/run/wiki_safe_router_20260831T154022Z/logs/%x-%j.out
#SBATCH --error=/scratch/project_2015042/mingyl/ATHENA/run/wiki_safe_router_20260831T154022Z/logs/%x-%j.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export HF_HOME=/scratch/project_2015042/cache/huggingface
export HF_HUB_CACHE=/scratch/project_2015042/cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/project_2015042/cache/huggingface/datasets
export XDG_CACHE_HOME=/scratch/project_2015042/cache/xdg
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export ATHENA_TMPDIR=/scratch/project_2015042/cache/tmp/athena-safe-router-${SLURM_JOB_ID}
mkdir -p "$ATHENA_TMPDIR"
export TMPDIR="$ATHENA_TMPDIR"

cd /scratch/project_2015042/mingyl/ATHENA
module load python-pytorch/2.10

srun python3 -u scripts/train_advantage_router.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --adaptor-dir run/wiki_safe_router_20260831T154022Z/source_experts \
  --source-memory run/wiki_safe_router_20260831T154022Z/source_experts/memory.pt \
  --memory-config run/wiki_safe_router_20260831T154022Z/source_experts/memory_config.json \
  --output-dir run/wiki_safe_router_20260831T154022Z/smoke \
  --max-tokens 2048 \
  --validation-max-tokens 2048 \
  --seq-len 2048 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --eval-every 1 \
  --log-every 1 \
  --warmup-steps 1 \
  --wikipedia2021-source-tokenizer mistralai/Mistral-7B-v0.3 \
  --wikipedia2021-require-tokenizer-match

printf '%s\n' ATHENA_ROIHU_SAFE_ROUTER_SMOKE_COMPLETE
