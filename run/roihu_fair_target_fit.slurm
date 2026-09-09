#!/bin/bash --login
#SBATCH --account=project_2020132
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=06:00:00
#SBATCH --export=HOME,CSC_ENV_INIT_NON_INTERACTIVE=yes
#SBATCH --chdir=/scratch/project_2015042/mingyl/ATHENA
#SBATCH --job-name=athena-roihu-fair-fit
#SBATCH --output=/scratch/project_2015042/mingyl/ATHENA/logs/wiki_fair_joint_roihu_20260902T111237Z/train-%x-%j.out
#SBATCH --error=/scratch/project_2015042/mingyl/ATHENA/logs/wiki_fair_joint_roihu_20260902T111237Z/train-%x-%j.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

VARIANT=${VARIANT:?Submit with --export=ALL,VARIANT=<engram|joint>}
case "$VARIANT" in
  engram|joint) ;;
  *) printf 'Unsupported VARIANT=%s\n' "$VARIANT" >&2; exit 2 ;;
esac

ROOT=/scratch/project_2015042/mingyl/ATHENA
TAG=wiki_fair_joint_roihu_20260902T111237Z
CODE_ROOT=$ROOT/snapshots/$TAG
RUN_ROOT=$ROOT/run/$TAG
SOURCE_ROOT=$ROOT/assets/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_2015042/cache
TRAIN_OUT=$RUN_ROOT/${VARIANT}_20m_train

export XDG_CACHE_HOME=$CACHE/xdg
export HF_HOME=$CACHE/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export ATHENA_TMPDIR=$CACHE/tmp/athena-fair-fit-$SLURM_JOB_ID
export TMPDIR=$ATHENA_TMPDIR

test -f "$CODE_ROOT/scripts/train_adaptor.py"
test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test ! -e "$TRAIN_OUT"
mkdir -p "$RUN_ROOT" "$TMPDIR"

module load python-pytorch/2.10

srun python3 - <<'PY'
import torch
print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
print("gpu=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"))
assert torch.cuda.is_available()
assert "GH200" in torch.cuda.get_device_name(0)
PY

cd "$CODE_ROOT"
case "$VARIANT" in
  engram)
    srun python3 -u scripts/train_adaptor.py \
      --condition transferred \
      --target-model mistralai/Mistral-7B-v0.3 \
      --source-memory "$SOURCE_ROOT/memory.pt" \
      --memory-config "$SOURCE_ROOT/memory_config.json" \
      --output-dir "$TRAIN_OUT" \
      --max-tokens 20000000 \
      --validation-max-tokens 2000000 \
      --seq-len 2048 \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --lr 3e-5 \
      --warmup-steps 500 \
      --eval-every 500 \
      --log-every 50 \
      --injection-layers 2:10 \
      --adaptor-branches 4 \
      --architecture legacy \
      --deterministic-engram-init-seed 42 \
      --canon-mode word_boundary \
      --corpus wikipedia-2021 \
      --gradient-checkpointing \
      --wikipedia2021-source-tokenizer mistralai/Mistral-7B-v0.3 \
      --wikipedia2021-require-tokenizer-match \
      --skip-final-test-eval \
      --seed 42
    ;;
  joint)
    srun python3 -u scripts/train_adaptor.py \
      --condition transferred \
      --target-model mistralai/Mistral-7B-v0.3 \
      --source-memory "$SOURCE_ROOT/memory.pt" \
      --memory-config "$SOURCE_ROOT/memory_config.json" \
      --output-dir "$TRAIN_OUT" \
      --max-tokens 20000000 \
      --validation-max-tokens 2000000 \
      --seq-len 2048 \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --lr 3e-5 \
      --router-lr 1e-3 \
      --warmup-steps 500 \
      --eval-every 500 \
      --log-every 50 \
      --injection-layers 2:10 \
      --adaptor-branches 4 \
      --architecture generative \
      --reader-type cross_attention \
      --generator-cue-source engram \
      --generator-num-latents 4 \
      --generator-hidden-size 256 \
      --generator-layers 2 \
      --generator-heads 4 \
      --generator-cue-window 3 \
      --generator-fusion-type dual_reader \
      --generator-router-hidden-size 16 \
      --joint-engram-generated-router \
      --deterministic-engram-init-seed 42 \
      --router-start-tokens 2000000 \
      --router-init-alpha 0.95 \
      --generated-residual-penalty 1e-4 \
      --canon-mode word_boundary \
      --corpus wikipedia-2021 \
      --gradient-checkpointing \
      --wikipedia2021-source-tokenizer mistralai/Mistral-7B-v0.3 \
      --wikipedia2021-require-tokenizer-match \
      --skip-final-test-eval \
      --seed 42
    ;;
esac

srun python3 - "$TRAIN_OUT/results.json" "$VARIANT" <<'PY'
import json
import math
import sys

path, variant = sys.argv[1], sys.argv[2]
results = json.load(open(path))
assert results["completed"] is True
assert results["training_data"] == "wikipedia-2021-causal-next-token-only"
assert results["max_tokens"] == 20_000_000
assert results["actual_steps"] == 9_765
assert results["best_step"] > 0
assert math.isfinite(results["best_val_ppl"])
assert results["trainable_params"] > 0
expected_modes = {"engram_only"} if variant == "engram" else {"engram_only", "both", "routed"}
assert set(results["validation"]) == expected_modes
assert all(math.isfinite(row["ppl"]) for row in results["validation"].values())
print(json.dumps({
    "variant": variant,
    "actual_steps": results["actual_steps"],
    "best_step": results["best_step"],
    "best_val_ppl": results["best_val_ppl"],
    "trainable_params": results["trainable_params"],
    "validation": results["validation"],
}))
PY

printf '%s %s\n' ATHENA_ROIHU_FAIR_TARGET_FIT_COMPLETE "$VARIANT"
