#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=07:00:00
#SBATCH --job-name=athena-fair-target-fit
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_fair_joint_20260901T173940Z/train-%x-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_fair_joint_20260901T173940Z/train-%x-%j.err

set -euo pipefail

VARIANT=${VARIANT:?Submit with --export=ALL,VARIANT=<engram|joint>}
case "$VARIANT" in
  engram|joint) ;;
  *) printf 'Unsupported VARIANT=%s\n' "$VARIANT" >&2; exit 2 ;;
esac

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

ROOT=/scratch/project_462001493/mingyl/ATHENA
CODE_ROOT=$ROOT/snapshots/wiki_fair_joint_20260901T173940Z
RUN_ROOT=$ROOT/run/wiki_fair_joint_20260901T173940Z
SOURCE_ROOT=/scratch/project_462001050/mingyl/llm-memory-transfer/publish_bundle/models/hf_repos/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
TRAIN_OUT=$RUN_ROOT/${VARIANT}_20m_train

test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test -f "$CODE_ROOT/scripts/train_adaptor.py"
test -f "$SIF"
test ! -e "$TRAIN_OUT"
mkdir -p "$RUN_ROOT" "$CACHE/miopen/$SLURM_JOB_ID" "$CACHE/tmp/$SLURM_JOB_ID"

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
export SINGULARITYENV_HF_DATASETS_OFFLINE=1
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR=$CACHE/miopen/$SLURM_JOB_ID
export SINGULARITYENV_MIOPEN_USER_DB=$CACHE/miopen/$SLURM_JOB_ID/config
export SINGULARITYENV_TMPDIR=$CACHE/tmp/$SLURM_JOB_ID

cd "$ROOT"
srun singularity run \
  -B "$CODE_ROOT:/workspace" \
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
case '$VARIANT' in
  engram)
    python scripts/train_adaptor.py \
      --condition transferred \
      --target-model mistralai/Mistral-7B-v0.3 \
      --source-memory '$SOURCE_ROOT/memory.pt' \
      --memory-config '$SOURCE_ROOT/memory_config.json' \
      --output-dir '$TRAIN_OUT' \
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
    python scripts/train_adaptor.py \
      --condition transferred \
      --target-model mistralai/Mistral-7B-v0.3 \
      --source-memory '$SOURCE_ROOT/memory.pt' \
      --memory-config '$SOURCE_ROOT/memory_config.json' \
      --output-dir '$TRAIN_OUT' \
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
"

python3 - "$TRAIN_OUT/results.json" "$VARIANT" <<'PY'
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

printf '%s %s\n' ATHENA_LUMI_FAIR_TARGET_FIT_COMPLETE "$VARIANT"
