#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
# The measured first run needs about 33--36 hours on one GCD.  Keep a
# correctness margin while avoiding the partition's full 3-day reservation.
#SBATCH --time=1-16:00:00
#SBATCH --job-name=athena-tri-memory20m
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/tri-memory-train-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/tri-memory-train-%j.err

set -euo pipefail

EXPERIMENT_TAG=${EXPERIMENT_TAG:?Submit with --export=ALL,EXPERIMENT_TAG=<tag>}
ROOT=/scratch/project_462001493/mingyl/ATHENA
CODE_ROOT=$ROOT/snapshots/$EXPERIMENT_TAG
RUN_ROOT=$ROOT/run/$EXPERIMENT_TAG
SOURCE_ROOT=/scratch/project_462001050/mingyl/llm-memory-transfer/publish_bundle/models/hf_repos/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
TRAIN_OUT=$RUN_ROOT/tri_20m_train

test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test -f "$CODE_ROOT/scripts/train_adaptor.py"
test -f "$CODE_ROOT/engram/tri_memory.py"
test -f "$SIF"
test ! -e "$TRAIN_OUT"
mkdir -p "$TRAIN_OUT" "$CACHE/miopen/$SLURM_JOB_ID" "$CACHE/tmp/$SLURM_JOB_ID"
test -w "$TRAIN_OUT"

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
  --generator-cue-source hybrid \
  --generator-num-latents 4 \
  --generator-hidden-size 256 \
  --generator-layers 2 \
  --generator-heads 4 \
  --generator-cue-window 3 \
  --generator-fusion-type tri_reader \
  --generator-router-hidden-size 64 \
  --generator-router-semantic-size 64 \
  --joint-tri-subset-reader \
  --deterministic-engram-init-seed 42 \
  --generated-residual-penalty 0 \
  --tri-router-distillation-weight 0.5 \
  --canon-mode word_boundary \
  --corpus wikipedia-2021 \
  --gradient-checkpointing \
  --wikipedia2021-source-tokenizer mistralai/Mistral-7B-v0.3 \
  --wikipedia2021-require-tokenizer-match \
  --skip-final-test-eval \
  --seed 42
"

python3 - "$TRAIN_OUT/results.json" "$TRAIN_OUT/train_log.jsonl" <<'PY'
import json
import math
import sys

results = json.load(open(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[2])]
training = next(row for row in rows if row.get("type") != "eval")
assert results["completed"] is True
assert results["training_design"] == "token_zero_joint_E_GE_GH_unified_seven_way_subset_reader"
assert results["training_data"] == "wikipedia-2021-causal-next-token-only"
assert results["max_tokens"] == 20_000_000
assert results["actual_steps"] == 9765
assert results["best_step"] > 0
assert math.isfinite(results["best_val_ppl"])
expected = {
    "engram_only",
    "generated_from_engram_only",
    "generated_from_context_only",
    "e_ge",
    "e_gh",
    "ge_gh",
    "tri_soft_fused",
    "tri_subset_routed",
    "tri_subset_soft_fused",
}
assert set(results["validation"]) == expected
assert set(training["path_lm_losses"]) == {
    "engram_only",
    "generated_from_engram_only",
    "generated_from_context_only",
    "e_ge",
    "e_gh",
    "ge_gh",
    "tri_soft_fused",
    "tri_subset_soft_fused",
}
assert len(training["router_weights"]) == 3
assert len(training["router_subset_weights"]) == 7
assert training["grad_norm_backbone"] == 0.0
assert training["grad_norm_memory"] == 0.0
assert all(value > 0 for value in training["tri_grad_norms"].values())
print(json.dumps({
    "actual_steps": results["actual_steps"],
    "best_step": results["best_step"],
    "best_val_ppl": results["best_val_ppl"],
    "validation": results["validation"],
    "router_weights": training["router_weights"],
    "router_subset_weights": training["router_subset_weights"],
    "tri_grad_norms": training["tri_grad_norms"],
}))
PY

printf '%s\n' ATHENA_TRI_MEMORY_JOINT_TRAINING_COMPLETE
