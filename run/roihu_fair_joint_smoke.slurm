#!/bin/bash --login
#SBATCH --job-name=athena-fair-joint-smoke
#SBATCH --account=project_2020132
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --export=HOME,CSC_ENV_INIT_NON_INTERACTIVE=yes
#SBATCH --chdir=/scratch/project_2015042/mingyl/ATHENA
#SBATCH --output=/scratch/project_2015042/mingyl/ATHENA/logs/wiki_fair_joint_roihu_20260902T111237Z/smoke-%x-%j.out
#SBATCH --error=/scratch/project_2015042/mingyl/ATHENA/logs/wiki_fair_joint_roihu_20260902T111237Z/smoke-%x-%j.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

ROOT=/scratch/project_2015042/mingyl/ATHENA
TAG=wiki_fair_joint_roihu_20260902T111237Z
CODE_ROOT=$ROOT/snapshots/$TAG
RUN_ROOT=$ROOT/run/$TAG
SOURCE_ROOT=$ROOT/assets/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_2015042/cache
SMOKE_OUT=$RUN_ROOT/smoke_joint

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
export ATHENA_TMPDIR=$CACHE/tmp/athena-fair-joint-smoke-$SLURM_JOB_ID
export TMPDIR=$ATHENA_TMPDIR

test -f "$CODE_ROOT/scripts/train_adaptor.py"
test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test ! -e "$SMOKE_OUT"
mkdir -p "$RUN_ROOT" "$TMPDIR"

module load python-pytorch/2.10

srun python3 - <<'PY'
import torch
import transformers
import datasets

print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
print("transformers=" + transformers.__version__)
print("datasets=" + datasets.__version__)
print("gpu=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"))
assert torch.cuda.is_available()
assert "GH200" in torch.cuda.get_device_name(0)
PY

cd "$CODE_ROOT"
srun python3 -u scripts/train_adaptor.py \
  --condition transferred \
  --target-model mistralai/Mistral-7B-v0.3 \
  --source-memory "$SOURCE_ROOT/memory.pt" \
  --memory-config "$SOURCE_ROOT/memory_config.json" \
  --output-dir "$SMOKE_OUT" \
  --max-tokens 4096 \
  --validation-max-tokens 2048 \
  --seq-len 2048 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --lr 3e-5 \
  --router-lr 1e-3 \
  --warmup-steps 1 \
  --eval-every 1 \
  --log-every 1 \
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
  --router-start-tokens 2048 \
  --router-init-alpha 0.95 \
  --generated-residual-penalty 1e-4 \
  --canon-mode word_boundary \
  --corpus wikipedia-2021 \
  --gradient-checkpointing \
  --wikipedia2021-source-tokenizer mistralai/Mistral-7B-v0.3 \
  --wikipedia2021-require-tokenizer-match \
  --skip-final-test-eval \
  --seed 42

srun python3 - "$SMOKE_OUT" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

out = Path(sys.argv[1])
results = json.loads((out / "results.json").read_text())
assert results["completed"] is True
assert results["training_data"] == "wikipedia-2021-causal-next-token-only"
assert results["training_design"] == "single_stage_joint_engram_generated_router"
assert results["max_tokens"] == 4096
assert results["actual_steps"] == 2
assert results["best_step"] in {1, 2}
assert results["trainable_params"] > 0
assert set(results["validation"]) == {"engram_only", "both", "routed"}
assert all(math.isfinite(row["ppl"]) for row in results["validation"].values())

rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
train_rows = [row for row in rows if row.get("type") != "eval"]
assert [row["reader_mode"] for row in train_rows] == ["both", "routed"]
assert all(row["grad_norm_backbone"] == 0.0 for row in train_rows)
assert all(row["grad_norm_memory"] == 0.0 for row in train_rows)
assert all(math.isfinite(row["lm_loss"]) for row in train_rows)

state = torch.load(out / "adaptor.pt", map_location="cpu", weights_only=True)
router_outputs = [value for key, value in state.items() if key.endswith("router.2.weight")]
generated_outputs = [value for key, value in state.items() if key.endswith("output_projection.weight")]
assert router_outputs and any(torch.count_nonzero(value).item() > 0 for value in router_outputs)
assert generated_outputs and any(torch.count_nonzero(value).item() > 0 for value in generated_outputs)
print(json.dumps({
    "actual_steps": results["actual_steps"],
    "best_step": results["best_step"],
    "best_val_ppl": results["best_val_ppl"],
    "trainable_params": results["trainable_params"],
    "reader_modes": [row["reader_mode"] for row in train_rows],
    "router_updated": True,
    "generated_output_updated": True,
    "validation": results["validation"],
}))
PY

printf '%s\n' ATHENA_ROIHU_FAIR_JOINT_SMOKE_COMPLETE
