#!/bin/bash --login
#SBATCH --account=project_2020132
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=10:00:00
#SBATCH --export=HOME,CSC_ENV_INIT_NON_INTERACTIVE=yes
#SBATCH --chdir=/scratch/project_2015042/mingyl/ATHENA
#SBATCH --job-name=athena-roihu-fair-eval
#SBATCH --output=/scratch/project_2015042/mingyl/ATHENA/logs/wiki_fair_joint_roihu_20260902T111237Z/eval-%x-%j.out
#SBATCH --error=/scratch/project_2015042/mingyl/ATHENA/logs/wiki_fair_joint_roihu_20260902T111237Z/eval-%x-%j.err

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

TASK=${TASK:?Submit with --export=ALL,TASK=<nq|webqa|triviaqa|truthfulqa|hotpotqa>}
case "$TASK" in
  nq) EXPECTED_COUNT=3609 ;;
  webqa) EXPECTED_COUNT=2032 ;;
  triviaqa) EXPECTED_COUNT=17944 ;;
  truthfulqa) EXPECTED_COUNT=817 ;;
  hotpotqa) EXPECTED_COUNT=7405 ;;
  *) printf 'Unsupported TASK=%s\n' "$TASK" >&2; exit 2 ;;
esac

ROOT=/scratch/project_2015042/mingyl/ATHENA
TAG=wiki_fair_joint_roihu_20260902T111237Z
CODE_ROOT=$ROOT/snapshots/$TAG
RUN_ROOT=$ROOT/run/$TAG
SOURCE_ROOT=$ROOT/assets/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_2015042/cache
BASELINE_OUT=$RUN_ROOT/engram_20m_train
JOINT_OUT=$RUN_ROOT/joint_20m_train
EVAL_OUT=$RUN_ROOT/${TASK}_full_eval

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
export ATHENA_TMPDIR=$CACHE/tmp/athena-fair-eval-$SLURM_JOB_ID
export TMPDIR=$ATHENA_TMPDIR

test -f "$BASELINE_OUT/adaptor_best.pt"
test -f "$BASELINE_OUT/config.json"
test -f "$JOINT_OUT/adaptor_best.pt"
test -f "$JOINT_OUT/config.json"
test -f "$CODE_ROOT/scripts/eval_fair_joint_openqa_paired.py"
test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test ! -e "$EVAL_OUT"
mkdir -p "$RUN_ROOT" "$TMPDIR"

module load python-pytorch/2.10

cd "$CODE_ROOT"
srun python3 -u scripts/eval_fair_joint_openqa_paired.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --baseline-adaptor-dir "$BASELINE_OUT" \
  --baseline-adaptor-checkpoint "$BASELINE_OUT/adaptor_best.pt" \
  --joint-adaptor-dir "$JOINT_OUT" \
  --joint-adaptor-checkpoint "$JOINT_OUT/adaptor_best.pt" \
  --source-memory "$SOURCE_ROOT/memory.pt" \
  --memory-config "$SOURCE_ROOT/memory_config.json" \
  --output-dir "$EVAL_OUT" \
  --tasks "$TASK" \
  --canon-mode word_boundary \
  --reasoning-mode vanilla \
  --max-new-tokens 15 \
  --bootstrap-samples 0 \
  --seed 42

srun python3 - "$EVAL_OUT/results.json" "$TASK" "$EXPECTED_COUNT" <<'PY'
import json
import math
import sys

path, task, expected_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
results = json.load(open(path))
assert results["completed"] is True
assert results["bootstrap_completed"] is False
assert list(results["tasks"]) == [task]
assert results["tasks"][task]["n_examples"] == expected_count
expected_modes = {"engram_baseline", "joint_engram_only", "joint_both", "joint_routed"}
assert set(results["evaluation"]) == expected_modes
summary = {}
for mode in expected_modes:
    metrics = results["evaluation"][mode][task]["metrics"]
    assert metrics["total"] == expected_count
    if task == "truthfulqa":
        keys = ("mc1", "mc2", "mc3", "mc_avg")
        assert len(metrics["sample_examples"]) == expected_count
    else:
        keys = ("em", "f1")
        assert len(metrics["sample_predictions"]) == expected_count
    assert all(math.isfinite(metrics[key]) for key in keys)
    summary[mode] = {key: metrics[key] for key in (*keys, "total", "elapsed_s")}
print(json.dumps({task: summary}))
PY

printf '%s %s\n' ATHENA_ROIHU_FAIR_JOINT_EVAL_COMPLETE "$TASK"
