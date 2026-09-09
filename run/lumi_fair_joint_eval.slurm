#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --job-name=athena-fair-joint-eval
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_fair_joint_20260901T173940Z/eval-%x-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_fair_joint_20260901T173940Z/eval-%x-%j.err

set -euo pipefail

TASK=${TASK:?Submit with --export=ALL,TASK=<nq|webqa|triviaqa|truthfulqa|hotpotqa>}
case "$TASK" in
  nq) EXPECTED_COUNT=3609 ;;
  webqa) EXPECTED_COUNT=2032 ;;
  triviaqa) EXPECTED_COUNT=17944 ;;
  truthfulqa) EXPECTED_COUNT=817 ;;
  hotpotqa) EXPECTED_COUNT=7405 ;;
  *) printf 'Unsupported TASK=%s\n' "$TASK" >&2; exit 2 ;;
esac

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

ROOT=/scratch/project_462001493/mingyl/ATHENA
CODE_ROOT=$ROOT/snapshots/wiki_fair_joint_20260901T173940Z
RUN_ROOT=$ROOT/run/wiki_fair_joint_20260901T173940Z
BASELINE_OUT=$RUN_ROOT/engram_20m_train
JOINT_OUT=$RUN_ROOT/joint_20m_train
SOURCE_ROOT=/scratch/project_462001050/mingyl/llm-memory-transfer/publish_bundle/models/hf_repos/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
EVAL_OUT=$RUN_ROOT/${TASK}_full_eval

test -f "$BASELINE_OUT/adaptor_best.pt"
test -f "$BASELINE_OUT/config.json"
test -f "$JOINT_OUT/adaptor_best.pt"
test -f "$JOINT_OUT/config.json"
test -f "$CODE_ROOT/scripts/eval_fair_joint_openqa_paired.py"
test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test -f "$SIF"
test ! -e "$EVAL_OUT"
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
python scripts/eval_fair_joint_openqa_paired.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --baseline-adaptor-dir '$BASELINE_OUT' \
  --baseline-adaptor-checkpoint '$BASELINE_OUT/adaptor_best.pt' \
  --joint-adaptor-dir '$JOINT_OUT' \
  --joint-adaptor-checkpoint '$JOINT_OUT/adaptor_best.pt' \
  --source-memory '$SOURCE_ROOT/memory.pt' \
  --memory-config '$SOURCE_ROOT/memory_config.json' \
  --output-dir '$EVAL_OUT' \
  --tasks '$TASK' \
  --canon-mode word_boundary \
  --reasoning-mode vanilla \
  --max-new-tokens 15 \
  --bootstrap-samples 0 \
  --seed 42
"

python3 - "$EVAL_OUT/results.json" "$TASK" "$EXPECTED_COUNT" <<'PY'
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

printf '%s %s\n' ATHENA_LUMI_FAIR_JOINT_EVAL_COMPLETE "$TASK"
