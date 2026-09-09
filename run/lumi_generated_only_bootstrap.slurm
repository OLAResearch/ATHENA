#!/usr/bin/env bash
#SBATCH --account=project_462001493
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=athena-ge-correction-stats
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/ge-correction-stats-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/ge-correction-stats-%j.err

set -euo pipefail

EXPERIMENT_TAG=${EXPERIMENT_TAG:?Submit with --export=ALL,EXPERIMENT_TAG=<tag>}
ROOT=/scratch/project_462001493/mingyl/ATHENA
CODE_ROOT=$ROOT/snapshots/$EXPERIMENT_TAG
INPUT=$ROOT/run/wiki_fair_joint_20260901T173940Z/joint_generated_only_correction_nq/results.json
OUTPUT=$ROOT/run/wiki_fair_joint_20260901T173940Z/joint_generated_only_correction_nq/results_with_bootstrap.json
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

test -f "$INPUT"
test ! -e "$OUTPUT"
test -f "$CODE_ROOT/scripts/bootstrap_fair_joint_openqa.py"
test -f "$SIF"
mkdir -p "$CACHE/tmp/$SLURM_JOB_ID"

export SINGULARITYENV_PYTHONPATH=/workspace
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_XDG_CACHE_HOME=$CACHE/xdg
export SINGULARITYENV_HF_HOME=$CACHE/huggingface
export SINGULARITYENV_HF_HUB_CACHE=$CACHE/huggingface/hub
export SINGULARITYENV_HF_DATASETS_CACHE=$CACHE/huggingface/datasets
export SINGULARITYENV_TRANSFORMERS_CACHE=$CACHE/huggingface/transformers
export SINGULARITYENV_TMPDIR=$CACHE/tmp/$SLURM_JOB_ID

cd "$ROOT"
srun singularity run \
  -B "$CODE_ROOT:/workspace" \
  -B "$ROOT:$ROOT:rw" \
  -B "/scratch/project_462001050:/scratch/project_462001050" \
  "$SIF" bash -lc "
set -euo pipefail
cd /workspace
python scripts/bootstrap_fair_joint_openqa.py \
  --results '$INPUT' \
  --bootstrap-samples 2000 \
  --seed 42
"

python3 - "$OUTPUT" <<'PY'
import json
import math
import sys

results = json.load(open(sys.argv[1]))
assert results["completed"] is True
assert results["bootstrap_completed"] is True
assert results["bootstrap_samples"] == 2000
comparison = results["paired_comparisons"]["nq"]["joint_generated_only"]
assert comparison["n_examples"] == 3609
for stats in comparison["metrics"].values():
    assert len(stats["delta_95pct_paired_bootstrap"]) == 2
    assert all(math.isfinite(value) for value in [stats["delta"], *stats["delta_95pct_paired_bootstrap"]])
print(json.dumps(comparison))
PY

printf '%s\n' ATHENA_GE_CORRECTION_BOOTSTRAP_COMPLETE
