#!/usr/bin/env bash
#SBATCH --account=project_462001493
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=athena-router-cross-stats
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_safe_router_20260831T154022Z/cross-stats-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/wiki_safe_router_20260831T154022Z/cross-stats-%j.err

set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

ROOT=/scratch/project_462001493/mingyl/ATHENA
RUN_ROOT=$ROOT/run/wiki_safe_router_20260831T154022Z
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
TASKS=(webqa triviaqa truthfulqa hotpotqa)
INPUTS=()

for task in "${TASKS[@]}"; do
  input=$RUN_ROOT/${task}_full_eval/results.json
  output=$RUN_ROOT/${task}_full_eval/results_with_bootstrap.json
  test -f "$input"
  test ! -e "$output"
  INPUTS+=("$input")
done
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
  -B "$ROOT:/workspace" \
  -B "/scratch/project_462001050:/scratch/project_462001050" \
  "$SIF" bash -lc "
set -euo pipefail
cd /workspace
python scripts/bootstrap_dual_reader_openqa.py \
  --input-results ${INPUTS[*]} \
  --bootstrap-samples 2000 \
  --seed 42
"

python3 - "$RUN_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
for task in ("webqa", "triviaqa", "truthfulqa", "hotpotqa"):
    path = run_root / f"{task}_full_eval" / "results_with_bootstrap.json"
    results = json.load(open(path))
    assert results["completed"] is True
    assert results["bootstrap_completed"] is True
    assert results["bootstrap_samples"] == 2000
    assert set(results["paired_comparisons"][task]) == {"both", "routed"}
    summary = {}
    for mode, comparison in results["paired_comparisons"][task].items():
        summary[mode] = {}
        for metric, stats in comparison["metrics"].items():
            interval = stats["delta_95pct_paired_bootstrap"]
            assert len(interval) == 2
            assert all(math.isfinite(value) for value in [stats["delta"], *interval])
            summary[mode][metric] = {
                "delta": stats["delta"],
                "ci": interval,
            }
    print(json.dumps({task: summary}))
PY

printf '%s\n' ATHENA_LUMI_SAFE_ROUTER_CROSS_DATASET_BOOTSTRAP_COMPLETE
