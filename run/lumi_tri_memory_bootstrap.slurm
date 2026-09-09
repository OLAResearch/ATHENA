#!/usr/bin/env bash
#SBATCH --account=project_462001493
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=athena-tri-memory-stats
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/tri-memory-stats-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/tri-memory-stats-%j.err

set -euo pipefail

EXPERIMENT_TAG=${EXPERIMENT_TAG:?Submit with --export=ALL,EXPERIMENT_TAG=<tag>}
ROOT=/scratch/project_462001493/mingyl/ATHENA
CODE_ROOT=$ROOT/snapshots/$EXPERIMENT_TAG
RUN_ROOT=$ROOT/run/$EXPERIMENT_TAG
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
RESULTS=(
  "$RUN_ROOT/nq_tri_full_eval/results.json"
  "$RUN_ROOT/webqa_tri_full_eval/results.json"
  "$RUN_ROOT/triviaqa_tri_full_eval/results.json"
  "$RUN_ROOT/truthfulqa_tri_full_eval/results.json"
  "$RUN_ROOT/hotpotqa_tri_full_eval/results.json"
)

for input in "${RESULTS[@]}"; do
  test -f "$input"
  test ! -e "${input%/results.json}/results_with_bootstrap.json"
done
test -f "$CODE_ROOT/scripts/bootstrap_tri_memory_openqa.py"
test -f "$SIF"
mkdir -p "$CACHE/tmp/$SLURM_JOB_ID"

export SINGULARITYENV_PYTHONPATH=/workspace
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_XDG_CACHE_HOME=$CACHE/xdg
export SINGULARITYENV_HF_HOME=$CACHE/huggingface
export SINGULARITYENV_HF_HUB_CACHE=$CACHE/huggingface/hub
export SINGULARITYENV_HF_DATASETS_CACHE=$CACHE/huggingface/datasets
export SINGULARITYENV_TRANSFORMERS_CACHE=$CACHE/transformers
export SINGULARITYENV_TMPDIR=$CACHE/tmp/$SLURM_JOB_ID

cd "$ROOT"
srun singularity run \
  -B "$CODE_ROOT:/workspace" \
  -B "$ROOT:$ROOT:rw" \
  -B "/scratch/project_462001050:/scratch/project_462001050" \
  "$SIF" bash -lc "
set -euo pipefail
cd /workspace
python scripts/bootstrap_tri_memory_openqa.py \
  --results ${RESULTS[*]} \
  --bootstrap-samples 2000 \
  --seed 42
"

python3 - "$RUN_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
for task in ("nq", "webqa", "triviaqa", "truthfulqa", "hotpotqa"):
    path = root / f"{task}_tri_full_eval" / "results_with_bootstrap.json"
    results = json.load(open(path))
    assert results["completed"] is True
    assert results["bootstrap_completed"] is True
    assert results["bootstrap_samples"] == 2000
    comparisons = results["paired_comparisons"][task]
    required = {
        "tri_E", "tri_GE", "tri_GH", "tri_E_GE", "tri_E_GH", "tri_GE_GH",
        "tri_reader_hard", "tri_reader_soft", "tri_subset_reader_hard",
        "tri_subset_reader_soft", "oracle_E_GE", "oracle_E_GH", "oracle_GE_GH",
        "oracle_E_GE_GH", "oracle_all_nonempty_subsets",
    }
    assert required <= set(comparisons)
    for comparison in comparisons.values():
        for stats in comparison["metrics"].values():
            assert len(stats["delta_95pct_paired_bootstrap"]) == 2
            assert all(math.isfinite(value) for value in [stats["delta"], *stats["delta_95pct_paired_bootstrap"]])
print("validated five paired bootstrap result files")
PY

printf '%s\n' ATHENA_TRI_MEMORY_BOOTSTRAP_COMPLETE
