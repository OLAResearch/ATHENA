#!/usr/bin/env bash
# Recovery wrapper for the timed-out TriviaQA evaluation.
# The original output directory exists but is empty; remove only that empty
# directory so the unchanged evaluation script can create its output safely.
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=athena-tri-eval-trivia-recovery
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/tri-eval-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/tri-eval-%j.err

set -euo pipefail

: "${EXPERIMENT_TAG:?Submit with --export=ALL,EXPERIMENT_TAG=<tag>}"
: "${TASK:?Submit with --export=ALL,TASK=triviaqa}"
test "$TASK" = triviaqa

ROOT=/scratch/project_462001493/mingyl/ATHENA
EVAL_OUT="$ROOT/run/$EXPERIMENT_TAG/triviaqa_tri_full_eval"
if test -d "$EVAL_OUT"; then
  test -z "$(find "$EVAL_OUT" -mindepth 1 -maxdepth 1 -print -quit)"
  rmdir "$EVAL_OUT"
else
  test ! -e "$EVAL_OUT"
fi

exec bash "$ROOT/scripts/$EXPERIMENT_TAG/lumi_tri_memory_eval.slurm"
