#!/usr/bin/env bash
#SBATCH --account=project_462001616
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=athena-adv-20m
#SBATCH --output=/scratch/project_462001493/mingyl/ATHENA/logs/adv-train-%j.out
#SBATCH --error=/scratch/project_462001493/mingyl/ATHENA/logs/adv-train-%j.err

set -euo pipefail

EXPERIMENT_TAG=tri_advantage_formal_20260908T133625Z
ROOT=/scratch/project_462001493/mingyl/ATHENA
CODE_ROOT=$ROOT/snapshots/$EXPERIMENT_TAG
RUN_ROOT=$ROOT/run/$EXPERIMENT_TAG
SOURCE_ROOT=/scratch/project_462001050/mingyl/llm-memory-transfer/publish_bundle/models/hf_repos/olaresearchx-xmemtransfer-llama2-source-20m
CACHE=/scratch/project_462001050/cache
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
TRAIN_OUT=$RUN_ROOT/reader_20m_train

test -f "$SOURCE_ROOT/memory.pt"
test -f "$SOURCE_ROOT/memory_config.json"
test -f "$CODE_ROOT/scripts/train_adaptor.py"
test -f "$CODE_ROOT/engram/tri_memory.py"
test -f "$SIF"
test ! -e "$TRAIN_OUT"
mkdir -p "$TRAIN_OUT" "$CACHE/miopen/$SLURM_JOB_ID" "$CACHE/tmp/$SLURM_JOB_ID"
test -w "$TRAIN_OUT"

export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
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

export XDG_CACHE_HOME=$CACHE/xdg
export SINGULARITY_CACHEDIR=$CACHE/singularity
export TMPDIR=$CACHE/tmp/$SLURM_JOB_ID
export SINGULARITYENV_PIP_CACHE_DIR=$CACHE/pip
cd "$ROOT"
srun singularity run \
  -B "$CODE_ROOT:/workspace" \
  -B "$ROOT:$ROOT:rw" \
  -B "/scratch/project_462001050:/scratch/project_462001050" \
  "$SIF" bash -lc "
set -euo pipefail
cd /workspace
python - <<'PY'
import torch,socket
print('hostname=' + socket.gethostname())
print('torch=' + torch.__version__)
print('hip=' + str(torch.version.hip))
print('gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))
assert torch.cuda.is_available()
PY
python scripts/train_counterfactual_router.py \
  --adaptor-dir '$ROOT/run/tri_memory_joint_softsubset_20260903T053716Z/tri_20m_train' \
  --source-memory '$SOURCE_ROOT/memory.pt' \
  --memory-config '$SOURCE_ROOT/memory_config.json' \
  --output-dir '$TRAIN_OUT' \
  --candidate-space sources \
  --max-tokens 20000000 --validation-max-tokens 2000000 --seq-len 2048 \
  --batch-size 1 --grad-accum-steps 1 --warmup-steps 200 \
  --eval-every 2000 --log-every 50 \
  --wikipedia2021-source-tokenizer mistralai/Mistral-7B-v0.3 \
  --wikipedia2021-require-tokenizer-match --seed 42
python - '$TRAIN_OUT/results.json' <<'PYVERIFY'
import json,sys,math
r=json.load(open(sys.argv[1]))
assert r['completed'] is True
assert r['max_tokens'] == 20000000
assert r['actual_steps'] == 9765
assert r['best_step'] > 0
assert all('advantage_router' in n for n in r['trainable_names'])
assert set(r['validation']) == {'engram_only','tri_soft_fused','tri_routed','tri_advantage_routed'}
assert all(math.isfinite(v['ppl']) for v in r['validation'].values())
print(json.dumps(r))
print('ATHENA_TRI_ADVANTAGE_FORMAL_TRAINING_COMPLETE')
PYVERIFY
"
