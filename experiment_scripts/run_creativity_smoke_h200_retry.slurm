#!/bin/bash

#SBATCH --job-name=uc_psi_iem_retry
#SBATCH --output=/home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/slurm_logs/%x_%j.err
#SBATCH --partition=gpu-h200
#SBATCH --account=gpu-research
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@1200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=h200
#SBATCH --mail-user=vishnevsky@mail.tau.ac.il
#SBATCH --mail-type=ALL

set -eo pipefail

export PIP_CACHE_DIR=/home/dcor/vishnevsky/.cache/pip
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID:-manual}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-psi-h200-torch2_5
export TORCHINDUCTOR_COMPILE_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/dcor/vishnevsky/unsupervised_creativity_psi
CONDA_ROOT=/home/dcor/vishnevsky/anaconda3
ENV_NAME=psi_sampler
PYTHON="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
CONFIG="$REPO/config/creativity.yaml"
DATA="$REPO/data/aesthetic_smoke.txt"
RESULTS="$REPO/results_creativity_iem_h200_retry_${SLURM_JOB_ID:-manual}"
SHARED_HF_HUB=/home/dcor/vishnevsky/.cache/huggingface/hub
LOCAL_HF_HOME=/tmp/huggingface-psi
LOCAL_HF_HUB="$LOCAL_HF_HOME/hub"
FLUX_CACHE=models--black-forest-labs--FLUX.1-schnell

mkdir -p \
  "$REPO/experiment_scripts/slurm_logs" \
  "$RESULTS" \
  "$MPLCONFIGDIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$LOCAL_HF_HUB"

echo "Job starting at $(date)"
echo "Repo: $REPO"
echo "Config: $CONFIG"
echo "Data: $DATA"
echo "Results: $RESULTS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

stage_start=$SECONDS
echo "Staging FLUX cache to $LOCAL_HF_HUB"
(
  flock 9
  rsync -a "$SHARED_HF_HUB/$FLUX_CACHE/" "$LOCAL_HF_HUB/$FLUX_CACHE/"
) 9>/tmp/huggingface-psi-stage.lock
echo "FLUX cache staged in $((SECONDS - stage_start)) seconds"

FLUX_REVISION=$(tr -d '\r\n' <"$LOCAL_HF_HUB/$FLUX_CACHE/refs/main")
FLUX_SNAPSHOT="$LOCAL_HF_HUB/$FLUX_CACHE/snapshots/$FLUX_REVISION"
if [[ ! -f "$FLUX_SNAPSHOT/model_index.json" ]]; then
  echo "Missing staged FLUX model_index.json: $FLUX_SNAPSHOT" >&2
  exit 1
fi
BROKEN_LINK=$(find -L "$FLUX_SNAPSHOT" -type l -print -quit)
if [[ -n "$BROKEN_LINK" ]]; then
  echo "Broken staged FLUX snapshot link: $BROKEN_LINK" >&2
  exit 1
fi
echo "FLUX snapshot: $FLUX_SNAPSHOT"

export HF_HOME="$LOCAL_HF_HOME"
export HF_DATASETS_CACHE="$LOCAL_HF_HOME/datasets"
export HF_HUB_CACHE="$LOCAL_HF_HUB"
export TRANSFORMERS_CACHE="$LOCAL_HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

NVIDIA_LIB_PATH=
for lib_dir in "$CONDA_PREFIX"/lib/python3.10/site-packages/nvidia/*/lib; do
  NVIDIA_LIB_PATH="${NVIDIA_LIB_PATH:+$NVIDIA_LIB_PATH:}$lib_dir"
done
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set -u

cd "$REPO"

"$PYTHON" - <<'PY'
import ctypes
import torch

ctypes.CDLL("libnvrtc.so.12")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

"$PYTHON" -u main.py \
  --tag smoke_h200_retry \
  --config "$CONFIG" \
  --data_path "$DATA" \
  --save_dir "$RESULTS" \
  --save_reward \
  --save_baseline_comparison \
  log_timing=true \
  pretrained_model_name_or_path="$FLUX_SNAPSHOT" \
  noise_table_mode=fixed \
  noise_table_count=1 \
  reference_sample_count=3 \
  reference_batch_size=1 \
  num_steps=2 \
  level_batch_size=1 \
  num_mcmc_steps=1 \
  num_chains=1 \
  burn_in=1 \
  num_particles=1 \
  num_inference_steps=2 \
  grad_minibatch_size=1 \
  mini_batch_size=1

echo "Job finished at $(date)"
