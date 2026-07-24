#!/bin/bash

#SBATCH --job-name=uc_psi_iem_full
#SBATCH --output=/home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/slurm_logs/%x_%j.err
#SBATCH --partition=gpu-h200
#SBATCH --account=gpu-research
#SBATCH --time=24:00:00
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
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-psi-${SLURM_JOB_ID:-manual}
export TORCHINDUCTOR_COMPILE_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/dcor/vishnevsky/unsupervised_creativity_psi
CONDA_ROOT=/home/dcor/vishnevsky/anaconda3
ENV_NAME=psi_sampler
PYTHON="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
CONFIG="$REPO/config/creativity.yaml"
DATA="$REPO/data/aesthetic_10.txt"
RESULTS="$REPO/results_creativity_iem_full_h200_${SLURM_JOB_ID:-manual}"
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
rsync -a "$SHARED_HF_HUB/$FLUX_CACHE/" "$LOCAL_HF_HUB/$FLUX_CACHE/"
echo "FLUX cache staged in $((SECONDS - stage_start)) seconds"

export HF_HOME="$LOCAL_HF_HOME"
export HF_DATASETS_CACHE="$LOCAL_HF_HOME/datasets"
export HF_HUB_CACHE="$LOCAL_HF_HUB"
export TRANSFORMERS_CACHE="$LOCAL_HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

cd "$REPO"

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

"$PYTHON" -u main.py \
  --tag full_h200_fixed_noise \
  --config "$CONFIG" \
  --data_path "$DATA" \
  --save_dir "$RESULTS" \
  --save_reward \
  --save_baseline_comparison \
  log_timing=true \
  noise_table_mode=fixed \
  noise_table_count=1 \
  reference_sample_count=48 \
  reference_batch_size=1 \
  reference_num_inference_steps=4 \
  num_steps=64 \
  level_batch_size=4 \
  num_mcmc_steps=25 \
  num_chains=5 \
  burn_in=25 \
  num_particles=10 \
  num_inference_steps=25 \
  grad_minibatch_size=1 \
  mini_batch_size=1

echo "Job finished at $(date)"
