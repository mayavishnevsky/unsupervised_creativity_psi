#!/bin/bash

#SBATCH --job-name=uc_psi_iem_smoke
#SBATCH --output=/home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/slurm_logs/%x_%j.err
#SBATCH --partition=gpu-h100-killable
#SBATCH --account=gpu-research
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@1200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --mail-user=vishnevsky@mail.tau.ac.il
#SBATCH --mail-type=ALL

set -eo pipefail

export HF_HOME=/home/dcor/vishnevsky/.cache/huggingface
export HF_DATASETS_CACHE=/home/dcor/vishnevsky/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/dcor/vishnevsky/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dcor/vishnevsky/.cache/huggingface/transformers
export PIP_CACHE_DIR=/home/dcor/vishnevsky/.cache/pip
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID:-manual}
export TORCHINDUCTOR_CACHE_DIR=/home/dcor/vishnevsky/.cache/torchinductor/psi_sampler_h100_torch2_5
export TORCHINDUCTOR_COMPILE_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/dcor/vishnevsky/unsupervised_creativity_psi
CONDA_ROOT=/home/dcor/vishnevsky/anaconda3
ENV_NAME=psi_sampler
PYTHON="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
CONFIG="$REPO/config/creativity.yaml"
DATA="$REPO/data/aesthetic_smoke.txt"
RESULTS="$REPO/results_creativity_smoke"

mkdir -p "$REPO/experiment_scripts/slurm_logs" "$RESULTS" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "Job starting at $(date)"
echo "Repo: $REPO"
echo "Config: $CONFIG"
echo "Data: $DATA"
echo "Results: $RESULTS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

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
  --tag smoke_h100 \
  --config "$CONFIG" \
  --data_path "$DATA" \
  --save_dir "$RESULTS" \
  --save_reward \
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
