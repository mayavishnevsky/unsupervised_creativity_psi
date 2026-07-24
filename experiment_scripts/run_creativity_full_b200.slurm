#!/bin/bash

#SBATCH --job-name=uc_psi_iem_full_b200
#SBATCH --output=/home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/slurm_logs/%x_%j.err
#SBATCH --partition=gpu-b200
#SBATCH --account=gpu-research
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@1200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=b200
#SBATCH --mail-user=vishnevsky@mail.tau.ac.il
#SBATCH --mail-type=ALL

set -eo pipefail

export ENV_NAME_OVERRIDE=psi_sampler_b200
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-psi-b200-torch2_7

exec bash /home/dcor/vishnevsky/unsupervised_creativity_psi/experiment_scripts/run_creativity_full_h200.slurm
