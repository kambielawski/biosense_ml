#!/bin/bash
#SBATCH --job-name=biosense-autoencoder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# Load modules (customize for your cluster)
# module load cuda/12.x python/3.10

mkdir -p logs

source .venv/bin/activate

python scripts/train_autoencoder.py \
    model=autoencoder \
    "$@"  # pass additional overrides
