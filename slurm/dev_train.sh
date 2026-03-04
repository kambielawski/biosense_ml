#!/bin/bash
#SBATCH --job-name=biosense-dev
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# Rung 2 — Dev Set Stability
# Goal: verify training is stable on a small real subset (~10 shards) for 5 epochs.
# Expected: loss decreasing each epoch, no NaN/inf, checkpoints saved, epoch_time logged.
# Uses max_shards=10 to approximate the dev batch set (the manifest doesn't filter by batch ID).
# shuffle_buffer reduced to 500 to avoid memory pressure on small dataset.

mkdir -p logs

source .venv/bin/activate

python scripts/train_autoencoder.py \
    model=autoencoder \
    data=dev \
    training.epochs=5 \
    +training.max_shards=10 \
    +experiment_tag=dev-stability \
    "$@"
