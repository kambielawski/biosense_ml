#!/bin/bash
#SBATCH --job-name=biosense-sanity
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# Rung 1 — Sanity Check
# Goal: verify the model can overfit a single shard (~1000 samples) in 50 epochs.
# Expected: val_loss starts ~0.5-1.0 and drops significantly (ideally <0.1), indicating
# the model and data pipeline are fundamentally working.
# Runtime: should complete in <15 minutes.

mkdir -p logs

source .venv/bin/activate

python scripts/train_autoencoder.py \
    model=autoencoder \
    training.epochs=50 \
    data.batch_size=8 \
    data.num_workers=0 \
    data.shuffle_buffer=100 \
    +training.max_shards=1 \
    +experiment_tag=sanity-check \
    "$@"
