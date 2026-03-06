#!/bin/bash
#SBATCH --job-name=train-rssm
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_rssm_%j.out
#SBATCH --error=logs/train_rssm_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

# Latent H5 path (relative to project root)
LATENT_H5=${1:-"data/rssm/latents.h5"}

echo "Training RSSM with latents: $LATENT_H5"

python scripts/train_rssm.py \
    model=rssm \
    training=rssm \
    +training.latent_h5="$LATENT_H5" \
    +training.checkpoint_dir=outputs/rssm/checkpoints \
    +experiment_tag=rssm-phase1

echo "Training complete."
