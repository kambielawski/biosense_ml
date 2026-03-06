#!/bin/bash
#SBATCH --job-name=train-conv-rssm
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_conv_rssm_%j.out
#SBATCH --error=logs/train_conv_rssm_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

LATENT_H5=${1:-"data/rssm/latents_16x32.h5"}

echo "Training ConvRSSM with latents: $LATENT_H5"

python scripts/train_conv_rssm.py \
    model=conv_rssm \
    training=rssm \
    +training.latent_h5="$LATENT_H5" \
    +training.checkpoint_dir=outputs/conv_rssm/checkpoints \
    +experiment_tag=conv-rssm-v1 \
    training.free_bits=0.1

echo "Training complete."
