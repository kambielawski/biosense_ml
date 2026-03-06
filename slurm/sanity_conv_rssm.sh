#!/bin/bash
#SBATCH --job-name=sanity-conv-rssm
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_conv_rssm_%j.out
#SBATCH --error=logs/sanity_conv_rssm_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

echo "Sanity check: ConvRSSM overfit 20 epochs"

python scripts/train_conv_rssm.py \
    model=conv_rssm \
    training=rssm \
    +training.latent_h5=data/rssm/latents_16x32.h5 \
    +training.checkpoint_dir=outputs/conv_rssm_sanity/checkpoints \
    +experiment_tag=conv-rssm-sanity \
    training.epochs=20 \
    training.free_bits=0.1 \
    training.noise_std_init=0.0 \
    training.noise_std_final=0.0

echo "Sanity check complete."
