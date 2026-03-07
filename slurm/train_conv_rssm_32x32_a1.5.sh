#!/bin/bash
#SBATCH --job-name=crssm32-a1.5
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/crssm32_a1.5_%j.out
#SBATCH --error=logs/crssm32_a1.5_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

echo "Training ConvRSSM 32x32 SDW v2 with alpha=1.5, beta_kl=0.1"

python scripts/train_conv_rssm.py \
    model=conv_rssm_32x32 \
    training=rssm \
    +training.latent_h5=data/rssm/latents_32x32x8.h5 \
    +training.checkpoint_dir=outputs/conv_rssm_32x32_a1.5/checkpoints \
    +experiment_tag=conv-rssm-32x32-a1.5 \
    training.free_bits=0.1 \
    training.loss.beta_kl=0.1 \
    +training.spatial_delta_weighting=true \
    +training.sdw_alpha=1.5

echo "Training complete."
