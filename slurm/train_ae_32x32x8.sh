#!/bin/bash
#SBATCH --job-name=ae-32x32x8
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ae_32x32x8_%j.out
#SBATCH --error=logs/ae_32x32x8_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

echo "Training 32x32x8 autoencoder (num_encoder_blocks=4, latent_channels=8)"

python scripts/train_autoencoder.py \
    model=autoencoder_32x32x8 \
    +training.checkpoint_dir=outputs/bottleneck_32x32x8/checkpoints \
    +experiment_tag=ae-32x32x8

echo "Training complete."
