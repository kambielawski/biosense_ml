#!/bin/bash
#SBATCH --job-name=encode-32x32x8
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/encode_32x32x8_%j.out
#SBATCH --error=logs/encode_32x32x8_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

# Use epoch 50 checkpoint (checkpoint_best.pt was deleted by known bug)
CKPT="outputs/bottleneck_32x32x8/checkpoints/checkpoint_epoch0050.pt"

echo "Encoding latents through 32x32x8 AE: $CKPT"

python scripts/encode_latents.py \
    --checkpoint "$CKPT" \
    --manifest data/processed/manifest.json \
    --output data/rssm/latents_32x32x8.h5 \
    --action_dim 3 \
    --batch_size 64 \
    --num_workers 4

echo "Encoding complete. Computing temporal variance..."

python scripts/compute_temporal_variance.py \
    --latent_h5 data/rssm/latents_32x32x8.h5

echo "All done."
