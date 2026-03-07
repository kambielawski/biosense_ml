#!/bin/bash
#SBATCH --job-name=recon-32x32x8
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/recon_32x32x8_%j.out
#SBATCH --error=logs/recon_32x32x8_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

# Use epoch 40 checkpoint (best available — checkpoint_best.pt was deleted by known bug)
CKPT="outputs/bottleneck_32x32x8/checkpoints/checkpoint_epoch0040.pt"
SHARD="data/processed/batch-000068/shard-000000.tar"
OUTPUT="outputs/bottleneck_32x32x8/reconstruction_32x32x8.mp4"

python scripts/vis_scripts/make_reconstruction_video.py \
    --checkpoint "$CKPT" \
    --shard "$SHARD" \
    --output "$OUTPUT" \
    --num_frames 120 \
    --fps 10 \
    --num_encoder_blocks 4 \
    --latent_channels 8

echo "Reconstruction video complete: $OUTPUT"
