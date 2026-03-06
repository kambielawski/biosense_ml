#!/bin/bash
#SBATCH --job-name=encode-latents
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/encode_latents_%j.out
#SBATCH --error=logs/encode_latents_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

# Arguments: pass checkpoint path and output path
CHECKPOINT=${1:-"outputs/bottleneck_16x32/checkpoints/checkpoint_best.pt"}
OUTPUT=${2:-"data/rssm/latents.h5"}

echo "Encoding latents with checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"

python scripts/encode_latents.py \
    --checkpoint "$CHECKPOINT" \
    --manifest data/processed/manifest.json \
    --output "$OUTPUT" \
    --action_dim 3 \
    --batch_size 64 \
    --num_workers 8

echo "Encoding complete."
