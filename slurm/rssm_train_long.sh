#!/bin/bash
#SBATCH --job-name=rssm-long
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rssm_long_%j.out
#SBATCH --error=logs/rssm_long_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

FREE_BITS=${1:-0.5}
EPOCHS=${2:-500}
LATENT_H5=${3:-"data/rssm/latents_16x32.h5"}
TAG="rssm-fb${FREE_BITS}-ep${EPOCHS}"
CKPT_DIR="outputs/rssm_fb${FREE_BITS}/checkpoints"

echo "Training RSSM: free_bits=$FREE_BITS, epochs=$EPOCHS, tag=$TAG"
echo "  Checkpoint dir: $CKPT_DIR"

python scripts/train_rssm.py \
    model=rssm \
    training=rssm \
    +training.latent_h5="$LATENT_H5" \
    +training.checkpoint_dir="$CKPT_DIR" \
    +experiment_tag="$TAG" \
    training.free_bits="$FREE_BITS" \
    training.epochs="$EPOCHS"

echo "Training complete."
