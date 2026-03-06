#!/bin/bash
#SBATCH --job-name=rssm-rollout-vid
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/rssm_rollout_video_%j.out
#SBATCH --error=logs/rssm_rollout_video_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

RSSM_CKPT=${1:-"outputs/rssm/checkpoints/checkpoint_best.pt"}
AE_CKPT=${2:-"outputs/bottleneck_16x32/checkpoints/checkpoint_best.pt"}
LATENT_H5=${3:-"data/rssm/latents_16x32.h5"}
OUTPUT=${4:-"outputs/rssm/rssm_rollout_baseline.mp4"}

echo "Generating RSSM rollout video"
echo "  RSSM: $RSSM_CKPT"
echo "  AE:   $AE_CKPT"
echo "  H5:   $LATENT_H5"
echo "  Out:  $OUTPUT"

python scripts/vis_scripts/make_rssm_rollout_video.py \
    --rssm_checkpoint "$RSSM_CKPT" \
    --ae_checkpoint "$AE_CKPT" \
    --latent_h5 "$LATENT_H5" \
    --output "$OUTPUT" \
    --context_len 16 \
    --rollout_len 48 \
    --fps 10

echo "Done. Video at: $OUTPUT"
