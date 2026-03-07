#!/bin/bash
#SBATCH --job-name=crssm32-vis
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/crssm32_vis_%j.out
#SBATCH --error=logs/crssm32_vis_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

RUN_DIR=${1:?"Usage: sbatch $0 <run_dir> (e.g. outputs/conv_rssm_32x32_bkl05)"}
RSSM_CKPT="${RUN_DIR}/checkpoints/checkpoint_best.pt"
AE_CKPT="outputs/bottleneck_32x32x8/checkpoints/checkpoint_best.pt"
LATENT_H5="data/rssm/latents_32x32x8.h5"
OUTPUT_DIR="${RUN_DIR}/videos"
mkdir -p "$OUTPUT_DIR"

for SEQ_IDX in 139 126 49 45 46; do
    echo "Generating rollout video for sequence $SEQ_IDX..."
    python scripts/vis_scripts/make_conv_rssm_rollout_video.py \
        --rssm_checkpoint "$RSSM_CKPT" \
        --ae_checkpoint "$AE_CKPT" \
        --latent_h5 "$LATENT_H5" \
        --output "$OUTPUT_DIR/rollout_seq${SEQ_IDX}.mp4" \
        --context_len 16 \
        --rollout_len 48 \
        --sequence_idx "$SEQ_IDX" \
        --fps 10
done

echo "All rollout videos complete."
ls -lh "$OUTPUT_DIR"
