#!/bin/bash
#SBATCH --job-name=conv-rssm-vis
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/conv_rssm_vis_%j.out
#SBATCH --error=logs/conv_rssm_vis_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

python scripts/vis_scripts/make_conv_rssm_rollout_video.py \
    --rssm_checkpoint outputs/conv_rssm/checkpoints/checkpoint_best.pt \
    --ae_checkpoint outputs/bottleneck_16x32/checkpoints/checkpoint_best.pt \
    --latent_h5 data/rssm/latents_16x32.h5 \
    --output conv_rssm_rollout.mp4 \
    --context_len 16 \
    --rollout_len 48 \
    --fps 10

echo "Visualization complete."
