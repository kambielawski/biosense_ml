#!/bin/bash
#SBATCH --job-name=rssm-traj-dist
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/rssm_traj_dist_%j.out
#SBATCH --error=logs/rssm_traj_dist_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

RSSM_CKPT=${1:-"outputs/rssm/checkpoints/checkpoint_best.pt"}
AE_CKPT=${2:-"outputs/bottleneck_16x32/checkpoints/checkpoint_best.pt"}
LATENT_H5=${3:-"data/rssm/latents_16x32.h5"}
OUTPUT=${4:-"outputs/rssm/trajectory_distribution.mp4"}

echo "Generating trajectory distribution video"

python scripts/vis_scripts/make_trajectory_distribution_video.py \
    --rssm_checkpoint "$RSSM_CKPT" \
    --ae_checkpoint "$AE_CKPT" \
    --latent_h5 "$LATENT_H5" \
    --output "$OUTPUT" \
    --num_rollouts 30 \
    --context_len 16 \
    --rollout_len 48 \
    --fps 10

echo "Done. Video at: $OUTPUT"
