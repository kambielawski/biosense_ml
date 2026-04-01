#!/bin/bash
#SBATCH --job-name=gaze_rollout
#SBATCH --partition=goldenmaple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gaze_rollout_%j.log
#SBATCH --exclude=h2node14,h2node15

set -euo pipefail
mkdir -p logs
cd ~/projects/biosense_ml
source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start:     $(date)"
echo "================"

python scripts/generate_gaze_rollout.py \
    --checkpoint outputs/gaze_dynamics/checkpoints/checkpoint_best.pt \
    --h5 data/gaze/gaze_dataset.h5 \
    --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
    --output_dir outputs/gaze_dynamics/rollout_videos \
    --split test \
    --num_videos 3 \
    --fps 5 \
    --side_by_side \
    --wandb_project biosense_ml

echo "=== Done: $(date) ==="
