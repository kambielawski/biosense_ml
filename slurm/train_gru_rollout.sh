#!/bin/bash
#SBATCH --job-name=gru-rollout
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/train_gru_rollout_%j.out
#SBATCH --error=logs/train_gru_rollout_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "================"

python scripts/train_trajectory.py \
    model=trajectory_gru \
    training=trajectory_rollout \
    +training.trajectory_h5=data/trajectory/trajectory_dataset.h5 \
    +training.checkpoint_dir=outputs/trajectory_gru_rollout/checkpoints

echo "=== Done: $(date) ==="
