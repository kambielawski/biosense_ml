#!/bin/bash
#SBATCH --job-name=traj-gru
#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/train_traj_gru_%j.out
#SBATCH --error=logs/train_traj_gru_%j.err

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "================"

python scripts/train_trajectory.py \
    model=trajectory_gru \
    training=trajectory \
    training.epochs=100 \
    training.num_workers=4 \
    +training.trajectory_h5=data/trajectory/trajectory_dataset.h5 \
    +training.checkpoint_dir=outputs/trajectory_gru/checkpoints

echo "=== Done: $(date) ==="
