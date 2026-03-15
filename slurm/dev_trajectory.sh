#!/bin/bash
#SBATCH --job-name=traj-dev
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/traj_dev_%j.out
#SBATCH --error=logs/traj_dev_%j.err

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "================"

echo "--- GRU dev training (10 epochs) ---"
python scripts/train_trajectory.py \
    model=trajectory_gru \
    training=trajectory \
    training.epochs=10 \
    training.batch_size=32 \
    training.num_workers=2 \
    +training.trajectory_h5=data/trajectory/trajectory_dataset.h5 \
    +training.checkpoint_dir=outputs/trajectory_gru_dev/checkpoints

echo ""
echo "--- MLP dev training (10 epochs) ---"
python scripts/train_trajectory.py \
    model=trajectory_mlp \
    training=trajectory \
    training.epochs=10 \
    training.batch_size=32 \
    training.num_workers=2 \
    +training.trajectory_h5=data/trajectory/trajectory_dataset.h5 \
    +training.checkpoint_dir=outputs/trajectory_mlp_dev/checkpoints

echo "=== Done: $(date) ==="
