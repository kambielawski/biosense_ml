#!/bin/bash
#SBATCH --job-name=gaze-dyn
#SBATCH --partition=goldenmaple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_gaze_%j.out
#SBATCH --error=logs/train_gaze_%j.err
#SBATCH --exclude=h2node14,h2node15

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "================"

python scripts/train_gaze.py \
    model=gaze_dynamics \
    training=gaze \
    training.epochs=100 \
    training.num_workers=4 \
    +training.gaze_h5=data/gaze/gaze_dataset.h5 \
    +training.checkpoint_dir=outputs/gaze_dynamics/checkpoints

echo "=== Done: $(date) ==="
