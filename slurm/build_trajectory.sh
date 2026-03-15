#!/bin/bash
#SBATCH --job-name=build-trajectory
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/build_trajectory_%j.out
#SBATCH --error=logs/build_trajectory_%j.err

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "================"

python scripts/build_trajectory_dataset.py \
    --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
    --output data/trajectory/trajectory_dataset.h5 \
    --num_workers 8

echo "=== Done: $(date) ==="
