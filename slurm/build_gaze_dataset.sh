#!/bin/bash
#SBATCH --job-name=build-gaze
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/build_gaze_%j.out
#SBATCH --error=logs/build_gaze_%j.err
#SBATCH --exclude=h2node14,h2node15

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "================"

python scripts/build_gaze_dataset.py \
    --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
    --output data/gaze/gaze_dataset.h5 \
    --num_workers 8

echo "=== Done: $(date) ==="
