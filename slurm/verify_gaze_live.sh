#!/bin/bash
#SBATCH --job-name=gaze-verify-live
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_gaze_live_%j.log
#SBATCH --exclude=h2node14,h2node15

set -euo pipefail
mkdir -p logs
cd ~/projects/biosense_ml
source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "================"

python scripts/verify_gaze_dataset.py \
    --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
    --output_dir outputs/gaze_verification_live \
    --live \
    --batch_ids 121 68 257 130 \
    --fps 10

echo "=== Done: $(date) ==="
