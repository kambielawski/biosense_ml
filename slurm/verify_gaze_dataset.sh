#!/bin/bash
#SBATCH --job-name=verify-gaze
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_gaze_%j.log
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
    --h5 data/gaze/gaze_dataset.h5 \
    --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
    --output_dir outputs/gaze_verification \
    --num_batches 3

echo "=== Done: $(date) ==="
echo ""
echo "To download results locally:"
echo "  scp -r ktbielaw@login.vacc.uvm.edu:~/projects/biosense_ml/outputs/gaze_verification ./outputs/"
