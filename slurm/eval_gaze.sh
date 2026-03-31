#!/bin/bash
#SBATCH --job-name=eval_gaze
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/eval_gaze_%j.log
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

python scripts/eval_gaze.py \
    --checkpoint outputs/gaze_dynamics/checkpoints/checkpoint_best.pt \
    --h5 data/gaze/gaze_dataset.h5 \
    --output_dir outputs/gaze_dynamics/eval \
    --splits val test \
    --filmstrips 3 \
    --overlays 3 \
    --wandb_project biosense_ml

echo "=== Done: $(date) ==="
