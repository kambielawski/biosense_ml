#!/bin/bash
#SBATCH --job-name=traj-sanity
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/traj_sanity_%j.out
#SBATCH --error=logs/traj_sanity_%j.err

set -euo pipefail
mkdir -p logs

source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "================"

echo "--- GRU sanity check ---"
python scripts/sanity_check_trajectory.py --model gru --epochs 100

echo ""
echo "--- MLP sanity check ---"
python scripts/sanity_check_trajectory.py --model mlp --epochs 100

echo "=== Done: $(date) ==="
