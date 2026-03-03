#!/bin/bash
#SBATCH --job-name=biosense-preprocess
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Partition is set automatically based on preprocessing mode:
#   resize      -> general (CPU-only, no GPU needed)
#   autoencoder -> gpu     (needs GPU for encoding)
# Override via: sbatch --partition=<name> slurm/submit_preprocess.sh

set -euo pipefail

mkdir -p logs

source .venv/bin/activate

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "Args:      $@"
echo "================"

python scripts/preprocess.py "$@"

echo "=== Done: $(date) ==="
