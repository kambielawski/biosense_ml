#!/bin/bash
# Convenience wrapper for submitting preprocessing jobs.
#
# Usage:
#   ./slurm/preprocess.sh                          # resize mode, all batches
#   ./slurm/preprocess.sh data=dev                  # resize mode, dev batches
#   ./slurm/preprocess.sh data.preprocessing.mode=autoencoder  # autoencoder mode
#
# After submission, monitor with:
#   tail -f logs/preprocess_<JOBID>.out

set -euo pipefail

# Detect preprocessing mode from args to pick the right partition
MODE="resize"
for arg in "$@"; do
    if [[ "$arg" == *"preprocessing.mode=autoencoder"* ]] || [[ "$arg" == *"preprocessing=autoencoder"* ]]; then
        MODE="autoencoder"
    fi
done

if [ "$MODE" = "autoencoder" ]; then
    PARTITION="gpu"
    EXTRA="--gres=gpu:1"
    echo "Autoencoder mode -> submitting to gpu partition"
else
    PARTITION="general"
    EXTRA=""
    echo "Resize mode -> submitting to general (CPU) partition"
fi

JOB_ID=$(sbatch --parsable --partition="$PARTITION" $EXTRA slurm/submit_preprocess.sh "$@")
echo "Submitted job $JOB_ID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/preprocess_${JOB_ID}.out"
echo "  squeue -j $JOB_ID"
