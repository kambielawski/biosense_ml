#!/bin/bash
# Submit the full gaze pipeline as a chain of dependent jobs.
# Usage: bash slurm/submit_gaze_pipeline.sh
#
# Job chain:
#   1. build_gaze_dataset (CPU, ~1-2h)
#   2. train_gaze (GPU, ~2-8h depending on data size)
#   3. eval_gaze (GPU, ~5min)
#   4. generate_gaze_rollout (GPU, ~5min)

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Submitting Gaze Pipeline ==="
echo "Working dir: $(pwd)"
echo ""

# Step 1: Build dataset
JOB1=$(sbatch --parsable slurm/build_gaze_dataset.sh)
echo "Step 1 — build_gaze_dataset:    Job $JOB1"

# Step 2: Train (after dataset is built)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/train_gaze.sh)
echo "Step 2 — train_gaze:            Job $JOB2  (after $JOB1)"

# Step 3: Evaluate (after training)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/eval_gaze.sh)
echo "Step 3 — eval_gaze:             Job $JOB3  (after $JOB2)"

# Step 4: Generate rollout videos (after training, can run parallel with eval)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/generate_gaze_rollout.sh)
echo "Step 4 — generate_gaze_rollout: Job $JOB4  (after $JOB2)"

echo ""
echo "=== Pipeline Submitted ==="
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel $JOB1 $JOB2 $JOB3 $JOB4"
