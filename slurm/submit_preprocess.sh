#!/bin/bash
#SBATCH --job-name=biosense-preprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-15  # 16 parallel workers; adjust based on dataset size
#SBATCH --output=logs/%j_%A_%a.out
#SBATCH --error=logs/%j_%A_%a.err

# Load modules (customize for your cluster)
# module load cuda/12.x python/3.10

mkdir -p logs

source .venv/bin/activate

python scripts/preprocess.py \
    data.raw_data_dir="/path/to/biosense/prod/archive" \
    data.processed_data_dir="/scratch/$USER/processed" \
    "$@"
