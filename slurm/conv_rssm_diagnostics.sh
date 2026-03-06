#!/bin/bash
#SBATCH --job-name=conv-rssm-diag
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/conv_rssm_diag_%j.out
#SBATCH --error=logs/conv_rssm_diag_%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/biosense_ml
source .venv/bin/activate

SEQ_IDX=${1:-139}

python scripts/diagnostics/conv_rssm_spatial_diagnostics.py \
    --rssm_checkpoint outputs/conv_rssm/checkpoints/checkpoint_best.pt \
    --latent_h5 data/rssm/latents_16x32.h5 \
    --sequence_idx "$SEQ_IDX" \
    --output_dir outputs/diagnostics/conv_rssm_spatial

echo "Diagnostics complete."
