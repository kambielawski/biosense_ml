"""Compute per-dimension temporal variance of AE latents and add to HDF5.

For each sequence, computes variance across timesteps for each of the 8192
latent dimensions. Averages variances across all sequences to produce a
global variance vector. Saves as `/temporal_variance` in the HDF5 file.

Usage:
    python scripts/compute_temporal_variance.py --latent_h5 data/rssm/latents_16x32.h5
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute temporal variance of AE latents")
    parser.add_argument("--latent_h5", type=str, required=True, help="Path to HDF5 latents file")
    args = parser.parse_args()

    h5_path = Path(args.latent_h5)
    logger.info("Loading %s", h5_path)

    with h5py.File(h5_path, "r") as f:
        latents = f["latents"][:]  # (N_total, ae_latent_dim)
        seq_starts = f["sequence_starts"][:]
        seq_lengths = f["sequence_lengths"][:]
        ae_latent_dim = int(f.attrs["ae_latent_dim"])

    num_sequences = len(seq_starts)
    logger.info("Dataset: %d samples, %d sequences, ae_latent_dim=%d",
                latents.shape[0], num_sequences, ae_latent_dim)

    # Compute per-sequence temporal variance, then average across sequences
    all_variances = []
    for i in range(num_sequences):
        start = int(seq_starts[i])
        length = int(seq_lengths[i])
        seq = latents[start:start + length]  # (T, ae_latent_dim)
        var = np.var(seq, axis=0)  # (ae_latent_dim,)
        all_variances.append(var)

    # Average across sequences
    temporal_variance = np.mean(all_variances, axis=0)  # (ae_latent_dim,)

    # Print statistics
    logger.info("Temporal variance statistics:")
    logger.info("  min:    %.6f", temporal_variance.min())
    logger.info("  max:    %.6f", temporal_variance.max())
    logger.info("  mean:   %.6f", temporal_variance.mean())
    logger.info("  median: %.6f", np.median(temporal_variance))
    logger.info("  std:    %.6f", temporal_variance.std())

    # Top-10 highest variance dimensions
    top_10_idx = np.argsort(temporal_variance)[-10:][::-1]
    logger.info("  Top-10 highest-variance dimensions:")
    for rank, idx in enumerate(top_10_idx, 1):
        logger.info("    #%d: dim %d = %.6f", rank, idx, temporal_variance[idx])

    # Bottom-10 lowest variance dimensions
    bot_10_idx = np.argsort(temporal_variance)[:10]
    logger.info("  Bottom-10 lowest-variance dimensions:")
    for rank, idx in enumerate(bot_10_idx, 1):
        logger.info("    #%d: dim %d = %.6f", rank, idx, temporal_variance[idx])

    # Distribution summary
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    logger.info("  Percentiles:")
    for p in percentiles:
        logger.info("    p%d: %.6f", p, np.percentile(temporal_variance, p))

    # Ratio of max to min (dynamic range)
    nonzero_min = temporal_variance[temporal_variance > 0].min() if (temporal_variance > 0).any() else 0
    if nonzero_min > 0:
        logger.info("  Dynamic range (max/min nonzero): %.1f", temporal_variance.max() / nonzero_min)

    # Save to HDF5
    with h5py.File(h5_path, "a") as f:
        if "temporal_variance" in f:
            del f["temporal_variance"]
        f.create_dataset("temporal_variance", data=temporal_variance.astype(np.float32))

    logger.info("Saved temporal_variance (%d,) to %s", ae_latent_dim, h5_path)


if __name__ == "__main__":
    main()
