#!/usr/bin/env python3
"""Build trajectory HDF5 dataset from raw archive images.

Extracts organoid centroids from all batches, computes velocity and stimulus
features, and writes a single HDF5 file for autoregressive trajectory modeling.

Must run on VACC where raw archive images are accessible.

Usage:
    python scripts/build_trajectory_dataset.py \
        --archive_dir /gpfs2/scratch/ks} \
        --output data/trajectory_dataset.h5 \
        --num_workers 8

    # Process specific batches:
    python scripts/build_trajectory_dataset.py \
        --archive_dir /archive \
        --output data/trajectory_dataset.h5 \
        --batches 328 329 330
"""

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from biosense_ml.pipeline.preprocessing import discover_batch_dirs
from biosense_ml.pipeline.trajectory import (
    assign_splits,
    compute_timestamps_since_start,
    compute_velocity,
    encode_trajectory_actions,
    process_single_batch,
    scan_max_current_ma,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _process_batch_wrapper(kwargs: dict) -> list[dict] | None:
    """Wrapper for ProcessPoolExecutor — unpacks kwargs for process_single_batch."""
    return process_single_batch(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Build trajectory HDF5 dataset from raw archive images"
    )
    parser.add_argument(
        "--archive_dir", type=str, required=True,
        help="Root archive directory containing batch-NNNNNN/ subdirs",
    )
    parser.add_argument(
        "--output", type=str, default="data/trajectory_dataset.h5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--batches", type=int, nargs="*", default=None,
        help="Specific batch IDs to process (default: all)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel workers for batch processing",
    )
    parser.add_argument(
        "--max_gap_frames", type=int, default=2,
        help="Max consecutive missing frames to interpolate (longer gaps split sequences)",
    )
    parser.add_argument(
        "--dimension", type=int, default=512,
        help="Image downsampling dimension",
    )
    parser.add_argument(
        "--block_size", type=int, default=17,
        help="Adaptive threshold block size",
    )
    parser.add_argument("--C", type=int, default=29, help="Adaptive threshold constant")
    parser.add_argument(
        "--mask_radius", type=int, default=240,
        help="Circular mask radius (at default 512px dimension)",
    )
    parser.add_argument(
        "--split_seed", type=int, default=42,
        help="Random seed for train/val/test splitting",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover batches
    batch_dirs = discover_batch_dirs(archive_dir, batches=args.batches)
    if not batch_dirs:
        logger.error("No batch directories found in %s", archive_dir)
        return

    logger.info("Found %d batches to process", len(batch_dirs))

    # -----------------------------------------------------------------------
    # Step 1: Extract centroids from all batches (parallelizable)
    # -----------------------------------------------------------------------
    all_sequences: list[dict] = []
    wall_start = time.monotonic()
    completed = 0
    total = len(batch_dirs)

    # Build kwargs for each batch
    batch_kwargs = [
        {
            "batch_id": bid,
            "batch_dir": bdir,
            "max_gap_frames": args.max_gap_frames,
            "dimension": args.dimension,
            "block_size": args.block_size,
            "C": args.C,
            "mask_radius": args.mask_radius,
        }
        for bid, bdir in batch_dirs
    ]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(_process_batch_wrapper, kw): kw["batch_id"]
            for kw in batch_kwargs
        }
        for future in as_completed(futures):
            bid = futures[future]
            completed += 1
            try:
                sequences = future.result()
                if sequences:
                    all_sequences.extend(sequences)
            except Exception:
                logger.exception("Failed to process batch %06d", bid)

            elapsed = time.monotonic() - wall_start
            avg = elapsed / completed
            eta = avg * (total - completed)
            eta_m, eta_s = divmod(eta, 60)
            logger.info(
                "[%3d/%d] Batch %06d done | ETA %dm%02ds",
                completed, total, bid, int(eta_m), int(eta_s),
            )

    if not all_sequences:
        logger.error("No sequences extracted — nothing to write")
        return

    logger.info("Extracted %d sequences total", len(all_sequences))

    # -----------------------------------------------------------------------
    # Step 2: Two-pass normalization — find max current_ma
    # -----------------------------------------------------------------------
    max_current_ma = scan_max_current_ma(all_sequences)
    logger.info("Max current_ma across dataset: %.2f", max_current_ma)

    # -----------------------------------------------------------------------
    # Step 3: Compute derived features and flatten into arrays
    # -----------------------------------------------------------------------
    all_centroid_xy = []
    all_velocity_xy = []
    all_dt = []
    all_actions = []
    all_valid_mask = []
    all_batch_ids = []
    all_frame_indices = []
    all_timestamps = []
    sequence_starts = []
    sequence_lengths = []

    offset = 0
    for seq in all_sequences:
        T = seq["centroid_xy"].shape[0]
        velocity_xy, dt = compute_velocity(seq["centroid_xy"], seq["timestamps"])
        actions = encode_trajectory_actions(
            seq["stimulus_annotations"], dt, max_current_ma
        )
        ts_since_start = compute_timestamps_since_start(seq["timestamps"])

        all_centroid_xy.append(seq["centroid_xy"])
        all_velocity_xy.append(velocity_xy)
        all_dt.append(dt)
        all_actions.append(actions)
        all_valid_mask.append(seq["valid_mask"])
        all_batch_ids.append(np.full(T, seq["batch_id"], dtype=np.int32))
        all_frame_indices.append(seq["frame_indices"])
        all_timestamps.append(ts_since_start)

        sequence_starts.append(offset)
        sequence_lengths.append(T)
        offset += T

    # Stack everything
    centroid_xy = np.concatenate(all_centroid_xy)
    velocity_xy = np.concatenate(all_velocity_xy)
    dt = np.concatenate(all_dt)
    actions = np.concatenate(all_actions)
    valid_mask = np.concatenate(all_valid_mask)
    batch_ids = np.concatenate(all_batch_ids)
    frame_indices = np.concatenate(all_frame_indices)
    timestamps = np.concatenate(all_timestamps)
    seq_starts = np.array(sequence_starts, dtype=np.int64)
    seq_lengths = np.array(sequence_lengths, dtype=np.int32)

    # -----------------------------------------------------------------------
    # Step 4: Train/val/test split (batch-level, stratified)
    # -----------------------------------------------------------------------
    batch_id_list = [seq["batch_id"] for seq in all_sequences]
    split_labels = assign_splits(
        batch_id_list, all_sequences, seed=args.split_seed
    )

    # Compute normalization stats
    norm_stats = {
        "centroid_xy_mean": centroid_xy.mean(axis=0).tolist(),
        "centroid_xy_std": centroid_xy.std(axis=0).tolist(),
        "velocity_xy_mean": velocity_xy.mean(axis=0).tolist(),
        "velocity_xy_std": velocity_xy.std(axis=0).tolist(),
        "dt_mean": float(dt[dt > 0].mean()) if (dt > 0).any() else 0.0,
        "dt_std": float(dt[dt > 0].std()) if (dt > 0).any() else 1.0,
        "max_current_ma": max_current_ma,
    }

    preprocessing_params = {
        "dimension": args.dimension,
        "block_size": args.block_size,
        "C": args.C,
        "mask_radius": args.mask_radius,
        "max_gap_frames": args.max_gap_frames,
        "crop_left": 1358,
        "crop_right": 850,
    }

    # -----------------------------------------------------------------------
    # Step 5: Write HDF5
    # -----------------------------------------------------------------------
    split_arr = np.array(split_labels, dtype="S5")

    # Count splits
    n_train = sum(1 for s in split_labels if s == "train")
    n_val = sum(1 for s in split_labels if s == "val")
    n_test = sum(1 for s in split_labels if s == "test")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("centroid_xy", data=centroid_xy, dtype="float32")
        f.create_dataset("velocity_xy", data=velocity_xy, dtype="float32")
        f.create_dataset("dt", data=dt, dtype="float32")
        f.create_dataset("actions", data=actions, dtype="float32")
        f.create_dataset("valid_mask", data=valid_mask, dtype="uint8")
        f.create_dataset("batch_ids", data=batch_ids, dtype="int32")
        f.create_dataset("frame_indices", data=frame_indices, dtype="int32")
        f.create_dataset("timestamps", data=timestamps, dtype="float32")
        f.create_dataset("sequence_starts", data=seq_starts, dtype="int64")
        f.create_dataset("sequence_lengths", data=seq_lengths, dtype="int32")
        f.create_dataset("split_assignments", data=split_arr)

        f.attrs["centroid_dim"] = 2
        f.attrs["action_dim"] = 5
        f.attrs["num_sequences"] = len(sequence_starts)
        f.attrs["num_samples"] = offset
        f.attrs["norm_stats"] = json.dumps(norm_stats)
        f.attrs["preprocessing_params"] = json.dumps(preprocessing_params)
        f.attrs["split_seed"] = args.split_seed

    total_elapsed = time.monotonic() - wall_start
    mins, secs = divmod(total_elapsed, 60)

    logger.info(
        "Wrote %s: %d samples, %d sequences (train=%d, val=%d, test=%d) in %dm%02ds",
        output_path, offset, len(sequence_starts), n_train, n_val, n_test,
        int(mins), int(secs),
    )
    logger.info("Normalization stats: %s", json.dumps(norm_stats, indent=2))


if __name__ == "__main__":
    main()
