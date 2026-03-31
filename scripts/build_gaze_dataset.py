#!/usr/bin/env python3
"""Build gaze crop HDF5 dataset from raw archive images.

Runs motion detection on consecutive frames, extracts 32×32 crops centered on
the detected motion, and writes an HDF5 file for training the GazeDynamics model.

Must run on VACC where raw archive images are accessible.

Usage:
    python scripts/build_gaze_dataset.py \
        --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
        --output data/gaze/gaze_dataset.h5 \
        --num_workers 8
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
from biosense_ml.pipeline.gaze import process_single_batch_gaze
from biosense_ml.pipeline.trajectory import assign_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _process_batch_wrapper(kwargs: dict) -> dict | None:
    """Wrapper for ProcessPoolExecutor — unpacks kwargs."""
    return process_single_batch_gaze(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Build gaze crop HDF5 dataset from raw archive images"
    )
    parser.add_argument(
        "--archive_dir", type=str, required=True,
        help="Root archive directory containing batch-NNNNNN/ subdirs",
    )
    parser.add_argument(
        "--output", type=str, default="data/gaze/gaze_dataset.h5",
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
        "--dimension", type=int, default=512,
        help="Image downsampling dimension",
    )
    parser.add_argument(
        "--sigma", type=float, default=2.0,
        help="Gaussian blur sigma for motion detection",
    )
    parser.add_argument(
        "--threshold", type=float, default=10.0,
        help="Motion detection intensity threshold",
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
    # Step 1: Extract gaze crops from all batches (parallelizable)
    # -----------------------------------------------------------------------
    all_batch_results: list[dict] = []
    wall_start = time.monotonic()
    completed = 0
    total = len(batch_dirs)

    batch_kwargs = [
        {
            "batch_id": bid,
            "batch_dir": bdir,
            "dimension": args.dimension,
            "sigma": args.sigma,
            "threshold": args.threshold,
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
                result = future.result()
                if result is not None:
                    all_batch_results.append(result)
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

    if not all_batch_results:
        logger.error("No batches produced results — nothing to write")
        return

    logger.info("Processed %d batches successfully", len(all_batch_results))

    # -----------------------------------------------------------------------
    # Step 2: Train/val/test split (batch-level, stratified — same as trajectory)
    # -----------------------------------------------------------------------
    # Build a compatible structure for assign_splits
    batch_id_list = [r["batch_id"] for r in all_batch_results]
    split_labels = assign_splits(
        batch_id_list, all_batch_results, seed=args.split_seed
    )

    # -----------------------------------------------------------------------
    # Step 3: Write HDF5 — one group per batch
    # -----------------------------------------------------------------------
    preprocessing_params = {
        "dimension": args.dimension,
        "sigma": args.sigma,
        "threshold": args.threshold,
        "crop_size": 32,
    }

    total_frames = 0
    total_motion_frames = 0
    n_train = n_val = n_test = 0

    with h5py.File(output_path, "w") as f:
        for result, split in zip(all_batch_results, split_labels):
            bid = result["batch_id"]
            grp_name = f"batch_{bid:06d}"
            grp = f.create_group(grp_name)

            grp.create_dataset("crops", data=result["crops"], dtype="float32")
            grp.create_dataset("centers", data=result["centers"], dtype="float32")
            grp.create_dataset("deltas", data=result["deltas"], dtype="float32")
            grp.create_dataset("timestamps", data=result["timestamps"], dtype="float64")
            grp.create_dataset("has_motion", data=result["has_motion"], dtype="bool")
            grp.attrs["split"] = split
            grp.attrs["batch_id"] = bid

            T = result["crops"].shape[0]
            total_frames += T
            total_motion_frames += int(result["has_motion"].sum())

            if split == "train":
                n_train += 1
            elif split == "val":
                n_val += 1
            else:
                n_test += 1

        # File-level attributes
        f.attrs["num_batches"] = len(all_batch_results)
        f.attrs["total_frames"] = total_frames
        f.attrs["total_motion_frames"] = total_motion_frames
        f.attrs["preprocessing_params"] = json.dumps(preprocessing_params)
        f.attrs["split_seed"] = args.split_seed

    total_elapsed = time.monotonic() - wall_start
    mins, secs = divmod(total_elapsed, 60)

    logger.info(
        "Wrote %s: %d batches, %d total frames (%d with motion), "
        "splits: train=%d, val=%d, test=%d in %dm%02ds",
        output_path, len(all_batch_results), total_frames, total_motion_frames,
        n_train, n_val, n_test, int(mins), int(secs),
    )


if __name__ == "__main__":
    main()
