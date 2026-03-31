#!/usr/bin/env python3
"""Verify gaze dataset Step 1 acceptance criteria.

Produces visual verification outputs:
1. For N batches: overlay crop bounding box on original 512x512 frames
   at multiple timepoints, saved as image grids.
2. Report has_motion statistics per batch.
3. Check that deltas cumsum ≈ centers.
4. Check crop value ranges (ImageNet-normalized).
5. (--video mode) Full-sequence MP4 videos with bounding box overlay.

Usage:
    # Static images only:
    python scripts/verify_gaze_dataset.py \
        --h5 data/gaze/gaze_dataset.h5 \
        --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
        --output_dir outputs/gaze_verification \
        --num_batches 3

    # Full videos:
    python scripts/verify_gaze_dataset.py \
        --h5 data/gaze/gaze_dataset.h5 \
        --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
        --output_dir outputs/gaze_verification \
        --num_batches 3 --video --fps 10
"""

import argparse
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np

from biosense_ml.pipeline.preprocessing import discover_batch_dirs, discover_batch_files
from biosense_ml.pipeline.trajectory import crop_and_downsample
from biosense_ml.pipeline.gaze import process_single_batch_gaze

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ImageNet stats for denormalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm_crop(crop_chw: np.ndarray) -> np.ndarray:
    """Convert (3,32,32) ImageNet-normalized crop back to uint8 RGB."""
    rgb = crop_chw.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def draw_crop_box(frame_bgr: np.ndarray, cy: float, cx: float, half: int = 16,
                  color=(0, 255, 0), thickness=1) -> np.ndarray:
    """Draw a rectangle on the frame around the crop region."""
    h, w = frame_bgr.shape[:2]
    y0 = int(np.clip(round(cy) - half, 0, h))
    y1 = int(np.clip(round(cy) + half, 0, h))
    x0 = int(np.clip(round(cx) - half, 0, w))
    x1 = int(np.clip(round(cx) + half, 0, w))
    out = frame_bgr.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    cv2.circle(out, (int(round(cx)), int(round(cy))), 2, (0, 0, 255), -1)
    return out


def render_overlay_frame(
    frame_bgr: np.ndarray,
    crop_chw: np.ndarray,
    cy: float,
    cx: float,
    has_motion_flag: bool,
    frame_idx: int,
    batch_label: str,
) -> np.ndarray:
    """Render a single annotated overlay frame (512x512 BGR)."""
    motion_str = "motion" if has_motion_flag else "static"
    color = (0, 255, 0) if has_motion_flag else (0, 128, 255)  # green=motion, orange=static

    overlay = draw_crop_box(frame_bgr, cy, cx, color=color)

    # Text: frame info top-left
    cv2.putText(overlay, f"{batch_label}  f={frame_idx}/{motion_str}  ({cy:.0f},{cx:.0f})",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Denormalize the stored crop and resize for preview in top-right
    crop_rgb = denorm_crop(crop_chw)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    crop_big = cv2.resize(crop_bgr, (128, 128), interpolation=cv2.INTER_NEAREST)

    overlay[5:133, 379:507] = crop_big
    cv2.rectangle(overlay, (379, 5), (507, 133), (255, 255, 0), 1)
    cv2.putText(overlay, "crop", (385, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return overlay


def verify_batch(grp, batch_key: str, archive_dir: Path, output_dir: Path,
                 video: bool = False, fps: int = 10):
    """Verify a single batch and save overlay images and/or video."""
    batch_id = grp.attrs["batch_id"]
    split = grp.attrs.get("split", "unknown")
    crops = grp["crops"][:]        # (T, 3, 32, 32)
    centers = grp["centers"][:]    # (T, 2)  (y, x)
    deltas = grp["deltas"][:]      # (T, 2)
    has_motion = grp["has_motion"][:]
    timestamps = grp["timestamps"][:]

    T = len(crops)
    n_motion = int(has_motion.sum())
    pct_motion = 100.0 * n_motion / T if T > 0 else 0

    logger.info("--- %s (batch %06d, split=%s) ---", batch_key, batch_id, split)
    logger.info("  Frames: %d, with motion: %d (%.1f%%)", T, n_motion, pct_motion)

    # --- Check 1: crop value ranges ---
    cmin, cmax = crops.min(), crops.max()
    logger.info("  Crop range: [%.3f, %.3f] (expected approx [-2.1, 2.6])", cmin, cmax)
    if cmin < -3.0 or cmax > 3.5:
        logger.warning("  ⚠ Crop values outside expected ImageNet range!")

    # --- Check 2: deltas cumsum ≈ centers ---
    reconstructed = centers[0] + np.cumsum(deltas, axis=0)
    max_drift = np.abs(reconstructed - centers).max()
    logger.info("  Delta cumsum max drift from centers: %.4f px (should be ~0)", max_drift)
    if max_drift > 1.0:
        logger.warning("  ⚠ Cumsum of deltas diverges from centers by %.2f px!", max_drift)

    # --- Check 3: has_motion[0] should be False ---
    if has_motion[0]:
        logger.warning("  ⚠ First frame has_motion=True (expected False)")
    else:
        logger.info("  First frame has_motion=False ✓")

    # --- Check 4: visual overlay ---
    batch_dir = archive_dir / f"batch-{batch_id:06d}"
    if not batch_dir.exists():
        logger.warning("  Archive dir not found: %s — skipping visual overlay", batch_dir)
        return

    image_files = discover_batch_files(batch_dir)
    if len(image_files) < T:
        logger.warning("  Archive has %d images but H5 has %d frames", len(image_files), T)

    batch_out = output_dir / batch_key
    batch_out.mkdir(parents=True, exist_ok=True)
    batch_label = f"batch {batch_id:06d}"

    if video:
        # --- Full-sequence video ---
        video_path = batch_out / f"{batch_key}_gaze_overlay.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (512, 512))

        for idx in range(min(T, len(image_files))):
            frame = crop_and_downsample(image_files[idx])
            overlay = render_overlay_frame(
                frame, crops[idx], centers[idx, 0], centers[idx, 1],
                bool(has_motion[idx]), idx, batch_label,
            )
            writer.write(overlay)

            if (idx + 1) % 200 == 0:
                logger.info("  Video: rendered %d/%d frames", idx + 1, T)

        writer.release()
        logger.info("  Saved video (%d frames, %d fps): %s", min(T, len(image_files)), fps, video_path)

    else:
        # --- Sampled overlay images ---
        motion_indices = np.where(has_motion)[0]
        if len(motion_indices) == 0:
            sample_indices = np.linspace(0, T - 1, min(8, T), dtype=int)
        else:
            n_samples = min(8, len(motion_indices))
            sample_indices = motion_indices[np.linspace(0, len(motion_indices) - 1, n_samples, dtype=int)]

        for idx in sample_indices:
            if idx >= len(image_files):
                continue

            frame = crop_and_downsample(image_files[idx])
            motion_str = "motion" if has_motion[idx] else "static"
            overlay = render_overlay_frame(
                frame, crops[idx], centers[idx, 0], centers[idx, 1],
                bool(has_motion[idx]), idx, batch_label,
            )
            out_path = batch_out / f"frame_{idx:04d}_{motion_str}.png"
            cv2.imwrite(str(out_path), overlay)

        logger.info("  Saved %d overlay images to %s", len(sample_indices), batch_out)


def verify_batch_live(batch_id: int, archive_dir: Path, output_dir: Path,
                      fps: int = 10):
    """Run gaze detection live on raw archive and render video directly."""
    batch_dir = archive_dir / f"batch-{batch_id:06d}"
    if not batch_dir.exists():
        logger.error("  Archive dir not found: %s", batch_dir)
        return

    batch_key = f"batch_{batch_id:06d}"
    batch_label = f"batch {batch_id:06d}"
    logger.info("--- %s (live from archive) ---", batch_key)

    # Run gaze detection from scratch
    result = process_single_batch_gaze(batch_id, batch_dir)
    if result is None:
        logger.error("  No usable frames in batch %06d", batch_id)
        return

    crops = result["crops"]
    centers = result["centers"]
    has_motion = result["has_motion"]
    T = len(crops)
    n_motion = int(has_motion.sum())
    logger.info("  Frames: %d, with motion: %d (%.1f%%)", T, n_motion,
                100.0 * n_motion / T if T > 0 else 0)

    # Load the image file list for rendering
    image_files = discover_batch_files(batch_dir)

    batch_out = output_dir / batch_key
    batch_out.mkdir(parents=True, exist_ok=True)
    video_path = batch_out / f"{batch_key}_gaze_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (512, 512))

    for idx in range(min(T, len(image_files))):
        frame = crop_and_downsample(image_files[idx])
        overlay = render_overlay_frame(
            frame, crops[idx], centers[idx, 0], centers[idx, 1],
            bool(has_motion[idx]), idx, batch_label,
        )
        writer.write(overlay)
        if (idx + 1) % 200 == 0:
            logger.info("  Video: rendered %d/%d frames", idx + 1, T)

    writer.release()
    logger.info("  Saved video (%d frames, %d fps): %s", min(T, len(image_files)), fps, video_path)


def main():
    parser = argparse.ArgumentParser(description="Verify gaze dataset acceptance criteria")
    parser.add_argument("--h5", default="data/gaze/gaze_dataset.h5")
    parser.add_argument("--archive_dir", default="/users/k/k/kkannans/scratch/biosense_training_data")
    parser.add_argument("--output_dir", default="outputs/gaze_verification")
    parser.add_argument("--num_batches", type=int, default=3,
                        help="Number of random batches to visually verify")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true",
                        help="Generate full-sequence MP4 videos instead of sampled images")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video frame rate (default 10)")
    parser.add_argument("--live", action="store_true",
                        help="Run gaze detection live from raw archive (skip H5)")
    parser.add_argument("--batch_ids", type=int, nargs="*", default=None,
                        help="Specific batch IDs to verify (used with --live)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = Path(args.archive_dir)

    # --- Live mode: run gaze detection from raw archive, no H5 needed ---
    if args.live:
        batch_ids = args.batch_ids or [121, 68, 257, 130]
        logger.info("=== Live gaze verification for batches: %s ===", batch_ids)
        for bid in batch_ids:
            verify_batch_live(bid, archive_dir, output_dir, fps=args.fps)
        logger.info("")
        logger.info("=== Verification complete ===")
        logger.info("Videos saved to: %s", output_dir)
        return

    with h5py.File(args.h5, "r") as f:
        logger.info("=== Gaze Dataset Verification ===")
        logger.info("H5 file: %s", args.h5)
        logger.info("Num batches in file: %d", len(f.keys()))
        logger.info("File attrs: %s", dict(f.attrs))
        logger.info("")

        # --- Global has_motion report ---
        logger.info("=== Per-batch has_motion summary ===")
        batch_keys = sorted(f.keys())
        motion_stats = []
        for bk in batch_keys:
            grp = f[bk]
            hm = grp["has_motion"][:]
            T = len(hm)
            n_m = int(hm.sum())
            split = grp.attrs.get("split", "?")
            motion_stats.append((bk, T, n_m, split))

        # Report batches with 0% motion (potential dead/stationary organoids)
        zero_motion = [(bk, T, n_m, sp) for bk, T, n_m, sp in motion_stats if n_m == 0]
        low_motion = [(bk, T, n_m, sp) for bk, T, n_m, sp in motion_stats if 0 < n_m < T * 0.05]

        logger.info("Batches with 0%% motion (dead/stationary): %d", len(zero_motion))
        for bk, T, n_m, sp in zero_motion:
            logger.info("  %s: %d frames, 0 motion, split=%s", bk, T, sp)

        logger.info("Batches with <5%% motion: %d", len(low_motion))
        for bk, T, n_m, sp in low_motion:
            logger.info("  %s: %d frames, %d motion (%.1f%%), split=%s",
                        bk, T, n_m, 100 * n_m / T, sp)

        high_motion = [(bk, T, n_m, sp) for bk, T, n_m, sp in motion_stats if n_m > T * 0.5]
        logger.info("Batches with >50%% motion: %d", len(high_motion))
        logger.info("")

        # --- Pick batches to verify visually ---
        # Choose from high-motion batches (known organoid activity)
        rng = np.random.RandomState(args.seed)
        if len(high_motion) >= args.num_batches:
            indices = rng.choice(len(high_motion), size=args.num_batches, replace=False)
            chosen_keys = [high_motion[i][0] for i in indices]
        else:
            chosen_keys = [bk for bk, _, _, _ in high_motion[:args.num_batches]]

        # Also include one zero/low-motion batch if available, for contrast
        if zero_motion:
            chosen_keys.append(zero_motion[0][0])
        elif low_motion:
            chosen_keys.append(low_motion[0][0])

        mode_str = "video" if args.video else "images"
        logger.info("=== Visual verification (%s) for %d batches ===", mode_str, len(chosen_keys))
        for bk in chosen_keys:
            verify_batch(f[bk], bk, archive_dir, output_dir,
                         video=args.video, fps=args.fps)

    logger.info("")
    logger.info("=== Verification complete ===")
    logger.info("Overlay images saved to: %s", output_dir)
    logger.info("Review them to confirm crop bounding boxes track the organoid.")


if __name__ == "__main__":
    main()
