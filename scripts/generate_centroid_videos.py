#!/usr/bin/env python3
"""Generate verification videos and trajectory plots for centroid extraction.

Renders original video frames with detected centroid overlaid as a colored dot,
plus static trajectory plots. Used to visually verify that centroid extraction
is working correctly before committing to a full dataset build.

Usage:
    # Specific batches:
    python scripts/generate_centroid_videos.py \
        --archive_dir /archive \
        --batches 328 329 330

    # Random sample of N batches:
    python scripts/generate_centroid_videos.py \
        --archive_dir /archive \
        --sample 5

    # Include batches with stimulus:
    python scripts/generate_centroid_videos.py \
        --archive_dir /archive \
        --sample 5 --require_stim
"""

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from biosense_ml.pipeline.preprocessing import (
    discover_batch_dirs,
    discover_batch_files,
    load_commands,
    parse_image_timestamp,
)
from biosense_ml.pipeline.trajectory import (
    compute_timestamps_since_start,
    compute_velocity,
    crop_and_downsample,
    extract_centroid,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_C,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_DIMENSION,
    DEFAULT_MASK_RADIUS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def render_centroid_video(
    batch_id: int,
    batch_dir: Path,
    output_path: Path,
    dimension: int = DEFAULT_DIMENSION,
    block_size: int = DEFAULT_BLOCK_SIZE,
    C: int = DEFAULT_C,
    mask_radius: int = DEFAULT_MASK_RADIUS,
    crop_left: int = DEFAULT_CROP_LEFT,
    crop_right: int = DEFAULT_CROP_RIGHT,
    fps: int = 10,
) -> tuple[list[tuple[int, int] | None], list]:
    """Render a video with centroid dots overlaid on original frames.

    Detected centroids shown as green dots; interpolated as yellow dots.
    Frame number and timestamp shown as text overlay.

    Returns (raw_centroids, timestamps) for subsequent plot generation.
    """
    image_files = discover_batch_files(batch_dir)
    if not image_files:
        logger.warning("Batch %06d: no images", batch_id)
        return [], []

    # Extract centroids and timestamps
    raw_centroids: list[tuple[float, float] | None] = []
    timestamps = []

    for img_path in image_files:
        try:
            ts = parse_image_timestamp(img_path)
        except ValueError:
            continue
        centroid = extract_centroid(
            img_path, crop_left, crop_right, block_size, C, dimension, mask_radius
        )
        raw_centroids.append(centroid)
        timestamps.append(ts)

    if not raw_centroids:
        return [], []

    # Write video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (dimension, dimension))

    t0 = timestamps[0]
    trail_length = 30  # number of past centroids to show as trail

    for i, img_path in enumerate(image_files):
        if i >= len(raw_centroids):
            break

        frame = crop_and_downsample(img_path, crop_left, crop_right, dimension)
        elapsed = (timestamps[i] - t0).total_seconds()

        # Draw centroid trail
        start_trail = max(0, i - trail_length)
        for j in range(start_trail, i + 1):
            c = raw_centroids[j]
            if c is not None:
                # Cast to int for drawing (sub-pixel centroids)
                c_int = (int(round(c[0])), int(round(c[1])))
                # Current frame: larger green dot; trail: smaller blue dots
                if j == i:
                    cv2.circle(frame, c_int, 6, (0, 255, 0), -1)
                    cv2.circle(frame, c_int, 6, (0, 200, 0), 1)
                else:
                    alpha = 0.3 + 0.7 * (j - start_trail) / max(1, i - start_trail)
                    color = (int(255 * alpha), int(100 * alpha), 0)
                    cv2.circle(frame, c_int, 2, color, -1)
            elif j == i:
                # Missing detection — red X
                cx, cy = dimension // 2, dimension // 2
                cv2.putText(frame, "?", (cx - 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Text overlay
        cv2.putText(
            frame, f"#{i:04d}  t={elapsed:.1f}s",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

        out.write(frame)

    out.release()
    logger.info("Wrote video: %s (%d frames)", output_path, len(raw_centroids))
    return raw_centroids, timestamps


def generate_trajectory_plots(
    batch_id: int,
    raw_centroids: list[tuple[float, float] | None],
    timestamps: list,
    output_path: Path,
    dimension: int = DEFAULT_DIMENSION,
    commands: list[dict] | None = None,
):
    """Generate static trajectory plots: x(t), y(t), and x-y scatter."""
    # Filter to detected frames
    detected = [(i, c, t) for i, (c, t) in enumerate(zip(raw_centroids, timestamps)) if c is not None]
    if len(detected) < 2:
        logger.warning("Batch %06d: too few detections for plots", batch_id)
        return

    indices, centroids, times = zip(*detected)
    xy = np.array(centroids, dtype=np.float32) / dimension
    t0 = times[0]
    elapsed = np.array([(t - t0).total_seconds() for t in times])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Batch {batch_id:06d} — {len(detected)}/{len(raw_centroids)} frames detected")

    # x(t)
    axes[0].plot(elapsed, xy[:, 0], "b.-", markersize=2, linewidth=0.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("x (normalized)")
    axes[0].set_title("X position over time")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # y(t)
    axes[1].plot(elapsed, xy[:, 1], "r.-", markersize=2, linewidth=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("y (normalized)")
    axes[1].set_title("Y position over time")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # x-y scatter
    scatter = axes[2].scatter(xy[:, 0], xy[:, 1], c=elapsed, cmap="viridis", s=3)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("X-Y trajectory")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_aspect("equal")
    plt.colorbar(scatter, ax=axes[2], label="Time (s)")

    # Add stimulus shading if commands available
    if commands:
        from biosense_ml.pipeline.preprocessing import annotate_stimulus, _parse_command_time
        for cmd in commands:
            if cmd.get("type") == "electrical":
                start = _parse_command_time(cmd.get("start_time"))
                end = _parse_command_time(cmd.get("end_time"))
                if start and end:
                    s_sec = (start - t0).total_seconds()
                    e_sec = (end - t0).total_seconds()
                    for ax in axes[:2]:
                        ax.axvspan(s_sec, e_sec, alpha=0.2, color="orange", label="E-stim")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote plot: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate centroid verification videos and trajectory plots"
    )
    parser.add_argument(
        "--archive_dir", type=str, required=True,
        help="Root archive directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/verification",
        help="Output directory for videos and plots",
    )
    parser.add_argument(
        "--batches", type=int, nargs="*", default=None,
        help="Specific batch IDs to render",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N batches instead of specifying IDs",
    )
    parser.add_argument(
        "--require_stim", action="store_true",
        help="When sampling, ensure at least half have electrical stimulus",
    )
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--C", type=int, default=DEFAULT_C)
    parser.add_argument("--mask_radius", type=int, default=DEFAULT_MASK_RADIUS)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    output_dir = Path(args.output_dir)

    # Discover batches
    all_batch_dirs = discover_batch_dirs(archive_dir, batches=args.batches)

    if args.sample and not args.batches:
        rng = np.random.RandomState(42)

        if args.require_stim:
            # Split into stim/no-stim
            stim_batches = []
            no_stim_batches = []
            for bid, bdir in all_batch_dirs:
                cmds = load_commands(bdir)
                has_stim = any(c.get("type") == "electrical" for c in cmds)
                if has_stim:
                    stim_batches.append((bid, bdir))
                else:
                    no_stim_batches.append((bid, bdir))

            n_stim = min(len(stim_batches), args.sample // 2)
            n_no = min(len(no_stim_batches), args.sample - n_stim)
            stim_idx = rng.choice(len(stim_batches), n_stim, replace=False)
            no_idx = rng.choice(len(no_stim_batches), n_no, replace=False)
            selected = [stim_batches[i] for i in stim_idx] + [no_stim_batches[i] for i in no_idx]
        else:
            idx = rng.choice(len(all_batch_dirs), min(args.sample, len(all_batch_dirs)), replace=False)
            selected = [all_batch_dirs[i] for i in idx]

        all_batch_dirs = sorted(selected)

    logger.info("Processing %d batches", len(all_batch_dirs))

    for bid, bdir in all_batch_dirs:
        video_path = output_dir / "centroid_videos" / f"batch_{bid:06d}.mp4"
        plot_path = output_dir / "trajectory_plots" / f"batch_{bid:06d}.png"

        raw_centroids, timestamps = render_centroid_video(
            bid, bdir, video_path,
            dimension=args.dimension,
            block_size=args.block_size,
            C=args.C,
            mask_radius=args.mask_radius,
            fps=args.fps,
        )

        if raw_centroids:
            commands = load_commands(bdir)
            generate_trajectory_plots(
                bid, raw_centroids, timestamps, plot_path,
                dimension=args.dimension, commands=commands,
            )

    # Write a README
    readme_path = output_dir / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        "# Centroid Verification\n\n"
        "Generated by `scripts/generate_centroid_videos.py`.\n\n"
        "## Contents\n"
        "- `centroid_videos/` — MP4 videos with centroid dot overlaid on original frames\n"
        "  - Green dot = detected centroid, blue trail = recent positions\n"
        "  - Red '?' = no detection for that frame\n"
        "- `trajectory_plots/` — Static PNG plots showing x(t), y(t), and x-y scatter\n"
        "  - Orange shading indicates electrical stimulation periods\n\n"
        "## Regenerate\n"
        "```bash\n"
        "python scripts/generate_centroid_videos.py \\\n"
        "    --archive_dir /archive \\\n"
        "    --sample 10 --require_stim\n"
        "```\n"
    )
    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
