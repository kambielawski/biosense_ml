#!/usr/bin/env python3
"""Generate trajectory rollout overlay videos.

Overlays predicted and ground-truth trajectories on raw video frames:
- Red trail: burn-in (ground truth fed to model)
- Green trail: ground truth for rollout period
- Blue trail: model's autoregressive prediction

Usage:
    python scripts/vis_trajectory_rollout.py \
        --h5 data/trajectory/trajectory_dataset.h5 \
        --checkpoint outputs/trajectory_mlp/checkpoints/checkpoint_epoch0080.pt \
        --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
        --output_dir outputs/rollout_videos \
        --seq_indices 14 132 169 100 187
"""

import argparse
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from biosense_ml.models.trajectory_mlp import TrajectoryMLP
from biosense_ml.pipeline.preprocessing import discover_batch_files
from biosense_ml.pipeline.trajectory import crop_and_downsample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Colors (BGR for OpenCV)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 100, 0)
WHITE = (255, 255, 255)

DIMENSION = 512  # image dimension used during centroid extraction


def load_model(checkpoint_path: str, device: torch.device) -> TrajectoryMLP:
    """Load trained MLP model from checkpoint."""
    cfg = OmegaConf.create({
        "name": "trajectory_mlp", "input_dim": 10, "hidden_dim": 64,
        "output_dim": 2, "context_len": 10, "num_layers": 2, "dropout": 0.0,
    })
    model = TrajectoryMLP(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded MLP checkpoint: epoch=%d", ckpt["epoch"])
    return model


def run_rollout(model, features, burn_in_frac=0.2, context_len=10):
    """Run autoregressive rollout on a sequence.

    Returns:
        burn_in_xy: (B, 2) ground truth positions during burn-in
        gt_xy: (H, 2) ground truth positions during rollout
        pred_xy: (H, 2) predicted positions during rollout
        burn_in_end: index where burn-in ends
    """
    device = next(model.parameters()).device
    T = features.shape[0]
    B_idx = max(int(T * burn_in_frac), context_len)
    H = T - B_idx - 1

    if H < 1:
        return None, None, None, 0

    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, 10)

    with torch.no_grad():
        context = feat_t[:, B_idx - context_len : B_idx]  # (1, K, 10)
        future_actions = feat_t[:, B_idx : B_idx + H, 5:10]  # (1, H, 5)
        future_dt = feat_t[:, B_idx + 1 : B_idx + 1 + H, 4]  # (1, H) — dt at each future step
        predictions = model.rollout(context, future_actions, future_dt)

    burn_in_xy = features[:B_idx, :2]  # (B, 2)
    gt_xy = features[B_idx : B_idx + H, :2]  # (H, 2)
    pred_xy = predictions.squeeze(0).cpu().numpy()  # (H, 2)

    # Clamp predictions to [0, 1] to avoid drawing outside frame
    pred_xy = np.clip(pred_xy, 0, 1)

    return burn_in_xy, gt_xy, pred_xy, B_idx


def draw_trail(frame, points_norm, color, radius=4, alpha=0.6, max_trail=50):
    """Draw a trail of dots on the frame with fading opacity."""
    overlay = frame.copy()
    n = len(points_norm)
    start = max(0, n - max_trail)
    for i in range(start, n):
        px = int(points_norm[i, 0] * DIMENSION)
        py = int(points_norm[i, 1] * DIMENSION)
        # Older points are more transparent
        t = (i - start) / max(n - start - 1, 1)
        r = max(2, int(radius * (0.5 + 0.5 * t)))
        cv2.circle(overlay, (px, py), r, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_dot(frame, xy_norm, color, radius=6):
    """Draw a single dot at the current position."""
    px = int(xy_norm[0] * DIMENSION)
    py = int(xy_norm[1] * DIMENSION)
    cv2.circle(frame, (px, py), radius, color, -1)
    cv2.circle(frame, (px, py), radius, WHITE, 1)
    return frame


def generate_video(
    seq_idx, h5_data, model, archive_dir, output_dir, burn_in_frac=0.2,
):
    """Generate a single trajectory rollout overlay video."""
    seq_start = int(h5_data["seq_starts"][seq_idx])
    seq_len = int(h5_data["seq_lengths"][seq_idx])
    batch_id = int(h5_data["batch_ids"][seq_start])
    frame_indices = h5_data["frame_indices"][seq_start : seq_start + seq_len]

    # Build feature vector for this sequence
    features = np.concatenate([
        h5_data["centroid_xy"][seq_start : seq_start + seq_len],
        h5_data["velocity_xy"][seq_start : seq_start + seq_len],
        h5_data["dt"][seq_start : seq_start + seq_len, None],
        h5_data["actions"][seq_start : seq_start + seq_len],
    ], axis=1).astype(np.float32)

    stim_frac = float(h5_data["actions"][seq_start:seq_start+seq_len, 0].sum()) / seq_len

    logger.info(
        "Seq %d: batch=%d, length=%d, stim=%.1f%%, frames=%d-%d",
        seq_idx, batch_id, seq_len, stim_frac * 100,
        frame_indices[0], frame_indices[-1],
    )

    # Run rollout
    burn_in_xy, gt_xy, pred_xy, B_idx = run_rollout(
        model, features, burn_in_frac=burn_in_frac,
    )
    if burn_in_xy is None:
        logger.warning("Seq %d: too short for rollout", seq_idx)
        return

    # Load raw images for this batch
    batch_dir = Path(archive_dir) / f"batch-{batch_id:06d}"
    image_files = discover_batch_files(batch_dir)
    if not image_files:
        logger.error("No images found for batch %d at %s", batch_id, batch_dir)
        return

    # Set up video writer
    output_path = Path(output_dir) / f"rollout_seq{seq_idx}_batch{batch_id}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10  # speed up from ~0.2fps real-time
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (DIMENSION, DIMENSION))

    # Subsample frames for reasonable video length (every 2nd frame)
    stride = max(1, seq_len // 500)

    for t in range(0, seq_len, stride):
        fi = int(frame_indices[t])
        if fi >= len(image_files):
            continue

        # Load and crop raw image
        frame = crop_and_downsample(str(image_files[fi]))

        # Determine which phase we're in and draw accordingly
        if t < B_idx:
            # Burn-in phase: draw red trail up to current frame
            trail_points = burn_in_xy[: t + 1]
            frame = draw_trail(frame, trail_points, RED, max_trail=80)
            frame = draw_dot(frame, burn_in_xy[t], RED, radius=7)
            phase_text = "BURN-IN (ground truth)"
            phase_color = RED
        else:
            # Rollout phase: draw all trails
            # Red: full burn-in trail
            frame = draw_trail(frame, burn_in_xy, RED, alpha=0.3, max_trail=200)

            rollout_t = t - B_idx
            if rollout_t < len(gt_xy):
                # Green: ground truth trail
                gt_trail = gt_xy[: rollout_t + 1]
                frame = draw_trail(frame, gt_trail, GREEN, max_trail=100)
                frame = draw_dot(frame, gt_xy[rollout_t], GREEN, radius=7)

            if rollout_t < len(pred_xy):
                # Blue: prediction trail
                pred_trail = pred_xy[: rollout_t + 1]
                frame = draw_trail(frame, pred_trail, BLUE, max_trail=100)
                frame = draw_dot(frame, pred_xy[rollout_t], BLUE, radius=7)

            phase_text = "ROLLOUT (green=GT, blue=pred)"
            phase_color = GREEN

        # Add text overlay
        cv2.putText(frame, phase_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)
        cv2.putText(frame, f"t={t}/{seq_len}  batch={batch_id}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        if t >= B_idx:
            rollout_t = t - B_idx
            if rollout_t < len(gt_xy) and rollout_t < len(pred_xy):
                err = np.linalg.norm(pred_xy[rollout_t] - gt_xy[rollout_t])
                cv2.putText(frame, f"L2 err: {err:.4f}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)

        # Draw legend
        cv2.circle(frame, (DIMENSION - 120, 20), 5, RED, -1)
        cv2.putText(frame, "burn-in", (DIMENSION - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
        cv2.circle(frame, (DIMENSION - 120, 40), 5, GREEN, -1)
        cv2.putText(frame, "GT", (DIMENSION - 110, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
        cv2.circle(frame, (DIMENSION - 120, 60), 5, BLUE, -1)
        cv2.putText(frame, "predicted", (DIMENSION - 110, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)

        writer.write(frame)

    writer.release()
    file_size = output_path.stat().st_size / 1024
    logger.info("Saved: %s (%.0f KB, %d frames)", output_path, file_size, seq_len // stride)


def main():
    parser = argparse.ArgumentParser(description="Trajectory rollout overlay videos")
    parser.add_argument("--h5", default="data/trajectory/trajectory_dataset.h5")
    parser.add_argument("--checkpoint",
                        default="outputs/trajectory_mlp/checkpoints/checkpoint_epoch0080.pt")
    parser.add_argument("--archive_dir",
                        default="/users/k/k/kkannans/scratch/biosense_training_data")
    parser.add_argument("--output_dir", default="outputs/rollout_videos")
    parser.add_argument("--seq_indices", type=int, nargs="+",
                        default=[14, 132, 169, 100, 187])
    parser.add_argument("--burn_in_frac", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Load all HDF5 data into memory
    with h5py.File(args.h5, "r") as f:
        h5_data = {
            "centroid_xy": f["centroid_xy"][:],
            "velocity_xy": f["velocity_xy"][:],
            "dt": f["dt"][:],
            "actions": f["actions"][:],
            "batch_ids": f["batch_ids"][:],
            "frame_indices": f["frame_indices"][:],
            "seq_starts": f["sequence_starts"][:],
            "seq_lengths": f["sequence_lengths"][:],
        }

    for seq_idx in args.seq_indices:
        generate_video(
            seq_idx, h5_data, model, args.archive_dir, args.output_dir,
            burn_in_frac=args.burn_in_frac,
        )

    logger.info("Done! Videos saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
