#!/usr/bin/env python3
"""Visualize RSSM trajectory distribution: 30 stochastic rollouts overlaid on ground truth.

For each timestep, this script:
  1. Decodes all 30 rollout predictions through the AE decoder
  2. Extracts organoid centroids from each decoded frame via adaptive thresholding
  3. Draws all centroid positions as colored dots on the ground truth frame
  4. Accumulates trajectory paths showing how possible futures fan out

Usage:
    python scripts/vis_scripts/make_trajectory_distribution_video.py \
        --rssm_checkpoint outputs/rssm/checkpoints/checkpoint_best.pt \
        --ae_checkpoint outputs/bottleneck_16x32/checkpoints/checkpoint_best.pt \
        --latent_h5 data/rssm/latents_16x32.h5 \
        --output outputs/rssm/trajectory_distribution.mp4 \
        --num_rollouts 30 \
        --context_len 16 \
        --rollout_len 48
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biosense_ml.models.autoencoder import ConvAutoencoder
from biosense_ml.models.rssm import RSSM

# ImageNet denormalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Centroid detection params (tuned for 512x512 AE-decoded images)
BLOCK_SIZE = 17
C_VALUE = 29
MASK_RADIUS = 240
MIN_AREA = 50  # minimum contour area to count as organoid


def parse_args():
    parser = argparse.ArgumentParser(description="RSSM trajectory distribution visualization")
    parser.add_argument("--rssm_checkpoint", required=True, type=Path)
    parser.add_argument("--ae_checkpoint", required=True, type=Path)
    parser.add_argument("--latent_h5", required=True, type=Path)
    parser.add_argument("--output", default="trajectory_distribution.mp4", type=Path)
    parser.add_argument("--num_rollouts", default=30, type=int)
    parser.add_argument("--context_len", default=16, type=int)
    parser.add_argument("--rollout_len", default=48, type=int)
    parser.add_argument("--sequence_idx", default=None, type=int)
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_ae(checkpoint_path: Path, device: torch.device) -> ConvAutoencoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["config"]["model"])
    model = ConvAutoencoder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_rssm(checkpoint_path: Path, device: torch.device) -> RSSM:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["config"]["model"])
    model = RSSM(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def decode_latents_to_images(ae_latents, ae_model, num_encoder_blocks, latent_channels):
    """Decode flat AE latents to images. Returns (N, 3, 512, 512)."""
    spatial = 512 // (2 ** num_encoder_blocks)
    bottleneck = ae_latents.view(-1, latent_channels, spatial, spatial)
    return ae_model.decoder(bottleneck)


def denormalize(tensor):
    mean = _IMAGENET_MEAN.to(tensor.device)
    std = _IMAGENET_STD.to(tensor.device)
    return (tensor * std + mean).clamp(0.0, 1.0)


def tensor_to_bgr_uint8(t):
    """(3, H, W) float [0,1] -> (H, W, 3) uint8 BGR."""
    rgb = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def extract_centroid(frame_bgr):
    """Extract the largest organoid centroid from a 512x512 BGR frame.

    Uses multiple strategies to handle both raw and AE-decoded images:
    1. Otsu thresholding on blurred grayscale (robust to varying contrast)
    2. Fallback to center-of-mass of darkest 10% of pixels

    Returns (cx, cy) or None if no organoid detected.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Apply circular mask to ignore edges
    circular = np.zeros_like(gray)
    cv2.circle(circular, (w // 2, h // 2), MASK_RADIUS, 255, -1)
    gray_masked = cv2.bitwise_and(gray, circular)

    # Strategy 1: Otsu thresholding on blurred image
    blurred = cv2.GaussianBlur(gray_masked, (15, 15), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, circular)

    # Clean up small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_centroid = None
    best_area = MIN_AREA
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > best_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_centroid = (cx, cy)
                best_area = area

    if best_centroid is not None:
        return best_centroid

    # Strategy 2: center of mass of darkest pixels within circular mask
    # This works even with blurry/low-contrast AE reconstructions
    roi_pixels = gray_masked[circular > 0]
    if len(roi_pixels) == 0:
        return None
    threshold_val = np.percentile(roi_pixels[roi_pixels > 0], 10)
    dark_mask = (gray_masked > 0) & (gray_masked < threshold_val)
    if dark_mask.sum() < 10:
        return None
    ys, xs = np.where(dark_mask)
    return (int(xs.mean()), int(ys.mean()))


def generate_color_palette(n):
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color_hsv = np.array([[[hue, 200, 230]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in color_bgr))
    return colors


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    total_len = args.context_len + args.rollout_len

    # Load models
    print(f"Loading AE from {args.ae_checkpoint}")
    ae_model = load_ae(args.ae_checkpoint, device)
    ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    ae_cfg = ae_ckpt["config"]["model"]
    num_encoder_blocks = ae_cfg.get("num_encoder_blocks", 5)
    latent_channels = ae_cfg.get("latent_channels", 512)

    print(f"Loading RSSM from {args.rssm_checkpoint}")
    rssm = load_rssm(args.rssm_checkpoint, device)

    # Load HDF5
    print(f"Loading latents from {args.latent_h5}")
    with h5py.File(args.latent_h5, "r") as f:
        all_latents = f["latents"][:]
        all_actions = f["actions"][:]
        seq_starts = f["sequence_starts"][:]
        seq_lengths = f["sequence_lengths"][:]

    # Pick a sequence
    si = args.sequence_idx if args.sequence_idx is not None else int(np.argmax(seq_lengths))
    start = int(seq_starts[si])
    slen = int(seq_lengths[si])
    print(f"Sequence {si}: length={slen}")

    if slen < total_len:
        args.rollout_len = slen - args.context_len
        total_len = slen

    seq_latents = torch.tensor(all_latents[start:start + total_len], dtype=torch.float32).to(device)
    seq_actions = torch.tensor(all_actions[start:start + total_len], dtype=torch.float32).to(device)

    # --- Context phase: teacher-force through posterior (residual mode) ---
    print(f"Running {args.context_len} context frames through posterior (residual)...")
    B = 1
    h_t, z_t = rssm.initial_state(B, device)
    ctx_latents = seq_latents[:args.context_len].unsqueeze(0)
    ctx_actions = seq_actions[:args.context_len].unsqueeze(0)

    # Compute deltas for context
    zeros = torch.zeros(B, 1, rssm.ae_latent_dim, device=device)
    ctx_deltas = torch.cat([zeros, ctx_latents[:, 1:] - ctx_latents[:, :-1]], dim=1)

    for t in range(args.context_len):
        a_prev = torch.zeros(B, rssm.action_dim, device=device) if t == 0 else ctx_actions[:, t - 1]
        delta_t = ctx_deltas[:, t]
        step = rssm.forward_single_step(h_t, z_t, a_prev, delta_t)
        h_t = step["h_t"]
        z_t = step["z_t"]

    print(f"Context done. h_t norm={h_t.norm().item():.2f}, z_t norm={z_t.norm().item():.2f}")

    # --- Generate N stochastic rollouts from the same state ---
    print(f"Generating {args.num_rollouts} stochastic rollouts of {args.rollout_len} steps...")
    rollout_actions = seq_actions[args.context_len - 1:args.context_len - 1 + args.rollout_len]
    rollout_actions = rollout_actions.unsqueeze(0)  # (1, T, action_dim)

    # Repeat initial state and x_last for all rollouts
    h_batch = h_t.expand(args.num_rollouts, -1).clone()
    z_batch = z_t.expand(args.num_rollouts, -1).clone()
    actions_batch = rollout_actions.expand(args.num_rollouts, -1, -1)
    x_last = ctx_latents[:, -1].expand(args.num_rollouts, -1).clone()

    imagination = rssm.imagine(h_batch, z_batch, actions_batch, x_last)
    # imagination["obs_pred"]: (num_rollouts, rollout_len, ae_latent_dim)

    # --- Decode all rollouts and extract centroids ---
    print("Decoding rollouts and extracting centroids...")
    # centroids_per_rollout[r][t] = (cx, cy) or None
    centroids_per_rollout = [[] for _ in range(args.num_rollouts)]

    for t in tqdm(range(args.rollout_len), desc="Decoding timesteps"):
        # Get predicted obs for all rollouts at this timestep
        obs_pred_t = imagination["obs_pred"][:, t, :]  # (num_rollouts, ae_latent_dim)
        ae_latents_t = obs_pred_t  # already in AE space, no projector

        # Decode through AE in one batch
        images_t = decode_latents_to_images(ae_latents_t, ae_model, num_encoder_blocks, latent_channels)
        images_t = denormalize(images_t.cpu())

        for r in range(args.num_rollouts):
            bgr = tensor_to_bgr_uint8(images_t[r])
            centroid = extract_centroid(bgr)
            centroids_per_rollout[r].append(centroid)

    # --- Decode ground truth frames ---
    print("Decoding ground truth frames...")
    gt_latents = seq_latents[args.context_len:args.context_len + args.rollout_len]
    gt_images = decode_latents_to_images(gt_latents, ae_model, num_encoder_blocks, latent_channels)
    gt_images = denormalize(gt_images.cpu())

    # Also decode context frames
    ctx_gt_latents = seq_latents[:args.context_len]
    ctx_gt_images = decode_latents_to_images(ctx_gt_latents, ae_model, num_encoder_blocks, latent_channels)
    ctx_gt_images = denormalize(ctx_gt_images.cpu())

    # --- Write video ---
    print("Writing video...")
    H, W = 512, 512
    label_h = 40
    video_h = H + label_h
    video_w = W
    colors = generate_color_palette(args.num_rollouts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (video_w, video_h))

    if not writer.isOpened():
        sys.exit(f"Failed to open video writer: {args.output}")

    # Context frames (no overlay)
    for i in range(args.context_len):
        frame_bgr = tensor_to_bgr_uint8(ctx_gt_images[i])
        label_bar = np.zeros((label_h, W, 3), dtype=np.uint8)
        cv2.putText(label_bar, f"CONTEXT {i+1}/{args.context_len}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        combined = np.concatenate([frame_bgr, label_bar], axis=0)
        writer.write(combined)

    # Rollout frames with trajectory overlay
    # Accumulate trajectory paths
    trajectory_paths = [[] for _ in range(args.num_rollouts)]

    for t in range(args.rollout_len):
        frame_bgr = tensor_to_bgr_uint8(gt_images[t])

        # Update trajectory paths
        for r in range(args.num_rollouts):
            c = centroids_per_rollout[r][t]
            if c is not None:
                trajectory_paths[r].append(c)

        # Draw trajectory lines (fading older points)
        for r in range(args.num_rollouts):
            path = trajectory_paths[r]
            if len(path) >= 2:
                for j in range(1, len(path)):
                    alpha = 0.3 + 0.7 * (j / len(path))  # fade in
                    thickness = max(1, int(2 * alpha))
                    cv2.line(frame_bgr, path[j-1], path[j], colors[r], thickness,
                             lineType=cv2.LINE_AA)

        # Draw current centroid positions as dots
        for r in range(args.num_rollouts):
            c = centroids_per_rollout[r][t]
            if c is not None:
                cv2.circle(frame_bgr, c, 4, colors[r], -1, lineType=cv2.LINE_AA)
                cv2.circle(frame_bgr, c, 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Count detected centroids
        n_detected = sum(1 for r in range(args.num_rollouts)
                         if centroids_per_rollout[r][t] is not None)

        label_bar = np.zeros((label_h, W, 3), dtype=np.uint8)
        cv2.putText(label_bar,
                    f"ROLLOUT step {t+1}/{args.rollout_len} | {n_detected}/{args.num_rollouts} detected",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        combined = np.concatenate([frame_bgr, label_bar], axis=0)
        writer.write(combined)

    writer.release()
    total_frames = args.context_len + args.rollout_len

    # Print detection stats
    total_detections = sum(
        1 for r in range(args.num_rollouts)
        for t in range(args.rollout_len)
        if centroids_per_rollout[r][t] is not None
    )
    total_possible = args.num_rollouts * args.rollout_len
    print(f"\nCentroid detection rate: {total_detections}/{total_possible} "
          f"({100*total_detections/total_possible:.1f}%)")
    print(f"Wrote {total_frames}-frame video to {args.output} ({video_w}x{video_h}, {args.fps}fps)")


if __name__ == "__main__":
    main()
