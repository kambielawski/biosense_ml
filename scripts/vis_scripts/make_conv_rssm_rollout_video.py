#!/usr/bin/env python3
"""Produce a side-by-side video: real frames vs ConvRSSM predicted frames.

Adapted from make_rssm_rollout_video.py for the spatial ConvRSSM model.
The ConvRSSM works in (32, 16, 16) spatial bottleneck space throughout.

Flow:
  1. Load ConvRSSM checkpoint + frozen AE checkpoint
  2. Pick a validation sequence from the HDF5
  3. Run context_len frames through the posterior (teacher forcing) to build state
  4. Roll out rollout_len steps via the prior (imagination)
  5. Decode both real and predicted spatial latents through the AE decoder
  6. Show original (top) vs predicted (bottom) as MP4

Usage:
    python scripts/vis_scripts/make_conv_rssm_rollout_video.py \
        --rssm_checkpoint outputs/conv_rssm/checkpoints/checkpoint_best.pt \
        --ae_checkpoint outputs/bottleneck_16x32/checkpoints/checkpoint_best.pt \
        --latent_h5 data/rssm/latents_16x32.h5 \
        --output conv_rssm_rollout.mp4 \
        --context_len 16 \
        --rollout_len 48 \
        --fps 10
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
from biosense_ml.models.conv_rssm import ConvRSSM

# ImageNet denormalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="ConvRSSM rollout visualization")
    parser.add_argument("--rssm_checkpoint", required=True, type=Path)
    parser.add_argument("--ae_checkpoint", required=True, type=Path)
    parser.add_argument("--latent_h5", required=True, type=Path)
    parser.add_argument("--output", default="conv_rssm_rollout.mp4", type=Path)
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


def load_conv_rssm(checkpoint_path: Path, device: torch.device) -> ConvRSSM:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["config"]["model"])
    model = ConvRSSM(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def decode_spatial_latents_to_images(
    spatial_latents: torch.Tensor,
    ae_model: ConvAutoencoder,
) -> torch.Tensor:
    """Decode spatial AE latents (N, C, H, W) to images via the AE decoder.

    Returns:
        (N, 3, 512, 512) decoded images (ImageNet-normalized scale)
    """
    return ae_model.decoder(spatial_latents)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(tensor.device)
    std = _IMAGENET_STD.to(tensor.device)
    return (tensor * std + mean).clamp(0.0, 1.0)


def tensor_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    rgb = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    total_len = args.context_len + args.rollout_len

    # Load models
    print(f"Loading AE from {args.ae_checkpoint}")
    ae_model = load_ae(args.ae_checkpoint, device)

    print(f"Loading ConvRSSM from {args.rssm_checkpoint}")
    rssm_model = load_conv_rssm(args.rssm_checkpoint, device)

    C_ae = rssm_model.ae_latent_channels
    S = rssm_model.spatial_size

    # Load HDF5 data
    print(f"Loading latents from {args.latent_h5}")
    with h5py.File(args.latent_h5, "r") as f:
        all_latents = f["latents"][:]
        all_actions = f["actions"][:]
        seq_starts = f["sequence_starts"][:]
        seq_lengths = f["sequence_lengths"][:]

    # Pick a sequence
    if args.sequence_idx is not None:
        si = args.sequence_idx
    else:
        si = int(np.argmax(seq_lengths))

    start = int(seq_starts[si])
    slen = int(seq_lengths[si])
    print(f"Sequence {si}: length={slen}, start={start}")

    if slen < total_len:
        print(f"Warning: sequence length {slen} < context+rollout {total_len}, truncating rollout")
        args.rollout_len = slen - args.context_len
        total_len = slen

    # Extract sequence data and reshape to spatial
    seq_latents_flat = torch.tensor(all_latents[start:start + total_len], dtype=torch.float32).to(device)
    seq_actions = torch.tensor(all_actions[start:start + total_len], dtype=torch.float32).to(device)

    # Reshape to spatial: (T, D) -> (T, C, H, W)
    seq_latents = seq_latents_flat.view(-1, C_ae, S, S)

    # --- Context phase: run posterior ---
    print(f"Running posterior for {args.context_len} context frames...")
    context_latents = seq_latents[:args.context_len].unsqueeze(0)  # (1, T_ctx, C, H, W)
    context_actions = seq_actions[:args.context_len].unsqueeze(0)

    # Compute deltas for context
    B = 1
    zeros = torch.zeros(B, 1, C_ae, S, S, device=device)
    context_deltas = torch.cat([zeros, context_latents[:, 1:] - context_latents[:, :-1]], dim=1)

    h_t, z_t = rssm_model.initial_state(B, device)
    ctx_delta_preds = []

    for t in range(args.context_len):
        if t == 0:
            a_prev = torch.zeros(B, rssm_model.action_dim, device=device)
        else:
            a_prev = context_actions[:, t - 1]

        delta_t = context_deltas[:, t]
        step = rssm_model.forward_single_step(h_t, z_t, a_prev, delta_t)
        h_t = step["h_t"]
        z_t = step["z_t"]
        ctx_delta_preds.append(step["obs_pred"])

    print(f"Context done. h_t norm={h_t.norm().item():.2f}, z_t norm={z_t.norm().item():.2f}")

    # Accumulate context delta predictions back to absolute latents
    ctx_delta_preds = torch.stack(ctx_delta_preds, dim=0)  # (T_ctx, B, C, H, W)
    ctx_abs_preds = []
    x_acc = context_latents[0, 0]  # First real frame as anchor (C, H, W)
    ctx_abs_preds.append(x_acc)
    for t in range(1, args.context_len):
        x_acc = x_acc + ctx_delta_preds[t, 0]
        ctx_abs_preds.append(x_acc)
    ctx_rssm_latents = torch.stack(ctx_abs_preds, dim=0)  # (T_ctx, C, H, W)

    # --- Rollout phase: imagine using prior ---
    print(f"Imagining {args.rollout_len} steps via prior...")
    rollout_actions = seq_actions[args.context_len - 1:args.context_len - 1 + args.rollout_len].unsqueeze(0)
    x_last = context_latents[:, -1]  # (1, C, H, W)
    imagination = rssm_model.imagine(h_t, z_t, rollout_actions, x_last)

    # obs_pred is accumulated absolute spatial latents (1, T_roll, C, H, W)
    imagined_latents = imagination["obs_pred"][0]  # (T_roll, C, H, W)

    # Ground truth for rollout period
    gt_latents_rollout = seq_latents[args.context_len:args.context_len + args.rollout_len]

    # Decode through AE decoder
    print("Decoding predicted frames through AE decoder...")
    pred_images = decode_spatial_latents_to_images(imagined_latents, ae_model)
    print("Decoding ground truth frames through AE decoder...")
    gt_images_rollout = decode_spatial_latents_to_images(gt_latents_rollout, ae_model)

    # Also decode context frames
    ctx_gt_images = decode_spatial_latents_to_images(seq_latents[:args.context_len], ae_model)
    ctx_rssm_images = decode_spatial_latents_to_images(ctx_rssm_latents, ae_model)

    # Denormalize all images
    gt_all = denormalize(torch.cat([ctx_gt_images, gt_images_rollout], dim=0).cpu())
    pred_all = denormalize(torch.cat([ctx_rssm_images, pred_images], dim=0).cpu())

    # --- Write video ---
    N = gt_all.shape[0]
    H, W = 512, 512
    video_h = H * 2 + 40
    video_w = W

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (video_w, video_h))

    if not writer.isOpened():
        sys.exit(f"Failed to open video writer: {args.output}")

    print(f"Writing {N}-frame video to {args.output}")
    for i in tqdm(range(N), desc="Writing video"):
        gt_bgr = tensor_to_bgr_uint8(gt_all[i])
        pred_bgr = tensor_to_bgr_uint8(pred_all[i])

        label_bar = np.zeros((40, W, 3), dtype=np.uint8)
        if i < args.context_len:
            label = f"CONTEXT frame {i+1}/{args.context_len} (posterior)"
            color = (0, 255, 0)
        else:
            step = i - args.context_len + 1
            label = f"ROLLOUT step {step}/{args.rollout_len} (prior imagination)"
            color = (0, 165, 255)
        cv2.putText(label_bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        combined = np.concatenate([gt_bgr, label_bar, pred_bgr], axis=0)
        writer.write(combined)

    writer.release()
    print(f"Wrote {N}-frame video to {args.output} ({video_w}x{video_h}, {args.fps}fps)")
    print(f"  Context frames: {args.context_len} (posterior reconstruction)")
    print(f"  Rollout frames: {args.rollout_len} (prior imagination)")


if __name__ == "__main__":
    main()
