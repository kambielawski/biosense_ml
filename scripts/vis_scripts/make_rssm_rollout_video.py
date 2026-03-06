#!/usr/bin/env python3
"""Produce a side-by-side video: real frames vs RSSM predicted frames.

Flow:
  1. Load RSSM checkpoint + frozen AE checkpoint
  2. Pick a validation sequence from the HDF5
  3. Run context_len frames through the posterior (teacher forcing) to build state
  4. Roll out rollout_len steps via the prior (imagination)
  5. Decode both real and predicted AE latents through the AE decoder
  6. Show original (top) vs predicted (bottom) as MP4

Usage:
    python scripts/vis_scripts/make_rssm_rollout_video.py \
        --rssm_checkpoint outputs/rssm/checkpoints/checkpoint_best.pt \
        --ae_checkpoint outputs/bottleneck_16x32/checkpoints/checkpoint_best.pt \
        --latent_h5 data/rssm/latents_16x32.h5 \
        --output rssm_rollout.mp4 \
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
from biosense_ml.models.rssm import RSSM

# ImageNet denormalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="RSSM rollout visualization")
    parser.add_argument("--rssm_checkpoint", required=True, type=Path)
    parser.add_argument("--ae_checkpoint", required=True, type=Path)
    parser.add_argument("--latent_h5", required=True, type=Path)
    parser.add_argument("--output", default="rssm_rollout.mp4", type=Path)
    parser.add_argument("--context_len", default=16, type=int,
                        help="Number of frames to process through posterior for context")
    parser.add_argument("--rollout_len", default=48, type=int,
                        help="Number of imagination steps via prior")
    parser.add_argument("--sequence_idx", default=None, type=int,
                        help="Which sequence to use (default: longest val sequence)")
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_ae(checkpoint_path: Path, device: torch.device) -> ConvAutoencoder:
    """Load frozen autoencoder from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["config"]["model"])
    model = ConvAutoencoder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_rssm(checkpoint_path: Path, device: torch.device) -> RSSM:
    """Load RSSM from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["config"]["model"])
    model = RSSM(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def decode_latents_to_images(
    ae_latents: torch.Tensor,
    ae_model: ConvAutoencoder,
    num_encoder_blocks: int,
    latent_channels: int,
) -> torch.Tensor:
    """Decode flat AE latents to images via the AE decoder.

    Args:
        ae_latents: (N, ae_latent_dim) flat latents
        ae_model: Frozen autoencoder
        num_encoder_blocks: To compute spatial size
        latent_channels: Channel depth

    Returns:
        (N, 3, 512, 512) decoded images (ImageNet-normalized scale)
    """
    spatial = 512 // (2 ** num_encoder_blocks)
    bottleneck = ae_latents.view(-1, latent_channels, spatial, spatial)
    return ae_model.decoder(bottleneck)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization, clamp to [0, 1]."""
    mean = _IMAGENET_MEAN.to(tensor.device)
    std = _IMAGENET_STD.to(tensor.device)
    return (tensor * std + mean).clamp(0.0, 1.0)


def tensor_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] -> (H, W, 3) uint8 BGR."""
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
    ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    ae_cfg = ae_ckpt["config"]["model"]
    num_encoder_blocks = ae_cfg.get("num_encoder_blocks", 5)
    latent_channels = ae_cfg.get("latent_channels", 512)

    print(f"Loading RSSM from {args.rssm_checkpoint}")
    rssm_model = load_rssm(args.rssm_checkpoint, device)

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
        # Pick the longest sequence
        si = int(np.argmax(seq_lengths))

    start = int(seq_starts[si])
    slen = int(seq_lengths[si])
    print(f"Sequence {si}: length={slen}, start={start}")

    if slen < total_len:
        print(f"Warning: sequence length {slen} < context+rollout {total_len}, truncating rollout")
        args.rollout_len = slen - args.context_len
        total_len = slen

    # Extract sequence data
    seq_latents = torch.tensor(all_latents[start:start + total_len], dtype=torch.float32).to(device)
    seq_actions = torch.tensor(all_actions[start:start + total_len], dtype=torch.float32).to(device)

    # Add batch dimension
    latents_batch = seq_latents.unsqueeze(0)  # (1, T, ae_dim)
    actions_batch = seq_actions.unsqueeze(0)  # (1, T, action_dim)

    # --- Context phase: run posterior for context_len steps ---
    print(f"Running posterior for {args.context_len} context frames...")
    context_latents = latents_batch[:, :args.context_len]
    context_actions = actions_batch[:, :args.context_len]
    context_out = rssm_model(context_latents, context_actions)

    # Get final state from context
    h_final = context_out["z_t"][:, -1]  # Wait, need h_t not z_t
    # h_t isn't directly returned by forward(). Need to re-run manually.

    # Re-run context to get h_t at the end
    B = 1
    h_t, z_t = rssm_model.initial_state(B, device)

    for t in range(args.context_len):
        if t == 0:
            a_prev = torch.zeros(B, rssm_model.action_dim, device=device)
        else:
            a_prev = context_actions[:, t - 1]

        x_t = context_latents[:, t]
        step = rssm_model.forward_single_step(h_t, z_t, a_prev, x_t)
        h_t = step["h_t"]
        z_t = step["z_t"]

    print(f"Context done. h_t norm={h_t.norm().item():.2f}, z_t norm={z_t.norm().item():.2f}")

    # --- Rollout phase: imagine using prior ---
    print(f"Imagining {args.rollout_len} steps via prior...")
    rollout_actions = actions_batch[:, args.context_len - 1:args.context_len - 1 + args.rollout_len]
    imagination = rssm_model.imagine(h_t, z_t, rollout_actions)

    # obs_pred is already in AE latent space (no projector)
    imagined_obs_pred = imagination["obs_pred"]  # (1, rollout_len, ae_latent_dim)
    imagined_ae_latents = imagined_obs_pred.reshape(-1, imagined_obs_pred.shape[-1])

    # Decode ground truth AE latents for the rollout period
    gt_ae_latents = seq_latents[args.context_len:args.context_len + args.rollout_len]

    print("Decoding predicted frames through AE decoder...")
    pred_images = decode_latents_to_images(
        imagined_ae_latents, ae_model, num_encoder_blocks, latent_channels
    )
    print("Decoding ground truth frames through AE decoder...")
    gt_images = decode_latents_to_images(
        gt_ae_latents, ae_model, num_encoder_blocks, latent_channels
    )

    # Also decode context frames for a complete video
    ctx_ae_latents = seq_latents[:args.context_len]
    ctx_images = decode_latents_to_images(
        ctx_ae_latents, ae_model, num_encoder_blocks, latent_channels
    )

    # Context obs_pred is already in AE latent space (no projector)
    ctx_obs_pred = context_out["obs_pred"]  # (1, ctx_len, ae_latent_dim)
    ctx_rssm_ae_latents = ctx_obs_pred.reshape(-1, ctx_obs_pred.shape[-1])
    ctx_rssm_images = decode_latents_to_images(
        ctx_rssm_ae_latents, ae_model, num_encoder_blocks, latent_channels
    )

    # Denormalize all images
    gt_all = denormalize(torch.cat([ctx_images, gt_images], dim=0).cpu())
    pred_all = denormalize(torch.cat([ctx_rssm_images, pred_images], dim=0).cpu())

    # --- Write video ---
    N = gt_all.shape[0]
    H, W = 512, 512
    video_h = H * 2 + 40  # room for label bar
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

        # Label bar
        label_bar = np.zeros((40, W, 3), dtype=np.uint8)
        if i < args.context_len:
            label = f"CONTEXT frame {i+1}/{args.context_len} (posterior)"
            color = (0, 255, 0)  # green
        else:
            step = i - args.context_len + 1
            label = f"ROLLOUT step {step}/{args.rollout_len} (prior imagination)"
            color = (0, 165, 255)  # orange
        cv2.putText(label_bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Stack: ground truth | label bar | prediction
        combined = np.concatenate([gt_bgr, label_bar, pred_bgr], axis=0)
        writer.write(combined)

    writer.release()
    print(f"Wrote {N}-frame video to {args.output} ({video_w}x{video_h}, {args.fps}fps)")
    print(f"  Context frames: {args.context_len} (posterior reconstruction)")
    print(f"  Rollout frames: {args.rollout_len} (prior imagination)")


if __name__ == "__main__":
    main()
