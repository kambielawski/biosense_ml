#!/usr/bin/env python3
"""Produce a side-by-side reconstruction video from a trained autoencoder checkpoint.

The output MP4 has each frame split vertically:
  - Top half:    original image from the shard
  - Bottom half: autoencoder reconstruction

Usage example:
    python scripts/vis_scripts/make_reconstruction_video.py \
        --checkpoint outputs/checkpoints/best.pt \
        --shard data/processed/batch-000000/shard-000000.tar \
        --output reconstruction.mp4 \
        --num_frames 120 \
        --fps 10 \
        --skip_frames 1
"""

import argparse
import io
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow running from repo root without installing the package
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biosense_ml.models.autoencoder import ConvAutoencoder
from biosense_ml.pipeline.transforms import get_transforms
from biosense_ml.utils.checkpoint import load_checkpoint

# ---------------------------------------------------------------------------
# ImageNet denormalization constants (must match transforms.py defaults)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CHUNK_SIZE = 16  # frames per forward pass (avoids OOM on large batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise autoencoder reconstructions as an MP4 video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a .pt checkpoint saved by biosense_ml/utils/checkpoint.py.",
    )
    parser.add_argument(
        "--shard",
        required=True,
        type=Path,
        help="Path to a WebDataset .tar shard to sample frames from.",
    )
    parser.add_argument(
        "--output",
        default="reconstruction.mp4",
        type=Path,
        help="Output video path (MP4).",
    )
    parser.add_argument(
        "--num_frames",
        default=120,
        type=int,
        help="Number of consecutive (post-skip) frames to include in the video.",
    )
    parser.add_argument(
        "--fps",
        default=10,
        type=int,
        help="Output video framerate.",
    )
    parser.add_argument(
        "--skip_frames",
        default=1,
        type=int,
        help="Take every Nth frame from the shard (1=no skip, 2=every other, etc.).",
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        default=default_device,
        help="Torch device to run inference on.",
    )
    parser.add_argument(
        "--num_encoder_blocks",
        default=5,
        type=int,
        help="Number of stride-2 encoder blocks (controls spatial bottleneck size).",
    )
    parser.add_argument(
        "--latent_channels",
        default=512,
        type=int,
        help="Channel depth at the bottleneck.",
    )
    parser.add_argument(
        "--bottleneck_spatial",
        default=None,
        type=int,
        help="Adaptive pool target spatial size (e.g. 12 for 12x12). None = use natural size.",
    )
    return parser.parse_args()


def _minimal_cfg(
    num_encoder_blocks: int = 5,
    latent_channels: int = 512,
    bottleneck_spatial: int | None = None,
) -> OmegaConf:
    """Build the minimal OmegaConf DictConfig that get_transforms and ConvAutoencoder need."""
    model_cfg = {
        "input_size": 512,
        "latent_channels": latent_channels,
        "num_encoder_blocks": num_encoder_blocks,
    }
    if bottleneck_spatial is not None:
        model_cfg["bottleneck_spatial"] = bottleneck_spatial
    return OmegaConf.create(
        {
            "model": model_cfg,
            "data": {
                "preprocessing": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
        }
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_encoder_blocks: int = 5,
    latent_channels: int = 512,
    bottleneck_spatial: int | None = None,
) -> ConvAutoencoder:
    """Instantiate ConvAutoencoder and load weights from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Torch device to move the model onto.
        num_encoder_blocks: Number of stride-2 encoder blocks.
        latent_channels: Channel depth at bottleneck.
        bottleneck_spatial: Adaptive pool target spatial size (None = natural).

    Returns:
        Model in eval mode on the requested device.
    """
    cfg = _minimal_cfg(
        num_encoder_blocks=num_encoder_blocks,
        latent_channels=latent_channels,
        bottleneck_spatial=bottleneck_spatial,
    )
    model = ConvAutoencoder(cfg.model)
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()
    return model


def load_frames_from_shard(
    shard_path: Path,
    num_frames: int,
    skip_frames: int,
    transform,
) -> torch.Tensor:
    """Extract a contiguous segment of frames from a WebDataset tar shard.

    Reads the tar directly with Python's tarfile module for determinism —
    no WebDataset async machinery.

    Args:
        shard_path: Path to the .tar shard.
        num_frames: Number of frames to return after applying the skip.
        skip_frames: Keep every Nth file (1 = all, 2 = every other, etc.).
        transform: Torchvision transform to apply to each PIL image.

    Returns:
        Float tensor of shape (N, 3, 512, 512) where N <= num_frames.
    """
    with tarfile.open(shard_path, "r") as tar:
        # Collect and sort all JPEG members
        jpg_members = sorted(
            [m for m in tar.getmembers() if m.name.endswith(".jpg")],
            key=lambda m: m.name,
        )

    # We need num_frames after skipping, so grab num_frames * skip_frames
    # candidates up front, then subsample.
    candidates = jpg_members[: num_frames * skip_frames]
    selected = candidates[::skip_frames][:num_frames]

    if len(selected) == 0:
        raise RuntimeError(
            f"No JPEG frames found in shard: {shard_path}. "
            "Check that the tar contains .jpg files."
        )
    if len(selected) < num_frames:
        print(
            f"Warning: shard only has {len(selected)} frames after skip "
            f"(requested {num_frames}). Proceeding with {len(selected)} frames."
        )

    tensors = []
    with tarfile.open(shard_path, "r") as tar:
        for member in tqdm(selected, desc="Loading frames", unit="frame"):
            f = tar.extractfile(member)
            if f is None:
                continue
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            tensors.append(transform(img))

    return torch.stack(tensors)  # (N, 3, H, W)


@torch.no_grad()
def run_inference(
    model: ConvAutoencoder,
    frames: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run autoencoder forward pass in chunks to avoid OOM.

    Args:
        model: Trained ConvAutoencoder in eval mode.
        frames: Input tensor (N, 3, H, W) — normalised, on CPU.
        device: Device to run inference on.

    Returns:
        Reconstruction tensor (N, 3, H, W) on CPU.
    """
    reconstructions = []
    for start in tqdm(
        range(0, len(frames), CHUNK_SIZE),
        desc="Running inference",
        unit="chunk",
    ):
        chunk = frames[start : start + CHUNK_SIZE].to(device)
        recon, _ = model(chunk)
        reconstructions.append(recon.cpu())
    return torch.cat(reconstructions, dim=0)  # (N, 3, H, W)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalisation, clamp to [0, 1].

    Args:
        tensor: Normalised tensor (N, 3, H, W) or (3, H, W).

    Returns:
        Denormalised tensor in [0, 1], same shape.
    """
    mean = _IMAGENET_MEAN.to(tensor.device)
    std = _IMAGENET_STD.to(tensor.device)
    return (tensor * std + mean).clamp(0.0, 1.0)


def tensor_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a single (3, H, W) float tensor in [0,1] to (H, W, 3) uint8 BGR numpy.

    Args:
        t: Tensor of shape (3, H, W), values in [0, 1].

    Returns:
        NumPy array (H, W, 3) in BGR uint8 suitable for cv2.VideoWriter.
    """
    rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_video(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    output_path: Path,
    fps: int,
) -> None:
    """Assemble and write the side-by-side MP4.

    Each video frame is:
        [original     ]  <- top half  (512 x 512)
        [reconstruction]  <- bottom half (512 x 512)
    Total frame size: 512 wide x 1024 tall.

    Args:
        originals: Denormalised tensor (N, 3, 512, 512) in [0, 1].
        reconstructions: Denormalised tensor (N, 3, 512, 512) in [0, 1].
        output_path: Where to write the MP4 file.
        fps: Output framerate.
    """
    assert originals.shape == reconstructions.shape, (
        f"Shape mismatch: {originals.shape} vs {reconstructions.shape}"
    )
    N, _, H, W = originals.shape
    video_h, video_w = H * 2, W  # stacked vertically

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h))

    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open output path: {output_path}. "
            "Check that OpenCV was built with video support."
        )

    for i in tqdm(range(N), desc="Writing video", unit="frame"):
        orig_bgr = tensor_to_bgr_uint8(originals[i])
        recon_bgr = tensor_to_bgr_uint8(reconstructions[i])
        combined = np.concatenate([orig_bgr, recon_bgr], axis=0)  # (1024, 512, 3)
        writer.write(combined)

    writer.release()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    if not args.shard.exists():
        sys.exit(f"Shard not found: {args.shard}")
    if args.skip_frames < 1:
        sys.exit("--skip_frames must be >= 1")

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(
        args.checkpoint,
        device,
        num_encoder_blocks=args.num_encoder_blocks,
        latent_channels=args.latent_channels,
        bottleneck_spatial=args.bottleneck_spatial,
    )

    # ------------------------------------------------------------------
    # Load and transform frames from shard
    # ------------------------------------------------------------------
    cfg = _minimal_cfg(
        num_encoder_blocks=args.num_encoder_blocks,
        latent_channels=args.latent_channels,
        bottleneck_spatial=args.bottleneck_spatial,
    )
    transform = get_transforms(cfg, split="val")
    print(f"Loading frames from shard: {args.shard}")
    frames = load_frames_from_shard(
        args.shard,
        num_frames=args.num_frames,
        skip_frames=args.skip_frames,
        transform=transform,
    )
    print(f"Loaded {len(frames)} frames, shape: {tuple(frames.shape)}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    reconstructions = run_inference(model, frames, device)

    # ------------------------------------------------------------------
    # Denormalize
    # ------------------------------------------------------------------
    orig_dn = denormalize(frames)
    recon_dn = denormalize(reconstructions)

    # ------------------------------------------------------------------
    # Write video
    # ------------------------------------------------------------------
    N, _, H, W = orig_dn.shape
    print(f"Writing video: {args.output}")
    write_video(orig_dn, recon_dn, args.output, fps=args.fps)

    print(
        f"Wrote {N}-frame video to {args.output} "
        f"({W}x{H * 2}, {args.fps}fps)"
    )


if __name__ == "__main__":
    main()
