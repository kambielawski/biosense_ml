"""Encode the full WebDataset through a frozen autoencoder to HDF5.

Produces an HDF5 file with temporal sequences grouped by batch, suitable for
RSSM training. Each frame gets a flattened AE bottleneck vector and an
action vector derived from stimulus metadata.

HDF5 structure:
    /latents          (N_total, ae_latent_dim) float32 — flattened AE bottleneck
    /actions          (N_total, action_dim) float32 — intervention vectors
    /timestamps       (N_total,) float32 — time_since_batch_start in seconds
    /batch_ids        (N_total,) int32 — batch ID for each frame
    /frame_indices    (N_total,) int32 — frame index within batch
    /sequence_starts  (num_sequences,) int64 — start index of each sequence
    /sequence_lengths (num_sequences,) int32 — length of each sequence

Usage:
    python scripts/encode_latents.py \
        --checkpoint outputs/bottleneck_16x32/checkpoints/checkpoint_best.pt \
        --manifest data/processed/manifest.json \
        --output data/rssm/latents.h5 \
        --action_dim 3 \
        --batch_size 64 \
        --num_workers 4
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

from biosense_ml.models.autoencoder import ConvAutoencoder
from biosense_ml.pipeline.interventions import encode_action_3d, encode_action_2d, estimate_stimulus_duration
from biosense_ml.pipeline.transforms import get_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_autoencoder(checkpoint_path: str, device: torch.device) -> ConvAutoencoder:
    """Load a frozen autoencoder from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Target device.

    Returns:
        Frozen ConvAutoencoder in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]

    # Reconstruct a minimal DictConfig-like object for the model
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.create(cfg_dict["model"])

    model = ConvAutoencoder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    logger.info(
        "Loaded autoencoder from %s (epoch %d, metric %.4f)",
        checkpoint_path, ckpt["epoch"], ckpt["best_metric"],
    )
    return model


def build_shard_loader(
    shard_paths: list[str],
    project_root: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build a WebDataset loader that yields images + full metadata.

    No augmentation — just resize, center crop, and normalize for inference.
    """
    from torchvision import transforms

    # Deterministic val-like transform
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    abs_paths = [str(project_root / p) for p in shard_paths]
    safe_workers = min(num_workers, len(abs_paths))

    def decode_fn(sample: dict) -> dict:
        img = sample["jpg"]
        metadata = json.loads(sample["json"]) if isinstance(sample["json"], str) else sample["json"]
        if isinstance(metadata, bytes):
            metadata = json.loads(metadata.decode("utf-8"))
        return {"image": transform(img), "metadata": metadata}

    dataset = (
        wds.WebDataset(abs_paths, shardshuffle=False)
        .decode("pil")
        .map(decode_fn)
    )

    # We need individual samples, not batched — we'll batch manually
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=safe_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def main():
    parser = argparse.ArgumentParser(description="Encode dataset through frozen autoencoder to HDF5")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to AE checkpoint")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.json")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--action_dim", type=int, default=3, choices=[2, 3])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_shards", type=int, default=None, help="Cap shard count for testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load manifest
    project_root = Path(args.manifest).parent.parent  # manifest is at data/processed/manifest.json
    with open(args.manifest) as f:
        manifest = json.load(f)
    shard_paths = manifest["shard_paths"]

    if args.max_shards:
        shard_paths = shard_paths[:args.max_shards]
    logger.info("Processing %d shards", len(shard_paths))

    # Load frozen autoencoder
    model = load_autoencoder(args.checkpoint, device)

    # First pass: collect all samples grouped by batch_id
    # We need to group by batch to form temporal sequences
    logger.info("Pass 1: Reading all samples and encoding latents...")

    batch_samples: dict[int, list] = defaultdict(list)
    loader = build_shard_loader(shard_paths, project_root, args.batch_size, args.num_workers)

    # Process samples one at a time, accumulate in a buffer for batched encoding
    image_buffer = []
    meta_buffer = []
    ENCODE_BATCH_SIZE = args.batch_size

    def flush_buffer(image_buf, meta_buf, model, device):
        """Encode a batch of images through the frozen AE."""
        if not image_buf:
            return []
        images = torch.stack(image_buf).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            _, bottleneck = model(images)
        # Flatten spatial dims: (B, C, H, W) -> (B, C*H*W)
        flat = bottleneck.reshape(bottleneck.size(0), -1).cpu().numpy()
        results = []
        for i, meta in enumerate(meta_buf):
            results.append((flat[i], meta))
        return results

    n_processed = 0
    for sample in tqdm(loader, desc="Encoding"):
        image_buffer.append(sample["image"])
        meta_buffer.append(sample["metadata"])

        if len(image_buffer) >= ENCODE_BATCH_SIZE:
            for latent, meta in flush_buffer(image_buffer, meta_buffer, model, device):
                bid = meta["batch_id"]
                batch_samples[bid].append({
                    "latent": latent,
                    "metadata": meta,
                    "frame_index": meta["frame_index"],
                    "time_since_batch_start": meta.get("time_since_batch_start", 0.0),
                })
                n_processed += 1
            image_buffer.clear()
            meta_buffer.clear()

    # Flush remaining
    for latent, meta in flush_buffer(image_buffer, meta_buffer, model, device):
        bid = meta["batch_id"]
        batch_samples[bid].append({
            "latent": latent,
            "metadata": meta,
            "frame_index": meta["frame_index"],
            "time_since_batch_start": meta.get("time_since_batch_start", 0.0),
        })
        n_processed += 1

    logger.info("Encoded %d samples across %d batches", n_processed, len(batch_samples))

    # Sort each batch by frame_index to ensure temporal order
    for bid in batch_samples:
        batch_samples[bid].sort(key=lambda x: x["frame_index"])

    # Build action vectors per batch (need batch-level duration estimate)
    logger.info("Pass 2: Building action vectors and writing HDF5...")

    ae_latent_dim = batch_samples[next(iter(batch_samples))][0]["latent"].shape[0]
    action_dim = args.action_dim

    all_latents = []
    all_actions = []
    all_timestamps = []
    all_batch_ids = []
    all_frame_indices = []
    sequence_starts = []
    sequence_lengths = []

    offset = 0
    for bid in sorted(batch_samples.keys()):
        frames = batch_samples[bid]
        meta_list = [f["metadata"] for f in frames]
        stim_duration = estimate_stimulus_duration(meta_list)

        sequence_starts.append(offset)
        sequence_lengths.append(len(frames))

        for frame in frames:
            all_latents.append(frame["latent"])
            stimulus = frame["metadata"].get("stimulus", {})
            if action_dim == 3:
                action = encode_action_3d(stimulus, stim_duration)
            else:
                action = encode_action_2d(stimulus)
            all_actions.append(action)
            all_timestamps.append(frame["time_since_batch_start"])
            all_batch_ids.append(bid)
            all_frame_indices.append(frame["frame_index"])
            offset += 1

    # Write HDF5
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("latents", data=np.stack(all_latents), dtype="float32")
        f.create_dataset("actions", data=np.stack(all_actions), dtype="float32")
        f.create_dataset("timestamps", data=np.array(all_timestamps, dtype="float32"))
        f.create_dataset("batch_ids", data=np.array(all_batch_ids, dtype="int32"))
        f.create_dataset("frame_indices", data=np.array(all_frame_indices, dtype="int32"))
        f.create_dataset("sequence_starts", data=np.array(sequence_starts, dtype="int64"))
        f.create_dataset("sequence_lengths", data=np.array(sequence_lengths, dtype="int32"))

        # Store metadata
        f.attrs["ae_latent_dim"] = ae_latent_dim
        f.attrs["action_dim"] = action_dim
        f.attrs["num_sequences"] = len(sequence_starts)
        f.attrs["num_samples"] = offset
        f.attrs["checkpoint"] = args.checkpoint

    logger.info(
        "Wrote %s: %d samples, %d sequences, ae_latent_dim=%d, action_dim=%d",
        output_path, offset, len(sequence_starts), ae_latent_dim, action_dim,
    )


if __name__ == "__main__":
    main()
