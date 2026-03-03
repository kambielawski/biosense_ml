"""Preprocessing pipeline: converts raw data into training-ready formats.

Supports two modes:
  - resize: Downscale images and package into WebDataset tar shards.
  - autoencoder: Encode images into latent vectors and save as HDF5.
"""

import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from biosense_ml.pipeline.manifest import DatasetManifest, compute_config_hash
from biosense_ml.pipeline.webdataset_utils import ShardWriter

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def discover_raw_files(raw_dir: Path, batches: list[int] | None = None) -> list[Path]:
    """Find all image files in the specified batch directories.

    Args:
        raw_dir: Root archive directory containing batch-NNNNNN/ subdirs.
        batches: List of batch IDs to include. If empty or None, scans all batches.

    Returns:
        Sorted list of image file paths.
    """
    if batches:
        batch_dirs = []
        for batch_id in sorted(batches):
            batch_dir = raw_dir / f"batch-{batch_id:06d}"
            if batch_dir.is_dir():
                batch_dirs.append(batch_dir)
            else:
                logger.warning("Batch directory not found: %s", batch_dir)
        files = sorted(
            p
            for d in batch_dirs
            for p in d.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
    else:
        files = sorted(
            p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
    logger.info("Discovered %d image files in %s", len(files), raw_dir)
    return files


def load_metadata(image_path: Path) -> dict:
    """Load metadata associated with an image file.

    Stub implementation: looks for a .json sidecar file next to the image.
    Override this function to match your actual dataset structure.

    Args:
        image_path: Path to the image file.

    Returns:
        Metadata dict. Returns empty dict if no sidecar exists.
    """
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists():
        with open(sidecar) as f:
            return json.load(f)
    return {"filename": image_path.name}


def partition_files(files: list[Path]) -> list[Path]:
    """Partition file list for Slurm array jobs.

    If SLURM_ARRAY_TASK_ID is set, returns only the files assigned to this worker.
    Otherwise returns all files.
    """
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    num_tasks = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    if task_id is None or num_tasks is None:
        return files

    task_id = int(task_id)
    num_tasks = int(num_tasks)
    partition = [f for i, f in enumerate(files) if i % num_tasks == task_id]
    logger.info(
        "Slurm array task %d/%d: processing %d/%d files",
        task_id,
        num_tasks,
        len(partition),
        len(files),
    )
    return partition


def preprocess_resize(cfg: DictConfig, files: list[Path], output_dir: Path) -> DatasetManifest:
    """Resize images and write to WebDataset tar shards.

    Args:
        cfg: Full Hydra config.
        files: List of image file paths to process.
        output_dir: Directory to write tar shards to.

    Returns:
        DatasetManifest describing the output.
    """
    target_size = cfg.data.preprocessing.target_size
    shard_size = cfg.data.shard_size

    resize_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
    ])

    with ShardWriter(output_dir, shard_size=shard_size) as writer:
        for i, img_path in enumerate(tqdm(files, desc="Preprocessing (resize)")):
            try:
                image = Image.open(img_path).convert("RGB")
                image = resize_transform(image)
                metadata = load_metadata(img_path)
                key = f"sample_{i:08d}"
                writer.write(key=key, image=image, metadata=metadata)
            except Exception:
                logger.exception("Failed to process %s", img_path)
                continue

    manifest = DatasetManifest(
        config_hash=compute_config_hash(cfg),
        source_dir=str(cfg.data.biosense_archive_path),
        processed_dir=str(output_dir),
        num_samples=writer.total_samples_written,
        format="webdataset",
        shard_paths=writer.shard_paths,
    )
    return manifest


def preprocess_autoencoder(
    cfg: DictConfig, files: list[Path], output_dir: Path
) -> DatasetManifest:
    """Encode images with a pretrained autoencoder and save latent vectors to HDF5.

    Args:
        cfg: Full Hydra config.
        files: List of image file paths to encode.
        output_dir: Directory to write HDF5 file to.

    Returns:
        DatasetManifest describing the output.
    """
    # Load autoencoder — stub: user must provide their own model
    checkpoint_path = cfg.data.preprocessing.checkpoint_path
    if checkpoint_path is None:
        raise ValueError(
            "Autoencoder preprocessing requires data.preprocessing.checkpoint_path to be set."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading autoencoder from %s", checkpoint_path)

    # TODO: Replace with actual autoencoder loading logic
    # encoder = load_autoencoder_encoder(checkpoint_path).to(device).eval()
    raise NotImplementedError(
        "Autoencoder preprocessing is a stub. "
        "Implement your autoencoder loading and encoding logic here."
    )

    # When implemented, the flow should be:
    # 1. Batch-load images, resize, normalize
    # 2. Encode with autoencoder to get latent vectors
    # 3. Write latents + metadata to HDF5
    # See the resize path above for the metadata/manifest pattern.


def run_preprocessing(cfg: DictConfig) -> None:
    """Main preprocessing entry point.

    Args:
        cfg: Full Hydra config.
    """
    raw_dir = Path(cfg.data.biosense_archive_path)
    output_dir = Path(cfg.data.processed_data_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover and partition files
    batches = list(cfg.data.batches) if cfg.data.batches else None
    all_files = discover_raw_files(raw_dir, batches=batches)
    files = partition_files(all_files)

    if not files:
        logger.warning("No files to process.")
        return

    mode = cfg.data.preprocessing.mode
    logger.info("Running preprocessing in '%s' mode on %d files", mode, len(files))

    if mode == "resize":
        manifest = preprocess_resize(cfg, files, output_dir)
    elif mode == "autoencoder":
        manifest = preprocess_autoencoder(cfg, files, output_dir)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    manifest.save(output_dir / "manifest.json")
    logger.info("Preprocessing complete. %d samples written to %s", manifest.num_samples, output_dir)
