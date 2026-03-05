"""Preprocessing pipeline: converts raw data into training-ready formats.

Supports two modes:
  - resize: Downscale images and package into WebDataset tar shards.
  - autoencoder: Encode images into latent vectors and save as HDF5.

Batches are processed in parallel using multiprocessing.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from biosense_ml.pipeline.manifest import DatasetManifest, compute_config_hash
from biosense_ml.pipeline.webdataset_utils import ShardWriter

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def discover_batch_dirs(raw_dir: Path, batches: list[int] | None = None) -> list[tuple[int, Path]]:
    """Find batch directories in the archive.

    Args:
        raw_dir: Root archive directory containing batch-NNNNNN/ subdirs.
        batches: List of batch IDs to include. If empty or None, discovers all.

    Returns:
        Sorted list of (batch_id, batch_dir) tuples.
    """
    if batches:
        result = []
        for batch_id in sorted(batches):
            batch_dir = raw_dir / f"batch-{batch_id:06d}"
            if batch_dir.is_dir():
                result.append((batch_id, batch_dir))
            else:
                logger.warning("Batch directory not found: %s", batch_dir)
        return result

    # Auto-discover all batch-NNNNNN/ directories
    result = []
    for d in sorted(raw_dir.iterdir()):
        if d.is_dir() and d.name.startswith("batch-"):
            try:
                batch_id = int(d.name.split("-", 1)[1])
                result.append((batch_id, d))
            except ValueError:
                logger.warning("Skipping non-numeric batch dir: %s", d)
    logger.info("Discovered %d batch directories in %s", len(result), raw_dir)
    return result


def discover_batch_files(batch_dir: Path) -> list[Path]:
    """Find all image files in a single batch directory, sorted for temporal order.

    Args:
        batch_dir: Path to a batch-NNNNNN/ directory.

    Returns:
        Sorted list of image file paths.
    """
    return sorted(
        p for p in batch_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )


def parse_image_timestamp(image_path: Path) -> datetime:
    """Extract timestamp from an image filename.

    Expected format: {uuid}-{YYYY-MM-DD}T{HH.MM.SS}.jpg
    The dots in the time portion replace colons for filesystem compatibility.
    """
    stem = image_path.stem
    # Match the ISO-ish timestamp at the end: YYYY-MM-DDTHH.MM.SS
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2})$", stem)
    if not match:
        raise ValueError(f"Cannot parse timestamp from filename: {image_path.name}")
    ts_str = match.group(1).replace(".", ":")  # 20.55.31 -> 20:55:31
    return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


def load_commands(batch_dir: Path) -> list[dict]:
    """Load all command JSON files from a batch's commands/ directory.

    Args:
        batch_dir: Path to a batch-NNNNNN/ directory.

    Returns:
        List of parsed command dicts.
    """
    commands_dir = batch_dir / "commands"
    if not commands_dir.is_dir():
        return []
    commands = []
    for f in sorted(commands_dir.glob("*.json")):
        with open(f) as fh:
            commands.append(json.load(fh))
    return commands


def _parse_command_time(time_str: str | None) -> datetime | None:
    """Parse an ISO timestamp string from a command file."""
    if time_str is None:
        return None
    return datetime.fromisoformat(time_str.replace("Z", "+00:00"))


def annotate_stimulus(image_ts: datetime, commands: list[dict]) -> dict:
    """Build stimulus annotations for a single frame based on command timing.

    Rules:
      - electrical: annotate if image_ts falls within [start_time, end_time].
        Records current_ma, angle_degrees, frequency_hz from the first instruction.
      - chemical: annotate all images after start_time with a positive flag.
      - All other command types (vibration, temperature, camera) are ignored.
      - time_since_electrical_stimulus_onset: seconds since the earliest electrical
        command's start_time, or -1 if no electrical command exists in the batch.

    Args:
        image_ts: Timestamp of the image frame.
        commands: List of command dicts from load_commands().

    Returns:
        Dict with stimulus annotations.
    """
    annotations: dict = {
        "electrical": {},
        "chemical": {},
        "time_since_electrical_stimulus_onset": -1.0,
    }

    # Find earliest electrical start time for the onset field.
    # -1 if no electrical command exists or frame is before stimulus onset.
    electrical_starts = []
    for cmd in commands:
        if cmd.get("type") == "electrical":
            start = _parse_command_time(cmd.get("start_time"))
            if start:
                electrical_starts.append(start)

    if electrical_starts:
        earliest_start = min(electrical_starts)
        if image_ts >= earliest_start:
            annotations["time_since_electrical_stimulus_onset"] = (
                image_ts - earliest_start
            ).total_seconds()

    for cmd in commands:
        cmd_type = cmd.get("type", "")
        start = _parse_command_time(cmd.get("start_time"))
        end = _parse_command_time(cmd.get("end_time"))
        instructions = cmd.get("instructions", [])
        inst = instructions[0] if instructions else {}

        if cmd_type == "electrical":
            if start and end and start <= image_ts <= end:
                annotations["electrical"] = {
                    "active": True,
                    "current_ma": inst.get("current_ma", 0.0),
                    "angle_degrees": inst.get("angle_degrees", 0.0),
                    "frequency_hz": inst.get("frequency_hz", 0.0),
                }

        elif cmd_type == "chemical":
            if start and image_ts >= start:
                annotations["chemical"] = {
                    "active": True,
                }

    return annotations


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


def _resize_one_batch(
    batch_id: int,
    batch_dir: Path,
    files: list[Path],
    output_dir: Path,
    target_size: int,
    shard_size: int,
) -> tuple[int, int, list[str], float]:
    """Process a single batch: resize images and write to shards.

    Runs in a worker process. Each batch writes shards to its own subdirectory
    to avoid filename collisions between parallel workers.

    Returns:
        (batch_id, num_samples, shard_paths, elapsed_seconds)
    """
    t0 = time.monotonic()
    batch_output_dir = output_dir / f"batch-{batch_id:06d}"

    resize_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
    ])

    # Raw images are 9152x6944 (wider than tall). Crop to a 6944x6944 square
    # by removing 1358px from the left edge and 850px from the right edge.
    # PIL crop box: (left, upper, right, lower)
    CROP_BOX = (1358, 0, 8302, 6944)  # 8302 = 9152 - 850

    commands = load_commands(batch_dir)
    batch_start_ts = parse_image_timestamp(files[0]) if files else None

    with ShardWriter(batch_output_dir, shard_size=shard_size) as writer:
        for frame_idx, img_path in enumerate(files):
            try:
                image = Image.open(img_path).convert("RGB")
                image = image.crop(CROP_BOX)
                image = resize_transform(image)
                image_ts = parse_image_timestamp(img_path)
                stimulus = annotate_stimulus(image_ts, commands)
                metadata = {
                    "filename": img_path.name,
                    "batch_id": batch_id,
                    "frame_index": frame_idx,
                    "timestamp": image_ts.isoformat(),
                    "time_since_batch_start": (image_ts - batch_start_ts).total_seconds(),
                    "stimulus": stimulus,
                }
                key = f"b{batch_id:06d}_f{frame_idx:06d}"
                writer.write(key=key, image=image, metadata=metadata)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to process %s", img_path
                )
                continue

    elapsed = time.monotonic() - t0
    return batch_id, writer.total_samples_written, writer.shard_paths, elapsed


def preprocess_resize(
    cfg: DictConfig,
    batch_groups: list[tuple[int, Path, list[Path]]],
    output_dir: Path,
) -> DatasetManifest:
    """Resize images and write to WebDataset tar shards, parallelized by batch.

    Each batch is processed in a separate worker process. Results are logged
    as batches complete.

    Args:
        cfg: Full Hydra config.
        batch_groups: List of (batch_id, batch_dir, sorted_file_list) tuples.
        output_dir: Directory to write tar shards to.

    Returns:
        DatasetManifest describing the output.
    """
    target_size = cfg.data.preprocessing.target_size
    shard_size = cfg.data.shard_size
    # Use Slurm-allocated CPUs if available, otherwise fall back to config
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", cfg.data.num_workers))
    total_batches = len(batch_groups)

    logger.info(
        "Preprocessing %d batches across %d workers", total_batches, num_workers
    )

    all_shard_paths: list[str] = []
    total_samples = 0
    completed = 0
    wall_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _resize_one_batch,
                batch_id, batch_dir, files,
                output_dir, target_size, shard_size,
            ): batch_id
            for batch_id, batch_dir, files in batch_groups
        }

        for future in as_completed(futures):
            batch_id, n_samples, shard_paths, elapsed = future.result()
            all_shard_paths.extend(shard_paths)
            total_samples += n_samples
            completed += 1

            wall_elapsed = time.monotonic() - wall_start
            avg_per_batch = wall_elapsed / completed
            remaining = (total_batches - completed) * avg_per_batch
            eta_min, eta_sec = divmod(remaining, 60)

            logger.info(
                "[%3d/%d] Batch %06d: %d samples (%.1fs) | ETA %dm%02ds",
                completed, total_batches, batch_id, n_samples, elapsed,
                int(eta_min), int(eta_sec),
            )

    manifest = DatasetManifest(
        config_hash=compute_config_hash(cfg),
        source_dir=str(cfg.data.biosense_archive_path),
        processed_dir=str(output_dir),
        num_samples=total_samples,
        format="webdataset",
        shard_paths=sorted(all_shard_paths),
    )
    return manifest


def preprocess_autoencoder(
    cfg: DictConfig, batch_groups: list[tuple[int, Path, list[Path]]], output_dir: Path
) -> DatasetManifest:
    """Encode images with a pretrained autoencoder and save latent vectors to HDF5.

    Args:
        cfg: Full Hydra config.
        batch_groups: List of (batch_id, sorted_file_list) tuples.
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

    # Discover batches and their files
    batch_ids = list(cfg.data.batches) if cfg.data.batches else None
    batch_dirs = discover_batch_dirs(raw_dir, batches=batch_ids)

    if not batch_dirs:
        logger.warning("No batch directories found.")
        return

    batch_groups = []
    total_files = 0
    for batch_id, batch_dir in batch_dirs:
        files = discover_batch_files(batch_dir)
        files = partition_files(files)
        if files:
            batch_groups.append((batch_id, batch_dir, files))
            total_files += len(files)

    if not batch_groups:
        logger.warning("No files to process.")
        return

    mode = cfg.data.preprocessing.mode
    logger.info(
        "Running preprocessing in '%s' mode: %d batches, %d files",
        mode, len(batch_groups), total_files,
    )

    t0 = time.monotonic()

    if mode == "resize":
        manifest = preprocess_resize(cfg, batch_groups, output_dir)
    elif mode == "autoencoder":
        manifest = preprocess_autoencoder(cfg, batch_groups, output_dir)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    elapsed = time.monotonic() - t0
    minutes, seconds = divmod(elapsed, 60)
    manifest.save(output_dir / "manifest.json")
    logger.info(
        "Preprocessing complete. %d samples in %dm%02ds",
        manifest.num_samples, int(minutes), int(seconds),
    )
