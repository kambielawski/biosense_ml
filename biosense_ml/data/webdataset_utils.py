"""Utilities for writing and reading WebDataset tar shards."""

import io
import json
import logging
import tarfile
from pathlib import Path

import torch
import webdataset as wds
from omegaconf import DictConfig
from PIL import Image

from biosense_ml.data.transforms import get_transforms

logger = logging.getLogger(__name__)


class ShardWriter:
    """Context manager that accumulates samples and writes them as tar shards.

    Usage:
        with ShardWriter(output_dir, shard_size=1000) as writer:
            writer.write(key="sample_0001", image=pil_img, metadata={"label": 3})
    """

    def __init__(self, output_dir: Path, shard_size: int = 1000) -> None:
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._shard_idx = 0
        self._sample_count = 0
        self._total_count = 0
        self._current_tar: tarfile.TarFile | None = None
        self._shard_paths: list[str] = []

    def __enter__(self) -> "ShardWriter":
        self._open_new_shard()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._close_current_shard()

    def _open_new_shard(self) -> None:
        shard_name = f"shard-{self._shard_idx:06d}.tar"
        shard_path = self.output_dir / shard_name
        self._current_tar = tarfile.open(shard_path, "w")
        self._shard_paths.append(str(shard_path))
        logger.debug("Opened new shard: %s", shard_name)

    def _close_current_shard(self) -> None:
        if self._current_tar is not None:
            self._current_tar.close()
            self._current_tar = None
            logger.debug(
                "Closed shard %d with %d samples", self._shard_idx, self._sample_count
            )
            self._shard_idx += 1
            self._sample_count = 0

    def _add_member(self, name: str, data: bytes) -> None:
        """Add a named file to the current tar archive."""
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self._current_tar.addfile(info, io.BytesIO(data))

    def write(self, key: str, image: Image.Image, metadata: dict) -> None:
        """Write a single sample (image + metadata) to the current shard.

        Args:
            key: Unique sample identifier.
            image: PIL Image to save as JPEG.
            metadata: Dict of metadata to save as JSON.
        """
        if self._sample_count >= self.shard_size:
            self._close_current_shard()
            self._open_new_shard()

        # Write image as JPEG
        img_buf = io.BytesIO()
        image.save(img_buf, format="JPEG", quality=95)
        self._add_member(f"{key}.jpg", img_buf.getvalue())

        # Write metadata as JSON
        meta_bytes = json.dumps(metadata).encode("utf-8")
        self._add_member(f"{key}.json", meta_bytes)

        self._sample_count += 1
        self._total_count += 1

    @property
    def total_samples_written(self) -> int:
        return self._total_count

    @property
    def shard_paths(self) -> list[str]:
        return list(self._shard_paths)


def make_webdataset_loader(
    cfg: DictConfig,
    split: str = "train",
) -> torch.utils.data.DataLoader:
    """Create a DataLoader backed by WebDataset tar shards.

    Args:
        cfg: Full Hydra config.
        split: "train" or "val" — controls shuffle and transforms.

    Returns:
        A DataLoader yielding (image_tensor, metadata_dict) batches.
    """
    processed_dir = Path(cfg.data.processed_data_dir)
    shard_pattern = str(processed_dir / "shard-{000000..999999}.tar")

    transform = get_transforms(cfg, split=split)

    def decode_sample(sample: dict) -> tuple[torch.Tensor, dict]:
        image = sample["jpg"]
        metadata = json.loads(sample["json"])
        image = transform(image)
        return image, metadata

    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=(split == "train"))
        .decode("pil")
        .map(decode_sample)
    )

    if split == "train":
        dataset = dataset.shuffle(cfg.data.shuffle_buffer)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=_collate_with_metadata,
    )
    return loader


def _collate_with_metadata(
    batch: list[tuple[torch.Tensor, dict]],
) -> tuple[torch.Tensor, list[dict]]:
    """Custom collate that stacks images and collects metadata as a list of dicts."""
    images, metadatas = zip(*batch)
    return torch.stack(images), list(metadatas)
