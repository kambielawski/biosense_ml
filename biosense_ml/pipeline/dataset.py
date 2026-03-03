"""PyTorch Dataset classes and DataLoader factories for both image and latent modes."""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from biosense_ml.pipeline.webdataset_utils import make_webdataset_loader

logger = logging.getLogger(__name__)


class LatentDataset(Dataset):
    """Dataset for pre-computed latent vectors stored in HDF5 format.

    The HDF5 file is expected to have datasets:
        - "latents": shape (N, latent_dim), float32
        - "metadata": shape (N,), stored as JSON strings
        - "keys": shape (N,), sample identifiers
    """

    def __init__(self, hdf5_path: Path) -> None:
        self.hdf5_path = Path(hdf5_path)
        self._file: h5py.File | None = None
        # Read length without keeping file open (for multi-worker safety)
        with h5py.File(self.hdf5_path, "r") as f:
            self._length = len(f["latents"])
        logger.info("LatentDataset: %d samples from %s", self._length, self.hdf5_path)

    def _open(self) -> None:
        """Lazy-open HDF5 file (needed for multi-worker DataLoader)."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        self._open()
        latent = torch.from_numpy(np.array(self._file["latents"][idx], dtype=np.float32))
        metadata_str = self._file["metadata"][idx]
        if isinstance(metadata_str, bytes):
            metadata_str = metadata_str.decode("utf-8")
        metadata = json.loads(metadata_str)
        return latent, metadata


def make_image_dataloader(cfg: DictConfig, split: str = "train") -> DataLoader:
    """Create a DataLoader for the resize-images (WebDataset) path.

    Args:
        cfg: Full Hydra config.
        split: "train" or "val".

    Returns:
        DataLoader yielding (image_tensor, metadata_dict) batches.
    """
    return make_webdataset_loader(cfg, split=split)


def make_latent_dataloader(cfg: DictConfig, split: str = "train") -> DataLoader:
    """Create a DataLoader for the autoencoder (HDF5 latent) path.

    Args:
        cfg: Full Hydra config.
        split: "train" or "val".

    Returns:
        DataLoader yielding (latent_tensor, metadata_dict) batches.
    """
    processed_dir = Path(cfg.data.processed_data_dir)
    hdf5_path = processed_dir / "latents.h5"
    dataset = LatentDataset(hdf5_path)

    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
    )


def make_dataloader(cfg: DictConfig, split: str = "train") -> DataLoader:
    """Dispatcher: create the appropriate DataLoader based on cfg.data.format.

    Args:
        cfg: Full Hydra config.
        split: "train" or "val".

    Returns:
        DataLoader for the configured data format.
    """
    fmt = cfg.data.format
    if fmt == "webdataset":
        return make_image_dataloader(cfg, split=split)
    elif fmt == "hdf5":
        return make_latent_dataloader(cfg, split=split)
    else:
        raise ValueError(f"Unknown data format: {fmt}. Expected 'webdataset' or 'hdf5'.")
