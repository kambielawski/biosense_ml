"""Smoke tests for the data pipeline."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from biosense_ml.pipeline.dataset import LatentDataset
from biosense_ml.pipeline.manifest import DatasetManifest, compute_config_hash
from biosense_ml.pipeline.webdataset_utils import ShardWriter


def _make_fake_images(tmpdir: Path, n: int = 10) -> list[Path]:
    """Create fake image + metadata files for testing."""
    img_dir = tmpdir / "raw"
    img_dir.mkdir()
    paths = []
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img_path = img_dir / f"sample_{i:04d}.jpg"
        img.save(img_path)
        meta_path = img_dir / f"sample_{i:04d}.json"
        with open(meta_path, "w") as f:
            json.dump({"label": i % 3, "filename": img_path.name}, f)
        paths.append(img_path)
    return paths


def test_shard_writer_roundtrip():
    """Test that ShardWriter creates tar shards with the expected number of samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images = _make_fake_images(tmpdir, n=5)

        output_dir = tmpdir / "shards"
        with ShardWriter(output_dir, shard_size=3) as writer:
            for i, img_path in enumerate(images):
                img = Image.open(img_path)
                writer.write(key=f"s_{i:04d}", image=img, metadata={"label": i % 3})

        assert writer.total_samples_written == 5
        # Should have 2 shards (3 + 2)
        shards = list(output_dir.glob("shard-*.tar"))
        assert len(shards) == 2


def test_latent_dataset():
    """Test LatentDataset reads HDF5 files correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "latents.h5"
        n_samples = 20
        latent_dim = 64

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("latents", data=np.random.randn(n_samples, latent_dim).astype(np.float32))
            metadata = [json.dumps({"label": i % 5}) for i in range(n_samples)]
            f.create_dataset("metadata", data=metadata)
            f.create_dataset("keys", data=[f"sample_{i}" for i in range(n_samples)])

        dataset = LatentDataset(hdf5_path)
        assert len(dataset) == n_samples

        latent, meta = dataset[0]
        assert latent.shape == (latent_dim,)
        assert isinstance(meta, dict)
        assert "label" in meta


def test_manifest_roundtrip():
    """Test that DatasetManifest saves and loads correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = DatasetManifest(
            config_hash="abc123",
            source_dir="/data/raw",
            processed_dir="/data/processed",
            num_samples=1000,
            format="webdataset",
            shard_paths=["/data/processed/shard-000000.tar"],
        )
        path = Path(tmpdir) / "manifest.json"
        manifest.save(path)

        loaded = DatasetManifest.load(path)
        assert loaded.config_hash == "abc123"
        assert loaded.num_samples == 1000
        assert loaded.format == "webdataset"


def test_config_hash_deterministic():
    """Test that config hashing is deterministic."""
    cfg = OmegaConf.create({"a": 1, "b": "hello", "c": [1, 2, 3]})
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2
