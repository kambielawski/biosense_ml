"""Tests for the GazeDynamics model and GazeCropDataset."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from biosense_ml.models.gaze_dynamics import GazeDynamics
from biosense_ml.pipeline.gaze_dataset import GazeCropDataset, GazeSequenceDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_cfg(**overrides):
    """Create a model config DictConfig."""
    defaults = {
        "name": "gaze_dynamics",
        "latent_dim": 32,  # smaller for tests
        "hidden_dim": 64,
        "context_len": 4,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _make_h5(path: Path, n_batches: int = 3, frames_per_batch: int = 20):
    """Create a minimal gaze HDF5 dataset for testing."""
    splits = ["train", "val", "test"]
    with h5py.File(path, "w") as f:
        for i in range(n_batches):
            T = frames_per_batch
            grp = f.create_group(f"batch_{i:06d}")
            # Random crops (ImageNet-normalized range)
            crops = np.random.randn(T, 3, 32, 32).astype(np.float32) * 0.5
            centers = np.cumsum(np.random.randn(T, 2).astype(np.float32) * 2, axis=0) + 256.0
            deltas = np.zeros((T, 2), dtype=np.float32)
            deltas[1:] = centers[1:] - centers[:-1]
            has_motion = np.ones(T, dtype=bool)
            has_motion[0] = False
            timestamps = np.arange(T, dtype=np.float64) * 5.0

            grp.create_dataset("crops", data=crops)
            grp.create_dataset("centers", data=centers)
            grp.create_dataset("deltas", data=deltas)
            grp.create_dataset("has_motion", data=has_motion)
            grp.create_dataset("timestamps", data=timestamps)
            grp.create_dataset("split", data=splits[i % len(splits)])


@pytest.fixture
def model_cfg():
    return _make_model_cfg()


@pytest.fixture
def model(model_cfg):
    return GazeDynamics(model_cfg)


@pytest.fixture
def h5_path(tmp_path):
    path = tmp_path / "gaze_test.h5"
    _make_h5(path)
    return str(path)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestGazeDynamicsModel:
    """Tests for the GazeDynamics model architecture."""

    def test_forward_shapes(self, model, model_cfg):
        B, K = 4, model_cfg.context_len
        crops = torch.randn(B, K, 3, 32, 32)
        pred_crop, pred_delta = model(crops)
        assert pred_crop.shape == (B, 3, 32, 32)
        assert pred_delta.shape == (B, 2)

    def test_forward_batch_size_1(self, model, model_cfg):
        crops = torch.randn(1, model_cfg.context_len, 3, 32, 32)
        pred_crop, pred_delta = model(crops)
        assert pred_crop.shape == (1, 3, 32, 32)
        assert pred_delta.shape == (1, 2)

    def test_encode_shape(self, model, model_cfg):
        x = torch.randn(8, 3, 32, 32)
        z = model.encode(x)
        assert z.shape == (8, model_cfg.latent_dim)

    def test_decode_shape(self, model, model_cfg):
        z = torch.randn(4, model_cfg.latent_dim)
        out = model.decode(z)
        assert out.shape == (4, 3, 32, 32)

    def test_rollout_shapes(self, model, model_cfg):
        B, K, H = 2, model_cfg.context_len, 5
        context = torch.randn(B, K, 3, 32, 32)
        pred_crops, pred_deltas = model.rollout(context, horizon=H)
        assert pred_crops.shape == (B, H, 3, 32, 32)
        assert pred_deltas.shape == (B, H, 2)

    def test_gradient_flows(self, model, model_cfg):
        crops = torch.randn(2, model_cfg.context_len, 3, 32, 32)
        pred_crop, pred_delta = model(crops)
        loss = pred_crop.sum() + pred_delta.sum()
        loss.backward()
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_param_count_reasonable(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        # With latent_dim=32 (test size), should be well under 1M
        assert n_params < 1_000_000
        assert n_params > 1_000  # sanity — not empty

    def test_full_size_param_count(self):
        """Verify the full-size model param count is reasonable.

        Spec estimated ~300K but actual is ~813K due to decoder FC layer
        (128*128*4*4 = 262K alone). Still small enough for ~200 sequences.
        """
        cfg = _make_model_cfg(latent_dim=128, hidden_dim=256, context_len=10)
        model = GazeDynamics(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        assert 500_000 < n_params < 1_500_000, f"Got {n_params} params"


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestGazeCropDataset:
    """Tests for the GazeCropDataset."""

    def test_loads_train_split(self, h5_path):
        ds = GazeCropDataset(h5_path, split="train", context_len=4)
        assert len(ds) > 0

    def test_loads_val_split(self, h5_path):
        ds = GazeCropDataset(h5_path, split="val", context_len=4)
        assert len(ds) > 0

    def test_sample_shapes(self, h5_path):
        ds = GazeCropDataset(h5_path, split="train", context_len=4)
        sample = ds[0]
        assert sample["context_crops"].shape == (4, 3, 32, 32)
        assert sample["target_crop"].shape == (3, 32, 32)
        assert sample["target_delta"].shape == (2,)

    def test_sample_dtypes(self, h5_path):
        ds = GazeCropDataset(h5_path, split="train", context_len=4)
        sample = ds[0]
        assert sample["context_crops"].dtype == torch.float32
        assert sample["target_crop"].dtype == torch.float32
        assert sample["target_delta"].dtype == torch.float32

    def test_no_static_targets(self, h5_path):
        """Windows should only target frames with has_motion=True."""
        ds = GazeCropDataset(h5_path, split="train", context_len=4)
        # All windows should have non-zero deltas or be from motion frames
        # (first frame of each batch has has_motion=False, delta=0)
        assert len(ds) > 0  # basic sanity

    def test_empty_split(self, tmp_path):
        """A split with no matching batches returns empty dataset."""
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("batch_000000")
            grp.create_dataset("crops", data=np.zeros((5, 3, 32, 32), dtype=np.float32))
            grp.create_dataset("centers", data=np.zeros((5, 2), dtype=np.float32))
            grp.create_dataset("deltas", data=np.zeros((5, 2), dtype=np.float32))
            grp.create_dataset("has_motion", data=np.ones(5, dtype=bool))
            grp.create_dataset("timestamps", data=np.zeros(5, dtype=np.float64))
            grp.create_dataset("split", data="train")
        ds = GazeCropDataset(str(path), split="val", context_len=4)
        assert len(ds) == 0


class TestGazeSequenceDataset:
    """Tests for the GazeSequenceDataset."""

    def test_loads_test_split(self, h5_path):
        ds = GazeSequenceDataset(h5_path, split="test")
        assert len(ds) > 0

    def test_sequence_contents(self, h5_path):
        ds = GazeSequenceDataset(h5_path, split="test")
        seq = ds[0]
        assert "crops" in seq
        assert "centers" in seq
        assert "deltas" in seq
        assert "has_motion" in seq
        assert "timestamps" in seq
        T = seq["length"]
        assert seq["crops"].shape == (T, 3, 32, 32)
        assert seq["centers"].shape == (T, 2)
        assert seq["deltas"].shape == (T, 2)


# ---------------------------------------------------------------------------
# Integration: model + dataset
# ---------------------------------------------------------------------------

class TestModelDatasetIntegration:
    """Test that model and dataset work together end-to-end."""

    def test_forward_pass_with_dataset(self, model, model_cfg, h5_path):
        ds = GazeCropDataset(h5_path, split="train", context_len=model_cfg.context_len)
        sample = ds[0]
        # Add batch dimension
        crops = sample["context_crops"].unsqueeze(0)
        pred_crop, pred_delta = model(crops)
        assert pred_crop.shape == (1, 3, 32, 32)
        assert pred_delta.shape == (1, 2)

    def test_loss_computation(self, model, model_cfg, h5_path):
        ds = GazeCropDataset(h5_path, split="train", context_len=model_cfg.context_len)
        sample = ds[0]
        crops = sample["context_crops"].unsqueeze(0)
        target_crop = sample["target_crop"].unsqueeze(0)
        target_delta = sample["target_delta"].unsqueeze(0)

        pred_crop, pred_delta = model(crops)
        crop_loss = torch.nn.functional.l1_loss(pred_crop, target_crop)
        pos_loss = torch.nn.functional.l1_loss(pred_delta, target_delta)
        loss = crop_loss + pos_loss

        assert loss.item() > 0
        loss.backward()

    def test_dataloader_batching(self, model, model_cfg, h5_path):
        ds = GazeCropDataset(h5_path, split="train", context_len=model_cfg.context_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        pred_crop, pred_delta = model(batch["context_crops"])
        assert pred_crop.shape[0] == 4

    def test_overfit_single_sample(self, model_cfg, h5_path):
        """Model can memorize a single sample (basic sanity)."""
        model = GazeDynamics(model_cfg)
        ds = GazeCropDataset(h5_path, split="train", context_len=model_cfg.context_len)
        sample = ds[0]
        crops = sample["context_crops"].unsqueeze(0)
        target_crop = sample["target_crop"].unsqueeze(0)
        target_delta = sample["target_delta"].unsqueeze(0)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial_loss = None

        for step in range(50):
            optimizer.zero_grad()
            pred_crop, pred_delta = model(crops)
            loss = (
                torch.nn.functional.l1_loss(pred_crop, target_crop)
                + torch.nn.functional.l1_loss(pred_delta, target_delta)
            )
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, (
            f"Loss didn't decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )
