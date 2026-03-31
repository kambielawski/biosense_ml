"""Tests for the gaze evaluation harness."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from biosense_ml.evaluation.gaze_eval import (
    GazeRolloutMetrics,
    evaluate_dataset,
    evaluate_rollout,
)
from biosense_ml.models.gaze_dynamics import GazeDynamics
from biosense_ml.pipeline.gaze_dataset import GazeSequenceDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_cfg(**overrides):
    defaults = {
        "name": "gaze_dynamics",
        "latent_dim": 32,
        "hidden_dim": 64,
        "context_len": 4,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _make_sequence(T: int = 30, moving: bool = True) -> dict[str, torch.Tensor]:
    """Create a synthetic sequence dict matching GazeSequenceDataset output."""
    crops = torch.randn(T, 3, 32, 32) * 0.5
    centers = torch.zeros(T, 2)
    if moving:
        centers = torch.cumsum(torch.randn(T, 2) * 2, dim=0) + 256.0
    else:
        centers[:] = 256.0
    deltas = torch.zeros(T, 2)
    deltas[1:] = centers[1:] - centers[:-1]
    has_motion = torch.ones(T, dtype=torch.bool)
    has_motion[0] = False
    timestamps = torch.arange(T, dtype=torch.float64) * 5.0

    return {
        "crops": crops,
        "centers": centers,
        "deltas": deltas,
        "has_motion": has_motion,
        "timestamps": timestamps,
        "batch_key": "batch_000000",
        "length": T,
    }


def _make_h5(path: Path, n_batches: int = 3, frames_per_batch: int = 30):
    splits = ["train", "val", "test"]
    with h5py.File(path, "w") as f:
        for i in range(n_batches):
            T = frames_per_batch
            grp = f.create_group(f"batch_{i:06d}")
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
            grp.attrs["split"] = splits[i % len(splits)]


@pytest.fixture
def model_cfg():
    return _make_model_cfg()


@pytest.fixture
def model(model_cfg):
    return GazeDynamics(model_cfg)


@pytest.fixture
def h5_path(tmp_path):
    path = tmp_path / "gaze_eval_test.h5"
    _make_h5(path)
    return str(path)


# ---------------------------------------------------------------------------
# GazeRolloutMetrics
# ---------------------------------------------------------------------------

class TestGazeRolloutMetrics:
    def test_default_values(self):
        m = GazeRolloutMetrics()
        assert m.crop_l1 == 0.0
        assert m.ade == 0.0
        assert m.fde == 0.0
        assert m.hde == {}
        assert m.rollout_length == 0
        assert m.diverged is False
        assert m.baseline_crop_l1 == 0.0

    def test_custom_values(self):
        m = GazeRolloutMetrics(
            crop_l1=0.5, ade=10.0, fde=15.0,
            hde={10.0: 5.0, 30.0: 8.0},
            rollout_length=50, diverged=False,
            baseline_crop_l1=0.8, velocity_error=2.0,
        )
        assert m.crop_l1 == 0.5
        assert m.ade == 10.0
        assert m.hde[10.0] == 5.0
        assert m.velocity_error == 2.0


# ---------------------------------------------------------------------------
# evaluate_rollout
# ---------------------------------------------------------------------------

class TestEvaluateRollout:
    def test_returns_metrics(self, model):
        seq = _make_sequence(T=30)
        metrics = evaluate_rollout(model, seq, context_len=4, burn_in_frac=0.2)
        assert metrics.rollout_length > 0
        assert metrics.crop_l1 > 0
        assert metrics.baseline_crop_l1 > 0
        assert np.isfinite(metrics.ade)
        assert np.isfinite(metrics.fde)

    def test_short_sequence_returns_empty(self, model):
        seq = _make_sequence(T=5)
        metrics = evaluate_rollout(model, seq, context_len=4, burn_in_frac=0.5)
        assert metrics.rollout_length == 0

    def test_hde_computed_for_reachable_horizons(self, model):
        # T=30, timestamps 5s apart -> max time = ~125s
        seq = _make_sequence(T=30)
        metrics = evaluate_rollout(
            model, seq, context_len=4, burn_in_frac=0.2,
            time_horizons=[10.0, 30.0, 60.0, 300.0],
        )
        # 10s and 30s should be reachable, 300s probably not
        assert metrics.rollout_length > 0
        if 10.0 in metrics.hde:
            assert np.isfinite(metrics.hde[10.0])

    def test_velocity_error_computed(self, model):
        seq = _make_sequence(T=30)
        metrics = evaluate_rollout(model, seq, context_len=4, burn_in_frac=0.2)
        assert metrics.rollout_length > 1
        assert metrics.velocity_error >= 0

    def test_rollout_length_matches_expected(self, model):
        T = 30
        seq = _make_sequence(T=T)
        burn_in = max(int(T * 0.2), 4)  # 6
        expected_H = T - burn_in - 1  # 23
        metrics = evaluate_rollout(model, seq, context_len=4, burn_in_frac=0.2)
        assert metrics.rollout_length == expected_H

    def test_crop_l1_is_positive(self, model):
        seq = _make_sequence(T=30)
        metrics = evaluate_rollout(model, seq, context_len=4, burn_in_frac=0.2)
        assert metrics.crop_l1 > 0

    def test_baseline_crop_l1_is_positive(self, model):
        seq = _make_sequence(T=30)
        metrics = evaluate_rollout(model, seq, context_len=4, burn_in_frac=0.2)
        assert metrics.baseline_crop_l1 > 0


# ---------------------------------------------------------------------------
# evaluate_dataset
# ---------------------------------------------------------------------------

class TestEvaluateDataset:
    def test_aggregated_metrics(self, model):
        sequences = [_make_sequence(T=30) for _ in range(5)]
        results = evaluate_dataset(model, sequences, context_len=4, burn_in_frac=0.2)
        assert results["num_sequences"] == 5
        assert "ade_median" in results
        assert "ade_mean" in results
        assert "fde_median" in results
        assert "crop_l1_mean" in results
        assert "baseline_crop_l1_mean" in results
        assert "crop_improvement_pct" in results
        assert "divergence_rate" in results

    def test_empty_sequences(self, model):
        results = evaluate_dataset(model, [], context_len=4)
        assert results["num_sequences"] == 0

    def test_all_short_sequences(self, model):
        sequences = [_make_sequence(T=3) for _ in range(3)]
        results = evaluate_dataset(model, sequences, context_len=4, burn_in_frac=0.5)
        assert results["num_sequences"] == 0

    def test_ade_iqr_format(self, model):
        sequences = [_make_sequence(T=30) for _ in range(5)]
        results = evaluate_dataset(model, sequences, context_len=4, burn_in_frac=0.2)
        iqr = results["ade_iqr"]
        assert isinstance(iqr, tuple)
        assert len(iqr) == 2
        assert iqr[0] <= iqr[1]

    def test_divergence_rate_zero_for_valid(self, model):
        sequences = [_make_sequence(T=30) for _ in range(3)]
        results = evaluate_dataset(model, sequences, context_len=4, burn_in_frac=0.2)
        assert results["divergence_rate"] == 0.0

    def test_length_weighted_ade(self, model):
        sequences = [_make_sequence(T=30) for _ in range(3)]
        results = evaluate_dataset(model, sequences, context_len=4, burn_in_frac=0.2)
        assert "ade_weighted" in results
        assert np.isfinite(results["ade_weighted"])

    def test_hde_aggregation(self, model):
        # Use long enough sequences with small enough time intervals
        sequences = [_make_sequence(T=50) for _ in range(3)]
        results = evaluate_dataset(
            model, sequences, context_len=4, burn_in_frac=0.2,
            time_horizons=[10.0, 30.0],
        )
        # At least the 10s horizon should be reachable (T=50, dt=5s -> 200s after burn-in)
        if "hde_10s_median" in results:
            assert np.isfinite(results["hde_10s_median"])
            assert results["hde_10s_count"] > 0


# ---------------------------------------------------------------------------
# Integration: eval with real dataset loading
# ---------------------------------------------------------------------------

class TestEvalIntegration:
    def test_evaluate_from_h5(self, model, h5_path):
        """End-to-end: load dataset, evaluate, get results."""
        eval_ds = GazeSequenceDataset(h5_path, split="test")
        assert len(eval_ds) > 0

        sequences = [eval_ds[i] for i in range(len(eval_ds))]
        results = evaluate_dataset(
            model, sequences, context_len=4, burn_in_frac=0.2,
        )
        assert results["num_sequences"] > 0
        assert results["ade_median"] > 0
        assert results["crop_l1_mean"] > 0

    def test_checkpoint_round_trip(self, model, model_cfg, tmp_path):
        """Save and reload checkpoint, then evaluate."""
        from biosense_ml.utils.checkpoint import save_checkpoint, load_checkpoint

        ckpt_path = tmp_path / "test_ckpt.pt"
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        cfg = OmegaConf.create({"model": OmegaConf.to_container(model_cfg), "training": {}})

        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=5, best_metric=0.1, cfg=cfg)

        # Reload into a new model
        model2 = GazeDynamics(model_cfg)
        load_checkpoint(ckpt_path, model2)

        # Evaluate both — should get identical results
        seq = _make_sequence(T=30)
        m1 = evaluate_rollout(model, seq, context_len=4, burn_in_frac=0.2)
        m2 = evaluate_rollout(model2, seq, context_len=4, burn_in_frac=0.2)
        assert abs(m1.ade - m2.ade) < 1e-5
        assert abs(m1.crop_l1 - m2.crop_l1) < 1e-5
