"""Smoke tests for the training pipeline."""

import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from biosense_ml.models import build_model
from biosense_ml.training.metrics import MetricTracker
from biosense_ml.utils.checkpoint import load_checkpoint, save_checkpoint


def _make_test_cfg(**overrides):
    """Create a minimal config for testing."""
    cfg = OmegaConf.create({
        "seed": 42,
        "project_name": "test",
        "output_dir": "/tmp/test_output",
        "model": {
            "name": "baseline",
            "architecture": "cnn",
            "hidden_dim": 32,
            "num_layers": 2,
            "dropout": 0.0,
            "input_type": "image",
            "input_channels": 3,
            "input_size": 32,
            "latent_dim": 64,
            "num_classes": 5,
        },
        "data": {
            "preprocessing": {"mode": "resize"},
            "batch_size": 4,
        },
        "training": {
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0, "betas": [0.9, 0.999]},
            "scheduler": {"name": "cosine", "warmup_epochs": 0, "min_lr": 1e-6},
            "loss": {"name": "cross_entropy"},
            "gradient_clip": 1.0,
            "mixed_precision": False,
            "epochs": 2,
            "log_every": 1,
            "val_every": 1,
            "checkpoint_every": 1,
            "keep_top_k": 2,
        },
    })
    for dotkey, value in overrides.items():
        OmegaConf.update(cfg, dotkey, value)
    return cfg


def test_model_forward_image():
    """Test that the baseline CNN produces correct output shape."""
    cfg = _make_test_cfg()
    model = build_model(cfg)

    batch = torch.randn(4, 3, 32, 32)
    output = model(batch)
    assert output.shape == (4, 5)


def test_model_forward_latent():
    """Test that the baseline MLP produces correct output shape."""
    cfg = _make_test_cfg(**{"model.input_type": "latent"})
    model = build_model(cfg)

    batch = torch.randn(4, 64)
    output = model(batch)
    assert output.shape == (4, 5)


def test_one_training_step():
    """Test a single forward + backward + optimizer step produces finite loss."""
    cfg = _make_test_cfg()
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 5, (4,))

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_checkpoint_roundtrip():
    """Test that saving and loading a checkpoint preserves model weights."""
    cfg = _make_test_cfg()
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=5, best_metric=0.42, cfg=cfg)

        # Load into a fresh model
        model2 = build_model(cfg)
        info = load_checkpoint(ckpt_path, model2)
        assert info["epoch"] == 5
        assert info["best_metric"] == 0.42

        # Verify weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


def test_metric_tracker():
    """Test MetricTracker computes correct running averages."""
    tracker = MetricTracker()
    tracker.update("loss", 1.0, n=2)
    tracker.update("loss", 2.0, n=3)
    # Weighted average: (1.0*2 + 2.0*3) / 5 = 8/5 = 1.6
    assert abs(tracker.average("loss") - 1.6) < 1e-6

    tracker.reset()
    assert tracker.average("loss") == 0.0
