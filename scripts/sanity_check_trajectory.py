#!/usr/bin/env python3
"""Sanity check: overfit trajectory models on a single sequence.

Verifies the training pipeline works by memorizing one sequence.
Loss should approach ~0 within 50 epochs.

Usage:
    python scripts/sanity_check_trajectory.py --model gru
    python scripts/sanity_check_trajectory.py --model mlp
"""

import argparse
import logging
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from biosense_ml.models.trajectory_gru import TrajectoryGRU
from biosense_ml.models.trajectory_mlp import TrajectoryMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gru", "mlp"], required=True)
    parser.add_argument("--h5", default="data/trajectory/trajectory_dataset.h5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_idx", type=int, default=0,
                        help="Which sequence to overfit on (picks first with >=100 steps)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load a single long sequence
    with h5py.File(args.h5, "r") as f:
        centroid_xy = f["centroid_xy"][:]
        velocity_xy = f["velocity_xy"][:]
        dt = f["dt"][:]
        actions = f["actions"][:]
        seq_starts = f["sequence_starts"][:]
        seq_lengths = f["sequence_lengths"][:]

    # Find a sequence with >= 100 steps
    long_seqs = [(i, int(seq_lengths[i])) for i in range(len(seq_starts)) if seq_lengths[i] >= 100]
    if not long_seqs:
        logger.error("No sequences with >= 100 steps")
        return
    idx, length = long_seqs[args.seq_idx]
    start = int(seq_starts[idx])
    logger.info("Using sequence %d (batch start=%d, length=%d)", idx, start, length)

    # Build feature matrix for this sequence
    features = np.concatenate([
        centroid_xy[start:start+length],
        velocity_xy[start:start+length],
        dt[start:start+length, None],
        actions[start:start+length],
    ], axis=1).astype(np.float32)

    targets = centroid_xy[start:start+length].astype(np.float32)

    # Input: features[:-1], Target: targets[1:]
    feat_t = torch.from_numpy(features[:-1]).unsqueeze(0).to(device)  # (1, T-1, 10)
    tgt_t = torch.from_numpy(targets[1:]).unsqueeze(0).to(device)     # (1, T-1, 2)

    T = feat_t.shape[1]
    context_len = 10

    # Build model
    if args.model == "gru":
        cfg = OmegaConf.create({
            "name": "trajectory_gru", "input_dim": 10, "hidden_dim": 64,
            "output_dim": 2, "num_layers": 1, "dropout": 0.0,
        })
        model = TrajectoryGRU(cfg).to(device)
    else:
        cfg = OmegaConf.create({
            "name": "trajectory_mlp", "input_dim": 10, "hidden_dim": 64,
            "output_dim": 2, "context_len": context_len, "num_layers": 2, "dropout": 0.0,
        })
        model = TrajectoryMLP(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%d params)", args.model, n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        if args.model == "gru":
            pred, _ = model(feat_t)  # (1, T-1, 2)
            loss = nn.functional.mse_loss(pred, tgt_t)
        else:
            # Slide windows
            windows = []
            window_targets = []
            for t in range(context_len, T):
                windows.append(feat_t[0, t - context_len : t])
                window_targets.append(tgt_t[0, t - 1])
            windows = torch.stack(windows)  # (N, K, 10)
            window_targets = torch.stack(window_targets)  # (N, 2)
            pred = model(windows)  # (N, 2)
            loss = nn.functional.mse_loss(pred, window_targets)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d | loss=%.8f", epoch, loss.item())

    logger.info("Final loss: %.8f", loss.item())
    if loss.item() < 1e-4:
        logger.info("PASS — loss < 1e-4, model can memorize a single sequence")
    else:
        logger.warning("WARN — loss=%.6f, expected < 1e-4 for single-sequence overfit", loss.item())


if __name__ == "__main__":
    main()
