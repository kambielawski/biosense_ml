#!/usr/bin/env python3
"""Generate trajectory prediction comparison plots (GRU vs MLP).

Produces:
1. XY scatter: predicted vs ground-truth centroid paths
2. Time series: x(t) and y(t) with predictions overlaid
3. Representative examples: best, worst, and stimulus-varied sequences
4. Summary error-over-time plot across all sequences

Usage:
    python scripts/plot_trajectory_eval.py \
        --h5 data/trajectory/trajectory_dataset.h5 \
        --output_dir outputs/trajectory_plots
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from biosense_ml.models.trajectory_gru import TrajectoryGRU
from biosense_ml.models.trajectory_mlp import TrajectoryMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_models(device):
    """Load both GRU and MLP models."""
    gru_cfg = OmegaConf.create({
        "name": "trajectory_gru", "input_dim": 10, "hidden_dim": 64,
        "output_dim": 2, "num_layers": 1, "dropout": 0.0,
    })
    gru = TrajectoryGRU(gru_cfg).to(device)
    ckpt = torch.load(
        "outputs/trajectory_gru/checkpoints/checkpoint_epoch0100.pt",
        map_location=device, weights_only=False,
    )
    gru.load_state_dict(ckpt["model_state_dict"])
    gru.eval()
    logger.info("GRU: epoch=%d, best_metric=%.6f", ckpt["epoch"], ckpt["best_metric"])

    mlp_cfg = OmegaConf.create({
        "name": "trajectory_mlp", "input_dim": 10, "hidden_dim": 64,
        "output_dim": 2, "context_len": 10, "num_layers": 2, "dropout": 0.0,
    })
    mlp = TrajectoryMLP(mlp_cfg).to(device)
    ckpt = torch.load(
        "outputs/trajectory_mlp/checkpoints/checkpoint_epoch0100.pt",
        map_location=device, weights_only=False,
    )
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp.eval()
    logger.info("MLP: epoch=%d, best_metric=%.6f", ckpt["epoch"], ckpt["best_metric"])

    return gru, mlp


def rollout_both(gru, mlp, features, dt, burn_in_frac=0.2, context_len=10):
    """Run autoregressive rollout for both models on one sequence.

    Returns dict with burn_in_xy, gt_xy, gru_pred, mlp_pred, cum_time, B_idx.
    """
    device = next(gru.parameters()).device
    T = features.shape[0]
    B_idx = max(int(T * burn_in_frac), context_len)
    H = T - B_idx - 1

    if H < 1:
        return None

    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    dt_t = torch.from_numpy(dt).to(device)

    with torch.no_grad():
        # GRU rollout
        gru_ctx = feat_t[:, :B_idx]
        future_actions = feat_t[:, B_idx : B_idx + H, 5:10]
        future_dt = dt_t[B_idx + 1 : B_idx + 1 + H].unsqueeze(0)
        gru_pred = gru.rollout(gru_ctx, future_actions, future_dt)

        # MLP rollout
        mlp_ctx = feat_t[:, B_idx - context_len : B_idx]
        mlp_pred = mlp.rollout(mlp_ctx, future_actions, future_dt)

    burn_in_xy = features[:B_idx, :2]
    gt_xy = features[B_idx : B_idx + H, :2]
    gru_pred_np = gru_pred.squeeze(0).cpu().numpy()
    mlp_pred_np = mlp_pred.squeeze(0).cpu().numpy()

    # Cumulative time from burn-in end
    cum_time = np.cumsum(dt[B_idx + 1 : B_idx + 1 + H])

    # ADE for ranking
    actual_H = min(len(gt_xy), len(gru_pred_np), len(mlp_pred_np))
    gru_ade = float(np.linalg.norm(gru_pred_np[:actual_H] - gt_xy[:actual_H], axis=1).mean())
    mlp_ade = float(np.linalg.norm(mlp_pred_np[:actual_H] - gt_xy[:actual_H], axis=1).mean())

    return {
        "burn_in_xy": burn_in_xy,
        "gt_xy": gt_xy[:actual_H],
        "gru_pred": gru_pred_np[:actual_H],
        "mlp_pred": mlp_pred_np[:actual_H],
        "cum_time": cum_time[:actual_H],
        "B_idx": B_idx,
        "gru_ade": gru_ade,
        "mlp_ade": mlp_ade,
    }


def plot_xy_trajectory(result, seq_idx, stim_pct, output_dir):
    """Plot predicted vs ground-truth XY paths for one sequence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, name, pred in [
        (axes[0], "GRU", result["gru_pred"]),
        (axes[1], "MLP", result["mlp_pred"]),
    ]:
        # Burn-in trail
        ax.plot(result["burn_in_xy"][:, 0], result["burn_in_xy"][:, 1],
                ".-", color="gray", alpha=0.4, markersize=1, linewidth=0.5, label="burn-in")
        # Ground truth
        ax.plot(result["gt_xy"][:, 0], result["gt_xy"][:, 1],
                ".-", color="green", alpha=0.7, markersize=1, linewidth=0.8, label="ground truth")
        # Prediction
        ax.plot(pred[:, 0], pred[:, 1],
                ".-", color="blue", alpha=0.7, markersize=1, linewidth=0.8, label="predicted")
        # Start/end markers
        ax.plot(result["gt_xy"][0, 0], result["gt_xy"][0, 1], "go", markersize=8, label="rollout start")
        ax.plot(result["gt_xy"][-1, 0], result["gt_xy"][-1, 1], "gx", markersize=8)
        ax.plot(pred[-1, 0], pred[-1, 1], "bx", markersize=8)

        ade = result[f"{name.lower()}_ade"]
        ax.set_title(f"{name} — ADE={ade:.4f}")
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sequence {seq_idx} — XY Trajectories (stim={stim_pct:.0f}%)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"xy_seq{seq_idx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(result, seq_idx, stim_pct, output_dir):
    """Plot x(t) and y(t) time series with predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex="col")

    t = result["cum_time"]

    for col, (name, pred) in enumerate([("GRU", result["gru_pred"]), ("MLP", result["mlp_pred"])]):
        for row, (coord, label) in enumerate([(0, "x"), (1, "y")]):
            ax = axes[row, col]
            ax.plot(t, result["gt_xy"][:, coord], "-", color="green", linewidth=1, label="ground truth")
            ax.plot(t, pred[:, coord], "-", color="blue", linewidth=1, alpha=0.8, label="predicted")
            ax.fill_between(
                t,
                result["gt_xy"][:, coord],
                pred[:, coord],
                alpha=0.15, color="red",
            )
            ax.set_ylabel(f"{label}(t)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ade = result[f"{name.lower()}_ade"]
                ax.set_title(f"{name} (ADE={ade:.4f})")

        axes[1, col].set_xlabel("Time (seconds)")

    fig.suptitle(f"Sequence {seq_idx} — Time Series (stim={stim_pct:.0f}%)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"ts_seq{seq_idx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_over_time(all_results, output_dir):
    """Plot displacement error vs time, averaged across sequences."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, pred_key) in [
        (axes[0], ("GRU", "gru_pred")),
        (axes[1], ("MLP", "mlp_pred")),
    ]:
        # Bin errors by time
        time_bins = np.arange(0, 901, 10)  # 0 to 900s in 10s bins
        bin_errors = [[] for _ in range(len(time_bins) - 1)]

        for r in all_results:
            if r is None:
                continue
            pred = r[pred_key]
            gt = r["gt_xy"]
            t = r["cum_time"]
            errs = np.linalg.norm(pred - gt, axis=1)

            # Skip if diverged
            if not np.all(np.isfinite(errs)):
                continue

            for i in range(len(time_bins) - 1):
                mask = (t >= time_bins[i]) & (t < time_bins[i + 1])
                if mask.any():
                    bin_errors[i].extend(errs[mask].tolist())

        # Compute median and IQR per bin
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        medians = []
        q25s = []
        q75s = []
        valid_centers = []

        for i, be in enumerate(bin_errors):
            if len(be) >= 3:
                arr = np.array(be)
                medians.append(np.median(arr))
                q25s.append(np.percentile(arr, 25))
                q75s.append(np.percentile(arr, 75))
                valid_centers.append(bin_centers[i])

        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        valid_centers = np.array(valid_centers)

        ax.plot(valid_centers, medians, "-", color="blue", linewidth=1.5, label="median")
        ax.fill_between(valid_centers, q25s, q75s, alpha=0.2, color="blue", label="IQR")
        ax.set_xlabel("Time since rollout start (seconds)")
        ax.set_ylabel("L2 displacement error")
        ax.set_title(f"{name} — Error vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Displacement Error Over Rollout Time (all test sequences)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_bar(all_results, output_dir):
    """Bar chart comparing GRU vs MLP on key metrics across all sequences."""
    gru_ades = [r["gru_ade"] for r in all_results if r is not None and np.isfinite(r["gru_ade"])]
    mlp_ades = [r["mlp_ade"] for r in all_results if r is not None and np.isfinite(r["mlp_ade"])]

    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["ADE (median)", "ADE (mean)", "ADE (75th pctl)"]
    gru_vals = [np.median(gru_ades), np.mean(gru_ades), np.percentile(gru_ades, 75)]
    mlp_vals = [np.median(mlp_ades), np.mean(mlp_ades), np.percentile(mlp_ades, 75)]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, gru_vals, width, label="GRU", color="coral", alpha=0.8)
    ax.bar(x + width / 2, mlp_vals, width, label="MLP", color="steelblue", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Displacement Error (normalized coords)")
    ax.set_title("GRU vs MLP — Test Set Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (gv, mv) in enumerate(zip(gru_vals, mlp_vals)):
        ax.text(i - width / 2, gv + 0.005, f"{gv:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, mv + 0.005, f"{mv:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="data/trajectory/trajectory_dataset.h5")
    parser.add_argument("--output_dir", default="outputs/trajectory_plots")
    parser.add_argument("--split", default="test")
    parser.add_argument("--burn_in_frac", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    gru, mlp = load_models(device)

    # Load dataset
    with h5py.File(args.h5, "r") as f:
        centroid_xy = f["centroid_xy"][:]
        velocity_xy = f["velocity_xy"][:]
        dt = f["dt"][:]
        actions = f["actions"][:]
        seq_starts = f["sequence_starts"][:]
        seq_lengths = f["sequence_lengths"][:]
        split_assignments = f["split_assignments"][:]
        norm_stats = json.loads(f.attrs["norm_stats"])

    split_bytes = args.split.encode("utf-8")

    # Collect all test sequences
    seq_indices = []
    for i in range(len(seq_starts)):
        if split_assignments[i].strip() == split_bytes:
            seq_indices.append(i)

    logger.info("Found %d %s sequences", len(seq_indices), args.split)

    # Run rollouts on all test sequences
    all_results = []
    stim_pcts = []

    for seq_idx in seq_indices:
        start = int(seq_starts[seq_idx])
        length = int(seq_lengths[seq_idx])
        if length < 20:
            all_results.append(None)
            stim_pcts.append(0.0)
            continue

        features = np.concatenate([
            centroid_xy[start : start + length],
            velocity_xy[start : start + length],
            dt[start : start + length, None],
            actions[start : start + length],
        ], axis=1).astype(np.float32)

        dt_seq = dt[start : start + length].astype(np.float32)

        # Stimulus fraction (action[0] = stimulus on/off)
        stim_pct = float(actions[start : start + length, 0].sum()) / length * 100
        stim_pcts.append(stim_pct)

        result = rollout_both(gru, mlp, features, dt_seq, burn_in_frac=args.burn_in_frac)
        all_results.append(result)

    # Rank by MLP ADE to find best/worst
    ranked = [(i, r, stim_pcts[i]) for i, r in enumerate(all_results)
              if r is not None and np.isfinite(r["mlp_ade"])]
    ranked.sort(key=lambda x: x[1]["mlp_ade"])

    logger.info("Valid rollouts: %d/%d", len(ranked), len(all_results))

    # --- Plot 1: Summary bar chart ---
    plot_model_comparison_bar(all_results, output_dir)
    logger.info("Saved model_comparison.png")

    # --- Plot 2: Error over time ---
    plot_error_over_time(all_results, output_dir)
    logger.info("Saved error_vs_time.png")

    # --- Plot 3-5: Best 3 MLP predictions ---
    for rank, (idx, result, stim) in enumerate(ranked[:3]):
        si = seq_indices[idx]
        plot_xy_trajectory(result, si, stim, output_dir)
        plot_time_series(result, si, stim, output_dir)
        logger.info("Best #%d: seq %d, MLP ADE=%.4f, stim=%.0f%%",
                     rank + 1, si, result["mlp_ade"], stim)

    # --- Plot 6-8: Worst 3 MLP predictions ---
    for rank, (idx, result, stim) in enumerate(ranked[-3:]):
        si = seq_indices[idx]
        plot_xy_trajectory(result, si, stim, output_dir)
        plot_time_series(result, si, stim, output_dir)
        logger.info("Worst #%d: seq %d, MLP ADE=%.4f, stim=%.0f%%",
                     rank + 1, si, result["mlp_ade"], stim)

    # --- Plot 9-10: High vs low stimulus sequences ---
    high_stim = [(i, r, s) for i, r, s in ranked if s > 50]
    low_stim = [(i, r, s) for i, r, s in ranked if s < 10]

    if high_stim:
        # Pick median-performing high-stim sequence
        mid = len(high_stim) // 2
        idx, result, stim = high_stim[mid]
        si = seq_indices[idx]
        plot_xy_trajectory(result, si, stim, output_dir)
        plot_time_series(result, si, stim, output_dir)
        logger.info("High-stim example: seq %d, MLP ADE=%.4f, stim=%.0f%%",
                     si, result["mlp_ade"], stim)

    if low_stim:
        mid = len(low_stim) // 2
        idx, result, stim = low_stim[mid]
        si = seq_indices[idx]
        plot_xy_trajectory(result, si, stim, output_dir)
        plot_time_series(result, si, stim, output_dir)
        logger.info("Low-stim example: seq %d, MLP ADE=%.4f, stim=%.0f%%",
                     si, result["mlp_ade"], stim)

    # Print summary table
    logger.info("\n=== Summary ===")
    logger.info("Sequences plotted:")
    logger.info("  Best 3 (by MLP ADE): %s", [seq_indices[r[0]] for r in ranked[:3]])
    logger.info("  Worst 3 (by MLP ADE): %s", [seq_indices[r[0]] for r in ranked[-3:]])
    logger.info("  High-stim count: %d, Low-stim count: %d", len(high_stim), len(low_stim))
    logger.info("Plots saved to: %s", output_dir)


if __name__ == "__main__":
    main()
