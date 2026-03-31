#!/usr/bin/env python3
"""Evaluate trained gaze dynamics model on val and test splits.

Computes rollout metrics (Crop L1, ADE, FDE, HDE, divergence rate)
and generates crop filmstrip visualizations.

Usage:
    python scripts/eval_gaze.py \
        --checkpoint outputs/gaze_dynamics/checkpoints/checkpoint_best.pt \
        --h5 data/gaze/gaze_dataset.h5 \
        --output_dir outputs/gaze_dynamics/eval
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import wandb
from biosense_ml.evaluation.gaze_eval import evaluate_dataset
from biosense_ml.models.gaze_dynamics import GazeDynamics
from biosense_ml.pipeline.gaze_dataset import GazeSequenceDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def generate_filmstrip(
    model: torch.nn.Module,
    sequence: dict[str, torch.Tensor],
    context_len: int,
    burn_in_frac: float,
    output_path: Path,
    every_n: int = 10,
) -> None:
    """Generate side-by-side GT vs predicted crop filmstrip.

    Saves a PNG showing GT crops (top) and predicted crops (bottom)
    at regular intervals throughout the rollout.

    Args:
        model: GazeDynamics model.
        sequence: Sequence dict from GazeSequenceDataset.
        context_len: Context window size.
        burn_in_frac: Fraction for burn-in.
        output_path: Where to save the filmstrip image.
        every_n: Show every Nth rollout step.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping filmstrip generation")
        return

    crops = sequence["crops"]  # (T, 3, 32, 32)
    T = crops.shape[0]
    burn_in = max(int(T * burn_in_frac), context_len)
    H = T - burn_in - 1

    if H < 1:
        return

    device = next(model.parameters()).device
    context = crops[burn_in - context_len : burn_in].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_crops, _ = model.rollout(context, horizon=H)

    pred_crops_np = pred_crops.squeeze(0).cpu().numpy()  # (H, 3, 32, 32)
    gt_crops_np = crops[burn_in + 1 : burn_in + 1 + H].numpy()

    # Select frames at regular intervals
    indices = list(range(0, H, every_n))
    if len(indices) > 20:
        indices = indices[:20]  # cap at 20 frames
    n_frames = len(indices)

    if n_frames == 0:
        return

    # ImageNet denormalization for visualization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 1.5, 3.5))
    if n_frames == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        gt_img = np.clip(gt_crops_np[idx] * std + mean, 0, 1).transpose(1, 2, 0)
        pred_img = np.clip(pred_crops_np[idx] * std + mean, 0, 1).transpose(1, 2, 0)

        axes[0, col].imshow(gt_img)
        axes[0, col].set_title(f"t+{idx}", fontsize=7)
        axes[0, col].axis("off")

        axes[1, col].imshow(pred_img)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("GT", fontsize=9)
    axes[1, 0].set_ylabel("Pred", fontsize=9)

    batch_key = sequence.get("batch_key", "unknown")
    fig.suptitle(f"Crop Filmstrip: {batch_key}", fontsize=10)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved filmstrip: %s", output_path)


def generate_overlay_video(
    model: torch.nn.Module,
    sequence: dict[str, torch.Tensor],
    context_len: int,
    burn_in_frac: float,
    output_path: Path,
) -> None:
    """Generate overlay visualization: predicted crop bbox on GT frame centers.

    Saves a PNG with predicted vs GT trajectory overlaid on the sequence.

    Args:
        model: GazeDynamics model.
        sequence: Sequence dict from GazeSequenceDataset.
        context_len: Context window size.
        burn_in_frac: Fraction for burn-in.
        output_path: Where to save the overlay image.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        logger.warning("matplotlib not available — skipping overlay generation")
        return

    crops = sequence["crops"]
    centers = sequence["centers"].numpy()  # (T, 2)
    deltas = sequence["deltas"].numpy()
    T = crops.shape[0]
    burn_in = max(int(T * burn_in_frac), context_len)
    H = T - burn_in - 1

    if H < 1:
        return

    device = next(model.parameters()).device
    context = crops[burn_in - context_len : burn_in].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        _, pred_deltas = model.rollout(context, horizon=H)

    pred_deltas_np = pred_deltas.squeeze(0).cpu().numpy()  # (H, 2)
    start_center = centers[burn_in]
    pred_centers = start_center + np.cumsum(pred_deltas_np, axis=0)
    gt_centers = centers[burn_in + 1 : burn_in + 1 + H]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot trajectories
    ax.plot(gt_centers[:, 1], gt_centers[:, 0], "g-", linewidth=1.5, label="GT", alpha=0.8)
    ax.plot(pred_centers[:, 1], pred_centers[:, 0], "r--", linewidth=1.5, label="Pred", alpha=0.8)

    # Mark start and end
    ax.plot(gt_centers[0, 1], gt_centers[0, 0], "go", markersize=8, label="Start")
    ax.plot(gt_centers[-1, 1], gt_centers[-1, 0], "gs", markersize=8, label="GT End")
    ax.plot(pred_centers[-1, 1], pred_centers[-1, 0], "rs", markersize=8, label="Pred End")

    # Draw crop boxes at intervals
    for idx in range(0, H, max(1, H // 5)):
        for c, color in [(gt_centers[idx], "green"), (pred_centers[idx], "red")]:
            rect = Rectangle(
                (c[1] - 16, c[0] - 16), 32, 32,
                linewidth=0.5, edgecolor=color, facecolor="none", alpha=0.4,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)  # image coordinates
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    batch_key = sequence.get("batch_key", "unknown")
    ax.set_title(f"Trajectory Overlay: {batch_key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved overlay: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate gaze dynamics model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--h5", type=str, default="data/gaze/gaze_dataset.h5", help="Path to HDF5 dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/gaze_dynamics/eval", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Splits to evaluate")
    parser.add_argument("--burn_in_frac", type=float, default=0.2, help="Burn-in fraction")
    parser.add_argument("--filmstrips", type=int, default=3, help="Number of filmstrips to generate per split")
    parser.add_argument("--overlays", type=int, default=3, help="Number of overlay plots per split")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and reconstruct model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = OmegaConf.create(ckpt["config"]["model"])
    model = GazeDynamics(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    context_len = model_cfg.context_len
    logger.info(
        "Loaded checkpoint: epoch=%d, best_metric=%.6f, context_len=%d",
        ckpt["epoch"], ckpt["best_metric"], context_len,
    )

    # Init W&B
    if args.wandb_project:
        try:
            wandb.init(
                project=args.wandb_project,
                name=f"eval-gaze-v1_{time.strftime('%Y%m%d_%H%M%S')}",
                tags=["eval-gaze-v1", "gaze_dynamics"],
                config={
                    "checkpoint": args.checkpoint,
                    "epoch": ckpt["epoch"],
                    "burn_in_frac": args.burn_in_frac,
                    "context_len": context_len,
                },
            )
        except Exception as e:
            logger.warning("W&B init failed: %s", e)

    for split in args.splits:
        logger.info("\n=== %s ===", split.upper())

        eval_ds = GazeSequenceDataset(args.h5, split=split)
        if len(eval_ds) == 0:
            logger.info("  %s: no sequences", split)
            continue

        sequences = [eval_ds[i] for i in range(len(eval_ds))]

        results = evaluate_dataset(
            model=model,
            sequences=sequences,
            context_len=context_len,
            burn_in_frac=args.burn_in_frac,
            time_horizons=[10, 30, 60, 120, 300],
        )

        # Log results
        logger.info("  %s results:", split)
        for k in sorted(results.keys()):
            v = results[k]
            if isinstance(v, float):
                logger.info("    %s: %.6f", k, v)
            elif isinstance(v, int):
                logger.info("    %s: %d", k, v)
            elif isinstance(v, tuple):
                logger.info("    %s: (%.6f, %.6f)", k, v[0], v[1])

        # Log to W&B
        if wandb.run:
            wandb.log({f"{split}/{k}": v for k, v in results.items() if isinstance(v, (int, float))})

        # Generate filmstrips
        n_filmstrips = min(args.filmstrips, len(sequences))
        for i in range(n_filmstrips):
            filmstrip_path = output_dir / f"filmstrip_{split}_{sequences[i]['batch_key']}.png"
            generate_filmstrip(
                model, sequences[i], context_len, args.burn_in_frac,
                filmstrip_path, every_n=10,
            )
            if wandb.run:
                wandb.log({f"{split}/filmstrip_{i}": wandb.Image(str(filmstrip_path))})

        # Generate overlay plots
        n_overlays = min(args.overlays, len(sequences))
        for i in range(n_overlays):
            overlay_path = output_dir / f"overlay_{split}_{sequences[i]['batch_key']}.png"
            generate_overlay_video(
                model, sequences[i], context_len, args.burn_in_frac,
                overlay_path,
            )
            if wandb.run:
                wandb.log({f"{split}/overlay_{i}": wandb.Image(str(overlay_path))})

    if wandb.run:
        wandb.finish()

    logger.info("\nEvaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
