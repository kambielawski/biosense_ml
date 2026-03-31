#!/usr/bin/env python3
"""Generate full-frame 512×512 rollout videos from a trained gaze dynamics model.

Composites predicted 32×32 crops onto a static background frame with feathered
alpha blending. Optionally produces side-by-side GT vs predicted videos.

Usage:
    python scripts/generate_gaze_rollout.py \
        --checkpoint outputs/gaze_dynamics/checkpoints/checkpoint_best.pt \
        --h5 data/gaze/gaze_dataset.h5 \
        --archive_dir /users/k/k/kkannans/scratch/biosense_training_data \
        --output_dir outputs/gaze_dynamics/rollout_videos \
        --split test \
        --num_videos 3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from biosense_ml.evaluation.gaze_eval import evaluate_rollout
from biosense_ml.models.gaze_dynamics import GazeDynamics
from biosense_ml.pipeline.compositing import (
    build_alpha_mask,
    denormalize_crop,
    generate_rollout_video,
    load_background_frame,
)
from biosense_ml.pipeline.gaze_dataset import GazeSequenceDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def find_batch_dir(archive_dir: Path, batch_key: str) -> Path | None:
    """Resolve a batch directory from a batch key like 'batch_000163'.

    Tries common naming patterns: batch-NNNNNN, batch_NNNNNN.
    """
    # Extract numeric part
    num_str = batch_key.replace("batch_", "").replace("batch-", "")
    for pattern in [f"batch-{num_str}", f"batch_{num_str}"]:
        candidate = archive_dir / pattern
        if candidate.is_dir():
            return candidate
    return None


def generate_single_video(
    model: torch.nn.Module,
    sequence: dict[str, torch.Tensor],
    archive_dir: Path | None,
    output_path: Path,
    context_len: int,
    burn_in_frac: float,
    fps: int,
    side_by_side: bool,
) -> dict | None:
    """Generate one rollout video for a single sequence.

    Returns:
        Dict with metrics and path, or None on failure.
    """
    crops = sequence["crops"]  # (T, 3, 32, 32)
    centers = sequence["centers"]  # (T, 2)
    T = crops.shape[0]
    burn_in = max(int(T * burn_in_frac), context_len)
    H = T - burn_in - 1

    if H < 1:
        logger.warning("Sequence too short for rollout (T=%d, burn_in=%d)", T, burn_in)
        return None

    device = next(model.parameters()).device
    context = crops[burn_in - context_len : burn_in].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_crops, pred_deltas = model.rollout(context, horizon=H)

    pred_crops_np = pred_crops.squeeze(0).cpu().numpy()  # (H, 3, 32, 32)
    pred_deltas_np = pred_deltas.squeeze(0).cpu().numpy()  # (H, 2)

    # Compute predicted centers via cumulative deltas
    start_center = centers[burn_in].numpy()
    pred_centers_np = start_center + np.cumsum(pred_deltas_np, axis=0)  # (H, 2)

    # Ground truth for comparison
    gt_crops_np = crops[burn_in + 1 : burn_in + 1 + H].numpy()
    gt_centers_np = centers[burn_in + 1 : burn_in + 1 + H].numpy()

    # Load or synthesize background frame
    background = None
    batch_key = sequence.get("batch_key", "unknown")

    if archive_dir is not None:
        batch_dir = find_batch_dir(archive_dir, batch_key)
        if batch_dir is not None:
            try:
                # Use the burn-in frame as background
                background = load_background_frame(batch_dir, frame_index=burn_in)
                logger.info("  Loaded background from %s (frame %d)", batch_dir.name, burn_in)
            except Exception as e:
                logger.warning("  Failed to load background from archive: %s", e)

    if background is None:
        # Fallback: create a dark gray background
        logger.info("  Using synthetic background (no archive frame available)")
        background = np.full((512, 512, 3), 40, dtype=np.uint8)

    # Generate video
    if side_by_side:
        generate_rollout_video(
            background=background,
            pred_crops=pred_crops_np,
            pred_centers=pred_centers_np,
            output_path=output_path,
            fps=fps,
            gt_crops=gt_crops_np,
            gt_centers=gt_centers_np,
        )
    else:
        generate_rollout_video(
            background=background,
            pred_crops=pred_crops_np,
            pred_centers=pred_centers_np,
            output_path=output_path,
            fps=fps,
        )

    # Compute metrics for this sequence
    metrics = evaluate_rollout(
        model=model,
        sequence=sequence,
        context_len=context_len,
        burn_in_frac=burn_in_frac,
    )

    logger.info(
        "  Video saved: %s (%d frames, ADE=%.2f, CropL1=%.4f)",
        output_path.name, H, metrics.ade, metrics.crop_l1,
    )

    return {
        "path": str(output_path),
        "batch_key": batch_key,
        "rollout_length": H,
        "ade": metrics.ade,
        "fde": metrics.fde,
        "crop_l1": metrics.crop_l1,
        "baseline_crop_l1": metrics.baseline_crop_l1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate full-frame gaze rollout videos"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--h5", type=str, default="data/gaze/gaze_dataset.h5",
        help="Path to HDF5 gaze dataset",
    )
    parser.add_argument(
        "--archive_dir", type=str, default=None,
        help="Path to raw image archive (for background frames)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/gaze_dynamics/rollout_videos",
        help="Output directory for videos",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to generate videos for",
    )
    parser.add_argument(
        "--num_videos", type=int, default=3,
        help="Number of videos to generate",
    )
    parser.add_argument(
        "--burn_in_frac", type=float, default=0.2,
        help="Burn-in fraction of each sequence",
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Output video framerate",
    )
    parser.add_argument(
        "--side_by_side", action="store_true",
        help="Generate GT (left) vs Pred (right) side-by-side videos",
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="W&B project name for logging",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and model
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

    # Load dataset
    eval_ds = GazeSequenceDataset(args.h5, split=args.split)
    if len(eval_ds) == 0:
        logger.error("No sequences found in %s split", args.split)
        sys.exit(1)

    num_videos = min(args.num_videos, len(eval_ds))
    logger.info("Generating %d rollout videos from %s split (%d sequences total)",
                num_videos, args.split, len(eval_ds))

    archive_dir = Path(args.archive_dir) if args.archive_dir else None

    # Init W&B
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"rollout-gaze_{time.strftime('%Y%m%d_%H%M%S')}",
                tags=["rollout-gaze-v1", "gaze_dynamics"],
                config={
                    "checkpoint": args.checkpoint,
                    "epoch": ckpt["epoch"],
                    "split": args.split,
                    "num_videos": num_videos,
                    "burn_in_frac": args.burn_in_frac,
                    "fps": args.fps,
                    "side_by_side": args.side_by_side,
                },
            )
        except Exception as e:
            logger.warning("W&B init failed: %s", e)

    # Generate videos
    all_results = []
    for i in range(num_videos):
        seq = eval_ds[i]
        batch_key = seq.get("batch_key", f"seq_{i}")
        logger.info("\n[%d/%d] Generating video for %s...", i + 1, num_videos, batch_key)

        video_path = output_dir / f"rollout_{args.split}_{batch_key}.mp4"

        result = generate_single_video(
            model=model,
            sequence=seq,
            archive_dir=archive_dir,
            output_path=video_path,
            context_len=context_len,
            burn_in_frac=args.burn_in_frac,
            fps=args.fps,
            side_by_side=args.side_by_side,
        )

        if result is not None:
            all_results.append(result)
            if wandb_run:
                try:
                    import wandb
                    wandb.log({
                        f"rollout/{batch_key}/ade": result["ade"],
                        f"rollout/{batch_key}/fde": result["fde"],
                        f"rollout/{batch_key}/crop_l1": result["crop_l1"],
                        f"rollout/{batch_key}/video": wandb.Video(
                            result["path"], fps=args.fps, format="mp4"
                        ),
                    })
                except Exception as e:
                    logger.warning("W&B logging failed for %s: %s", batch_key, e)

    # Summary
    if all_results:
        ades = [r["ade"] for r in all_results]
        crop_l1s = [r["crop_l1"] for r in all_results]
        logger.info("\n=== Summary ===")
        logger.info("Videos generated: %d", len(all_results))
        logger.info("Mean ADE: %.2f", np.mean(ades))
        logger.info("Mean Crop L1: %.4f", np.mean(crop_l1s))

        if wandb_run:
            try:
                import wandb
                wandb.log({
                    "rollout/mean_ade": np.mean(ades),
                    "rollout/mean_crop_l1": np.mean(crop_l1s),
                    "rollout/num_videos": len(all_results),
                })
            except Exception:
                pass

    if wandb_run:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    logger.info("\nDone. Videos saved to %s", output_dir)


if __name__ == "__main__":
    main()
