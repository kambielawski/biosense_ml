#!/usr/bin/env python3
"""Evaluate trained trajectory models on val and test splits.

Usage:
    python scripts/eval_trajectory.py
"""

import logging
import torch
from omegaconf import OmegaConf
from biosense_ml.models.trajectory_gru import TrajectoryGRU
from biosense_ml.models.trajectory_mlp import TrajectoryMLP
from biosense_ml.pipeline.trajectory_dataset import TrajectorySequenceDataset
from biosense_ml.evaluation.trajectory_eval import evaluate_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

H5_PATH = "data/trajectory/trajectory_dataset.h5"

MODELS = [
    {
        "name": "GRU",
        "cls": TrajectoryGRU,
        "ckpt": "outputs/trajectory_gru/checkpoints/checkpoint_epoch0100.pt",
        "cfg": {"name": "trajectory_gru", "input_dim": 10, "hidden_dim": 64,
                "output_dim": 2, "num_layers": 1, "dropout": 0.0},
        "model_type": "gru",
    },
    {
        "name": "MLP",
        "cls": TrajectoryMLP,
        "ckpt": "outputs/trajectory_mlp/checkpoints/checkpoint_epoch0100.pt",
        "cfg": {"name": "trajectory_mlp", "input_dim": 10, "hidden_dim": 64,
                "output_dim": 2, "context_len": 10, "num_layers": 2, "dropout": 0.0},
        "model_type": "mlp",
    },
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    for m in MODELS:
        logger.info("\n=== %s ===", m["name"])
        cfg = OmegaConf.create(m["cfg"])
        model = m["cls"](cfg).to(device)

        ckpt = torch.load(m["ckpt"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded: epoch=%d, best_metric=%.6f",
            ckpt["epoch"], ckpt["best_metric"],
        )

        for split in ["val", "test"]:
            eval_ds = TrajectorySequenceDataset(H5_PATH, split=split)
            if len(eval_ds) == 0:
                logger.info("  %s: no sequences", split)
                continue
            results = evaluate_dataset(
                model=model,
                sequences=[eval_ds[i] for i in range(len(eval_ds))],
                burn_in_frac=0.2,
                time_horizons=[10, 30, 60, 120, 300],
                model_type=m["model_type"],
                context_len=10,
            )
            logger.info("  %s:", split)
            for k in sorted(results.keys()):
                v = results[k]
                if isinstance(v, float):
                    logger.info("    %s: %.6f", k, v)
                elif isinstance(v, int):
                    logger.info("    %s: %d", k, v)
                elif isinstance(v, tuple):
                    logger.info("    %s: (%.6f, %.6f)", k, v[0], v[1])


if __name__ == "__main__":
    main()
