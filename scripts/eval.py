"""Entry point for model evaluation on held-out data."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO: Implement evaluation pipeline
    # This should:
    # 1. Load a trained model from checkpoint
    # 2. Load the validation/test dataset
    # 3. Run inference and compute metrics
    # 4. Log results to wandb and/or save to disk
    raise NotImplementedError(
        "Evaluation script is a stub. Implement your evaluation logic here."
    )


if __name__ == "__main__":
    main()
