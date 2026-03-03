"""Entry point for model training."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.training.trainer import Trainer

    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
