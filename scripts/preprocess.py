"""Entry point for data preprocessing."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from biosense_ml.pipeline.preprocessing import run_preprocessing

    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
