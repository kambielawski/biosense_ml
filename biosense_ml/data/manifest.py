"""Dataset manifest for tracking preprocessing runs and ensuring reproducibility."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class DatasetManifest:
    """Records metadata about a preprocessing run for reproducibility."""

    config_hash: str
    source_dir: str
    processed_dir: str
    num_samples: int
    format: str  # "webdataset" or "hdf5"
    shard_paths: list[str] = field(default_factory=list)
    hdf5_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save(self, path: Path) -> None:
        """Save manifest to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Saved manifest to %s", path)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        """Load manifest from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def compute_config_hash(cfg: DictConfig) -> str:
    """Compute a deterministic hash of the preprocessing config for cache invalidation."""
    config_str = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
