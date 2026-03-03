"""Image transforms and augmentations for training and validation."""

import logging

from omegaconf import DictConfig
from torchvision import transforms

logger = logging.getLogger(__name__)


def get_transforms(cfg: DictConfig, split: str = "train") -> transforms.Compose:
    """Build a torchvision transform pipeline from config.

    Args:
        cfg: Full config (uses cfg.data.preprocessing and cfg.model.input_size).
        split: One of "train" or "val". Training includes augmentation.

    Returns:
        A composed transform pipeline.
    """
    preproc = cfg.data.preprocessing
    img_size = cfg.model.input_size

    mean = list(preproc.get("mean", [0.485, 0.456, 0.406]))
    std = list(preproc.get("std", [0.229, 0.224, 0.225]))

    if split == "train":
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        transform_list = [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    logger.debug("Built %s transforms with %d stages", split, len(transform_list))
    return transforms.Compose(transform_list)
