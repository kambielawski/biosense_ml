"""Multi-GPU / DDP utilities for distributed training on Slurm clusters."""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed process group from Slurm environment variables.

    Returns:
        Tuple of (rank, world_size, local_rank).
    """
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        # Set MASTER_ADDR and MASTER_PORT from Slurm if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        logger.info("Initialized distributed: rank %d/%d, local_rank %d", rank, world_size, local_rank)
    else:
        logger.info("Running in single-process mode")

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Destroy the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    return not dist.is_initialized() or dist.get_rank() == 0
