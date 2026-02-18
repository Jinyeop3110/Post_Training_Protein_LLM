"""Distributed training utilities."""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> int:
    """Initialize distributed training.

    Args:
        backend: Distributed backend (nccl, gloo, etc.).
        init_method: URL for initialization.

    Returns:
        Local rank.
    """
    if not dist.is_initialized():
        if "RANK" in os.environ:
            # Launched with torchrun or similar
            dist.init_process_group(backend=backend, init_method=init_method)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # Single GPU
            local_rank = 0
    else:
        local_rank = dist.get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    log.info(f"Initialized distributed training: rank={local_rank}")
    return local_rank


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
