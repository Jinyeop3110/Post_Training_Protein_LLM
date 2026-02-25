"""Checkpoint utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

log = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    path: Union[str, Path],
    filename: str = "checkpoint.pt",
) -> Path:
    """Save training checkpoint.

    Args:
        state: State dictionary to save.
        path: Directory or file path.
        filename: Filename if path is a directory.

    Returns:
        Path to saved checkpoint.
    """
    path = Path(path)

    if path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        filepath = path

    torch.save(state, filepath)
    log.info(f"Checkpoint saved to {filepath}")

    return filepath


def load_checkpoint(
    path: Union[str, Path],
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        map_location: Device to map tensors to.

    Returns:
        Loaded state dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=map_location, weights_only=True)
    log.info(f"Checkpoint loaded from {path}")

    return state
