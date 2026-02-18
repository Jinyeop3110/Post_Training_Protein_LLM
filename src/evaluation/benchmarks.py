"""Benchmark runners."""

import logging
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from .go_prediction import evaluate_go
from .ppi_prediction import evaluate_ppi
from .stability import evaluate_stability

log = logging.getLogger(__name__)


def run_all_benchmarks(
    cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """Run all evaluation benchmarks.

    Args:
        cfg: Hydra configuration.
        checkpoint_path: Path to model checkpoint.

    Returns:
        Dictionary of all metric names to values.
    """
    log.info("Running all benchmarks...")

    results = {}

    # GO term prediction
    try:
        go_results = evaluate_go(cfg, checkpoint_path)
        results.update({f"go_{k}": v for k, v in go_results.items()})
    except NotImplementedError:
        log.warning("GO prediction not implemented, skipping")

    # PPI prediction
    try:
        ppi_results = evaluate_ppi(cfg, checkpoint_path)
        results.update({f"ppi_{k}": v for k, v in ppi_results.items()})
    except NotImplementedError:
        log.warning("PPI prediction not implemented, skipping")

    # Stability prediction
    try:
        stability_results = evaluate_stability(cfg, checkpoint_path)
        results.update({f"stability_{k}": v for k, v in stability_results.items()})
    except NotImplementedError:
        log.warning("Stability prediction not implemented, skipping")

    return results
