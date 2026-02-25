"""Benchmark runners for all evaluation tasks.

Aggregates GO prediction, PPI prediction, stability prediction, and SFT
evaluations into a single ``run_all_benchmarks`` entry point.  Results are
returned as a flat ``Dict[str, float]`` with task-specific prefixes
(``go_``, ``ppi_``, ``stability_``, ``sft_``).
"""

import logging
import math
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from .go_prediction import evaluate_go
from .ppi_prediction import evaluate_ppi
from .sft_eval import evaluate_sft
from .stability import evaluate_stability

log = logging.getLogger(__name__)


def run_all_benchmarks(
    cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
    model=None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run all evaluation benchmarks and aggregate results.

    Each task is run independently -- a failure in one task does not prevent
    the remaining tasks from executing.  Results from each task are prefixed
    with the task name so that keys never collide.

    Args:
        cfg: Hydra configuration.
        checkpoint_path: Path to model checkpoint.
        model: Pre-loaded model instance (shared across all tasks).

    Returns:
        Dictionary of all metric names to values, with task prefixes.
    """
    log.info("Running all benchmarks...")

    results: Dict[str, Any] = {}
    task_results: Dict[str, Dict[str, Any]] = {}

    # ---- GO term prediction ----
    try:
        go_results = evaluate_go(cfg, checkpoint_path, model=model, output_dir=output_dir)
        task_results["go"] = go_results
        results.update({f"go_{k}": v for k, v in go_results.items()})
        log.info(f"GO prediction: {len(go_results)} metrics computed")
    except NotImplementedError:
        log.warning("GO prediction not implemented, skipping")
    except Exception as e:
        log.error(f"GO prediction failed: {e}", exc_info=True)

    # ---- PPI prediction ----
    try:
        ppi_results = evaluate_ppi(cfg, checkpoint_path, model=model, output_dir=output_dir)
        task_results["ppi"] = ppi_results
        results.update({f"ppi_{k}": v for k, v in ppi_results.items()})
        log.info(f"PPI prediction: {len(ppi_results)} metrics computed")
    except NotImplementedError:
        log.warning("PPI prediction not implemented, skipping")
    except Exception as e:
        log.error(f"PPI prediction failed: {e}", exc_info=True)

    # ---- Stability prediction ----
    try:
        stability_results = evaluate_stability(cfg, checkpoint_path, model=model, output_dir=output_dir)
        task_results["stability"] = stability_results
        results.update({f"stability_{k}": v for k, v in stability_results.items()})
        log.info(f"Stability prediction: {len(stability_results)} metrics computed")
    except NotImplementedError:
        log.warning("Stability prediction not implemented, skipping")
    except Exception as e:
        log.error(f"Stability prediction failed: {e}", exc_info=True)

    # ---- SFT evaluation (perplexity / BLEU / ROUGE) ----
    try:
        sft_results = evaluate_sft(cfg, checkpoint_path, model=model, output_dir=output_dir)
        task_results["sft"] = sft_results
        results.update({f"sft_{k}": v for k, v in sft_results.items()})
        log.info(f"SFT evaluation: {len(sft_results)} metrics computed")
    except NotImplementedError:
        log.warning("SFT evaluation not implemented, skipping")
    except Exception as e:
        log.error(f"SFT evaluation failed: {e}", exc_info=True)

    # ---- Aggregated logging ----
    log.info(f"All benchmarks complete: {len(results)} total metrics across {len(task_results)} tasks")

    # Log to wandb if configured
    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("wandb", {}).get("enabled", False):
        _log_aggregated_to_wandb(results, task_results)

    return results


def _log_aggregated_to_wandb(
    results: Dict[str, Any],
    task_results: Dict[str, Dict[str, Any]],
) -> None:
    """Log aggregated benchmark results to Weights & Biases."""
    try:
        import wandb

        # Log flat results dict
        wandb_safe = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                if isinstance(v, float) and math.isnan(v):
                    continue
                wandb_safe[f"benchmark/{k}"] = v
        if wandb_safe:
            wandb.log(wandb_safe)

        # Log per-task summary table
        summary_rows = []
        for task_name, task_metrics in task_results.items():
            row = {"task": task_name}
            for mk, mv in task_metrics.items():
                if isinstance(mv, (int, float)) and not (isinstance(mv, float) and math.isnan(mv)):
                    row[mk] = mv
            summary_rows.append(row)

        if summary_rows:
            # Build columns from union of all keys
            all_cols = list(dict.fromkeys(
                k for row in summary_rows for k in row.keys()
            ))
            table = wandb.Table(
                columns=all_cols,
                data=[[row.get(c, None) for c in all_cols] for row in summary_rows],
            )
            wandb.log({"benchmark_summary": table})

    except ImportError:
        log.warning("wandb not installed, skipping aggregated wandb logging")
    except Exception as e:
        log.warning(f"Failed to log aggregated results to wandb: {e}")
