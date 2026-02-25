"""Shared metrics utilities for all evaluation tasks.

This module provides common metric computation helpers used across
GO prediction, PPI prediction, and stability evaluation.  All public
functions return ``Dict[str, float]`` for consistent downstream handling.
"""

import logging
import math
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_float(value: Any) -> float:
    """Convert a value to float, returning 0.0 for non-numeric inputs."""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def sanitise_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Ensure every value in *metrics* is a plain float or int.

    Non-numeric values (lists, strings, NaN) are replaced with 0.0 or
    converted where possible so that the output is always
    ``Dict[str, float]`` safe for JSON / wandb / tensorboard.
    """
    clean: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = 0.0
            else:
                clean[k] = float(v)
        elif isinstance(v, (list, np.ndarray)):
            # Skip complex structures (e.g. confusion_matrix)
            continue
        else:
            continue
    return clean


# ---------------------------------------------------------------------------
# Multi-label helpers
# ---------------------------------------------------------------------------

def multilabel_aupr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute macro-averaged AUPR across labels with ground-truth support.

    Args:
        y_true: Binary ground-truth matrix (n_samples, n_labels).
        y_pred: Binary prediction matrix (n_samples, n_labels).

    Returns:
        Macro-averaged AUPR over labels that appear at least once in y_true.
    """
    if not SKLEARN_AVAILABLE:
        log.warning("sklearn not available; returning 0.0 for AUPR")
        return 0.0

    label_mask = y_true.sum(axis=0) > 0
    if not label_mask.any():
        return 0.0

    aupr_values = []
    n_labels = y_true.shape[1]
    for j in range(n_labels):
        if label_mask[j]:
            try:
                ap = average_precision_score(y_true[:, j], y_pred[:, j])
                if not np.isnan(ap):
                    aupr_values.append(ap)
            except Exception:
                pass

    return float(np.mean(aupr_values)) if aupr_values else 0.0


# ---------------------------------------------------------------------------
# Correlation helpers (work without scipy)
# ---------------------------------------------------------------------------

def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient, falling back to manual computation."""
    if len(y_true) < 2:
        return 0.0
    if SCIPY_AVAILABLE:
        try:
            r, _ = pearsonr(y_true, y_pred)
            return float(r) if not math.isnan(r) else 0.0
        except Exception:
            pass
    # Manual fallback
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    num = np.sum((y_true - mean_t) * (y_pred - mean_p))
    den = np.sqrt(np.sum((y_true - mean_t) ** 2) * np.sum((y_pred - mean_p) ** 2))
    return float(num / den) if den != 0 else 0.0


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation, falling back to manual computation."""
    if len(y_true) < 2:
        return 0.0
    if SCIPY_AVAILABLE:
        try:
            r, _ = spearmanr(y_true, y_pred)
            return float(r) if not math.isnan(r) else 0.0
        except Exception:
            pass
    # Manual fallback via rank-based Pearson
    def _rankdata(x: np.ndarray) -> np.ndarray:
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        return ranks.astype(float) + 1.0

    return pearson_correlation(_rankdata(y_true), _rankdata(y_pred))


# ---------------------------------------------------------------------------
# Wandb logging helper
# ---------------------------------------------------------------------------

def log_metrics_to_wandb(
    metrics: Dict[str, Any],
    prefix: str = "",
) -> None:
    """Attempt to log *metrics* to wandb under the given prefix.

    Silently skipped if wandb is not installed or not initialised.
    """
    try:
        import wandb
        if wandb.run is None:
            return
        payload = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                key = f"{prefix}/{k}" if prefix else k
                payload[key] = v
        if payload:
            wandb.log(payload)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"wandb logging failed: {e}")
