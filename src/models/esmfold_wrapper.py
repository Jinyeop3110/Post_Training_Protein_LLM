"""
ESMFold Wrapper for Structural Quality Prediction

Provides lazy-loaded, cached ESMFold predictions (pLDDT, pTM) for use
as GRPO reward signals. The model is loaded on first predict() call
and results are cached to avoid re-folding the same protein.

Usage:
    predictor = get_esmfold_predictor()
    result = predictor.predict("MKTAYIAK...")
    print(result["plddt"])  # e.g., 78.5
    print(result["ptm"])    # e.g., 0.82
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Module-level singleton
_esmfold_predictor: Optional["ESMFoldPredictor"] = None


class ESMFoldPredictor:
    """
    Wrapper around ESMFold for structure prediction.

    Features:
    - Lazy loading: model loaded on first predict() call
    - LRU caching: caches predictions keyed by sequence
    - Returns pLDDT (0-100) and pTM (0-1) scores

    Args:
        device: Device to run ESMFold on. Default: "cuda".
        cache_size: Maximum number of cached predictions. Default: 1000.
        max_sequence_length: Maximum sequence length for folding. Default: 1024.
    """

    def __init__(
        self,
        device: str = "cuda",
        cache_size: int = 1000,
        max_sequence_length: int = 1024,
    ) -> None:
        self.device = device
        self.cache_size = cache_size
        self.max_sequence_length = max_sequence_length
        self.model = None
        self.tokenizer = None
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _load_model(self) -> None:
        """Lazy load ESMFold model on first use."""
        if self.model is not None:
            return

        try:
            from transformers import AutoTokenizer, EsmForProteinFolding

            logger.info("Loading ESMFold model (facebook/esmfold_v1)...")
            self.model = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1"
            ).to(self.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            logger.info("ESMFold model loaded and frozen")

        except ImportError:
            raise ImportError(
                "ESMFold requires transformers with ESM support. "
                "Install with: pip install transformers"
            )

    @torch.no_grad()
    def predict(self, sequence: str) -> Dict[str, Any]:
        """
        Predict structural quality for a protein sequence.

        Args:
            sequence: Amino acid sequence string.

        Returns:
            Dictionary containing:
                - plddt: Mean pLDDT score (0-100), higher = more confident
                - ptm: pTM score (0-1), higher = better topology
                - plddt_per_residue: Per-residue pLDDT tensor [L]
                - sequence_length: Length of the input sequence
                - truncated: Whether sequence was truncated
        """
        # Check cache
        if sequence in self._cache:
            self._cache.move_to_end(sequence)
            return self._cache[sequence]

        self._load_model()

        # Truncate if needed
        truncated = len(sequence) > self.max_sequence_length
        seq = sequence[: self.max_sequence_length] if truncated else sequence

        try:
            inputs = self.tokenizer(
                seq,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)

            outputs = self.model(**inputs)

            # Extract pLDDT: per-residue confidence (0-100)
            plddt_per_residue = outputs.plddt[0, : len(seq), 0]
            mean_plddt = plddt_per_residue.mean().item()

            # Extract pTM score
            ptm = outputs.ptm.item() if hasattr(outputs, "ptm") else 0.0

            result = {
                "plddt": mean_plddt,
                "ptm": ptm,
                "plddt_per_residue": plddt_per_residue.cpu(),
                "sequence_length": len(seq),
                "truncated": truncated,
            }

        except Exception as e:
            logger.warning(
                f"ESMFold prediction failed for sequence (len={len(seq)}): {e}"
            )
            result = {
                "plddt": 0.0,
                "ptm": 0.0,
                "plddt_per_residue": torch.zeros(len(seq)),
                "sequence_length": len(seq),
                "truncated": truncated,
                "error": str(e),
            }

        # Cache with LRU eviction
        self._cache[sequence] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return result

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self._cache.clear()


def get_esmfold_predictor(
    device: str = "cuda",
    cache_size: int = 1000,
) -> ESMFoldPredictor:
    """
    Get or create the singleton ESMFoldPredictor.

    Args:
        device: Device for ESMFold. Default: "cuda".
        cache_size: Max cache entries. Default: 1000.

    Returns:
        ESMFoldPredictor instance (singleton).
    """
    global _esmfold_predictor
    if _esmfold_predictor is None:
        _esmfold_predictor = ESMFoldPredictor(
            device=device,
            cache_size=cache_size,
        )
    return _esmfold_predictor
