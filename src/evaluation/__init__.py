"""Evaluation metrics and benchmarks.

This module provides evaluation functions for various protein prediction tasks:

- GO Prediction: Evaluate Gene Ontology term prediction
- PPI Prediction: Evaluate protein-protein interaction prediction
- Stability: Evaluate protein stability prediction
- Benchmarks: Run all evaluation benchmarks

Example usage:
    >>> from src.evaluation import evaluate_go, parse_go_terms, compute_go_metrics
    >>>
    >>> # Parse GO terms from model output
    >>> go_terms = parse_go_terms("The protein has GO:0003674 and GO:0005634")
    >>> print(go_terms)  # ['GO:0003674', 'GO:0005634']
    >>>
    >>> # Run full GO evaluation
    >>> metrics = evaluate_go(cfg, checkpoint_path="path/to/checkpoint")
"""

from .go_prediction import (
    # Main evaluation function
    evaluate_go,
    # Data classes
    GOTestSample,
    GOPredictionResult,
    # Helper functions
    parse_go_terms,
    categorize_go_terms,
    compute_go_metrics,
    load_go_test_dataset,
    create_go_prompt,
    # Utility functions
    evaluate_go_from_predictions,
)

# These will be imported when implemented
# from .ppi_prediction import evaluate_ppi
# from .stability import evaluate_stability
# from .benchmarks import run_all_benchmarks

__all__ = [
    # GO Prediction
    "evaluate_go",
    "GOTestSample",
    "GOPredictionResult",
    "parse_go_terms",
    "categorize_go_terms",
    "compute_go_metrics",
    "load_go_test_dataset",
    "create_go_prompt",
    "evaluate_go_from_predictions",
]
