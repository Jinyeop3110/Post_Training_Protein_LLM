"""Evaluation metrics and benchmarks.

This module provides evaluation functions for various protein prediction tasks:

- GO Prediction: Evaluate Gene Ontology term prediction
- PPI Prediction: Evaluate protein-protein interaction prediction
- Stability: Evaluate protein stability prediction (ddG)
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
    >>>
    >>> # PPI Prediction example
    >>> from src.evaluation import evaluate_ppi, parse_ppi_prediction, compute_ppi_metrics
    >>>
    >>> # Parse PPI prediction from model output
    >>> label, confidence = parse_ppi_prediction("Yes, these proteins interact.")
    >>> print(label, confidence)  # 1, 0.8
    >>>
    >>> # Run full PPI evaluation
    >>> metrics = evaluate_ppi(cfg, checkpoint_path="path/to/checkpoint")
    >>>
    >>> # Stability Prediction example
    >>> from src.evaluation import evaluate_stability, parse_stability_prediction
    >>>
    >>> # Parse stability prediction from model output
    >>> ddg, stability_class = parse_stability_prediction("The ddG is -2.5 kcal/mol. Stabilizing.")
    >>> print(ddg, stability_class)  # -2.5, 'stabilizing'
    >>>
    >>> # Run full stability evaluation
    >>> metrics = evaluate_stability(cfg, checkpoint_path="path/to/checkpoint")
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

from .ppi_prediction import (
    # Main evaluation function
    evaluate_ppi,
    # Data classes
    PPITestSample,
    PPIPredictionResult,
    # Helper functions
    parse_ppi_prediction,
    compute_ppi_metrics,
    load_ppi_test_dataset,
    create_ppi_prompt,
    # Utility functions
    evaluate_ppi_from_predictions,
)

from .stability import (
    # Main evaluation function
    evaluate_stability,
    # Data classes
    StabilityTestSample,
    StabilityPredictionResult,
    # Helper functions
    parse_stability_prediction,
    classify_ddg,
    compute_stability_metrics,
    load_stability_test_dataset,
    create_stability_prompt,
    # Utility functions
    evaluate_stability_from_predictions,
    # Constants
    STABILITY_CLASSES,
    STABILIZING_THRESHOLD,
    DESTABILIZING_THRESHOLD,
)

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
    # PPI Prediction
    "evaluate_ppi",
    "PPITestSample",
    "PPIPredictionResult",
    "parse_ppi_prediction",
    "compute_ppi_metrics",
    "load_ppi_test_dataset",
    "create_ppi_prompt",
    "evaluate_ppi_from_predictions",
    # Stability Prediction
    "evaluate_stability",
    "StabilityTestSample",
    "StabilityPredictionResult",
    "parse_stability_prediction",
    "classify_ddg",
    "compute_stability_metrics",
    "load_stability_test_dataset",
    "create_stability_prompt",
    "evaluate_stability_from_predictions",
    "STABILITY_CLASSES",
    "STABILIZING_THRESHOLD",
    "DESTABILIZING_THRESHOLD",
]
