#!/usr/bin/env python3
"""Evaluation entry point with Hydra configuration."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration object.
    """
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Get evaluation task
    eval_name = cfg.get("evaluation", {}).get("name", "go_prediction")
    log.info(f"Running evaluation: {eval_name}")

    # Load checkpoint if specified
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path:
        log.info(f"Loading checkpoint from: {checkpoint_path}")

    # Run evaluation
    if eval_name == "go_prediction":
        from src.evaluation.go_prediction import evaluate_go
        results = evaluate_go(cfg, checkpoint_path)
    elif eval_name == "ppi":
        from src.evaluation.ppi_prediction import evaluate_ppi
        results = evaluate_ppi(cfg, checkpoint_path)
    elif eval_name == "stability":
        from src.evaluation.stability import evaluate_stability
        results = evaluate_stability(cfg, checkpoint_path)
    elif eval_name == "all":
        from src.evaluation.benchmarks import run_all_benchmarks
        results = run_all_benchmarks(cfg, checkpoint_path)
    else:
        raise ValueError(f"Unknown evaluation: {eval_name}")

    # Log results
    log.info("Evaluation Results:")
    for metric, value in results.items():
        log.info(f"  {metric}: {value:.4f}")

    # Save results
    output_dir = Path(cfg.get("output_dir", "./results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / f"{eval_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
