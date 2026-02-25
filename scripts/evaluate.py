#!/usr/bin/env python3
"""Unified evaluation entry point with Hydra configuration.

Supports both vanilla HuggingFace baselines and ProteinLLM checkpoints.
The model is loaded once and shared across all requested evaluation tasks.

Usage examples::

    # Evaluate by experiment name (auto-detects checkpoint)
    python scripts/evaluate.py experiment_name=sft_esm3_mlp_50k evaluation.name=all

    # Evaluate specific checkpoint
    python scripts/evaluate.py checkpoint_path=results/.../checkpoints/protein_llm evaluation.name=all

    # Vanilla baseline — Qwen3-4B on all tasks
    python scripts/evaluate.py eval_mode=vanilla model=qwen3_4b evaluation.name=all

    # SFT evaluation only
    python scripts/evaluate.py eval_mode=vanilla model=qwen3_4b evaluation.name=sft

    # Single downstream task with sample limit
    python scripts/evaluate.py eval_mode=vanilla model=qwen3_4b evaluation.name=go_prediction evaluation.max_samples=5
"""

import json
import logging
import math
import os
import sys
from pathlib import Path

# Set Triton cache to local filesystem (CRITICAL: must be before any torch import)
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_{os.environ.get('USER', 'unknown')}")

# Ensure the project root is on sys.path so `from src.xxx import yyy` works
# even when Hydra changes CWD to the output directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def load_model(cfg: DictConfig, eval_mode: str, checkpoint_path):
    """Load model once based on eval_mode.

    Args:
        cfg: Hydra configuration.
        eval_mode: ``"vanilla"`` for bare HuggingFace model,
                   ``"protein_llm"`` for ProteinLLM checkpoint.
        checkpoint_path: Path to ProteinLLM checkpoint (used when
                         eval_mode is ``"protein_llm"``).

    Returns:
        Model instance with ``generate()`` and ``eval()`` methods.
    """
    if eval_mode == "vanilla":
        from src.models.vanilla_llm import VanillaLLMWrapper

        model = VanillaLLMWrapper.from_config(cfg)
    else:
        from src.models.multimodal_llm import ProteinLLM

        if checkpoint_path:
            log.info(f"Loading ProteinLLM from checkpoint: {checkpoint_path}")
            model = ProteinLLM.from_pretrained(checkpoint_path)
        else:
            log.info("Creating ProteinLLM from config (no checkpoint)")
            model = ProteinLLM.from_config(cfg)

    model.eval()
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    eval_name = cfg.get("evaluation", {}).get("name", "go_prediction")
    eval_mode = cfg.get("eval_mode", "protein_llm")
    checkpoint_path = cfg.get("checkpoint_path", None)

    # Auto-detect checkpoint from experiment_name if not set explicitly
    if eval_mode != "vanilla" and not checkpoint_path:
        experiment_dir = Path(cfg.paths.experiment_dir)
        auto_checkpoint = experiment_dir / "checkpoints" / "protein_llm"
        if auto_checkpoint.exists():
            checkpoint_path = str(auto_checkpoint)
            log.info(f"Auto-detected checkpoint from experiment: {checkpoint_path}")

    log.info(f"Evaluation mode: {eval_mode}")
    log.info(f"Running evaluation: {eval_name}")

    if eval_mode != "vanilla" and not checkpoint_path:
        log.warning(
            "No checkpoint_path specified and eval_mode is not 'vanilla'. "
            "Evaluation will use the base ProteinLLM without fine-tuned weights. "
            "Set checkpoint_path=<path> or eval_mode=vanilla."
        )

    # Prepare output directories — save to experiment_dir/eval/
    eval_dir = Path(cfg.paths.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Also ensure experiment dir exists for config save
    experiment_dir = Path(cfg.paths.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        log.info(f"Config saved to {config_path}")

    # Load model once
    model = load_model(cfg, eval_mode, checkpoint_path)

    eval_dir_str = str(eval_dir)

    # Dispatch evaluation
    results = None
    if eval_name == "go_prediction":
        from src.evaluation.go_prediction import evaluate_go

        results = evaluate_go(cfg, checkpoint_path, model=model, output_dir=eval_dir_str)

    elif eval_name == "ppi":
        from src.evaluation.ppi_prediction import evaluate_ppi

        results = evaluate_ppi(cfg, checkpoint_path, model=model, output_dir=eval_dir_str)

    elif eval_name == "stability":
        from src.evaluation.stability import evaluate_stability

        results = evaluate_stability(cfg, checkpoint_path, model=model, output_dir=eval_dir_str)

    elif eval_name == "sft":
        from src.evaluation.sft_eval import evaluate_sft

        results = evaluate_sft(cfg, checkpoint_path, model=model, output_dir=eval_dir_str)

    elif eval_name == "all":
        from src.evaluation.benchmarks import run_all_benchmarks

        results = run_all_benchmarks(cfg, checkpoint_path, model=model, output_dir=eval_dir_str)

    else:
        raise ValueError(f"Unknown evaluation: {eval_name}")

    # Log results
    log.info("Evaluation Results:")
    for metric, value in sorted(results.items()):
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                log.info(f"  {metric}: {value}")
            else:
                log.info(f"  {metric}: {value:.4f}")
        else:
            log.info(f"  {metric}: {value}")

    # Save metrics
    safe_results = {}
    for k, v in results.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            safe_results[k] = None
        else:
            safe_results[k] = v

    metrics_path = eval_dir / f"{eval_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(safe_results, f, indent=2, default=str)

    log.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
