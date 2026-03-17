#!/usr/bin/env python3
"""Merge SSL LoRA adapter into the base model.

After SSL continued pre-training, this script:
1. Loads the BASE model (e.g., Qwen3-8B-Base)
2. Loads the SSL LoRA adapter from the experiment checkpoint
3. Merges via PEFT merge_and_unload()
4. Saves the full merged model to results/{experiment}/merged/

The merged model becomes the base for SFT, which applies a fresh LoRA.

Usage:
    python scripts/merge_ssl.py --experiment ssl_text_qwen3_8b_base_0316_190000

    # Or specify paths directly:
    python scripts/merge_ssl.py \
        --base-model Qwen/Qwen3-8B-Base \
        --adapter results/my_ssl/checkpoints/lora_adapter \
        --output results/my_ssl/merged
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def find_adapter_path(experiment_dir: Path) -> Path:
    """Find the LoRA adapter directory within an experiment.

    Searches in order:
    1. checkpoints/lora_adapter/ (SSL trainer saves here)
    2. checkpoints/ (direct adapter at checkpoint root)
    3. Any directory containing adapter_config.json
    """
    candidates = [
        experiment_dir / "checkpoints" / "lora_adapter",
        experiment_dir / "checkpoints",
    ]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate

    # Search recursively
    for adapter_config in experiment_dir.rglob("adapter_config.json"):
        return adapter_config.parent

    raise FileNotFoundError(
        f"No adapter_config.json found in {experiment_dir}. "
        "Is the SSL experiment complete?"
    )


def find_base_model(experiment_dir: Path) -> str:
    """Determine the base model path from experiment config."""
    # Try training_args.json first
    training_args_path = experiment_dir / "training_args.json"
    if training_args_path.exists():
        with open(training_args_path) as f:
            args = json.load(f)
        model_path = args.get("model_path")
        if model_path:
            return model_path

    # Try config.yaml
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(config_path))
        return cfg.get("model", {}).get("path", "Qwen/Qwen3-8B-Base")

    return "Qwen/Qwen3-8B-Base"


def main():
    parser = argparse.ArgumentParser(
        description="Merge SSL LoRA adapter into base model"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="SSL experiment name (under results/)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path (auto-detected from experiment if not set)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="LoRA adapter path (auto-detected from experiment if not set)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for merged model (default: results/{experiment}/merged)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Results root directory",
    )
    args = parser.parse_args()

    if not args.experiment and not (args.base_model and args.adapter):
        parser.error("Either --experiment or both --base-model and --adapter required")

    t0 = time.time()

    # Resolve paths
    if args.experiment:
        experiment_dir = Path(args.results_dir) / args.experiment
        if not experiment_dir.exists():
            logger.error(f"Experiment directory not found: {experiment_dir}")
            sys.exit(1)

        adapter_path = args.adapter or str(find_adapter_path(experiment_dir))
        base_model = args.base_model or find_base_model(experiment_dir)
        output_dir = args.output or str(experiment_dir / "merged")
    else:
        adapter_path = args.adapter
        base_model = args.base_model
        output_dir = args.output or "./merged_model"

    logger.info(f"Base model: {base_model}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_dir}")

    # Load base model
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU to avoid OOM
    )
    logger.info(f"Base model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load LoRA adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    logger.info("Adapter loaded")

    # Merge
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    logger.info(f"Merged model: {sum(p.numel() for p in model.parameters()):,} params")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save merge metadata
    metadata = {
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "merge_method": "peft_merge_and_unload",
        "dtype": "bfloat16",
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    if args.experiment:
        metadata["ssl_experiment"] = args.experiment

    with open(output_path / "merge_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s. Merged model saved to {output_path}")
    logger.info(
        f"\nNext step: SFT with merged model as base:\n"
        f"  python scripts/train.py model.path={output_path} training=sft_lora"
    )


if __name__ == "__main__":
    main()
