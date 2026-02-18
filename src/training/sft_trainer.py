"""SFT trainer implementation."""

import logging
from typing import Any, Dict, Optional

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def get_qlora_config(cfg: DictConfig) -> Dict[str, Any]:
    """Get QLoRA configuration from Hydra config.

    Args:
        cfg: Hydra configuration.

    Returns:
        LoRA configuration dict.
    """
    lora_cfg = cfg.training.lora

    return {
        "r": lora_cfg.get("r", 8),
        "lora_alpha": lora_cfg.get("alpha", 16),
        "lora_dropout": lora_cfg.get("dropout", 0.05),
        "target_modules": list(lora_cfg.get("target_modules", ["k_proj", "v_proj"])),
        "bias": lora_cfg.get("bias", "none"),
        "task_type": lora_cfg.get("task_type", "CAUSAL_LM"),
    }


def run_sft_qlora(cfg: DictConfig) -> None:
    """Run SFT training with QLoRA.

    Args:
        cfg: Hydra configuration.
    """
    log.info("Starting SFT with QLoRA...")
    log.info(f"Model: {cfg.model.name}")
    log.info(f"Encoder: {cfg.encoder.name}")
    log.info(f"Learning rate: {cfg.training.lr}")

    # TODO: Implement full SFT training
    # 1. Load model with quantization
    # 2. Apply LoRA adapters
    # 3. Load dataset
    # 4. Train with TRL SFTTrainer

    raise NotImplementedError(
        "SFT trainer not yet implemented. "
        "See docs/training_guide.md for implementation details."
    )


def run_sft_lora(cfg: DictConfig) -> None:
    """Run SFT training with LoRA (no quantization).

    Args:
        cfg: Hydra configuration.
    """
    log.info("Starting SFT with LoRA...")

    raise NotImplementedError(
        "SFT LoRA trainer not yet implemented. "
        "See docs/training_guide.md for implementation details."
    )


class SFTTrainer:
    """SFT trainer class.

    Args:
        cfg: Hydra configuration.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def setup(self) -> None:
        """Set up model, tokenizer, and dataset."""
        raise NotImplementedError("Setup not yet implemented")

    def train(self) -> None:
        """Run training loop."""
        raise NotImplementedError("Training not yet implemented")

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        raise NotImplementedError("Checkpoint saving not yet implemented")
