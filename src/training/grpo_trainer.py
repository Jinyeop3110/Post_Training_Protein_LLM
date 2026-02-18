"""GRPO trainer implementation."""

import logging
from typing import Any, Dict

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def get_grpo_config(cfg: DictConfig) -> Dict[str, Any]:
    """Get GRPO configuration from Hydra config.

    Args:
        cfg: Hydra configuration.

    Returns:
        GRPO configuration dict.
    """
    grpo_cfg = cfg.training.get("grpo", {})

    return {
        "group_size": grpo_cfg.get("group_size", 4),
        "temperature": grpo_cfg.get("temperature", 1.0),
        "use_kl_penalty": grpo_cfg.get("use_kl_penalty", False),
        "normalize_advantages": grpo_cfg.get("normalize_advantages", False),
    }


def run_grpo(cfg: DictConfig) -> None:
    """Run GRPO training.

    Args:
        cfg: Hydra configuration.
    """
    log.info("Starting GRPO training...")
    log.info(f"Model: {cfg.model.name}")
    log.info(f"Learning rate: {cfg.training.lr}")
    log.info(f"Group size: {cfg.training.grpo.group_size}")

    # TODO: Implement GRPO training with veRL
    # 1. Load SFT checkpoint
    # 2. Set up rollout engine (vLLM)
    # 3. Configure GRPO algorithm
    # 4. Run training loop

    raise NotImplementedError(
        "GRPO trainer not yet implemented. "
        "See .claude/skills/rl-training/SKILL.md for details."
    )


class GRPOTrainer:
    """GRPO trainer class.

    Args:
        cfg: Hydra configuration.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        self.rollout_engine = None

    def setup(self) -> None:
        """Set up model and rollout engine."""
        raise NotImplementedError("Setup not yet implemented")

    def train(self) -> None:
        """Run GRPO training loop."""
        raise NotImplementedError("Training not yet implemented")
