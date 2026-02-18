#!/usr/bin/env python3
"""Training entry point with Hydra configuration."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    # Print resolved config
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Create directories
    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logging
    if cfg.logging.wandb.enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.logging.wandb.project,
                name=cfg.logging.wandb.name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except ImportError:
            log.warning("wandb not installed, skipping")

    # Select training method
    method = cfg.training.method
    log.info(f"Starting training with method: {method}")

    if method == "sft_qlora":
        from src.training.sft_trainer import run_sft_qlora
        run_sft_qlora(cfg)
    elif method == "sft_lora":
        from src.training.sft_trainer import run_sft_lora
        run_sft_lora(cfg)
    elif method == "grpo":
        from src.training.grpo_trainer import run_grpo
        run_grpo(cfg)
    elif method == "dpo":
        from src.training.dpo_trainer import run_dpo
        run_dpo(cfg)
    else:
        raise ValueError(f"Unknown training method: {method}")

    log.info("Training complete!")


if __name__ == "__main__":
    main()
