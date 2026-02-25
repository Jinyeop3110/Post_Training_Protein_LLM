#!/usr/bin/env python3
"""Training entry point with Hydra configuration.

Parallelization:
  Single GPU:  python scripts/train.py experiment=sft_esm3_mlp
  Multi-GPU:   bash scripts/launch_train.sh experiment=sft_esm3_mlp
               (uses torchrun --nproc_per_node=NUM_GPUS)

DDP layout (multi-GPU):
  - LLM (LoRA): DDP-wrapped by HF Trainer, one replica per GPU
  - ESM-3 encoder: replicated per GPU, frozen (no gradient sync needed)
  - Pooling + Projector: NOT DDP-wrapped; gradients manually all-reduced
    in ProteinLLMTrainer.training_step() via dist.all_reduce(op=AVG)
  - Data: auto-sharded by HF Trainer's DistributedSampler
"""

import logging
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

# When launched via torchrun, give each rank a unique Hydra output dir
# to prevent file locking conflicts between DDP processes.
if "LOCAL_RANK" in os.environ:
    rank = os.environ["LOCAL_RANK"]
    if not any("hydra.run.dir" in arg for arg in sys.argv):
        sys.argv.append(f"hydra.run.dir=./logs/ddp_rank_{rank}")

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0) in DDP."""
    return int(os.environ.get("RANK", 0)) == 0


def _build_wandb_tags(cfg: DictConfig) -> list:
    """Build wandb tags from config, ensuring method, model, dataset, lr, epochs are included.

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of tag strings.
    """
    tags = []

    # Add tags from training config
    training_tags = cfg.training.get("wandb", {}).get("tags", [])
    if training_tags:
        tags.extend(list(training_tags))

    # Add extra tags (lr, epochs) from logging config
    extra_tags = cfg.logging.get("wandb", {}).get("extra_tags", [])
    if extra_tags:
        tags.extend(list(extra_tags))

    # Ensure required tags are always present
    method = cfg.training.get("method", "unknown")
    model_name = cfg.model.get("name", "unknown")
    dataset_name = cfg.data.get("name", cfg.data.get("source", "unknown"))

    required = {
        f"method:{method}",
        f"model:{model_name}",
        f"dataset:{dataset_name}",
        f"lr:{cfg.training.get('lr', 'unknown')}",
        f"epochs:{cfg.training.get('epochs', 'unknown')}",
    }

    # Only add required tags if not already covered by a matching tag
    existing_lower = {t.lower() for t in tags}
    for req in required:
        key = req.split(":")[0]
        if not any(key in t.lower() for t in existing_lower):
            tags.append(req)

    return tags


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    import torch

    # DDP: set CUDA device early so torch.cuda.current_device() returns
    # the correct GPU for this rank (otherwise all ranks default to GPU 0).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

        # TF32: use TensorFloat-32 for float32 matmuls on H100/A100.
        # ~10-15% faster matmuls (including ESM-3 forward) with negligible precision loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Print resolved config (rank 0 only)
    if _is_main_process():
        log.info("Configuration:")
        log.info(OmegaConf.to_yaml(cfg))

    # Resolve experiment directory (all artifacts go here)
    experiment_dir = Path(cfg.paths.experiment_dir)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    log_dir = Path(cfg.paths.log_dir)

    # Create directories (rank 0 only, then barrier)
    if _is_main_process():
        experiment_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler so all log output is captured to train.log
        log_file = experiment_dir / "train.log"
        file_handler = logging.FileHandler(str(log_file), mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s - %(message)s")
        )
        # Attach to root logger so all modules' log output is captured
        logging.getLogger().addHandler(file_handler)
        log.info(f"Logging to file: {log_file}")

        # Write lineage.json
        from src.utils.experiment import resolve_parent_checkpoint, write_lineage

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        method = cfg.training.get("method", "unknown")
        experiment_name = cfg.get("experiment_name", "default")
        parent_experiment = cfg.get("parent_experiment", None)
        parent_checkpoint = cfg.get("parent_checkpoint", None)

        # Auto-resolve parent checkpoint from parent_experiment
        if parent_experiment and not parent_checkpoint:
            results_dir = Path(cfg.paths.results_dir)
            parent_checkpoint = resolve_parent_checkpoint(
                results_dir, parent_experiment
            )
            if parent_checkpoint:
                parent_checkpoint = str(parent_checkpoint)

        write_lineage(
            experiment_dir=experiment_dir,
            stage=method,
            experiment_name=experiment_name,
            cfg_dict=cfg_dict,
            parent_experiment=parent_experiment,
            parent_checkpoint=parent_checkpoint,
        )

        # Save resolved config to experiment dir
        config_path = experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        log.info(f"Config saved to {config_path}")

    # Barrier so non-rank-0 processes wait for dirs to be created
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # Initialize wandb logging (rank 0 only)
    if cfg.logging.wandb.enabled and _is_main_process():
        try:
            import wandb

            # Build tags including: method, model, dataset, lr, epochs
            tags = _build_wandb_tags(cfg)

            # Project comes from training config: protein-llm-sft or protein-llm-rl
            project = cfg.logging.wandb.get("project", "protein-llm")

            wandb.init(
                project=project,
                name=cfg.logging.wandb.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=tags,
            )
            log.info(f"wandb initialized: project={project}, tags={tags}")
        except ImportError:
            log.warning("wandb not installed, skipping")

    # Select training method
    method = cfg.training.method
    if _is_main_process():
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

    if _is_main_process():
        # Mark lineage as completed
        from src.utils.experiment import complete_lineage
        complete_lineage(experiment_dir)

        log.info(f"Training complete! Experiment dir: {experiment_dir}")


if __name__ == "__main__":
    main()
