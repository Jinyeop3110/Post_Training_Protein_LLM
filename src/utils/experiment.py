"""
Experiment management utilities.

Provides lineage tracking for the base -> SFT -> GRPO pipeline,
with all artifacts stored under a single experiment directory.

Experiment directory structure:
    results/{experiment_name}/
    ├── config.yaml            # Full resolved Hydra config
    ├── lineage.json           # Stage, parent, timestamps
    ├── training_args.json     # Hyperparameters
    ├── metrics.json           # Final train/eval metrics
    ├── checkpoints/
    │   └── protein_llm/      # ProteinLLM save
    ├── logs/
    │   ├── .hydra/            # Hydra config snapshots
    │   └── tensorboard/       # TensorBoard events
    └── eval/
        └── {task}_metrics.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def write_lineage(
    experiment_dir: str | Path,
    stage: str,
    experiment_name: str,
    cfg_dict: Dict[str, Any],
    parent_experiment: Optional[str] = None,
    parent_checkpoint: Optional[str] = None,
) -> Path:
    """Write lineage.json with experiment metadata.

    Args:
        experiment_dir: Path to the experiment directory.
        stage: Training stage (e.g., "sft_qlora", "sft_lora", "grpo", "dpo").
        experiment_name: Name of this experiment.
        cfg_dict: Resolved config as a dict.
        parent_experiment: Name of the parent experiment (for GRPO/DPO).
        parent_checkpoint: Path to the parent checkpoint.

    Returns:
        Path to the written lineage.json file.
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    lineage = {
        "experiment_name": experiment_name,
        "stage": stage,
        "parent_experiment": parent_experiment,
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "base_model": cfg_dict.get("model", {}).get("path", "unknown"),
        "encoder": cfg_dict.get("encoder", {}).get("model_name", "none"),
        "approach": cfg_dict.get("approach", "text"),
        "projector_type": cfg_dict.get("encoder", {}).get("projector", {}).get("type", "mlp"),
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    }

    lineage_path = experiment_dir / "lineage.json"
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)

    log.info(f"Lineage written to {lineage_path}")
    return lineage_path


def complete_lineage(experiment_dir: str | Path) -> None:
    """Mark an experiment as completed by setting completed_at timestamp.

    Args:
        experiment_dir: Path to the experiment directory.
    """
    lineage_path = Path(experiment_dir) / "lineage.json"
    if not lineage_path.exists():
        log.warning(f"No lineage.json found at {lineage_path}")
        return

    with open(lineage_path) as f:
        lineage = json.load(f)

    lineage["completed_at"] = datetime.now().isoformat()

    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)

    log.info(f"Lineage completed: {lineage_path}")


def read_lineage(experiment_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Read lineage.json from an experiment directory.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        Lineage dict if found, None otherwise.
    """
    lineage_path = Path(experiment_dir) / "lineage.json"
    if not lineage_path.exists():
        return None

    with open(lineage_path) as f:
        return json.load(f)


def resolve_parent_checkpoint(
    results_dir: str | Path,
    parent_experiment: str,
) -> Optional[Path]:
    """Resolve checkpoint path from parent experiment name.

    Looks for the checkpoint at:
        results/{parent_experiment}/checkpoints/protein_llm

    Args:
        results_dir: Root results directory.
        parent_experiment: Name of the parent experiment.

    Returns:
        Path to parent checkpoint if found, None otherwise.
    """
    results_dir = Path(results_dir)
    parent_dir = results_dir / parent_experiment
    checkpoint_path = parent_dir / "checkpoints" / "protein_llm"

    if checkpoint_path.exists():
        log.info(f"Resolved parent checkpoint: {checkpoint_path}")
        return checkpoint_path

    # Fallback: check if there's a checkpoint dir with any content
    checkpoints_dir = parent_dir / "checkpoints"
    if checkpoints_dir.exists():
        # Look for protein_llm subdirectory in any checkpoint
        for item in sorted(checkpoints_dir.iterdir()):
            if item.is_dir():
                protein_llm_path = item / "protein_llm"
                if protein_llm_path.exists():
                    log.info(f"Resolved parent checkpoint (fallback): {protein_llm_path}")
                    return protein_llm_path

    log.warning(
        f"Could not resolve parent checkpoint for '{parent_experiment}' "
        f"in {results_dir}"
    )
    return None


def list_experiments(results_dir: str | Path) -> List[Dict[str, Any]]:
    """List all experiments with their stage, parent, and status.

    Args:
        results_dir: Root results directory.

    Returns:
        List of experiment info dicts sorted by creation time.
    """
    results_dir = Path(results_dir)
    experiments = []

    if not results_dir.exists():
        return experiments

    for item in sorted(results_dir.iterdir()):
        if not item.is_dir():
            continue

        lineage = read_lineage(item)
        if lineage is not None:
            experiments.append({
                "name": item.name,
                "path": str(item),
                **lineage,
            })
        else:
            # Directory exists but no lineage — legacy or incomplete
            experiments.append({
                "name": item.name,
                "path": str(item),
                "stage": "unknown",
                "parent_experiment": None,
                "created_at": None,
                "completed_at": None,
            })

    return experiments
