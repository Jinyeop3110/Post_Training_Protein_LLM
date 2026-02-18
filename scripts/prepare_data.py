#!/usr/bin/env python3
"""Data preparation entry point."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main data preparation function.

    Args:
        cfg: Hydra configuration object.
    """
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Get data config
    data_name = cfg.data.name
    log.info(f"Preparing dataset: {data_name}")

    # Create output directories
    raw_dir = Path(cfg.data.paths.raw)
    processed_dir = Path(cfg.data.paths.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Process dataset
    if data_name == "mol_instructions":
        from src.data.instruction_dataset import prepare_mol_instructions
        prepare_mol_instructions(raw_dir, processed_dir, cfg)
    elif data_name == "ipd_pdb":
        from src.data.pdb_dataset import prepare_ipd_pdb
        prepare_ipd_pdb(raw_dir, processed_dir, cfg)
    elif data_name == "swissprot":
        from src.data.download import prepare_swissprot
        prepare_swissprot(raw_dir, processed_dir, cfg)
    elif data_name == "all":
        log.info("Preparing all datasets...")
        from src.data.instruction_dataset import prepare_mol_instructions
        from src.data.pdb_dataset import prepare_ipd_pdb
        from src.data.download import prepare_swissprot
        # Process each dataset
        # (Would need dataset-specific paths from config)
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    log.info(f"Data preparation complete! Output: {processed_dir}")


if __name__ == "__main__":
    main()
