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
        log.info("Mol-Instructions: already in instruction format, no conversion needed.")
        log.info(f"Raw data at: {raw_dir}")
    elif data_name == "ipd_pdb":
        from src.data.ipd_pdb_converter import prepare_ipd_pdb
        stats = prepare_ipd_pdb(raw_dir, processed_dir, cfg)
        log.info(f"IPD-PDB conversion stats: {stats}")
    elif data_name == "swissprot":
        from src.data.swissprot_converter import prepare_swissprot
        stats = prepare_swissprot(raw_dir, processed_dir, cfg)
        log.info(f"Swiss-Prot conversion stats: {stats}")
    elif data_name == "wikipedia_protein":
        from src.data.wikipedia_protein_converter import prepare_wikipedia_protein
        stats = prepare_wikipedia_protein(raw_dir, processed_dir, cfg)
        log.info(f"Wikipedia protein conversion stats: {stats}")
    elif data_name == "proteinlm":
        from src.data.proteinlm_converter import prepare_proteinlm
        stats = prepare_proteinlm(raw_dir, processed_dir, cfg)
        log.info(f"ProteinLMDataset conversion stats: {stats}")
    elif data_name == "swissprotclap":
        from src.data.swissprotclap_converter import prepare_swissprotclap
        stats = prepare_swissprotclap(raw_dir, processed_dir, cfg)
        log.info(f"SwissProtCLAP conversion stats: {stats}")
    elif data_name == "protdescribe":
        from src.data.protdescribe_converter import prepare_protdescribe
        stats = prepare_protdescribe(raw_dir, processed_dir, cfg)
        log.info(f"ProtDescribe conversion stats: {stats}")
    elif data_name == "protein2text_qa":
        from src.data.protein2text_qa_converter import prepare_protein2text_qa
        stats = prepare_protein2text_qa(raw_dir, processed_dir, cfg)
        log.info(f"Protein2Text-QA conversion stats: {stats}")
    elif data_name in ("combined_sft_260225",):
        from src.data.assemble_combined import assemble_combined
        stats = assemble_combined(
            data_root=Path(cfg.paths.data_dir) if hasattr(cfg, "paths") else raw_dir.parent.parent,
            output_dir=processed_dir,
            verify=True,
        )
        log.info(f"Combined assembly stats: {stats}")
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    log.info(f"Data preparation complete! Output: {processed_dir}")


if __name__ == "__main__":
    main()
