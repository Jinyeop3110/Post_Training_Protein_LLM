"""
Assemble Combined SFT Dataset Directory

Creates a unified directory with symbolic links to all processed
SFT data sources. Each file is prefixed with its source abbreviation:
    mol_  = Mol-Instructions
    sp_   = Swiss-Prot
    wp_   = Wikipedia Protein
    plm_  = ProteinLMDataset
    clap_ = SwissProtCLAP
    pd_   = ProtDescribe
    p2t_  = Protein2Text-QA

Usage:
    python src/data/assemble_combined.py
    python src/data/assemble_combined.py --output data/processed/combined_sft_260225
    python src/data/assemble_combined.py --verify
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

# Each entry maps a prefix to its source directory (relative to data_root)
# and optionally lists explicit files. If files is None, auto-detect all
# .json files except conversion_stats.json.
SOURCES = {
    "mol": {
        "dir": "raw/mol_instructions/data/Protein-oriented_Instructions",
        "files": {
            "catalytic_activity": "catalytic_activity.json",
            "domain_motif": "domain_motif.json",
            "general_function": "general_function.json",
            "protein_design": "protein_design.json",
            "protein_function": "protein_function.json",
        },
    },
    "sp": {
        "dir": "processed/swissprot",
        "files": {
            "gene_prediction": "gene_prediction.json",
            "general_function": "general_function.json",
            "organism_prediction": "organism_prediction.json",
        },
    },
    "plm": {
        "dir": "processed/proteinlm",
        "files": None,  # Auto-detect
    },
    "clap": {
        "dir": "processed/swissprotclap",
        "files": None,
    },
    "pd": {
        "dir": "processed/protdescribe",
        "files": None,
    },
    "p2t": {
        "dir": "processed/protein2text_qa",
        "files": None,
    },
}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_combined(
    data_root: Path,
    output_dir: Path,
    verify: bool = False,
) -> Dict[str, Any]:
    """Create combined directory with symlinks to all sources.

    Args:
        data_root: Root data directory (e.g., data/)
        output_dir: Combined output directory
        verify: If True, load each file and count records

    Returns:
        Assembly statistics dict
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {"sources": {}, "total_files": 0, "total_records": 0}

    for prefix, config in SOURCES.items():
        source_dir = data_root / config["dir"]
        if not source_dir.exists():
            logger.warning(f"Source dir not found, skipping: {source_dir}")
            continue

        # Determine file mapping
        if config["files"] is not None:
            file_map = config["files"]
        else:
            file_map = {}
            for f in sorted(source_dir.glob("*.json")):
                if f.name != "conversion_stats.json":
                    file_map[f.stem] = f.name

        source_stats = {"files": 0, "records": 0, "file_list": []}

        for task_name, filename in file_map.items():
            source_file = source_dir / filename
            if not source_file.exists():
                logger.warning(f"Missing file: {source_file}")
                continue

            link_name = f"{prefix}_{task_name}.json"
            link_path = output_dir / link_name

            # Remove existing link/file
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()

            # Create relative symlink
            rel_target = os.path.relpath(source_file, output_dir)
            link_path.symlink_to(rel_target)

            source_stats["files"] += 1
            source_stats["file_list"].append(link_name)
            stats["total_files"] += 1

            if verify:
                try:
                    with open(source_file) as f:
                        data = json.load(f)
                    n = len(data)
                    source_stats["records"] += n
                    stats["total_records"] += n
                    logger.info(f"  {link_name}: {n:,} records")
                except Exception as e:
                    logger.error(f"  {link_name}: failed to load — {e}")

        stats["sources"][prefix] = source_stats
        logger.info(
            f"Source '{prefix}': {source_stats['files']} files"
            + (f", {source_stats['records']:,} records" if verify else "")
        )

    # Write assembly manifest
    manifest_path = output_dir / "assembly_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Assembly complete: {stats['total_files']} files"
        + (f", {stats['total_records']:,} total records" if verify else "")
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Assemble combined SFT dataset directory")
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root data directory",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/combined_sft_260225",
        help="Output directory for combined symlinks",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Load each file and verify record counts",
    )
    args = parser.parse_args()

    stats = assemble_combined(
        data_root=Path(args.data_root),
        output_dir=Path(args.output),
        verify=args.verify,
    )

    print("\n=== Assembly Statistics ===")
    for prefix, source_stats in stats["sources"].items():
        print(f"  {prefix}: {source_stats['files']} files", end="")
        if source_stats.get("records"):
            print(f", {source_stats['records']:,} records", end="")
        print()
        for fname in source_stats.get("file_list", []):
            print(f"    - {fname}")

    print(f"\n  Total: {stats['total_files']} files", end="")
    if stats.get("total_records"):
        print(f", {stats['total_records']:,} records", end="")
    print()
