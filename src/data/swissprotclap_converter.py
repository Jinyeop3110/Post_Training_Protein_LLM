"""
SwissProtCLAP → Mol-Instructions Format Converter

Reads parallel protein_sequence.txt and text_sequence.txt files from the
SwissProtCLAP dataset (chao1224/ProteinDT), pairs sequences with their
text descriptions, and generates instruction-following pairs matching
the Mol-Instructions JSON schema.

File format:
    Both files have alternating lines:
        - Even lines (0, 2, 4, ...): UniProt accession IDs
        - Odd lines (1, 3, 5, ...): Actual data (sequence or description)

Task types generated:
    - protein_description: "Describe this protein" → text description
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction templates (style-matched to Mol-Instructions)
# ---------------------------------------------------------------------------

PROTEIN_DESCRIPTION_INSTRUCTIONS = [
    "Describe the properties and function of this protein:",
    "What can you tell about the protein with the following sequence?",
    "Analyze the protein sequence below and provide a description:",
    "Provide a comprehensive description of the protein represented by this sequence:",
    "What is known about this protein? Describe its key properties based on its sequence:",
    "Examine the protein with the following amino acid sequence and summarize its characteristics:",
]


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _read_parallel_files(
    protein_path: Path, text_path: Path
) -> List[Tuple[str, str, str]]:
    """Read the parallel SwissProtCLAP files and return (accession, sequence, description) tuples.

    The files have alternating lines: accession ID, then data, then next accession, etc.
    """
    pairs = []

    with open(protein_path, "r") as pf, open(text_path, "r") as tf:
        prot_lines = pf.read().splitlines()
        text_lines = tf.read().splitlines()

    if len(prot_lines) != len(text_lines):
        logger.warning(
            f"Line count mismatch: protein={len(prot_lines)}, text={len(text_lines)}"
        )

    # Process pairs of lines (accession + data)
    n_lines = min(len(prot_lines), len(text_lines))
    for i in range(0, n_lines - 1, 2):
        accession = prot_lines[i].strip()
        sequence = prot_lines[i + 1].strip()
        # text file has same accession on even lines, description on odd lines
        description = text_lines[i + 1].strip()

        if sequence and description:
            pairs.append((accession, sequence, description))

    return pairs


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def _format_input(sequence: str) -> str:
    """Wrap sequence in backtick block matching Mol-Instructions format."""
    return f"```\n{sequence}\n```"


def _make_description_record(
    accession: str,
    sequence: str,
    description: str,
    rng: random.Random,
) -> Dict[str, Any]:
    """Create a protein_description instruction record."""
    instruction = rng.choice(PROTEIN_DESCRIPTION_INSTRUCTIONS)

    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": description,
        "metadata": {
            "seq_len": len(sequence),
            "task": "protein_description",
            "source": "swissprotclap",
        },
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_swissprotclap(
    source_dir: Path,
    output_dir: Path,
    *,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert SwissProtCLAP parallel text files to Mol-Instructions JSON.

    Args:
        source_dir: Directory containing SwissProtCLAP/ subdirectory with
                     protein_sequence.txt and text_sequence.txt
        output_dir: Directory to write JSON files
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        seed: Random seed for instruction selection
        limit: Max number of pairs to process (None = all)

    Returns:
        Dict with statistics about the conversion.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Locate the text files
    clap_dir = source_dir / "SwissProtCLAP"
    protein_path = clap_dir / "protein_sequence.txt"
    text_path = clap_dir / "text_sequence.txt"

    if not protein_path.exists() or not text_path.exists():
        raise FileNotFoundError(
            f"SwissProtCLAP files not found in {clap_dir}. "
            "Expected protein_sequence.txt and text_sequence.txt"
        )

    stats = {
        "total_pairs": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_empty_description": 0,
    }

    logger.info(f"Reading SwissProtCLAP files from: {clap_dir}")

    # Read parallel files
    all_pairs = _read_parallel_files(protein_path, text_path)
    logger.info(f"Loaded {len(all_pairs)} raw pairs")

    records: List[Dict[str, Any]] = []

    for accession, sequence, description in all_pairs:
        stats["total_pairs"] += 1

        if limit and stats["total_pairs"] > limit:
            break

        # Filter by length
        seq_len = len(sequence)
        if seq_len < min_length:
            stats["skipped_too_short"] += 1
            continue
        if seq_len > max_length:
            stats["skipped_too_long"] += 1
            continue

        # Skip empty descriptions
        if not description.strip():
            stats["skipped_empty_description"] += 1
            continue

        records.append(
            _make_description_record(accession, sequence, description, rng)
        )

    # Write JSON file
    if records:
        output_path = output_dir / "protein_description.json"
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        stats["records_protein_description"] = len(records)
        logger.info(f"  Wrote {len(records)} records to {output_path.name}")

    stats["total_records"] = len(records)
    logger.info(f"SwissProtCLAP conversion complete: {len(records)} total records")

    # Write stats
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def prepare_swissprotclap(
    raw_dir: Path, processed_dir: Path, cfg: Any = None
) -> Dict[str, Any]:
    """Entry point for prepare_data.py integration.

    Args:
        raw_dir: Path to raw swissprotclap directory containing SwissProtCLAP/ subdir
        processed_dir: Output directory for JSON files
        cfg: Optional Hydra config

    Returns:
        Conversion statistics dict
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # Verify source files exist
    clap_dir = raw_dir / "SwissProtCLAP"
    if not clap_dir.exists():
        raise FileNotFoundError(
            f"SwissProtCLAP directory not found in {raw_dir}. "
            "Run: python src/data/download.py --dataset swissprotclap"
        )

    # Extract config filters
    min_length = 50
    max_length = 1000
    if cfg is not None:
        filters = cfg.get("filters", cfg.get("data", {}).get("filters", {}))
        if hasattr(filters, "to_container"):
            filters = filters.to_container(resolve=True)
        elif not isinstance(filters, dict):
            filters = {}
        min_length = filters.get("min_length", 50)
        max_length = filters.get("max_length", 1000)

    return convert_swissprotclap(
        source_dir=raw_dir,
        output_dir=processed_dir,
        min_length=min_length,
        max_length=max_length,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert SwissProtCLAP to instruction format"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw/swissprotclap",
        help="Source directory containing SwissProtCLAP/ subdirectory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/swissprotclap",
        help="Output directory for JSON files",
    )
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit pairs for testing"
    )
    parser.add_argument(
        "--show-samples", type=int, default=0, help="Show N sample records"
    )
    args = parser.parse_args()

    stats = convert_swissprotclap(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        min_length=args.min_length,
        max_length=args.max_length,
        limit=args.limit,
    )

    print("\n=== Conversion Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.show_samples > 0:
        output_dir = Path(args.output)
        for json_file in sorted(output_dir.glob("*.json")):
            if json_file.name == "conversion_stats.json":
                continue
            with open(json_file) as f:
                records = json.load(f)
            print(
                f"\n=== {json_file.name} "
                f"(showing {min(args.show_samples, len(records))} of {len(records)}) ==="
            )
            for rec in records[: args.show_samples]:
                print(json.dumps(rec, indent=2))
                print()
