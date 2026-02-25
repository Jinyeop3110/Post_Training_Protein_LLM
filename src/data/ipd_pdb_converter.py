"""
IPD-PDB → Mol-Instructions Format Converter

Converts the IPD PDB training set (556K chains with sequences, coordinates,
and structural metadata) into instruction-following pairs matching the
Mol-Instructions JSON schema.

Note: IPD-PDB has NO functional annotations (no GO terms, no descriptions).
Tasks generated are based on available structural metadata:
    - structure_description: Characterize the protein from its structural metadata
    - sequence_properties: Analyze amino acid composition and properties

For richer functional tasks, consider cross-referencing with UniProt/Swiss-Prot
via PDB ID mapping, or use IPD-PDB primarily for multimodal (structure-aware)
training rather than text-only SFT.
"""

import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

STRUCTURE_DESCRIPTION_INSTRUCTIONS = [
    "Given the following protein sequence from a crystallographic study, describe the structural characteristics you can infer:",
    "Analyze the protein represented by this amino acid sequence and provide structural insights based on its properties:",
    "Examine this protein sequence and characterize its likely structural features:",
    "What structural properties can you infer about the protein with the following sequence?",
    "Based on the amino acid sequence below, describe the protein's structural characteristics:",
]

SEQUENCE_PROPERTIES_INSTRUCTIONS = [
    "Analyze the amino acid composition and properties of the following protein sequence:",
    "Provide a summary of the sequence characteristics for this protein:",
    "Examine the following protein sequence and describe its compositional features:",
    "What can you tell about the amino acid distribution and properties of this protein sequence?",
    "Characterize the sequence properties of the protein below:",
]


# ---------------------------------------------------------------------------
# Amino acid analysis helpers
# ---------------------------------------------------------------------------

# Amino acid property groups
AA_HYDROPHOBIC = set("AILMFWVP")
AA_POLAR = set("STNQY")
AA_CHARGED_POS = set("RHK")
AA_CHARGED_NEG = set("DE")
AA_AROMATIC = set("FWY")
AA_TINY = set("AGS")


def _analyze_sequence(sequence: str) -> Dict[str, Any]:
    """Compute basic sequence properties."""
    seq_upper = sequence.upper()
    length = len(seq_upper)
    counts = Counter(seq_upper)

    hydrophobic_frac = sum(counts.get(aa, 0) for aa in AA_HYDROPHOBIC) / length
    polar_frac = sum(counts.get(aa, 0) for aa in AA_POLAR) / length
    charged_pos_frac = sum(counts.get(aa, 0) for aa in AA_CHARGED_POS) / length
    charged_neg_frac = sum(counts.get(aa, 0) for aa in AA_CHARGED_NEG) / length
    aromatic_frac = sum(counts.get(aa, 0) for aa in AA_AROMATIC) / length

    # Net charge estimate at pH 7
    net_charge = (
        sum(counts.get(aa, 0) for aa in AA_CHARGED_POS)
        - sum(counts.get(aa, 0) for aa in AA_CHARGED_NEG)
    )

    # Cysteine count (potential disulfide bonds)
    cys_count = counts.get("C", 0)

    return {
        "length": length,
        "hydrophobic_fraction": round(hydrophobic_frac, 3),
        "polar_fraction": round(polar_frac, 3),
        "positive_charge_fraction": round(charged_pos_frac, 3),
        "negative_charge_fraction": round(charged_neg_frac, 3),
        "aromatic_fraction": round(aromatic_frac, 3),
        "net_charge_estimate": net_charge,
        "cysteine_count": cys_count,
        "potential_disulfide_bonds": cys_count // 2,
    }


def _classify_resolution(resolution: float) -> str:
    """Classify X-ray resolution quality."""
    if resolution <= 1.5:
        return "very high"
    elif resolution <= 2.0:
        return "high"
    elif resolution <= 2.5:
        return "good"
    elif resolution <= 3.0:
        return "moderate"
    else:
        return "low"


def _size_description(length: int) -> str:
    """Describe protein size category."""
    if length < 100:
        return "small"
    elif length < 300:
        return "medium-sized"
    elif length < 500:
        return "large"
    else:
        return "very large"


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def _format_input(sequence: str) -> str:
    """Wrap sequence in backtick block matching Mol-Instructions format."""
    return f"```\n{sequence}\n```"


def _make_structure_description_record(
    entry: Dict[str, Any], props: Dict[str, Any], rng: random.Random
) -> Dict[str, Any]:
    """Create a structure_description instruction record."""
    instruction = rng.choice(STRUCTURE_DESCRIPTION_INSTRUCTIONS)
    resolution = entry["resolution"]
    length = len(entry["sequence"])
    res_quality = _classify_resolution(resolution)
    size = _size_description(length)

    # Build informative output
    parts = [
        f"This is a {size} protein of {length} amino acids.",
        f"Its crystal structure was solved at {resolution:.2f} Angstrom resolution ({res_quality} quality).",
    ]

    if props["hydrophobic_fraction"] > 0.45:
        parts.append("It has a high hydrophobic content, suggesting a well-packed core or membrane association.")
    elif props["hydrophobic_fraction"] < 0.25:
        parts.append("It has relatively low hydrophobic content, suggesting significant solvent exposure.")

    if props["cysteine_count"] >= 4:
        parts.append(
            f"The sequence contains {props['cysteine_count']} cysteine residues, "
            f"potentially forming up to {props['potential_disulfide_bonds']} disulfide bonds."
        )

    if abs(props["net_charge_estimate"]) > 10:
        charge_type = "positive" if props["net_charge_estimate"] > 0 else "negative"
        parts.append(f"The protein has a strong net {charge_type} charge ({props['net_charge_estimate']:+d}).")

    output = " ".join(parts)

    return {
        "instruction": instruction,
        "input": _format_input(entry["sequence"]),
        "output": output,
        "metadata": {
            "chain_id": entry["chain_id"],
            "seq_len": length,
            "task": "structure_description",
            "annots": f"resolution={resolution:.2f}A",
            "source": "ipd_pdb",
            "resolution": resolution,
            "cluster": entry.get("cluster", ""),
        },
    }


def _make_sequence_properties_record(
    entry: Dict[str, Any], props: Dict[str, Any], rng: random.Random
) -> Dict[str, Any]:
    """Create a sequence_properties instruction record."""
    instruction = rng.choice(SEQUENCE_PROPERTIES_INSTRUCTIONS)
    length = len(entry["sequence"])

    output = (
        f"This protein has {length} amino acids. "
        f"Amino acid composition analysis shows: "
        f"{props['hydrophobic_fraction']*100:.1f}% hydrophobic residues, "
        f"{props['polar_fraction']*100:.1f}% polar residues, "
        f"{props['positive_charge_fraction']*100:.1f}% positively charged, "
        f"and {props['negative_charge_fraction']*100:.1f}% negatively charged. "
        f"The estimated net charge at neutral pH is {props['net_charge_estimate']:+d}. "
        f"It contains {props['aromatic_fraction']*100:.1f}% aromatic residues "
        f"and {props['cysteine_count']} cysteine residues."
    )

    return {
        "instruction": instruction,
        "input": _format_input(entry["sequence"]),
        "output": output,
        "metadata": {
            "chain_id": entry["chain_id"],
            "seq_len": length,
            "task": "sequence_properties",
            "annots": f"hydrophobic={props['hydrophobic_fraction']:.3f}",
            "source": "ipd_pdb",
            "resolution": entry["resolution"],
            "cluster": entry.get("cluster", ""),
        },
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_ipd_pdb(
    data_dir: Path,
    output_dir: Path,
    *,
    min_length: int = 50,
    max_length: int = 1000,
    max_resolution: float = 3.5,
    seed: int = 42,
    task_weights: Optional[Dict[str, float]] = None,
    limit: Optional[int] = None,
    cluster_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convert IPD-PDB data to Mol-Instructions JSON files.

    Args:
        data_dir: Path to pdb_2021aug02_sample directory
        output_dir: Directory to write JSON files
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        max_resolution: Maximum resolution in Angstroms
        seed: Random seed
        task_weights: Sampling weights per task type
        limit: Max entries to process (None = all)
        cluster_file: Optional cluster file to filter entries (train_clusters.txt)

    Returns:
        Dict with conversion statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if task_weights is None:
        task_weights = {
            "structure_description": 1.0,
            "sequence_properties": 0.5,
        }

    # Load cluster filter if provided
    valid_clusters = None
    if cluster_file and cluster_file.exists():
        with open(cluster_file) as f:
            valid_clusters = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(valid_clusters)} clusters from {cluster_file}")

    csv_path = Path(data_dir) / "list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"list.csv not found in {data_dir}")

    records_by_task: Dict[str, List] = {
        "structure_description": [],
        "sequence_properties": [],
    }
    stats = {
        "total_entries": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_low_resolution": 0,
        "skipped_cluster_filter": 0,
    }

    logger.info(f"Reading IPD-PDB list.csv: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_entries"] += 1

            if limit and stats["total_entries"] > limit:
                break

            chain_id = row["CHAINID"]
            sequence = row["SEQUENCE"]
            seq_len = len(sequence)
            resolution = float(row["RESOLUTION"]) if row["RESOLUTION"] else None
            cluster = row.get("CLUSTER", "")

            # Apply filters
            if seq_len < min_length:
                stats["skipped_too_short"] += 1
                continue
            if seq_len > max_length:
                stats["skipped_too_long"] += 1
                continue
            if resolution and resolution > max_resolution:
                stats["skipped_low_resolution"] += 1
                continue
            if valid_clusters and cluster not in valid_clusters:
                stats["skipped_cluster_filter"] += 1
                continue

            entry = {
                "chain_id": chain_id,
                "sequence": sequence,
                "resolution": resolution or 0.0,
                "cluster": cluster,
                "deposition": row.get("DEPOSITION", ""),
            }

            props = _analyze_sequence(sequence)

            # Generate records based on task weights
            if rng.random() < task_weights.get("structure_description", 1.0):
                records_by_task["structure_description"].append(
                    _make_structure_description_record(entry, props, rng)
                )

            if rng.random() < task_weights.get("sequence_properties", 0.5):
                records_by_task["sequence_properties"].append(
                    _make_sequence_properties_record(entry, props, rng)
                )

    # Write JSON files
    for task_name, records in records_by_task.items():
        if not records:
            continue
        output_path = output_dir / f"{task_name}.json"
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        stats[f"records_{task_name}"] = len(records)
        logger.info(f"  Wrote {len(records)} records to {output_path.name}")

    total_records = sum(len(v) for v in records_by_task.values())
    stats["total_records"] = total_records
    logger.info(f"IPD-PDB conversion complete: {total_records} total records")

    # Write stats
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def prepare_ipd_pdb(raw_dir: Path, processed_dir: Path, cfg: Any = None) -> Dict[str, Any]:
    """Entry point for prepare_data.py integration.

    Args:
        raw_dir: Path to pdb_2021aug02_sample directory
        processed_dir: Output directory for JSON files
        cfg: Optional Hydra config

    Returns:
        Conversion statistics dict
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    csv_path = raw_dir / "list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"list.csv not found in {raw_dir}. "
            "Run: python src/data/download.py --dataset ipd_pdb_sample"
        )

    # Extract config
    min_length = 50
    max_length = 1000
    max_resolution = 3.5
    cluster_file = None

    if cfg is not None:
        processing = cfg.get("processing", cfg.get("data", {}).get("processing", {}))
        if hasattr(processing, "to_container"):
            processing = processing.to_container(resolve=True)
        elif not isinstance(processing, dict):
            processing = {}
        max_length = processing.get("max_seq_length", 1024)

        # Try to get train cluster file for filtering
        splits = cfg.get("splits", cfg.get("data", {}).get("splits", {}))
        if hasattr(splits, "to_container"):
            splits = splits.to_container(resolve=True)
        elif not isinstance(splits, dict):
            splits = {}
        train_clusters = splits.get("train_clusters")
        if train_clusters:
            cluster_file = Path(str(train_clusters))

    return convert_ipd_pdb(
        data_dir=raw_dir,
        output_dir=processed_dir,
        min_length=min_length,
        max_length=max_length,
        max_resolution=max_resolution,
        cluster_file=cluster_file,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Convert IPD-PDB to instruction format")
    parser.add_argument(
        "--data-dir", type=str,
        default="data/raw/pdb_2021aug02_sample",
        help="Path to IPD-PDB directory",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/processed/ipd_pdb",
        help="Output directory for JSON files",
    )
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--max-resolution", type=float, default=3.5)
    parser.add_argument("--limit", type=int, default=None, help="Limit entries for testing")
    parser.add_argument("--show-samples", type=int, default=0, help="Show N sample records")
    args = parser.parse_args()

    stats = convert_ipd_pdb(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output),
        min_length=args.min_length,
        max_length=args.max_length,
        max_resolution=args.max_resolution,
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
            print(f"\n=== {json_file.name} (showing {min(args.show_samples, len(records))} of {len(records)}) ===")
            for rec in records[:args.show_samples]:
                print(json.dumps(rec, indent=2))
                print()
