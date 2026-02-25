"""
ProtDescribe → Mol-Instructions Format Converter

Reads the ProtDescribe TSV dataset (katarinayuan/ProtDescribe) containing
UniProt protein entries with annotations for function, subcellular location,
protein naming, and similarity, then generates instruction-following pairs
matching the Mol-Instructions JSON schema.

TSV columns:
    EntryName, ProteinName, Function, SubcellularLocation, Similarity, Sequence

Task types generated:
    - function_description: "What is the function of this protein?" → Function text
    - subcellular_location: "Where is this protein localized?" → SubcellularLocation text
    - protein_naming: "What is the name of this protein?" → ProteinName
    - similarity_analysis: "What protein family does this belong to?" → Similarity text
"""

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction templates (style-matched to Mol-Instructions)
# ---------------------------------------------------------------------------

FUNCTION_INSTRUCTIONS = [
    "What is the function of this protein?",
    "Describe the biological function of the protein with the following sequence:",
    "Analyze the following protein sequence and explain its functional role:",
    "Based on the amino acid sequence below, describe the protein's function:",
    "Provide a functional description for the protein represented by this sequence:",
]

LOCATION_INSTRUCTIONS = [
    "Where is this protein localized within the cell?",
    "Predict the subcellular location of this protein:",
    "In which cellular compartment is this protein found?",
    "Describe the subcellular localization of the protein with this sequence:",
    "Based on the sequence, determine where this protein is located in the cell:",
]

NAMING_INSTRUCTIONS = [
    "What is the name of this protein?",
    "Identify the protein represented by the following sequence:",
    "Based on the amino acid sequence, predict the protein name:",
    "What protein does this sequence encode?",
    "Name the protein with the following amino acid sequence:",
]

SIMILARITY_INSTRUCTIONS = [
    "What protein family does this protein belong to?",
    "Describe the evolutionary relationships of this protein:",
    "Based on the sequence, identify similar proteins or protein families:",
    "What proteins share similarity with the one represented by this sequence?",
    "Describe the sequence similarity and family classification for this protein:",
]


# ---------------------------------------------------------------------------
# Column prefix stripping
# ---------------------------------------------------------------------------

_COLUMN_PREFIXES = {
    "ProteinName": "PROTEIN NAME: ",
    "Function": "FUNCTION: ",
    "SubcellularLocation": "SUBCELLULAR LOCATION: ",
    "Similarity": "SIMILARITY: ",
}


def _strip_prefix(column: str, value: str) -> str:
    """Remove the known prefix from a ProtDescribe column value."""
    prefix = _COLUMN_PREFIXES.get(column, "")
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    return value


def _is_non_empty(value: Optional[str]) -> bool:
    """Check if a value is non-null and non-empty after stripping."""
    return bool(value and value.strip())


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def _format_input(sequence: str) -> str:
    """Wrap sequence in backtick block matching Mol-Instructions format."""
    return f"```\n{sequence}\n```"


def _make_function_record(
    sequence: str, function_text: str, rng: random.Random
) -> Dict[str, Any]:
    """Create a function_description instruction record."""
    instruction = rng.choice(FUNCTION_INSTRUCTIONS)
    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": function_text,
        "metadata": {
            "seq_len": len(sequence),
            "task": "function_description",
            "source": "protdescribe",
        },
    }


def _make_location_record(
    sequence: str, location_text: str, rng: random.Random
) -> Dict[str, Any]:
    """Create a subcellular_location instruction record."""
    instruction = rng.choice(LOCATION_INSTRUCTIONS)
    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": location_text,
        "metadata": {
            "seq_len": len(sequence),
            "task": "subcellular_location",
            "source": "protdescribe",
        },
    }


def _make_naming_record(
    sequence: str, protein_name: str, rng: random.Random
) -> Dict[str, Any]:
    """Create a protein_naming instruction record."""
    instruction = rng.choice(NAMING_INSTRUCTIONS)
    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": f"This protein is {protein_name}.",
        "metadata": {
            "seq_len": len(sequence),
            "task": "protein_naming",
            "source": "protdescribe",
        },
    }


def _make_similarity_record(
    sequence: str, similarity_text: str, rng: random.Random
) -> Dict[str, Any]:
    """Create a similarity_analysis instruction record."""
    instruction = rng.choice(SIMILARITY_INSTRUCTIONS)
    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": similarity_text,
        "metadata": {
            "seq_len": len(sequence),
            "task": "similarity_analysis",
            "source": "protdescribe",
        },
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_protdescribe(
    source_dir: Path,
    output_dir: Path,
    *,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert ProtDescribe TSV to Mol-Instructions JSON files.

    Args:
        source_dir: Directory containing ProtDescribe TSV file(s)
        output_dir: Directory to write JSON files
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        seed: Random seed for instruction selection
        limit: Max number of rows to process (None = all)

    Returns:
        Dict with statistics about the conversion.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Find the TSV file
    tsv_path = None
    for candidate in [
        source_dir / "uniprot_sprot_filtered.tsv",
    ]:
        if candidate.exists():
            tsv_path = candidate
            break

    # Also check for parquet files (alternative format)
    parquet_paths = list(source_dir.glob("**/*.parquet"))

    if tsv_path is None and not parquet_paths:
        raise FileNotFoundError(
            f"ProtDescribe data not found in {source_dir}. "
            "Expected uniprot_sprot_filtered.tsv or parquet files. "
            "Run: python src/data/download.py --dataset protdescribe"
        )

    records_by_task: Dict[str, List] = {
        "function_description": [],
        "subcellular_location": [],
        "protein_naming": [],
        "similarity_analysis": [],
    }
    stats = {
        "total_rows": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_uncharacterized_name": 0,
    }

    if tsv_path is not None:
        logger.info(f"Reading ProtDescribe TSV: {tsv_path}")
        _process_tsv(tsv_path, records_by_task, stats, rng, min_length, max_length, limit)
    else:
        # Fall back to parquet
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for reading parquet files: pip install pandas pyarrow")
        for pq_path in sorted(parquet_paths):
            logger.info(f"Reading ProtDescribe parquet: {pq_path}")
            df = pd.read_parquet(pq_path)
            _process_dataframe(df, records_by_task, stats, rng, min_length, max_length, limit)

    # Write JSON files (one per task)
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
    logger.info(f"ProtDescribe conversion complete: {total_records} total records")

    # Write stats
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _process_tsv(
    tsv_path: Path,
    records_by_task: Dict[str, List],
    stats: Dict[str, Any],
    rng: random.Random,
    min_length: int,
    max_length: int,
    limit: Optional[int],
) -> None:
    """Process a ProtDescribe TSV file and populate records_by_task."""
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["total_rows"] += 1

            if limit and stats["total_rows"] > limit:
                break

            sequence = row.get("Sequence", "").strip()
            if not sequence:
                continue

            # Filter by length
            seq_len = len(sequence)
            if seq_len < min_length:
                stats["skipped_too_short"] += 1
                continue
            if seq_len > max_length:
                stats["skipped_too_long"] += 1
                continue

            _generate_records_for_row(row, sequence, records_by_task, stats, rng)


def _process_dataframe(
    df,
    records_by_task: Dict[str, List],
    stats: Dict[str, Any],
    rng: random.Random,
    min_length: int,
    max_length: int,
    limit: Optional[int],
) -> None:
    """Process a pandas DataFrame and populate records_by_task."""
    for _, row in df.iterrows():
        stats["total_rows"] += 1

        if limit and stats["total_rows"] > limit:
            break

        sequence = str(row.get("Sequence", "")).strip()
        if not sequence:
            continue

        seq_len = len(sequence)
        if seq_len < min_length:
            stats["skipped_too_short"] += 1
            continue
        if seq_len > max_length:
            stats["skipped_too_long"] += 1
            continue

        row_dict = {col: str(row[col]) if row[col] is not None else "" for col in row.index}
        _generate_records_for_row(row_dict, sequence, records_by_task, stats, rng)


def _generate_records_for_row(
    row: Dict[str, str],
    sequence: str,
    records_by_task: Dict[str, List],
    stats: Dict[str, Any],
    rng: random.Random,
) -> None:
    """Generate instruction records from a single data row."""
    # Function
    raw_func = row.get("Function", "").strip()
    func_text = _strip_prefix("Function", raw_func)
    if _is_non_empty(func_text):
        records_by_task["function_description"].append(
            _make_function_record(sequence, func_text, rng)
        )

    # Subcellular Location
    raw_loc = row.get("SubcellularLocation", "").strip()
    loc_text = _strip_prefix("SubcellularLocation", raw_loc)
    if _is_non_empty(loc_text):
        records_by_task["subcellular_location"].append(
            _make_location_record(sequence, loc_text, rng)
        )

    # Protein Naming
    raw_name = row.get("ProteinName", "").strip()
    protein_name = _strip_prefix("ProteinName", raw_name)
    if _is_non_empty(protein_name):
        # Skip "Uncharacterized protein" entries
        if protein_name.lower().startswith("uncharacterized protein"):
            stats["skipped_uncharacterized_name"] += 1
        else:
            records_by_task["protein_naming"].append(
                _make_naming_record(sequence, protein_name, rng)
            )

    # Similarity
    raw_sim = row.get("Similarity", "").strip()
    sim_text = _strip_prefix("Similarity", raw_sim)
    if _is_non_empty(sim_text):
        records_by_task["similarity_analysis"].append(
            _make_similarity_record(sequence, sim_text, rng)
        )


def prepare_protdescribe(
    raw_dir: Path, processed_dir: Path, cfg: Any = None
) -> Dict[str, Any]:
    """Entry point for prepare_data.py integration.

    Args:
        raw_dir: Path to raw protdescribe directory
        processed_dir: Output directory for JSON files
        cfg: Optional Hydra config

    Returns:
        Conversion statistics dict
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # Verify source data exists
    tsv_path = raw_dir / "uniprot_sprot_filtered.tsv"
    parquet_files = list(raw_dir.glob("**/*.parquet"))
    if not tsv_path.exists() and not parquet_files:
        raise FileNotFoundError(
            f"ProtDescribe data not found in {raw_dir}. "
            "Run: python src/data/download.py --dataset protdescribe"
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

    return convert_protdescribe(
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
        description="Convert ProtDescribe to instruction format"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw/protdescribe",
        help="Source directory containing ProtDescribe TSV/parquet files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/protdescribe",
        help="Output directory for JSON files",
    )
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit rows for testing"
    )
    parser.add_argument(
        "--show-samples", type=int, default=0, help="Show N sample records"
    )
    args = parser.parse_args()

    stats = convert_protdescribe(
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
