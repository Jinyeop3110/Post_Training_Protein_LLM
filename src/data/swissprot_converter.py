"""
Swiss-Prot FASTA → Mol-Instructions Format Converter

Parses Swiss-Prot FASTA headers to extract functional annotations,
then generates instruction-following pairs matching the Mol-Instructions
JSON schema for seamless integration with existing data loaders.

Swiss-Prot header format:
    >sp|ACCESSION|ENTRY_NAME Description OS=Organism OX=TaxID [GN=GeneName] PE=Level SV=Version

Task types generated:
    - general_function: "Describe this protein's function" → description
    - organism_prediction: "What organism does this protein come from?" → organism
    - gene_prediction: "What gene encodes this protein?" → gene name (when available)
"""

import gzip
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction templates (style-matched to Mol-Instructions)
# ---------------------------------------------------------------------------

GENERAL_FUNCTION_INSTRUCTIONS = [
    "Analyze the protein with the following sequence and describe its properties:",
    "Inspect the protein with the subsequent sequence and offer a concise description of its properties:",
    "Examine the given protein sequence and share a brief overview of its attributes:",
    "Assess the following protein sequence and provide a brief report on its primary characteristics:",
    "Please provide a summary of the key features and characteristics of the protein with the following amino acid sequence:",
    "Could you evaluate the protein with this amino acid sequence and present a summary of its features? The sequence is:",
    "Conduct a quick analysis of the protein represented by the following sequence and outline its main characteristics:",
]

ORGANISM_INSTRUCTIONS = [
    "Given the following protein sequence, identify the organism it originates from:",
    "Which organism does this protein sequence come from? Analyze the sequence below:",
    "Determine the source organism for the protein with the following amino acid sequence:",
    "Predict the organism of origin for the protein represented by this sequence:",
    "Based on the amino acid sequence below, identify which organism this protein belongs to:",
]

GENE_INSTRUCTIONS = [
    "What is the gene name that encodes the protein with the following sequence?",
    "Predict the gene name associated with this protein sequence:",
    "Given the protein sequence below, identify the gene that encodes it:",
    "Determine which gene encodes the protein represented by the following sequence:",
    "Based on the amino acid sequence, predict the corresponding gene name:",
]


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_swissprot_header(header: str) -> Optional[Dict[str, str]]:
    """Parse a Swiss-Prot FASTA header line.

    Args:
        header: Header line starting with '>sp|...'

    Returns:
        Dict with keys: accession, entry_name, description, organism, taxid,
        gene_name (may be None), pe, sv.  Returns None on parse failure.
    """
    if not header.startswith(">sp|"):
        return None

    # Remove '>' prefix
    line = header[1:]

    # Parse sp|ACCESSION|ENTRY_NAME
    parts = line.split("|", 2)
    if len(parts) < 3:
        return None

    accession = parts[1]
    remainder = parts[2]

    # Split entry_name from the rest
    entry_name, _, desc_rest = remainder.partition(" ")

    # Extract OS=... OX=... GN=... PE=... SV=...
    os_match = re.search(r'\bOS=(.+?)(?:\s+OX=)', desc_rest)
    ox_match = re.search(r'\bOX=(\d+)', desc_rest)
    gn_match = re.search(r'\bGN=(\S+)', desc_rest)
    pe_match = re.search(r'\bPE=(\d+)', desc_rest)
    sv_match = re.search(r'\bSV=(\d+)', desc_rest)

    # Description is everything before " OS="
    os_pos = desc_rest.find(" OS=")
    description = desc_rest[:os_pos].strip() if os_pos >= 0 else desc_rest.strip()

    organism = os_match.group(1).strip() if os_match else None
    taxid = ox_match.group(1) if ox_match else None
    gene_name = gn_match.group(1) if gn_match else None
    pe = pe_match.group(1) if pe_match else None
    sv = sv_match.group(1) if sv_match else None

    if not description or not organism:
        return None

    return {
        "accession": accession,
        "entry_name": entry_name,
        "description": description,
        "organism": organism,
        "taxid": taxid,
        "gene_name": gene_name,
        "pe": pe,
        "sv": sv,
    }


def iter_fasta(fasta_path: Path) -> Iterator[Tuple[str, str]]:
    """Iterate over (header, sequence) pairs from a FASTA file (gzipped or plain)."""
    open_fn = gzip.open if str(fasta_path).endswith(".gz") else open

    header = None
    seq_parts: List[str] = []

    with open_fn(fasta_path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)

    # Last entry
    if header is not None:
        yield header, "".join(seq_parts)


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def _format_input(sequence: str) -> str:
    """Wrap sequence in backtick block matching Mol-Instructions format."""
    return f"```\n{sequence}\n```"


def _make_general_function_record(
    parsed: Dict[str, str], sequence: str, rng: random.Random
) -> Dict[str, Any]:
    """Create a general_function instruction record."""
    instruction = rng.choice(GENERAL_FUNCTION_INSTRUCTIONS)

    # Build a natural-language functional description
    desc = parsed["description"]
    organism = parsed["organism"]
    output = f"This protein is {desc}, found in {organism}."

    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": output,
        "metadata": {
            "protein_accession": parsed["accession"],
            "seq_len": len(sequence),
            "task": "general_function",
            "annots": desc,
            "source": "swissprot",
            "organism": organism,
        },
    }


def _make_organism_record(
    parsed: Dict[str, str], sequence: str, rng: random.Random
) -> Dict[str, Any]:
    """Create an organism_prediction instruction record."""
    instruction = rng.choice(ORGANISM_INSTRUCTIONS)
    organism = parsed["organism"]
    taxid = parsed.get("taxid", "")

    output = f"This protein originates from {organism}"
    if taxid:
        output += f" (taxonomy ID: {taxid})"
    output += "."

    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": output,
        "metadata": {
            "protein_accession": parsed["accession"],
            "seq_len": len(sequence),
            "task": "organism_prediction",
            "annots": organism,
            "source": "swissprot",
            "organism": organism,
        },
    }


def _make_gene_record(
    parsed: Dict[str, str], sequence: str, rng: random.Random
) -> Dict[str, Any]:
    """Create a gene_prediction instruction record (only when gene name is available)."""
    instruction = rng.choice(GENE_INSTRUCTIONS)
    gene = parsed["gene_name"]

    output = f"The gene encoding this protein is {gene}."

    return {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": output,
        "metadata": {
            "protein_accession": parsed["accession"],
            "seq_len": len(sequence),
            "task": "gene_prediction",
            "annots": gene,
            "source": "swissprot",
            "organism": parsed["organism"],
        },
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_swissprot(
    fasta_path: Path,
    output_dir: Path,
    *,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
    task_weights: Optional[Dict[str, float]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert Swiss-Prot FASTA to Mol-Instructions JSON files.

    Args:
        fasta_path: Path to uniprot_sprot.fasta.gz
        output_dir: Directory to write JSON files
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        seed: Random seed for instruction selection
        task_weights: Sampling weights per task type (default: all equal)
        limit: Max number of sequences to process (None = all)

    Returns:
        Dict with statistics about the conversion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if task_weights is None:
        task_weights = {
            "general_function": 1.0,
            "organism_prediction": 0.5,
            "gene_prediction": 0.5,
        }

    records_by_task: Dict[str, List] = {
        "general_function": [],
        "organism_prediction": [],
        "gene_prediction": [],
    }
    stats = {
        "total_sequences": 0,
        "skipped_parse_fail": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_no_description": 0,
    }

    logger.info(f"Reading Swiss-Prot FASTA: {fasta_path}")

    for header, sequence in iter_fasta(fasta_path):
        stats["total_sequences"] += 1

        if limit and stats["total_sequences"] > limit:
            break

        # Parse header
        parsed = parse_swissprot_header(header)
        if parsed is None:
            stats["skipped_parse_fail"] += 1
            continue

        # Filter by length
        seq_len = len(sequence)
        if seq_len < min_length:
            stats["skipped_too_short"] += 1
            continue
        if seq_len > max_length:
            stats["skipped_too_long"] += 1
            continue

        # Skip entries with uninformative descriptions
        desc_lower = parsed["description"].lower()
        if desc_lower in ("uncharacterized protein", "putative uncharacterized protein"):
            stats["skipped_no_description"] += 1
            continue

        # Generate records based on task weights (probabilistic sampling)
        if rng.random() < task_weights.get("general_function", 1.0):
            records_by_task["general_function"].append(
                _make_general_function_record(parsed, sequence, rng)
            )

        if rng.random() < task_weights.get("organism_prediction", 0.5):
            records_by_task["organism_prediction"].append(
                _make_organism_record(parsed, sequence, rng)
            )

        if parsed["gene_name"] and rng.random() < task_weights.get("gene_prediction", 0.5):
            records_by_task["gene_prediction"].append(
                _make_gene_record(parsed, sequence, rng)
            )

    # Write JSON files (one per task, matching Mol-Instructions layout)
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
    logger.info(f"Swiss-Prot conversion complete: {total_records} total records")

    # Write stats
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def prepare_swissprot(raw_dir: Path, processed_dir: Path, cfg: Any = None) -> Dict[str, Any]:
    """Entry point for prepare_data.py integration.

    Args:
        raw_dir: Path to raw swissprot directory containing the FASTA file
        processed_dir: Output directory for JSON files
        cfg: Optional Hydra config

    Returns:
        Conversion statistics dict
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # Find the FASTA file
    fasta_candidates = [
        raw_dir / "uniprot_sprot.fasta.gz",
        raw_dir / "uniprot_sprot.fasta",
    ]
    fasta_path = None
    for candidate in fasta_candidates:
        if candidate.exists():
            fasta_path = candidate
            break

    if fasta_path is None:
        raise FileNotFoundError(
            f"Swiss-Prot FASTA not found in {raw_dir}. "
            "Run: python src/data/download.py --dataset swissprot"
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

    return convert_swissprot(
        fasta_path=fasta_path,
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

    parser = argparse.ArgumentParser(description="Convert Swiss-Prot FASTA to instruction format")
    parser.add_argument(
        "--fasta", type=str,
        default="data/raw/swissprot/uniprot_sprot.fasta.gz",
        help="Path to Swiss-Prot FASTA file",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/processed/swissprot",
        help="Output directory for JSON files",
    )
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Limit sequences for testing")
    parser.add_argument("--show-samples", type=int, default=0, help="Show N sample records")
    args = parser.parse_args()

    stats = convert_swissprot(
        fasta_path=Path(args.fasta),
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
            print(f"\n=== {json_file.name} (showing {min(args.show_samples, len(records))} of {len(records)}) ===")
            for rec in records[:args.show_samples]:
                print(json.dumps(rec, indent=2))
                print()
