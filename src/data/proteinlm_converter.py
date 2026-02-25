"""
ProteinLMDataset → Mol-Instructions Format Converter

Parses the swissProt2Text.json from tsynbio/ProteinLMDataset (HuggingFace)
and extracts UniProt annotation fields into instruction-following pairs
matching the Mol-Instructions JSON schema.

Raw data format:
    Each entry is a conversation dict with a single output text containing
    structured sections: Introduction, Function, Subunit Structure,
    Tissue Specificity, Post-translational Modifications (PTM), Induction,
    Disease Association, and Domain. Protein sequences are in <seq>...</seq>
    tags with space-separated amino acids.

Task types generated (6 from swissProt2Text.json):
    - functionality  (~465K) — from "Function" section
    - subunit        (~291K) — from "Subunit Structure" section
    - tissue_specificity (~50K) — from "Tissue Specificity" section
    - ptm            (~46K)  — from "Post-translational Modifications (PTM)" section
    - induction      (~25K)  — from "Induction" section
    - disease        (~5.6K) — from "Disease Association" section

Note: The ECoT (Enzyme Chain of Thought) task (~10.8K) comes from IUBMB enzyme
data and is NOT present in swissProt2Text.json. It would require a separate
data source and converter.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction templates (style-matched to Mol-Instructions)
# ---------------------------------------------------------------------------

FUNCTIONALITY_INSTRUCTIONS = [
    "Describe the function of the protein with the following sequence:",
    "What is the biological function of this protein?",
    "Analyze the following protein sequence and explain its functional role:",
    "Provide a description of the function for the protein represented by this sequence:",
    "What biological activities does this protein perform? Analyze the sequence below:",
    "Explain the molecular function of the protein with the following amino acid sequence:",
    "Based on the protein sequence provided, describe its biological function:",
]

SUBUNIT_INSTRUCTIONS = [
    "Describe the subunit structure of this protein:",
    "What is the quaternary structure and subunit composition of this protein?",
    "Analyze the following protein and describe its subunit organization:",
    "Identify the subunit structure for the protein with this sequence:",
    "What oligomeric state does this protein adopt? Examine the sequence below:",
    "Describe the protein-protein interaction and subunit composition for this protein:",
    "What is the subunit arrangement of the protein represented by this sequence?",
]

TISSUE_SPECIFICITY_INSTRUCTIONS = [
    "In which tissues is this protein expressed?",
    "Describe the tissue-specific expression pattern of this protein:",
    "Identify the tissues where the following protein is predominantly found:",
    "What is the tissue distribution of the protein with this sequence?",
    "Analyze the protein sequence and predict its tissue specificity:",
    "Where in the body is this protein primarily expressed?",
    "Describe the expression profile of this protein across different tissues:",
]

PTM_INSTRUCTIONS = [
    "What post-translational modifications does this protein undergo?",
    "Describe the post-translational modifications associated with this protein:",
    "Identify any known PTMs for the protein with the following sequence:",
    "Analyze the protein and describe its post-translational modifications:",
    "What chemical modifications occur on this protein after translation?",
    "List the post-translational modifications that have been observed for this protein:",
    "Describe the PTM landscape of the protein with this amino acid sequence:",
]

INDUCTION_INSTRUCTIONS = [
    "What conditions induce the expression of this protein?",
    "Describe the factors that regulate the expression of this protein:",
    "Under what conditions is this protein upregulated?",
    "Identify the induction signals for the protein with this sequence:",
    "What environmental or cellular stimuli trigger expression of this protein?",
    "Describe the regulatory conditions that control this protein's expression:",
    "What stimuli or conditions lead to increased expression of this protein?",
]

DISEASE_INSTRUCTIONS = [
    "What diseases are associated with this protein?",
    "Describe any known disease associations for this protein:",
    "Is this protein involved in any pathological conditions?",
    "Identify diseases linked to mutations or dysregulation of this protein:",
    "What is the clinical significance of this protein?",
    "Describe the role of this protein in human disease:",
    "What pathological conditions are associated with dysfunction of this protein?",
]

# Mapping from section headers to task names and instruction templates
SECTION_TO_TASK = {
    "Function:": {
        "task": "functionality",
        "instructions": FUNCTIONALITY_INSTRUCTIONS,
    },
    "Subunit Structure:": {
        "task": "subunit",
        "instructions": SUBUNIT_INSTRUCTIONS,
    },
    "Tissue Specificity:": {
        "task": "tissue_specificity",
        "instructions": TISSUE_SPECIFICITY_INSTRUCTIONS,
    },
    "Post-translational Modifications (PTM):": {
        "task": "ptm",
        "instructions": PTM_INSTRUCTIONS,
    },
    "Induction:": {
        "task": "induction",
        "instructions": INDUCTION_INSTRUCTIONS,
    },
    "Disease Association:": {
        "task": "disease",
        "instructions": DISEASE_INSTRUCTIONS,
    },
}

# All known section headers (used for detecting section boundaries)
ALL_SECTION_HEADERS = {
    "Introduction:",
    "Function:",
    "Subunit Structure:",
    "Domain:",
    "Tissue Specificity:",
    "Post-translational Modifications (PTM):",
    "Induction:",
    "Disease Association:",
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_annotation(text: str) -> str:
    """Remove evidence codes, references, and clean whitespace from UniProt annotations.

    Handles:
        - {ECO:...} evidence codes (with optional PubMed refs)
        - {PubMed:...} references
        - [MIM:...] OMIM references
        - (By similarity) markers are kept as they provide useful context
        - Extra whitespace
    """
    # Remove {ECO:...} evidence codes (may include |PubMed:NNN)
    text = re.sub(r'\s*\{ECO:\S+(?:\|PubMed:\d+)*(?:\|Ref\.\d+)*\}\s*', ' ', text)
    # Remove standalone {PubMed:...} references
    text = re.sub(r'\s*\{PubMed:\d+\}\s*', ' ', text)
    # Remove [MIM:NNNNNN] references
    text = re.sub(r'\s*\[MIM:\d+\]\s*', ' ', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove spaces before punctuation (artifacts from removed evidence codes)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+,', ',', text)
    # Remove trailing periods and re-add single one
    text = text.rstrip('. ')
    if text:
        text += '.'
    return text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_sequence(output_text: str) -> Optional[str]:
    """Extract protein sequence from <seq>...</seq> tags.

    The ProteinLMDataset stores sequences as space-separated amino acids
    inside <seq> tags. This function joins them into a continuous string.

    Returns:
        Protein sequence string, or None if not found.
    """
    match = re.search(r'<seq>\s*(.+?)\s*</seq>', output_text)
    if not match:
        return None
    raw_seq = match.group(1)
    # Remove spaces between amino acid letters
    sequence = raw_seq.replace(' ', '')
    # Validate: should only contain standard amino acid letters
    if not re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', sequence):
        # Allow some non-standard but known amino acid codes (U=selenocysteine, O=pyrrolysine, X=unknown)
        if not re.match(r'^[ACDEFGHIKLMNPQRSTVWYUOX]+$', sequence):
            return None
    return sequence


def _extract_accession(output_text: str) -> Optional[str]:
    """Extract UniProt accession number from the Introduction section."""
    match = re.search(r'UniProt accession number (\S+?)(?:,|\s)', output_text)
    if match:
        return match.group(1).rstrip(',')
    return None


def _extract_sections(output_text: str) -> Dict[str, str]:
    """Parse the output text into sections.

    Returns:
        Dict mapping section header to content text (after boilerplate removal).
    """
    lines = output_text.split('\n')
    sections = {}

    current_section = None
    section_lines = []
    boilerplate_skipped = False

    for line in lines:
        stripped = line.strip()

        # Check if this is a section header
        if stripped in ALL_SECTION_HEADERS:
            # Save previous section
            if current_section is not None and section_lines:
                content = ' '.join(section_lines).strip()
                if content:
                    sections[current_section] = content
            current_section = stripped
            section_lines = []
            boilerplate_skipped = False
            continue

        # If we're in a section
        if current_section is not None:
            if not stripped:
                # Empty line - skip
                continue
            if not boilerplate_skipped:
                # First non-empty line after header is the boilerplate
                # e.g., "The functions of X are as follows:"
                # Check if it's a boilerplate line by looking for common patterns
                if _is_boilerplate(stripped, current_section):
                    boilerplate_skipped = True
                    continue
                else:
                    # No boilerplate - this line IS the content
                    boilerplate_skipped = True
                    section_lines.append(stripped)
            else:
                section_lines.append(stripped)

    # Save the last section
    if current_section is not None and section_lines:
        content = ' '.join(section_lines).strip()
        if content:
            sections[current_section] = content

    return sections


def _is_boilerplate(line: str, section: str) -> bool:
    """Detect boilerplate prefix lines that introduce each section.

    These lines follow predictable patterns like:
        "The functions of X are as follows:"
        "Regarding the subunit composition of X, the details are as follows:"
        "In terms of tissue specificity, X exhibits the following characteristics:"
        etc.
    """
    boilerplate_patterns = {
        "Function:": [
            r"The functions of .+ are as follows",
        ],
        "Subunit Structure:": [
            r"Regarding the subunit composition of .+, the details are as follows",
        ],
        "Tissue Specificity:": [
            r"In terms of tissue specificity, .+ exhibits the following characteristics",
        ],
        "Post-translational Modifications (PTM):": [
            r"The post-translational modifications \(PTMs\) of .+ are as follows",
        ],
        "Induction:": [
            r"Concerning the induction of .+, it is observed that",
        ],
        "Disease Association:": [
            r"In relation to disease association, .+ is linked with the following conditions",
        ],
        "Domain:": [
            r"Regarding the domain structure of .+, it is detailed as follows",
        ],
        "Introduction:": [
            r"The protein with UniProt accession",
            r"The sequence of .+ with UniProt accession",
        ],
    }

    patterns = boilerplate_patterns.get(section, [])
    for pattern in patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def _format_input(sequence: str) -> str:
    """Wrap sequence in backtick block matching Mol-Instructions format."""
    return f"```\n{sequence}\n```"


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def _make_record(
    task: str,
    instructions: List[str],
    sequence: str,
    annotation: str,
    accession: Optional[str],
    rng: random.Random,
) -> Dict[str, Any]:
    """Create a single instruction-following record."""
    instruction = rng.choice(instructions)
    cleaned = _clean_annotation(annotation)

    record = {
        "instruction": instruction,
        "input": _format_input(sequence),
        "output": cleaned,
        "metadata": {
            "seq_len": len(sequence),
            "task": task,
            "source": "proteinlm",
        },
    }
    if accession:
        record["metadata"]["protein_accession"] = accession

    return record


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_proteinlm(
    source_dir: Path,
    output_dir: Path,
    *,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert ProteinLMDataset swissProt2Text.json to instruction JSON files.

    Args:
        source_dir: Directory containing swissProt2Text.json
        output_dir: Directory to write per-task JSON files
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        seed: Random seed for instruction template selection
        limit: Max number of raw entries to process (None = all)

    Returns:
        Dict with conversion statistics.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Find source file
    source_file = source_dir / "swissProt2Text.json"
    if not source_file.exists():
        raise FileNotFoundError(
            f"swissProt2Text.json not found in {source_dir}. "
            "Download it from HuggingFace: tsynbio/ProteinLMDataset"
        )

    logger.info(f"Loading ProteinLMDataset from {source_file}")
    with open(source_file) as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} entries")

    # Initialize per-task record lists
    records_by_task: Dict[str, List] = {
        info["task"]: [] for info in SECTION_TO_TASK.values()
    }

    stats = {
        "total_entries": 0,
        "skipped_no_sequence": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_no_annotations": 0,
        "entries_processed": 0,
    }

    for i, entry in enumerate(raw_data):
        if limit is not None and i >= limit:
            break

        stats["total_entries"] += 1

        # Extract conversation output text
        try:
            output_text = entry["conversation"][0]["output"]
        except (KeyError, IndexError):
            stats["skipped_no_annotations"] += 1
            continue

        # Extract sequence
        sequence = _extract_sequence(output_text)
        if sequence is None:
            stats["skipped_no_sequence"] += 1
            continue

        # Filter by sequence length
        seq_len = len(sequence)
        if seq_len < min_length:
            stats["skipped_too_short"] += 1
            continue
        if seq_len > max_length:
            stats["skipped_too_long"] += 1
            continue

        # Extract accession
        accession = _extract_accession(output_text)

        # Extract annotation sections
        sections = _extract_sections(output_text)

        has_annotation = False
        for section_header, task_info in SECTION_TO_TASK.items():
            if section_header in sections:
                annotation_text = sections[section_header]
                if not annotation_text or len(annotation_text.strip()) < 5:
                    continue

                has_annotation = True
                record = _make_record(
                    task=task_info["task"],
                    instructions=task_info["instructions"],
                    sequence=sequence,
                    annotation=annotation_text,
                    accession=accession,
                    rng=rng,
                )
                records_by_task[task_info["task"]].append(record)

        if has_annotation:
            stats["entries_processed"] += 1
        else:
            stats["skipped_no_annotations"] += 1

        # Progress logging
        if (i + 1) % 100000 == 0:
            logger.info(f"  Processed {i + 1} / {len(raw_data)} entries...")

    # Write JSON files (one per task)
    for task_name, records in records_by_task.items():
        if not records:
            logger.info(f"  {task_name}: 0 records (skipped)")
            continue
        output_path = output_dir / f"{task_name}.json"
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        stats[f"records_{task_name}"] = len(records)
        logger.info(f"  Wrote {len(records):,} records to {output_path.name}")

    total_records = sum(len(v) for v in records_by_task.values())
    stats["total_records"] = total_records
    logger.info(f"ProteinLMDataset conversion complete: {total_records:,} total records")

    # Note about missing ECoT task
    stats["note_ecot"] = (
        "ECoT (Enzyme Chain of Thought) task is not in swissProt2Text.json. "
        "It requires IUBMB enzyme data from a separate source."
    )

    # Write stats
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to {stats_path}")

    return stats


def prepare_proteinlm(
    raw_dir: Path, processed_dir: Path, cfg: Any = None
) -> Dict[str, Any]:
    """Entry point for prepare_data.py integration.

    Args:
        raw_dir: Path to raw proteinlm directory containing swissProt2Text.json
        processed_dir: Output directory for per-task JSON files
        cfg: Optional Hydra config

    Returns:
        Conversion statistics dict
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

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

    return convert_proteinlm(
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
        description="Convert ProteinLMDataset (swissProt2Text.json) to instruction format"
    )
    parser.add_argument(
        "--source", type=str,
        default="data/raw/proteinlm",
        help="Directory containing swissProt2Text.json",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/processed/proteinlm",
        help="Output directory for per-task JSON files",
    )
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Limit entries for testing")
    parser.add_argument("--show-samples", type=int, default=0, help="Show N sample records per task")
    args = parser.parse_args()

    stats = convert_proteinlm(
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
            n_show = min(args.show_samples, len(records))
            print(f"\n=== {json_file.name} (showing {n_show} of {len(records):,}) ===")
            for rec in records[:n_show]:
                print(json.dumps(rec, indent=2))
                print()
