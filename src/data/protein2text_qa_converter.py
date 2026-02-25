"""
Protein2Text-QA → Standard SFT JSON Format Converter

Converts the Protein2Text-QA dataset (tumorailab/Protein2Text-QA) from
conversational format to instruction-following JSON format.

Source format:
    conversations: [
        {"from": "human", "value": "<protein_sequence>\nQuestion?"},
        {"from": "gpt", "value": "Answer."}
    ]

The human message contains a protein sequence reference tag followed
by a question. The actual sequence is in the amino_seq column.

Single task type:
    - protein_qa: Free-form question answering about proteins
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_input(sequence: str) -> str:
    """Wrap sequence in backtick block matching project convention."""
    return f"```\n{sequence}\n```"


def _extract_question(human_value: str) -> str:
    """Extract the question from the human conversation turn.

    The human value typically contains a protein sequence placeholder
    (e.g. <protein_sequence> or the actual sequence) followed by the question.
    """
    # Remove common protein placeholders
    text = re.sub(r'<protein[_\s]*sequence>\s*', '', human_value, flags=re.IGNORECASE)
    # Remove raw sequence blocks if inlined (long uppercase AA strings)
    text = re.sub(r'\b[A-Z]{20,}\b', '', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_protein2text_qa(
    source_dir: Path,
    output_dir: Path,
    *,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert Protein2Text-QA to standard SFT JSON format.

    Args:
        source_dir: Path to downloaded dataset directory
        output_dir: Output directory for JSON files
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        seed: Random seed
        limit: Max records to process

    Returns:
        Conversion statistics dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    stats = {
        "total_entries": 0,
        "skipped_no_sequence": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_bad_conversation": 0,
        "skipped_empty_qa": 0,
    }

    # Load dataset — try HuggingFace datasets library first, then local parquet
    records_raw = None
    try:
        from datasets import load_dataset
        ds = load_dataset("tumorailab/Protein2Text-QA", split="test")
        records_raw = list(ds)
        logger.info(f"Loaded {len(records_raw)} records from HuggingFace")
    except Exception as e:
        logger.warning(f"HuggingFace load failed: {e}, trying local files...")

    if records_raw is None:
        try:
            import pandas as pd
            parquet_files = list(Path(source_dir).rglob("*.parquet"))
            if parquet_files:
                df = pd.concat([pd.read_parquet(f) for f in parquet_files])
                records_raw = df.to_dict("records")
                logger.info(f"Loaded {len(records_raw)} records from local parquet")
            else:
                # Try JSON files
                json_files = list(Path(source_dir).rglob("*.json"))
                records_raw = []
                for jf in json_files:
                    with open(jf) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        records_raw.extend(data)
                logger.info(f"Loaded {len(records_raw)} records from local JSON")
        except Exception as e:
            raise RuntimeError(f"Could not load Protein2Text-QA data: {e}")

    if not records_raw:
        raise RuntimeError(f"No data found in {source_dir}")

    records = []

    for item in records_raw:
        stats["total_entries"] += 1
        if limit and stats["total_entries"] > limit:
            break

        # Extract sequence
        sequence = str(item.get("amino_seq", "")).strip()
        if not sequence:
            stats["skipped_no_sequence"] += 1
            continue

        # Filter by length
        seq_len = len(sequence)
        if seq_len < min_length:
            stats["skipped_too_short"] += 1
            continue
        if seq_len > max_length:
            stats["skipped_too_long"] += 1
            continue

        # Parse conversations
        conversations = item.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            stats["skipped_bad_conversation"] += 1
            continue

        human_turn = conversations[0]
        gpt_turn = conversations[1]

        question = _extract_question(str(human_turn.get("value", "")))
        answer = str(gpt_turn.get("value", "")).strip()

        if not question or not answer:
            stats["skipped_empty_qa"] += 1
            continue

        records.append({
            "instruction": question,
            "input": _format_input(sequence),
            "output": answer,
            "metadata": {
                "seq_len": seq_len,
                "task": "protein_qa",
                "source": "protein2text_qa",
                "protein_name": str(item.get("protein", "")),
                "uniprot_id": str(item.get("id", "")),
            },
        })

    # Write output
    if records:
        output_path = output_dir / "protein_qa.json"
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        stats["records_protein_qa"] = len(records)
        logger.info(f"  Wrote {len(records)} records to {output_path.name}")

    stats["total_records"] = len(records)
    logger.info(f"Protein2Text-QA conversion complete: {len(records)} total records")

    # Write stats
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def prepare_protein2text_qa(
    raw_dir: Path, processed_dir: Path, cfg: Any = None
) -> Dict[str, Any]:
    """Entry point for prepare_data.py integration.

    Args:
        raw_dir: Path to raw protein2text_qa directory
        processed_dir: Output directory for JSON files
        cfg: Optional Hydra config

    Returns:
        Conversion statistics dict
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

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

    return convert_protein2text_qa(
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
        description="Convert Protein2Text-QA to instruction format"
    )
    parser.add_argument(
        "--source", type=str, default="data/raw/protein2text_qa",
        help="Path to downloaded dataset directory",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/protein2text_qa",
        help="Output directory for JSON files",
    )
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Limit records for testing")
    parser.add_argument("--show-samples", type=int, default=0, help="Show N sample records")
    args = parser.parse_args()

    stats = convert_protein2text_qa(
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
            print(f"\n=== {json_file.name} (showing {min(args.show_samples, len(records))} of {len(records)}) ===")
            for rec in records[:args.show_samples]:
                print(json.dumps(rec, indent=2))
                print()
