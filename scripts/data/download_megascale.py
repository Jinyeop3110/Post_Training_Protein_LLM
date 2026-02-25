"""
Download and convert Mega-Scale protein stability (ddG) data.

Downloads the Mega-Scale dataset (Tsuboyama et al., Nature 2023) from HuggingFace,
which contains ~776K curated protein stability measurements, and converts to our
JSON instruction format for RL training and evaluation.

Usage:
    python scripts/data/download_megascale.py
    python scripts/data/download_megascale.py --output_dir data/processed/megascale_stability
    python scripts/data/download_megascale.py --max_samples 10000
    python scripts/data/download_megascale.py --subset dataset3_single
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Stability classification thresholds (kcal/mol)
STABILIZING_THRESHOLD = -1.0
DESTABILIZING_THRESHOLD = 1.0


def classify_ddg(ddg: float) -> str:
    """Classify ddG value into stability category."""
    if ddg < STABILIZING_THRESHOLD:
        return "stabilizing"
    elif ddg > DESTABILIZING_THRESHOLD:
        return "destabilizing"
    else:
        return "neutral"


def load_megascale_from_huggingface(
    subset: str = "dataset3_single",
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load Mega-Scale dataset from HuggingFace.

    Args:
        subset: Dataset subset. Options:
            - "dataset3_single": Single mutations (~776K, recommended)
            - "dataset3_double": Double mutations
            - "dataset3_all": All mutations
        cache_dir: HuggingFace cache directory.

    Returns:
        List of raw records from the dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    log.info(f"Loading Mega-Scale ({subset}) from HuggingFace...")

    try:
        ds = load_dataset(
            "RosettaCommons/MegaScale",
            subset,
            cache_dir=cache_dir,
        )
    except Exception:
        # Try without subset (some versions don't have named configs)
        log.info("Trying without subset name...")
        ds = load_dataset(
            "RosettaCommons/MegaScale",
            cache_dir=cache_dir,
        )

    log.info(f"Loaded dataset with splits: {list(ds.keys())}")

    records = []
    for split_name in ds:
        split = ds[split_name]
        log.info(f"  {split_name}: {len(split)} samples")
        log.info(f"  Columns: {split.column_names}")
        for item in split:
            records.append(dict(item))

    return records


def parse_mutation_from_record(record: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract wild-type sequence, mutation notation, and mutant sequence.

    Mega-Scale columns vary by version. Common columns:
    - aa_seq: mutant amino acid sequence
    - WT_name: wild-type protein name/PDB ID
    - mut_type: mutation type (e.g., "single", "double")
    - deltaG / ddG_ML: stability values
    - Stabilizing_mut: boolean flag

    Returns:
        Tuple of (wild_type_seq, mutation_str, mutant_seq).
    """
    mutant_seq = record.get("aa_seq", record.get("mutant_sequence", ""))
    wt_seq = record.get("wt_aa_seq", record.get("wild_type_sequence", ""))
    wt_name = record.get("WT_name", record.get("protein_name", "unknown"))

    # Try to extract mutation notation
    mutation = record.get("mutation", record.get("mut_type", ""))
    if not mutation:
        # Try to infer mutation from sequence difference
        if wt_seq and mutant_seq and len(wt_seq) == len(mutant_seq):
            diffs = []
            for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mutant_seq)):
                if wt_aa != mut_aa:
                    diffs.append(f"{wt_aa}{i+1}{mut_aa}")
            mutation = "+".join(diffs) if diffs else "WT"
        else:
            mutation = "unknown"

    return wt_seq, mutation, mutant_seq


def convert_to_instruction_format(
    records: List[Dict[str, Any]],
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert Mega-Scale records to JSON instruction format.

    Args:
        records: Raw records from the dataset.
        max_samples: Limit number of output samples.

    Returns:
        List of instruction-format dicts.
    """
    import random

    samples = []
    class_counter = Counter()
    skipped = 0

    # Shuffle records for random sampling when max_samples is set
    if max_samples:
        random.shuffle(records)

    for record in records:
        if max_samples and len(samples) >= max_samples:
            break

        # Get ddG value - try multiple column names
        ddg = record.get("ddG_ML", record.get("deltaG", record.get("ddG", None)))
        if ddg is None:
            # Try other common column names
            for key in ["score", "stability_score", "dG", "delta_g"]:
                if key in record:
                    ddg = record[key]
                    break

        if ddg is None:
            skipped += 1
            continue

        try:
            ddg = float(ddg)
        except (ValueError, TypeError):
            skipped += 1
            continue

        # Skip NaN/Inf values
        if np.isnan(ddg) or np.isinf(ddg):
            skipped += 1
            continue

        # Extract sequences and mutation
        wt_seq, mutation, mutant_seq = parse_mutation_from_record(record)

        # Need at least a mutant sequence
        seq = mutant_seq or wt_seq
        if not seq:
            skipped += 1
            continue

        # Classify stability
        stability_class = classify_ddg(ddg)
        class_counter[stability_class] += 1

        # Build input text
        if wt_seq and mutation != "unknown":
            input_text = f"Wild-type: {wt_seq}\nMutation: {mutation}"
        else:
            input_text = seq

        # Build output text
        output_text = f"ddG = {ddg:.2f} kcal/mol. This mutation is {stability_class}."

        wt_name = record.get("WT_name", record.get("protein_name", "unknown"))

        sample = {
            "instruction": (
                "Predict the change in protein stability (ddG in kcal/mol) for this mutation. "
                "Classify as stabilizing (ddG < -1.0), neutral (-1.0 to 1.0), or "
                "destabilizing (ddG > 1.0)."
            ),
            "input": input_text,
            "output": output_text,
            "metadata": {
                "ddG": ddg,
                "wt_name": wt_name,
                "mutation": mutation,
                "stability_class": stability_class,
                "source": "megascale",
            },
        }
        samples.append(sample)

    log.info(f"Created {len(samples)} instruction samples (skipped {skipped})")
    log.info(f"Class distribution: {dict(class_counter)}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Download and convert Mega-Scale stability data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/megascale_stability",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="dataset3_single",
        help="Mega-Scale subset (dataset3_single, dataset3_double, dataset3_all)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of output samples",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    records = load_megascale_from_huggingface(args.subset, args.cache_dir)
    log.info(f"Loaded {len(records)} raw records")

    # Show sample record for debugging
    if records:
        sample_record = records[0]
        log.info(f"Sample record keys: {list(sample_record.keys())}")
        # Show a few key columns
        for key in ["aa_seq", "wt_aa_seq", "WT_name", "mut_type", "deltaG", "ddG_ML", "mutation"]:
            if key in sample_record:
                val = sample_record[key]
                if isinstance(val, str) and len(val) > 100:
                    val = val[:100] + "..."
                log.info(f"  {key}: {val}")

    # Convert to instruction format
    samples = convert_to_instruction_format(records, args.max_samples)

    # Save
    output_path = output_dir / "stability.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    log.info(f"Saved {len(samples)} samples to {output_path}")

    # Log statistics
    if samples:
        ddg_values = [s["metadata"]["ddG"] for s in samples]
        seq_lengths = [len(s["input"]) for s in samples]
        log.info("Statistics:")
        log.info(f"  Samples: {len(samples)}")
        log.info(f"  ddG range: [{min(ddg_values):.2f}, {max(ddg_values):.2f}] kcal/mol")
        log.info(f"  ddG mean: {np.mean(ddg_values):.2f} ± {np.std(ddg_values):.2f}")
        log.info(f"  Input length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
                 f"mean={np.mean(seq_lengths):.0f}")


if __name__ == "__main__":
    main()
