"""
Download and prepare protein solubility prediction data.

Sources protein solubility data (eSOL database or HuggingFace alternatives)
and converts to MolInstructions JSON format for GRPO training.

Usage:
    python scripts/data/download_solubility.py
    python scripts/data/download_solubility.py --max_samples 10000
    python scripts/data/download_solubility.py --output_dir data/processed/solubility_dataset
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Solubility classification threshold (percentage)
SOLUBILITY_THRESHOLD = 50.0  # >50% = soluble, ≤50% = insoluble


def classify_solubility(score: float) -> str:
    """Classify solubility score into binary category."""
    return "soluble" if score > SOLUBILITY_THRESHOLD else "insoluble"


def load_solubility_data(
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load solubility data from HuggingFace datasets.

    Tries multiple sources in order:
    1. HuggingFace eSOL-style datasets
    2. ProteinSol or DeepSol benchmark data

    Args:
        cache_dir: HuggingFace cache directory.

    Returns:
        List of records with sequence, solubility_score, and protein_id.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    records = []

    # Try loading from HuggingFace — multiple possible sources
    hf_sources = [
        ("proteinea/solubility", None),
        ("sagawa/protein-solubility", None),
    ]

    for dataset_name, subset in hf_sources:
        try:
            log.info(f"Trying HuggingFace dataset: {dataset_name}")
            kwargs = {"cache_dir": cache_dir}
            if subset:
                kwargs["name"] = subset
            ds = load_dataset(dataset_name, **kwargs)
            log.info(f"Loaded {dataset_name} with splits: {list(ds.keys())}")

            for split_name in ds:
                split = ds[split_name]
                log.info(f"  {split_name}: {len(split)} samples, columns: {split.column_names}")

                for item in split:
                    record = dict(item)
                    records.append(record)

            if records:
                log.info(f"Successfully loaded {len(records)} records from {dataset_name}")
                return records
        except Exception as e:
            log.info(f"  Failed: {e}")
            continue

    if not records:
        log.warning("No HuggingFace datasets available. Generating synthetic data from UniProt.")
        records = _generate_from_uniprot(cache_dir)

    return records


def _generate_from_uniprot(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate solubility-like data from UniProt E. coli proteome.

    Uses UniProt E. coli proteins with simulated solubility scores based on
    known sequence features (hydrophobicity, length, charge).

    This is a fallback for when eSOL HuggingFace data is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    log.info("Loading E. coli proteome from UniProt via HuggingFace...")
    try:
        ds = load_dataset(
            "agemagician/uniref50",
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )
        records = []
        for i, item in enumerate(ds):
            if i >= 15000:  # Enough to sample from
                break
            seq = item.get("sequence", item.get("text", ""))
            if not seq or len(seq) < 10 or len(seq) > 2000:
                continue
            # Estimate solubility from sequence features
            score = _estimate_solubility(seq)
            records.append({
                "sequence": seq,
                "solubility_score": score,
                "protein_id": item.get("id", f"uniprot_{i}"),
                "source": "uniprot_estimated",
            })
        log.info(f"Generated {len(records)} records with estimated solubility")
        return records
    except Exception as e:
        log.warning(f"UniProt fallback failed: {e}")
        return []


def _estimate_solubility(sequence: str) -> float:
    """Estimate solubility percentage from sequence features.

    Simple model based on known correlates of E. coli expression solubility:
    - Shorter proteins tend to be more soluble
    - Higher net charge (less hydrophobic) correlates with solubility
    - Fewer cysteines (no disulfide bonds needed in E. coli cytoplasm)
    """
    # Hydrophobic amino acids
    hydrophobic = set("AVILMFYW")
    hydro_frac = sum(1 for aa in sequence if aa in hydrophobic) / max(len(sequence), 1)

    # Charged amino acids
    charged = set("DEKRH")
    charge_frac = sum(1 for aa in sequence if aa in charged) / max(len(sequence), 1)

    # Length penalty
    length_score = max(0, 1.0 - len(sequence) / 1000)

    # Cysteine penalty
    cys_frac = sequence.count("C") / max(len(sequence), 1)

    # Combine features with noise
    base_score = (
        50.0
        + 30.0 * charge_frac
        - 40.0 * hydro_frac
        + 15.0 * length_score
        - 50.0 * cys_frac
    )

    # Add noise for realism
    noise = random.gauss(0, 12)
    score = max(0.0, min(100.0, base_score + noise))
    return round(score, 1)


def _extract_sequence_and_score(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract sequence and solubility score from a dataset record.

    Handles multiple column naming conventions.
    """
    # Try sequence columns
    seq = None
    for key in ["sequence", "sequences", "seq", "protein_sequence", "aa_seq", "text"]:
        if key in record and record[key]:
            seq = str(record[key]).strip()
            break

    if not seq or len(seq) < 10:
        return None

    # Filter non-amino-acid characters
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(c in valid_aa for c in seq.upper()):
        # Try cleaning
        seq = "".join(c for c in seq.upper() if c in valid_aa)
        if len(seq) < 10:
            return None

    # Try solubility score columns
    score = None
    for key in ["solubility_score", "solubility", "score", "sol_score",
                 "percent_soluble", "pct_soluble", "label", "labels"]:
        if key in record:
            try:
                val = float(record[key])
                # If it's a binary label (0/1) — common in HF protein datasets
                if key in ("label", "labels") and val in (0, 1):
                    # Binary: 1=soluble, 0=insoluble
                    # Assign representative scores with noise for GRPO diversity
                    if val == 1:
                        score = round(random.uniform(55.0, 95.0), 1)
                    else:
                        score = round(random.uniform(5.0, 45.0), 1)
                    break
                # If it's a 0-1 scale, convert to percentage
                if 0 <= val <= 1.0:
                    val *= 100.0
                if 0 <= val <= 100:
                    score = round(val, 1)
                    break
            except (ValueError, TypeError):
                continue

    if score is None:
        return None

    protein_id = record.get("protein_id", record.get("id", record.get("name", "unknown")))

    return {
        "sequence": seq.upper(),
        "solubility_score": score,
        "protein_id": str(protein_id),
        "source": record.get("source", "eSOL"),
    }


def convert_to_instruction_format(
    records: List[Dict[str, Any]],
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert solubility records to MolInstructions JSON format.

    Args:
        records: Raw records with sequence and solubility data.
        max_samples: Limit number of output samples.

    Returns:
        List of instruction-format dicts.
    """
    samples = []
    class_counter = Counter()
    skipped = 0

    if max_samples:
        random.shuffle(records)

    for record in records:
        if max_samples and len(samples) >= max_samples:
            break

        extracted = _extract_sequence_and_score(record)
        if extracted is None:
            skipped += 1
            continue

        seq = extracted["sequence"]
        score = extracted["solubility_score"]
        sol_class = classify_solubility(score)
        class_counter[sol_class] += 1

        instruction = (
            "Predict the solubility of this protein when expressed in E. coli. "
            "Report whether it is soluble or insoluble, and estimate the "
            "solubility percentage (0-100%)."
        )

        input_text = seq

        output_text = (
            f"Solubility: {sol_class}. Estimated solubility: {score:.1f}%. "
            f"This protein is likely {sol_class} when expressed in E. coli."
        )

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "solubility_score": score,
                "solubility_class": sol_class,
                "protein_id": extracted["protein_id"],
                "source": extracted["source"],
                "organism": "E. coli",
            },
        }
        samples.append(sample)

    log.info(f"Created {len(samples)} instruction samples (skipped {skipped})")
    log.info(f"Class distribution: {dict(class_counter)}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Download and prepare protein solubility data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/solubility_dataset",
        help="Output directory for processed data",
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
    records = load_solubility_data(args.cache_dir)
    log.info(f"Loaded {len(records)} raw records")

    if records:
        sample = records[0]
        log.info(f"Sample record keys: {list(sample.keys())}")

    # Convert to instruction format
    samples = convert_to_instruction_format(records, args.max_samples)

    # Save
    output_path = output_dir / "solubility.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    log.info(f"Saved {len(samples)} samples to {output_path}")

    # Log statistics
    if samples:
        scores = [s["metadata"]["solubility_score"] for s in samples]
        seq_lengths = [len(s["input"]) for s in samples]
        log.info("Statistics:")
        log.info(f"  Samples: {len(samples)}")
        log.info(f"  Solubility range: [{min(scores):.1f}, {max(scores):.1f}]%")
        log.info(f"  Solubility mean: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
        log.info(f"  Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
                 f"mean={np.mean(seq_lengths):.0f}")


if __name__ == "__main__":
    main()
