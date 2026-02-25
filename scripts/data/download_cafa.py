"""
Download and convert CAFA 5 GO term prediction data.

Downloads CAFA 5 protein function prediction data from HuggingFace,
parses FASTA sequences and GO term annotations, and converts to our
JSON instruction format for RL training.

Usage:
    python scripts/data/download_cafa.py
    python scripts/data/download_cafa.py --output_dir data/processed/cafa5_go
    python scripts/data/download_cafa.py --max_samples 10000
"""

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# GO aspect mapping based on GO ID prefix ranges (simplified)
# For production, use the GO OBO file for accurate namespace assignment
GO_ASPECT_MAP = {
    # Molecular Function (MF)
    "GO:0003": "MF", "GO:0004": "MF", "GO:0016": "MF", "GO:0015": "MF",
    "GO:0140": "MF", "GO:0060": "MF",
    # Biological Process (BP)
    "GO:0006": "BP", "GO:0007": "BP", "GO:0008": "BP", "GO:0009": "BP",
    "GO:0010": "BP", "GO:0019": "BP", "GO:0030": "BP", "GO:0031": "BP",
    "GO:0032": "BP", "GO:0033": "BP", "GO:0034": "BP", "GO:0035": "BP",
    "GO:0036": "BP", "GO:0042": "BP", "GO:0043": "BP", "GO:0044": "BP",
    "GO:0045": "BP", "GO:0046": "BP", "GO:0048": "BP", "GO:0050": "BP",
    "GO:0051": "BP", "GO:0055": "BP", "GO:0061": "BP", "GO:0065": "BP",
    "GO:0070": "BP", "GO:0071": "BP", "GO:0072": "BP", "GO:0097": "BP",
    "GO:0098": "BP", "GO:0099": "BP", "GO:0140": "BP",
    # Cellular Component (CC)
    "GO:0005": "CC", "GO:0012": "CC", "GO:0014": "CC", "GO:0016": "CC",
    "GO:0019": "CC", "GO:0031": "CC", "GO:0032": "CC", "GO:0033": "CC",
    "GO:0043": "CC", "GO:0044": "CC", "GO:0045": "CC", "GO:0048": "CC",
    "GO:0055": "CC", "GO:0062": "CC", "GO:0070": "CC", "GO:0098": "CC",
    "GO:0099": "CC", "GO:0110": "CC", "GO:0120": "CC",
}


def infer_go_aspect(go_term: str) -> str:
    """Infer GO aspect (MF/BP/CC) from term ID prefix.

    This is a heuristic. For accurate assignment, use the GO OBO ontology
    file with goatools or obonet. Many GO ID ranges overlap between aspects.
    """
    prefix = go_term[:7]  # e.g., "GO:0003"
    return GO_ASPECT_MAP.get(prefix, "unknown")


def parse_cafa_tsv(tsv_path: str) -> Dict[str, List[str]]:
    """Parse CAFA train_terms.tsv: protein_id -> list of GO terms."""
    protein_go = defaultdict(list)
    with open(tsv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                protein_id = parts[0].strip()
                go_term = parts[1].strip()
                if re.match(r"GO:\d{7}", go_term):
                    protein_go[protein_id].append(go_term)
    return dict(protein_go)


def parse_cafa_fasta(fasta_path: str) -> Dict[str, str]:
    """Parse CAFA train_sequences.fasta: protein_id -> sequence."""
    proteins = {}
    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    proteins[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id and current_seq:
        proteins[current_id] = "".join(current_seq)

    return proteins


def load_cafa5_from_huggingface(cache_dir: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Load CAFA 5 dataset from HuggingFace.

    Returns:
        Tuple of (protein_sequences, protein_go_terms).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    log.info("Loading protein function prediction data from HuggingFace...")

    # Primary source: nikolayvV preprocessed dataset (GO terms by aspect)
    # Load only data_train.csv (skip features_train.csv which has different columns)
    try:
        ds = load_dataset(
            "nikolayvV/protein-function-prediction-preprocessed",
            data_files={"train": "Train/data_train.csv"},
            cache_dir=cache_dir,
        )
        log.info(f"Loaded dataset with splits: {list(ds.keys())}")

        proteins = {}
        protein_go = defaultdict(list)

        for split_name in ds:
            split = ds[split_name]
            for item in split:
                pid = item.get("ProteinID", item.get("EntryID", ""))
                seq = item.get("Sequence", item.get("sequence", ""))

                if not pid or not seq:
                    continue

                proteins[pid] = seq

                # Collect GO terms from all three aspects (BPO, CCO, MFO)
                for aspect_key in ["BPO", "CCO", "MFO"]:
                    terms = item.get(aspect_key, [])
                    if isinstance(terms, list):
                        protein_go[pid].extend(terms)
                    elif isinstance(terms, str):
                        found = re.findall(r"GO:\d{7}", terms)
                        protein_go[pid].extend(found)

        if proteins and protein_go:
            log.info(f"Loaded {len(proteins)} proteins with GO annotations")
            return proteins, dict(protein_go)

    except Exception as e:
        log.warning(f"Could not load nikolayvV dataset: {e}")

    # Fallback: try AmelieSchreiber version
    try:
        ds = load_dataset(
            "AmelieSchreiber/cafa_5_protein_function_prediction",
            cache_dir=cache_dir,
        )
        log.info(f"Loaded AmelieSchreiber with splits: {list(ds.keys())}")

        proteins = {}
        protein_go = defaultdict(list)

        for split_name in ds:
            split = ds[split_name]
            for item in split:
                pid = item.get("EntryID", item.get("protein_id", ""))
                seq = item.get("Sequence", item.get("sequence", ""))
                go_terms = item.get("GO_terms", item.get("go_terms", []))

                if not pid or not seq:
                    continue

                proteins[pid] = seq
                if isinstance(go_terms, str):
                    terms = re.findall(r"GO:\d{7}", go_terms)
                    protein_go[pid].extend(terms)
                elif isinstance(go_terms, list):
                    protein_go[pid].extend(go_terms)

        return proteins, dict(protein_go)

    except Exception as e:
        log.warning(f"Could not load AmelieSchreiber: {e}")
        raise RuntimeError(
            "Could not load protein function data. Tried nikolayvV and AmelieSchreiber datasets. "
            "Use --local_dir to load from local CAFA 5 files instead."
        )


def load_cafa5_from_local(data_dir: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Load CAFA 5 from local files (downloaded from Kaggle).

    Expects:
        data_dir/train_sequences.fasta
        data_dir/train_terms.tsv
    """
    data_dir = Path(data_dir)
    fasta_path = data_dir / "train_sequences.fasta"
    tsv_path = data_dir / "train_terms.tsv"

    if not fasta_path.exists() or not tsv_path.exists():
        raise FileNotFoundError(
            f"Expected {fasta_path} and {tsv_path}. "
            "Download from https://www.kaggle.com/competitions/cafa-5-protein-function-prediction"
        )

    log.info(f"Loading sequences from {fasta_path}")
    proteins = parse_cafa_fasta(str(fasta_path))
    log.info(f"Loaded {len(proteins)} protein sequences")

    log.info(f"Loading GO terms from {tsv_path}")
    protein_go = parse_cafa_tsv(str(tsv_path))
    log.info(f"Loaded GO annotations for {len(protein_go)} proteins")

    return proteins, protein_go


def convert_to_instruction_format(
    proteins: Dict[str, str],
    protein_go: Dict[str, List[str]],
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert CAFA data to JSON instruction format.

    Args:
        proteins: protein_id -> sequence mapping.
        protein_go: protein_id -> list of GO terms.
        max_samples: Limit number of output samples.

    Returns:
        List of instruction-format dicts.
    """
    # Only include proteins that have both sequence and GO annotations
    import random

    valid_ids = sorted(set(proteins.keys()) & set(protein_go.keys()))
    log.info(f"Proteins with both sequence and GO terms: {len(valid_ids)}")

    if max_samples and len(valid_ids) > max_samples:
        valid_ids = sorted(random.sample(valid_ids, max_samples))

    samples = []
    aspect_counter = Counter()

    for pid in valid_ids:
        seq = proteins[pid]
        go_terms = sorted(set(protein_go[pid]))

        if not go_terms:
            continue

        # Categorize GO terms by aspect
        aspects = []
        for term in go_terms:
            aspect = infer_go_aspect(term)
            aspects.append(aspect)
            aspect_counter[aspect] += 1

        # Format output: comma-separated GO terms
        output_text = ", ".join(go_terms)

        sample = {
            "instruction": (
                "Predict the Gene Ontology (GO) terms for this protein. "
                "List molecular functions (MF), biological processes (BP), "
                "and cellular components (CC). Format: GO:XXXXXXX"
            ),
            "input": seq,
            "output": output_text,
            "metadata": {
                "protein_id": pid,
                "go_aspect": aspects,
                "num_terms": len(go_terms),
                "source": "cafa5",
            },
        }
        samples.append(sample)

    log.info(f"Created {len(samples)} instruction samples")
    log.info(f"GO aspect distribution: {dict(aspect_counter)}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Download and convert CAFA 5 GO data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/cafa5_go",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Path to local CAFA 5 data (with train_sequences.fasta and train_terms.tsv)",
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
    if args.local_dir:
        proteins, protein_go = load_cafa5_from_local(args.local_dir)
    else:
        proteins, protein_go = load_cafa5_from_huggingface(args.cache_dir)

    # Convert to instruction format
    samples = convert_to_instruction_format(proteins, protein_go, args.max_samples)

    # Save
    output_path = output_dir / "go_prediction.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    log.info(f"Saved {len(samples)} samples to {output_path}")

    # Log statistics
    num_terms = [s["metadata"]["num_terms"] for s in samples]
    seq_lengths = [len(s["input"]) for s in samples]
    log.info("Statistics:")
    log.info(f"  Samples: {len(samples)}")
    log.info(f"  GO terms per protein: min={min(num_terms)}, max={max(num_terms)}, "
             f"mean={sum(num_terms)/len(num_terms):.1f}")
    log.info(f"  Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
             f"mean={sum(seq_lengths)/len(seq_lengths):.1f}")


if __name__ == "__main__":
    main()
