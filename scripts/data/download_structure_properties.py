"""Generate structural property prediction training data using ESMFold.

Takes protein sequences, runs ESMFold to predict structures, then extracts
per-residue structural properties (secondary structure, dihedrals, SASA,
contacts) using biotite. Outputs JSON instruction format compatible with
MolInstructionsDataset.

Supports three task variants (--task):
- composition: Predict SS composition fractions + mean RSA (Option A reward)
- per_residue: Predict per-residue SS3 string (Option B reward)
- full: Predict comprehensive structural properties (Option C reward)

Usage:
    # Generate 10K samples with full properties (requires GPU)
    python scripts/data/download_structure_properties.py --max_samples 10000

    # Composition-only task (smaller output, faster reward)
    python scripts/data/download_structure_properties.py --task composition --max_samples 10000

    # Per-residue SS prediction task
    python scripts/data/download_structure_properties.py --task per_residue --max_samples 10000

    # Use existing SFT data as protein source
    python scripts/data/download_structure_properties.py --source sft_data \
        --sft_data_dir data/processed/combined_sft_260225_arrow

    # Custom output directory
    python scripts/data/download_structure_properties.py --output_dir data/processed/structure_props
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# Task-specific instruction prompts
TASK_INSTRUCTIONS = {
    "composition": (
        "Analyze the structural properties of this protein. "
        "Report the secondary structure composition as percentages "
        "(helix, sheet, coil), mean relative solvent accessibility (RSA), "
        "and the fraction of buried residues."
    ),
    "per_residue": (
        "Predict the per-residue secondary structure of this protein. "
        "Use H for helix, E for sheet, and C for coil. "
        "Report the full SS3 string matching the sequence length, "
        "followed by the overall composition percentages."
    ),
    "full": (
        "Provide a comprehensive structural analysis of this protein. "
        "Report: (1) secondary structure composition and per-residue SS3 string, "
        "(2) Ramachandran region distribution, (3) mean relative solvent "
        "accessibility and burial fraction, (4) number of long-range contacts "
        "and contact density, (5) overall fold confidence (pLDDT)."
    ),
}


def load_sequences_from_huggingface(
    max_sequences: int, cache_dir: Optional[str] = None
) -> List[Dict[str, str]]:
    """Load protein sequences from HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    log.info("Loading protein sequences from HuggingFace...")
    ds = load_dataset(
        "nikolayvV/protein-function-prediction-preprocessed",
        data_files={"train": "Train/data_train.csv"},
        cache_dir=cache_dir,
    )

    proteins = []
    for split_name in ds:
        for item in ds[split_name]:
            pid = item.get("ProteinID", "")
            seq = item.get("Sequence", "")
            if pid and seq and 20 <= len(seq) <= 1024:
                proteins.append({"id": pid, "sequence": seq})

    log.info(f"Found {len(proteins)} valid proteins (20-1024 residues)")

    if len(proteins) > max_sequences:
        proteins = random.sample(proteins, max_sequences)
    return proteins


def load_sequences_from_sft_data(
    sft_data_dir: str, max_sequences: int
) -> List[Dict[str, str]]:
    """Load protein sequences from existing SFT training data."""
    data_dir = Path(sft_data_dir)

    # Try Arrow format first
    arrow_dir = data_dir / "train"
    if arrow_dir.exists():
        try:
            from datasets import load_from_disk

            log.info(f"Loading from Arrow: {data_dir}")
            ds = load_from_disk(str(data_dir))
            if "train" in ds:
                ds = ds["train"]
        except Exception:
            arrow_dir = None

    # Fallback to JSON
    if not arrow_dir or not arrow_dir.exists():
        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No data files in {data_dir}")

        log.info(f"Loading from JSON: {len(json_files)} files")
        all_items = []
        for jf in json_files:
            with open(jf) as f:
                all_items.extend(json.load(f))
        ds = all_items

    # Extract unique protein sequences
    seen = set()
    proteins = []
    for item in ds:
        inp = item.get("input", "") if isinstance(item, dict) else ""
        # Extract protein sequence (may be wrapped in <protein> tags)
        seq = inp.strip()
        if seq.startswith("<protein>"):
            seq = seq.replace("<protein>", "").replace("</protein>", "").strip()

        # Filter: only standard amino acids, reasonable length
        if seq and 20 <= len(seq) <= 1024 and seq not in seen:
            valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
            if all(c in valid_aas for c in seq):
                seen.add(seq)
                proteins.append({"id": f"sft_{len(proteins)}", "sequence": seq})

    log.info(f"Found {len(proteins)} unique protein sequences from SFT data")

    if len(proteins) > max_sequences:
        proteins = random.sample(proteins, max_sequences)
    return proteins


def format_composition_output(props: Dict[str, Any]) -> str:
    """Format ground truth for composition task (Option A)."""
    h = props.get("helix_fraction", 0)
    e = props.get("sheet_fraction", 0)
    c = props.get("coil_fraction", 0)
    rsa = props.get("mean_rsa", 0)
    buried = props.get("buried_fraction", 0)

    return (
        f"Secondary structure composition: "
        f"{h*100:.1f}% helix, {e*100:.1f}% sheet, {c*100:.1f}% coil. "
        f"Mean RSA: {rsa:.2f}. Buried fraction: {buried*100:.1f}%."
    )


def format_per_residue_output(props: Dict[str, Any]) -> str:
    """Format ground truth for per-residue SS task (Option B)."""
    ss3 = props.get("ss3_string", "")
    h = props.get("helix_fraction", 0)
    e = props.get("sheet_fraction", 0)
    c = props.get("coil_fraction", 0)

    return (
        f"SS3: {ss3}\n"
        f"Composition: {h*100:.1f}% helix, {e*100:.1f}% sheet, {c*100:.1f}% coil."
    )


def format_full_output(props: Dict[str, Any]) -> str:
    """Format ground truth for full structural analysis task (Option C)."""
    from src.utils.structure_properties import properties_to_summary
    return properties_to_summary(props)


def run_esmfold_pipeline(
    proteins: List[Dict[str, str]],
    task: str,
) -> List[Dict[str, Any]]:
    """Run ESMFold on proteins and extract structural properties.

    Args:
        proteins: List of {"id": str, "sequence": str}.
        task: Task variant ("composition", "per_residue", "full").

    Returns:
        List of instruction-format samples.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ESMFold requires GPU.")

    from src.models.esmfold_wrapper import get_esmfold_predictor
    from src.utils.structure_properties import extract_properties

    predictor = get_esmfold_predictor()
    instruction = TASK_INSTRUCTIONS[task]

    format_fn = {
        "composition": format_composition_output,
        "per_residue": format_per_residue_output,
        "full": format_full_output,
    }[task]

    samples = []
    failed = 0
    category_counter = Counter()

    for i, protein in enumerate(proteins):
        if (i + 1) % 50 == 0:
            log.info(
                f"Progress: {i+1}/{len(proteins)} "
                f"({len(samples)} successful, {failed} failed)"
            )

        seq = protein["sequence"]

        try:
            full_output = predictor.predict_full(seq)
            props = extract_properties(full_output)

            if "error" in props:
                failed += 1
                continue

            output_text = format_fn(props)

            # Classify for distribution stats
            h = props.get("helix_fraction", 0)
            e = props.get("sheet_fraction", 0)
            if h > 0.4:
                dominant = "helix-rich"
            elif e > 0.3:
                dominant = "sheet-rich"
            elif h > 0.15 and e > 0.15:
                dominant = "mixed"
            else:
                dominant = "coil-rich"
            category_counter[dominant] += 1

            # Build metadata (store all props for reward computation)
            metadata = {
                "protein_id": protein["id"],
                "task": task,
                "plddt": round(props["plddt"], 2),
                "ptm": round(props["ptm"], 3),
                "helix_fraction": props["helix_fraction"],
                "sheet_fraction": props["sheet_fraction"],
                "coil_fraction": props["coil_fraction"],
                "mean_rsa": props["mean_rsa"],
                "buried_fraction": props["buried_fraction"],
                "num_contacts": props["num_contacts"],
                "contact_density": props["contact_density"],
                "dominant_ss": dominant,
            }

            # Add ramachandran fractions
            if "ramachandran" in props:
                metadata["ramachandran"] = props["ramachandran"]

            # For per_residue and full tasks, include SS3 string in metadata
            if task in ("per_residue", "full"):
                metadata["ss3_string"] = props.get("ss3_string", "")

            sample = {
                "instruction": instruction,
                "input": seq,
                "output": output_text,
                "metadata": metadata,
            }
            samples.append(sample)

        except Exception as e:
            logger_msg = str(e)[:200]
            log.warning(f"Failed protein {protein['id']} (len={len(seq)}): {logger_msg}")
            failed += 1

    log.info(f"Pipeline complete: {len(samples)} successful, {failed} failed")
    log.info(f"SS distribution: {dict(category_counter)}")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate structural property prediction training data"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="full",
        choices=["composition", "per_residue", "full"],
        help="Task variant: composition (Option A), per_residue (Option B), full (Option C)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "sft_data"],
        help="Protein sequence source",
    )
    parser.add_argument(
        "--sft_data_dir",
        type=str,
        default="data/processed/combined_sft_260225_arrow",
        help="SFT data directory (when --source sft_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/structure_properties",
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
        default=10000,
        help="Maximum number of output samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load protein sequences
    if args.source == "sft_data":
        proteins = load_sequences_from_sft_data(args.sft_data_dir, args.max_samples)
    else:
        proteins = load_sequences_from_huggingface(args.max_samples, args.cache_dir)

    if not proteins:
        log.error("No protein sequences loaded. Exiting.")
        sys.exit(1)

    log.info(f"Loaded {len(proteins)} protein sequences")

    # Run ESMFold pipeline
    samples = run_esmfold_pipeline(proteins, args.task)

    if not samples:
        log.error("No samples generated. Exiting.")
        sys.exit(1)

    # Save
    output_filename = f"structure_properties_{args.task}.json"
    output_path = output_dir / output_filename
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    log.info(f"Saved {len(samples)} samples to {output_path}")

    # Statistics
    plddt_values = [s["metadata"]["plddt"] for s in samples]
    seq_lengths = [len(s["input"]) for s in samples]
    helix_fracs = [s["metadata"]["helix_fraction"] for s in samples]
    sheet_fracs = [s["metadata"]["sheet_fraction"] for s in samples]

    log.info("Statistics:")
    log.info(f"  Samples: {len(samples)}")
    log.info(f"  Task: {args.task}")
    log.info(
        f"  Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, "
        f"mean={sum(seq_lengths)/len(seq_lengths):.0f}"
    )
    log.info(
        f"  pLDDT: min={min(plddt_values):.1f}, max={max(plddt_values):.1f}, "
        f"mean={sum(plddt_values)/len(plddt_values):.1f}"
    )
    log.info(
        f"  Helix fraction: mean={sum(helix_fracs)/len(helix_fracs):.3f}"
    )
    log.info(
        f"  Sheet fraction: mean={sum(sheet_fracs)/len(sheet_fracs):.3f}"
    )


if __name__ == "__main__":
    main()
