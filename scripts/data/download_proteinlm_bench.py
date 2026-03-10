"""
Download and convert ProteinLMBench multiple-choice questions for GRPO training.

Downloads from HuggingFace (tsynbio/ProteinLMBench) and converts to the
instruction/input/output/metadata JSON format used by downstream task datasets.

Usage:
    python scripts/data/download_proteinlm_bench.py
    python scripts/data/download_proteinlm_bench.py --max_samples 500
    python scripts/data/download_proteinlm_bench.py --output_dir data/processed/proteinlm_bench
"""

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def load_proteinlm_bench_from_huggingface(
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load ProteinLMBench dataset from HuggingFace.

    Downloads 944 multiple-choice questions covering protein knowledge:
    enzyme catalysis, protein function, PTMs, subunit structure, etc.

    Args:
        cache_dir: HuggingFace cache directory.
        max_samples: Limit number of samples.

    Returns:
        List of raw records from the dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    log.info("Downloading ProteinLMBench from HuggingFace (tsynbio/ProteinLMBench)...")
    try:
        ds = load_dataset(
            "tsynbio/ProteinLMBench",
            "evaluation",
            split="train",
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(f"Could not load ProteinLMBench: {e}")

    records = [dict(ds[i]) for i in range(len(ds))]
    log.info(f"Downloaded {len(records)} records from HuggingFace")

    if max_samples and len(records) > max_samples:
        records = random.sample(records, max_samples)
        log.info(f"Sampled {max_samples} records")

    return records


def convert_to_instruction_format(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert ProteinLMBench records to instruction/input/output/metadata format.

    Each question becomes an instruction-format dict compatible with
    MolInstructionsDataset and the GRPO trainer reward pipeline.

    Args:
        records: Raw records from the HuggingFace dataset.

    Returns:
        List of instruction-format dicts.
    """
    samples = []
    option_count_dist = Counter()

    for i, record in enumerate(records):
        question = record.get("question", "")
        options = record.get("options", [])
        answer = record.get("answer", "")
        explanation = record.get("explanation", "")

        if not question or not options:
            continue

        # Parse correct answer index from "option N" format
        match = re.match(r"option\s+(\d+)", answer, re.IGNORECASE)
        if not match:
            log.warning(f"Could not parse answer '{answer}' for record {i}, skipping")
            continue
        correct_idx = int(match.group(1)) - 1
        num_options = len(options)
        option_count_dist[num_options] += 1

        # Build option text (clean "option N: " prefix)
        option_lines = []
        for j, opt in enumerate(options):
            clean = re.sub(r"^option\s+\d+\s*:\s*", "", opt)
            option_lines.append(f"{j + 1}) {clean}")
        options_text = "\n".join(option_lines)

        # Instruction: the question with options
        instruction = (
            f"{question}\n\n{options_text}\n\n"
            f"Answer with just the option number (e.g., \"option 1\")."
        )

        # Input: empty (no protein sequence for MC questions)
        input_text = ""

        # Output: the correct answer in "option N" format
        output_text = answer

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "sample_id": f"plb_{i:04d}",
                "correct_answer": answer,
                "correct_index": correct_idx,
                "num_options": num_options,
                "explanation": explanation,
                "source": "tsynbio/ProteinLMBench",
            },
        }
        samples.append(sample)

    log.info(f"Created {len(samples)} instruction samples")
    log.info(f"Option count distribution: {dict(option_count_dist)}")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert ProteinLMBench for GRPO training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/proteinlm_bench",
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
        help="Limit number of output samples (null = use all ~944)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace
    records = load_proteinlm_bench_from_huggingface(args.cache_dir, args.max_samples)

    # Convert to instruction format
    samples = convert_to_instruction_format(records)

    # Save
    output_path = output_dir / "proteinlm_bench.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    log.info(f"Saved {len(samples)} samples to {output_path}")

    # Log statistics
    if samples:
        num_options = [s["metadata"]["num_options"] for s in samples]
        log.info("Statistics:")
        log.info(f"  Total samples: {len(samples)}")
        log.info(f"  Options range: [{min(num_options)}, {max(num_options)}]")
        log.info(f"  Mean options: {sum(num_options)/len(num_options):.1f}")


if __name__ == "__main__":
    main()
