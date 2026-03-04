#!/usr/bin/env python3
"""One-time Arrow preprocessing for combined SFT dataset.

Reads all JSON files from the combined data directory, applies temperature-based
upsampling, computes approximate token lengths, splits into train/val/test, and
saves as Arrow datasets via ``save_to_disk()``.  Subsequent training loads use
memory-mapped Arrow files (<5s) instead of JSON parsing (~10min).

Usage:
    python scripts/prepare_arrow.py \
        --input data/processed/combined_sft_260225 \
        --output data/processed/combined_sft_260225_arrow \
        --sampling-temperature 0.7 \
        --max-protein-length 1024 \
        --exclude mol_protein_design.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from pathlib import Path

from datasets import Dataset

# Add project root to path so we can import src.data.protein_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_json_files(
    input_dir: Path, exclude: set[str]
) -> dict[str, list[dict]]:
    """Load all JSON files grouped by source prefix."""
    source_groups: dict[str, list] = {}
    for json_file in sorted(input_dir.glob("*.json")):
        if json_file.name in exclude:
            logger.info(f"  {json_file.name}: EXCLUDED")
            continue
        with open(json_file) as f:
            records = json.load(f)
        if not isinstance(records, list):
            logger.info(f"  {json_file.name}: skipped (not a record list)")
            continue
        prefix_match = re.match(r"^([a-z]+)_", json_file.name)
        source = prefix_match.group(1) if prefix_match else "other"
        for record in records:
            record["__source__"] = source
            record["__filename__"] = json_file.stem
        source_groups.setdefault(source, [])
        source_groups[source].extend(records)
        logger.info(f"  {json_file.name}: {len(records):,} samples (source: {source})")
    return source_groups


def apply_temperature_sampling(
    source_groups: dict[str, list], alpha: float, seed: int = 42
) -> list[dict]:
    """Upsample smaller sources using temperature-based sampling."""
    if alpha >= 1.0 or len(source_groups) <= 1:
        all_records = []
        for records in source_groups.values():
            all_records.extend(records)
        return all_records

    sizes = {src: len(recs) for src, recs in source_groups.items()}

    # Compute target proportions: p_i ∝ n_i^α
    raw_weights = {src: n ** alpha for src, n in sizes.items()}
    total_weight = sum(raw_weights.values())
    target_props = {src: w / total_weight for src, w in raw_weights.items()}

    # Normalize so largest source stays at 1x
    max_source = max(sizes, key=lambda s: sizes[s])
    baseline_ratio = target_props[max_source] / sizes[max_source]

    all_records = []
    rng = random.Random(seed)
    for src, records in source_groups.items():
        factor = (target_props[src] / sizes[src]) / baseline_ratio
        full_repeats = int(factor)
        fractional = factor - full_repeats

        repeated = records * full_repeats
        if fractional > 0:
            extra_count = int(math.ceil(fractional * len(records)))
            extra = rng.sample(records, min(extra_count, len(records)))
            repeated.extend(extra)

        logger.info(
            f"  Source '{src}': {sizes[src]:,} -> {len(repeated):,} "
            f"(x{factor:.1f}, target {target_props[src]:.1%})"
        )
        all_records.extend(repeated)

    return all_records


def protein_seq_length(text: str) -> int:
    """Extract protein sequence length from input text.

    Delegates to the canonical implementation in src.data.protein_utils.
    """
    from src.data.protein_utils import protein_sequence_length as _psl
    return _psl(text)


def compute_length(record: dict) -> int:
    """Approximate token length: (chars of instruction+input+output) // 4."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    output = record.get("output", "")
    return (len(instruction) + len(input_text) + len(output)) // 4


def main():
    parser = argparse.ArgumentParser(
        description="Convert combined SFT JSON files to Arrow format"
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input directory with JSON files"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output directory for Arrow datasets"
    )
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=1.0,
        help="Temperature for multi-source balancing (default: 1.0, no rebalancing)",
    )
    parser.add_argument(
        "--max-protein-length",
        type=int,
        default=None,
        help="Filter proteins longer than this (default: no filter)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="JSON filenames to exclude (e.g., mol_protein_design.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train-split", type=float, default=0.9, help="Train split ratio"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.05, help="Validation split ratio"
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    t0 = time.time()

    # 1. Load JSON files
    logger.info(f"Loading JSON files from {args.input}")
    exclude = set(args.exclude)
    source_groups = load_json_files(args.input, exclude)
    if not source_groups:
        logger.error("No JSON files found!")
        sys.exit(1)

    total_raw = sum(len(v) for v in source_groups.values())
    logger.info(f"Raw records: {total_raw:,}")

    # 2. Apply temperature sampling
    logger.info(f"Applying temperature sampling (alpha={args.sampling_temperature})")
    all_records = apply_temperature_sampling(
        source_groups, args.sampling_temperature, args.seed
    )
    logger.info(f"After sampling: {len(all_records):,} records")

    # 3. Compute __length__ column
    logger.info("Computing __length__ column...")
    for record in all_records:
        record["__length__"] = compute_length(record)

    # 4. Filter by max protein length
    if args.max_protein_length:
        before = len(all_records)
        all_records = [
            r
            for r in all_records
            if protein_seq_length(r.get("input", "")) <= args.max_protein_length
        ]
        dropped = before - len(all_records)
        logger.info(
            f"Filtered {dropped:,} records with protein > {args.max_protein_length} AA "
            f"({dropped / before * 100:.1f}% dropped, {len(all_records):,} remaining)"
        )

    # 5. Convert to HF Dataset
    logger.info("Converting to HuggingFace Dataset...")
    full_dataset = Dataset.from_list(all_records)
    logger.info(f"Dataset: {len(full_dataset):,} rows, columns: {full_dataset.column_names}")

    # 6. Shuffle and split
    logger.info(f"Shuffling (seed={args.seed}) and splitting...")
    full_dataset = full_dataset.shuffle(seed=args.seed)

    total = len(full_dataset)
    train_end = int(total * args.train_split)
    val_end = train_end + int(total * args.val_split)

    splits = {
        "train": full_dataset.select(range(train_end)),
        "validation": full_dataset.select(range(train_end, val_end)),
        "test": full_dataset.select(range(val_end, total)),
    }

    for name, ds in splits.items():
        logger.info(f"  {name}: {len(ds):,} samples")

    # 7. Save to disk
    args.output.mkdir(parents=True, exist_ok=True)
    for name, ds in splits.items():
        split_dir = args.output / name
        logger.info(f"Saving {name} to {split_dir}")
        ds.save_to_disk(str(split_dir))

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s. Arrow dataset saved to {args.output}")


if __name__ == "__main__":
    main()
