#!/usr/bin/env python3
"""One-time Arrow preprocessing for SSL pre-training dataset.

Reads conversation-format JSON files from the SSL data directory, extracts the
``output`` field as plain text, computes approximate token lengths, splits into
train/validation, and saves as Arrow datasets via ``save_to_disk()``.

The SSL data uses ProteinLMDataset format:
    [{"conversation": [{"system": "", "input": "", "output": "..."}]}, ...]

Handles large files (>2GB) via streaming boundary-based parsing and builds
Arrow datasets incrementally per-file to avoid OOM.

Usage:
    python scripts/prepare_ssl_arrow.py \\
        --input data/processed/combined_ssl_260316 \\
        --output data/processed/combined_ssl_260316_arrow \\
        --train-split 0.98 --val-split 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from datasets import Dataset, concatenate_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_text(record: dict) -> str:
    """Extract plain text from a conversation-format record."""
    conversation = record.get("conversation")
    if conversation and isinstance(conversation, list):
        parts = []
        for turn in conversation:
            output = turn.get("output", "")
            if output:
                parts.append(output)
        if parts:
            return "\n\n".join(parts)

    # Fallback
    if "output" in record:
        return record["output"]
    if "text" in record:
        return record["text"]
    return ""


def _stream_json_records(path: Path):
    """Stream JSON records from a file, yielding one at a time.

    Handles:
    - Files under 2GB: standard json.load() then yield
    - Files over 2GB: boundary-based streaming (detects record starts/ends
      by ``{`` and ``}`` at column 0)
    - Files starting mid-record (no ``[`` prefix, e.g., continuation files)
    - Both ``\\r\\n`` and ``\\n`` line endings
    """
    file_size = path.stat().st_size

    # Fast path for small files
    if file_size < 2 * 1024**3:
        try:
            with open(path) as f:
                records = json.load(f)
            if isinstance(records, list):
                yield from records
            return
        except json.JSONDecodeError as e:
            logger.warning(
                f"  {path.name}: json.load failed at char {e.pos:,} — "
                "switching to streaming parser..."
            )
    else:
        logger.info(
            f"  {path.name}: {file_size / 1024**3:.1f} GB — "
            "using streaming parser"
        )

    # Streaming: split on top-level record boundaries.
    # Record starts: "{" at column 0 (or "[{" for first record)
    # Record ends: "}" at column 0 (possibly followed by "," or "]")
    obj_lines: list[str] = []
    in_record = False
    skipped_header = False
    count = 0

    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

            if not skipped_header:
                stripped = line.lstrip()
                if stripped.startswith("[{"):
                    skipped_header = True
                    in_record = True
                    obj_lines.append(stripped[1:])  # Keep the "{"
                    continue
                elif stripped == "[":
                    skipped_header = True
                    continue
                else:
                    # File starts mid-record (no "[" prefix, e.g., part2)
                    if line and line[0] == "{":
                        skipped_header = True
                        in_record = True
                        obj_lines = ["{"]
                    continue

            # Record boundary: "}" or "}," or "}]" at column 0
            if line and line[0] == "}":
                if in_record:
                    obj_lines.append("}")
                    obj_str = "\n".join(obj_lines)
                    obj_lines = []
                    in_record = False
                    try:
                        record = json.loads(obj_str)
                        yield record
                        count += 1
                    except json.JSONDecodeError:
                        pass

                    if count % 500_000 == 0:
                        logger.info(
                            f"  {path.name}: streaming... {count:,} records"
                        )
            elif line and line[0] == "{":
                in_record = True
                obj_lines = ["{"]
            elif in_record:
                obj_lines.append(line)

    logger.info(f"  {path.name}: {count:,} records total")


def _process_file_to_dataset(
    json_file: Path, min_length: int
) -> Dataset | None:
    """Process a single JSON file into an Arrow Dataset.

    Streams records and builds the dataset in batches to limit memory usage.
    """
    t0 = time.time()
    texts = []
    sources = []
    lengths = []
    source_name = json_file.stem
    batch_datasets = []
    batch_size = 500_000  # Flush to Arrow every 500K records

    for record in _stream_json_records(json_file):
        text = extract_text(record)
        if not text or len(text) < min_length:
            continue

        texts.append(text)
        sources.append(source_name)
        lengths.append(len(text) // 4)

        if len(texts) >= batch_size:
            batch_ds = Dataset.from_dict({
                "text": texts,
                "__source__": sources,
                "__length__": lengths,
            })
            batch_datasets.append(batch_ds)
            logger.info(
                f"  {json_file.name}: flushed batch "
                f"({len(batch_ds):,} records)"
            )
            texts = []
            sources = []
            lengths = []

    # Final batch
    if texts:
        batch_ds = Dataset.from_dict({
            "text": texts,
            "__source__": sources,
            "__length__": lengths,
        })
        batch_datasets.append(batch_ds)

    # Free temporary lists
    del texts, sources, lengths

    if not batch_datasets:
        return None

    # Concatenate batches for this file
    if len(batch_datasets) == 1:
        result = batch_datasets[0]
    else:
        result = concatenate_datasets(batch_datasets)
        del batch_datasets

    elapsed = time.time() - t0
    logger.info(
        f"  {json_file.name}: {len(result):,} samples ({elapsed:.1f}s)"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert SSL JSON files to Arrow format"
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input directory with JSON files"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output directory for Arrow datasets"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="JSON filenames to exclude",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train-split", type=float, default=0.98, help="Train split ratio"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.01, help="Validation split ratio"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum text length in characters (filter noise)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    t0 = time.time()

    # 1. Process each JSON file into an Arrow dataset (incremental, no OOM)
    logger.info(f"Loading JSON files from {args.input}")
    exclude = set(args.exclude)
    file_datasets = []

    for json_file in sorted(args.input.glob("*.json")):
        if json_file.name in exclude:
            logger.info(f"  {json_file.name}: EXCLUDED")
            continue

        logger.info(f"  Processing {json_file.name}...")
        ds = _process_file_to_dataset(json_file, args.min_length)
        if ds is not None:
            file_datasets.append(ds)

    if not file_datasets:
        logger.error("No records found!")
        sys.exit(1)

    # 2. Concatenate all file datasets
    logger.info(f"Concatenating {len(file_datasets)} file datasets...")
    if len(file_datasets) == 1:
        full_dataset = file_datasets[0]
    else:
        full_dataset = concatenate_datasets(file_datasets)
        del file_datasets

    logger.info(
        f"Dataset: {len(full_dataset):,} rows, "
        f"columns: {full_dataset.column_names}"
    )

    # 3. Shuffle and split
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

    # 4. Save to disk
    args.output.mkdir(parents=True, exist_ok=True)
    for name, ds in splits.items():
        split_dir = args.output / name
        logger.info(f"Saving {name} to {split_dir}")
        ds.save_to_disk(str(split_dir))

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s. Arrow dataset saved to {args.output}")


if __name__ == "__main__":
    main()
