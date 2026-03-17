#!/usr/bin/env python3
"""Pre-compute ESM-3 embeddings and store in LMDB cache.

Scans dataset JSON/Arrow files, extracts unique protein sequences, runs
ESM-3 forward pass in batches, and stores per-residue embeddings in LMDB.

Eliminates ESM-3 GPU memory (~6 GB) and ~40% of forward pass time during
SFT/GRPO training.

Supports both SFT data (protein in ``input`` field) and SSL data
(``<seq>M K M R F...</seq>`` tags in conversation ``output`` field).

Usage:
    # Single data dir:
    python scripts/precompute_esm_embeddings.py \
        --data-dir data/processed/combined_sft_260225 \
        --output data/esm_cache/all_proteins.lmdb

    # Multiple data dirs (SFT + SSL, deduplicated):
    python scripts/precompute_esm_embeddings.py \
        --data-dir data/processed/combined_sft_260225 \
                   data/processed/combined_ssl_260316 \
        --output data/esm_cache/all_proteins.lmdb

    # Multi-GPU (torchrun):
    torchrun --nproc_per_node=8 scripts/precompute_esm_embeddings.py \
        --data-dir data/processed/combined_sft_260225 \
                   data/processed/combined_ssl_260316 \
        --output data/esm_cache/all_proteins.lmdb

    # Resume (skip existing keys):
    python scripts/precompute_esm_embeddings.py \
        --data-dir data/processed/combined_sft_260225 \
        --output data/esm_cache/all_proteins.lmdb \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _stream_json_records(json_file: Path):
    """Stream records from a JSON file using ijson for memory efficiency.

    Falls back to full json.load() if ijson is not available.
    """
    try:
        import ijson

        with open(json_file, "rb") as f:
            for record in ijson.items(f, "item"):
                yield record
    except ImportError:
        # Fallback: load entire file (works for smaller files)
        logger.warning(f"ijson not available, loading {json_file.name} fully into memory")
        with open(json_file) as f:
            records = json.load(f)
        if isinstance(records, list):
            yield from records


def extract_sequences_from_json(
    data_dir: Path, max_protein_length: int
) -> set[str]:
    """Extract unique protein sequences from JSON files.

    Handles two formats:
    - SFT format: {"input": "<protein sequence>", "output": "..."}
    - SSL conversation format: {"conversation": [{"output": "...<seq>M K...</seq>..."}]}
    """
    from src.data.protein_utils import extract_protein_sequence, extract_seq_tag_proteins

    sequences = set()
    for json_file in sorted(data_dir.glob("*.json")):
        logger.info(f"  Scanning {json_file.name}...")
        count_before = len(sequences)

        for record in _stream_json_records(json_file):
            # SFT format: protein in "input" field
            input_text = record.get("input", "")
            if input_text:
                seq = extract_protein_sequence(input_text)
                if seq and len(seq) <= max_protein_length:
                    # Verify it's actually a protein (not just text)
                    from src.data.protein_utils import AA_CHARS
                    if all(c in AA_CHARS for c in seq.upper()):
                        sequences.add(seq)

            # SSL conversation format: <seq>...</seq> tags in output
            conversation = record.get("conversation", [])
            for turn in conversation:
                output_text = turn.get("output", "")
                if "<seq>" in output_text:
                    seq_proteins = extract_seq_tag_proteins(output_text)
                    for seq in seq_proteins:
                        if len(seq) <= max_protein_length:
                            sequences.add(seq)

        added = len(sequences) - count_before
        logger.info(
            f"  {json_file.name}: +{added:,} new sequences "
            f"({len(sequences):,} total unique)"
        )
    return sequences


def extract_sequences_from_arrow(
    data_dir: Path, max_protein_length: int
) -> set[str]:
    """Extract unique protein sequences from Arrow dataset."""
    from datasets import load_from_disk

    from src.data.protein_utils import AA_CHARS, extract_protein_sequence, extract_seq_tag_proteins

    sequences = set()
    arrow_dir = Path(str(data_dir) + "_arrow")

    for split in ["train", "validation", "test"]:
        split_dir = arrow_dir / split
        if not split_dir.exists():
            continue
        logger.info(f"  Loading Arrow split: {split}")
        ds = load_from_disk(str(split_dir))
        for i in range(len(ds)):
            row = ds[i]
            # SFT format
            input_text = row.get("input", "")
            if input_text:
                seq = extract_protein_sequence(input_text)
                if seq and len(seq) <= max_protein_length:
                    if all(c in AA_CHARS for c in seq.upper()):
                        sequences.add(seq)
            # SSL format: check "text" or "output" for <seq> tags
            for field in ["text", "output"]:
                text = row.get(field, "")
                if text and "<seq>" in text:
                    for seq in extract_seq_tag_proteins(text):
                        if len(seq) <= max_protein_length:
                            sequences.add(seq)
        logger.info(f"  {split}: {len(sequences):,} unique sequences so far")

    return sequences


def extract_sequences_from_dir(
    data_dir: Path, max_protein_length: int
) -> set[str]:
    """Extract sequences from a data directory (Arrow preferred, JSON fallback)."""
    arrow_dir = Path(str(data_dir) + "_arrow")
    if arrow_dir.exists():
        logger.info(f"  Using Arrow format from {arrow_dir}")
        return extract_sequences_from_arrow(data_dir, max_protein_length)
    else:
        logger.info(f"  Using JSON format from {data_dir}")
        return extract_sequences_from_json(data_dir, max_protein_length)


def _merge_shards(output_path: str, world_size: int, map_size_gb: int):
    """Merge per-rank LMDB shards into a single output database."""
    import shutil


    logger.info(f"Merging {world_size} LMDB shards into {output_path}")

    try:
        import lmdb
    except ImportError:
        logger.error("lmdb not available for merge")
        return

    # Open/create the merged database
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    merged_env = lmdb.open(
        str(output_path),
        map_size=map_size_gb * 1024**3,
        readahead=False,
        meminit=False,
    )

    total_merged = 0
    for shard_rank in range(world_size):
        shard_path = f"{output_path}.shard{shard_rank}"
        if not Path(shard_path).exists():
            logger.warning(f"Shard {shard_rank} not found at {shard_path}")
            continue

        shard_env = lmdb.open(str(shard_path), readonly=True, readahead=True, lock=False)
        shard_count = shard_env.stat()["entries"]
        logger.info(f"  Shard {shard_rank}: {shard_count:,} entries")

        with shard_env.begin(write=False) as src_txn:
            with merged_env.begin(write=True) as dst_txn:
                cursor = src_txn.cursor()
                for key, value in cursor:
                    dst_txn.put(key, value)
                    total_merged += 1

        shard_env.close()
        # Remove shard after merge
        shutil.rmtree(shard_path)
        logger.info(f"  Removed shard {shard_rank}")

    merged_env.close()
    logger.info(f"Merged total: {total_merged:,} entries at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute ESM-3 embeddings into LMDB cache"
    )
    parser.add_argument(
        "--data-dir",
        nargs="+",
        type=Path,
        default=[],
        help="One or more data directories with JSON files (sequences deduplicated across all)",
    )
    parser.add_argument(
        "--protein-list",
        type=Path,
        default=None,
        help="Pre-computed text file with one protein sequence per line (skips extraction)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output LMDB path (e.g., data/esm_cache/all_proteins.lmdb)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="ESM-3 encoding batch size (default: 4)",
    )
    parser.add_argument(
        "--max-protein-length",
        type=int,
        default=1024,
        help="Skip proteins longer than this (default: 1024)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sequences already in cache",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm3-sm-open-v1",
        help="ESM-3 model name",
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=100,
        help="LMDB max map size in GB (default: 100)",
    )
    args = parser.parse_args()

    if not args.data_dir and not args.protein_list:
        parser.error("Must specify --data-dir or --protein-list")

    for d in args.data_dir:
        if not d.is_dir():
            logger.error(f"Data directory not found: {d}")
            sys.exit(1)

    # Multi-GPU setup
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        import torch
        import torch.distributed as dist

        # Set device BEFORE init_process_group to avoid "duplicate GPU" NCCL error
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        logger.info(f"Rank {rank}/{world_size}, GPU {local_rank}")

    t0 = time.time()

    # 1. Extract unique protein sequences
    if rank == 0:
        if args.protein_list:
            # Load pre-computed list (one sequence per line)
            logger.info(f"Loading protein list from {args.protein_list}")
            with open(args.protein_list) as f:
                all_sequences = [
                    line.strip()
                    for line in f
                    if line.strip() and len(line.strip()) <= args.max_protein_length
                ]
            logger.info(f"Loaded {len(all_sequences):,} sequences from file")
        else:
            sequences = set()
            for data_dir in args.data_dir:
                logger.info(f"Extracting protein sequences from {data_dir}")
                dir_seqs = extract_sequences_from_dir(
                    data_dir, args.max_protein_length
                )
                logger.info(f"  {data_dir.name}: {len(dir_seqs):,} unique sequences")
                sequences |= dir_seqs

            logger.info(f"Total unique sequences across all dirs: {len(sequences):,}")
            all_sequences = sorted(sequences)  # Deterministic order
    else:
        all_sequences = None

    # Broadcast sequence list in multi-GPU mode
    if world_size > 1:
        import torch
        import torch.distributed as dist

        if rank == 0:
            # Serialize and broadcast
            seq_bytes = json.dumps(all_sequences).encode()
            size_tensor = torch.tensor([len(seq_bytes)], dtype=torch.long).cuda()
        else:
            size_tensor = torch.tensor([0], dtype=torch.long).cuda()

        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()

        if rank == 0:
            data_tensor = torch.frombuffer(
                bytearray(seq_bytes), dtype=torch.uint8
            ).cuda()
        else:
            data_tensor = torch.zeros(size, dtype=torch.uint8).cuda()

        dist.broadcast(data_tensor, src=0)

        if rank != 0:
            all_sequences = json.loads(
                data_tensor.cpu().numpy().tobytes().decode()
            )

    # Sort by length so batches have similar-length proteins (less padding waste)
    all_sequences.sort(key=len)

    # 2. Shard sequences across GPUs (interleaved for balanced lengths)
    my_sequences = all_sequences[rank::world_size]
    logger.info(
        f"Rank {rank}: processing {len(my_sequences):,} sequences "
        f"(of {len(all_sequences):,} total)"
    )

    # 3. Open LMDB cache (per-rank shards for multi-GPU, single for single-GPU)
    from src.data.esm_embedding_cache import ESMEmbeddingCache

    if world_size > 1:
        # Each rank writes its own shard to avoid LMDB single-writer contention
        cache_path = f"{args.output}.shard{rank}"
    else:
        cache_path = args.output

    cache = ESMEmbeddingCache(
        cache_path,
        map_size=args.map_size_gb * 1024**3,
        readonly=False,
    )

    # Filter already-cached sequences if resuming
    if args.resume:
        before = len(my_sequences)
        # For multi-GPU resume, also check the merged output
        if world_size > 1 and Path(args.output).exists():
            merged_cache = ESMEmbeddingCache(args.output, readonly=True)
            my_sequences = [
                seq for seq in my_sequences
                if seq not in cache and seq not in merged_cache
            ]
            merged_cache.close()
        else:
            my_sequences = [seq for seq in my_sequences if seq not in cache]
        skipped = before - len(my_sequences)
        if skipped > 0:
            logger.info(f"Rank {rank}: skipped {skipped:,} cached sequences")

    if not my_sequences:
        logger.info(f"Rank {rank}: nothing to compute")
        cache.close()
        if world_size > 1:
            import torch.distributed as dist

            dist.barrier()
            # Rank 0 merges shards
            if rank == 0:
                _merge_shards(args.output, world_size, args.map_size_gb)
            dist.barrier()
            dist.destroy_process_group()
        return

    # 4. Load ESM-3 encoder
    import torch

    if world_size == 1:
        torch.cuda.set_device(local_rank)

    from src.models.protein_encoder import ESM3ProteinEncoder

    encoder = ESM3ProteinEncoder(
        model_name=args.model,
        device=f"cuda:{local_rank}",
        dtype="bfloat16",
        encoder_batch_size=args.batch_size,
    )

    # 5. Encode and cache
    batch_size = args.batch_size
    total = len(my_sequences)
    processed = 0
    t_start = time.time()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_seqs = my_sequences[start:end]

        # Encode
        with torch.no_grad():
            output = encoder.encode(batch_seqs)
        embeddings = output["embeddings"]  # [B, L_max, 1536]

        # Store each sequence's embedding (trimmed to actual length)
        for i, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            emb = embeddings[i, :seq_len, :]  # [L_i, 1536]
            cache.put(seq, emb)

        processed += len(batch_seqs)
        if processed % (batch_size * 50) == 0 or processed == total:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            logger.info(
                f"Rank {rank}: {processed:,}/{total:,} "
                f"({processed / total * 100:.1f}%, {rate:.1f} seq/s, "
                f"ETA {eta / 60:.1f}min)"
            )

    cache.close()
    total_elapsed = time.time() - t0
    logger.info(
        f"Rank {rank}: done. {processed:,} embeddings computed in {total_elapsed:.1f}s"
    )

    # Cleanup and merge
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

        # Rank 0 merges all shards into the final output
        if rank == 0:
            _merge_shards(args.output, world_size, args.map_size_gb)

        dist.barrier()
        dist.destroy_process_group()
    else:
        if rank == 0:
            cache_final = ESMEmbeddingCache(args.output, readonly=True)
            logger.info(
                f"Cache total: {len(cache_final):,} entries at {args.output}"
            )
            cache_final.close()


if __name__ == "__main__":
    main()
