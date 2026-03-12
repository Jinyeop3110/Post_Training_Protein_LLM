#!/usr/bin/env python3
"""Extract protein embeddings from ESM3 + AttentionPooling + MLP projector.

Produces three embedding sets:
  1. esm3_pooled        — ESM3 per-residue embeddings, attention-pooled to (N, 32, 1536)
  2. trained_projected  — pooled embeddings projected by trained MLP   (N, 32, output_dim)
  3. random_projected   — pooled embeddings projected by random MLP    (N, 32, output_dim)

Loads proteins from the Arrow dataset with explicit train/validation splits.
Each protein is tagged with its task type (from metadata.task) and split.

Usage
-----
  # Quick test (100 proteins from train)
  python scripts/analysis/extract_embeddings.py \
      --experiment sft_esm3_mlp_combined_qwen3_8b_it_0306_010408 \
      --n_samples 100 --output_dir blog/data/03-10/embeddings

  # Full extraction: 10K train + 10K validation
  python scripts/analysis/extract_embeddings.py \
      --experiment sft_esm3_mlp_combined_qwen3_8b_it_0306_010408 \
      --n_samples 10000 --split train

  python scripts/analysis/extract_embeddings.py \
      --experiment sft_esm3_mlp_combined_qwen3_8b_it_0306_010408 \
      --n_samples 10000 --split validation

  # Both splits combined (default)
  python scripts/analysis/extract_embeddings.py \
      --experiment sft_esm3_mlp_combined_qwen3_8b_it_0306_010408 \
      --n_samples 10000 --split both
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pooling import AttentionPooling
from src.models.projector import MLPProjector
from src.models.protein_encoder import ESM3ProteinEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_embeddings")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_from_metadata(record: dict) -> str:
    """Extract task type from a dataset record.

    Priority: metadata.task > __filename__ > 'unknown'
    """
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("task"):
        return metadata["task"]
    # Fallback: derive from __filename__
    fname = record.get("__filename__", "")
    if fname:
        # Strip common prefixes: mol_, pd_, plm_, sp_, p2t_, clap_
        name = fname
        for prefix in ("mol_", "pd_", "plm_", "sp_", "p2t_", "clap_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name
    return "unknown"


def _extract_sequence(record: dict) -> str:
    """Extract protein sequence from a dataset record.

    The sequence is typically in the 'input' field, wrapped in backticks.
    """
    text = record.get("input", "") or ""
    # Pattern: ```\nSEQUENCE\n``` or just a raw sequence
    match = re.search(r"```\s*\n?([A-Z]+)\n?\s*```", text)
    if match:
        return match.group(1)
    # Fallback: look for a stretch of uppercase amino acid letters
    match = re.search(r"\b([ACDEFGHIKLMNPQRSTVWY]{20,})\b", text)
    if match:
        return match.group(1)
    return ""


def load_proteins_from_arrow(
    arrow_dir: Path, split: str, n_samples: int, seed: int = 42,
) -> list[dict]:
    """Load protein sequences from Arrow dataset split.

    Returns a list of dicts with keys:
        protein_sequence, task_type, source_file, split, index, seq_len, accession
    """
    from datasets import load_from_disk

    split_dir = arrow_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Arrow split not found: {split_dir}")

    logger.info(f"  Loading Arrow dataset: {split_dir}")
    ds = load_from_disk(str(split_dir))
    logger.info(f"  Total records in {split}: {len(ds):,}")

    # Random sample of indices
    rng = np.random.RandomState(seed)
    # Over-sample to account for invalid sequences
    n_try = min(int(n_samples * 1.5), len(ds))
    indices = rng.choice(len(ds), n_try, replace=False)

    proteins = []
    seen_seqs = set()
    for idx in indices:
        if len(proteins) >= n_samples:
            break
        record = ds[int(idx)]
        seq = _extract_sequence(record)
        if not seq or len(seq) < 20 or len(seq) > 1024:
            continue
        if seq in seen_seqs:
            continue
        seen_seqs.add(seq)

        metadata = record.get("metadata", {}) or {}
        proteins.append({
            "protein_sequence": seq,
            "task_type": _task_from_metadata(record),
            "source_file": record.get("__filename__", ""),
            "split": split,
            "index": int(idx),
            "seq_len": len(seq),
            "accession": metadata.get("protein_accession", "") or "",
        })

    logger.info(f"  Extracted {len(proteins)} unique proteins from {split} "
                f"(20-1024 AA, seed={seed})")

    # Log task distribution
    from collections import Counter
    task_counts = Counter(p["task_type"] for p in proteins)
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1])[:15]:
        logger.info(f"    {task}: {count}")
    if len(task_counts) > 15:
        logger.info(f"    ... and {len(task_counts) - 15} more task types")

    return proteins


def load_proteins_from_json(
    data_dir: Path, n_samples: int, seed: int = 42,
) -> list[dict]:
    """Fallback: load protein sequences from JSON files.

    Returns same format as load_proteins_from_arrow.
    """
    import random
    rng = random.Random(seed)

    all_proteins = []
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    for jf in json_files:
        logger.info(f"  Scanning {jf.name} ...")
        with open(jf) as f:
            records = json.load(f)

        for idx, rec in enumerate(records):
            seq = _extract_sequence(rec)
            if seq and 20 <= len(seq) <= 1024:
                all_proteins.append({
                    "protein_sequence": seq,
                    "task_type": _task_from_metadata(rec),
                    "source_file": jf.name,
                    "split": "unknown",
                    "index": idx,
                    "seq_len": len(seq),
                    "accession": (rec.get("metadata", {}) or {}).get(
                        "protein_accession", ""),
                })

    logger.info(f"  Found {len(all_proteins)} valid proteins (20-1024 AA)")

    # Deduplicate by sequence
    seen = set()
    unique = []
    for p in all_proteins:
        if p["protein_sequence"] not in seen:
            seen.add(p["protein_sequence"])
            unique.append(p)
    logger.info(f"  {len(unique)} unique sequences after deduplication")

    if n_samples < len(unique):
        rng.shuffle(unique)
        unique = unique[:n_samples]
        logger.info(f"  Sampled {n_samples} proteins (seed={seed})")

    return unique


def load_proteins_from_downstream(
    data_dir: Path, datasets: list[str], n_samples: int, seed: int = 42,
) -> list[dict]:
    """Load protein sequences from downstream task JSON files.

    Supported datasets:
      - stability: data/processed/megascale_stability/stability.json
        label = stability_class (neutral/stabilizing/destabilizing)
      - structure: data/processed/structure_quality/structure_quality.json
        label = fold_category (high/medium/low)
      - go: data/processed/cafa5_go/go_prediction.json
        label = go (GO prediction task)

    Returns same format as load_proteins_from_arrow, with task_type set to
    the biological label (e.g., "stability:neutral", "structure:high").
    """
    dataset_configs = {
        "stability": {
            "path": data_dir / "megascale_stability" / "stability.json",
            "label_key": "stability_class",
            "prefix": "stability",
        },
        "structure": {
            "path": data_dir / "structure_quality" / "structure_quality.json",
            "label_key": "fold_category",
            "prefix": "structure",
        },
        "go": {
            "path": data_dir / "cafa5_go" / "go_prediction.json",
            "label_key": None,  # All same label — use dataset name
            "prefix": "go",
        },
    }

    rng = np.random.RandomState(seed)
    all_proteins = []

    for ds_name in datasets:
        if ds_name not in dataset_configs:
            logger.warning(f"Unknown downstream dataset: {ds_name}")
            continue

        cfg = dataset_configs[ds_name]
        path = cfg["path"]
        if not path.exists():
            logger.error(f"  Dataset not found: {path}")
            continue

        logger.info(f"\n--- Loading downstream: {ds_name} ---")
        with open(path) as f:
            records = json.load(f)
        logger.info(f"  Total records: {len(records)}")

        # Extract sequences and labels
        valid = []
        seen_seqs = set()
        for idx, rec in enumerate(records):
            seq = _extract_sequence(rec)
            if not seq or len(seq) < 20 or len(seq) > 1024:
                continue
            if seq in seen_seqs:
                continue
            seen_seqs.add(seq)

            metadata = rec.get("metadata", {}) or {}
            label_key = cfg["label_key"]
            if label_key and label_key in metadata:
                label = f"{cfg['prefix']}:{metadata[label_key]}"
            else:
                label = cfg["prefix"]

            valid.append({
                "protein_sequence": seq,
                "task_type": label,
                "source_file": ds_name,
                "split": "downstream",
                "index": idx,
                "seq_len": len(seq),
                "accession": metadata.get("protein_id", "") or "",
            })

        logger.info(f"  Valid proteins: {len(valid)} (20-1024 AA, unique)")

        # Sample n_samples per dataset
        if n_samples < len(valid):
            indices = rng.choice(len(valid), n_samples, replace=False)
            valid = [valid[i] for i in sorted(indices)]
            logger.info(f"  Sampled {n_samples} proteins (seed={seed})")

        # Log label distribution
        from collections import Counter
        label_counts = Counter(p["task_type"] for p in valid)
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {label}: {count}")

        all_proteins.extend(valid)

    return all_proteins


def resolve_checkpoint_dir(experiment_dir: Path, step: int = None) -> Path:
    """Find the checkpoint directory (latest step or specific step)."""
    ckpt_root = experiment_dir / "checkpoints"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoints/ in {experiment_dir}")

    if step is not None:
        target = ckpt_root / f"checkpoint-{step}"
        if not target.exists():
            raise FileNotFoundError(f"Checkpoint step {step} not found at {target}")
        return target

    # Find latest checkpoint by step number
    ckpt_dirs = sorted(
        ckpt_root.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoint-* dirs in {ckpt_root}")
    return ckpt_dirs[-1]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def build_pooling(encoder_embed_dim: int, num_output_tokens: int, device: str) -> AttentionPooling:
    """Create an AttentionPooling module (for random baseline)."""
    return AttentionPooling(
        embed_dim=encoder_embed_dim,
        num_output_tokens=num_output_tokens,
        num_heads=8,
        dropout=0.1,
        layer_norm=True,
    ).to(device)


def build_projector(
    input_dim: int, hidden_dim: int, output_dim: int,
    num_layers: int, dropout: float, device: str,
) -> MLPProjector:
    """Create an MLPProjector module (for random baseline)."""
    return MLPProjector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        activation="gelu",
        dropout=dropout,
    ).to(device)


@torch.no_grad()
def extract_embeddings(
    sequences: list[str],
    encoder: ESM3ProteinEncoder,
    pooling_trained: AttentionPooling,
    projector_trained: MLPProjector,
    pooling_random: AttentionPooling,
    projector_random: MLPProjector,
    encoder_batch_size: int = 4,
    device: str = "cuda:0",
) -> dict:
    """Run the full extraction pipeline.

    Returns dict with:
      esm3_pooled:         (N, 32, 1536)  -- attention-pooled ESM3 embeddings (trained pooling)
      esm3_pooled_random:  (N, 32, 1536)  -- attention-pooled ESM3 embeddings (random pooling)
      trained_projected:   (N, 32, dim)   -- after trained projector
      random_projected:    (N, 32, dim)   -- after random projector
    """
    all_esm3_pooled = []
    all_esm3_pooled_random = []
    all_trained = []
    all_random = []

    N = len(sequences)
    logger.info(f"Extracting embeddings for {N} proteins ...")

    for start in range(0, N, encoder_batch_size):
        end = min(start + encoder_batch_size, N)
        batch_seqs = sequences[start:end]

        # 1. ESM3 encode (frozen, bf16 autocast inside encoder)
        encoder_output = encoder.encode(batch_seqs)
        raw_embeds = encoder_output["embeddings"]  # [B, L, 1536] on GPU

        # 2. Trained pooling + projection
        pooled_trained = pooling_trained(raw_embeds)          # [B, 32, 1536]
        projected_trained = projector_trained(pooled_trained)  # [B, 32, dim]

        # 3. Random pooling + projection
        pooled_random = pooling_random(raw_embeds)            # [B, 32, 1536]
        projected_random = projector_random(pooled_random)     # [B, 32, dim]

        all_esm3_pooled.append(pooled_trained.cpu().float())
        all_esm3_pooled_random.append(pooled_random.cpu().float())
        all_trained.append(projected_trained.cpu().float())
        all_random.append(projected_random.cpu().float())

        # Free GPU memory between batches
        del raw_embeds, pooled_trained, projected_trained, pooled_random, projected_random
        if end % 200 == 0:
            torch.cuda.empty_cache()

        if (end % 200 == 0) or end == N:
            logger.info(f"  Processed {end}/{N} proteins")

    return {
        "esm3_pooled": torch.cat(all_esm3_pooled, dim=0).numpy(),
        "esm3_pooled_random": torch.cat(all_esm3_pooled_random, dim=0).numpy(),
        "trained_projected": torch.cat(all_trained, dim=0).numpy(),
        "random_projected": torch.cat(all_random, dim=0).numpy(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract protein embeddings from ESM3 + pooling + projector",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="sft_esm3_mlp_combined_qwen3_8b_it_0306_010408",
        help="Experiment name under results/",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Specific checkpoint step (default: latest)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of proteins per split (default: 10000)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "validation", "both"],
        help="Which dataset split(s) to use (default: both)",
    )
    parser.add_argument(
        "--encoder_batch_size",
        type=int,
        default=4,
        help="ESM3 sub-batch size (default: 4)",
    )
    parser.add_argument(
        "--arrow_dir",
        type=str,
        default=None,
        help="Path to Arrow dataset (default: data/processed/combined_sft_260225_arrow)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Fallback: path to JSON data dir (used only if Arrow not found)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: blog/data/03-10/embeddings)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for protein sampling",
    )
    parser.add_argument(
        "--downstream",
        nargs="*",
        default=None,
        help="Use downstream datasets instead of SFT data. "
             "Options: stability, structure, go (default: all three)",
    )
    args = parser.parse_args()

    # If --downstream given with no args, use all three
    if args.downstream is not None and len(args.downstream) == 0:
        args.downstream = ["stability", "structure", "go"]

    # Resolve paths
    results_dir = PROJECT_ROOT / "results"
    experiment_dir = results_dir / args.experiment
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    arrow_dir = Path(args.arrow_dir) if args.arrow_dir else (
        PROJECT_ROOT / "data" / "processed" / "combined_sft_260225_arrow"
    )
    data_dir = Path(args.data_dir) if args.data_dir else (
        PROJECT_ROOT / "data" / "processed" / "combined_sft_260225"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "blog" / "data" / "03-10" / "embeddings"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------------
    # 1. Read experiment config
    # -----------------------------------------------------------------------
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(experiment_dir / "config.yaml")

    encoder_embed_dim = cfg.encoder.embedding_dim        # 1536
    num_output_tokens = cfg.encoder.pooling.num_output_tokens  # 32
    proj_hidden_dim = cfg.encoder.projector.hidden_dim   # 5120
    proj_output_dim = cfg.encoder.projector.output_dim   # output_dim
    proj_num_layers = cfg.encoder.projector.num_layers   # 2
    proj_dropout = cfg.encoder.projector.dropout         # 0.1

    data_mode = "downstream" if args.downstream else "sft"

    logger.info("=" * 60)
    logger.info("Embedding Extraction")
    logger.info("=" * 60)
    logger.info(f"Experiment:     {args.experiment}")
    logger.info(f"Data mode:      {data_mode}")
    if args.downstream:
        logger.info(f"Downstream:     {args.downstream}")
    else:
        logger.info(f"Split:          {args.split}")
    logger.info(f"N samples/split:{args.n_samples}")
    logger.info(f"Encoder dim:    {encoder_embed_dim}")
    logger.info(f"Pool tokens:    {num_output_tokens}")
    logger.info(f"Projector:      {encoder_embed_dim} -> {proj_hidden_dim} -> {proj_output_dim}")
    logger.info(f"Device:         {device}")

    # -----------------------------------------------------------------------
    # 2. Load protein sequences
    # -----------------------------------------------------------------------
    logger.info("\nLoading proteins ...")

    all_proteins = []

    if args.downstream:
        # Load from downstream task datasets
        processed_dir = Path(args.data_dir) if args.data_dir else (
            PROJECT_ROOT / "data" / "processed"
        )
        all_proteins = load_proteins_from_downstream(
            processed_dir, args.downstream, args.n_samples, seed=args.seed,
        )
    else:
        # Load from SFT dataset (Arrow or JSON)
        splits_to_load = (
            ["train", "validation"] if args.split == "both"
            else [args.split]
        )

        use_arrow = arrow_dir.exists()

        for split_name in splits_to_load:
            if use_arrow:
                logger.info(f"\n--- Loading {split_name} from Arrow ---")
                proteins = load_proteins_from_arrow(
                    arrow_dir, split_name, args.n_samples, seed=args.seed,
                )
            else:
                logger.info("\n--- Arrow not found, falling back to JSON ---")
                proteins = load_proteins_from_json(
                    data_dir, args.n_samples, seed=args.seed,
                )
            all_proteins.extend(proteins)

    sequences = [p["protein_sequence"] for p in all_proteins]
    logger.info(f"\nTotal proteins: {len(sequences)}")

    # Log split distribution
    from collections import Counter
    split_counts = Counter(p["split"] for p in all_proteins)
    for s, c in split_counts.items():
        logger.info(f"  {s}: {c}")

    # -----------------------------------------------------------------------
    # 3. Load ESM3 encoder (frozen)
    # -----------------------------------------------------------------------
    logger.info("\nLoading ESM3 encoder ...")
    t0 = time.time()
    encoder = ESM3ProteinEncoder(
        model_name=cfg.encoder.model_name,
        device=device,
        dtype="bfloat16",
        encoder_batch_size=args.encoder_batch_size,
    )
    # Force load now
    encoder._load_model()
    logger.info(f"ESM3 loaded in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # 4. Build TRAINED pooling + projector (load weights from checkpoint)
    # -----------------------------------------------------------------------
    ckpt_dir = resolve_checkpoint_dir(experiment_dir, args.checkpoint_step)
    logger.info(f"\nLoading trained weights from: {ckpt_dir.name}")

    pooling_trained = build_pooling(encoder_embed_dim, num_output_tokens, device)
    projector_trained = build_projector(
        encoder_embed_dim, proj_hidden_dim, proj_output_dim,
        proj_num_layers, proj_dropout, device,
    )

    # Load trained state dicts
    pooling_path = ckpt_dir / "pooling.pt"
    projector_path = ckpt_dir / "projector.pt"

    if not pooling_path.exists():
        logger.error(f"pooling.pt not found in {ckpt_dir}")
        sys.exit(1)
    if not projector_path.exists():
        logger.error(f"projector.pt not found in {ckpt_dir}")
        sys.exit(1)

    # Handle _orig_mod. prefix from FSDP + torch.compile checkpoints
    def _load_state_dict(path, module, name):
        state = torch.load(path, map_location=device, weights_only=True)
        # Strip _orig_mod. prefix if present
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            logger.info(f"  Stripped _orig_mod. prefix from {name}")
        module.load_state_dict(state)
        logger.info(f"  Loaded {name}: {sum(p.numel() for p in module.parameters()):,} params")

    _load_state_dict(pooling_path, pooling_trained, "pooling (trained)")
    _load_state_dict(projector_path, projector_trained, "projector (trained)")

    pooling_trained.eval()
    projector_trained.eval()

    # -----------------------------------------------------------------------
    # 5. Build RANDOM pooling + projector (same architecture, fresh init)
    # -----------------------------------------------------------------------
    logger.info("\nBuilding random baseline (same architecture, random init) ...")
    # Use a fixed seed for the random baseline so it is reproducible
    torch.manual_seed(0)
    pooling_random = build_pooling(encoder_embed_dim, num_output_tokens, device)
    projector_random = build_projector(
        encoder_embed_dim, proj_hidden_dim, proj_output_dim,
        proj_num_layers, proj_dropout, device,
    )
    pooling_random.eval()
    projector_random.eval()
    logger.info(f"  Random pooling:    {sum(p.numel() for p in pooling_random.parameters()):,} params")
    logger.info(f"  Random projector:  {sum(p.numel() for p in projector_random.parameters()):,} params")

    # -----------------------------------------------------------------------
    # 6. Extract embeddings
    # -----------------------------------------------------------------------
    logger.info("\nExtracting embeddings ...")
    t0 = time.time()
    results = extract_embeddings(
        sequences=sequences,
        encoder=encoder,
        pooling_trained=pooling_trained,
        projector_trained=projector_trained,
        pooling_random=pooling_random,
        projector_random=projector_random,
        encoder_batch_size=args.encoder_batch_size,
        device=device,
    )
    elapsed = time.time() - t0
    logger.info(f"Extraction done in {elapsed:.1f}s ({elapsed / len(sequences):.2f}s/protein)")

    # -----------------------------------------------------------------------
    # 7. Save outputs
    # -----------------------------------------------------------------------
    logger.info(f"\nSaving to {output_dir} ...")

    # Embeddings as numpy arrays
    for name, arr in results.items():
        path = output_dir / f"{name}.npy"
        np.save(str(path), arr)
        logger.info(f"  {name}.npy  shape={arr.shape}  dtype={arr.dtype}")

    # Protein identifiers (with task_type and split)
    protein_ids = []
    for p in all_proteins:
        protein_ids.append({
            "accession": p["accession"],
            "seq_len": p["seq_len"],
            "source_file": p["source_file"],
            "task_type": p["task_type"],
            "split": p["split"],
        })
    with open(output_dir / "protein_ids.json", "w") as f:
        json.dump(protein_ids, f, indent=2)

    # Metadata
    metadata = {
        "experiment": args.experiment,
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_step": ckpt_dir.name,
        "num_proteins": len(sequences),
        "splits": dict(split_counts),
        "encoder": cfg.encoder.model_name,
        "encoder_embed_dim": encoder_embed_dim,
        "num_output_tokens": num_output_tokens,
        "projector_hidden_dim": proj_hidden_dim,
        "projector_output_dim": proj_output_dim,
        "projector_num_layers": proj_num_layers,
        "device": device,
        "seed": args.seed,
        "extraction_time_sec": round(elapsed, 1),
        "shapes": {k: list(v.shape) for k, v in results.items()},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\nDone.")
    logger.info(f"Output directory: {output_dir}")
    for name, arr in results.items():
        logger.info(f"  {name}: {arr.shape}")


if __name__ == "__main__":
    main()
