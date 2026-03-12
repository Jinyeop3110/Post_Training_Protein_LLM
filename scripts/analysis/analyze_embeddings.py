#!/usr/bin/env python3
"""Analyze extracted protein embeddings: quality metrics, UMAP, classification.

Takes the output of extract_embeddings.py and produces:
  - analysis_results.json   : All metrics (kNN, linear probe, silhouette, CKA, etc.)
  - umap_esm3_raw.npy       : (N, 2) UMAP coordinates for ESM3 pooled embeddings
  - umap_trained.npy         : (N, 2) UMAP coordinates for trained projector
  - umap_random.npy          : (N, 2) UMAP coordinates for random projector
  - labels.npy               : (N,) integer class labels
  - label_names.json         : {int_label: label_name, ...}
  - cosine_distributions.json: Intra/inter-class cosine similarity statistics

Three embedding conditions are compared:
  1. esm3_raw      — ESM3 attention-pooled embeddings (mean over 32 tokens -> N x 1536)
  2. trained       — Trained MLP projector output    (mean over 32 tokens -> N x proj_dim)
  3. random        — Random MLP projector output     (mean over 32 tokens -> N x proj_dim)

Labels are derived from the source_file field in protein_ids.json (task type).

Usage
-----
  python scripts/analysis/analyze_embeddings.py \\
      --embeddings_dir blog/data/03-10/embeddings/ \\
      --label_source auto

  # With explicit label field
  python scripts/analysis/analyze_embeddings.py \\
      --embeddings_dir blog/data/03-10/embeddings/ \\
      --label_source source_file

  # Limit UMAP to speed up
  python scripts/analysis/analyze_embeddings.py \\
      --embeddings_dir blog/data/03-10/embeddings/ \\
      --max_umap_samples 2000
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_embeddings")


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _task_from_source_file(source_file: str) -> str:
    """Map source_file to a clean task label.

    source_file looks like 'mol_general_function.json' -> 'general_function'
    """
    name = source_file.replace(".json", "")
    # Strip common prefixes: mol_, pd_, plm_, sp_, p2t_, clap_
    for prefix in ("mol_", "pd_", "plm_", "sp_", "p2t_", "clap_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def load_labels(protein_ids: list[dict], label_source: str = "auto") -> tuple:
    """Extract labels from protein_ids metadata.

    Returns:
        labels: np.ndarray of int labels (N,)
        label_names: dict mapping int -> string label name
        label_field: str describing what field was used
    """
    if label_source == "auto":
        # Prefer task_type (from Arrow dataset), then source_file
        if all("task_type" in p and p["task_type"] for p in protein_ids):
            label_source = "task_type"
        elif all("source_file" in p for p in protein_ids):
            label_source = "source_file"
        else:
            raise ValueError(
                "Cannot auto-detect labels. protein_ids.json has no task_type or source_file field. "
                "Specify --label_source explicitly."
            )

    if label_source == "source_file":
        raw_labels = [_task_from_source_file(p["source_file"]) for p in protein_ids]
    elif label_source == "task_type":
        raw_labels = [p.get("task_type", "unknown") for p in protein_ids]
    elif label_source == "split":
        raw_labels = [p.get("split", "unknown") for p in protein_ids]
    else:
        # Generic: try to use the specified field directly
        raw_labels = [p.get(label_source, "unknown") for p in protein_ids]

    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    label_names = {int(i): name for i, name in enumerate(le.classes_)}

    logger.info(f"Labels from '{label_source}': {len(label_names)} classes, "
                f"{len(labels)} samples")
    for idx, name in sorted(label_names.items()):
        count = int(np.sum(labels == idx))
        logger.info(f"  [{idx:2d}] {name}: {count}")

    return labels, label_names, label_source


def filter_rare_classes(labels: np.ndarray, min_samples: int = 10) -> np.ndarray:
    """Return boolean mask keeping only classes with >= min_samples."""
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = set(unique[counts >= min_samples])
    mask = np.array([l in valid_classes for l in labels])
    n_dropped = len(labels) - mask.sum()
    if n_dropped > 0:
        logger.info(f"Filtered out {n_dropped} samples from classes with < {min_samples} members")
    return mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def knn_accuracy(embeddings: np.ndarray, labels: np.ndarray,
                 ks: list[int] = None, n_splits: int = 5) -> dict:
    """k-NN classification accuracy via stratified k-fold cross-validation."""
    if ks is None:
        ks = [1, 5, 10, 20]

    results = {}
    for k in ks:
        # Cap k at number of samples in smallest fold
        effective_k = min(k, len(labels) // n_splits - 1)
        if effective_k < 1:
            effective_k = 1

        knn = KNeighborsClassifier(n_neighbors=effective_k, metric="cosine")
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(knn, embeddings, labels, cv=skf, scoring="accuracy")
            results[f"k{k}"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
        except ValueError as e:
            logger.warning(f"kNN k={k} failed: {e}")
            results[f"k{k}"] = {"mean": 0.0, "std": 0.0}

    return results


def linear_probe_accuracy(embeddings: np.ndarray, labels: np.ndarray,
                          n_splits: int = 5) -> dict:
    """Logistic regression linear probe via stratified k-fold CV."""
    lr = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(lr, embeddings, labels, cv=skf, scoring="accuracy")
        return {
            "accuracy": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "per_fold": [float(s) for s in scores],
        }
    except Exception as e:
        logger.warning(f"Linear probe failed: {e}")
        return {"accuracy": 0.0, "std": 0.0, "per_fold": []}


def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray,
                       max_samples: int = 5000) -> float:
    """Silhouette score with cosine distance. Sub-samples if too large."""
    n = len(labels)
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        logger.warning("Silhouette requires >= 2 classes, got %d", n_unique)
        return 0.0

    return float(silhouette_score(embeddings, labels, metric="cosine"))


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment (linear) between two embedding matrices.

    Both X and Y should be (N, D) with the same N.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalized singular values).

    Higher means more dimensions are used (less anisotropic).
    """
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    # Use min(N, D) SVD for efficiency
    s = np.linalg.svd(centered, compute_uv=False)
    s = s[s > 1e-10]  # Remove near-zero
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def isotropy_score(embeddings: np.ndarray, n_pairs: int = 10000) -> float:
    """Average cosine similarity between random pairs.

    Lower = more isotropic (uniform angular distribution).
    """
    n = len(embeddings)
    rng = np.random.RandomState(42)
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    normed = normalize(embeddings, norm="l2")
    cos_sims = np.sum(normed[idx_a] * normed[idx_b], axis=1)
    return float(np.mean(cos_sims))


def cosine_distributions(embeddings: np.ndarray, labels: np.ndarray,
                         n_pairs: int = 20000) -> dict:
    """Compute intra-class and inter-class cosine similarity distributions."""
    n = len(labels)
    rng = np.random.RandomState(42)
    normed = normalize(embeddings, norm="l2")

    intra_sims = []
    inter_sims = []

    # Sample random pairs
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)

    for a, b in zip(idx_a, idx_b):
        if a == b:
            continue
        cos = float(np.dot(normed[a], normed[b]))
        if labels[a] == labels[b]:
            intra_sims.append(cos)
        else:
            inter_sims.append(cos)

    def _stats(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "n": 0}
        arr = np.array(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
        }

    return {
        "intra_class": _stats(intra_sims),
        "inter_class": _stats(inter_sims),
        "separation": (
            float(np.mean(intra_sims) - np.mean(inter_sims))
            if intra_sims and inter_sims else 0.0
        ),
        # Raw arrays for histogram plotting
        "intra_class_values": intra_sims,
        "inter_class_values": inter_sims,
    }


def retrieval_metrics(embeddings: np.ndarray, labels: np.ndarray,
                      ks: list[int] = None, max_samples: int = 2000) -> dict:
    """Compute retrieval metrics: Precision@k and Mean Reciprocal Rank (MRR).

    For each query protein, find nearest neighbors by cosine similarity
    and check if they share the same label.
    """
    if ks is None:
        ks = [1, 5, 10, 20]

    n = len(labels)
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        n = max_samples

    # Cosine distance matrix
    normed = normalize(embeddings, norm="l2")
    # Similarity matrix
    sim_matrix = normed @ normed.T
    # Zero out self-similarity
    np.fill_diagonal(sim_matrix, -1.0)

    # For each sample, sort neighbors by similarity (descending)
    sorted_indices = np.argsort(-sim_matrix, axis=1)

    precision_at_k = {}
    mrr_scores = []

    for k in ks:
        precisions = []
        for i in range(n):
            top_k = sorted_indices[i, :k]
            same_label = np.sum(labels[top_k] == labels[i])
            precisions.append(same_label / k)
        precision_at_k[f"precision_at_{k}"] = float(np.mean(precisions))

    # MRR: rank of first same-label neighbor
    for i in range(n):
        for rank, j in enumerate(sorted_indices[i], start=1):
            if labels[j] == labels[i]:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

    results = dict(precision_at_k)
    results["mrr"] = float(np.mean(mrr_scores))
    return results


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def compute_umap(embeddings: np.ndarray, max_samples: int = None,
                 n_neighbors: int = 15, min_dist: float = 0.1,
                 random_state: int = 42) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute 2D UMAP coordinates.

    Returns:
        coords: (M, 2) UMAP coordinates (M <= N if sub-sampled)
        indices: (M,) indices into original array, or None if no sub-sampling
    """
    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed. Install with: pip install umap-learn")
        return np.zeros((len(embeddings), 2)), None

    indices = None
    if max_samples is not None and len(embeddings) > max_samples:
        logger.info(f"  Sub-sampling {max_samples}/{len(embeddings)} for UMAP")
        rng = np.random.RandomState(random_state)
        indices = np.sort(rng.choice(len(embeddings), max_samples, replace=False))
        embeddings = embeddings[indices]

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        n_jobs=-1,
    )
    coords = reducer.fit_transform(embeddings)
    return coords.astype(np.float32), indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze extracted protein embeddings",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="blog/data/03-10/embeddings",
        help="Directory containing .npy files and protein_ids.json",
    )
    parser.add_argument(
        "--label_source",
        type=str,
        default="auto",
        help="Label source: 'auto', 'source_file', or a field name in protein_ids.json",
    )
    parser.add_argument(
        "--min_class_size",
        type=int,
        default=10,
        help="Minimum samples per class for classification metrics (default: 10)",
    )
    parser.add_argument(
        "--max_umap_samples",
        type=int,
        default=None,
        help="Max samples for UMAP (default: all)",
    )
    parser.add_argument(
        "--skip_umap",
        action="store_true",
        help="Skip UMAP computation (much faster)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=128,
        help="PCA dimensions after flattening 32 tokens (default: 128). "
             "Set to 0 to use mean-pooling instead of flatten+PCA.",
    )
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.exists():
        logger.error(f"Embeddings directory not found: {emb_dir}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Embedding Quality Analysis")
    logger.info("=" * 60)
    logger.info(f"Directory: {emb_dir}")

    # -------------------------------------------------------------------
    # 1. Load embeddings
    # -------------------------------------------------------------------
    logger.info("\n--- Loading embeddings ---")

    def _load_npy(name: str) -> np.ndarray:
        path = emb_dir / f"{name}.npy"
        if not path.exists():
            logger.error(f"Missing: {path}")
            sys.exit(1)
        arr = np.load(str(path))
        logger.info(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
        return arr

    esm3_pooled = _load_npy("esm3_pooled")           # (N, 32, 1536)
    trained_projected = _load_npy("trained_projected") # (N, 32, proj_dim)
    random_projected = _load_npy("random_projected")   # (N, 32, proj_dim)

    # Optionally load esm3_pooled_random for CKA comparison
    esm3_pooled_random_path = emb_dir / "esm3_pooled_random.npy"
    esm3_pooled_random = None
    if esm3_pooled_random_path.exists():
        esm3_pooled_random = np.load(str(esm3_pooled_random_path))
        logger.info(f"  esm3_pooled_random: shape={esm3_pooled_random.shape}")

    N = esm3_pooled.shape[0]

    if args.pca_dim > 0:
        # Flatten 32 tokens -> (N, 32*dim), then PCA reduce
        logger.info(f"\nFlatten + PCA (target dim={args.pca_dim}) ...")
        from sklearn.decomposition import PCA as _PCA

        raw_flat = esm3_pooled.reshape(N, -1)           # (N, 32*1536)
        trained_flat = trained_projected.reshape(N, -1)  # (N, 32*proj_dim)
        random_flat = random_projected.reshape(N, -1)    # (N, 32*proj_dim)

        logger.info(f"  Flattened: esm3_raw={raw_flat.shape}, "
                    f"trained={trained_flat.shape}, random={random_flat.shape}")

        # Fit PCA independently per condition (different dim spaces)
        pca_results = {}
        for name, flat in [("esm3_raw", raw_flat),
                           ("trained", trained_flat),
                           ("random", random_flat)]:
            t0 = time.time()
            n_components = min(args.pca_dim, flat.shape[0], flat.shape[1])
            pca = _PCA(n_components=n_components, random_state=42)
            reduced = pca.fit_transform(flat)
            var_explained = pca.explained_variance_ratio_.sum()
            logger.info(f"  {name}: {flat.shape} -> {reduced.shape}  "
                        f"({var_explained:.1%} variance explained, {time.time() - t0:.1f}s)")
            pca_results[name] = {
                "embeddings": reduced,
                "variance_explained": float(var_explained),
                "n_components": n_components,
            }

        esm3_raw = pca_results["esm3_raw"]["embeddings"]
        trained = pca_results["trained"]["embeddings"]
        random_emb = pca_results["random"]["embeddings"]
    else:
        # Fallback: mean-pool across 32 tokens -> (N, dim)
        logger.info("\nMean-pooling across token dimension ...")
        esm3_raw = esm3_pooled.mean(axis=1)       # (N, 1536)
        trained = trained_projected.mean(axis=1)    # (N, proj_dim)
        random_emb = random_projected.mean(axis=1)  # (N, proj_dim)
        pca_results = None

    logger.info(f"  esm3_raw:  ({N}, {esm3_raw.shape[1]})")
    logger.info(f"  trained:   ({N}, {trained.shape[1]})")
    logger.info(f"  random:    ({N}, {random_emb.shape[1]})")

    conditions = {
        "esm3_raw": esm3_raw,
        "trained": trained,
        "random": random_emb,
    }

    # -------------------------------------------------------------------
    # 2. Load labels
    # -------------------------------------------------------------------
    logger.info("\n--- Loading labels ---")
    with open(emb_dir / "protein_ids.json") as f:
        protein_ids = json.load(f)

    labels, label_names, label_field = load_labels(protein_ids, args.label_source)

    # Filter rare classes for classification metrics
    class_mask = filter_rare_classes(labels, min_samples=args.min_class_size)
    labels_filtered = labels[class_mask]
    conditions_filtered = {k: v[class_mask] for k, v in conditions.items()}
    n_filtered = class_mask.sum()
    logger.info(f"After filtering: {n_filtered}/{N} samples, "
                f"{len(np.unique(labels_filtered))} classes")

    # -------------------------------------------------------------------
    # 3. Compute all metrics
    # -------------------------------------------------------------------
    results = {
        "metadata": {
            "n_samples": N,
            "n_samples_filtered": int(n_filtered),
            "n_classes": len(label_names),
            "n_classes_filtered": int(len(np.unique(labels_filtered))),
            "label_field": label_field,
            "min_class_size": args.min_class_size,
            "n_splits": args.n_splits,
            "embeddings_dir": str(emb_dir),
            "reduction": "flatten_pca" if pca_results else "mean_pool",
            "pca_dim": args.pca_dim if pca_results else None,
            "dimensions": {k: v.shape[1] for k, v in conditions.items()},
        },
    }
    if pca_results:
        results["metadata"]["pca_variance_explained"] = {
            name: pca_results[name]["variance_explained"]
            for name in pca_results
        }

    # --- kNN accuracy ---
    logger.info("\n--- kNN classification ---")
    results["knn_accuracy"] = {}
    for name, emb in conditions_filtered.items():
        logger.info(f"  {name} ...")
        t0 = time.time()
        results["knn_accuracy"][name] = knn_accuracy(
            emb, labels_filtered, ks=[1, 5, 10, 20], n_splits=args.n_splits,
        )
        logger.info(f"    Done in {time.time() - t0:.1f}s")
        for k, v in results["knn_accuracy"][name].items():
            logger.info(f"    {k}: {v['mean']:.4f} +/- {v['std']:.4f}")

    # --- Linear probe ---
    logger.info("\n--- Linear probe (logistic regression) ---")
    results["linear_probe"] = {}
    for name, emb in conditions_filtered.items():
        logger.info(f"  {name} ...")
        t0 = time.time()
        results["linear_probe"][name] = linear_probe_accuracy(
            emb, labels_filtered, n_splits=args.n_splits,
        )
        logger.info(f"    Done in {time.time() - t0:.1f}s")
        lp = results["linear_probe"][name]
        logger.info(f"    accuracy: {lp['accuracy']:.4f} +/- {lp['std']:.4f}")

    # --- Silhouette score ---
    logger.info("\n--- Silhouette score ---")
    results["silhouette"] = {}
    for name, emb in conditions.items():
        results["silhouette"][name] = compute_silhouette(emb, labels)
        logger.info(f"  {name}: {results['silhouette'][name]:.4f}")

    # --- CKA ---
    logger.info("\n--- CKA (Centered Kernel Alignment) ---")
    cka_results = {}
    cka_results["raw_trained"] = linear_CKA(esm3_raw, trained)
    cka_results["raw_random"] = linear_CKA(esm3_raw, random_emb)
    cka_results["trained_random"] = linear_CKA(trained, random_emb)
    if esm3_pooled_random is not None:
        if pca_results:
            # Flatten + PCA same as other conditions
            from sklearn.decomposition import PCA as _PCA2
            flat_rp = esm3_pooled_random.reshape(N, -1)
            n_comp = min(args.pca_dim, flat_rp.shape[0], flat_rp.shape[1])
            esm3_raw_random = _PCA2(n_components=n_comp, random_state=42).fit_transform(flat_rp)
        else:
            esm3_raw_random = esm3_pooled_random.mean(axis=1)
        cka_results["raw_trained_pool_vs_raw_random_pool"] = linear_CKA(
            esm3_raw, esm3_raw_random,
        )
    results["cka"] = cka_results
    for k, v in cka_results.items():
        logger.info(f"  {k}: {v:.4f}")

    # --- Embedding statistics ---
    logger.info("\n--- Embedding statistics ---")
    emb_stats = {"effective_rank": {}, "isotropy": {}}
    for name, emb in conditions.items():
        logger.info(f"  {name} ...")
        emb_stats["effective_rank"][name] = effective_rank(emb)
        emb_stats["isotropy"][name] = isotropy_score(emb)
        logger.info(f"    effective_rank: {emb_stats['effective_rank'][name]:.1f}")
        logger.info(f"    isotropy: {emb_stats['isotropy'][name]:.4f}")
    results["embedding_stats"] = emb_stats

    # --- Cosine distributions ---
    logger.info("\n--- Cosine similarity distributions ---")
    cos_dists = {}
    for name, emb in conditions.items():
        logger.info(f"  {name} ...")
        cos_dists[name] = cosine_distributions(emb, labels)
        logger.info(f"    intra-class mean: {cos_dists[name]['intra_class']['mean']:.4f}")
        logger.info(f"    inter-class mean: {cos_dists[name]['inter_class']['mean']:.4f}")
        logger.info(f"    separation: {cos_dists[name]['separation']:.4f}")
    results["cosine_distributions_summary"] = {
        name: {
            "intra_mean": d["intra_class"]["mean"],
            "inter_mean": d["inter_class"]["mean"],
            "separation": d["separation"],
        }
        for name, d in cos_dists.items()
    }

    # --- Retrieval metrics ---
    logger.info("\n--- Retrieval metrics ---")
    results["retrieval"] = {}
    for name, emb in conditions.items():
        logger.info(f"  {name} ...")
        t0 = time.time()
        results["retrieval"][name] = retrieval_metrics(
            emb, labels, ks=[1, 5, 10, 20],
        )
        logger.info(f"    Done in {time.time() - t0:.1f}s")
        for k, v in results["retrieval"][name].items():
            logger.info(f"    {k}: {v:.4f}")

    # -------------------------------------------------------------------
    # 4. UMAP
    # -------------------------------------------------------------------
    if not args.skip_umap:
        logger.info("\n--- UMAP ---")
        umap_indices = None  # Will be set if sub-sampling occurs
        for name, emb in conditions.items():
            logger.info(f"  Computing UMAP for {name} ...")
            t0 = time.time()
            coords, indices = compute_umap(
                emb,
                max_samples=args.max_umap_samples,
                n_neighbors=15,
                min_dist=0.1,
            )
            out_path = emb_dir / f"umap_{name}.npy"
            np.save(str(out_path), coords)
            logger.info(f"    Saved {out_path.name}: shape={coords.shape} ({time.time() - t0:.1f}s)")
            # Save sub-sampling indices (same for all conditions due to same seed)
            if indices is not None and umap_indices is None:
                umap_indices = indices
        # Save UMAP indices and corresponding labels
        if umap_indices is not None:
            np.save(str(emb_dir / "umap_indices.npy"), umap_indices)
            np.save(str(emb_dir / "umap_labels.npy"), labels[umap_indices])
            logger.info(f"    Saved umap_indices.npy and umap_labels.npy ({len(umap_indices)} samples)")
    else:
        logger.info("\n--- UMAP skipped ---")

    # -------------------------------------------------------------------
    # 5. Save all outputs
    # -------------------------------------------------------------------
    logger.info("\n--- Saving outputs ---")

    # Analysis results
    out_path = emb_dir / "analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  {out_path}")

    # Labels
    np.save(str(emb_dir / "labels.npy"), labels)
    logger.info(f"  labels.npy: shape=({N},)")

    # Label names
    with open(emb_dir / "label_names.json", "w") as f:
        json.dump(label_names, f, indent=2)
    logger.info(f"  label_names.json: {len(label_names)} classes")

    # Full cosine distributions
    with open(emb_dir / "cosine_distributions.json", "w") as f:
        json.dump(cos_dists, f, indent=2)
    logger.info("  cosine_distributions.json")

    # -------------------------------------------------------------------
    # 6. Summary
    # -------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Compact summary table
    header = f"{'Metric':<25} {'ESM3 Raw':>12} {'Trained':>12} {'Random':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    # kNN k=5
    knn_vals = {c: results["knn_accuracy"][c].get("k5", {}).get("mean", 0.0)
                for c in ["esm3_raw", "trained", "random"]}
    logger.info(f"{'kNN (k=5)':<25} {knn_vals['esm3_raw']:>12.4f} "
                f"{knn_vals['trained']:>12.4f} {knn_vals['random']:>12.4f}")

    # Linear probe
    lp_vals = {c: results["linear_probe"][c]["accuracy"]
               for c in ["esm3_raw", "trained", "random"]}
    logger.info(f"{'Linear Probe':<25} {lp_vals['esm3_raw']:>12.4f} "
                f"{lp_vals['trained']:>12.4f} {lp_vals['random']:>12.4f}")

    # Silhouette
    sil_vals = results["silhouette"]
    logger.info(f"{'Silhouette':<25} {sil_vals['esm3_raw']:>12.4f} "
                f"{sil_vals['trained']:>12.4f} {sil_vals['random']:>12.4f}")

    # Effective rank
    er_vals = emb_stats["effective_rank"]
    logger.info(f"{'Effective Rank':<25} {er_vals['esm3_raw']:>12.1f} "
                f"{er_vals['trained']:>12.1f} {er_vals['random']:>12.1f}")

    # Isotropy
    iso_vals = emb_stats["isotropy"]
    logger.info(f"{'Isotropy (lower=better)':<25} {iso_vals['esm3_raw']:>12.4f} "
                f"{iso_vals['trained']:>12.4f} {iso_vals['random']:>12.4f}")

    # Retrieval P@5
    ret_vals = {c: results["retrieval"][c].get("precision_at_5", 0.0)
                for c in ["esm3_raw", "trained", "random"]}
    logger.info(f"{'Precision@5':<25} {ret_vals['esm3_raw']:>12.4f} "
                f"{ret_vals['trained']:>12.4f} {ret_vals['random']:>12.4f}")

    # MRR
    mrr_vals = {c: results["retrieval"][c].get("mrr", 0.0)
                for c in ["esm3_raw", "trained", "random"]}
    logger.info(f"{'MRR':<25} {mrr_vals['esm3_raw']:>12.4f} "
                f"{mrr_vals['trained']:>12.4f} {mrr_vals['random']:>12.4f}")

    logger.info("")
    logger.info("CKA:")
    for k, v in results["cka"].items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info(f"\nAll outputs saved to: {emb_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
