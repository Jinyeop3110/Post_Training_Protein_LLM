#!/usr/bin/env python3
"""
Protein embedding quality analysis — visualization.

Generates 6 figures comparing embeddings from three conditions:
  1. Raw ESM-3 (frozen encoder, attention-pooled)
  2. Trained MLP projector
  3. Random MLP projector

Usage:
    python scripts/analysis/plot_embedding_analysis.py \
        --embeddings_dir blog/data/03-10/embeddings/

Expects the following files in embeddings_dir:
    analysis_results.json    — All quantitative metrics
    umap_esm3_raw.npy        — (N, 2) UMAP coordinates
    umap_trained.npy         — (N, 2)
    umap_random.npy          — (N, 2)
    labels.npy               — (N,) integer labels
    label_names.json         — {0: "label_name", ...}
    cosine_distributions.json — intra/inter-class cosine distributions
"""

import matplotlib

matplotlib.use("Agg")

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import shared style module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "analysis"))
from figure_style import (  # noqa: E402
    BASE_DIR,
    MAIN_FIGURES_DIR,
    SUPPLE_FIGURES_DIR,
    annotate_bars,
    get_label,  # noqa: F401 — available for axis labels
    save_figure,
    save_main_figure,
    style,
    to_rgba_alpha,
)

# ═══════════════════════════════════════════════════════════════════════
# CONDITION STYLING
# ═══════════════════════════════════════════════════════════════════════

CONDITION_KEYS = ["esm3_raw", "trained", "random"]

CONDITION_LABELS = {
    "esm3_raw": "Raw ESM-3 (pooled)",
    "trained": "Trained Projector",
    "random": "Random Projector",
}

JEKYLL_DIR = Path(
    "/home/yeopjin/orcd/pool/workspace/Jinyeop3110.github.io/assets/img/blog/protein-llm/"
)


def _condition_colors():
    """Get condition colors using the active style palette (adapts to blog/paper/web)."""
    return {
        "esm3_raw": style.color("text"),
        "trained": style.color("mlp"),
        "random": style.color("flamingo"),
    }


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data(embeddings_dir):
    """Load all embedding analysis data files.

    Returns:
        dict with keys: umap, labels, label_names, metrics, cosine_dists
    """
    edir = Path(embeddings_dir)
    data = {}

    # UMAP coordinates
    data["umap"] = {}
    for cond in CONDITION_KEYS:
        fname = {
            "esm3_raw": "umap_esm3_raw.npy",
            "trained": "umap_trained.npy",
            "random": "umap_random.npy",
        }[cond]
        path = edir / fname
        if path.exists():
            data["umap"][cond] = np.load(str(path))
        else:
            print(f"  WARNING: {path} not found, skipping UMAP for {cond}")

    # Labels
    labels_path = edir / "labels.npy"
    if labels_path.exists():
        data["labels"] = np.load(str(labels_path))
    else:
        data["labels"] = None
        print("  WARNING: labels.npy not found")

    # UMAP-specific labels (sub-sampled to match UMAP coordinates)
    umap_labels_path = edir / "umap_labels.npy"
    if umap_labels_path.exists():
        data["umap_labels"] = np.load(str(umap_labels_path))
    else:
        data["umap_labels"] = None

    # Label names
    label_names_path = edir / "label_names.json"
    if label_names_path.exists():
        with open(label_names_path) as f:
            raw = json.load(f)
        # Keys may be strings of ints; normalize
        data["label_names"] = {int(k): v for k, v in raw.items()}
    else:
        data["label_names"] = {}

    # Analysis results (metrics)
    metrics_path = edir / "analysis_results.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)
    else:
        data["metrics"] = {}
        print("  WARNING: analysis_results.json not found")

    # Cosine distributions
    cosine_path = edir / "cosine_distributions.json"
    if cosine_path.exists():
        with open(cosine_path) as f:
            data["cosine_dists"] = json.load(f)
    else:
        data["cosine_dists"] = {}

    return data


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: UMAP 3-PANEL (HERO FIGURE)
# ═══════════════════════════════════════════════════════════════════════

def plot_umap_3panel(data):
    """3-panel UMAP scatter: Raw ESM-3 | Trained MLP | Random MLP."""
    umap_data = data["umap"]
    label_names = data["label_names"]

    # Use UMAP-specific labels if available (handles sub-sampling)
    labels = data.get("umap_labels")
    if labels is None:
        labels = data["labels"]

    if not umap_data or labels is None:
        print("  SKIP: UMAP 3-panel (missing data)")
        return []

    # Verify label count matches UMAP coordinate count
    first_umap = next(iter(umap_data.values()))
    if len(labels) != len(first_umap):
        print(f"  WARNING: labels ({len(labels)}) != UMAP coords ({len(first_umap)}), using full labels")
        labels = data["labels"]
        if labels is None or len(labels) != len(first_umap):
            print("  SKIP: UMAP 3-panel (label/coord mismatch)")
            return []

    colors = _condition_colors()
    n_labels = len(label_names) if label_names else len(np.unique(labels))

    # Build a categorical colormap for biological labels
    if n_labels <= 10:
        cmap = matplotlib.colormaps.get_cmap("tab10").resampled(n_labels)
    elif n_labels <= 20:
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_labels)
    else:
        cmap = matplotlib.colormaps.get_cmap("viridis").resampled(n_labels)

    label_colors = [cmap(i) for i in range(n_labels)]

    # --- Blog version ---
    style.apply("blog")
    w, h = style.figsize("wide")
    fig, axes = plt.subplots(1, 3, figsize=(w + 2, h - 1))
    fig.subplots_adjust(wspace=0.08)

    panels = [
        ("esm3_raw", "Raw ESM-3 (pooled)"),
        ("trained", "Trained Projector"),
        ("random", "Random Projector"),
    ]

    scatter_handles = []
    for ax, (cond, title) in zip(axes, panels):
        if cond not in umap_data:
            ax.set_visible(False)
            continue

        coords = umap_data[cond]
        for lab_idx in range(n_labels):
            mask = labels == lab_idx
            if not np.any(mask):
                continue
            name = label_names.get(lab_idx, f"Class {lab_idx}")
            sc = ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[label_colors[lab_idx]],
                s=8, alpha=0.6, edgecolors="none",
                label=name if ax is axes[0] else None,
                rasterized=True,
            )
            if ax is axes[0]:
                scatter_handles.append(sc)

        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2" if ax is axes[0] else "")
        style.clean_axes(ax)

        # Subtle colored border matching the condition
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[cond])
            spine.set_linewidth(2.0)

    # Shared legend below
    if scatter_handles:
        legend_labels = [label_names.get(i, f"Class {i}") for i in range(n_labels)
                         if np.any(labels == i)]
        # Truncate long legends
        max_cols = min(n_labels, 6)
        fig.legend(
            handles=scatter_handles,
            labels=legend_labels,
            loc="lower center",
            ncol=max_cols,
            bbox_to_anchor=(0.5, -0.05),
            frameon=False,
            markerscale=2.5,
            fontsize=9,
        )

    fig.suptitle("Protein Embedding Space: Effect of Projection Training",
                 fontsize=14, fontweight="bold", y=1.02)

    saved = []
    # Save as main figure (fig12)
    saved.extend(save_main_figure(fig, "fig12_embedding_umap", close=False))
    # Also save as supplementary
    saved.extend(save_figure(fig, "embedding_umap_3panel"))

    # --- Paper version ---
    style.apply("paper")
    pw, _ = style.figsize("wide")
    fig_p, axes_p = plt.subplots(1, 3, figsize=(pw, 2.2))
    fig_p.subplots_adjust(wspace=0.08)

    for ax, (cond, title) in zip(axes_p, panels):
        if cond not in umap_data:
            ax.set_visible(False)
            continue
        coords = umap_data[cond]
        for lab_idx in range(n_labels):
            mask = labels == lab_idx
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[label_colors[lab_idx]],
                s=3, alpha=0.5, edgecolors="none",
                rasterized=True,
            )
        # Paper: sub-panel labels only (no full title)
        ax.text(0.5, 1.02, title, transform=ax.transAxes,
                ha="center", fontsize=7, fontstyle="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        style.clean_axes(ax)

    saved.extend(save_figure(fig_p, "embedding_umap_3panel", target="paper"))

    # Copy to Jekyll
    _copy_to_jekyll("fig12_embedding_umap.png", MAIN_FIGURES_DIR)

    # Restore blog style for subsequent plots
    style.apply("blog")
    return saved


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: METRICS COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════

def plot_metrics_bars(data):
    """Grouped bar chart: kNN@5, linear probe accuracy, silhouette score."""
    metrics = data.get("metrics", {})
    if not metrics:
        print("  SKIP: Metrics bar chart (no analysis_results.json)")
        return []

    colors = _condition_colors()

    # Extract metrics for each condition
    metric_names = ["knn_accuracy_k5", "linear_probe_accuracy", "silhouette_score"]
    metric_pretty = ["kNN@5 Accuracy", "Linear Probe Acc.", "Silhouette Score"]

    # Try to find the metrics under different possible key structures
    values = {cond: [] for cond in CONDITION_KEYS}
    for m in metric_names:
        for cond in CONDITION_KEYS:
            val = _get_metric(metrics, cond, m)
            values[cond].append(val)

    # Check we have data
    if all(v is None for cond in CONDITION_KEYS for v in values[cond]):
        print("  SKIP: Metrics bar chart (no matching metrics found)")
        return []

    style.apply("blog")
    fig, ax = style.subplots()

    x = np.arange(len(metric_names))
    bar_width = 0.25

    for i, cond in enumerate(CONDITION_KEYS):
        vals = [v if v is not None else 0 for v in values[cond]]
        bars = ax.bar(
            x + i * bar_width, vals, bar_width,
            label=CONDITION_LABELS[cond],
            color=colors[cond],
            edgecolor="white", linewidth=0.5,
        )
        annotate_bars(ax, bars, vals, fmt="{:.3f}")

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(metric_pretty)
    ax.set_ylabel("Score")
    ax.set_title("Embedding Quality Metrics", fontweight="bold")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    style.clean_axes(ax)

    saved = save_figure(fig, "embedding_metrics_comparison")
    _copy_to_jekyll("embedding_metrics_comparison.png", SUPPLE_FIGURES_DIR)
    return saved


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: CKA HEATMAP
# ═══════════════════════════════════════════════════════════════════════

def plot_cka_heatmap(data):
    """3x3 CKA similarity heatmap between conditions."""
    metrics = data.get("metrics", {})

    # Try to find CKA matrix
    cka_matrix = _get_nested(metrics, "cka_matrix")
    if cka_matrix is None:
        # Try to build from pairwise values
        cka_matrix = _build_cka_matrix(metrics)

    if cka_matrix is None:
        print("  SKIP: CKA heatmap (no CKA data)")
        return []

    cka_matrix = np.array(cka_matrix)
    labels_list = [CONDITION_LABELS[c] for c in CONDITION_KEYS]

    style.apply("blog")
    fig, ax = style.subplots()

    mask = None  # Show full matrix
    sns.heatmap(
        cka_matrix,
        annot=True, fmt=".3f",
        xticklabels=labels_list,
        yticklabels=labels_list,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        square=True,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "CKA Similarity", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Centered Kernel Alignment (CKA)", fontweight="bold", pad=12)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)

    saved = save_figure(fig, "embedding_cka_heatmap")

    # Paper version
    style.apply("paper")
    fig_p, ax_p = style.subplots("tall")
    sns.heatmap(
        cka_matrix,
        annot=True, fmt=".3f", annot_kws={"size": 7},
        xticklabels=[s.replace(" ", "\n") for s in labels_list],
        yticklabels=[s.replace(" ", "\n") for s in labels_list],
        cmap="YlOrRd",
        vmin=0, vmax=1,
        square=True,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": "CKA", "shrink": 0.7},
        ax=ax_p,
    )
    plt.xticks(rotation=25, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    saved.extend(save_figure(fig_p, "embedding_cka_heatmap", target="paper"))

    style.apply("blog")
    return saved


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: COSINE SIMILARITY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════

def plot_cosine_distributions(data):
    """Intra-class vs inter-class cosine similarity distributions per condition."""
    cosine_dists = data.get("cosine_dists", {})
    if not cosine_dists:
        print("  SKIP: Cosine distributions (no data)")
        return []

    colors = _condition_colors()

    style.apply("blog")
    w, h = style.figsize("wide")
    fig, axes = plt.subplots(1, 3, figsize=(w + 2, h - 1.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, cond in zip(axes, CONDITION_KEYS):
        cond_data = cosine_dists.get(cond, {})

        # Raw arrays: prefer *_values keys (from updated analyze_embeddings.py)
        intra_vals = cond_data.get("intra_class_values")
        inter_vals = cond_data.get("inter_class_values")

        # Fallback: check if intra_class/inter_class are arrays (list) or dicts (stats)
        if intra_vals is None:
            intra_raw = cond_data.get("intra_class")
            if isinstance(intra_raw, list):
                intra_vals = intra_raw
        if inter_vals is None:
            inter_raw = cond_data.get("inter_class")
            if isinstance(inter_raw, list):
                inter_vals = inter_raw

        has_histograms = (intra_vals is not None and len(intra_vals) > 0) or \
                         (inter_vals is not None and len(inter_vals) > 0)

        if not has_histograms:
            # No raw arrays — show stats text instead
            intra_stats = cond_data.get("intra_class", {})
            inter_stats = cond_data.get("inter_class", {})
            if not intra_stats and not inter_stats:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="#999")
            else:
                # Display stats as text
                lines = []
                if isinstance(intra_stats, dict):
                    lines.append(f"Intra-class: mean={intra_stats.get('mean', 0):.4f}")
                if isinstance(inter_stats, dict):
                    lines.append(f"Inter-class: mean={inter_stats.get('mean', 0):.4f}")
                sep = cond_data.get("separation")
                if sep is not None:
                    lines.append(f"Separation: {sep:.4f}")
                ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#333")
            ax.set_title(CONDITION_LABELS[cond], fontweight="bold")
            continue

        base_color = colors[cond]
        intra_color = base_color
        inter_color = to_rgba_alpha(base_color, 0.4)

        if intra_vals is not None and len(intra_vals) > 0:
            intra_arr = np.array(intra_vals)
            ax.hist(intra_arr, bins=60, alpha=0.7, color=intra_color,
                    label="Intra-class", density=True, edgecolor="none")
        if inter_vals is not None and len(inter_vals) > 0:
            inter_arr = np.array(inter_vals)
            ax.hist(inter_arr, bins=60, alpha=0.4, color=inter_color,
                    label="Inter-class", density=True, edgecolor="none",
                    hatch="//")

        ax.set_title(CONDITION_LABELS[cond], fontweight="bold")
        ax.set_xlabel("Cosine Similarity")
        if ax is axes[0]:
            ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style.clean_axes(ax)

        # Add separation metric if available
        sep = cond_data.get("separation")
        if sep is not None:
            ax.text(0.97, 0.95, f"Sep={sep:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=base_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=base_color,
                              alpha=0.8))

    fig.suptitle("Cosine Similarity: Intra- vs Inter-Class Separation",
                 fontsize=13, fontweight="bold", y=1.02)

    saved = save_figure(fig, "embedding_cosine_distributions")
    _copy_to_jekyll("embedding_cosine_distributions.png", SUPPLE_FIGURES_DIR)
    return saved


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: kNN ACCURACY vs k
# ═══════════════════════════════════════════════════════════════════════

def plot_knn_vs_k(data):
    """Line plot: kNN accuracy vs k for each condition."""
    metrics = data.get("metrics", {})

    colors = _condition_colors()
    k_values = [1, 5, 10, 20]

    # Try to extract kNN accuracy for various k
    has_data = False
    cond_curves = {}

    for cond in CONDITION_KEYS:
        means = []
        stds = []
        for k in k_values:
            mean_val = _get_metric(metrics, cond, f"knn_accuracy_k{k}")
            std_val = _get_metric(metrics, cond, f"knn_accuracy_k{k}_std")
            if mean_val is not None:
                has_data = True
            means.append(mean_val)
            stds.append(std_val if std_val is not None else 0.0)
        cond_curves[cond] = {"means": means, "stds": stds}

    if not has_data:
        print("  SKIP: kNN vs k (no kNN data for multiple k)")
        return []

    style.apply("blog")
    fig, ax = style.subplots()

    for cond in CONDITION_KEYS:
        curve = cond_curves[cond]
        means = np.array([v if v is not None else np.nan for v in curve["means"]])
        stds = np.array(curve["stds"])

        valid = ~np.isnan(means)
        if not np.any(valid):
            continue

        k_valid = np.array(k_values)[valid]
        m_valid = means[valid]
        s_valid = stds[valid]

        ax.plot(k_valid, m_valid, "o-", color=colors[cond],
                label=CONDITION_LABELS[cond], linewidth=2, markersize=7)
        if np.any(s_valid > 0):
            ax.fill_between(k_valid, m_valid - s_valid, m_valid + s_valid,
                            color=to_rgba_alpha(colors[cond], 0.2))

    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("kNN Accuracy")
    ax.set_title("kNN Classification Accuracy vs k", fontweight="bold")
    ax.set_xticks(k_values)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    style.clean_axes(ax)

    saved = save_figure(fig, "embedding_knn_vs_k")
    return saved


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: EFFECTIVE RANK & ISOTROPY
# ═══════════════════════════════════════════════════════════════════════

def plot_rank_isotropy(data):
    """Bar chart for effective rank and isotropy across conditions."""
    metrics = data.get("metrics", {})
    colors = _condition_colors()

    # Extract effective rank and isotropy
    metric_keys = ["effective_rank", "isotropy"]
    metric_pretty = ["Effective Rank", "Isotropy"]

    values = {cond: [] for cond in CONDITION_KEYS}
    has_data = False

    for m in metric_keys:
        for cond in CONDITION_KEYS:
            val = _get_metric(metrics, cond, m)
            values[cond].append(val)
            if val is not None:
                has_data = True

    if not has_data:
        print("  SKIP: Rank & isotropy (no data)")
        return []

    style.apply("blog")
    fig, axes = plt.subplots(1, 2, figsize=style.figsize())

    for idx, (m_key, m_label) in enumerate(zip(metric_keys, metric_pretty)):
        ax = axes[idx]
        bar_vals = []
        bar_colors = []
        bar_labels = []
        for cond in CONDITION_KEYS:
            val = values[cond][idx]
            bar_vals.append(val if val is not None else 0)
            bar_colors.append(colors[cond])
            bar_labels.append(CONDITION_LABELS[cond])

        bars = ax.bar(bar_labels, bar_vals, color=bar_colors,
                      edgecolor="white", linewidth=0.5, width=0.6)

        # Determine format based on value magnitude
        if m_key == "effective_rank":
            fmt = "{:.1f}"
        else:
            fmt = "{:.4f}"
        annotate_bars(ax, bars, bar_vals, fmt=fmt)

        ax.set_title(m_label, fontweight="bold")
        ax.set_ylabel(m_label)
        ax.set_ylim(bottom=0)
        style.clean_axes(ax)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Representation Geometry: Rank & Isotropy",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    saved = save_figure(fig, "embedding_rank_isotropy")
    return saved


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _get_metric(metrics, condition, metric_name):
    """Flexibly extract a metric value from analysis_results.json.

    Handles the nested structure produced by analyze_embeddings.py:
      - knn_accuracy_k5      -> metrics["knn_accuracy"][condition]["k5"]["mean"]
      - linear_probe_accuracy -> metrics["linear_probe"][condition]["accuracy"]
      - silhouette_score     -> metrics["silhouette"][condition]
      - effective_rank       -> metrics["embedding_stats"]["effective_rank"][condition]
      - isotropy             -> metrics["embedding_stats"]["isotropy"][condition]
      - knn_accuracy_k{N}_std -> metrics["knn_accuracy"][condition]["k{N}"]["std"]
    """
    import re as _re

    # knn_accuracy_k{N} or knn_accuracy_k{N}_std
    m = _re.match(r"knn_accuracy_k(\d+)(_std)?$", metric_name)
    if m:
        k = m.group(1)
        is_std = m.group(2) is not None
        knn = metrics.get("knn_accuracy", {}).get(condition, {}).get(f"k{k}", {})
        val = knn.get("std" if is_std else "mean")
        return float(val) if val is not None else None

    # linear_probe_accuracy
    if metric_name == "linear_probe_accuracy":
        lp = metrics.get("linear_probe", {}).get(condition, {})
        val = lp.get("accuracy")
        return float(val) if val is not None else None

    # silhouette_score
    if metric_name == "silhouette_score":
        sil = metrics.get("silhouette", {})
        val = sil.get(condition)
        return float(val) if val is not None else None

    # effective_rank, isotropy (under embedding_stats)
    if metric_name in ("effective_rank", "isotropy"):
        stats = metrics.get("embedding_stats", {}).get(metric_name, {})
        val = stats.get(condition)
        return float(val) if val is not None else None

    # retrieval metrics: precision_at_{k}, mrr
    if metric_name.startswith("precision_at_") or metric_name == "mrr":
        ret = metrics.get("retrieval", {}).get(condition, {})
        val = ret.get(metric_name)
        return float(val) if val is not None else None

    # Generic fallback: try metrics[metric_name][condition]
    if metric_name in metrics and isinstance(metrics[metric_name], dict):
        val = metrics[metric_name].get(condition)
        if val is not None:
            return float(val) if not isinstance(val, dict) else None

    # Flat key fallback
    flat_key = f"{metric_name}_{condition}"
    if flat_key in metrics:
        return float(metrics[flat_key])

    return None


def _get_nested(d, key):
    """Get a key from dict, searching one level deep."""
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return None


def _build_cka_matrix(metrics):
    """Try to build a 3x3 CKA matrix from pairwise values in metrics.

    analyze_embeddings.py stores CKA as:
      metrics["cka"]["raw_trained"], metrics["cka"]["raw_random"], metrics["cka"]["trained_random"]
    where "raw" = esm3_raw in our condition naming.
    """
    cka_data = metrics.get("cka", {})
    if not cka_data:
        return None

    # Map condition keys to CKA key names
    # analyze uses: "raw" for esm3_raw, "trained" for trained, "random" for random
    cka_name_map = {"esm3_raw": "raw", "trained": "trained", "random": "random"}

    matrix = np.zeros((3, 3))
    found_any = False

    for ci, cond_i in enumerate(CONDITION_KEYS):
        for cj, cond_j in enumerate(CONDITION_KEYS):
            if ci == cj:
                matrix[ci, cj] = 1.0
                continue

            name_i = cka_name_map[cond_i]
            name_j = cka_name_map[cond_j]

            # Try both orderings
            for key in [f"{name_i}_{name_j}", f"{name_j}_{name_i}"]:
                val = cka_data.get(key)
                if val is not None:
                    matrix[ci, cj] = float(val)
                    found_any = True
                    break

    return matrix if found_any else None


def _copy_to_jekyll(filename, source_dir):
    """Copy a figure to the Jekyll site assets directory."""
    src = source_dir / filename
    if not src.exists():
        return
    JEKYLL_DIR.mkdir(parents=True, exist_ok=True)
    dst = JEKYLL_DIR / filename
    shutil.copy2(str(src), str(dst))
    print(f"  COPY {dst}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE CATALOG UPDATE
# ═══════════════════════════════════════════════════════════════════════

def update_figure_catalog():
    """Append embedding analysis entries to figure_catalog.md."""
    catalog_path = BASE_DIR / "blog" / "figures" / "figure_catalog.md"
    if not catalog_path.exists():
        print("  WARNING: figure_catalog.md not found, skipping update")
        return

    content = catalog_path.read_text()

    # Check if already present
    if "embedding_umap_3panel" in content:
        print("  Figure catalog already has embedding entries, skipping update")
        return

    new_section = """
### Embedding Quality Analysis (Mar 2026)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `embedding_umap_3panel.png` | UMAP 3-panel: Raw ESM-3, Trained MLP, Random MLP | 2026-03-10 | Embedding quality comparison |
| `embedding_metrics_comparison.png` | kNN@5, linear probe, silhouette bar chart | 2026-03-10 | Quantitative metrics |
| `embedding_cka_heatmap.png` | 3x3 CKA similarity heatmap | 2026-03-10 | Representation similarity |
| `embedding_cosine_distributions.png` | Intra- vs inter-class cosine similarity | 2026-03-10 | Class separation quality |
| `embedding_knn_vs_k.png` | kNN accuracy vs k (1, 5, 10, 20) | 2026-03-10 | Robustness across k |
| `embedding_rank_isotropy.png` | Effective rank and isotropy bars | 2026-03-10 | Representation geometry |
"""

    # Insert before the "## Generation Scripts" section if it exists, otherwise append
    marker = "## Generation Scripts"
    if marker in content:
        content = content.replace(marker, new_section + "\n" + marker)
    else:
        content += "\n" + new_section

    # Also add main figure entry
    main_entry = "| 12 | `fig12_embedding_umap.png` | embedding | UMAP 3-panel: Raw ESM-3 vs Trained vs Random projector | Shows projection training effect | `scripts/analysis/plot_embedding_analysis.py` | 2026-03-10 | confirmed |"

    # Find the main figure table and append
    if "fig12_embedding_umap" not in content:
        # Insert after fig11 row
        fig11_marker = "fig11_generation_over_steps"
        if fig11_marker in content:
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if fig11_marker in line and line.strip().startswith("|"):
                    new_lines.append(main_entry)
            content = "\n".join(new_lines)

    catalog_path.write_text(content)
    print(f"  UPDATED {catalog_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot protein embedding quality analysis figures."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="blog/data/03-10/embeddings/",
        help="Directory containing embedding analysis data files.",
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        type=int,
        default=None,
        help="Which figures to generate (1-6). Default: all.",
    )
    args = parser.parse_args()

    # Resolve path relative to project root if not absolute
    edir = Path(args.embeddings_dir)
    if not edir.is_absolute():
        edir = BASE_DIR / edir

    print(f"Loading data from: {edir}")
    data = load_data(edir)

    # Determine which figures to generate
    all_figs = [1, 2, 3, 4, 5, 6]
    figs_to_plot = args.figures if args.figures else all_figs

    all_saved = []
    fig_funcs = {
        1: ("UMAP 3-panel", plot_umap_3panel),
        2: ("Metrics comparison bars", plot_metrics_bars),
        3: ("CKA heatmap", plot_cka_heatmap),
        4: ("Cosine distributions", plot_cosine_distributions),
        5: ("kNN accuracy vs k", plot_knn_vs_k),
        6: ("Effective rank & isotropy", plot_rank_isotropy),
    }

    for fig_num in figs_to_plot:
        if fig_num not in fig_funcs:
            print(f"  WARNING: Unknown figure number {fig_num}")
            continue
        name, func = fig_funcs[fig_num]
        print(f"\n--- Figure {fig_num}: {name} ---")
        saved = func(data)
        if saved:
            all_saved.extend(saved)

    # Update figure catalog
    print("\n--- Updating figure catalog ---")
    update_figure_catalog()

    # Summary
    print(f"\n{'='*60}")
    print(f"Generated {len(all_saved)} files:")
    for f in all_saved:
        print(f"  {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
