#!/usr/bin/env python3
"""ARCHIVED: Consolidated into scripts/analysis/plot_pub_figures.py (2026-03-09).
Use plot_pub_figures.py for equivalent output.

Original:
Publication-quality figures for three targets:
  1. Internal blog (blog/figures/)
  2. Personal website blog (blog/figures/web_*)
  3. NeurIPS paper (paper/figures/)

Uses data from blog/data/03-02/ (early experiments), blog/data/03-06/, and blog/data/03-07/ (latest).
"""

import matplotlib

matplotlib.use('Agg')
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import ArrowStyle, FancyBboxPatch

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
BASE = "/orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM"
BLOG_FIG = f"{BASE}/blog/figures"
PAPER_FIG = f"{BASE}/paper/figures"
DATA_02 = f"{BASE}/blog/data/03-02"
DATA_06 = f"{BASE}/blog/data/03-06"
DATA_07 = f"{BASE}/blog/data/03-07"
WEBSITE_DIR = "/home/yeopjin/orcd/pool/workspace/Jinyeop3110.github.io/assets/img/projects"

os.makedirs(BLOG_FIG, exist_ok=True)
os.makedirs(PAPER_FIG, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
df_early = pd.read_csv(f"{DATA_02}/run_histories.csv")
# Use latest 03-07 data (supersedes 03-06)
df_late = pd.read_csv(f"{DATA_07}/run_histories.csv")

with open(f"{DATA_02}/experiment_metadata.json") as f:
    meta_early = json.load(f)
with open(f"{DATA_07}/experiment_metadata.json") as f:
    meta_late = json.load(f)
with open(f"{DATA_06}/wandb_summaries.json") as f:
    wandb_data = json.load(f)

# ═══════════════════════════════════════════════════════════════════════
# COLOR SCHEMES
# ═══════════════════════════════════════════════════════════════════════
APPROACH_COLORS = {
    "mlp": "#1f77b4",
    "perceiver": "#ff7f0e",
    "text": "#2ca02c",
}

# Softer palette for website
WEB_COLORS = {
    "mlp": "#4A90D9",
    "perceiver": "#F5A623",
    "text": "#7ED321",
}

# Grayscale-safe palette for paper
PAPER_COLORS = {
    "mlp": "#2166ac",
    "perceiver": "#b2182b",
    "text": "#4d4d4d",
}

# ═══════════════════════════════════════════════════════════════════════
# STYLE CONTEXTS
# ═══════════════════════════════════════════════════════════════════════

def blog_style():
    """Internal blog style — existing convention."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
    })
    return APPROACH_COLORS

def web_style():
    """Personal website — clean, friendly, modern."""
    sns.set_theme(style="white", palette="muted")
    plt.rcParams.update({
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.labelsize': 13,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })
    return WEB_COLORS

def paper_style():
    """NeurIPS paper — LaTeX-compatible, PDF-exportable."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.4,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })
    return PAPER_COLORS

# NeurIPS column widths in inches
PAPER_COL = (3.25, 2.4)       # single column
PAPER_WIDE = (6.75, 2.8)      # full width
PAPER_TALL = (3.25, 3.2)      # single column tall

# Blog/web sizes
BLOG_SIZE = (10, 6)
WEB_SIZE = (9, 5.5)

# ═══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

# Key experiments for cross-approach comparison (from 03-02 data)
EARLY_EXPS = {
    "sft_lora_esm3_qwen3_8b_it_0225_203237": ("Perceiver", "perceiver"),
    "sft_lora_esm3_qwen3_8b_it_0226_151416": ("MLP (50K)", "mlp"),
    "sft_text_qwen3_8b_it_0227_145821": ("Text-only", "text"),
}

# Main run (from 03-07 data — updated to step 9750)
MAIN_RUN = "sft_esm3_mlp_combined_qwen3_8b_it_0306_010408"
TEXT_RUN = "sft_text_combined_qwen3_8b_it_0307_190324"

# Final metrics for comparison (updated with latest data)
FINAL_METRICS = {
    "Perceiver\n(50K, 3 ep)": {"eval_loss": 1.952, "train_loss": 2.296, "approach": "perceiver", "steps": 1677},
    "MLP\n(50K, 3 ep)": {"eval_loss": 1.942, "train_loss": 2.288, "approach": "mlp", "steps": 1677},
    "Text-only\n(50K, 3 ep)": {"eval_loss": 2.415, "train_loss": 2.551, "approach": "text", "steps": 2595},
    "MLP\n(4.89M, 0.34 ep)": {"eval_loss": 0.361, "train_loss": 0.438, "approach": "mlp", "steps": 9750},
}

# Generation quality from wandb (main run at step 7000)
GEN_METRICS = {
    "Overall": {"bleu": 0.307, "rouge_l": 0.483},
    "Function": {"bleu": 0.404, "rouge_l": 0.523},
    "Catalytic": {"bleu": 0.392, "rouge_l": 0.541},
    "General": {"bleu": 0.265, "rouge_l": 0.436},
    "Domain": {"bleu": 0.166, "rouge_l": 0.433},
}

# Dataset composition (from CLAUDE.md context)
DATASET_COMP = {
    "Protein Function": 2_147_000,
    "Catalytic Activity": 1_236_000,
    "General QA": 987_000,
    "Domain/Family": 456_000,
    "Other": 64_000,
}

print("Data loaded successfully.")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: Cross-Approach Loss Curves (Early experiments)
# ═══════════════════════════════════════════════════════════════════════

def plot_cross_approach_loss(figsize, colors, save_path, fmt="png", title_prefix=""):
    fig, ax = plt.subplots(figsize=figsize)
    for exp_name, (label, approach) in EARLY_EXPS.items():
        exp_data = df_early[df_early["experiment"] == exp_name].dropna(subset=["token_avg_loss"]).sort_values("step")
        if len(exp_data) > 0:
            ax.plot(exp_data["step"], exp_data["token_avg_loss"],
                    color=colors[approach], linewidth=1.5, alpha=0.85, label=label)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Token Average Loss")
    if title_prefix:
        ax.set_title(f"{title_prefix}Training Loss by Approach (50K Samples)")
    else:
        ax.set_title("Training Loss by Approach (50K Samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: Cross-Approach Eval Loss Curves
# ═══════════════════════════════════════════════════════════════════════

def plot_cross_approach_eval(figsize, colors, save_path, fmt="png"):
    fig, ax = plt.subplots(figsize=figsize)
    for exp_name, (label, approach) in EARLY_EXPS.items():
        exp_data = df_early[df_early["experiment"] == exp_name].dropna(subset=["eval_loss"]).sort_values("step")
        if len(exp_data) > 0:
            ax.plot(exp_data["step"], exp_data["eval_loss"],
                    color=colors[approach], marker="o", markersize=4,
                    linewidth=1.8, alpha=0.85, label=label)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval Loss")
    ax.set_title("Validation Loss by Approach")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Main Run Deep Dive (train + eval + smoothed)
# ═══════════════════════════════════════════════════════════════════════

def plot_main_run(figsize, colors, save_path, fmt="png"):
    df_main = df_late[df_late["experiment"] == MAIN_RUN].copy()
    train = df_main.dropna(subset=["token_avg_loss"]).sort_values("step")
    eval_d = df_main.dropna(subset=["eval_loss"]).sort_values("step")

    fig, ax = plt.subplots(figsize=figsize)
    # Raw (faint)
    ax.plot(train["step"], train["token_avg_loss"],
            color=colors["mlp"], linewidth=0.4, alpha=0.25, label="_raw")
    # Smoothed
    window = 50
    if len(train) > window:
        smoothed = train["token_avg_loss"].rolling(window, min_periods=1).mean()
        ax.plot(train["step"], smoothed,
                color=colors["mlp"], linewidth=1.8, alpha=0.9,
                label=f"Train (MA-{window})")
    # Eval
    if len(eval_d) > 0:
        ax.plot(eval_d["step"], eval_d["eval_loss"],
                color="#d62728", marker="D", markersize=5, linewidth=0,
                zorder=5, label="Eval Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("ESM3+MLP on 4.89M Samples: Training Progress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: Final Metrics Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════

def plot_final_comparison(figsize, colors, save_path, fmt="png"):
    names = list(FINAL_METRICS.keys())
    eval_vals = [FINAL_METRICS[n]["eval_loss"] for n in names]
    train_vals = [FINAL_METRICS[n]["train_loss"] for n in names]
    bar_colors = [colors[FINAL_METRICS[n]["approach"]] for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_vals, width, label="Train Loss",
                   color=[c + "99" for c in bar_colors], edgecolor=bar_colors, linewidth=1.2)
    bars2 = ax.bar(x + width/2, eval_vals, width, label="Eval Loss",
                   color=bar_colors, edgecolor="white", linewidth=0.5)

    # Annotate
    for bar, val in zip(bars2, eval_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=plt.rcParams['font.size'] - 1, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=plt.rcParams['font.size'] - 1)
    ax.set_ylabel("Loss")
    ax.set_title("Final Metrics: Approach Comparison")
    ax.legend()
    ax.set_ylim(0, max(max(train_vals), max(eval_vals)) * 1.25)
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: Generation Quality (BLEU / ROUGE-L)
# ═══════════════════════════════════════════════════════════════════════

def plot_generation_quality(figsize, colors, save_path, fmt="png"):
    categories = list(GEN_METRICS.keys())
    bleu = [GEN_METRICS[c]["bleu"] for c in categories]
    rouge = [GEN_METRICS[c]["rouge_l"] for c in categories]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, bleu, width, label="BLEU",
                   color=colors["mlp"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, rouge, width, label="ROUGE-L",
                   color=colors["perceiver"], alpha=0.85, edgecolor="white")

    for bar, val in zip(bars1, bleu):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=plt.rcParams['font.size'] - 2)
    for bar, val in zip(bars2, rouge):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=plt.rcParams['font.size'] - 2)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_title("Generation Quality: BLEU & ROUGE-L by Task (Step 7000)")
    ax.legend()
    ax.set_ylim(0, 0.7)
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 6: Dataset Composition
# ═══════════════════════════════════════════════════════════════════════

def plot_data_composition(figsize, colors, save_path, fmt="png"):
    labels = list(DATASET_COMP.keys())
    sizes = list(DATASET_COMP.values())
    total = sum(sizes)
    pcts = [s / total * 100 for s in sizes]

    # Use a nice color gradient
    cmap = plt.cm.Blues
    bar_colors = [cmap(0.3 + 0.6 * i / len(labels)) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, [s / 1e6 for s in sizes], color=bar_colors, edgecolor="white", linewidth=0.8)

    for bar, pct, size in zip(bars, pcts, sizes):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{size/1e6:.2f}M ({pct:.0f}%)",
                ha="left", va="center", fontsize=plt.rcParams['font.size'] - 1)

    ax.set_xlabel("Samples (millions)")
    ax.set_title(f"Training Data Composition ({total/1e6:.2f}M total)")
    ax.set_xlim(0, max(sizes) / 1e6 * 1.35)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 7: Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════

def plot_architecture(figsize, colors, save_path, fmt="png"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", linewidth=1.5)

    # ESM-3 encoder (frozen)
    ax.add_patch(FancyBboxPatch((0.3, 3.8), 2.0, 1.2, **box_style,
                                facecolor="#E8E8E8", edgecolor="#666666"))
    ax.text(1.3, 4.4, "ESM-3\n(frozen)", ha="center", va="center",
            fontsize=plt.rcParams['font.size'], fontweight="bold", color="#444444")

    # Attention Pooling
    ax.add_patch(FancyBboxPatch((3.2, 4.0), 1.6, 0.8, **box_style,
                                facecolor="#D4E6F1", edgecolor=colors["mlp"]))
    ax.text(4.0, 4.4, "Attn Pool\n(32 tok)", ha="center", va="center",
            fontsize=plt.rcParams['font.size'] - 1, color=colors["mlp"])

    # MLP Projector
    ax.add_patch(FancyBboxPatch((5.6, 4.0), 1.6, 0.8, **box_style,
                                facecolor="#D4E6F1", edgecolor=colors["mlp"]))
    ax.text(6.4, 4.4, "MLP\n1536→2560", ha="center", va="center",
            fontsize=plt.rcParams['font.size'] - 1, color=colors["mlp"])

    # LLM
    ax.add_patch(FancyBboxPatch((7.8, 2.5), 1.8, 2.5, **box_style,
                                facecolor="#FDE8D0", edgecolor="#E67E22"))
    ax.text(8.7, 3.75, "Qwen3-8B\n(LoRA)", ha="center", va="center",
            fontsize=plt.rcParams['font.size'], fontweight="bold", color="#D35400")

    # Protein input
    ax.add_patch(FancyBboxPatch((0.3, 1.5), 2.0, 1.0, **box_style,
                                facecolor="#E8F8E8", edgecolor="#27AE60"))
    ax.text(1.3, 2.0, "Protein\nSequence", ha="center", va="center",
            fontsize=plt.rcParams['font.size'], color="#27AE60")

    # Text prompt
    ax.add_patch(FancyBboxPatch((4.0, 1.5), 2.4, 1.0, **box_style,
                                facecolor="#FFF3E0", edgecolor="#F39C12"))
    ax.text(5.2, 2.0, "Text Prompt\n(chat template)", ha="center", va="center",
            fontsize=plt.rcParams['font.size'] - 1, color="#E67E22")

    # Output
    ax.add_patch(FancyBboxPatch((7.8, 0.5), 1.8, 1.0, **box_style,
                                facecolor="#FDEDEC", edgecolor="#E74C3C"))
    ax.text(8.7, 1.0, "Response", ha="center", va="center",
            fontsize=plt.rcParams['font.size'], fontweight="bold", color="#C0392B")

    # Arrows
    arrow_kw = dict(arrowstyle=ArrowStyle("->", head_length=0.3, head_width=0.15),
                    color="#555555", linewidth=1.2)
    ax.annotate("", xy=(3.2, 4.4), xytext=(2.3, 4.4), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.6, 4.4), xytext=(4.8, 4.4), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.8, 4.0), xytext=(7.2, 4.4), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.8, 3.8), xytext=(0.8, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.8, 3.0), xytext=(6.4, 2.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.7, 2.5), xytext=(8.7, 1.5), arrowprops=arrow_kw)

    # Title
    ax.text(5.0, 5.7, "Protein-LLM Architecture (ESM3 + MLP Approach)",
            ha="center", va="center",
            fontsize=plt.rcParams['axes.titlesize'], fontweight="bold")

    # Param counts
    ax.text(4.0, 3.4, "Trainable: ~32.5M params",
            ha="center", va="center",
            fontsize=plt.rcParams['font.size'] - 2, style="italic", color="#777777")
    ax.text(4.0, 3.0, "(Pooling 9.5M + MLP 21M + LoRA 2M)",
            ha="center", va="center",
            fontsize=plt.rcParams['font.size'] - 2, color="#999999")

    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 8: Gradient Norms (main run)
# ═══════════════════════════════════════════════════════════════════════

def plot_gradient_norms(figsize, colors, save_path, fmt="png"):
    df_main = df_late[df_late["experiment"] == MAIN_RUN].copy()
    gn = df_main.dropna(subset=["grad_norm"]).sort_values("step")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(gn["step"], gn["grad_norm"],
            color=colors["mlp"], linewidth=0.6, alpha=0.5)
    # Smoothed
    window = 30
    if len(gn) > window:
        smoothed = gn["grad_norm"].rolling(window, min_periods=1).mean()
        ax.plot(gn["step"], smoothed,
                color=colors["mlp"], linewidth=1.5, alpha=0.9,
                label=f"MA-{window}")
    mean_gn = gn["grad_norm"].mean()
    ax.axhline(mean_gn, color="gray", linestyle="--", alpha=0.5,
               label=f"Mean: {mean_gn:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm (log)")
    ax.set_title("Gradient Norms: Main Run (ESM3+MLP, 4.89M)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 9: Convergence Summary Table
# ═══════════════════════════════════════════════════════════════════════

def plot_convergence_table(figsize, colors, save_path, fmt="png"):
    headers = ["Experiment", "Approach", "Dataset", "Steps", "Train Loss", "Eval Loss"]
    rows = [
        ["Perceiver (0225)", "Perceiver", "50K", "1,677", "2.296", "1.952"],
        ["MLP (0226)", "MLP", "50K", "1,677", "2.288", "1.942"],
        ["Text-only (0227)", "Text", "50K", "2,595", "2.551", "2.415"],
        ["MLP-Combined (0306)", "MLP", "4.89M", "9,750*", "0.438", "0.361"],
        ["Text-Combined (0307)", "Text", "4.89M", "200*", "2.356", "—"],
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(plt.rcParams['font.size'] - 1)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by approach
    approach_row_colors = {
        "perceiver": "#FDE8D0",
        "mlp": "#D4E6F1",
        "text": "#D5F5E3",
    }
    approach_order = ["perceiver", "mlp", "text", "mlp", "text"]
    for i in range(len(rows)):
        bg = approach_row_colors.get(approach_order[i], "#FFFFFF")
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(bg)

    # Highlight best eval loss
    table[4, 5].set_text_props(fontweight="bold", color="#C0392B")

    ax.set_title("Experiment Summary (* = still running)",
                 fontsize=plt.rcParams['axes.titlesize'], pad=15)
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 10: Scaling comparison (50K→4.89M effect)
# ═══════════════════════════════════════════════════════════════════════

def plot_scaling_effect(figsize, colors, save_path, fmt="png"):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: eval loss bars
    ax = axes[0]
    names = ["50K\n(3 ep)", "4.89M\n(0.34 ep)"]
    eval_vals = [1.942, 0.361]
    bar_colors = [colors["mlp"], colors["mlp"]]
    alphas = [0.6, 1.0]
    bars = ax.bar(names, eval_vals, color=bar_colors, alpha=1.0,
                  edgecolor="white", linewidth=0.8)
    bars[0].set_alpha(0.6)
    for bar, val in zip(bars, eval_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=plt.rcParams['font.size'], fontweight="bold")
    ax.set_ylabel("Eval Loss")
    ax.set_title("MLP Eval Loss")
    ax.set_ylim(0, 2.5)

    # Right: improvement arrow / percentage
    ax = axes[1]
    improvement = (1.942 - 0.361) / 1.942 * 100
    ax.text(0.5, 0.6, f"{improvement:.0f}%", ha="center", va="center",
            fontsize=plt.rcParams['font.size'] * 4, fontweight="bold",
            color=colors["mlp"],
            transform=ax.transAxes)
    ax.text(0.5, 0.35, "eval loss\nimprovement", ha="center", va="center",
            fontsize=plt.rcParams['font.size'] + 1, color="#666666",
            transform=ax.transAxes)
    ax.text(0.5, 0.15, "50K → 4.89M samples\n(98× more data)",
            ha="center", va="center",
            fontsize=plt.rcParams['font.size'] - 1, color="#999999",
            transform=ax.transAxes)
    ax.axis("off")

    fig.suptitle("Effect of Scaling Training Data (MLP Approach)",
                 fontsize=plt.rcParams['axes.titlesize'], fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 11: MLP vs Text on 4.89M (early comparison)
# ═══════════════════════════════════════════════════════════════════════

def plot_mlp_vs_text_4m(figsize, colors, save_path, fmt="png"):
    """Compare MLP (9750 steps) vs Text (200 steps) on 4.89M combined dataset."""
    fig, ax = plt.subplots(figsize=figsize)

    # MLP main run
    df_mlp = df_late[df_late["experiment"] == MAIN_RUN].dropna(subset=["token_avg_loss"]).sort_values("step")
    if len(df_mlp) > 0:
        # Smoothed
        window = 50
        if len(df_mlp) > window:
            smoothed = df_mlp["token_avg_loss"].rolling(window, min_periods=1).mean()
            ax.plot(df_mlp["step"], smoothed,
                    color=colors["mlp"], linewidth=1.8, alpha=0.9,
                    label="ESM3+MLP (encoder)")
        else:
            ax.plot(df_mlp["step"], df_mlp["token_avg_loss"],
                    color=colors["mlp"], linewidth=1.5, alpha=0.85,
                    label="ESM3+MLP (encoder)")

    # Text baseline
    df_txt = df_late[df_late["experiment"] == TEXT_RUN].dropna(subset=["token_avg_loss"]).sort_values("step")
    if len(df_txt) > 0:
        ax.plot(df_txt["step"], df_txt["token_avg_loss"],
                color=colors["text"], linewidth=1.8, alpha=0.85,
                marker="o", markersize=3, label="Text-only (no encoder)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Token Average Loss")
    ax.set_title("ESM3+MLP vs Text-only on 4.89M Combined Dataset")
    ax.legend()
    # Zoom to shared x range for fair comparison
    if len(df_txt) > 0:
        ax.set_xlim(0, max(df_txt["step"].max(), 250) * 1.1)
    fig.tight_layout()
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()
    print(f"  OK {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════════════
# GENERATE ALL FIGURES
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("GENERATING BLOG FIGURES")
print("=" * 60)
colors = blog_style()
plot_cross_approach_loss(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_approach_loss_curves.png")
plot_cross_approach_eval(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_approach_eval_curves.png")
plot_main_run(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_main_run_progress.png")
plot_final_comparison(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_final_comparison.png")
plot_generation_quality(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_generation_quality.png")
plot_data_composition(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_data_composition.png")
plot_architecture(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_architecture.png")
plot_gradient_norms(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_gradient_norms.png")
plot_convergence_table((14, 4), colors, f"{BLOG_FIG}/pub_convergence_table.png")
plot_scaling_effect(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_scaling_effect.png")
plot_mlp_vs_text_4m(BLOG_SIZE, colors, f"{BLOG_FIG}/pub_mlp_vs_text_4m.png")

print("\n" + "=" * 60)
print("GENERATING WEBSITE FIGURES")
print("=" * 60)
colors = web_style()
plot_cross_approach_loss(WEB_SIZE, colors, f"{BLOG_FIG}/web_approach_loss_curves.png")
plot_cross_approach_eval(WEB_SIZE, colors, f"{BLOG_FIG}/web_approach_eval_curves.png")
plot_main_run(WEB_SIZE, colors, f"{BLOG_FIG}/web_main_run_progress.png")
plot_final_comparison(WEB_SIZE, colors, f"{BLOG_FIG}/web_final_comparison.png")
plot_generation_quality(WEB_SIZE, colors, f"{BLOG_FIG}/web_generation_quality.png")
plot_data_composition(WEB_SIZE, colors, f"{BLOG_FIG}/web_data_composition.png")
plot_architecture(WEB_SIZE, colors, f"{BLOG_FIG}/web_architecture.png")
plot_scaling_effect(WEB_SIZE, colors, f"{BLOG_FIG}/web_scaling_effect.png")
plot_mlp_vs_text_4m(WEB_SIZE, colors, f"{BLOG_FIG}/web_mlp_vs_text_4m.png")

print("\n" + "=" * 60)
print("GENERATING PAPER FIGURES (PDF + PNG)")
print("=" * 60)
colors = paper_style()
# Paper figures as both PDF (vector) and PNG (raster fallback)
for ext in ["pdf", "png"]:
    plot_cross_approach_loss(PAPER_COL, colors, f"{PAPER_FIG}/loss_curves.{ext}", fmt=ext)
    plot_cross_approach_eval(PAPER_COL, colors, f"{PAPER_FIG}/eval_curves.{ext}", fmt=ext)
    plot_main_run(PAPER_COL, colors, f"{PAPER_FIG}/main_run.{ext}", fmt=ext)
    plot_final_comparison(PAPER_WIDE, colors, f"{PAPER_FIG}/final_comparison.{ext}", fmt=ext)
    plot_generation_quality(PAPER_WIDE, colors, f"{PAPER_FIG}/generation_quality.{ext}", fmt=ext)
    plot_data_composition(PAPER_COL, colors, f"{PAPER_FIG}/data_composition.{ext}", fmt=ext)
    plot_architecture((6.75, 3.5), colors, f"{PAPER_FIG}/architecture.{ext}", fmt=ext)
    plot_gradient_norms(PAPER_COL, colors, f"{PAPER_FIG}/gradient_norms.{ext}", fmt=ext)
    plot_convergence_table(PAPER_WIDE, colors, f"{PAPER_FIG}/convergence_table.{ext}", fmt=ext)
    plot_scaling_effect(PAPER_WIDE, colors, f"{PAPER_FIG}/scaling_effect.{ext}", fmt=ext)
    plot_mlp_vs_text_4m(PAPER_COL, colors, f"{PAPER_FIG}/mlp_vs_text_4m.{ext}", fmt=ext)

# ═══════════════════════════════════════════════════════════════════════
# COPY WEBSITE FIGURES TO PERSONAL SITE
# ═══════════════════════════════════════════════════════════════════════
import shutil

print("\n" + "=" * 60)
print("COPYING TO WEBSITE ASSETS")
print("=" * 60)
if os.path.isdir(WEBSITE_DIR):
    web_files = {
        "web_approach_loss_curves.png": "protein_llm_approach_loss_curves.png",
        "web_approach_eval_curves.png": "protein_llm_approach_eval_curves.png",
        "web_main_run_progress.png": "protein_llm_main_run_progress.png",
        "web_final_comparison.png": "protein_llm_final_comparison.png",
        "web_generation_quality.png": "protein_llm_generation_quality.png",
        "web_data_composition.png": "protein_llm_data_composition.png",
        "web_architecture.png": "protein_llm_architecture.png",
        "web_scaling_effect.png": "protein_llm_scaling_effect.png",
        "web_mlp_vs_text_4m.png": "protein_llm_mlp_vs_text_4m.png",
    }
    for src_name, dst_name in web_files.items():
        shutil.copy2(f"{BLOG_FIG}/{src_name}", f"{WEBSITE_DIR}/{dst_name}")
        print(f"  {src_name} -> {dst_name}")
    print(f"Copied {len(web_files)} figures to website.")
else:
    print(f"  WARNING: {WEBSITE_DIR} not found, skipping website copy.")

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"\nBlog figures:    {BLOG_FIG}/pub_*.png, web_*.png")
print(f"Paper figures:   {PAPER_FIG}/*.pdf, *.png")
print(f"Website:         {WEBSITE_DIR}/protein_llm_*.png")
