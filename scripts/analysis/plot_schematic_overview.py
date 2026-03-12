#!/usr/bin/env python3
"""
Schematic overview figure for paper and blog (v2).

Generates panels separately then composes:
  - fig1a_pathways.png        (Panel A: Four Pathways)
  - fig1b_training_pipeline.png (Panel B: Training Pipeline with SFT+GRPO detail)
  - fig1_schematic_overview.png (Composed with research question banner)

Publication-quality design with soft pastel fills, frozen/trainable markers,
clean arrows, and generous whitespace.
"""

import os
import sys

# Ensure this directory is on sys.path for figure_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from figure_style import (
    BLOG_FIGURES_DIR,
    MAIN_FIGURES_DIR,
    PAPER_MAIN_DIR,
    save_main_figure,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ===================================================================
# SOFT PASTEL PALETTE
# ===================================================================
PAL = {
    # Frozen components (cool blues)
    "frozen_bg":       "#DCEEFB",
    "frozen_border":   "#5B9BD5",
    "frozen_text":     "#2B5C8A",

    # Trainable components (warm corals/oranges)
    "train_bg":        "#FDE8D0",
    "train_border":    "#E8875B",
    "train_text":      "#8B4513",

    # LLM (soft greens)
    "llm_bg":          "#D5F5E3",
    "llm_border":      "#5BAD7C",
    "llm_text":        "#2D6A4F",

    # Protein input (light lavender)
    "protein_bg":      "#E8E0F0",
    "protein_border":  "#7B68AE",
    "protein_text":    "#4A3878",

    # Text-only path (neutral gray)
    "text_bg":         "#EDEDED",
    "text_border":     "#999999",
    "text_text":       "#555555",

    # Flamingo (soft red/pink)
    "flam_bg":         "#FADBD8",
    "flam_border":     "#C0392B",
    "flam_text":       "#922B21",

    # Perceiver (soft orange)
    "perc_bg":         "#FFF3E0",
    "perc_border":     "#E67E22",
    "perc_text":       "#935116",

    # MLP (soft blue)
    "mlp_bg":          "#D6EAF8",
    "mlp_border":      "#2E86C1",
    "mlp_text":        "#1A5276",

    # SFT / GRPO stages
    "sft_bg":          "#EBF5FB",
    "sft_border":      "#5DADE2",
    "grpo_bg":         "#FEF9E7",
    "grpo_border":     "#D4AC0D",

    # Research question banner
    "rq_bg":           "#F0F4F8",
    "rq_border":       "#B0BEC5",
    "rq_text":         "#2C3E50",

    # Structural
    "arrow":           "#4A4A4A",
    "arrow_light":     "#AAAAAA",
    "separator":       "#C0C0C0",
    "title_color":     "#2C3E50",
    "subtitle":        "#7B8D9E",
}

# Unicode icons that render in DejaVu Sans
ICON_FROZEN = "\u2744"    # snowflake
ICON_TRAIN  = "\u2731"    # heavy asterisk (trainable marker)


def _box(ax, x, y, w, h, bg, border, text="", fontsize=9,
         fontweight="normal", text_color="#2C3E50", lw=1.5,
         pad=0.2, alpha=1.0, zorder=2, sub_text=None,
         sub_fontsize=7.5, icon=None):
    """Draw a rounded rectangle with text, optional sub-text, and optional icon."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=bg, edgecolor=border,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    cx = x + w / 2
    cy = y + h / 2

    if text and sub_text:
        ax.text(cx, cy + h * 0.15, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight,
                color=text_color, zorder=zorder + 1)
        ax.text(cx, cy - h * 0.2, sub_text, ha="center", va="center",
                fontsize=sub_fontsize, color=PAL["subtitle"],
                style="italic", zorder=zorder + 1)
    elif text:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight,
                color=text_color, zorder=zorder + 1)

    # Frozen/trainable icon in top-right
    if icon:
        ix = x + w - 0.35
        iy = y + h - 0.35
        icon_color = PAL["frozen_text"] if icon == ICON_FROZEN else PAL["train_text"]
        ax.text(ix, iy, icon, ha="center", va="center",
                fontsize=max(fontsize, 8), color=icon_color,
                fontweight="bold", zorder=zorder + 2)

    return box


def _arrow(ax, start, end, color="#4A4A4A", lw=1.3, style="-|>",
           connectionstyle="arc3,rad=0", zorder=3):
    """Draw a clean arrow."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        connectionstyle=connectionstyle,
        zorder=zorder,
        mutation_scale=11,
    )
    ax.add_patch(arrow)
    return arrow


def _section_label(ax, x, y, label, fontsize=12, color=None):
    """Draw a section label."""
    c = color or PAL["title_color"]
    ax.text(x, y, label, ha="left", va="center",
            fontsize=fontsize, fontweight="bold", color=c, zorder=6)


# ===================================================================
# PANEL A: Four Pathways
# ===================================================================
def draw_panel_a():
    """Create Panel A: Four Pathways (architecture overview).

    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 9.5))
    ax.set_xlim(-0.5, 21)
    ax.set_ylim(-0.5, 14.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _section_label(ax, 0.3, 13.8, "(A) Four Pathways")

    # --- Protein Input (far left) ---
    prot_x, prot_y = 0.3, 5.5
    prot_w, prot_h = 2.8, 2.8
    _box(ax, prot_x, prot_y, prot_w, prot_h,
         PAL["protein_bg"], PAL["protein_border"],
         text="Protein\nSequence", fontsize=11, fontweight="bold",
         text_color=PAL["protein_text"], lw=2.0, pad=0.25)
    ax.text(prot_x + prot_w / 2, prot_y + 0.4,
            "MKTL...AVFG", ha="center", va="center",
            fontsize=8, fontfamily="monospace", color=PAL["protein_text"],
            style="italic", zorder=4)

    # --- Fork point ---
    fork_x = prot_x + prot_w + 0.8
    fork_y = prot_y + prot_h / 2

    # Horizontal stem from protein
    ax.plot([prot_x + prot_w + 0.1, fork_x], [fork_y, fork_y],
            color=PAL["arrow"], linewidth=1.8, zorder=3)
    # Fork dot
    ax.plot(fork_x, fork_y, "o", color=PAL["arrow"], markersize=6, zorder=4)

    # -- (a) Text-Only: top path --
    text_y = 12.0
    ax.plot([fork_x, fork_x], [fork_y + 0.2, text_y - 0.7],
            color=PAL["text_border"], linewidth=1.5, linestyle=(0, (4, 3)),
            alpha=0.6, zorder=2)

    tb_x = fork_x + 0.6
    tb_w, tb_h = 4.8, 1.4
    _box(ax, tb_x, text_y - tb_h / 2, tb_w, tb_h,
         PAL["text_bg"], PAL["text_border"],
         text="(a) Text-Only", fontsize=10, fontweight="bold",
         text_color=PAL["text_text"], lw=1.5,
         sub_text="Tokenize AA sequence directly")
    _arrow(ax, (fork_x, text_y), (tb_x - 0.05, text_y),
           color=PAL["text_border"], lw=1.3)

    ax.text(fork_x + 0.35, (fork_y + text_y) / 2 + 1.2, "direct",
            ha="center", va="center", fontsize=8, style="italic",
            color=PAL["text_text"], alpha=0.7, zorder=4)

    # -- ESM-3 Encoder: shared for (b)(c)(d) --
    esm_cy = 5.5
    ax.plot([fork_x, fork_x], [fork_y - 0.2, esm_cy + 1.1],
            color=PAL["frozen_border"], linewidth=1.8, zorder=2)

    esm_x = fork_x + 0.6
    esm_w, esm_h = 3.0, 2.2
    esm_y = esm_cy - esm_h / 2
    _box(ax, esm_x, esm_y, esm_w, esm_h,
         PAL["frozen_bg"], PAL["frozen_border"],
         text="ESM-3", fontsize=12, fontweight="bold",
         text_color=PAL["frozen_text"], lw=2.0,
         sub_text="Protein Encoder", icon=ICON_FROZEN)
    _arrow(ax, (fork_x, esm_cy), (esm_x - 0.05, esm_cy),
           color=PAL["frozen_border"], lw=1.5)

    ax.text(fork_x + 0.35, (fork_y + esm_cy) / 2 - 0.5, "encode",
            ha="center", va="center", fontsize=8, style="italic",
            color=PAL["frozen_text"], alpha=0.7, zorder=4)

    # -- Three projector variants from ESM-3 --
    proj_x = esm_x + esm_w + 1.4
    proj_w = 4.2

    variants = [
        ("(b) MLP Projector", PAL["mlp_bg"], PAL["mlp_border"], PAL["mlp_text"],
         "AttnPool + MLP", 9.2, ICON_TRAIN),
        ("(c) Perceiver", PAL["perc_bg"], PAL["perc_border"], PAL["perc_text"],
         "Perceiver Resampler", 5.8, ICON_TRAIN),
        ("(d) Flamingo", PAL["flam_bg"], PAL["flam_border"], PAL["flam_text"],
         "Gated Cross-Attn", 2.4, ICON_TRAIN),
    ]

    esm_right = esm_x + esm_w
    proj_row_ys = []

    for label, bg, border, tc, desc, row_y, icon in variants:
        row_h = 1.8
        _box(ax, proj_x, row_y - row_h / 2, proj_w, row_h,
             bg, border, text=label, fontsize=10, fontweight="bold",
             text_color=tc, lw=1.5, sub_text=desc, icon=icon)
        proj_row_ys.append(row_y)

        rad = 0.0 if abs(esm_cy - row_y) < 0.5 else (
            -0.25 if esm_cy > row_y else 0.25)
        _arrow(ax, (esm_right + 0.15, esm_cy),
               (proj_x - 0.15, row_y),
               color=PAL["frozen_border"], lw=1.2,
               connectionstyle=f"arc3,rad={rad}")

    # -- LLM block (right side) --
    llm_x = proj_x + proj_w + 1.2
    llm_w, llm_h = 3.4, 10.0
    llm_y = 1.5
    _box(ax, llm_x, llm_y, llm_w, llm_h,
         PAL["llm_bg"], PAL["llm_border"],
         lw=2.0, pad=0.3)

    ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.8,
            "Qwen3", ha="center", va="center",
            fontsize=13, fontweight="bold", color=PAL["llm_text"], zorder=4)
    ax.text(llm_x + llm_w / 2, llm_y + llm_h - 1.5,
            "LLM", ha="center", va="center",
            fontsize=10, color=PAL["llm_text"], zorder=4)

    n_layers = 8
    layer_w = llm_w - 0.8
    layer_h = 0.5
    layer_gap = 0.15
    layer_start_y = llm_y + 0.5
    for i in range(n_layers):
        ly = layer_start_y + i * (layer_h + layer_gap)
        alpha_val = 0.35 + i * 0.07
        _box(ax, llm_x + 0.4, ly, layer_w, layer_h,
             "#C8E6C9", PAL["llm_border"],
             lw=0.8, pad=0.05, alpha=alpha_val)

    for i in [0, 2, 4, 6]:
        ly = layer_start_y + i * (layer_h + layer_gap) + layer_h / 2
        ax.text(llm_x + 0.2, ly, "\u2726",
                ha="center", va="center",
                fontsize=9, color=PAL["flam_border"],
                fontweight="bold", zorder=5)

    ax.text(llm_x + llm_w / 2, llm_y + 0.35,
            "LoRA", ha="center", va="center",
            fontsize=9, color="#27AE60", fontweight="bold",
            style="italic", zorder=4)

    # Arrow: Text-only -> LLM
    _arrow(ax, (tb_x + tb_w + 0.1, text_y),
           (llm_x - 0.1, llm_y + llm_h - 1.0),
           color=PAL["text_border"], lw=1.5)

    # Arrows: projectors -> LLM
    for row_y in proj_row_ys:
        target_y = max(llm_y + 1.0, min(row_y, llm_y + llm_h - 2.0))
        _arrow(ax, (proj_x + proj_w + 0.1, row_y),
               (llm_x - 0.1, target_y),
               color=PAL["arrow"], lw=1.2)

    # Output label above LLM
    out_y = llm_y + llm_h + 0.8
    _box(ax, llm_x + 0.2, out_y - 0.4, llm_w - 0.4, 0.9,
         "#F5F5F5", "#BBBBBB",
         text="Text Output", fontsize=9, fontweight="bold",
         text_color=PAL["subtitle"], lw=1.0, pad=0.1)
    _arrow(ax, (llm_x + llm_w / 2, llm_y + llm_h + 0.1),
           (llm_x + llm_w / 2, out_y - 0.35),
           color=PAL["arrow_light"], lw=1.2)

    # -- Legend (bottom) --
    legend_y = 0.2
    legend_items = [
        (ICON_FROZEN + "  Frozen", PAL["frozen_bg"], PAL["frozen_border"]),
        (ICON_TRAIN + "  Trainable", PAL["train_bg"], PAL["train_border"]),
        ("LLM + LoRA", PAL["llm_bg"], PAL["llm_border"]),
        ("\u2726  Cross-Attn", PAL["flam_bg"], PAL["flam_border"]),
    ]
    legend_x_start = 1.5
    legend_gap = 4.8
    for i, (label, bg, border) in enumerate(legend_items):
        lx = legend_x_start + i * legend_gap
        _box(ax, lx, legend_y - 0.35, 2.0, 0.7,
             bg, border, lw=1.2, pad=0.08)
        ax.text(lx + 2.4, legend_y, label, ha="left", va="center",
                fontsize=9, color=PAL["title_color"], zorder=4)

    return fig


# ===================================================================
# PANEL B: Training Pipeline (detailed SFT + GRPO)
# ===================================================================
def draw_panel_b():
    """Create Panel B: Training Pipeline with detailed SFT and GRPO stages.

    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-0.5, 20)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _section_label(ax, 0.3, 19.5, "(B) Training Pipeline")

    b_cx = 7.5  # center x

    # ===================== Stage 1: SFT =====================
    sft_w, sft_h = 13.5, 9.5
    sft_x = b_cx - sft_w / 2
    sft_y = 9.8
    _box(ax, sft_x, sft_y, sft_w, sft_h,
         PAL["sft_bg"], PAL["sft_border"],
         lw=2.0, pad=0.3)
    ax.text(b_cx, sft_y + sft_h - 0.55,
            "Stage 1: Supervised Fine-Tuning (SFT)", ha="center", va="center",
            fontsize=11, fontweight="bold", color=PAL["frozen_text"], zorder=4)

    # Dataset categories with examples and sample counts
    categories = [
        ("Function Prediction", '"What is the function of this protein?"', "2.15M"),
        ("Catalytic Activity", '"Predict catalytic activity..."', "1.24M"),
        ("General QA", '"Describe this protein\'s properties..."', "0.99M"),
        ("Domain Classification", '"What domains does this contain?"', "0.46M"),
    ]

    cat_start_y = sft_y + sft_h - 2.0
    cat_w = 12.0
    cat_h = 1.35
    cat_gap = 0.2
    cat_x = b_cx - cat_w / 2

    for i, (name, example, count) in enumerate(categories):
        cy = cat_start_y - i * (cat_h + cat_gap)
        # Category box
        _box(ax, cat_x, cy, cat_w, cat_h,
             "#FFFDE7", PAL["sft_border"],
             lw=0.8, pad=0.1)
        # Category name (bold, left-aligned)
        ax.text(cat_x + 0.4, cy + cat_h * 0.65, name,
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                color=PAL["frozen_text"], zorder=4)
        # Example prompt (italic, smaller)
        ax.text(cat_x + 0.4, cy + cat_h * 0.25, example,
                ha="left", va="center", fontsize=7.5,
                color=PAL["subtitle"], style="italic", zorder=4)
        # Sample count (right-aligned, bold)
        ax.text(cat_x + cat_w - 0.4, cy + cat_h * 0.5, count,
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=PAL["frozen_text"], zorder=4)

    # Total and training details
    details_y = sft_y + 0.5
    ax.text(b_cx, details_y + 0.7,
            "Total: 4.89M samples, ~2.1B tokens",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=PAL["frozen_text"], zorder=4)
    ax.text(b_cx, details_y,
            "Chat template  |  LoRA r=8  |  1 epoch",
            ha="center", va="center", fontsize=8.5,
            color=PAL["subtitle"], zorder=4)

    # ===================== Arrow SFT -> GRPO =====================
    _arrow(ax, (b_cx, sft_y - 0.1), (b_cx, sft_y - 0.9),
           color=PAL["arrow"], lw=2.2)
    ax.text(b_cx + 0.5, sft_y - 0.5, "init from SFT",
            ha="left", va="center", fontsize=7.5, style="italic",
            color=PAL["subtitle"], zorder=4)

    # ===================== Stage 2: GRPO =====================
    grpo_h = 8.0
    grpo_y = 1.0
    _box(ax, sft_x, grpo_y, sft_w, grpo_h,
         PAL["grpo_bg"], PAL["grpo_border"],
         lw=2.0, pad=0.3)
    ax.text(b_cx, grpo_y + grpo_h - 0.55,
            "Stage 2: Group Relative Policy Optimization (GRPO)",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="#7D6608", zorder=4)

    # Reward tasks with examples
    rewards = [
        ("GO F1", '"Predict Gene Ontology terms"', "F1 score reward", "10K"),
        ("Stability \u0394\u0394G", '"Predict stability change"', "Numerical accuracy reward", "10K"),
        ("Structure pLDDT", '"Describe structure quality"', "ESMFold pLDDT reward", "10K"),
    ]

    rew_start_y = grpo_y + grpo_h - 2.0
    rew_w = 12.0
    rew_h = 1.55
    rew_gap = 0.2
    rew_x = b_cx - rew_w / 2

    for i, (name, prompt, reward_type, count) in enumerate(rewards):
        ry = rew_start_y - i * (rew_h + rew_gap)
        # Reward box
        _box(ax, rew_x, ry, rew_w, rew_h,
             "#FFF8E1", "#D4AC0D",
             lw=0.8, pad=0.1)
        # Task name (bold, left-aligned)
        ax.text(rew_x + 0.4, ry + rew_h * 0.72, name,
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                color="#7D6608", zorder=4)
        # Example prompt (italic)
        ax.text(rew_x + 0.4, ry + rew_h * 0.42, prompt,
                ha="left", va="center", fontsize=7.5,
                color=PAL["subtitle"], style="italic", zorder=4)
        # Reward type (smaller, left below prompt)
        ax.text(rew_x + 0.4, ry + rew_h * 0.15, reward_type,
                ha="left", va="center", fontsize=7,
                color="#B7950B", fontweight="bold", zorder=4)
        # Sample count (right-aligned)
        ax.text(rew_x + rew_w - 0.4, ry + rew_h * 0.5, count,
                ha="right", va="center", fontsize=9, fontweight="bold",
                color="#7D6608", zorder=4)

    # GRPO details
    grpo_details_y = grpo_y + 0.4
    ax.text(b_cx, grpo_details_y + 0.6,
            "Total: 30K samples",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#7D6608", zorder=4)
    ax.text(b_cx, grpo_details_y,
            "Verifiable rewards  |  No human labels  |  KL penalty",
            ha="center", va="center", fontsize=8.5,
            color=PAL["subtitle"], zorder=4)

    return fig


# ===================================================================
# COMPOSED FIGURE: Research Question + Panel A + Panel B
# ===================================================================
def draw_figure():
    """Create the composed 2-panel schematic overview figure with research question.

    This is called by plot_pub_figures.py for fig1.
    Uses direct matplotlib drawing (not image composition) for crisp text.
    Returns a matplotlib Figure.
    """
    # Total figure: research question banner + two panels side by side
    fig = plt.figure(figsize=(24, 12))
    fig.patch.set_facecolor("white")

    # Research question banner at top (dedicated axes)
    rq_text = ("Does ESM-3 structural embeddings improve post-training "
               "(SFT + task-specific RL) of open-source LLMs for protein understanding?")

    rq_ax = fig.add_axes([0.02, 0.925, 0.96, 0.055])
    rq_ax.set_xlim(0, 1)
    rq_ax.set_ylim(0, 1)
    rq_ax.axis("off")

    banner_box = FancyBboxPatch(
        (0.005, 0.05), 0.99, 0.9,
        boxstyle="round,pad=0.015",
        facecolor=PAL["rq_bg"], edgecolor=PAL["rq_border"],
        linewidth=1.5, zorder=2,
    )
    rq_ax.add_patch(banner_box)
    rq_ax.text(0.5, 0.5, rq_text,
               ha="center", va="center",
               fontsize=14, style="italic", fontweight="normal",
               color=PAL["rq_text"], zorder=3)

    # Panel A: Four Pathways (left, wider)
    ax_a = fig.add_axes([0.01, 0.01, 0.55, 0.9])
    ax_a.set_xlim(-0.5, 21)
    ax_a.set_ylim(-0.5, 14.5)
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    _draw_panel_a_on_ax(ax_a)

    # Panel B: Training Pipeline (right)
    ax_b = fig.add_axes([0.57, 0.01, 0.42, 0.9])
    ax_b.set_xlim(-0.5, 15)
    ax_b.set_ylim(-0.5, 20)
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    _draw_panel_b_on_ax(ax_b)

    # Vertical separator line
    sep_line = fig.add_axes([0.565, 0.03, 0.001, 0.85])
    sep_line.set_xlim(0, 1)
    sep_line.set_ylim(0, 1)
    sep_line.axis("off")
    sep_line.axvline(0.5, color=PAL["separator"], linewidth=1.2, alpha=0.3)

    return fig


def _draw_panel_a_on_ax(ax):
    """Draw Panel A content onto a given axes (for composed figure)."""
    _section_label(ax, 0.3, 13.8, "(A) Four Pathways")

    # --- Protein Input (far left) ---
    prot_x, prot_y = 0.3, 5.5
    prot_w, prot_h = 2.8, 2.8
    _box(ax, prot_x, prot_y, prot_w, prot_h,
         PAL["protein_bg"], PAL["protein_border"],
         text="Protein\nSequence", fontsize=11, fontweight="bold",
         text_color=PAL["protein_text"], lw=2.0, pad=0.25)
    ax.text(prot_x + prot_w / 2, prot_y + 0.4,
            "MKTL...AVFG", ha="center", va="center",
            fontsize=8, fontfamily="monospace", color=PAL["protein_text"],
            style="italic", zorder=4)

    # --- Fork point ---
    fork_x = prot_x + prot_w + 0.8
    fork_y = prot_y + prot_h / 2

    ax.plot([prot_x + prot_w + 0.1, fork_x], [fork_y, fork_y],
            color=PAL["arrow"], linewidth=1.8, zorder=3)
    ax.plot(fork_x, fork_y, "o", color=PAL["arrow"], markersize=6, zorder=4)

    # -- (a) Text-Only: top path --
    text_y = 12.0
    ax.plot([fork_x, fork_x], [fork_y + 0.2, text_y - 0.7],
            color=PAL["text_border"], linewidth=1.5, linestyle=(0, (4, 3)),
            alpha=0.6, zorder=2)

    tb_x = fork_x + 0.6
    tb_w, tb_h = 4.8, 1.4
    _box(ax, tb_x, text_y - tb_h / 2, tb_w, tb_h,
         PAL["text_bg"], PAL["text_border"],
         text="(a) Text-Only", fontsize=10, fontweight="bold",
         text_color=PAL["text_text"], lw=1.5,
         sub_text="Tokenize AA sequence directly")
    _arrow(ax, (fork_x, text_y), (tb_x - 0.05, text_y),
           color=PAL["text_border"], lw=1.3)

    ax.text(fork_x + 0.35, (fork_y + text_y) / 2 + 1.2, "direct",
            ha="center", va="center", fontsize=8, style="italic",
            color=PAL["text_text"], alpha=0.7, zorder=4)

    # -- ESM-3 Encoder --
    esm_cy = 5.5
    ax.plot([fork_x, fork_x], [fork_y - 0.2, esm_cy + 1.1],
            color=PAL["frozen_border"], linewidth=1.8, zorder=2)

    esm_x = fork_x + 0.6
    esm_w, esm_h = 3.0, 2.2
    esm_y = esm_cy - esm_h / 2
    _box(ax, esm_x, esm_y, esm_w, esm_h,
         PAL["frozen_bg"], PAL["frozen_border"],
         text="ESM-3", fontsize=12, fontweight="bold",
         text_color=PAL["frozen_text"], lw=2.0,
         sub_text="Protein Encoder", icon=ICON_FROZEN)
    _arrow(ax, (fork_x, esm_cy), (esm_x - 0.05, esm_cy),
           color=PAL["frozen_border"], lw=1.5)

    ax.text(fork_x + 0.35, (fork_y + esm_cy) / 2 - 0.5, "encode",
            ha="center", va="center", fontsize=8, style="italic",
            color=PAL["frozen_text"], alpha=0.7, zorder=4)

    # -- Three projector variants --
    proj_x = esm_x + esm_w + 1.4
    proj_w = 4.2

    variants = [
        ("(b) MLP Projector", PAL["mlp_bg"], PAL["mlp_border"], PAL["mlp_text"],
         "AttnPool + MLP", 9.2, ICON_TRAIN),
        ("(c) Perceiver", PAL["perc_bg"], PAL["perc_border"], PAL["perc_text"],
         "Perceiver Resampler", 5.8, ICON_TRAIN),
        ("(d) Flamingo", PAL["flam_bg"], PAL["flam_border"], PAL["flam_text"],
         "Gated Cross-Attn", 2.4, ICON_TRAIN),
    ]

    esm_right = esm_x + esm_w
    proj_row_ys = []

    for label, bg, border, tc, desc, row_y, icon in variants:
        row_h = 1.8
        _box(ax, proj_x, row_y - row_h / 2, proj_w, row_h,
             bg, border, text=label, fontsize=10, fontweight="bold",
             text_color=tc, lw=1.5, sub_text=desc, icon=icon)
        proj_row_ys.append(row_y)

        rad = 0.0 if abs(esm_cy - row_y) < 0.5 else (
            -0.25 if esm_cy > row_y else 0.25)
        _arrow(ax, (esm_right + 0.15, esm_cy),
               (proj_x - 0.15, row_y),
               color=PAL["frozen_border"], lw=1.2,
               connectionstyle=f"arc3,rad={rad}")

    # -- LLM block --
    llm_x = proj_x + proj_w + 1.2
    llm_w, llm_h = 3.4, 10.0
    llm_y = 1.5
    _box(ax, llm_x, llm_y, llm_w, llm_h,
         PAL["llm_bg"], PAL["llm_border"],
         lw=2.0, pad=0.3)

    ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.8,
            "Qwen3", ha="center", va="center",
            fontsize=13, fontweight="bold", color=PAL["llm_text"], zorder=4)
    ax.text(llm_x + llm_w / 2, llm_y + llm_h - 1.5,
            "LLM", ha="center", va="center",
            fontsize=10, color=PAL["llm_text"], zorder=4)

    n_layers = 8
    layer_w = llm_w - 0.8
    layer_h = 0.5
    layer_gap = 0.15
    layer_start_y = llm_y + 0.5
    for i in range(n_layers):
        ly = layer_start_y + i * (layer_h + layer_gap)
        alpha_val = 0.35 + i * 0.07
        _box(ax, llm_x + 0.4, ly, layer_w, layer_h,
             "#C8E6C9", PAL["llm_border"],
             lw=0.8, pad=0.05, alpha=alpha_val)

    for i in [0, 2, 4, 6]:
        ly = layer_start_y + i * (layer_h + layer_gap) + layer_h / 2
        ax.text(llm_x + 0.2, ly, "\u2726",
                ha="center", va="center",
                fontsize=9, color=PAL["flam_border"],
                fontweight="bold", zorder=5)

    ax.text(llm_x + llm_w / 2, llm_y + 0.35,
            "LoRA", ha="center", va="center",
            fontsize=9, color="#27AE60", fontweight="bold",
            style="italic", zorder=4)

    # Arrow: Text-only -> LLM
    _arrow(ax, (tb_x + tb_w + 0.1, text_y),
           (llm_x - 0.1, llm_y + llm_h - 1.0),
           color=PAL["text_border"], lw=1.5)

    # Arrows: projectors -> LLM
    for row_y in proj_row_ys:
        target_y = max(llm_y + 1.0, min(row_y, llm_y + llm_h - 2.0))
        _arrow(ax, (proj_x + proj_w + 0.1, row_y),
               (llm_x - 0.1, target_y),
               color=PAL["arrow"], lw=1.2)

    # Output
    out_y = llm_y + llm_h + 0.8
    _box(ax, llm_x + 0.2, out_y - 0.4, llm_w - 0.4, 0.9,
         "#F5F5F5", "#BBBBBB",
         text="Text Output", fontsize=9, fontweight="bold",
         text_color=PAL["subtitle"], lw=1.0, pad=0.1)
    _arrow(ax, (llm_x + llm_w / 2, llm_y + llm_h + 0.1),
           (llm_x + llm_w / 2, out_y - 0.35),
           color=PAL["arrow_light"], lw=1.2)

    # Legend
    legend_y = 0.2
    legend_items = [
        (ICON_FROZEN + "  Frozen", PAL["frozen_bg"], PAL["frozen_border"]),
        (ICON_TRAIN + "  Trainable", PAL["train_bg"], PAL["train_border"]),
        ("LLM + LoRA", PAL["llm_bg"], PAL["llm_border"]),
        ("\u2726  Cross-Attn", PAL["flam_bg"], PAL["flam_border"]),
    ]
    legend_x_start = 1.5
    legend_gap = 4.8
    for i, (label, bg, border) in enumerate(legend_items):
        lx = legend_x_start + i * legend_gap
        _box(ax, lx, legend_y - 0.35, 2.0, 0.7,
             bg, border, lw=1.2, pad=0.08)
        ax.text(lx + 2.4, legend_y, label, ha="left", va="center",
                fontsize=9, color=PAL["title_color"], zorder=4)


def _draw_panel_b_on_ax(ax):
    """Draw Panel B content onto a given axes (for composed figure)."""
    _section_label(ax, 0.3, 19.5, "(B) Training Pipeline")

    b_cx = 7.5

    # ===================== Stage 1: SFT =====================
    sft_w, sft_h = 13.5, 9.5
    sft_x = b_cx - sft_w / 2
    sft_y = 9.8
    _box(ax, sft_x, sft_y, sft_w, sft_h,
         PAL["sft_bg"], PAL["sft_border"],
         lw=2.0, pad=0.3)
    ax.text(b_cx, sft_y + sft_h - 0.55,
            "Stage 1: Supervised Fine-Tuning (SFT)", ha="center", va="center",
            fontsize=11, fontweight="bold", color=PAL["frozen_text"], zorder=4)

    categories = [
        ("Function Prediction", '"What is the function of this protein?"', "2.15M"),
        ("Catalytic Activity", '"Predict catalytic activity..."', "1.24M"),
        ("General QA", '"Describe this protein\'s properties..."', "0.99M"),
        ("Domain Classification", '"What domains does this contain?"', "0.46M"),
    ]

    cat_start_y = sft_y + sft_h - 2.0
    cat_w = 12.0
    cat_h = 1.35
    cat_gap = 0.2
    cat_x = b_cx - cat_w / 2

    for i, (name, example, count) in enumerate(categories):
        cy = cat_start_y - i * (cat_h + cat_gap)
        _box(ax, cat_x, cy, cat_w, cat_h,
             "#FFFDE7", PAL["sft_border"],
             lw=0.8, pad=0.1)
        ax.text(cat_x + 0.4, cy + cat_h * 0.65, name,
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                color=PAL["frozen_text"], zorder=4)
        ax.text(cat_x + 0.4, cy + cat_h * 0.25, example,
                ha="left", va="center", fontsize=7.5,
                color=PAL["subtitle"], style="italic", zorder=4)
        ax.text(cat_x + cat_w - 0.4, cy + cat_h * 0.5, count,
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=PAL["frozen_text"], zorder=4)

    details_y = sft_y + 0.5
    ax.text(b_cx, details_y + 0.7,
            "Total: 4.89M samples, ~2.1B tokens",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=PAL["frozen_text"], zorder=4)
    ax.text(b_cx, details_y,
            "Chat template  |  LoRA r=8  |  1 epoch",
            ha="center", va="center", fontsize=8.5,
            color=PAL["subtitle"], zorder=4)

    # Arrow SFT -> GRPO
    _arrow(ax, (b_cx, sft_y - 0.1), (b_cx, sft_y - 0.9),
           color=PAL["arrow"], lw=2.2)
    ax.text(b_cx + 0.5, sft_y - 0.5, "init from SFT",
            ha="left", va="center", fontsize=7.5, style="italic",
            color=PAL["subtitle"], zorder=4)

    # ===================== Stage 2: GRPO =====================
    grpo_h = 8.0
    grpo_y = 1.0
    _box(ax, sft_x, grpo_y, sft_w, grpo_h,
         PAL["grpo_bg"], PAL["grpo_border"],
         lw=2.0, pad=0.3)
    ax.text(b_cx, grpo_y + grpo_h - 0.55,
            "Stage 2: Group Relative Policy Optimization (GRPO)",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="#7D6608", zorder=4)

    rewards = [
        ("GO F1", '"Predict Gene Ontology terms"', "F1 score reward", "10K"),
        ("Stability \u0394\u0394G", '"Predict stability change"', "Numerical accuracy reward", "10K"),
        ("Structure pLDDT", '"Describe structure quality"', "ESMFold pLDDT reward", "10K"),
    ]

    rew_start_y = grpo_y + grpo_h - 2.0
    rew_w = 12.0
    rew_h = 1.55
    rew_gap = 0.2
    rew_x = b_cx - rew_w / 2

    for i, (name, prompt, reward_type, count) in enumerate(rewards):
        ry = rew_start_y - i * (rew_h + rew_gap)
        _box(ax, rew_x, ry, rew_w, rew_h,
             "#FFF8E1", "#D4AC0D",
             lw=0.8, pad=0.1)
        ax.text(rew_x + 0.4, ry + rew_h * 0.72, name,
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                color="#7D6608", zorder=4)
        ax.text(rew_x + 0.4, ry + rew_h * 0.42, prompt,
                ha="left", va="center", fontsize=7.5,
                color=PAL["subtitle"], style="italic", zorder=4)
        ax.text(rew_x + 0.4, ry + rew_h * 0.15, reward_type,
                ha="left", va="center", fontsize=7,
                color="#B7950B", fontweight="bold", zorder=4)
        ax.text(rew_x + rew_w - 0.4, ry + rew_h * 0.5, count,
                ha="right", va="center", fontsize=9, fontweight="bold",
                color="#7D6608", zorder=4)

    grpo_details_y = grpo_y + 0.4
    ax.text(b_cx, grpo_details_y + 0.6,
            "Total: 30K samples",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#7D6608", zorder=4)
    ax.text(b_cx, grpo_details_y,
            "Verifiable rewards  |  No human labels  |  KL penalty",
            ha="center", va="center", fontsize=8.5,
            color=PAL["subtitle"], zorder=4)


def main():
    print("Generating schematic overview (v2 - separate panels + composed)...")

    # Ensure output directories exist
    MAIN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_MAIN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Panel A separately
    print("\n--- Panel A: Four Pathways ---")
    fig_a = draw_panel_a()
    save_main_figure(fig_a, "fig1a_pathways")

    # 2. Save Panel B separately
    print("\n--- Panel B: Training Pipeline ---")
    fig_b = draw_panel_b()
    save_main_figure(fig_b, "fig1b_training_pipeline")

    # 3. Save composed figure
    print("\n--- Composed: fig1_schematic_overview ---")
    fig = draw_figure()
    save_main_figure(fig, "fig1_schematic_overview")

    # 4. Backward compat copy
    blog_path = str(BLOG_FIGURES_DIR / "schematic_overview.png")
    fig2 = draw_figure()
    fig2.savefig(blog_path, dpi=300, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    plt.close(fig2)
    print(f"  OK {blog_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
