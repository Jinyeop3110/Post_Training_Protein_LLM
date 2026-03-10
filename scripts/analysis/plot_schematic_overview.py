#!/usr/bin/env python3
"""
Schematic overview figure for paper and blog.

4 panels: Question → Approach → Training → Key Results
"""

import os
import sys

# Ensure this directory is on sys.path for figure_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from figure_style import (
    BLOG_FIGURES_DIR,
    SCHEMATIC_COLORS,
    STATUS_COLORS,
    save_main_figure,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ═══════════════════════════════════════════════════════════════════════
# COLOR PALETTE (from figure_style.SCHEMATIC_COLORS + extras for this diagram)
# ═══════════════════════════════════════════════════════════════════════
C = dict(SCHEMATIC_COLORS)
# Extras only used in this schematic
C.update({
    "result_bg":      "#F2F3F4",
    "result_border":  STATUS_COLORS["pending"],
    "result_text":    STATUS_COLORS["pending"],
    "pending_bg":     "#FDEDEC",
    "section_num":    "#FFFFFF",
    "bg":             "#FAFAFA",
})


def add_box(ax, xy, w, h, bg, border, text="", fontsize=8, fontweight="normal",
            text_color="#2C3E50", pad=0.3, lw=1.5, alpha=1.0, zorder=2,
            va="center", ha="center", linestyle="-"):
    """Add a rounded box with centered text."""
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=bg, edgecolor=border,
        linewidth=lw, alpha=alpha, zorder=zorder,
        linestyle=linestyle,
    )
    ax.add_patch(box)
    if text:
        cx = xy[0] + w / 2
        cy = xy[1] + h / 2
        ax.text(cx, cy, text, ha=ha, va=va, fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=zorder + 1)
    return box


def add_arrow(ax, start, end, color="#566573", style="-|>", lw=1.2,
              connectionstyle="arc3,rad=0", zorder=3):
    """Add a fancy arrow."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        connectionstyle=connectionstyle,
        zorder=zorder,
        mutation_scale=12,
    )
    ax.add_patch(arrow)
    return arrow


def add_panel_label(ax, x, y, num, title, fontsize=10):
    """Add a circled number + panel title."""
    circle = plt.Circle((x, y), 0.28, facecolor=C["panel_title_bg"],
                         edgecolor="none", zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, str(num), ha="center", va="center",
            fontsize=fontsize - 1, fontweight="bold", color="white", zorder=6)
    ax.text(x + 0.45, y, title, ha="left", va="center",
            fontsize=fontsize, fontweight="bold", color=C["panel_title_bg"], zorder=6)


def draw_helix(ax, cx, cy, w=0.8, h=0.6, color="#2C7BB6", n_turns=3):
    """Draw a simplified protein helix icon."""
    t = np.linspace(0, n_turns * 2 * np.pi, 100)
    x = cx + (w / 2) * np.sin(t) * np.linspace(0.3, 1, 100)
    y = cy + np.linspace(-h / 2, h / 2, 100)
    # Draw ribbon effect
    for i in range(len(t) - 1):
        alpha = 0.4 + 0.4 * (0.5 + 0.5 * np.sin(t[i]))
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]],
                color=color, linewidth=2.5, alpha=alpha, zorder=3,
                solid_capstyle='round')


def draw_panel_separator(ax, x, y_bot, y_top):
    """Draw a subtle vertical separator between panels."""
    ax.plot([x, x], [y_bot, y_top], color="#D5D8DC", linewidth=1.0,
            linestyle=":", alpha=0.45, zorder=1)


def draw_figure():
    """Create the 4-panel schematic overview."""
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(-0.5, 39)
    ax.set_ylim(-0.5, 14)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor("white")

    # Y baseline for vertical centering
    BOTTOM = 2.5
    TOP = 13.0
    MID = (BOTTOM + TOP) / 2

    # Panel separators
    draw_panel_separator(ax, 8.8, 2.0, 13.0)
    draw_panel_separator(ax, 21.0, 2.0, 13.0)
    draw_panel_separator(ax, 28.8, 2.0, 13.0)

    # ─────────────────────────────────────────────────────────────
    # PANEL 1: QUESTION (x: 0-8.5)
    # ─────────────────────────────────────────────────────────────
    add_panel_label(ax, 0.3, 13.3, 1, "Question")

    # Protein world (left)
    add_box(ax, (0.2, 4.5), 3.4, 7.8,
            C["protein_bg"], C["protein_border"],
            lw=1.8, pad=0.35)
    ax.text(1.9, 11.5, "Protein\nWorld", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C["protein_text"], zorder=4)

    # Helix icon (bigger, centered better)
    draw_helix(ax, 1.9, 9.0, w=1.5, h=1.8, color=C["protein_border"])

    # Sequence text (shifted up slightly for better centering)
    ax.text(1.9, 7.0, "MKTLIF...", ha="center", va="center",
            fontsize=9, fontfamily="monospace", color=C["protein_text"],
            zorder=4, style="italic")
    ax.text(1.9, 5.8, "structure\nfunction\ncatalysis", ha="center", va="center",
            fontsize=8, color="#7B8D9E", zorder=4)

    # Question mark / gap
    ax.text(4.6, 8.5, "?", ha="center", va="center",
            fontsize=42, fontweight="bold", color="#E74C3C", zorder=4, alpha=0.55)

    # LLM world (right of gap)
    add_box(ax, (5.6, 4.5), 3.0, 7.8,
            C["llm_bg"], C["llm_border"],
            lw=1.8, pad=0.35)
    ax.text(7.1, 11.5, "Language\nWorld", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C["llm_text"], zorder=4)

    # LLM icon (stacked layers)
    for i, (yy, a) in enumerate([(10.0, 0.45), (9.4, 0.6), (8.8, 0.75), (8.2, 0.9)]):
        add_box(ax, (6.1, yy), 2.0, 0.4,
                C["llm_bg"], C["llm_border"],
                alpha=a, lw=0.8, pad=0.08, zorder=3)
    ax.text(7.1, 9.0, "LLM", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C["llm_text"], zorder=5)

    ax.text(7.1, 6.8, "reasoning\ngeneration\nknowledge", ha="center", va="center",
            fontsize=8, color="#7B8D9E", zorder=4)

    # Panel question text (lighter to reduce competition)
    ax.text(4.5, 3.5, "Can LLMs understand proteins\nvia structural embeddings?",
            ha="center", va="center", fontsize=8, style="italic",
            color="#AAB7C0", zorder=4)

    # ─────────────────────────────────────────────────────────────
    # PANEL 2: APPROACH (x: 9.2-20.5)
    # Forking layout: Protein Sequence → fork
    #   Top branch (a): Text path → direct to LLM (short, simple)
    #   Bottom branch: ESM-3 → split into (b) MLP, (c) Perceiver, (d) Flamingo → LLM
    # ─────────────────────────────────────────────────────────────
    add_panel_label(ax, 9.5, 13.3, 2, "Approach: Four Pathways")

    # --- Protein Sequence input (left, centered vertically) ---
    prot_cx, prot_cy = 10.2, 8.0
    add_box(ax, (prot_cx - 1.0, prot_cy - 0.7), 2.0, 1.4,
            C["protein_bg"], C["protein_border"],
            lw=1.5, pad=0.2)
    ax.text(prot_cx, prot_cy, "Protein\nSequence", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C["protein_text"], zorder=4)

    # --- Fork point ---
    fork_x = prot_cx + 1.2
    fork_y = prot_cy

    # Fork lines: vertical stem then two branches
    ax.plot([prot_cx + 1.0, fork_x], [fork_y, fork_y],
            color=C["arrow"], linewidth=1.5, zorder=3)

    # ═══════════════════════════════════════════════════
    # TOP BRANCH: (a) Text path — simple, direct
    # ═══════════════════════════════════════════════════
    text_y = 11.0  # high up, visually separate

    # Vertical line up from fork
    ax.plot([fork_x, fork_x], [fork_y, text_y],
            color=C["proj_text_border"], linewidth=1.3, linestyle="--",
            alpha=0.7, zorder=3)

    # Text box
    text_box_x = fork_x + 0.6
    text_box_w = 3.2
    add_box(ax, (text_box_x, text_y - 0.65), text_box_w, 1.3,
            C["proj_text_bg"], C["proj_text_border"],
            lw=1.3, pad=0.18)
    ax.text(text_box_x + text_box_w / 2, text_y + 0.35,
            "(a) Text Only", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C["proj_text_border"], zorder=4)
    ax.text(text_box_x + text_box_w / 2, text_y - 0.2,
            "Raw AA sequence as tokens\nno encoder needed", ha="center", va="center",
            fontsize=8, color="#2C3E50", zorder=4)

    # Horizontal arrow from fork to text box
    add_arrow(ax, (fork_x, text_y), (text_box_x - 0.05, text_y),
              color=C["proj_text_border"], lw=1.3, style="-|>")

    # "no encoder" label on the branch
    ax.text(fork_x - 0.3, (fork_y + text_y) / 2 + 0.5, "direct",
            ha="center", va="center", fontsize=7, style="italic",
            color=C["proj_text_border"], alpha=0.7, rotation=90, zorder=4)

    # ═══════════════════════════════════════════════════
    # BOTTOM BRANCH: ESM-3 encoder → 3 projector variants
    # ═══════════════════════════════════════════════════
    esm_y_center = 5.8  # lower region

    # Vertical line down from fork
    ax.plot([fork_x, fork_x], [fork_y, esm_y_center],
            color=C["esm3_border"], linewidth=1.5, zorder=3)

    # ESM-3 block (SMALLER — proportional)
    esm_x = fork_x + 0.6
    esm_w, esm_h = 2.2, 1.8
    esm_box_y = esm_y_center - esm_h / 2
    add_box(ax, (esm_x, esm_box_y), esm_w, esm_h,
            C["esm3_bg"], C["esm3_border"],
            lw=1.8, pad=0.22)
    ax.text(esm_x + esm_w / 2, esm_y_center + 0.35, "ESM-3",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color=C["esm3_border"], zorder=4)
    ax.text(esm_x + esm_w / 2, esm_y_center - 0.35, "Encoder\n(frozen)",
            ha="center", va="center",
            fontsize=8, color="#95A5A6", style="italic", zorder=4)

    # Arrow from fork to ESM-3
    add_arrow(ax, (fork_x, esm_y_center), (esm_x - 0.05, esm_y_center),
              color=C["esm3_border"], lw=1.5, style="-|>")

    # "structural\nembeddings" label on the branch
    ax.text(fork_x - 0.3, (fork_y + esm_y_center) / 2, "encode",
            ha="center", va="center", fontsize=7, style="italic",
            color=C["esm3_border"], alpha=0.7, rotation=90, zorder=4)

    # --- Three projector variants fanning out from ESM-3 ---
    esm_right = esm_x + esm_w
    proj_x = esm_right + 1.0
    proj_w = 3.2

    esm_variants = [
        ("(b) MLP", C["proj_mlp_bg"], C["proj_mlp_border"],
         "AttnPool \u2192 MLP\n32 tokens \u00b7 30.5M", 7.8),
        ("(c) Perceiver", C["proj_perc_bg"], C["proj_perc_border"],
         "Perceiver Resampler\n32 queries \u00b7 29.4M", 5.8),
        ("(d) Flamingo", C["proj_flam_bg"], C["proj_flam_border"],
         "Gated Cross-Attn\nevery 4th LLM layer", 3.8),
    ]

    esm_row_centers = []
    for label, bg, border, desc, row_y in esm_variants:
        row_h = 1.35
        add_box(ax, (proj_x, row_y - row_h / 2), proj_w, row_h,
                bg, border, lw=1.3, pad=0.18)
        ax.text(proj_x + proj_w / 2, row_y + 0.4, label,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=border, zorder=4)
        ax.text(proj_x + proj_w / 2, row_y - 0.2, desc,
                ha="center", va="center",
                fontsize=8, color="#2C3E50", zorder=4)
        esm_row_centers.append(row_y)

        # Arrow from ESM-3 to projector
        rad = 0.0 if abs(esm_y_center - row_y) < 0.5 else (
            0.15 if esm_y_center > row_y else -0.15)
        add_arrow(ax, (esm_right + 0.1, esm_y_center),
                  (proj_x - 0.1, row_y),
                  color=C["esm3_border"], lw=1.1,
                  connectionstyle=f"arc3,rad={rad}")

    # ═══════════════════════════════════════════════════
    # LLM block (receiving end, right side)
    # ═══════════════════════════════════════════════════
    llm_x = 18.0
    llm_w = 2.6
    llm_h = 8.5
    llm_y = 3.0
    add_box(ax, (llm_x, llm_y), llm_w, llm_h,
            C["llm_bg"], C["llm_border"],
            lw=1.8, pad=0.3)
    ax.text(llm_x + llm_w / 2, 11.0, "Qwen3-8B", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C["llm_text"], zorder=4)
    ax.text(llm_x + llm_w / 2, 10.3, "(LoRA r=8)", ha="center", va="center",
            fontsize=8.5, color="#27AE60", zorder=4)

    # Stacked layer icons inside LLM
    layer_ys = [9.5, 8.8, 8.1, 7.4, 6.7, 6.0, 5.3]
    for j, yy in enumerate(layer_ys):
        a = 0.25 + j * 0.09
        add_box(ax, (llm_x + 0.3, yy), 2.0, 0.45,
                C["llm_bg"], C["llm_border"],
                alpha=a, lw=0.6, pad=0.06, zorder=3)

    # Flamingo cross-attention markers (every other layer)
    xattn_ys = [9.5, 8.1, 6.7, 5.3]
    for yy in xattn_ys:
        ax.plot(llm_x + 0.15, yy + 0.22, marker="*", color=C["proj_flam_border"],
                markersize=12, zorder=5)
    ax.plot([llm_x + 0.05, llm_x + 0.05],
            [xattn_ys[-1] + 0.22, xattn_ys[0] + 0.22],
            color=C["proj_flam_border"], linewidth=1.0, alpha=0.4, zorder=4)
    ax.text(llm_x + llm_w / 2, 4.0, "Flamingo:\ngated \u00d7-attn",
            ha="center", va="center",
            fontsize=8, color=C["proj_flam_border"], fontweight="bold",
            style="italic", zorder=4)

    # Arrow from Text box → LLM (top, direct — short path)
    add_arrow(ax, (text_box_x + text_box_w + 0.1, text_y),
              (llm_x - 0.1, text_y),
              color=C["proj_text_border"], lw=1.5, style="-|>")

    # Arrows from ESM projector variants → LLM
    for row_y in esm_row_centers:
        target_y = max(llm_y + 0.5, min(row_y, llm_y + llm_h - 0.5))
        add_arrow(ax, (proj_x + proj_w + 0.15, row_y),
                  (llm_x - 0.1, target_y),
                  color=C["arrow"], lw=1.1)

    # Approach summary text (lighter)
    ax.text(14.5, 2.5, "Which pathway best bridges\nprotein structure \u2192 language?",
            ha="center", va="center", fontsize=7.5, style="italic",
            color="#AAB7C0", zorder=4)

    # ─────────────────────────────────────────────────────────────
    # PANEL 3: TRAINING (x: 21.5-28.5)
    # ─────────────────────────────────────────────────────────────
    add_panel_label(ax, 21.8, 13.3, 3, "Training Pipeline")

    # SFT block
    sft_x, sft_y, sft_w, sft_h = 21.8, 8.0, 6.2, 3.8
    add_box(ax, (sft_x, sft_y), sft_w, sft_h,
            C["sft_bg"], C["train_border"],
            lw=1.8, pad=0.3)
    ax.text(sft_x + sft_w/2, 11.3, "Stage 1: SFT", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C["train_text"], zorder=4)

    # Dataset pills — spaced evenly across the width
    pill_w = 1.3
    pill_gap = 0.15
    total_pills_w = 4 * pill_w + 3 * pill_gap
    pill_start_x = sft_x + (sft_w - total_pills_w) / 2
    datasets = ["Function\n2.15M", "Catalytic\n1.24M", "General\n0.99M", "Domain\n0.46M"]
    for idx, label in enumerate(datasets):
        dx = pill_start_x + idx * (pill_w + pill_gap)
        add_box(ax, (dx, 9.6), pill_w, 0.95,
                "#FFF8E7", C["train_border"],
                label, fontsize=7, lw=0.8, pad=0.1,
                text_color=C["train_text"])

    ax.text(sft_x + sft_w/2, 8.6, "4.89M samples \u00b7 chat template \u00b7 LoRA r=8",
            ha="center", va="center", fontsize=8,
            color="#7B8D9E", zorder=4)

    # Arrow SFT → GRPO
    add_arrow(ax, (sft_x + sft_w/2, 7.8), (sft_x + sft_w/2, 7.2),
              color=C["train_border"], lw=2.0, style="-|>")

    # GRPO block
    grpo_x, grpo_y, grpo_w, grpo_h = 21.8, 4.0, 6.2, 3.0
    add_box(ax, (grpo_x, grpo_y), grpo_w, grpo_h,
            C["grpo_bg"], C["train_border"],
            lw=1.8, pad=0.3)
    ax.text(grpo_x + grpo_w/2, 6.6, "Stage 2: GRPO", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C["train_text"], zorder=4)

    # Reward pills — spaced evenly
    rpill_w = 1.5
    rpill_gap = 0.25
    total_rpills_w = 3 * rpill_w + 2 * rpill_gap
    rpill_start_x = grpo_x + (grpo_w - total_rpills_w) / 2
    rewards = ["GO F1\nscore", "Stability\n\u0394\u0394G", "Structure\npLDDT"]
    for idx, label in enumerate(rewards):
        dx = rpill_start_x + idx * (rpill_w + rpill_gap)
        add_box(ax, (dx, 4.8), rpill_w, 1.0,
                "#FEF9E7", "#D4AC0D",
                label, fontsize=7, lw=0.8, pad=0.1,
                text_color="#7D6608")

    ax.text(grpo_x + grpo_w/2, 4.4, "Verifiable rewards \u00b7 no human labels",
            ha="center", va="center", fontsize=8,
            color="#7B8D9E", zorder=4)

    # ─────────────────────────────────────────────────────────────
    # PANEL 4: KEY RESULTS (x: 29.2-38.5)
    # ─────────────────────────────────────────────────────────────
    add_panel_label(ax, 29.5, 13.3, 4, "Key Results")

    # Solid outer border (not dashed — only GRPO card is pending)
    res_x, res_y, res_w, res_h = 29.5, 4.0, 8.5, 7.8
    add_box(ax, (res_x, res_y), res_w, res_h,
            "#F8F9FA", "#BDC3C7",
            lw=1.5, pad=0.35, linestyle="-")

    # Four result cards in 2×2 grid
    card_w, card_h = 3.7, 1.7
    col_gap = 0.6
    row_gap = 0.6

    result_items = [
        ("Text vs MLP", "Structural embeddings\n\u2192 faster convergence",
         res_x + 0.3, res_y + res_h - card_h - 0.5, True),
        ("GRPO Impact", "Verifiable rewards\nimprove task accuracy",
         res_x + card_w + col_gap + 0.3, res_y + res_h - card_h - 0.5, False),
        ("Scaling Effect", "50K \u2192 4.89M:\n81% eval improvement",
         res_x + 0.3, res_y + 0.5, True),
        ("Generation", "BLEU=0.31, ROUGE-L=0.51\nprotein-aware output",
         res_x + card_w + col_gap + 0.3, res_y + 0.5, True),
    ]

    for title, desc, rx, ry, confirmed in result_items:
        if confirmed:
            add_box(ax, (rx, ry), card_w, card_h,
                    "#FFFFFF", "#27AE60",
                    lw=1.3, pad=0.18, linestyle="-")
            ax.text(rx + card_w / 2, ry + card_h - 0.35, title, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#2C3E50", zorder=4)
            ax.text(rx + card_w / 2, ry + 0.5, desc, ha="center", va="center",
                    fontsize=7.5, color="#5D6D7E", zorder=4)
            ax.text(rx + card_w - 0.25, ry + card_h - 0.25, "\u2713", ha="center", va="center",
                    fontsize=11, color="#27AE60", fontweight="bold", zorder=5)
        else:
            # GRPO pending — red dashed border
            add_box(ax, (rx, ry), card_w, card_h,
                    "#FFF5F5", C["result_border"],
                    lw=1.8, pad=0.18, linestyle="--")
            ax.text(rx + card_w / 2, ry + card_h - 0.35, title, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=C["result_border"], zorder=4)
            ax.text(rx + card_w / 2, ry + 0.45, desc, ha="center", va="center",
                    fontsize=7.5, color="#999999", zorder=4)
            # PENDING watermark — centered in card, less rotation for readability
            ax.text(rx + card_w / 2, ry + card_h / 2 + 0.05, "PENDING",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=C["result_border"],
                    alpha=0.28, rotation=8, zorder=5)

    # ─────────────────────────────────────────────────────────────
    # FLOW ARROWS between panels (in separator gaps)
    # ─────────────────────────────────────────────────────────────
    flow_y = MID + 0.3

    # Panel 1 → Panel 2
    add_arrow(ax, (8.9, flow_y), (9.3, flow_y),
              color=C["arrow_flow"], lw=2.8, style="-|>")

    # Panel 2 → Panel 3
    add_arrow(ax, (20.8, flow_y), (21.5, flow_y),
              color=C["arrow_flow"], lw=2.8, style="-|>")

    # Panel 3 → Panel 4
    add_arrow(ax, (28.2, flow_y), (29.2, flow_y),
              color=C["arrow_flow"], lw=2.8, style="-|>")

    # ─────────────────────────────────────────────────────────────
    # TITLE
    # ─────────────────────────────────────────────────────────────
    ax.text(19.5, 1.2,
            "Post-Training Protein LLM: Bridging Structural Biology and Language Models",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["panel_title_bg"], zorder=6)

    return fig


def main():
    print("Generating schematic overview...")

    fig = draw_figure()

    # Save as main figure (blog PNG + paper PDF)
    save_main_figure(fig, "fig1_schematic_overview")

    # Also save to blog/figures/ root for backward compat
    blog_path = str(BLOG_FIGURES_DIR / "schematic_overview.png")
    fig2 = draw_figure()
    fig2.savefig(blog_path, dpi=300, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    plt.close(fig2)
    print(f"  OK {blog_path}")

    print("Done.")


if __name__ == "__main__":
    main()
