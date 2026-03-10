#!/usr/bin/env python3
"""
Publication-quality main figures (fig1-fig11) for blog, website, and paper.

Consolidated from blog/data/03-07/pub_figures.py. Uses figure_style.py for
consistent styling across all targets.

Usage:
    # Generate all 11 main figures
    python scripts/analysis/plot_pub_figures.py

    # Generate specific figures only
    python scripts/analysis/plot_pub_figures.py --figures 4 5 6 10

    # Blog-only (skip paper PDFs)
    python scripts/analysis/plot_pub_figures.py --targets blog
"""

import os
import sys

# Ensure this directory is on sys.path for figure_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import (
    DATA_DIR,
    MAIN_FIGURES_DIR,
    PAPER_MAIN_DIR,
    SCHEMATIC_COLORS,
    annotate_bars,
    annotate_best,
    get_approach_label,
    get_label,
    save_main_figure,
    smooth,
    style,
    to_rgba_alpha,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

DATA_02 = DATA_DIR / "03-02"
DATA_06 = DATA_DIR / "03-06"
DATA_07 = DATA_DIR / "03-07"
DATA_09 = DATA_DIR / "03-09"

# Path to results/ directory (blog/data/../../results)
_RESULTS_DIR = DATA_DIR.parent.parent / "results"

# Key experiments for cross-approach comparison (from 03-02 data)
EARLY_EXPS = {
    "sft_lora_esm3_qwen3_8b_it_0225_203237": ("Perceiver", "perceiver"),
    "sft_lora_esm3_qwen3_8b_it_0226_151416": ("ESM3+MLP (50K)", "mlp"),
    "sft_text_qwen3_8b_it_0227_145821": ("Text-only", "text"),
}

MAIN_RUN = "sft_esm3_mlp_combined_qwen3_8b_it_0306_010408"
TEXT_RUN = "sft_text_combined_qwen3_8b_it_0307_190324"

# Hardcoded fallbacks — used when data files are missing
_FINAL_METRICS_FALLBACK = {
    "Base Text": {"eval_loss": 3.124, "train_loss": 3.124, "approach": "text", "steps": 0, "is_base": True},
    "Base ESM3\n(rand proj)": {"eval_loss": 3.341, "train_loss": 3.341, "approach": "mlp", "steps": 0, "is_base": True},
    "Perceiver\n(50K, 3 ep)": {"eval_loss": 1.952, "train_loss": 2.296, "approach": "perceiver", "steps": 1677},
    "ESM3+MLP\n(50K, 3 ep)": {"eval_loss": 1.942, "train_loss": 2.288, "approach": "mlp", "steps": 1677},
    "Text-only\n(50K, 3 ep)": {"eval_loss": 2.415, "train_loss": 2.551, "approach": "text", "steps": 2595},
    "Text-only\n(4.89M)": {"eval_loss": 1.207, "train_loss": 1.643, "approach": "text", "steps": 9647},
    "ESM3+MLP\n(4.89M, 0.34 ep)": {"eval_loss": 0.361, "train_loss": 0.438, "approach": "mlp", "steps": 9750},
}

_GEN_METRICS_FALLBACK = {
    "Overall": {"bleu": 0.307, "rouge_l": 0.483},
    "Function": {"bleu": 0.404, "rouge_l": 0.523},
    "Catalytic": {"bleu": 0.392, "rouge_l": 0.541},
    "General": {"bleu": 0.265, "rouge_l": 0.436},
    "Domain": {"bleu": 0.166, "rouge_l": 0.433},
}

_DATASET_COMP_FALLBACK = {
    "Protein Function": 2_147_000,
    "Catalytic Activity": 1_236_000,
    "General QA": 987_000,
    "Domain/Family": 456_000,
    "Other": 64_000,
}

# Base model vs SFT comparison data (from all_experiments.json 03-09)
# Base: evaluated with sft_eval on 5K samples + 200 generation samples
# SFT perplexity: exp(best_eval_loss) from training logs
# SFT BLEU/ROUGE-L: best generation metrics from eval callbacks
_BASE_VS_SFT_FALLBACK = {
    "ESM3+MLP": {
        "base": {"perplexity": 28.25, "bleu": 0.0025, "rouge_l": 0.0569},
        "sft":  {"perplexity": 1.43,  "bleu": 0.315,  "rouge_l": 0.515},
    },
    "Text-only": {
        "base": {"perplexity": 22.73, "bleu": 0.0033, "rouge_l": 0.0607},
        "sft":  {"perplexity": 3.35,  "bleu": 0.276,  "rouge_l": 0.466},
    },
}


def _find_latest_data_dir():
    """Find the latest date subfolder in blog/data/."""
    if not DATA_DIR.exists():
        return None
    date_dirs = sorted(DATA_DIR.iterdir())
    for d in reversed(date_dirs):
        if d.is_dir() and (d / "run_histories.csv").exists():
            return d
    return None


def _load_json_safe(path, fallback):
    """Load JSON file, return fallback on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARNING: Could not load {path}: {e}, using fallback")
        return fallback


def _load_generation_metrics(experiment_name):
    """Load real generation metrics from an experiment eval directory.

    Looks for generation_metrics.json (standalone evaluation) in
    results/<experiment_name>/eval/. Parses flat generation/bleu_<cat>
    keys into dict-of-dicts for fig6_generation_quality().

    Returns None if the file is not found.
    """
    eval_dir = _RESULTS_DIR / experiment_name / "eval"
    metrics_path = eval_dir / "generation_metrics.json"

    if not metrics_path.exists():
        print(f"  generation_metrics.json not found at {metrics_path}")
        return None

    try:
        with open(metrics_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: Could not load {metrics_path}: {exc}")
        return None

    overall_bleu = raw.get("generation/bleu")
    overall_rouge = raw.get("generation/rouge_l")
    if overall_bleu is None or overall_rouge is None:
        print("  WARNING: generation_metrics.json missing overall bleu/rouge_l")
        return None

    # Discover per-category keys (skip design -- too few samples)
    categories = []
    for key in raw:
        if key.startswith("generation/bleu_") and key != "generation/bleu_design":
            categories.append(key.replace("generation/bleu_", ""))

    gen_metrics = {
        "Overall": {"bleu": overall_bleu, "rouge_l": overall_rouge},
    }
    for cat in sorted(categories):
        label = cat.capitalize()
        gen_metrics[label] = {
            "bleu": raw.get(f"generation/bleu_{cat}", 0.0),
            "rouge_l": raw.get(f"generation/rouge_l_{cat}", 0.0),
        }

    print(f"  Loaded generation metrics from {metrics_path}")
    print(f"    Overall BLEU={overall_bleu:.4f}, ROUGE-L={overall_rouge:.4f}")
    return gen_metrics


def _build_metrics_from_all_experiments(all_exp, base_comp):
    """Build final_metrics dict from all_experiments.json + base_model_comparison.json.

    Returns a dict with base models first, then SFT experiments sorted by
    eval_loss descending (worst first, best last for left-to-right reading).
    """
    final_metrics = {}

    # --- Base model baselines from base_model_comparison.json ---
    if base_comp and "models" in base_comp:
        _BASE_MAP = {
            "base_qwen3_8b_text": "Base Text",
            "esm3_random_projector": "Base ESM3\n(rand proj)",
        }
        _APPROACH_MAP = {
            "text": "text",
            "esm3+mlp": "mlp",
            "esm3+mlp (untrained)": "mlp",
        }
        for key, model in base_comp["models"].items():
            label = _BASE_MAP.get(key)
            if label is None:
                continue
            approach = _APPROACH_MAP.get(model.get("approach", ""), "unknown")
            eval_loss = model.get("eval_loss")
            if eval_loss is not None:
                final_metrics[label] = {
                    "eval_loss": eval_loss,
                    "train_loss": eval_loss,
                    "approach": approach,
                    "steps": 0,
                    "is_base": True,
                }

    # --- SFT experiments from all_experiments.json ---
    if all_exp and "experiments" in all_exp:
        _LABEL_MAP = {
            "sft_lora_esm3_qwen3_8b_it_0225_203237": "Perceiver\n(50K, 3 ep)",
            "sft_lora_esm3_qwen3_8b_it_0226_151416": "ESM3+MLP\n(50K, 3 ep)",
            "sft_text_qwen3_8b_it_0227_145821": "Text-only\n(50K, 3 ep)",
            "sft_esm3_mlp_combined_qwen3_8b_it_0306_010408": "ESM3+MLP\n(4.89M, 0.34 ep)",
            "sft_text_combined_qwen3_8b_it_0307_190324": "Text-only\n(4.89M)",
        }
        for exp in all_exp["experiments"]:
            name = exp.get("name", "")
            label = _LABEL_MAP.get(name)
            if label is None:
                continue
            approach = exp.get("approach", "unknown")
            if approach == "esm3":
                ptype = exp.get("projector_type", "mlp")
                approach = "perceiver" if ptype == "perceiver" else "mlp"
            eval_loss = exp.get("best_eval_loss") or exp.get("final_eval_loss")
            train_loss = exp.get("final_token_avg_loss")
            if eval_loss is None:
                continue
            final_metrics[label] = {
                "eval_loss": round(eval_loss, 3),
                "train_loss": round(train_loss, 3) if train_loss else eval_loss,
                "approach": approach,
                "steps": exp.get("total_steps", 0) or 0,
            }

    # Sort: base models first, then SFT by eval_loss descending
    bases = {k: v for k, v in final_metrics.items() if v.get("is_base")}
    trained = {k: v for k, v in final_metrics.items() if not v.get("is_base")}
    sorted_trained = dict(sorted(trained.items(), key=lambda kv: kv[1]["eval_loss"], reverse=True))
    result = {}
    result.update(bases)
    result.update(sorted_trained)
    return result


def load_metrics(data_dir=None):
    """Load FINAL_METRICS, GEN_METRICS, DATASET_COMP from data files.

    Tries multiple data sources in order:
    1. analysis_summary.json (03-07 format)
    2. all_experiments.json + base_model_comparison.json (03-09 format)
    3. _FINAL_METRICS_FALLBACK (hardcoded)
    """
    data_path = data_dir or _find_latest_data_dir() or DATA_07

    final_metrics = None

    # Try 1: analysis_summary.json (03-07 format)
    summary = _load_json_safe(data_path / "analysis_summary.json", None)
    if summary and "experiments" in summary:
        fm = {}
        for exp_name, exp_data in summary["experiments"].items():
            approach = exp_data.get("approach", "unknown")
            if approach == "esm3":
                approach = "mlp"
            label = exp_data.get("label", exp_name[:20])
            fm[label] = {
                "eval_loss": exp_data.get("best_eval_loss"),
                "train_loss": exp_data.get("final_token_avg_loss"),
                "approach": approach,
                "steps": exp_data.get("total_steps_logged", exp_data.get("total_steps", 0)),
            }
        fm = {k: v for k, v in fm.items() if v.get("eval_loss") is not None}
        if fm:
            final_metrics = fm
            print(f"  Loaded {len(fm)} experiment metrics from {data_path / 'analysis_summary.json'}")

    # Try 2: all_experiments.json + base_model_comparison.json (03-09 format)
    if final_metrics is None:
        all_exp = _load_json_safe(data_path / "all_experiments.json", None)
        base_comp = _load_json_safe(data_path / "base_model_comparison.json", None)
        if all_exp and "experiments" in all_exp:
            fm = _build_metrics_from_all_experiments(all_exp, base_comp)
            if fm:
                # Merge with fallback for any entries not covered by the data files
                # (e.g., early 50K experiments not in the latest all_experiments.json)
                for label, entry in _FINAL_METRICS_FALLBACK.items():
                    if label not in fm:
                        fm[label] = entry
                # Re-sort: base first, then trained by eval_loss descending
                bases = {k: v for k, v in fm.items() if v.get("is_base")}
                trained = {k: v for k, v in fm.items() if not v.get("is_base")}
                sorted_trained = dict(sorted(trained.items(), key=lambda kv: kv[1]["eval_loss"], reverse=True))
                fm = {}
                fm.update(bases)
                fm.update(sorted_trained)
                final_metrics = fm
                print(f"  Loaded {len(fm)} experiment metrics from {data_path / 'all_experiments.json'} + fallbacks")

    # Try 3: fallback
    if final_metrics is None:
        final_metrics = _FINAL_METRICS_FALLBACK
        print("  Using fallback metrics")

    # Load generation metrics from experiment eval directory
    gen_metrics = _load_generation_metrics(MAIN_RUN)
    if gen_metrics is None:
        gen_metrics = _GEN_METRICS_FALLBACK
        print("  Using fallback generation metrics")

    dataset_comp = _DATASET_COMP_FALLBACK

    return final_metrics, gen_metrics, dataset_comp


def load_data(data_dir=None):
    """Load CSV data. Returns (df_early, df_late).

    Uses the latest date-stamped data directory by default so that
    newly exported run histories (e.g. longer text runs) are picked up
    automatically.
    """
    df_early = pd.read_csv(str(DATA_02 / "run_histories.csv"))
    late_path = (data_dir or _find_latest_data_dir() or DATA_07) / "run_histories.csv"
    df_late = pd.read_csv(str(late_path))
    return df_early, df_late


# ═══════════════════════════════════════════════════════════════════════
# FIGURE FUNCTIONS (fig1-fig9)
# Each returns a matplotlib Figure object.
# ═══════════════════════════════════════════════════════════════════════

def fig1_schematic_overview():
    """Fig 1: 4-panel project overview (Question, Approach, Training, Results).

    This is a diagram figure — uses SCHEMATIC_COLORS, not approach colors.
    Delegates to plot_schematic_overview.py for the actual drawing.
    """
    # Import the schematic drawer
    from plot_schematic_overview import draw_figure
    fig = draw_figure()
    return fig


def fig2_architecture(colors):
    """Fig 2: Four architecture approaches in a 4-column layout.

    Shows all four protein-LLM pathways side-by-side:
    (a) Text-only, (b) ESM-3+MLP, (c) ESM-3+Perceiver, (d) ESM-3+Flamingo.
    Bottom-to-top flow: Protein -> Encoder -> Projector -> LLM -> Output.
    """
    from matplotlib.patches import Patch
    SC = SCHEMATIC_COLORS

    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    fig.patch.set_facecolor("white")

    # ── Shared drawing helpers ──────────────────────────────────────

    def _box(ax, xy, w, h, bg, border, text="", fontsize=9,
             fontweight="normal", text_color="#2C3E50", lw=1.5,
             linestyle="-", sub_text=None):
        """Draw a rounded box with optional sub-text below main text."""
        patch = FancyBboxPatch(
            xy, w, h,
            boxstyle="round,pad=0.25",
            facecolor=bg, edgecolor=border,
            linewidth=lw, linestyle=linestyle, zorder=2,
        )
        ax.add_patch(patch)
        cx, cy = xy[0] + w / 2, xy[1] + h / 2
        if text and sub_text:
            ax.text(cx, cy + 0.18, text, ha="center", va="center",
                    fontsize=fontsize, fontweight=fontweight,
                    color=text_color, zorder=3)
            ax.text(cx, cy - 0.22, sub_text, ha="center", va="center",
                    fontsize=fontsize - 1.5, color="#7B8D9E",
                    style="italic", zorder=3)
        elif text:
            ax.text(cx, cy, text, ha="center", va="center",
                    fontsize=fontsize, fontweight=fontweight,
                    color=text_color, zorder=3)

    def _arrow(ax, x, y_start, y_end, color="#566573"):
        """Vertical upward arrow."""
        arrow = FancyArrowPatch(
            (x, y_start), (x, y_end),
            arrowstyle="-|>", color=color,
            linewidth=1.3, mutation_scale=12, zorder=3,
        )
        ax.add_patch(arrow)

    # ── Layout constants ────────────────────────────────────────────
    COL_W = 4.0
    BOX_W = 3.2
    BOX_X = (COL_W - BOX_W) / 2
    ARR_X = COL_W / 2

    Y_PROTEIN = 0.3
    Y_ENCODER = 2.2
    BOX_H     = 1.2
    BOX_H_SM  = 1.0
    Y_LLM     = 6.5
    Y_OUTPUT  = 8.8

    # ── Column definitions ──────────────────────────────────────────
    col_defs = [
        {
            "title": "(a) Text-Only",
            "approach": "text",
            "has_encoder": False,
            "note": "~2M trainable (LoRA only)",
        },
        {
            "title": "(b) ESM-3 + MLP",
            "approach": "mlp",
            "has_encoder": True,
            "note": "~32.5M trainable",
        },
        {
            "title": "(c) ESM-3 + Perceiver",
            "approach": "perceiver",
            "has_encoder": True,
            "note": "~29.4M trainable",
        },
        {
            "title": "(d) ESM-3 + Flamingo",
            "approach": "flamingo",
            "has_encoder": True,
            "note": "~120-150M trainable (no LoRA)",
        },
    ]

    for ax, col in zip(axes, col_defs):
        ax.set_xlim(-0.3, COL_W + 0.3)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect("equal")
        ax.axis("off")

        approach = col["approach"]
        approach_color = colors.get(approach, "#808080")

        # --- Title ---
        ax.text(COL_W / 2, 10.1, col["title"], ha="center", va="center",
                fontsize=11, fontweight="bold", color=approach_color, zorder=4)

        # --- Output box ---
        _box(ax, (BOX_X, Y_OUTPUT), BOX_W, BOX_H_SM,
             "#F2F3F4", "#95A5A6",
             text="Text Output", fontsize=9, fontweight="bold",
             text_color="#2C3E50")

        # --- LLM box ---
        is_flamingo = (approach == "flamingo")
        llm_h = 1.8 if is_flamingo else BOX_H
        llm_label = "LLM (frozen)" if is_flamingo else "LLM + LoRA"
        _box(ax, (BOX_X, Y_LLM), BOX_W, llm_h,
             SC["llm_bg"], SC["llm_border"],
             text=llm_label, fontsize=9, fontweight="bold",
             text_color=SC["llm_text"],
             sub_text="Qwen3-8B")

        # Flamingo cross-attention markers
        if is_flamingo:
            for yy in [Y_LLM + 0.5, Y_LLM + 0.95, Y_LLM + 1.4]:
                ax.plot(BOX_X + 0.15, yy, marker="*",
                        color=SC["proj_flam_border"], markersize=8, zorder=5)
            ax.text(BOX_X + BOX_W - 0.15, Y_LLM + 0.95,
                    "gated\n\u00d7-attn", ha="right", va="center",
                    fontsize=6.5, color=SC["proj_flam_border"],
                    fontweight="bold", style="italic", zorder=4)

        # Arrow: LLM -> Output
        _arrow(ax, ARR_X, Y_LLM + llm_h + 0.05, Y_OUTPUT - 0.05, "#566573")

        # --- Text-only path ---
        if approach == "text":
            tok_y = 3.5
            _box(ax, (BOX_X, tok_y), BOX_W, BOX_H,
                 "#F2F3F4", "#95A5A6",
                 text="embed_tokens()", fontsize=8.5,
                 text_color="#566573",
                 sub_text="tokenize AA sequence")

            ax.text(COL_W / 2, 2.7,
                    "protein is tokenized using\nthe LLM tokenizer",
                    ha="center", va="center",
                    fontsize=6.5, color="#AAB7C0", style="italic", zorder=3)

            # Arrows
            _arrow(ax, ARR_X, Y_PROTEIN + BOX_H_SM + 0.05, tok_y - 0.05,
                   "#566573")
            _arrow(ax, ARR_X, tok_y + BOX_H + 0.05, Y_LLM - 0.05,
                   "#566573")

            # Spacer text
            ax.text(COL_W / 2, 5.5,
                    "Text prompt\n+ AA tokens",
                    ha="center", va="center",
                    fontsize=7, color="#95A5A6", zorder=3)

        # --- Encoder-based paths ---
        else:
            # ESM-3 Encoder
            _box(ax, (BOX_X, Y_ENCODER), BOX_W, BOX_H,
                 SC["esm3_bg"], SC["esm3_border"],
                 text="ESM-3 Encoder", fontsize=9, fontweight="bold",
                 text_color=SC["esm3_border"],
                 sub_text="frozen, 1536-dim")
            _arrow(ax, ARR_X, Y_PROTEIN + BOX_H_SM + 0.05,
                   Y_ENCODER - 0.05, "#566573")

            # Projector boxes
            if approach == "mlp":
                pool_y = 3.7
                _box(ax, (BOX_X, pool_y), BOX_W, BOX_H_SM,
                     SC["proj_mlp_bg"], SC["proj_mlp_border"],
                     text="Attention Pooling", fontsize=8,
                     text_color=SC["proj_mlp_border"],
                     sub_text="32 tokens")
                _arrow(ax, ARR_X, Y_ENCODER + BOX_H + 0.05,
                       pool_y - 0.05, SC["proj_mlp_border"])

                proj_y = 5.0
                _box(ax, (BOX_X, proj_y), BOX_W, BOX_H,
                     SC["proj_mlp_bg"], SC["proj_mlp_border"],
                     text="MLP Projector", fontsize=9, fontweight="bold",
                     text_color=SC["proj_mlp_border"],
                     sub_text="1536 \u2192 2560")
                _arrow(ax, ARR_X, pool_y + BOX_H_SM + 0.05,
                       proj_y - 0.05, SC["proj_mlp_border"])
                _arrow(ax, ARR_X, proj_y + BOX_H + 0.05,
                       Y_LLM - 0.05, "#566573")

            elif approach == "perceiver":
                proj_y = 4.2
                proj_h = 1.5
                _box(ax, (BOX_X, proj_y), BOX_W, proj_h,
                     SC["proj_perc_bg"], SC["proj_perc_border"],
                     text="Perceiver Resampler", fontsize=9, fontweight="bold",
                     text_color=SC["proj_perc_border"],
                     sub_text="32 queries \u00b7 2560-dim")
                _arrow(ax, ARR_X, Y_ENCODER + BOX_H + 0.05,
                       proj_y - 0.05, SC["proj_perc_border"])
                _arrow(ax, ARR_X, proj_y + proj_h + 0.05,
                       Y_LLM - 0.05, "#566573")

            elif approach == "flamingo":
                proj_y = 4.2
                proj_h = 1.5
                _box(ax, (BOX_X, proj_y), BOX_W, proj_h,
                     SC["proj_flam_bg"], SC["proj_flam_border"],
                     text="Perceiver Resampler", fontsize=9, fontweight="bold",
                     text_color=SC["proj_flam_border"],
                     sub_text="64 queries \u00b7 6 layers")
                _arrow(ax, ARR_X, Y_ENCODER + BOX_H + 0.05,
                       proj_y - 0.05, SC["proj_flam_border"])
                _arrow(ax, ARR_X, proj_y + proj_h + 0.05,
                       Y_LLM - 0.05, "#566573")

        # --- Protein input (bottom) ---
        _box(ax, (BOX_X, Y_PROTEIN), BOX_W, BOX_H_SM,
             SC["protein_bg"], SC["protein_border"],
             text="Protein: MKTL...AVFG", fontsize=8,
             text_color=SC["protein_text"])

        # --- Param note ---
        ax.text(COL_W / 2, -0.25, col["note"],
                ha="center", va="center",
                fontsize=7.5, color="#7B8D9E", style="italic", zorder=3)

    # ── Legend ───────────────────────────────────────────────────────
    legend_items = [
        (SC["esm3_bg"], SC["esm3_border"], "Frozen (pretrained)"),
        (SC["proj_mlp_bg"], SC["proj_mlp_border"], "Trainable"),
        (SC["llm_bg"], SC["llm_border"], "LLM + LoRA"),
        (SC["proj_flam_bg"], SC["proj_flam_border"], "Gated Cross-Attention"),
    ]
    legend_patches = [
        Patch(facecolor=bg, edgecolor=border, linewidth=1.2, label=label)
        for bg, border, label in legend_items
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Protein-LLM Architecture: Four Approaches",
                 fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def fig3_data_composition(colors, dataset_comp=None):
    """Fig 3: Dataset breakdown (4.89M samples, 6 sources, 4 categories)."""
    comp = dataset_comp or _DATASET_COMP_FALLBACK
    labels = list(comp.keys())
    sizes = list(comp.values())
    total = sum(sizes)
    pcts = [s / total * 100 for s in sizes]

    cmap = plt.cm.Blues
    bar_colors = [cmap(0.3 + 0.6 * i / len(labels)) for i in range(len(labels))]

    fig, ax = style.subplots()
    bars = ax.barh(labels, [s / 1e6 for s in sizes], color=bar_colors,
                   edgecolor="white", linewidth=0.8)

    for bar, pct, size in zip(bars, pcts, sizes):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{size/1e6:.2f}M ({pct:.0f}%)",
                ha="left", va="center", fontsize=plt.rcParams['font.size'] - 1)

    ax.set_xlabel("Samples (millions)")
    ax.set_title(f"Training Data Composition ({total/1e6:.2f}M total)")
    ax.set_xlim(0, max(sizes) / 1e6 * 1.35)
    ax.invert_yaxis()
    style.clean_axes(ax)
    fig.tight_layout()
    return fig


def fig4_main_run_progress(df_late, colors):
    """Fig 4: Train + eval loss curves (MLP 4.89M, colored markers)."""
    df_main = df_late[df_late["experiment"] == MAIN_RUN].copy()
    train = df_main.dropna(subset=["token_avg_loss"]).sort_values("step")
    eval_d = df_main.dropna(subset=["eval_loss"]).sort_values("step")

    fig, ax = style.subplots()

    # Raw (faint)
    ax.plot(train["step"], train["token_avg_loss"],
            color=colors["mlp"], linewidth=0.4, alpha=0.25)
    # Smoothed
    window = 50
    if len(train) > window:
        smoothed = smooth(train["token_avg_loss"], window)
        ax.plot(train["step"], smoothed,
                color=colors["mlp"], linewidth=1.8, alpha=0.9,
                label=f"Train (MA-{window})")
    # Eval
    if len(eval_d) > 0:
        ax.plot(eval_d["step"], eval_d["eval_loss"],
                color="#d62728", marker="D", markersize=5, linewidth=0,
                zorder=5, label=get_label("eval_loss"))

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Loss")
    ax.set_title("ESM3+MLP on 4.89M Samples: Training Progress")
    ax.legend()
    style.clean_axes(ax)
    fig.tight_layout()
    return fig


def fig5_eval_loss(df_late, colors):
    """Fig 5: Eval loss with best-point annotation (MLP vs Text)."""
    fig, ax = style.subplots()

    # MLP eval loss
    df_main = df_late[df_late["experiment"] == MAIN_RUN].copy()
    eval_mlp = df_main.dropna(subset=["eval_loss"]).sort_values("step")

    if len(eval_mlp) > 0:
        ax.plot(eval_mlp["step"], eval_mlp["eval_loss"],
                color=colors["mlp"], marker="o", markersize=5,
                linewidth=1.8, alpha=0.85, label=f"{get_approach_label('mlp')} (4.89M)")

        best_idx = eval_mlp["eval_loss"].idxmin()
        best_step = eval_mlp.loc[best_idx, "step"]
        best_val = eval_mlp.loc[best_idx, "eval_loss"]
        annotate_best(ax, best_step, best_val, best_val, colors["mlp"])

    # Text eval loss
    df_text = df_late[df_late["experiment"] == TEXT_RUN].copy()
    eval_text = df_text.dropna(subset=["eval_loss"]).sort_values("step")

    if len(eval_text) > 0:
        ax.plot(eval_text["step"], eval_text["eval_loss"],
                color=colors["text"], marker="s", markersize=4,
                linewidth=1.8, alpha=0.85, label=f"{get_approach_label('text')} (4.89M)")

        best_idx_t = eval_text["eval_loss"].idxmin()
        best_step_t = eval_text.loc[best_idx_t, "step"]
        best_val_t = eval_text.loc[best_idx_t, "eval_loss"]
        annotate_best(ax, best_step_t, best_val_t, best_val_t, colors["text"],
                      offset=(10, 15))

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("eval_loss"))
    ax.set_title("Validation Loss: ESM3+MLP vs Text-only on 4.89M")
    ax.legend()
    style.clean_axes(ax)
    fig.tight_layout()
    return fig


def fig6_generation_quality(colors, gen_metrics=None):
    """Fig 6: BLEU & ROUGE-L by task category."""
    metrics = gen_metrics or _GEN_METRICS_FALLBACK
    categories = list(metrics.keys())
    bleu = [metrics[c]["bleu"] for c in categories]
    rouge = [metrics[c]["rouge_l"] for c in categories]

    fig, ax = style.subplots()
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, bleu, width, label=get_label("bleu"),
                   color=colors["mlp"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, rouge, width, label=get_label("rouge_l"),
                   color=colors["perceiver"], alpha=0.85, edgecolor="white")

    annotate_bars(ax, bars1, bleu, fmt="{:.3f}", offset=0.008)
    annotate_bars(ax, bars2, rouge, fmt="{:.3f}", offset=0.008)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_title("Generation Quality: BLEU & ROUGE-L by Task (Final Checkpoint)")
    ax.legend()
    ax.set_ylim(0, max(max(rouge), max(bleu)) * 1.3)
    style.clean_axes(ax)
    fig.tight_layout()
    return fig


def fig7_scaling_effect(colors):
    """Fig 7: 50K -> 4.89M: 81% eval_loss improvement."""
    fig, axes = style.subplots(nrows=1, ncols=2)

    # Left: eval loss bars
    ax = axes[0]
    names = ["50K\n(3 ep)", "4.89M\n(0.34 ep)"]
    eval_vals = [1.942, 0.361]
    bars = ax.bar(names, eval_vals, color=colors["mlp"], edgecolor="white", linewidth=0.8)
    bars[0].set_alpha(0.6)
    annotate_bars(ax, bars, eval_vals, offset=0.03)
    ax.set_ylabel(get_label("eval_loss"))
    ax.set_title("ESM3+MLP Eval Loss")
    ax.set_ylim(0, 2.5)
    style.clean_axes(ax)

    # Right: improvement percentage
    ax = axes[1]
    improvement = (1.942 - 0.361) / 1.942 * 100
    ax.text(0.5, 0.6, f"{improvement:.0f}%", ha="center", va="center",
            fontsize=plt.rcParams['font.size'] * 4, fontweight="bold",
            color=colors["mlp"], transform=ax.transAxes)
    ax.text(0.5, 0.35, "eval loss\nimprovement", ha="center", va="center",
            fontsize=plt.rcParams['font.size'] + 1, color="#666666",
            transform=ax.transAxes)
    ax.text(0.5, 0.15, "50K \u2192 4.89M samples\n(98\u00d7 more data)",
            ha="center", va="center",
            fontsize=plt.rcParams['font.size'] - 1, color="#999999",
            transform=ax.transAxes)
    ax.axis("off")

    fig.suptitle("Effect of Scaling Training Data (ESM3+MLP)",
                 fontsize=plt.rcParams['axes.titlesize'], fontweight="bold")
    fig.tight_layout()
    return fig


def fig8_mlp_vs_text(df_late, colors):
    """Fig 8: MLP vs Text-only on same 4.89M dataset."""
    fig, ax = style.subplots()

    # MLP main run
    df_mlp = df_late[df_late["experiment"] == MAIN_RUN].dropna(
        subset=["token_avg_loss"]).sort_values("step")
    max_step = 0
    if len(df_mlp) > 0:
        max_step = max(max_step, df_mlp["step"].max())
        window = 50
        if len(df_mlp) > window:
            smoothed = smooth(df_mlp["token_avg_loss"], window)
            ax.plot(df_mlp["step"], smoothed,
                    color=colors["mlp"], linewidth=1.8, alpha=0.9,
                    label=f"{get_approach_label('mlp')} (encoder)")
        else:
            ax.plot(df_mlp["step"], df_mlp["token_avg_loss"],
                    color=colors["mlp"], linewidth=1.5, alpha=0.85,
                    label=f"{get_approach_label('mlp')} (encoder)")

    # Text baseline
    df_txt = df_late[df_late["experiment"] == TEXT_RUN].dropna(
        subset=["token_avg_loss"]).sort_values("step")
    if len(df_txt) > 0:
        max_step = max(max_step, df_txt["step"].max())
        ax.plot(df_txt["step"], df_txt["token_avg_loss"],
                color=colors["text"], linewidth=1.8, alpha=0.85,
                marker="o", markersize=3,
                label=f"{get_approach_label('text')} (no encoder)")

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("token_avg_loss"))
    ax.set_title("ESM3+MLP vs Text-only on 4.89M Combined Dataset")
    ax.legend()
    if max_step > 0:
        ax.set_xlim(0, max_step * 1.05)
    style.clean_axes(ax)
    fig.tight_layout()
    return fig


def fig9_final_comparison(colors, final_metrics=None):
    """Fig 9: Cross-approach bar chart with base models and all SFT configs."""
    metrics = final_metrics or _FINAL_METRICS_FALLBACK
    names = list(metrics.keys())
    eval_vals = [metrics[n]["eval_loss"] for n in names]
    train_vals = [metrics[n]["train_loss"] for n in names]
    is_base = [metrics[n].get("is_base", False) for n in names]
    bar_colors = [colors.get(metrics[n]["approach"], colors.get("unknown", "#7f7f7f")) for n in names]

    fig, ax = style.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35

    # Train loss bars (lighter)
    bars1 = ax.bar(x - width/2, train_vals, width,
                   label=get_label("train_loss"),
                   color=[to_rgba_alpha(c, 0.6) for c in bar_colors],
                   edgecolor=bar_colors, linewidth=1.2)
    # Eval loss bars (solid)
    bars2 = ax.bar(x + width/2, eval_vals, width,
                   label=get_label("eval_loss"),
                   color=bar_colors, edgecolor="white", linewidth=0.5)

    # Add hatching to base model bars to visually distinguish them
    for i, base in enumerate(is_base):
        if base:
            bars1[i].set_hatch("///")
            bars1[i].set_alpha(0.5)
            bars2[i].set_hatch("///")
            bars2[i].set_alpha(0.7)

    annotate_bars(ax, bars2, eval_vals, offset=0.03)

    # Draw a vertical separator between base models and SFT models
    n_base = sum(is_base)
    if 0 < n_base < len(names):
        sep_x = n_base - 0.5
        ax.axvline(sep_x, color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)
        ax.text(sep_x - 0.1, ax.get_ylim()[1] * 0.97, "Base",
                ha="right", va="top", fontsize=plt.rcParams['font.size'] - 2,
                color="#999999", style="italic")
        ax.text(sep_x + 0.1, ax.get_ylim()[1] * 0.97, "Fine-tuned",
                ha="left", va="top", fontsize=plt.rcParams['font.size'] - 2,
                color="#999999", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=plt.rcParams['font.size'] - 1)
    ax.set_ylabel("Loss")
    ax.set_title("Final Metrics: Base Model Baselines vs Fine-tuned Approaches")
    ax.legend()
    ax.set_ylim(0, max(max(train_vals), max(eval_vals)) * 1.25)
    style.clean_axes(ax)
    fig.tight_layout()
    return fig


def fig10_base_vs_finetuned(colors, base_vs_sft=None):
    """Fig 10: Base model vs SFT — dramatic improvement across all metrics.

    Grouped bar chart with 3 metric panels (Perplexity, BLEU, ROUGE-L),
    comparing Base vs SFT for both ESM3+MLP and Text-only approaches.
    """
    data = base_vs_sft or _BASE_VS_SFT_FALLBACK
    approaches = list(data.keys())  # ["ESM3+MLP", "Text-only"]

    metrics_cfg = [
        ("perplexity", "Perplexity", "{:.1f}", True),    # (key, label, fmt, lower_is_better)
        ("bleu",       "BLEU",       "{:.3f}", False),
        ("rouge_l",    "ROUGE-L",    "{:.3f}", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(approaches))
    width = 0.32

    for ax, (metric_key, metric_label, fmt, lower_better) in zip(axes, metrics_cfg):
        base_vals = [data[a]["base"][metric_key] for a in approaches]
        sft_vals = [data[a]["sft"][metric_key] for a in approaches]

        # Base bars: lighter/alpha version of approach colors
        approach_keys = ["mlp", "text"]
        base_colors = [to_rgba_alpha(colors[k], 0.35) for k in approach_keys]
        sft_colors = [colors[k] for k in approach_keys]

        bars_base = ax.bar(x - width / 2, base_vals, width,
                           label="Base Model",
                           color=base_colors,
                           edgecolor=[colors[k] for k in approach_keys],
                           linewidth=1.0, linestyle="--")
        bars_sft = ax.bar(x + width / 2, sft_vals, width,
                          label="After SFT",
                          color=sft_colors,
                          edgecolor="white", linewidth=0.5)

        # Annotate bar values
        for bar, val in zip(bars_base, base_vals):
            y_pos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos * 1.02,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=plt.rcParams["font.size"] - 1,
                    fontweight="bold", color="#666666")

        for bar, val in zip(bars_sft, sft_vals):
            y_pos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos * 1.02,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=plt.rcParams["font.size"] - 1,
                    fontweight="bold", color="#333333")

        # Improvement annotations (between the bar pairs)
        for i, approach in enumerate(approaches):
            base_v = base_vals[i]
            sft_v = sft_vals[i]
            if lower_better:
                ratio = base_v / sft_v if sft_v > 0 else float("inf")
                arrow_text = f"{ratio:.0f}x lower"
            else:
                ratio = sft_v / base_v if base_v > 0 else float("inf")
                arrow_text = f"{ratio:.0f}x"

            # Place improvement text above the taller bar
            y_top = max(base_v, sft_v)
            y_annot = y_top * 1.18 if metric_key != "perplexity" else y_top * 1.12
            ax.annotate(
                arrow_text,
                xy=(i, y_annot),
                ha="center", va="bottom",
                fontsize=plt.rcParams["font.size"],
                fontweight="bold",
                color="#C0392B",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3F0",
                          edgecolor="#C0392B", linewidth=0.8, alpha=0.9),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(approaches)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)

        # Expand y-axis to fit annotations
        ymax = max(max(base_vals), max(sft_vals))
        ax.set_ylim(0, ymax * 1.45)

        ax.legend(loc="upper right", fontsize=plt.rcParams["legend.fontsize"])
        style.clean_axes(ax)

    fig.suptitle("Base Model vs Fine-tuned: Impact of SFT Training",
                 fontsize=plt.rcParams["axes.titlesize"] + 2, fontweight="bold",
                 y=1.02)
    fig.tight_layout()
    return fig


def _load_generation_over_steps():
    """Load generation_over_steps.csv from blog/data/03-09/.

    Returns a DataFrame with columns: approach, step, bleu_overall, rouge_l_overall.
    Filters to only the two main approaches (esm3_mlp and text).
    """
    csv_path = DATA_09 / "generation_over_steps.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return None

    df = pd.read_csv(str(csv_path))
    # Keep only the main MLP and Text runs (exclude thinking variant)
    df = df[df["approach"].isin(["esm3_mlp", "text"])].copy()
    print(f"  Loaded {len(df)} rows from {csv_path}")
    return df


def fig11_generation_over_steps(colors, gen_df=None):
    """Fig 11: Generation quality progression (BLEU & ROUGE-L) during training.

    2-panel figure: left=BLEU, right=ROUGE-L. Lines for MLP and Text with
    LOWESS smoothing and final-value annotations.
    """
    if gen_df is None:
        gen_df = _load_generation_over_steps()
    if gen_df is None or len(gen_df) == 0:
        print("  No generation-over-steps data available, skipping fig11")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    metric_panels = [
        ("bleu_overall", "BLEU", axes[0]),
        ("rouge_l_overall", "ROUGE-L", axes[1]),
    ]

    approach_cfg = {
        "esm3_mlp": {
            "color": colors["mlp"],
            "label": get_approach_label("mlp"),
            "marker": "o",
            "markersize": 4,
        },
        "text": {
            "color": colors["text"],
            "label": get_approach_label("text"),
            "marker": "s",
            "markersize": 3.5,
        },
    }

    for metric_col, metric_label, ax in metric_panels:
        for approach_key, cfg in approach_cfg.items():
            sub = gen_df[gen_df["approach"] == approach_key].sort_values("step")
            if len(sub) == 0:
                continue

            steps = sub["step"].values
            vals = sub[metric_col].values

            # Raw data points (faint)
            ax.plot(steps, vals,
                    color=cfg["color"], marker=cfg["marker"],
                    markersize=cfg["markersize"], linewidth=0.4,
                    alpha=0.3, zorder=2)

            # Smoothed trend (LOWESS-like via rolling mean)
            vals_series = pd.Series(vals)
            window = max(3, len(vals) // 6)
            smoothed = vals_series.rolling(window, min_periods=1, center=True).mean().values
            ax.plot(steps, smoothed,
                    color=cfg["color"], linewidth=2.0, alpha=0.9,
                    label=cfg["label"], zorder=3)

            # Annotate final value (use smoothed for cleaner annotation)
            final_step = steps[-1]
            final_val = smoothed[-1]
            # Offset: MLP on right, Text slightly below
            if approach_key == "esm3_mlp":
                offset = (8, 10)
            else:
                offset = (8, -18)
            ax.annotate(
                f"{final_val:.3f}",
                xy=(final_step, final_val),
                xytext=offset,
                textcoords="offset points",
                fontsize=plt.rcParams["font.size"] - 1,
                fontweight="bold",
                color=cfg["color"],
                arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=0.8),
                zorder=5,
            )

        ax.set_xlabel(get_label("step"))
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} Over Training Steps")
        ax.legend(loc="lower right")
        style.clean_axes(ax)

    fig.suptitle("Generation Quality During Training: BLEU & ROUGE-L",
                 fontsize=plt.rcParams["axes.titlesize"] + 2,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

ALL_FIGURES = list(range(1, 12))  # 1-11

def main():
    parser = argparse.ArgumentParser(
        description="Generate main publication figures (fig1-fig11)."
    )
    parser.add_argument("--figures", "-f", nargs="+", type=int,
                        default=ALL_FIGURES,
                        help="Figure numbers to generate (default: 1-11)")
    parser.add_argument("--targets", "-t", nargs="+",
                        default=["blog"],
                        choices=["blog", "web", "paper"],
                        help="Style targets (default: blog)")
    parser.add_argument("--data-dir", "-d", default=None,
                        help="Data directory (default: auto-detect latest in blog/data/)")
    args = parser.parse_args()

    from pathlib import Path
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Load data (needed for fig4, 5, 8)
    print("Loading data...")
    df_early, df_late = load_data(data_dir)
    print(f"  Early data: {len(df_early)} rows, Late data: {len(df_late)} rows")

    # Load metrics (needed for fig3, 6, 9)
    print("Loading metrics...")
    final_metrics, gen_metrics, dataset_comp = load_metrics(data_dir)

    for target in args.targets:
        print(f"\n{'=' * 60}")
        print(f"GENERATING MAIN FIGURES (target: {target})")
        print(f"{'=' * 60}")
        colors = style.apply(target)

        for fig_num in args.figures:
            print(f"\n--- Fig {fig_num} ---")

            try:
                if fig_num == 1:
                    fig = fig1_schematic_overview()
                elif fig_num == 2:
                    fig = fig2_architecture(colors)
                elif fig_num == 3:
                    fig = fig3_data_composition(colors, dataset_comp)
                elif fig_num == 4:
                    fig = fig4_main_run_progress(df_late, colors)
                elif fig_num == 5:
                    fig = fig5_eval_loss(df_late, colors)
                elif fig_num == 6:
                    fig = fig6_generation_quality(colors, gen_metrics)
                elif fig_num == 7:
                    fig = fig7_scaling_effect(colors)
                elif fig_num == 8:
                    fig = fig8_mlp_vs_text(df_late, colors)
                elif fig_num == 9:
                    fig = fig9_final_comparison(colors, final_metrics)
                elif fig_num == 10:
                    fig = fig10_base_vs_finetuned(colors)
                elif fig_num == 11:
                    fig = fig11_generation_over_steps(colors)
                else:
                    print(f"  Unknown figure number: {fig_num}")
                    continue

                if fig is None:
                    print(f"  Skipped fig{fig_num} (no data)")
                    continue

                # Figure name mapping
                fig_names = {
                    1: "fig1_schematic_overview",
                    2: "fig2_architecture",
                    3: "fig3_data_composition",
                    4: "fig4_main_run_progress",
                    5: "fig5_eval_loss",
                    6: "fig6_generation_quality",
                    7: "fig7_scaling_effect",
                    8: "fig8_mlp_vs_text",
                    9: "fig9_final_comparison",
                    10: "fig10_base_vs_finetuned",
                    11: "fig11_generation_over_steps",
                }
                name = fig_names[fig_num]

                # Save to main_figures/ and paper/figures/main/
                save_main_figure(fig, name)

            except Exception as e:
                print(f"  ERROR generating fig{fig_num}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("ALL MAIN FIGURES GENERATED")
    print(f"{'=' * 60}")
    print(f"Blog:  {MAIN_FIGURES_DIR}/fig*.png")
    print(f"Paper: {PAPER_MAIN_DIR}/fig*.pdf")


if __name__ == "__main__":
    main()
