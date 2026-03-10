#!/usr/bin/env python3
"""
Shared figure style module for all protein-LLM plots.

Enforces visual consistency across blog, website, and paper figures.
Import and call `apply_style(target)` once before creating any figures.

Usage:
    from figure_style import style, save_figure

    style.apply("blog")           # or "web", "paper"
    fig, ax = style.subplots()    # uses default figsize for target
    ax.plot(x, y, color=style.color("mlp"))
    style.clean_axes(ax)
    save_figure(fig, "my_plot", target="blog")
"""

import matplotlib

matplotlib.use("Agg")  # Headless rendering — MUST be before pyplot import

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root (scripts/analysis/ -> root)
BLOG_FIGURES_DIR = BASE_DIR / "blog" / "figures"
MAIN_FIGURES_DIR = BLOG_FIGURES_DIR / "main_figures"
SUPPLE_FIGURES_DIR = BLOG_FIGURES_DIR / "supple_figures"
PAPER_MAIN_DIR = BASE_DIR / "paper" / "figures" / "main"
PAPER_SUPPLE_DIR = BASE_DIR / "paper" / "figures" / "supplementary"
DATA_DIR = BASE_DIR / "blog" / "data"
RESULTS_DIR = BASE_DIR / "results"


# ═══════════════════════════════════════════════════════════════════════
# COLOR PALETTES
# ═══════════════════════════════════════════════════════════════════════

# Approach name aliases — normalize before color lookup
APPROACH_ALIASES = {
    "esm3": "mlp",       # lineage.json stores "esm3" for MLP approach
    "esm3_mlp": "mlp",
    "text_only": "text",
}

# Canonical approach colors — used across all targets unless overridden
APPROACH_COLORS = {
    "text":      "#808080",   # Gray
    "mlp":       "#1f77b4",   # Blue
    "perceiver": "#ff7f0e",   # Orange
    "flamingo":  "#d62728",   # Red
    "unknown":   "#7f7f7f",   # Fallback gray
}

# Paper variant — higher contrast, grayscale-safe
PAPER_COLORS = {
    "text":      "#4d4d4d",   # Dark gray
    "mlp":       "#2166ac",   # Deep blue
    "perceiver": "#b2182b",   # Deep red
    "flamingo":  "#762a83",   # Deep purple
    "unknown":   "#7f7f7f",
}

# Website variant — softer, modern
WEB_COLORS = {
    "text":      "#95a5a6",   # Soft gray
    "mlp":       "#4A90D9",   # Soft blue
    "perceiver": "#F5A623",   # Soft orange
    "flamingo":  "#E74C3C",   # Soft red
    "unknown":   "#bdc3c7",
}

# Status colors (for result cards, tables)
STATUS_COLORS = {
    "confirmed": "#27AE60",   # Green
    "pending":   "#E74C3C",   # Red
    "running":   "#F39C12",   # Amber
}

# Table styling
TABLE_HEADER_BG = "#2C3E50"
TABLE_HEADER_FG = "white"
TABLE_ROW_COLORS = {
    "text":      "#E8E8E8",   # Light gray
    "mlp":       "#D4E6F1",   # Light blue
    "perceiver": "#FDE8D0",   # Light orange
    "flamingo":  "#FADBD8",   # Light red
}

# Schematic / diagram palette
SCHEMATIC_COLORS = {
    "protein_bg":     "#DCEEFB",
    "protein_border": "#2C7BB6",
    "protein_text":   "#1A5276",
    "esm3_bg":        "#B3D4F0",
    "esm3_border":    "#2166AC",
    "llm_bg":         "#D5F5E3",
    "llm_border":     "#27AE60",
    "llm_text":       "#196F3D",
    "proj_mlp_bg":    "#D6EAF8",
    "proj_mlp_border":"#2E86C1",
    "proj_perc_bg":   "#FDEBD0",
    "proj_perc_border":"#E67E22",
    "proj_flam_bg":   "#FADBD8",
    "proj_flam_border":"#C0392B",
    "proj_text_bg":   "#E8F8F5",
    "proj_text_border":"#1ABC9C",
    "train_bg":       "#FEF3E2",
    "train_border":   "#E67E22",
    "train_text":     "#935116",
    "sft_bg":         "#FDE8D0",
    "grpo_bg":        "#F9E4B7",
    "arrow":          "#566573",
    "arrow_flow":     "#2C3E50",
    "panel_border":   "#AEB6BF",
    "panel_title_bg": "#2C3E50",
}


# ═══════════════════════════════════════════════════════════════════════
# FIGURE SIZES (width, height in inches)
# ═══════════════════════════════════════════════════════════════════════

# Blog / internal
BLOG_SIZE = (10, 6)
BLOG_WIDE = (14, 6)
BLOG_TALL = (10, 8)

# Website
WEB_SIZE = (9, 5.5)
WEB_WIDE = (12, 5.5)

# Paper (NeurIPS column widths)
PAPER_COL = (3.25, 2.4)       # Single column
PAPER_WIDE = (6.75, 2.8)      # Full width
PAPER_TALL = (3.25, 3.2)      # Single column tall

# DPI per target
DPI = {
    "blog":  150,
    "web":   200,
    "paper": 300,
}

# Default figsize per target
DEFAULT_FIGSIZE = {
    "blog":  BLOG_SIZE,
    "web":   WEB_SIZE,
    "paper": PAPER_COL,
}


# ═══════════════════════════════════════════════════════════════════════
# FONT SETTINGS
# ═══════════════════════════════════════════════════════════════════════

FONT_SETTINGS = {
    "blog": {
        "font.family":     "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size":       11,
        "axes.titlesize":  13,
        "axes.labelsize":  12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    },
    "web": {
        "font.family":     "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size":       12,
        "axes.titlesize":  15,
        "axes.labelsize":  13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    },
    "paper": {
        "font.family":     "serif",
        "font.serif":      ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size":       8,
        "axes.titlesize":  9,
        "axes.labelsize":  8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# LINE / MARKER STYLES
# ═══════════════════════════════════════════════════════════════════════

LINE_SETTINGS = {
    "blog": {
        "lines.linewidth":  1.5,
        "lines.markersize": 5,
        "axes.linewidth":   1.0,
        "grid.linewidth":   0.5,
        "grid.alpha":       0.5,
    },
    "web": {
        "lines.linewidth":  1.8,
        "lines.markersize": 5,
        "axes.linewidth":   0.8,
        "grid.linewidth":   0.5,
        "grid.alpha":       0.3,
    },
    "paper": {
        "lines.linewidth":  1.0,
        "lines.markersize": 3,
        "axes.linewidth":   0.5,
        "grid.linewidth":   0.3,
        "grid.alpha":       0.4,
    },
}

# Marker cycle for multi-series plots
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

# Linestyle cycle for secondary series (e.g., ROUGE-L vs BLEU)
LINESTYLES = ["-", "--", "-.", ":"]


# ═══════════════════════════════════════════════════════════════════════
# METRIC LABELS — pretty names for raw metric keys
# ═══════════════════════════════════════════════════════════════════════

METRIC_LABELS = {
    "token_avg_loss":    "Token Average Loss",
    "loss":              "Loss (Running Avg)",
    "eval_loss":         "Eval Loss",
    "grad_norm":         "Gradient Norm",
    "learning_rate":     "Learning Rate",
    "bleu":              "BLEU",
    "rouge_l":           "ROUGE-L",
    "train_loss":        "Train Loss",
    "best_eval_loss":    "Best Eval Loss",
    "final_train_loss":  "Final Train Loss",
    "step":              "Training Step",
    "epoch":             "Epoch",
    "total_tokens":      "Total Tokens",
    "gpu_memory_allocated_gb": "GPU Memory Allocated (GB)",
    "gpu_memory_reserved_gb":  "GPU Memory Reserved (GB)",
    "gpu_memory_max_allocated_gb": "Peak GPU Memory (GB)",
    # Downstream evaluation metrics
    "go_prediction_accuracy": "GO Prediction Accuracy",
    "stability_mae": "Stability MAE (ddG)",
    "structure_quality": "Structure Quality (pLDDT)",
    "perplexity": "Perplexity",
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "mae": "MAE",
    "plddt": "pLDDT",
}

# Approach pretty names for legends
APPROACH_LABELS = {
    "text":      "Text-only",
    "mlp":       "ESM3+MLP",
    "perceiver": "Perceiver",
    "flamingo":  "Flamingo",
    "unknown":   "Unknown",
}


# ═══════════════════════════════════════════════════════════════════════
# STYLE CLASS — main interface
# ═══════════════════════════════════════════════════════════════════════

class FigureStyle:
    """Central style manager. Call `apply(target)` once per script."""

    def __init__(self):
        self._target = None
        self._colors = APPROACH_COLORS

    @property
    def target(self):
        return self._target

    def apply(self, target="blog"):
        """Activate rcParams preset for the given target.

        Args:
            target: One of "blog", "web", "paper".
        """
        if target not in ("blog", "web", "paper"):
            raise ValueError(f"Unknown target: {target!r}. Use 'blog', 'web', or 'paper'.")

        self._target = target

        # Pick color palette
        if target == "paper":
            self._colors = PAPER_COLORS
        elif target == "web":
            self._colors = WEB_COLORS
        else:
            self._colors = APPROACH_COLORS

        # Seaborn theme
        if target == "web":
            sns.set_theme(style="white", palette="muted")
        else:
            sns.set_theme(style="whitegrid", palette="colorblind")

        # Merge all rcParams
        rc = {}
        rc["figure.dpi"] = DPI[target]
        rc["savefig.dpi"] = DPI[target]
        rc["savefig.bbox"] = "tight"
        rc["savefig.pad_inches"] = 0.02 if target == "paper" else 0.1
        rc.update(FONT_SETTINGS[target])
        rc.update(LINE_SETTINGS[target])

        # Web: remove top/right spines
        if target == "web":
            rc["axes.spines.top"] = False
            rc["axes.spines.right"] = False

        plt.rcParams.update(rc)
        return self._colors

    def color(self, approach):
        """Get the color for an approach, respecting active target palette.

        Handles aliases (e.g., "esm3" -> "mlp").
        """
        normalized = APPROACH_ALIASES.get(approach, approach)
        return self._colors.get(normalized, self._colors.get("unknown", "#333333"))

    def colors(self):
        """Return the full active color dict."""
        return dict(self._colors)

    def figsize(self, kind="default"):
        """Get figure size for the active target.

        Args:
            kind: "default", "wide", "tall", "col", or a tuple.
        """
        if isinstance(kind, tuple):
            return kind

        t = self._target or "blog"
        if kind == "default":
            return DEFAULT_FIGSIZE[t]
        elif kind == "wide":
            return {"blog": BLOG_WIDE, "web": WEB_WIDE, "paper": PAPER_WIDE}[t]
        elif kind == "tall":
            return {"blog": BLOG_TALL, "web": WEB_SIZE, "paper": PAPER_TALL}[t]
        elif kind == "col":
            return {"blog": BLOG_SIZE, "web": WEB_SIZE, "paper": PAPER_COL}[t]
        return DEFAULT_FIGSIZE.get(t, BLOG_SIZE)

    def dpi(self):
        """Get DPI for the active target."""
        return DPI.get(self._target, 150)

    def fmt(self):
        """Default save format for the active target."""
        return "pdf" if self._target == "paper" else "png"

    def subplots(self, kind="default", **kwargs):
        """Create fig, ax with correct figsize for active target.

        Args:
            kind: figsize specifier (see `figsize()`).
            **kwargs: passed to plt.subplots.
        """
        if "figsize" not in kwargs:
            kwargs["figsize"] = self.figsize(kind)
        return plt.subplots(**kwargs)

    def clean_axes(self, ax, remove_top=True, remove_right=True):
        """Remove top and right spines, tighten grid."""
        if remove_top:
            ax.spines["top"].set_visible(False)
        if remove_right:
            ax.spines["right"].set_visible(False)

    def style_table(self, table, approach_order=None, n_rows=0, n_cols=0):
        """Apply consistent styling to a matplotlib table.

        Args:
            table: matplotlib Table object.
            approach_order: list of approach names per data row (for row coloring).
            n_rows: number of data rows.
            n_cols: number of columns.
        """
        table.auto_set_font_size(False)
        table.set_fontsize(plt.rcParams.get("font.size", 10) - 1)
        table.scale(1.2, 1.6)

        # Style header
        for j in range(n_cols):
            table[0, j].set_facecolor(TABLE_HEADER_BG)
            table[0, j].set_text_props(color=TABLE_HEADER_FG, fontweight="bold")

        # Color rows by approach (resolve aliases)
        if approach_order:
            for i, approach in enumerate(approach_order):
                normalized = APPROACH_ALIASES.get(approach, approach)
                bg = TABLE_ROW_COLORS.get(normalized, "#FFFFFF")
                for j in range(n_cols):
                    table[i + 1, j].set_facecolor(bg)


# Module-level singleton
style = FigureStyle()


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_color(approach):
    """Get approach color from the active palette."""
    return style.color(approach)


def get_label(metric):
    """Get pretty label for a metric key."""
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def get_approach_label(approach):
    """Get pretty label for an approach name. Handles aliases."""
    normalized = APPROACH_ALIASES.get(approach, approach)
    return APPROACH_LABELS.get(normalized, approach.title())


def apply_style(target="blog"):
    """Apply style preset. Convenience wrapper around style.apply()."""
    return style.apply(target)


def save_figure(fig, name, target=None, subdir=None, also_pdf=False,
                close=True):
    """Save figure to the correct directory with correct format and DPI.

    Args:
        fig: matplotlib Figure.
        name: filename stem (no extension).
        target: "blog", "web", "paper", or None (uses active target).
        subdir: subdirectory override. Default: supple_figures/ for blog/web,
                main/ for paper.
        also_pdf: if True, also save PDF (for blog/web targets).
        close: close figure after saving.

    Returns:
        list of saved file paths.
    """
    target = target or style.target or "blog"
    saved = []

    if target == "paper":
        out_dir = Path(subdir) if subdir else PAPER_SUPPLE_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in ("pdf", "png"):
            path = out_dir / f"{name}.{fmt}"
            fig.savefig(str(path), format=fmt, dpi=DPI["paper"],
                        bbox_inches="tight")
            saved.append(str(path))
            print(f"  OK {path}")
    else:
        out_dir = Path(subdir) if subdir else SUPPLE_FIGURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.png"
        fig.savefig(str(path), dpi=DPI.get(target, 150), bbox_inches="tight")
        saved.append(str(path))
        print(f"  OK {path}")

        if also_pdf:
            pdf_path = out_dir / f"{name}.pdf"
            fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight")
            saved.append(str(pdf_path))
            print(f"  OK {pdf_path}")

    if close:
        plt.close(fig)

    return saved


def save_main_figure(fig, name, close=True):
    """Save a main figure to both blog/main_figures/ and paper/figures/main/.

    Args:
        fig: matplotlib Figure.
        name: filename stem (e.g., "fig1_schematic_overview").
        close: close figure after saving.

    Returns:
        list of saved file paths.
    """
    saved = []

    # Blog PNG (high DPI for quality)
    MAIN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = MAIN_FIGURES_DIR / f"{name}.png"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    saved.append(str(png_path))
    print(f"  OK {png_path}")

    # Paper PDF
    PAPER_MAIN_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PAPER_MAIN_DIR / f"{name}.pdf"
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    saved.append(str(pdf_path))
    print(f"  OK {pdf_path}")

    if close:
        plt.close(fig)

    return saved


def shorten(text, max_len=30):
    """Shorten text for legends."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def annotate_best(ax, x, y, value, color, offset=(10, -15)):
    """Annotate a point with its value and an arrow."""
    ax.annotate(
        f"{value:.3f}",
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=plt.rcParams["font.size"] - 2,
        color=color,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )


def annotate_bars(ax, bars, values, fmt="{:.3f}", offset=0.02, **text_kwargs):
    """Annotate bar chart with values above each bar."""
    defaults = dict(
        ha="center", va="bottom",
        fontsize=plt.rcParams["font.size"] - 1,
        fontweight="bold",
    )
    defaults.update(text_kwargs)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                fmt.format(val),
                **defaults,
            )


def smooth(series, window=50):
    """Apply rolling mean smoothing to a pandas Series.

    Args:
        series: pandas Series of values (e.g., loss, grad_norm).
        window: rolling window size (default 50).

    Returns:
        pandas Series with smoothed values (min_periods=1 for no leading NaNs).
    """
    import pandas as pd
    if isinstance(series, pd.Series):
        return series.rolling(window, min_periods=1).mean()
    # Numpy array fallback
    s = pd.Series(series)
    return s.rolling(window, min_periods=1).mean().values


def to_rgba_alpha(hex_color, alpha=0.6):
    """Convert a hex color to RGBA tuple with specified alpha.

    Args:
        hex_color: hex color string (e.g., "#1f77b4").
        alpha: alpha value 0-1.

    Returns:
        RGBA tuple (r, g, b, a).
    """
    import matplotlib.colors as mcolors
    return mcolors.to_rgba(hex_color, alpha=alpha)


def add_watermark(fig, text="DRAFT", alpha=0.08, fontsize=60, rotation=30):
    """Add a diagonal watermark to a figure.

    Args:
        fig: matplotlib Figure.
        text: watermark text.
        alpha: transparency (0=invisible, 1=opaque).
        fontsize: watermark font size.
        rotation: text rotation in degrees.
    """
    fig.text(0.5, 0.5, text,
             ha="center", va="center",
             fontsize=fontsize, color="gray",
             alpha=alpha, rotation=rotation,
             transform=fig.transFigure,
             fontweight="bold", zorder=0)
