# Figure Style Audit — 2026-03-09

## Overview

Audited all 9 Python scripts in the repo that generate matplotlib/seaborn figures.
Only **3 of 9 (33%)** use the centralized `scripts/analysis/figure_style.py` module.
The remaining 6 use inline hardcoded styles.

## Scripts Using figure_style.py (3)

| Script | Notes |
|--------|-------|
| `scripts/analysis/plot_training.py` | Full adoption: imports style, apply_style, save_figure, get_color |
| `scripts/analysis/plot_pub_figures.py` | Full adoption. Consolidated from blog/data/03-07/pub_figures.py |
| `scripts/analysis/plot_schematic_overview.py` | Uses SCHEMATIC_COLORS and other constants |

All three are in `scripts/analysis/` — the same directory as `figure_style.py`.

## Scripts NOT Using figure_style.py (6)

| Script | Style Method | Key Issue |
|--------|-------------|-----------|
| `blog/data/03-07/pub_figures.py` | Own apply_style() + save_figure() functions, 3 color palettes | Legacy predecessor of plot_pub_figures.py |
| `blog/data/03-06/analysis_plots.py` | Inline APPROACH_COLORS, FIG_DPI=150, sns.set_theme | text color mismatch |
| `blog/data/03-04/analysis_plots.py` | Inline APPROACH_COLORS, FIG_DPI=150, sns.set_theme | text color mismatch |
| `blog/data/03-04/draw_architecture.py` | Custom C dict for architecture diagrams, DPI=300 | Specialized diagram — partial justification |
| `blog/data/03-02/mlp_vs_text_analysis.py` | Inline APPROACH_COLORS, FIG_DPI=150, sns.set_theme | Earliest script, predates figure_style |
| `blog/data/03-02/three_way_analysis.py` | Inline APPROACH_COLORS, FIG_DPI=150, sns.set_theme | text color mismatch |

All 6 non-conforming scripts are in `blog/data/` — analysis scripts created before figure_style.py existed.

## Critical Color Mismatch: text approach

The most significant inconsistency is the "text" approach color:

| Source | text Color | Visual |
|--------|-----------|--------|
| `figure_style.py` (blog) | `#808080` | Gray |
| `figure_style.py` (paper) | `#4d4d4d` | Dark gray |
| `figure_style.py` (web) | `#95a5a6` | Soft gray |
| All 5 inline scripts | `#2ca02c` | Green |
| SCIENTIST_TEAM.md analyst spec | `#2ca02c` | Green |

The figure_style module deliberately changed text from green to gray (to position it as a "baseline" visually), but the inline scripts and agent specs still use the old green.

MLP (blue `#1f77b4`) and Perceiver (orange `#ff7f0e`) are consistent everywhere.

Flamingo (red `#d62728`) exists only in figure_style.py — no inline script defines it.

## SCIENTIST_TEAM.md Agent Spec Alignment: NOT ALIGNED

The analyst agent specification (SCIENTIST_TEAM.md lines 165-176) defines inline style:

```python
sns.set_theme(style="whitegrid", palette="colorblind")
APPROACH_COLORS = {"mlp": "#1f77b4", "perceiver": "#ff7f0e", "text": "#2ca02c"}
FIG_DPI = 150
FIG_SIZE = (10, 6)
```

Issues:
1. Does not reference `figure_style.py` at all
2. Uses old text color `#2ca02c` (green) instead of `#808080` (gray)
3. Does not include flamingo color
4. Instructs agents to hardcode style inline rather than import from the shared module
5. The spawn prompt (lines 224-260) also uses inline style instructions

This means every time the analyst agent is spawned, it will use the old inline style rather than figure_style.py.

## Recommendations

1. **Update SCIENTIST_TEAM.md** analyst spec to reference `figure_style.py` instead of inline colors.
   Replace the inline style block with:
   ```python
   import sys
   sys.path.insert(0, "scripts/analysis")
   from figure_style import style, save_figure, get_color
   style.apply("blog")
   ```

2. **Legacy blog/data/ scripts** (03-02, 03-04, 03-06) are historical artifacts.
   They generated figures for past reports and should not be modified.
   New scripts should import from figure_style.py.

3. **blog/data/03-07/pub_figures.py** is superseded by `scripts/analysis/plot_pub_figures.py`.
   Mark as legacy or add a comment pointing to the consolidated version.

4. **blog/data/03-04/draw_architecture.py** could use `SCHEMATIC_COLORS` from figure_style.py
   for consistency, but its custom palette is architecture-specific and mostly non-overlapping.

5. **Decide on text color**: The change from green (#2ca02c) to gray (#808080) was intentional
   in figure_style.py but never propagated to the agent spec or legacy scripts. This should
   be explicitly decided and documented.
