# Figure Style Guide

> Canonical style conventions for all protein-LLM figures.
> Enforced by `scripts/analysis/figure_style.py`.

## Quick Start

```python
from figure_style import style, save_figure, get_color, get_label

style.apply("blog")                    # "blog" | "web" | "paper"
fig, ax = style.subplots()             # auto-sized for target
ax.plot(x, y, color=style.color("mlp"))
style.clean_axes(ax)
save_figure(fig, "my_plot")            # saves to supple_figures/
```

## Color Palette

### Approach Colors (canonical)

| Approach   | Blog/Internal   | Paper           | Web             |
|-----------|-----------------|-----------------|-----------------|
| Text       | `#808080` gray  | `#4d4d4d` dark gray | `#95a5a6` soft gray |
| MLP        | `#1f77b4` blue  | `#2166ac` deep blue  | `#4A90D9` soft blue |
| Perceiver  | `#ff7f0e` orange| `#b2182b` deep red   | `#F5A623` soft orange |
| Flamingo   | `#d62728` red   | `#762a83` deep purple| `#E74C3C` soft red |

### Status Colors

| Status    | Color     | Hex       |
|-----------|-----------|-----------|
| Confirmed | Green     | `#27AE60` |
| Pending   | Red       | `#E74C3C` |
| Running   | Amber     | `#F39C12` |

### Table Styling

- Header: `#2C3E50` background, white text, bold
- Row backgrounds by approach: Text `#E8E8E8`, MLP `#D4E6F1`, Perceiver `#FDE8D0`, Flamingo `#FADBD8`

## Font Settings

| Property         | Blog | Web  | Paper |
|-----------------|------|------|-------|
| Family          | sans-serif | sans-serif | serif (DejaVu Serif) |
| Base size       | 11   | 12   | 8     |
| Title size      | 13   | 15   | 9     |
| Axis label size | 12   | 13   | 8     |
| Legend size     | 9    | 10   | 7     |
| Tick size       | 10   | 11   | 7     |

## Figure Sizes

| Size    | Blog       | Web        | Paper           |
|---------|-----------|------------|-----------------|
| Default | 10 × 6    | 9 × 5.5   | 3.25 × 2.4     |
| Wide    | 14 × 6    | 12 × 5.5  | 6.75 × 2.8     |
| Tall    | 10 × 8    | 9 × 5.5   | 3.25 × 3.2     |

Paper sizes follow NeurIPS column widths (3.25" single, 6.75" full).

## DPI

| Target | DPI |
|--------|-----|
| Blog   | 150 |
| Web    | 200 |
| Paper  | 300 |

## Line & Marker Styles

| Property        | Blog | Web  | Paper |
|----------------|------|------|-------|
| Line width     | 1.5  | 1.8  | 1.0   |
| Marker size    | 5    | 5    | 3     |
| Axis linewidth | 1.0  | 0.8  | 0.5   |
| Grid linewidth | 0.5  | 0.5  | 0.3   |
| Grid alpha     | 0.5  | 0.3  | 0.4   |

- Marker cycle: `o`, `s`, `D`, `^`, `v`, `P`, `X`
- Linestyle cycle: `-`, `--`, `-.`, `:`

## Axes Conventions

- **Blog**: whitegrid, all spines visible
- **Web**: white background, top/right spines removed
- **Paper**: whitegrid, all spines (thin at 0.5pt)
- Use `style.clean_axes(ax)` to remove top/right spines on any target

## Metric Labels

Use `get_label(metric_key)` for axis labels:

| Key                | Pretty Name             |
|--------------------|-------------------------|
| `token_avg_loss`   | Token Average Loss      |
| `eval_loss`        | Eval Loss               |
| `grad_norm`        | Gradient Norm           |
| `learning_rate`    | Learning Rate           |
| `bleu`             | BLEU                    |
| `rouge_l`          | ROUGE-L                 |
| `step`             | Training Step           |
| `train_loss`       | Train Loss              |
| `best_eval_loss`   | Best Eval Loss          |
| `final_train_loss` | Final Train Loss        |
| `epoch`            | Epoch                   |
| `perplexity`       | Perplexity              |
| `accuracy`         | Accuracy                |
| `f1`               | F1 Score                |
| `mae`              | MAE                     |
| `plddt`            | pLDDT                   |

**Critical**: Always use `token_avg_loss` for training loss, never `loss`.

## Approach Labels

Use `get_approach_label(approach)` for legends:

| Key         | Pretty Name  |
|-------------|-------------|
| `text`      | Text-only   |
| `mlp`       | ESM3+MLP    |
| `perceiver` | Perceiver   |
| `flamingo`  | Flamingo    |

### Approach Aliases

Some data sources use variant names. These are auto-resolved:

| Alias        | Resolves to |
|-------------|-------------|
| `esm3`      | `mlp`       |
| `esm3_mlp`  | `mlp`       |
| `text_only` | `text`      |

## Save Conventions

### `save_figure(fig, name)`
- Default: saves to `blog/figures/supple_figures/` as PNG
- Paper target: saves both PDF and PNG to `paper/figures/supplementary/`

### `save_main_figure(fig, name)`
- Saves PNG to `blog/figures/main_figures/`
- Saves PDF to `paper/figures/main/`
- Always 300 DPI, white background

### File Naming
- Main figures: `fig{N}_{content}` (e.g., `fig4_main_run_progress`)
- Supplementary: `{prefix}_{content}` (e.g., `mlp_epoch1_train_loss`)
- No spaces, lowercase, underscores

## Directory Structure

```
blog/figures/
├── main_figures/           # 9 key figures (fig1–fig9)
├── supple_figures/         # All supplementary figures
└── figure_catalog.md       # Single source of truth

paper/figures/
├── main/                   # PDFs of fig1–fig9
└── supplementary/          # Supplementary PDFs + PNGs
```

## Helper Functions

| Function | Purpose |
|----------|---------|
| `style.apply(target)` | Activate rcParams for blog/web/paper |
| `style.color(approach)` | Get color for approach |
| `style.colors()` | Get full color dict |
| `style.subplots(kind)` | Create fig, ax with correct size |
| `style.clean_axes(ax)` | Remove top/right spines |
| `style.style_table(table, ...)` | Style a table consistently |
| `save_figure(fig, name)` | Save to correct dir with correct format |
| `save_main_figure(fig, name)` | Save main figure (PNG + PDF) |
| `get_color(approach)` | Module-level color getter |
| `get_label(metric)` | Pretty metric name |
| `get_approach_label(approach)` | Pretty approach name |
| `annotate_best(ax, x, y, val, color)` | Annotate best point with arrow |
| `annotate_bars(ax, bars, values)` | Label bar chart values |
| `shorten(text, max_len)` | Truncate text for legends |
| `smooth(series, window=50)` | Rolling mean smoothing (pandas Series or array) |
| `to_rgba_alpha(hex_color, alpha)` | Convert hex color to RGBA tuple with alpha |
| `add_watermark(fig, text="DRAFT")` | Add diagonal watermark to figure |
