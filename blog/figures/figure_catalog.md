# Figure Catalog

> **Single source of truth** for all project figures. Updated 2026-03-09.

## Directory Structure

```
blog/figures/
├── main_figures/           # 11 key figures for paper + website
├── supple_figures/         # 50 supplementary blog figures
└── figure_catalog.md       # This file

paper/figures/
├── main/                   # 11 PDFs (NeurIPS-compatible)
└── supplementary/          # 30 PDFs + PNGs
```

## Main Figures (11)

These are the core figures used in the paper and website blog. Each tells a key part of the project narrative.

| Fig | Filename | Category | Content | Why Main | Source Script | Date | Status |
|-----|----------|----------|---------|----------|---------------|------|--------|
| 1 | `fig1_schematic_overview.png` | overview | 4-panel project overview (architecture, data, training, results) | Core narrative figure | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 2 | `fig2_architecture.png` | architecture | ESM-3 + MLP projector architecture diagram | Explains model design | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 3 | `fig3_data_composition.png` | data | Dataset breakdown (4.89M samples, 6 sources, 4 categories) | Training data overview | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 4 | `fig4_main_run_progress.png` | training | Train + eval loss curves (MLP 4.89M, colored markers) | Key result: training trajectory | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 5 | `fig5_eval_loss.png` | evaluation | Eval loss with best-point annotation (0.361 at step 9750) | Key result: model quality | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 6 | `fig6_generation_quality.png` | generation | BLEU & ROUGE-L over steps 3000–9750 | Generation quality trend | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 7 | `fig7_scaling_effect.png` | scaling | 50K → 4.89M: 81% eval_loss improvement bar chart | Key finding: data scaling | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 8 | `fig8_mlp_vs_text.png` | comparison | MLP vs Text-only on same 4.89M dataset (step 200) | Key comparison: ESM-3 advantage | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 9 | `fig9_final_comparison.png` | comparison | Cross-approach bar chart (4 configs, annotated eval_loss) | Summary comparison | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 10 | `fig10_base_vs_finetuned.png` | comparison | Base model vs SFT: perplexity, BLEU, ROUGE-L (126x BLEU improvement) | Key result: SFT impact | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 11 | `fig11_generation_over_steps.png` | generation | BLEU & ROUGE-L progression over training steps (MLP vs Text) | Generation quality trend over time | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |

### Paper PDF Counterparts

| Fig | Blog PNG | Paper PDF |
|-----|----------|-----------|
| 1 | `main_figures/fig1_schematic_overview.png` | `paper/figures/main/fig1_schematic_overview.pdf` |
| 2 | `main_figures/fig2_architecture.png` | `paper/figures/main/fig2_architecture.pdf` |
| 3 | `main_figures/fig3_data_composition.png` | `paper/figures/main/fig3_data_composition.pdf` |
| 4 | `main_figures/fig4_main_run_progress.png` | `paper/figures/main/fig4_main_run_progress.pdf` |
| 5 | `main_figures/fig5_eval_loss.png` | `paper/figures/main/fig5_eval_loss.pdf` |
| 6 | `main_figures/fig6_generation_quality.png` | `paper/figures/main/fig6_generation_quality.pdf` |
| 7 | `main_figures/fig7_scaling_effect.png` | `paper/figures/main/fig7_scaling_effect.pdf` |
| 8 | `main_figures/fig8_mlp_vs_text.png` | `paper/figures/main/fig8_mlp_vs_text.pdf` |
| 9 | `main_figures/fig9_final_comparison.png` | `paper/figures/main/fig9_final_comparison.pdf` |
| 10 | `main_figures/fig10_base_vs_finetuned.png` | `paper/figures/main/fig10_base_vs_finetuned.pdf` |
| 11 | `main_figures/fig11_generation_over_steps.png` | `paper/figures/main/fig11_generation_over_steps.pdf` |

---

## Supplementary Figures — Blog (50)

### Early Experiments (50K data, Feb 2026)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `protein_seq_length_histogram.png` | Protein sequence length distribution | 2026-02-20 | Data exploration |
| `mlp_vs_text_training_loss.png` | MLP vs Text-only train loss (50K) | 2026-03-02 | Early comparison |
| `mlp_vs_text_eval_loss.png` | MLP vs Text-only eval loss (50K) | 2026-03-02 | Early comparison |
| `mlp_vs_text_grad_norms.png` | MLP vs Text-only gradient norms (50K) | 2026-03-02 | Early comparison |
| `mlp_vs_text_lr_schedule.png` | MLP vs Text-only LR schedule (50K) | 2026-03-02 | Early comparison |
| `mlp_vs_text_final_metrics.png` | MLP vs Text-only final bar chart (50K) | 2026-03-02 | Early comparison |

### Three-Way Comparison (MLP vs Perceiver vs Text, 50K)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `three_way_loss_curves.png` | Overlaid train loss (3 approaches) | 2026-03-02 | 50K comparison |
| `three_way_eval_loss.png` | Eval loss (3 approaches) | 2026-03-02 | 50K comparison |
| `three_way_grad_norms.png` | Gradient norms (3 approaches) | 2026-03-02 | 50K comparison |
| `three_way_lr_schedule.png` | LR schedule (3 approaches) | 2026-03-02 | 50K comparison |
| `three_way_final_metrics.png` | Final metrics bar chart (3 approaches) | 2026-03-02 | 50K comparison |
| `three_way_convergence_table.png` | Convergence summary table (3 approaches) | 2026-03-02 | 50K comparison |

### Extended MLP Training (combined data, Mar 4-6)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `long_mlp_loss_curve.png` | Extended MLP training loss | 2026-03-04 | Mol-instructions long run |
| `long_mlp_gradient_norms.png` | Extended MLP grad norms | 2026-03-04 | Mol-instructions long run |
| `long_mlp_lr_schedule.png` | Extended MLP LR schedule | 2026-03-04 | Mol-instructions long run |
| `text_vs_esm_eval_loss.png` | Text vs ESM eval loss overlay | 2026-03-04 | Cross-approach |
| `text_vs_esm_final_comparison.png` | Text vs ESM final bar chart | 2026-03-04 | Cross-approach |
| `approach_convergence_comparison.png` | Convergence comparison across approaches | 2026-03-04 | Cross-approach |

### MLP Combined Run Detail (4.89M, step 7090)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `loss_curves.png` | Loss curves (combined run overlay) | 2026-03-06 | Combined dataset |
| `convergence_table.png` | Convergence summary (combined) | 2026-03-06 | Combined dataset |
| `architecture_comparison.png` | Architecture diagram (4 approaches) | 2026-03-06 | Architecture overview |
| `mlp_combined_train_loss.png` | Combined run training loss | 2026-03-06 | Step 7090 snapshot |
| `mlp_combined_eval_loss.png` | Combined run eval loss | 2026-03-06 | Step 7090 snapshot |
| `mlp_combined_grad_norms.png` | Combined run gradient norms | 2026-03-06 | Step 7090 snapshot |
| `mlp_combined_lr_schedule.png` | Combined run LR schedule | 2026-03-06 | Step 7090 snapshot |
| `mlp_combined_final_comparison.png` | Combined run final metrics | 2026-03-06 | Step 7090 snapshot |
| `mlp_combined_convergence_table.png` | Combined run convergence table | 2026-03-06 | Step 7090 snapshot |
| `main_run_loss_detail.png` | Main run loss with detail view | 2026-03-06 | Zoomed view |
| `main_run_smoothed_loss.png` | Main run smoothed loss (MA-50) | 2026-03-06 | Smoothed view |

### MLP Epoch 1 Deep-Dive (4.89M, step 9750)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `mlp_epoch1_train_loss.png` | Smoothed training loss (MA-50), 0–9750 | 2026-03-07 | Epoch 1 analysis |
| `mlp_epoch1_train_eval_combined.png` | Train + eval overlaid | 2026-03-07 | Epoch 1 analysis |
| `mlp_epoch1_lr_schedule.png` | Warmup + cosine decay schedule | 2026-03-07 | Epoch 1 analysis |
| `mlp_epoch1_grad_norms.png` | Log-scale grad norms with spike detection | 2026-03-07 | Epoch 1 analysis |
| `mlp_epoch1_comparison_bar.png` | Final metrics bar chart | 2026-03-07 | Epoch 1 analysis |
| `mlp_epoch1_convergence_table.png` | Summary convergence table | 2026-03-07 | Epoch 1 analysis |

### Publication-Quality Variants (pub_*)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `pub_approach_loss_curves.png` | Approach comparison train loss | 2026-03-07 | Publication style |
| `pub_approach_eval_curves.png` | Approach comparison eval loss | 2026-03-07 | Publication style |
| `pub_convergence_table.png` | Convergence table (styled) | 2026-03-07 | Publication style |
| `pub_generation_quality.png` | BLEU/ROUGE-L by task type | 2026-03-07 | Publication style |
| `pub_gradient_norms.png` | Gradient norms (log scale, styled) | 2026-03-07 | Publication style |

### Website Variants (web_*)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `web_approach_loss_curves.png` | Approach comparison train loss | 2026-03-07 | Website style |
| `web_approach_eval_curves.png` | Approach comparison eval loss | 2026-03-07 | Website style |
| `web_architecture.png` | Architecture diagram | 2026-03-07 | Website style |
| `web_data_composition.png` | Dataset composition | 2026-03-07 | Website style |
| `web_final_comparison.png` | Final comparison bar chart | 2026-03-07 | Website style |
| `web_generation_quality.png` | Generation quality metrics | 2026-03-07 | Website style |
| `web_main_run_progress.png` | Main run training progress | 2026-03-07 | Website style |
| `web_mlp_vs_text_4m.png` | MLP vs Text on 4.89M | 2026-03-07 | Website style |
| `web_scaling_effect.png` | Scaling effect visualization | 2026-03-07 | Website style |

### Other

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `architecture_comparison.pdf` | Architecture diagram (PDF) | 2026-03-06 | PDF variant in supple |

---

## Supplementary Figures — Paper (30)

Paper supplementary figures live at `paper/figures/supplementary/`. Mix of PDFs (high-quality plots from `plot_training.py`) and PNGs (from blog analysis scripts).

### PDFs (11)

| Filename | Content |
|----------|---------|
| `convergence_table.pdf` | Convergence summary table |
| `eval_curves.pdf` | Multi-approach eval loss curves |
| `generation_quality.pdf` | BLEU/ROUGE-L by task type |
| `gradient_norms.pdf` | Gradient norm trajectories |
| `loss_curves.pdf` | Multi-approach training loss |
| `mlp_epoch1_comparison_bar.pdf` | Epoch 1 metrics bar chart |
| `mlp_epoch1_convergence_table.pdf` | Epoch 1 convergence table |
| `mlp_epoch1_grad_norms.pdf` | Epoch 1 gradient norms |
| `mlp_epoch1_lr_schedule.pdf` | Epoch 1 LR schedule |
| `mlp_epoch1_train_eval_combined.pdf` | Epoch 1 train + eval overlaid |
| `mlp_epoch1_train_loss.pdf` | Epoch 1 training loss |

### PNGs (19)

| Filename | Content |
|----------|---------|
| `architecture_comparison.png` | 4-approach architecture diagram |
| `architecture.png` | Single-approach architecture |
| `convergence_table.png` | Convergence table (PNG) |
| `data_composition.png` | Dataset breakdown |
| `eval_curves.png` | Eval loss curves |
| `final_comparison.png` | Cross-approach bar chart |
| `generation_quality.png` | Generation quality metrics |
| `gradient_norms.png` | Gradient norms |
| `loss_curves.png` | Training loss curves |
| `main_run.png` | Main run progress |
| `mlp_combined_convergence_table.png` | Combined run convergence |
| `mlp_combined_eval_loss.png` | Combined run eval loss |
| `mlp_combined_final_comparison.png` | Combined run final metrics |
| `mlp_combined_grad_norms.png` | Combined run grad norms |
| `mlp_combined_lr_schedule.png` | Combined run LR schedule |
| `mlp_combined_train_loss.png` | Combined run train loss |
| `mlp_vs_text_4m.png` | MLP vs Text on 4.89M |
| `protein_seq_length_histogram.png` | Sequence length distribution |
| `scaling_effect.png` | Scaling effect visualization |

---

## Generation Scripts

All scripts use `scripts/analysis/figure_style.py` for consistent styling. See `scripts/analysis/STYLE_GUIDE.md`.

| Script | Figures Generated | Notes |
|--------|-------------------|-------|
| `scripts/analysis/plot_pub_figures.py` | `fig1`–`fig11` (main figures) | Consolidated from `blog/data/03-07/pub_figures.py` (archived). CLI: `--figures 4 5 6 10 11` |
| `scripts/analysis/plot_training.py` | Diagnostic plots (loss, eval, grad norms, etc.) | Reusable: `--experiments`, `--output`, `--paper-output`, `--prefix`, `--plots` |
| `scripts/analysis/plot_schematic_overview.py` | `fig1_schematic_overview` | 4-panel diagram, called by plot_pub_figures.py |
| `scripts/analysis/figure_style.py` | (style module) | Colors, fonts, sizes, save helpers. Import, don't run. |

### Archived Scripts (consolidated 2026-03-09)

| Script | Status | Replacement |
|--------|--------|-------------|
| `blog/data/03-07/pub_figures.py` | ARCHIVED | `scripts/analysis/plot_pub_figures.py` |
| `blog/data/03-04/analysis_plots.py` | ARCHIVED | `scripts/analysis/plot_training.py` |
| `blog/data/03-06/analysis_plots.py` | ARCHIVED | `scripts/analysis/plot_training.py` |

---

## Conventions

- **Style module**: `scripts/analysis/figure_style.py` — single source of truth for all styling
- **Blog PNGs**: 150 DPI, `(10, 6)` default size, seaborn whitegrid style
- **Paper PDFs**: 300 DPI, vector format, NeurIPS column widths (3.25" / 6.75")
- **Color scheme**: Text `#808080` (gray), MLP `#1f77b4` (blue), Perceiver `#ff7f0e` (orange), Flamingo `#d62728` (red)
- **Main figure naming**: `fig{N}_{content}.{ext}` — canonical numbered names (fig1–fig11)
- **Supplementary naming**: `{prefix}_{content}.{ext}` — prefix groups related figures
- **References from blog posts**: `../figures/main_figures/name.png` or `../figures/supple_figures/name.png`
- **References from paper**: `figures/main/name.pdf` or `figures/supplementary/name.pdf`
