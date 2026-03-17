# Figure Catalog

> **Single source of truth** for all project figures. Updated 2026-03-13.

## Directory Structure

```
blog/figures/
├── main_figures/           # 11 key figures for paper + website
├── supple_figures/         # 71 supplementary blog figures
└── figure_catalog.md       # This file

paper/figures/
├── main/                   # 11 PDFs (NeurIPS-compatible)
└── supplementary/          # 30 PDFs + PNGs
```

## Main Figures (11)

These are the core figures used in the paper and website blog. Each tells a key part of the project narrative.

| Fig | Filename | Category | Content | Why Main | Source Script | Date | Status |
|-----|----------|----------|---------|----------|---------------|------|--------|
| 1 | `fig1_schematic_overview.svg` | overview | Hand-coded SVG: 2-panel (A) Four pathways (Text/MLP/Perceiver/Flamingo) with ESM-3 encoder, fork-point routing, LLM layer bars with cross-attn markers; (B) Training pipeline (SFT 4 categories 4.89M + GRPO 3 reward tasks 30K). Research question banner. Mermaid-like flat style, 1400x720. Also: `fig1_schematic_overview.png` (raster). Separate panels: `fig1a_pathways.png`, `fig1b_training_pipeline.png` | Core narrative figure | hand-coded SVG | 2026-03-13 | confirmed |
| 1a | `fig1a_pathways.png` | overview | Panel A standalone: Four pathways diagram with protein input, ESM-3 encoder, 3 projector variants, LLM. Pastel fills, frozen/trainable icons | Standalone panel for flexible layout | `scripts/analysis/plot_schematic_overview.py` | 2026-03-11 | confirmed |
| 1b | `fig1b_training_pipeline.png` | overview | Panel B standalone: SFT (4 categories with example prompts, 4.89M samples) + GRPO (3 reward tasks with descriptions, 30K samples) | Standalone panel for flexible layout | `scripts/analysis/plot_schematic_overview.py` | 2026-03-11 | confirmed |
| 2 | `fig2_architecture.svg` | architecture | Hand-coded SVG: 4-column comparison (Text-Only, ESM-3+MLP, ESM-3+Perceiver, ESM-3+Flamingo). Bottom-to-top flow: protein seq, encoder, pooling/projector, LLM, text output. Mermaid-like flat style with frozen/trainable icons, token bars, dashed annotation boxes. Flamingo column shows frozen LLM + gated cross-attn. 1200x820. Also: `.png` (raster), paper `.pdf`. | Core architecture comparison figure | hand-coded SVG | 2026-03-13 | confirmed |
| 3 | `fig3_data_composition.png` | data | Dataset breakdown (4.89M samples, 6 sources, 4 categories) | Training data overview | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 4 | `fig4_main_run_progress.png` | training | Train + eval loss curves (MLP 4.89M, colored markers) | Key result: training trajectory | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 5 | `fig5_eval_loss.png` | evaluation | Eval loss with best-point annotation (0.361 at step 9750) | Key result: model quality | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 6 | `fig6_generation_quality.png` | generation | BLEU & ROUGE-L over steps 3000–9750 | Generation quality trend | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 7 | `fig7_scaling_effect.png` | scaling | 50K → 4.89M: 81% eval_loss improvement bar chart | Key finding: data scaling | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 8 | `fig8_mlp_vs_text.png` | comparison | MLP vs Text-only on same 4.89M dataset (step 200) | Key comparison: ESM-3 advantage | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 9 | `fig9_final_comparison.png` | comparison | Cross-approach bar chart (4 configs, annotated eval_loss) | Summary comparison | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 10 | `fig10_base_vs_finetuned.png` | comparison | Base model vs SFT: perplexity, BLEU, ROUGE-L (126x BLEU improvement) | Key result: SFT impact | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 11 | `fig11_generation_over_steps.png` | generation | BLEU & ROUGE-L progression over training steps (MLP vs Text) | Generation quality trend over time | `scripts/analysis/plot_pub_figures.py` | 2026-03-09 | confirmed |
| 12 | `fig12_embedding_umap.png` | embedding | UMAP 3-panel: Raw ESM-3 vs Trained vs Random projector | Shows projection training effect | `scripts/analysis/plot_embedding_analysis.py` | 2026-03-10 | confirmed |
| 5b | `fig5_sft_loss_curves.png` | training | SFT eval loss: ESM3+MLP vs Text-only (Qwen3-8B, 4.89M). Best MLP=0.361, best Text=1.207 | Updated SFT comparison with both approaches | `scripts/analysis/plot_grpo_diagnostics.py` | 2026-03-11 | confirmed |
| 7b | `fig7_grpo_reward_curves.png` | grpo | GRPO reward vs step: ProteinLM v1 (3ep), v2 (10ep), Structure (10ep). Structure reaches ~0.83, ProteinLM ~0.68 | GRPO training progress across tasks | `scripts/analysis/plot_grpo_diagnostics.py` | 2026-03-11 | confirmed |
| 8b | `fig8_grpo_reward_breakdown.png` | grpo | 2-panel: ProteinLM bench reward summary (mean/max/min/format) + Structure quality reward components (quality_align/numerical/category/format) | Reward component analysis per task | `scripts/analysis/plot_grpo_diagnostics.py` | 2026-03-11 | confirmed |
| 9b | `fig9_grpo_grad_norms.png` | grpo | 2-panel: LoRA grad norms + Multimodal grad norms for all GRPO runs. ProteinLM: LoRA only, Structure: multimodal only | Gradient flow diagnostic for GRPO | `scripts/analysis/plot_grpo_diagnostics.py` | 2026-03-11 | confirmed |
| 13 | `fig13_grpo_structure_reward_curves.png` | grpo | 2-panel: training reward + eval reward, Text-only vs ESM3+MLP on structure quality task. Text reaches 0.832, MLP stuck at 0.582 | Key GRPO comparison: text vs MLP reward trajectory | `scripts/analysis/plot_grpo_structure_comparison.py` | 2026-03-12 | confirmed |
| 14 | `fig14_grpo_structure_reward_breakdown.png` | grpo | Grouped bar chart: 4 reward components (quality_alignment, numerical_accuracy, category_match, format_bonus) for Text vs MLP | Reward component analysis shows MLP deficit across all components | `scripts/analysis/plot_grpo_structure_comparison.py` | 2026-03-12 | confirmed |
| 15 | `fig15_grpo_structure_format_rate.png` | grpo | 2-panel: format compliance rate at eval steps + PG loss curves. Text 100% format, MLP 74-76% | Format compliance gap explains reward difference | `scripts/analysis/plot_grpo_structure_comparison.py` | 2026-03-12 | confirmed |
| 16 | `fig16_grpo_structure_grad_norms.png` | grpo | 2-panel: LoRA grad norms (log scale) + Multimodal grad norms. MLP LoRA grad=0, Text MM grad=0 | Gradient flow diagnostic: MLP cannot update LoRA (format learning blocked) | `scripts/analysis/plot_grpo_structure_comparison.py` | 2026-03-12 | confirmed |

### Paper PDF Counterparts

| Fig | Blog PNG | Paper PDF |
|-----|----------|-----------|
| 1 | `main_figures/fig1_schematic_overview.png` | `paper/figures/main/fig1_schematic_overview.pdf` |
| 1a | `main_figures/fig1a_pathways.png` | `paper/figures/main/fig1a_pathways.pdf` |
| 1b | `main_figures/fig1b_training_pipeline.png` | `paper/figures/main/fig1b_training_pipeline.pdf` |
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
| 12 | `main_figures/fig12_embedding_umap.png` | `paper/figures/main/fig12_embedding_umap.pdf` |
| 5b | `main_figures/fig5_sft_loss_curves.png` | `paper/figures/main/fig5_sft_loss_curves.pdf` |
| 7b | `main_figures/fig7_grpo_reward_curves.png` | `paper/figures/main/fig7_grpo_reward_curves.pdf` |
| 8b | `main_figures/fig8_grpo_reward_breakdown.png` | `paper/figures/main/fig8_grpo_reward_breakdown.pdf` |
| 9b | `main_figures/fig9_grpo_grad_norms.png` | `paper/figures/main/fig9_grpo_grad_norms.pdf` |
| 13 | `main_figures/fig13_grpo_structure_reward_curves.png` | `paper/figures/main/fig13_grpo_structure_reward_curves.pdf` |
| 14 | `main_figures/fig14_grpo_structure_reward_breakdown.png` | `paper/figures/main/fig14_grpo_structure_reward_breakdown.pdf` |
| 15 | `main_figures/fig15_grpo_structure_format_rate.png` | `paper/figures/main/fig15_grpo_structure_format_rate.pdf` |
| 16 | `main_figures/fig16_grpo_structure_grad_norms.png` | `paper/figures/main/fig16_grpo_structure_grad_norms.pdf` |

---

## Supplementary Figures — Blog (59)

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
| `mlp_epoch1_eval_loss.png` | Eval loss curve with best-point annotation | 2026-03-09 | Epoch 1 analysis |
| `mlp_epoch1_generation_quality.png` | BLEU & ROUGE-L generation metrics | 2026-03-09 | Epoch 1 analysis |

### Publication-Quality Variants (pub_*)

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `pub_approach_loss_curves.png` | Approach comparison train loss | 2026-03-07 | Publication style |
| `pub_approach_eval_curves.png` | Approach comparison eval loss | 2026-03-07 | Publication style |
| `pub_convergence_table.png` | Convergence table (styled) | 2026-03-07 | Publication style |
| `pub_generation_quality.png` | BLEU/ROUGE-L by task type | 2026-03-07 | Publication style |
| `pub_gradient_norms.png` | Gradient norms (log scale, styled) | 2026-03-07 | Publication style |
| `pub_architecture.png` | ESM-3 + MLP architecture diagram | 2026-03-09 | Publication style |
| `pub_data_composition.png` | Dataset breakdown (4.89M, 6 sources) | 2026-03-09 | Publication style |
| `pub_final_comparison.png` | Cross-approach final metrics bar chart | 2026-03-09 | Publication style |
| `pub_main_run_progress.png` | Main run train + eval loss curves | 2026-03-09 | Publication style |
| `pub_mlp_vs_text_4m.png` | MLP vs Text-only on 4.89M dataset | 2026-03-09 | Publication style |
| `pub_scaling_effect.png` | 50K to 4.89M scaling effect bar chart | 2026-03-09 | Publication style |

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
| `schematic_overview.png` | Early 4-panel project overview diagram | 2026-03-09 | Architecture overview |

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


### Embedding Quality Analysis (Mar 2026)

**Dataset**: 27,818 downstream proteins (GO 8.8K + stability 10K + structure 10K), 7 biological classes. Flatten + PCA(128).

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `embedding_umap_3panel.png` | UMAP 3-panel: Raw ESM-3, Trained MLP, Random MLP | 2026-03-11 | Downstream biological labels |
| `embedding_metrics_comparison.png` | kNN@5, linear probe, silhouette bar chart | 2026-03-11 | All ~54%; random edges out |
| `embedding_cka_heatmap.png` | 3x3 CKA similarity heatmap | 2026-03-11 | Trained↔random=0.053 |
| `embedding_cosine_distributions.png` | Intra- vs inter-class cosine similarity | 2026-03-11 | Trained shows cosine~1.0 spike |
| `embedding_knn_vs_k.png` | kNN accuracy vs k (1, 5, 10, 20) | 2026-03-11 | Curves nearly overlap |
| `embedding_rank_isotropy.png` | Effective rank and isotropy bars | 2026-03-11 | Trained high isotropy, low rank |

**Blog post**: `blog/posts/2026-03-11_embedding-analysis-downstream.html`
**Key finding**: Trained projector does NOT improve embedding clustering — consistent with ACL 2024, VIRAL 2025, CVPR 2024 literature. Novel for protein domain.

### Comprehensive Diagnostics (Mar 15, 2026)

All-experiment diagnostic figures covering 3 SFT runs + 13 GRPO runs. Generated by artist agent from trainer_state.json + train.log + blog/data CSVs. All figures saved to both blog (PNG) and paper (PDF+PNG).

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `all_sft_loss_curves.png` | Smoothed token_avg_loss for all 3 SFT experiments (MLP 4.89M, Text 4.89M, MLP+Thinking) with raw data overlay | 2026-03-15 | MA-20 smoothing, paper PDF |
| `all_sft_eval_loss.png` | Eval loss curves with best-point annotations for all SFT runs. MLP best=0.361, Text best=1.207, Thinking best=0.613 | 2026-03-15 | Best points annotated, paper PDF |
| `all_sft_gradient_norms.png` | Log-scale gradient norms for all SFT runs with smoothed + raw overlay | 2026-03-15 | Log scale Y-axis, paper PDF |
| `all_sft_lr_schedule.png` | Learning rate warmup + cosine decay schedules for all SFT runs | 2026-03-15 | All use same LR=2e-4, paper PDF |
| `all_sft_loss_comparison_bar.png` | Grouped bar chart: final train loss + best eval loss per SFT experiment | 2026-03-15 | Annotated values, paper PDF |
| `all_grpo_reward_curves.png` | GRPO mean_reward vs step for all 13 runs parsed from train.log (MLP blue, Text gray, varied linestyles) | 2026-03-15 | 13 experiments, paper PDF |
| `all_grpo_gradient_norms.png` | 2-panel: LoRA grad norms (log scale) + Multimodal grad norms for all GRPO runs | 2026-03-15 | Key finding: MLP LoRA grad=0 in some runs, paper PDF |
| `all_grpo_format_compliance.png` | Format compliance rate (%) at eval steps over training for all GRPO runs with eval data | 2026-03-15 | Text=100%, MLP varies, paper PDF |
| `all_training_overview.png` | 4-panel overview: (A) SFT eval loss, (B) GRPO final rewards bar, (C) SFT grad norms, (D) GRPO reward trajectories (best per config) | 2026-03-15 | Summary view, paper PDF |

**Data sources**:
- SFT: trainer_state.json from latest checkpoints (3 experiments: MLP 1014 entries, Text 1002 entries, Thinking 52 entries)
- GRPO: train.log parsed for step-level data (13 experiments with step data), grpo_experiment_comparison.csv for final metrics

**Key findings across all experiments**:
- SFT: MLP (ESM3) achieves 0.361 eval loss vs Text-only 1.207 (3.3x better), Thinking mode 0.613 (early, 500 steps)
- GRPO Structure: Text-only reaches ~0.83 reward with 100% format compliance; MLP variants range 0.58-0.78
- GRPO gradient flow: MLP LoRA gradients are 0 in several structure runs, explaining format learning failure
- Format compliance is the key differentiator: Text=100%, MLP varies from 0% to 100% depending on config
- FrozenMM configs show improved MLP reward (0.77) vs standard MLP structure runs

### Comprehensive Diagnostics Refresh (Mar 16, 2026)

Updated diagnostic figures with `_0316` suffix. 12 figures total, each saved to blog (PNG 150 DPI) and paper (PDF+PNG 300 DPI). Generated directly from `results/` trainer_state.json + train.log.

| Filename | Content | Source Date | Notes |
|----------|---------|-------------|-------|
| `all_sft_loss_curves_0316.png` | Smoothed token_avg_loss for all 3 SFT experiments (MLP 4.89M, Text 4.89M, MLP+Thinking) with raw data overlay (MA-20) | 2026-03-16 | Paper PDF+PNG |
| `all_sft_eval_loss_0316.png` | Eval loss curves with best-point annotations. MLP best=0.361, Text best=1.207, Thinking best=0.613 | 2026-03-16 | Paper PDF+PNG |
| `all_sft_gradient_norms_0316.png` | Log-scale gradient norms for all SFT runs with smoothed + raw overlay | 2026-03-16 | Paper PDF+PNG |
| `all_sft_lr_schedule_0316.png` | Learning rate warmup + cosine decay schedules for all SFT runs | 2026-03-16 | Paper PDF+PNG |
| `all_sft_loss_comparison_bar_0316.png` | Grouped bar chart: final train loss + best eval loss per SFT experiment, annotated values | 2026-03-16 | Paper PDF+PNG |
| `all_grpo_reward_curves_0316.png` | GRPO mean_reward vs step for all 13 runs (MLP blue, Text gray, varied linestyles) | 2026-03-16 | Paper PDF+PNG |
| `all_grpo_gradient_norms_0316.png` | 2-panel: LoRA grad norms (log scale) + Multimodal grad norms for all GRPO runs | 2026-03-16 | Paper PDF+PNG |
| `all_grpo_eval_reward_trajectories_0316.png` | GRPO eval mean reward at eval checkpoints for runs with eval data | 2026-03-16 | Paper PDF+PNG |
| `all_grpo_format_compliance_0316.png` | Format compliance rate (%) at eval steps for all GRPO runs with eval data | 2026-03-16 | Paper PDF+PNG |
| `all_grpo_final_reward_bar_0316.png` | Final reward bar chart for all 13 GRPO experiments, colored by approach | 2026-03-16 | Paper PDF+PNG |
| `all_grpo_reward_step_level_0316.png` | Step-level reward for top 6 longest GRPO runs with MA-5 smoothing + raw overlay | 2026-03-16 | Paper PDF+PNG |
| `all_training_overview_0316.png` | 4-panel overview: (A) SFT eval loss, (B) GRPO final rewards bar, (C) SFT grad norms, (D) GRPO best reward trajectories | 2026-03-16 | Paper PDF+PNG |

**Script**: `scripts/analysis/plot_all_diagnostics_0316.py`
**Data**: 3 SFT experiments (1014+1002+52 log entries), 13 GRPO experiments (7-30 steps each)

## Generation Scripts

All scripts use `scripts/analysis/figure_style.py` for consistent styling. See `scripts/analysis/STYLE_GUIDE.md`.

| Script | Figures Generated | Notes |
|--------|-------------------|-------|
| `scripts/analysis/plot_pub_figures.py` | `fig1`–`fig11` (main figures) | Consolidated from `blog/data/03-07/pub_figures.py` (archived). CLI: `--figures 4 5 6 10 11` |
| `scripts/analysis/plot_training.py` | Diagnostic plots (loss, eval, grad norms, etc.) | Reusable: `--experiments`, `--output`, `--paper-output`, `--prefix`, `--plots` |
| `scripts/analysis/plot_grpo_diagnostics.py` | `fig5_sft_loss_curves`, `fig7_grpo_reward_curves`, `fig8_grpo_reward_breakdown`, `fig9_grpo_grad_norms` | GRPO + SFT comparison diagnostics |
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
