# Data Collection Summary for Figure Upgrade

**Date**: 2026-03-09
**Collector**: data-collector agent

---

## Experiment Inventory

### Currently on Disk (results/)

| Experiment | Approach | Stage | Steps | Best Eval Loss | Gen Evals | Status |
|------------|----------|-------|-------|----------------|-----------|--------|
| sft_esm3_mlp_combined_qwen3_8b_it_0306_010408 | esm3/mlp | sft_lora | 9750 | 0.3607 | 28 | completed |
| sft_text_combined_qwen3_8b_it_0307_190324 | text | sft_lora | 8500 | 1.2072 | 34 | in-progress (still training) |
| sft_esm3_mlp_thinking_qwen3_8b_it_0309_115011 | esm3/mlp | sft_lora | 500 | 0.6131 | 2 | in-progress (thinking mode) |
| grpo_structure_mlp_sft_qwen3_8b_it_0309_154144 | esm3/mlp | grpo | 0 | -- | 0 | just launched |
| grpo_structure_mlp_sft_qwen3_8b_it_0309_154545 | esm3/mlp | grpo | 0 | -- | 0 | just launched |
| sft_esm3_perceiver_combined_qwen3_8b_it_0308_165601 | esm3/perceiver | sft_lora | 0 | -- | 0 | failed (logs only) |
| base_qwen3_8b_esm3_random | esm3 | baseline | -- | -- | 0 | eval only |
| base_qwen3_8b_text | text | baseline | -- | -- | 0 | eval only |

### Previously Collected (no longer on disk)

These experiments were in earlier data collections but their result directories have been cleaned up:

**From 03-02 (50K dataset, early experiments)**:
- sft_lora_esm3_qwen3_8b_it_0225_203237 (esm3/mlp, 50K)
- sft_lora_esm3_qwen3_8b_it_0226_151416 (esm3/mlp, 50K)
- sft_text_qwen3_8b_it_0227_145821 (text, 50K)

**From 03-04 (long runs, intermediate combined)**:
- sft_esm3_mlp_long_qwen3_8b_it_0302_175459
- sft_esm3_mlp_combined_qwen3_8b_it_0302_201516
- sft_esm3_mlp_long_qwen3_8b_it_0303_004216
- sft_esm3_mlp_long_qwen3_8b_it_0303_154932
- sft_esm3_mlp_combined_qwen3_8b_it_0304_093325
- sft_esm3_mlp_combined_qwen3_8b_it_0304_102354

**From 03-06 (combined dataset iterations)**:
- sft_esm3_mlp_combined_qwen3_8b_it_0304_150544
- sft_esm3_mlp_combined_qwen3_8b_it_0305_100040
- sft_esm3_mlp_combined_qwen3_8b_it_0305_115307
- sft_esm3_mlp_combined_qwen3_8b_it_0305_115509
- sft_esm3_mlp_combined_qwen3_8b_it_0305_222552
- sft_esm3_mlp_combined_qwen3_8b_it_0306_002235 through _003113
- sft_esm3_mlp_long_qwen3_8b_it_0305_105828

**From 03-07 (epoch 1 deep dive)**:
- Same as 03-06 plus sft_text_combined_qwen3_8b_it_0307_061432

---

## Available Data for Figure Upgrades

### 1. Training Loss Curves (token_avg_loss)

| Experiment | Steps with data | Range |
|------------|----------------|-------|
| MLP combined (0306) | 975 train entries | 3.297 -> 0.438 |
| Text combined (0307) | 850 train entries | 4.073 -> 1.697 |
| MLP thinking (0309) | 50 train entries | 2.630 -> 0.634 |

**Historical data** available in `blog/data/03-02/run_histories.csv` (50K experiments) and `blog/data/03-07/run_histories.csv` (earlier combined runs).

### 2. Eval Loss Curves

| Experiment | Eval entries | Best eval_loss | Best step |
|------------|-------------|----------------|-----------|
| MLP combined (0306) | 39 | 0.3607 | 9750 |
| Text combined (0307) | 34 | 1.2072 | 7500 |
| MLP thinking (0309) | 2 | 0.6131 | 500 |

### 3. Generation Quality (BLEU, ROUGE-L)

| Experiment | Steps evaluated | Best BLEU | Best ROUGE-L |
|------------|----------------|-----------|-------------|
| MLP combined (0306) | 28 (steps 3000-9750) | 0.3147 (step 9250) | 0.5145 (step 9250) |
| Text combined (0307) | 34 (steps 250-8500) | 0.2652 (step 8500) | 0.4478 (step 8500) |
| MLP thinking (0309) | 2 (steps 250-500) | too early | too early |

### 4. Base Model Evaluation (unfinetuned)

| Model | Perplexity | BLEU | ROUGE-L |
|-------|-----------|------|---------|
| base_qwen3_8b_esm3_random | 28.25 | 0.0025 | 0.0569 |
| base_qwen3_8b_text | 22.73 | 0.0033 | 0.0607 |

### 5. GRPO/RL Experiments

Two GRPO runs just launched (2026-03-09 15:45) on structure quality task:
- Parent: sft_esm3_mlp_combined_qwen3_8b_it_0306_010408 (checkpoint-9750)
- Reward: ESMFold structure prediction
- LR: 5e-06, group_size=8, batch=4, grad_accum=4
- No training data yet (just started)

### 6. Perceiver Experiment

sft_esm3_perceiver_combined_qwen3_8b_it_0308_165601: Failed before producing checkpoints. Only has a logs directory.

---

## Figure Inventory Summary

### Main Figures (9) -- blog/figures/main_figures/

All updated 2026-03-09 02:32 via `scripts/analysis/plot_pub_figures.py`:

| Fig | Name | Content |
|-----|------|---------|
| 1 | fig1_schematic_overview.png | 4-panel project overview |
| 2 | fig2_architecture.png | ESM-3 + MLP architecture |
| 3 | fig3_data_composition.png | 4.89M dataset breakdown |
| 4 | fig4_main_run_progress.png | Train + eval loss (MLP 4.89M) |
| 5 | fig5_eval_loss.png | Eval loss with best-point (0.361 at 9750) |
| 6 | fig6_generation_quality.png | BLEU and ROUGE-L over steps |
| 7 | fig7_scaling_effect.png | 50K -> 4.89M scaling bar chart |
| 8 | fig8_mlp_vs_text.png | MLP vs Text on same dataset |
| 9 | fig9_final_comparison.png | Cross-approach bar chart |

### Supplementary Figures (59) -- blog/figures/supple_figures/

- 6 early MLP vs Text (50K) comparisons (03-02)
- 6 three-way comparisons (MLP/Perceiver/Text, 50K) (03-02)
- 6 extended MLP training (03-04)
- 11 MLP combined run detail (03-06)
- 6 MLP epoch 1 deep-dive (03-07)
- 5 pub-style variants (03-07)
- 9 web-style variants (03-07)
- Several newer figures: pub_architecture, pub_data_composition, pub_final_comparison, etc. (03-09)

### Paper Figures

- 9 main PDFs in paper/figures/main/ (match fig1-fig9)
- 30 supplementary files in paper/figures/supplementary/ (11 PDFs + 19 PNGs)

### Source Scripts

| Script | Purpose |
|--------|---------|
| `plot_pub_figures.py` | fig1-fig9 main figures |
| `plot_training.py` | Reusable diagnostic plots |
| `plot_schematic_overview.py` | fig1 schematic |
| `figure_style.py` | Style module (colors, fonts, save helpers) |
| `STYLE_GUIDE.md` | Style documentation |

---

## What Is NEW Since Last Collection (03-07)

1. **Text combined run progressed**: Was at ~3500 steps, now at 8500 steps with 34 generation evals
2. **MLP thinking mode experiment**: New run (sft_esm3_mlp_thinking_qwen3_8b_it_0309_115011) with 500 steps
3. **GRPO experiments launched**: Two grpo_structure runs just started
4. **Perceiver experiment attempted**: Failed before producing data
5. **Base model evaluations**: Two new baseline eval results (perplexity, BLEU, ROUGE-L on 5000 samples)
6. **Main figures regenerated**: All 9 main figures updated 2026-03-09 via plot_pub_figures.py
7. **Figure reorganization**: supple_figures now has 59 figures (was ~50 in catalog)

## What Is MISSING / Needed for Upgrades

1. **Perceiver combined run**: Failed, no training data. Need successful perceiver run for 4-way comparison.
2. **Flamingo run**: No flamingo experiment has been attempted yet. Needed for full thesis (4-way comparison).
3. **GRPO training curves**: Just launched, no data yet. Need to wait and re-collect.
4. **Text run completion**: Still training (epoch 0.026 of unknown total). Need more steps.
5. **MLP thinking run**: Only 500 steps so far. Too early for meaningful comparison.
6. **Cross-task generation metrics**: All generation evals use same SFT eval set. No downstream task-specific generation metrics yet.
7. **Downstream task evaluations**: No GO prediction, stability, or structure quality eval results from finetuned models.
