# Figure Upgrade Plan

**Date**: 2026-03-09
**Author**: Analyst Agent
**Scope**: Review of all 9 main figures and 50+ supplementary figures

---

## Executive Summary

Three main figures have critical issues requiring immediate attention:

1. **fig8 (MLP vs Text)** -- Text run shown at 200 steps but actually has 8500 steps of data
2. **fig6 (Generation Quality)** -- Uses hardcoded fallbacks that are 3x off from real metrics
3. **fig9 (Final Comparison)** -- Missing Text 4.89M results and base model baseline

Additionally, fig2 (Architecture) only shows the MLP approach when the project compares 4 approaches. Several high-value new figures can be created from existing but unused data.

---

## Part 1: Main Figure Review

### Fig 1: Schematic Overview -- STATUS: OK

- **Visual quality**: Good. 4-panel layout (Question, Approach, Training, Results) is clear.
- **Shows all 4 approaches**: Yes (text, MLP, perceiver, flamingo).
- **Issues**: Flamingo and GRPO marked "PENDING" -- this is accurate since no complete experiments exist.
- **Action**: No changes needed until flamingo/GRPO experiments complete.

### Fig 2: Architecture -- STATUS: NEEDS UPGRADE

- **Visual quality**: Functional but crude. Plain rectangles, no visual distinction between frozen and trainable.
- **Critical issue**: Only shows MLP approach. Title explicitly says "ESM3 + MLP Approach."
- **MLP dimension wrong**: Shows 1536->2560 but actual architecture is 1536->5120->2560 (2-layer MLP with hidden=5120).
- **Missing approaches**: Text (direct tokenization), Perceiver (resampler), Flamingo (gated cross-attention).
- **Action**: Replace with multi-approach architecture diagram. The existing `supple_figures/architecture_comparison.png` already shows all 4 -- consider promoting it.

### Fig 3: Data Composition -- STATUS: ACCEPTABLE

- **Visual quality**: Clean horizontal bar chart. Numbers total 4.89M correctly.
- **Data source**: Hardcoded fallback, but numbers appear accurate.
- **Minor issue**: Category names are high-level; no source-level breakdown.
- **Action**: Verify numbers against actual arrow dataset. Low priority.

### Fig 4: Main Run Progress -- STATUS: GOOD

- **Visual quality**: Clean. Raw loss (faint) with MA-50 smoothing + eval diamonds.
- **Data**: Real data from run_histories.csv. MLP run at step 9750 (latest available).
- **Action**: Update when MLP run progresses further. No immediate changes.

### Fig 5: Eval Loss -- STATUS: GOOD (ENHANCEMENT POSSIBLE)

- **Visual quality**: Clean with best-point annotation (0.361 at step 9750).
- **Issue**: Only shows MLP. Text eval loss now available (1.293 at step 8500).
- **Action**: Consider adding text eval loss overlay for comparison context.

### Fig 6: Generation Quality -- STATUS: CRITICAL FIX NEEDED

- **Visual quality**: The chart itself is well-formatted.
- **CRITICAL**: Uses hardcoded fallback values that are 3x off from real data:

| Metric | Fallback (shown) | Actual (generation_metrics.json) |
|--------|-------------------|----------------------------------|
| Overall BLEU | 0.307 | 0.097 |
| Overall ROUGE-L | 0.483 | 0.299 |
| Function BLEU | 0.404 | 0.111 |
| Catalytic BLEU | 0.392 | 0.283 |
| Domain BLEU | 0.166 | 0.060 |

- **Root cause**: `_GEN_METRICS_FALLBACK` in plot_pub_figures.py was never updated, and `load_metrics()` always falls through to fallback for generation metrics (comment says "no standard file format for these yet").
- **Additional data available**: Per-step generation files exist (33 for MLP, 34 for text). Text run at step 7000 has BLEU=0.233, ROUGE-L=0.422.
- **Action**: Load real generation_metrics.json. Also create a generation-over-training-steps figure.

### Fig 7: Scaling Effect -- STATUS: ACCEPTABLE

- **Visual quality**: Effective infographic with giant "81%" callout.
- **Numbers accurate**: 50K eval_loss=1.942, 4.89M eval_loss=0.361, improvement=81.4%.
- **Minor caveat**: Compares 3 epochs (50K) vs 0.34 epochs (4.89M) -- not noted in figure.
- **Action**: Add footnote about training duration. Low priority.

### Fig 8: MLP vs Text -- STATUS: CRITICAL FIX NEEDED

- **Visual quality**: The MLP curve looks good but text is a tiny dot cluster.
- **CRITICAL**: Text run shown at ~200 steps but has 8500 steps of data:
  - `blog/data/03-07/run_histories.csv` had text at 200 steps (data collection was early)
  - `blog/data/03-09/run_histories.csv` has text at 7000 steps (better but still not latest)
  - `checkpoint-8500/trainer_state.json` has full 884 entries through step 8500
  - Text final: token_avg_loss=1.70, eval_loss=1.293 at step 8500
- **Result**: Figure is misleading -- looks like text was barely attempted when it actually ran for 8500 steps.
- **Action**: Re-collect text data through step 8500 and regenerate. Show full training trajectories for both.

### Fig 9: Final Comparison -- STATUS: CRITICAL FIX NEEDED

- **Visual quality**: Clean grouped bar chart.
- **Issues**:
  - Shows 4 configs from hardcoded fallback: Perceiver 50K, MLP 50K, Text 50K, MLP 4.89M
  - **Missing**: Text 4.89M result (eval_loss=1.293, the second-best result)
  - **Missing**: Base model baseline (text ppl=22.73, ESM3+random ppl=28.25)
  - **Missing**: Generation metrics comparison
  - 50K experiments are from February -- superseded by 4.89M runs
- **Action**: Add Text 4.89M. Include base model baseline. Restructure to highlight 4.89M results.

---

## Part 2: Proposed New Figures

### HIGH PRIORITY (data exists, not yet visualized)

| # | Name | Description | Data Source |
|---|------|-------------|-------------|
| 10 | Generation Over Training | BLEU/ROUGE-L progression over steps for MLP (2750-9750) and text (250-8500) | 33+34 per-step generation_step_*.json files |
| 11 | Base vs Fine-Tuned | Perplexity/BLEU/ROUGE-L comparison: base Qwen3-8B vs SFT models | blog/data/03-09/base_model_comparison.json |
| 12 | Eval Loss Comparison | MLP vs Text eval loss curves over full training (9750 and 8500 steps) | trainer_state.json from both experiments |
| 13 | Generation Quality Comparison | MLP vs Text per-category BLEU/ROUGE-L at best checkpoint | Per-step generation JSONs |

### MEDIUM PRIORITY (data partially available)

| # | Name | Description | Data Source |
|---|------|-------------|-------------|
| 14 | Random Projector Ablation | Shows ESM3+random projector HURTS vs text-only (ppl 28.25 vs 22.73) | base_model_comparison.json |

### BLOCKED (waiting for experiments)

| # | Name | Description | Blocker |
|---|------|-------------|---------|
| 15 | GRPO Results | Reward curves and downstream task performance | GRPO runs just started 2026-03-09, no metrics |
| 16 | Perceiver 4.89M | MLP vs Perceiver on combined dataset | Perceiver run has no training data yet |

---

## Part 3: Supplementary Figure Review

### Current and up-to-date (no action needed)
- `mlp_epoch1_*.png` (6 files) -- Step 9750 analysis from 2026-03-07
- `pub_*.png` (9 files) -- Publication-style variants from 2026-03-09
- `web_*.png` (9 files) -- Website variants from 2026-03-08

### Outdated but retain as historical reference
- `three_way_*.png` (6 files) -- 50K three-way comparison from 2026-03-02
- `mlp_vs_text_*.png` (5 files) -- 50K two-way comparison from 2026-03-02
- `long_mlp_*.png` (3 files) -- Mol-instructions-only run from 2026-03-04

### Should be regenerated
- `mlp_combined_*.png` (7 files) -- Step 7090 snapshot from 2026-03-06. MLP run now at 9750.

### Candidate for promotion to main figures
- `architecture_comparison.png` -- Shows all 4 approaches. Should replace fig2.

### Redundant / cleanup candidates
- `pub_*.png` in supple_figures/ duplicates main_figures/fig*.png. Consider removing from supple.
- `schematic_overview.png` in supple_figures/ duplicates main_figures/fig1_schematic_overview.png.

---

## Priority Order for Fixes

1. **fig8** -- Text data stale (200 vs 8500 steps). Most misleading figure.
2. **fig6** -- Generation metrics wrong by 3x. Hardcoded fallbacks never replaced.
3. **fig9** -- Missing Text 4.89M and base model. Incomplete comparison.
4. **fig2** -- Only MLP architecture, should show all 4 approaches.
5. **fig10 (NEW)** -- Generation over training steps. Rich data unused.
6. **fig11 (NEW)** -- Base vs fine-tuned. Data collected but no figure.
7. **fig12 (NEW)** -- Eval loss comparison with full text data.
8. **fig13 (NEW)** -- MLP vs text generation quality.
9. **fig5** -- Enhancement: add text eval overlay.
10. **fig7** -- Minor: add training duration caveat.
11. **fig3** -- Minor: verify dataset numbers.
12. **fig1** -- OK as-is.
13. **fig4** -- OK as-is.

---

## Data Collection Gaps

| Gap | Description | Action |
|-----|-------------|--------|
| Text run full history | 03-09 CSV has 7000 steps, checkpoint has 8500 | Re-collect from checkpoint-8500/trainer_state.json |
| Generation per-step | 67 generation JSONs not aggregated into CSV | Data-collector should create generation_history.csv |
| Perceiver combined | Only logs dir, no training data | Wait for experiment |
| GRPO results | Two runs started today, no metrics | Wait for completion |

---

## Key Findings from Real Data vs Displayed Data

| Metric | Currently Displayed | Actual Value | Discrepancy |
|--------|-------------------|--------------|-------------|
| Overall BLEU (MLP) | 0.307 (fig6) | 0.097 (generation_metrics.json) | 3.2x overestimate |
| Overall ROUGE-L (MLP) | 0.483 (fig6) | 0.299 (generation_metrics.json) | 1.6x overestimate |
| Text training steps shown | ~200 (fig8) | 8500 (checkpoint) | 42x more data available |
| Text eval_loss | not shown (fig9) | 1.293 at step 8500 | Missing entirely |
| Text BLEU (step 7000) | not shown | 0.233 | Missing entirely |
| Text ROUGE-L (step 7000) | not shown | 0.422 | Missing entirely |
