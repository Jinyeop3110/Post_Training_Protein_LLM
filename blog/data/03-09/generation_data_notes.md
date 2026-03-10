# Generation Over Steps Data Notes

## Date: 2026-03-09

## Source: Local experiment eval directories (Priority 1)

All data was collected from local `generation_step_*.json` files under
`results/{experiment}/eval/`. No wandb data was needed -- local files had
equal or better coverage than wandb for all experiments.

## Experiments Collected

### 1. sft_esm3_mlp_combined_qwen3_8b_it_0306_010408 (approach: esm3_mlp)
- **Steps**: 3000 to 9750 (28 data points, every 250 steps)
- **BLEU range**: 0.1787 to 0.3147
- **ROUGE-L range**: 0.4137 to 0.5145
- **Samples per eval**: 40
- **Note**: Generation eval was not enabled before step 3000 for this run.
  wandb (run id: i8rebnp7) had data starting at step 3750, so local is more complete.

### 2. sft_text_combined_qwen3_8b_it_0307_190324 (approach: text)
- **Steps**: 250 to 9500 (38 data points, every 250 steps)
- **BLEU range**: 0.0000 to 0.2756
- **ROUGE-L range**: 0.0148 to 0.4664
- **Samples per eval**: 40 (except first few evals which had 3 samples due to early errors)
- **Note**: Step 250 shows BLEU=0 with `[ERROR: 'weight' must be 2-D]` in generated text,
  indicating the model was not yet producing valid output at that point.
  wandb had data across two crashed runs (cre9zvww: steps 250-5000, 5t71e8zs: steps 5250-8250).

### 3. sft_esm3_mlp_thinking_qwen3_8b_it_0309_115011 (approach: esm3_mlp_thinking)
- **Steps**: 250 to 500 (2 data points)
- **BLEU range**: 0.0031 to 0.0102
- **ROUGE-L range**: 0.0653 to 0.0734
- **Note**: This is a thinking-mode run still in progress. Very early stage.

## File Format

`generation_over_steps.csv` columns:
- `experiment`: Full experiment name
- `approach`: Short approach label (esm3_mlp, text, esm3_mlp_thinking)
- `step`: Training step number
- `num_samples`: Number of generation samples evaluated
- `bleu_overall`: Corpus-level BLEU across all categories
- `rouge_l_overall`: Average ROUGE-L across all categories
- `bleu_{category}`: Per-category BLEU (function, catalytic, domain, general)
- `rouge_l_{category}`: Per-category ROUGE-L

## Metric Details

- BLEU and ROUGE-L are computed by comparing model-generated text against reference answers
- Categories: catalytic (enzyme reactions), domain (protein domains/motifs),
  function (molecular function), general (protein name, family, subcellular location, etc.)
- Missing per-category values indicate that category was not represented in the eval sample

## Data Integrity

- All 68 rows sourced entirely from local JSON files
- No wandb API calls were needed for data collection
- JSON files contain full prediction-level detail (per-sample BLEU/ROUGE-L + generated text)
  but only aggregate metrics are included in the CSV
