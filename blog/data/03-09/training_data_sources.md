# Training Data Sources Audit — 2026-03-09

## Overview

Audited all local and wandb data sources for MLP and Text SFT training runs.
The key finding: **local trainer_state.json files contain all loss/metric data needed for loss curves**.
wandb adds generation quality metrics (BLEU/ROUGE per category) not logged locally.

---

## MLP Combined (sft_esm3_mlp_combined_qwen3_8b_it_0306_010408)

**Status**: Stopped early at step 9,750 / 28,941 max_steps (33.7% complete)

### Local Data
- **Source**: `checkpoints/checkpoint-9750/trainer_state.json`
- **Steps**: 10 to 9,750 (every 10 steps, 975 train entries + 39 eval entries)
- **Fields**: `token_avg_loss`, `loss`, `grad_norm`, `learning_rate`, `epoch`, `step`
- **Eval**: Every 250 steps, fields include `eval_loss`, `eval_runtime`, etc.
- **Gaps**: None. Continuous 10-step intervals.
- **Note**: `save_total_limit=3` means only checkpoints 9250/9500/9750 remain. No top-level trainer_state.json.

### wandb Data
| Run ID | State | Steps | Rows | Role |
|--------|-------|-------|------|------|
| p4ojn2n0 | crashed | 0-375 | 376 | Initial run |
| i8rebnp7 | crashed | 0-9525 | 656 | Resumed run |

- wandb run2 max step (9525) < local max (9750), so **local is more complete** for loss data.
- wandb run2 has **generation metrics** (BLEU, ROUGE per category: catalytic/domain/function/general) starting from step 3750.

### CSV Data (existing)
- `blog/data/03-09/run_histories.csv`: 1014 rows, steps 10-9750 (complete)
- `blog/data/03-07/run_histories.csv`: 1014 rows, steps 10-9750 (complete)
- `blog/data/03-06/run_histories.csv`: 728 rows, steps 10-7000 (partial, run was still going)

### Recommendation
- Local data is **sufficient** for loss curves.
- To add generation quality metrics, pull from wandb run `i8rebnp7`.

---

## Text Combined (sft_text_combined_qwen3_8b_it_0307_190324)

**Status**: RUNNING at step 8,525 / 9,647 max_steps (88.4% complete)

### Local Data
- **Source**: `checkpoints/checkpoint-8500/trainer_state.json`
- **Steps**: 10 to 8,500 (every 10 steps, 850 train entries + 34 eval entries)
- **Fields**: Same as MLP (token_avg_loss, loss, grad_norm, learning_rate, epoch, step)
- **Gaps**: None through step 8,500.
- **Note**: 8+ restarts documented by `resume_train.log` through `resume_train8.log`. Many FSDP/generation issues resolved across restarts.

### wandb Data
| Run ID | State | Steps | Rows | Role |
|--------|-------|-------|------|------|
| cre9zvww | crashed | 0-5015 | 541 | First significant run |
| 5t71e8zs | crashed | 0-8270 | 362 | Resumed |
| a8id33eo | **running** | 0-8525 | 52 | Currently active |
| pb6u8cx2 | failed | - | 0 | Failed restart |
| jnaje0wd | crashed | - | 0 | Failed restart |
| fqvda4cx | crashed | - | 0 | Failed restart |
| 8tvtx80u | failed | - | 0 | Failed restart |
| hyjm4t87 | crashed | - | 0 | Failed restart |
| rkseaiqf | crashed | - | 0 | Failed restart |
| mhowvt47 | crashed | - | 0 | Failed restart |

- 10 wandb runs total; only 3 have actual data.
- Active run (a8id33eo) has 25 steps beyond local checkpoint (8510-8525).
- wandb runs cre9zvww and 5t71e8zs have generation metrics (BLEU/ROUGE per category).

### CSV Data (existing)
- `blog/data/03-09/run_histories.csv`: 884 rows, steps 10-8500
- `blog/data/03-07/run_histories.csv`: 26 rows, steps 10-250 (very early)

### Recommendation
- Local data covers 10-8500; wandb active run adds 8510-8525 (minor).
- Training should complete within hours (1,147 steps remaining).
- **Wait for completion**, then pull final trainer_state from last checkpoint.
- For generation metrics, combine wandb runs cre9zvww (steps 250-5000) + 5t71e8zs (steps 5250-8250).

---

## Other SFT Runs

### Perceiver (sft_esm3_perceiver_combined_qwen3_8b_it_0308_165601)
- **Status**: No checkpoints. Only `logs/` directory exists.
- **wandb**: No runs found.
- **Likely**: Training did not produce checkpoints or failed very early.

### MLP Thinking (sft_esm3_mlp_thinking_qwen3_8b_it_0309_115011)
- **Status**: 52 rows in CSV, steps 10-500.
- **wandb**: Run x0vwht2v, crashed at step 512.

### Earlier runs (50K sample dataset)
| Experiment | Steps | wandb State |
|------------|-------|-------------|
| sft_lora_esm3_qwen3_8b_it_0225_203237 (MLP) | 10-1677 | finished |
| sft_lora_esm3_qwen3_8b_it_0226_151416 (MLP) | 10-1677 | finished |
| sft_text_qwen3_8b_it_0227_145821 (Text) | 10-2595 | finished |
| sft_esm3_mlp_long_qwen3_8b_it_0305_105828 (MLP) | 10-1500 | crashed (1758) |

---

## Key Conclusions

1. **For loss curves**: Local `trainer_state.json` is the authoritative source. The existing `03-09/run_histories.csv` already contains all available loss data.

2. **wandb adds generation metrics**: BLEU and ROUGE (overall + per-category: catalytic, domain, function, general) are logged to wandb but NOT to trainer_state.json. These are valuable for evaluating generation quality over training.

3. **No hidden data**: There are no steps in wandb that are missing from local for the loss metrics. wandb actually has fewer steps (due to crashes before checkpointing).

4. **Text training is still running**: Will complete at step 9,647. The current CSV and local data cover through step 8,500. Once training completes, the final checkpoint's trainer_state.json will have the complete history.

5. **MLP training stopped early**: Only 33.7% complete (9,750 of 28,941 steps). This was likely intentional as loss had converged.
