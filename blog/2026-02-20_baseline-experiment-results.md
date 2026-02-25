---
title: "Baseline Experiment Results: Where We Stand"
date: 2026-02-20
author: yeopjin
tags: [evaluation, sft, training, milestone]
---

# Baseline Experiment Results: Where We Stand

With the ESM-3 + Qwen3 multimodal pipeline fully operational, we ran a comprehensive set of baseline experiments to establish performance benchmarks across three protein understanding tasks: GO term prediction, protein-protein interaction (PPI) prediction, and protein stability prediction. This post documents the results, compares model scales, and outlines what they tell us about the path forward.

## Experiment Setup

All experiments share the same core architecture:

- **Protein Encoder**: ESM-3 small (1.4B params, frozen, 1536-dim embeddings)
- **Pooling**: Attention pooling compressing to 32 fixed tokens
- **Projector**: MLP (1536 → 2048 → 2560), trainable
- **LLM**: Qwen3-4B / Qwen3-8B / Qwen3-14B with LoRA on k/v matrices (r=8)
- **Trainable parameters**: ~20M (0.49% of 4B model)

Training used the Mol-Instructions dataset with differential learning rates (projector LR at 10x base) and cosine scheduling.

### Training Runs Summary

| Run | Samples | Epochs | Loss (Start → End) | Duration | Notes |
|-----|---------|--------|---------------------|----------|-------|
| 500-sample test | 500 | 3 | 17.35 → 4.08 | ~6 min | First multimodal success |
| 10K baseline | 10,000 | 3 | 35.80 → 27.84 | ~96 min | Convergence test |
| 50K full | 50,000 | 5 | 34.25 → 14.77 | ~2.35 hr | Full-scale baseline |

The 50K run used an effective batch size of 32 (4 per device x 8 gradient accumulation steps) across 7,815 total steps, peaking at ~39 GB GPU memory.

## Evaluation Results

### 1. Vanilla Qwen3-4B (Pre-SFT Baseline)

Before any fine-tuning, the vanilla Qwen3-4B establishes the floor — what the LLM can do with protein sequences it has never been explicitly trained on.

**GO Term Prediction** (10 proteins):

| Metric | Value |
|--------|-------|
| Accuracy | 0.0 |
| F1 Macro | 0.015 |
| F1 Micro | 0.047 |
| AUPR (macro) | 0.153 |
| Avg predicted terms | 11.9 |
| Avg ground truth terms | 5.0 |

The model over-predicts GO terms (11.9 vs. 5.0 ground truth) and the predictions are mostly incorrect. Only 3 out of 10 proteins had any correct term predicted at all.

**PPI Prediction** (15 pairs):

| Metric | Value |
|--------|-------|
| Accuracy | 0.667 |
| F1 | 0.286 |
| Recall | 0.2 |
| Specificity | 0.9 |
| AUROC | 0.51 |

The model is heavily biased toward predicting "No interaction" — it predicted negative for 13 out of 15 pairs. High specificity (0.9) but terrible recall (0.2). The AUROC of 0.51 is essentially random.

**Stability Prediction** (20 mutations):

| Metric | Value |
|--------|-------|
| Pearson r | 0.094 |
| MAE | 1.86 kcal/mol |
| RMSE | 2.38 |
| Classification Accuracy | 0.35 |

The model defaults to predicting "neutral" for most mutations, completely failing to distinguish stabilizing from destabilizing effects.

### 2. Post-SFT Results: Model Scale Comparison

After SFT on the Mol-Instructions dataset, we evaluated three model scales. The table below compares their performance.

#### SFT Text Generation Quality

| Model | Perplexity | BLEU | ROUGE-L |
|-------|-----------|------|---------|
| Qwen3-4B | 84.42 | 0.0076 | 0.114 |
| Qwen3-8B | 76.02 | 0.0075 | 0.126 |
| Qwen3-14B | 72.80 | 0.0074 | 0.116 |

Perplexity improves with scale as expected (84 → 76 → 73), but BLEU and ROUGE-L remain low across the board — the model generates relevant-sounding responses but doesn't closely match reference answers.

**ROUGE-L by task category (Qwen3-4B / 8B / 14B)**:

| Task Category | 4B | 8B | 14B |
|---------------|-----|-----|------|
| Protein Function | 0.113 | 0.123 | 0.113 |
| Domain/Motif | 0.137 | 0.180 | 0.160 |
| Protein Design | 0.050 | 0.059 | 0.065 |
| Catalytic Activity | 0.140 | 0.181 | 0.159 |
| Description | 0.130 | 0.122 | 0.118 |

Domain/Motif and Catalytic Activity see the largest gains from scaling, while Protein Design remains challenging at all scales.

#### GO Term Prediction (10 proteins)

| Metric | Vanilla 4B | SFT 4B | SFT 8B | SFT 14B |
|--------|-----------|--------|--------|---------|
| F1 Macro | 0.015 | 0.0 | 0.0 | 0.0 |
| F1 Micro | 0.047 | 0.0 | 0.0 | 0.0 |
| AUPR (macro) | 0.153 | 0.138 | 0.138 | 0.138 |
| Avg predicted terms | 11.9 | 2.2 | 0.0 | 0.3 |

SFT actually *degrades* GO prediction. The vanilla model at least guessed some correct terms among its many predictions; after SFT, the models predict too few terms (or none at all for 8B). This suggests the current SFT objective doesn't teach structured GO term output.

#### PPI Prediction (15 pairs)

| Metric | Vanilla 4B | SFT 4B | SFT 8B | SFT 14B |
|--------|-----------|--------|--------|---------|
| Accuracy | 0.667 | 0.667 | 0.667 | 0.667 |
| F1 | 0.286 | 0.286 | 0.0 | 0.0 |
| AUROC | 0.51 | 0.51 | 0.60 | 0.70 |
| AUPR | 0.367 | 0.367 | 0.385 | 0.493 |
| Specificity | 0.9 | 0.9 | 1.0 | 1.0 |

The 4B model is unchanged by SFT. The 8B and 14B models become *more conservative* — predicting all pairs as negative (hence F1=0 but Specificity=1.0). However, AUROC and AUPR improve with scale (0.51 → 0.60 → 0.70), suggesting the models learn better internal representations even though they fail to express positive predictions.

#### Stability Prediction (20-21 mutations)

| Metric | Vanilla 4B | SFT 4B | SFT 8B | SFT 14B |
|--------|-----------|--------|--------|---------|
| Pearson r | 0.094 | -0.093 | 0.064 | -0.068 |
| MAE | 1.86 | 1.61 | 1.62 | 1.64 |
| RMSE | 2.38 | 1.90 | 2.05 | 1.94 |
| Classification Acc. | 0.35 | 0.24 | 0.52 | 0.24 |
| F1 Macro | 0.240 | 0.155 | 0.236 | 0.167 |
| F1 (Destabilizing) | 0.267 | 0.118 | 0.643 | 0.300 |

Mixed picture. MAE and RMSE improve slightly after SFT (1.86 → 1.61), indicating the model learns to produce more reasonable magnitude estimates. But correlations hover around zero and classification accuracy is inconsistent across scales. The 8B model stands out with notably better destabilizing mutation detection (F1=0.643) and overall classification accuracy (0.52).

## Key Takeaways

### What Works

1. **Architecture is functional.** ESM-3 + Qwen3-4B trains stably at scale, using only ~39 GB of GPU memory for 50K samples. The 20M trainable parameters (0.49%) via attention pooling + projector + LoRA k/v is efficient.

2. **Training converges.** Loss drops consistently across runs (17.35 → 4.08 for 500 samples; 34.25 → 14.77 for 50K). Differential learning rates for the projector (10x base) were critical — without them, the projector couldn't bridge the embedding spaces.

3. **Scale helps internal representations.** AUROC for PPI improves with model size (0.51 → 0.60 → 0.70), and perplexity drops (84 → 76 → 73). The models are learning; they're just not expressing it well.

### What Doesn't Work (Yet)

1. **GO prediction degrades after SFT.** The model goes from over-predicting terms (at least hitting some) to under-predicting (hitting none). The instruction-tuning format may not be well-suited for multi-label structured output.

2. **PPI is stuck at majority-class prediction.** All models prefer "No" regardless of training. The class imbalance in training data (or evaluation prompting) needs addressing.

3. **Stability prediction has no signal.** Correlations near zero means the model isn't learning the relationship between mutations and thermodynamic stability. The raw magnitude estimates improve, but direction prediction remains random.

4. **Text generation quality is low.** BLEU scores below 0.01 and ROUGE-L around 0.12 indicate the generated answers don't closely match references — though this may partly reflect the open-ended nature of protein function description.

## What's Next

These baselines point to clear priorities for the next phase:

1. **GO prediction needs structured output training.** Explore constrained decoding or a classification head for multi-label GO term prediction instead of free-form generation.

2. **PPI needs balanced training.** Investigate class-weighted loss, balanced sampling, or explicit positive-example prompting to overcome the negative bias.

3. **GRPO alignment for reward-guided improvement.** The task-specific reward functions (GO F1, PPI accuracy, stability Gaussian) are already implemented. Once the gradient flow bug is fixed, GRPO should help push these metrics beyond what SFT alone achieves.

4. **Scale the evaluation set.** 10-21 samples per task is too small for reliable conclusions. We need at least 100-500 samples per benchmark for statistical significance.

5. **Try the 8B model as default.** The Qwen3-8B showed the most promising results on stability prediction and has better perplexity. The compute cost is manageable on our 8x H100 setup.

The architecture works. The training pipeline works. Now it's time to make the model actually learn protein biology.
