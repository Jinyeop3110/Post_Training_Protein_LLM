---
title: "Protein Boundary Tokens, Instruct Model Fix, and Full Pipeline Validation"
date: 2026-02-24
author: yeopjin
tags: [architecture, training, debug, milestone]
---

# Protein Boundary Tokens, Instruct Model Fix, and Full Pipeline Validation

Today's session addressed two critical issues and validated the entire pipeline end-to-end. We (1) fixed a model config bug where the 8B training run used a base model instead of Instruct, (2) introduced structured protein boundary tokens for the ESM-3 embedding path, and (3) ran comprehensive validation across all task types, reward functions, and configs.

## 1. The Instruct Model Bug

### Discovery

While reviewing the latest training run (`sft_lora_esm3_qwen3_8b_0224_011654`), we noticed degenerate generation at every evaluation step — outputs like `KQKQKQKQ...`, `SSSSSSSS...`, or empty strings. The model had been training for 5000 steps (epoch 2.27/3) with a reasonable loss curve:

| Step | Train Loss | Eval Loss |
|------|-----------|-----------|
| 250  | 8.30      | 2.09      |
| 1000 | 4.80      | 1.89      |
| 5000 | 3.86      | 1.84      |

The eval loss looked fine, but generation was garbage. The root cause: `configs/model/qwen3_8b.yaml` had `path: Qwen/Qwen3-8B` — the **base** model, not `Qwen/Qwen3-8B-Instruct-2507`.

### Why Base vs Instruct Matters

Using a base model for instruction-tuned training is harmful at two levels:

1. **Training waste**: The model must learn basic instruction-following from scratch (chat templates, turn-taking, stop tokens), consuming LoRA capacity that should go toward protein understanding.

2. **Generation failure**: The base model never learned to stop generating or follow the chat template's turn structure, causing repetitive loops. The LoRA adapter is bound to the base model's weight space and **cannot be transferred** to the Instruct variant.

The checkpoint is not salvageable — the entire run must be re-done.

### Fix

All model configs now use `-it` suffix to distinguish from base models:

| Config | `name` (before) | `name` (after) | `path` |
|--------|-----------------|-----------------|--------|
| qwen3_4b.yaml | `qwen3_4b` | `qwen3_4b_it` | Qwen/Qwen3-4B-Instruct-2507 (unchanged) |
| qwen3_8b.yaml | `qwen3_8b` | `qwen3_8b_it` | **Qwen/Qwen3-8B-Instruct-2507** (FIXED) |
| qwen3_14b.yaml | `qwen3_14b` | `qwen3_14b_it` | Qwen/Qwen3-14B-Instruct-2507 (unchanged) |
| llama3_8b.yaml | `llama3_8b` | `llama3_8b_it` | meta-llama/Llama-3.1-8B-Instruct (unchanged) |

Auto-generated experiment names now include `-it`: `sft_qlora_esm3_qwen3_8b_it_0224_HHMMSS`.


## 2. Protein Boundary Tokens

### Motivation

Previously, the ESM-3 embedding path used a single `<|protein_embed|>` placeholder token. The LLM had no explicit signal for where the protein representation starts and ends. We added structured boundary tokens so the LLM can learn to attend to protein embeddings as a distinct modality.

### Design

Three special tokens, registered in the tokenizer:

```
<|protein_start|>  <|protein_embed|>  <|protein_end|>
     ID 151669         ID 151670          ID 151671
```

At training/inference time, the full placeholder `<|protein_start|><|protein_embed|><|protein_end|>` is embedded as:

```
[..., text_tokens, start_embed, prot_1, prot_2, ..., prot_32, end_embed, text_tokens, ...]
                   ↑ regular LLM   ↑ 32 ESM-3 pooled tokens ↑  regular LLM
                     embedding       (replaced at forward)      embedding
```

- `<|protein_start|>` and `<|protein_end|>` remain as regular LLM token embeddings — the model learns what they mean through training.
- `<|protein_embed|>` is replaced at forward time by the 32 pooled ESM-3 embeddings.
- All three get `label = -100` in training — no cross-entropy loss on boundary or embedding tokens.

### Implementation

Changes across 4 files:
- `src/models/multimodal_llm.py`: Constants `PROTEIN_START_TOKEN`, `PROTEIN_EMBED_TOKEN`, `PROTEIN_END_TOKEN`, `PROTEIN_PLACEHOLDER`, `PROTEIN_SPECIAL_TOKENS`; `prepare_inputs()` searches only for `PROTEIN_EMBED_TOKEN` to replace.
- `src/training/sft_trainer.py`: `_load_tokenizer()` registers all 3 tokens via `add_special_tokens`.
- `src/training/grpo_trainer.py`: `_load_tokenizer()` now also registers all 3 tokens (was missing entirely before).


## 3. Task-by-Task: Example Prompts, Inputs, Answers

Here is what each task looks like in the pipeline, with actual data examples and how the loss/reward is computed.

### SFT Tasks (Mol-Instructions, 505K total)

All SFT tasks use the Qwen3 chat template with the protein expert system prompt. Loss is standard **causal LM cross-entropy on assistant tokens only** — instruction/system/protein embedding tokens are masked with `label = -100`.

#### Catalytic Activity Prediction (53K samples)

```
SYSTEM: You are a protein science expert...
USER:   Please evaluate the following protein sequence and provide an explanation
        of the enzyme's catalytic activity...

        <|protein_start|><|protein_embed|><|protein_end|>

ASSISTANT: By examining the input protein sequence, the enzyme catalyzes the
           subsequent chemical reaction: ATP + meso-2,6-diaminoheptanedioate =
           ADP + H+ + meso-2,6-diaminopimelate...
```

**Loss**: Cross-entropy on assistant tokens. The model learns to describe enzyme reactions from structural embeddings.

#### Protein Function Prediction (114K samples)

```
USER:   Using the given protein sequence, predict its functional role and the
        potential biological pathway it may be a part of:

        <|protein_start|><|protein_embed|><|protein_end|>

ASSISTANT: The analysis suggests its potential function as ATP binding, ligase
           activity, magnesium ion binding...
```

#### General Function Description (87K samples)

```
USER:   Inspect the protein with the subsequent sequence and offer a concise
        description of its properties:

        <|protein_start|><|protein_embed|><|protein_end|>

ASSISTANT: Catalyzes the NADPH-dependent formation of L-pipecolic acid from
           delta-1-piperideine-6-L-carboxylate (P6C)...
```

#### Domain/Motif Prediction (45K samples)

```
USER:   Find and list any domains or motifs that are likely present in this
        protein sequence:

        <|protein_start|><|protein_embed|><|protein_end|>

ASSISTANT: The prediction suggests the presence of these domains and motifs:
           Pkinase domain, ATP-binding region...
```

#### Protein Design (196K samples — largest category)

```
USER:   Synthesize a protein sequence that has the desired solubility or other
        physical properties for the intended use.

        1. The protein must exhibit the following characteristics: Also acts as
           a cofactor with GATA4, a key cardiac regulator...

ASSISTANT: Here's a protein sequence with the above functions:
           MIGMLESLQHESDLLQHDQIHTGEKPYECNECRKTFSL...
```

**Note**: Protein design is the only task where the protein sequence is in the *output*, not input. No ESM-3 encoding is used for the input.

### GRPO Downstream Tasks (10K samples each)

GRPO tasks use **verifiable reward functions** instead of cross-entropy. The model generates multiple completions per prompt, rewards are computed, and group-relative advantages guide policy optimization.

#### GO Term Prediction (CAFA5, 10K samples)

```
USER:   Predict the Gene Ontology (GO) terms for this protein across Molecular
        Function (MF), Biological Process (BP), and Cellular Component (CC):

        <|protein_start|><|protein_embed|><|protein_end|>

GROUND TRUTH: GO:0003674, GO:0005488, GO:0005515, GO:0019955, GO:0019956, GO:0019957
```

**Reward**: F1 score between predicted and ground truth GO terms (regex `GO:\d{7}`).
- Partial match (2/3 correct): reward = 0.80
- Exact match: reward = 1.00
- No match: reward = 0.00

#### Stability/ddG Prediction (MegaScale, 10K samples)

```
USER:   Predict the change in protein stability (ddG in kcal/mol) for this
        mutation. Classify as stabilizing (<-1.0), neutral (-1.0 to 1.0), or
        destabilizing (>1.0).

        <|protein_start|><|protein_embed|><|protein_end|>

GROUND TRUTH: ddG = 0.03 kcal/mol. This mutation is neutral.
METADATA:     {"ddG": 0.03, "mutation": "G22I", "stability_class": "neutral"}
```

**Reward**: Gaussian decay on prediction error: `R = exp(-error^2 / 2sigma^2)` with sigma=1.0 kcal/mol.
- Error of 0.2 kcal/mol: reward = 0.98
- Error of 1.0 kcal/mol: reward = 0.61
- No parseable number: reward = 0.00

#### Structure Quality Assessment (AlphaFold DB, 10K samples)

```
USER:   Assess the structural quality of this protein. Report the predicted fold
        quality (high/medium/low), estimated pLDDT (0-100), and whether the
        protein is well-folded or likely disordered.

        <|protein_start|><|protein_embed|><|protein_end|>

GROUND TRUTH: Fold quality: high. pLDDT: 88.4. This protein is well-folded
              with high confidence.
METADATA:     {"plddt": 88.44, "fold_category": "high", "source": "alphafold"}
```

**Reward**: Three components summing to max 1.0:
1. **Quality alignment** (0.4): "well-folded" claim matches pLDDT > 70? (or "disordered" matches pLDDT < 50?)
2. **Numerical pLDDT accuracy** (0.3): Gaussian on `|predicted - actual|` with sigma=10
3. **Category match** (0.3): high/medium/low matches pLDDT threshold

Example: claiming "well-folded" + predicting pLDDT=85 for actual 88.4 = 0.4 + 0.29 + 0.3 = 0.99.


## 4. Pipeline Validation Results

We ran 4 independent test agents covering the full pipeline. All passed.

### Agent 1: Data Pipeline & Prompt Formatting (4/4 PASS)

| Test | Status |
|------|--------|
| MolInstructions ESM-3 approach (boundary tokens in prompt) | PASS |
| MolInstructions text approach (raw sequence, no tokens) | PASS |
| Tokenization roundtrip (3 tokens → 3 IDs → 3 tokens) | PASS |
| DataCollator label masking (49% masked = prompt, 51% = response) | PASS |

### Agent 2: Evaluation Pipelines (5/5 PASS)

| Test | Status | Key Metrics |
|------|--------|-------------|
| GO prediction parsing + metrics | PASS | fmax=0.737, f1_micro=0.75 |
| PPI prediction parsing + metrics | PASS | 18 metrics including MCC, AUC |
| Stability parsing + metrics | PASS | Pearson=0.997, RMSE=0.24 |
| SFT evaluation module | PASS | perplexity + BLEU + ROUGE-L |
| Benchmarks runner | PASS | Orchestrates all 4 tasks |

### Agent 3: GRPO Rewards & Downstream Tasks (4/4 PASS)

| Test | Status |
|------|--------|
| 4 reward functions (GO, PPI, stability, ESMFold) | PASS |
| 3 downstream datasets loadable | PASS |
| Boundary tokens in all downstream data | PASS |
| ESMFold reward (pre-computed mode) | PASS |

### Agent 4: Config & Inference Consistency (7/7 PASS)

| Test | Status |
|------|--------|
| Default config resolves (qwen3_4b_it) | PASS |
| 8B config resolves (qwen3_8b_it, Instruct path) | PASS |
| Experiment name includes `-it` suffix | PASS |
| Protein token constants importable | PASS |
| Vanilla LLM (text path) doesn't use boundary tokens | PASS |
| All 9 experiment configs resolve | PASS |
| evaluate.py Hydra resolution | PASS |

**Total: 20/20 tests passed, 0 failures, 0 code changes needed.**


## 5. Code Review Findings (From Earlier Session)

A parallel code review identified issues to address in the next session:

| Severity | Issue | File |
|----------|-------|------|
| Critical | `DEFAULT_LORA_TARGET_MODULES` only has k/v (should be all 7) | multimodal_llm.py |
| Critical | `DEFAULT_LLM_NAME` uses base model variant | multimodal_llm.py |
| Warning | `torch.load()` without `weights_only=True` (5 locations) | multiple |
| Warning | GRPO `_load_protein_llm` missing perceiver params | grpo_trainer.py |
| Info | `from_config` defaults perceiver_layers=6 (should be 2) | multimodal_llm.py |
| Info | Code duplication between SFT and GRPO trainers | trainers |

These are tracked and will be addressed before the next training run.


## What's Next

1. **Re-run 8B training** with the fixed Instruct model and boundary tokens
2. **Fix the code review issues** (LoRA target modules, torch.load security)
3. **Launch GRPO** on downstream tasks (GO prediction first, as it has the clearest reward signal)
4. **Begin Perceiver Resampler comparison** — the MLP vs Perceiver ablation is the core thesis experiment
