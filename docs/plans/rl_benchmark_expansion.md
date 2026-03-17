# Implementation Plan: Solubility & Fold Classification GRPO Tasks

> **Date**: 2026-03-16
> **Goal**: Add two new verifiable reward tasks for GRPO training: **solubility prediction** and **fold classification**.
> **Motivation**: Surveyed [Protein-LLM-Survey](https://github.com/Yijia-Xiao/Protein-LLM-Survey) benchmarks; selected tasks with <0.5% SFT data contamination and clear verifiable labels.
> **Trial run**: text SFT model → GRPO, 50 steps, 2+ evaluations, verify reward signal changes over training.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Task 1: Solubility Prediction](#2-task-1-solubility-prediction)
3. [Task 2: Fold Classification](#3-task-2-fold-classification)
4. [Implementation Steps](#4-implementation-steps)
5. [File Change Summary](#5-file-change-summary)
6. [Trial Run Plan](#6-trial-run-plan)
7. [Acceptance Criteria](#7-acceptance-criteria)

---

## 1. Overview

### Current GRPO Tasks

| Task | Reward Function | Data Source | Labels |
|------|----------------|-------------|--------|
| GO prediction | `compute_go_reward` | CAFA5 | GO terms (F1) |
| Stability (ddG) | `compute_stability_reward` | MegaScale | ddG float + class (3-way) |
| Structure quality | `compute_esmfold_reward` | AlphaFoldDB | pLDDT + quality class |
| PPI | `compute_ppi_reward` | — | Binary yes/no |

### New Tasks to Add

| Task | SFT Contamination | Label Type | Reward Design |
|------|-------------------|------------|---------------|
| **Solubility** | <0.2% (excellent) | Binary (soluble/insoluble) + continuous score | Classification + regression |
| **Fold Classification** | <0.5% (clean) | Hierarchical CATH labels (4 levels) | Exact match + partial credit |

---

## 2. Task 1: Solubility Prediction

### Data Source

**Primary**: [eSOL database](https://www.tanpaku.org/tp-esol/) — experimental solubility data for E. coli proteins.
- ~4,000 proteins with measured solubility (0-100% scale)
- Binary classification: soluble (>50%) vs insoluble (≤50%)
- Alternative: [PSI:Biology solubility](https://doi.org/10.1038/s41587-019-0378-2) or [ProteinSol](https://protein-sol.manchester.ac.uk/)

**Fallback/supplement**: UniProt annotations with "solubility" keyword, or DeepSol benchmark data.

### JSON Format (matches existing pattern)

```json
{
  "instruction": "Predict the solubility of this protein when expressed in E. coli. Report whether it is soluble or insoluble, and estimate the solubility percentage (0-100%).",
  "input": "MKTLLLTLLAVVLAAGCAQKEDISRLG...",
  "output": "Solubility: soluble. Estimated solubility: 72.3%. This protein is likely soluble when expressed in E. coli.",
  "metadata": {
    "solubility_score": 72.3,
    "solubility_class": "soluble",
    "protein_id": "P12345",
    "source": "eSOL",
    "organism": "E. coli"
  }
}
```

### Reward Function Design: `compute_solubility_reward`

Three components (mirrors stability reward pattern):

| Component | Weight | Logic |
|-----------|--------|-------|
| **Classification** | 0.5 | Exact match: soluble/insoluble (binary) |
| **Numerical accuracy** | 0.3 | Gaussian decay on \|predicted% - true%\|, σ=15 |
| **Format compliance** | 0.2 | Correct output structure (solubility keyword + number) |

```python
def compute_solubility_reward(
    generated_text: str,
    ground_truth: Union[str, float, Dict[str, Any]],
    tolerance: float = 15.0,  # σ for Gaussian, in % points
    detailed: bool = False,
    focal_gamma: float = 0.0,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
```

**Parsing logic**:
- Extract predicted class: keywords "soluble"/"insoluble" (watch for "insoluble" containing "soluble")
- Extract predicted score: patterns like `solubility: 72.3%`, `estimated solubility: 72%`, or any number 0-100
- Ground truth: parse from JSON metadata dict or plain float

**Focal weighting**: if class distribution is skewed (e.g., 70% soluble / 30% insoluble), apply `_focal_weight()` with `_SOLUBILITY_CLASS_FREQ`.

### System Prompt

```python
_SOLUBILITY_SYSTEM_PROMPT_THINK = (
    "You are a protein solubility expert. Given a protein amino acid sequence, "
    "predict its solubility when expressed in E. coli.\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags about "
    "the protein's solubility-related features, then give your prediction inside "
    "<answer>...</answer> tags. Keep your thinking SHORT.\n\n"
    "Examples:\n"
    "<think>Small protein, no transmembrane domains, balanced charge. Likely soluble.</think>\n"
    "<answer>Solubility: soluble. Estimated solubility: 78.5%.</answer>\n\n"
    "<think>Large hydrophobic patches, many cysteines, membrane-associated. Likely insoluble.</think>\n"
    "<answer>Solubility: insoluble. Estimated solubility: 12.3%.</answer>"
)

_SOLUBILITY_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein solubility expert. Given a protein amino acid sequence, "
    "predict its solubility when expressed in E. coli.\n\n"
    "First write a brief reasoning about the protein's features on a line "
    "starting with \"Reasoning:\", then give your prediction inside "
    "<answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: Small protein, no transmembrane domains, balanced charge. Likely soluble.\n"
    "<answer>Solubility: soluble. Estimated solubility: 78.5%.</answer>\n\n"
    "Reasoning: Large hydrophobic patches, many cysteines, membrane-associated. Likely insoluble.\n"
    "<answer>Solubility: insoluble. Estimated solubility: 12.3%.</answer>"
)
```

---

## 3. Task 2: Fold Classification

### Data Source

**Primary**: [SCOPe (Structural Classification of Proteins — extended)](https://scop.berkeley.edu/)
- Hierarchical 4-level classification: Class → Fold → Superfamily → Family
- ~300K domain entries
- 7 major classes: all-alpha, all-beta, alpha/beta, alpha+beta, multi-domain, membrane, small proteins

**Alternative**: [CATH database](https://www.cathdb.info/)
- 4 levels: Class → Architecture → Topology → Homology (C.A.T.H)
- ~500K domain entries
- Well-maintained, clear labels

**Recommendation**: Use **CATH** for cleaner hierarchical labels and larger dataset. Download via CATH API or pre-built flat files.

### JSON Format

```json
{
  "instruction": "Classify the structural fold of this protein. Report the CATH classification at all four levels: Class (C), Architecture (A), Topology (T), and Homology (H).",
  "input": "MNIFEMLRIDEGLRLKIYKDTEGYYTI...",
  "output": "CATH classification: 1.10.490.10. Class: Mainly Alpha. Architecture: Orthogonal Bundle. Topology: Globin-like. Homology: Globin.",
  "metadata": {
    "cath_code": "1.10.490.10",
    "class": "1",
    "class_name": "Mainly Alpha",
    "architecture": "10",
    "architecture_name": "Orthogonal Bundle",
    "topology": "490",
    "topology_name": "Globin-like",
    "homology": "10",
    "homology_name": "Globin",
    "domain_id": "1a00A00",
    "pdb_id": "1a00",
    "source": "CATH"
  }
}
```

### Reward Function Design: `compute_fold_classification_reward`

Hierarchical partial credit (4 levels, each worth 0.25):

| Level | Weight | Match Logic |
|-------|--------|-------------|
| **Class (C)** | 0.25 | Exact match on first level (e.g., "1" or "Mainly Alpha") |
| **Architecture (A)** | 0.25 | Exact match on C.A (e.g., "1.10") |
| **Topology (T)** | 0.25 | Exact match on C.A.T (e.g., "1.10.490") |
| **Homology (H)** | 0.25 | Exact match on full C.A.T.H (e.g., "1.10.490.10") |

Total reward = sum of matched levels. Partial credit rewards the model for getting coarser levels right even if fine-grained is wrong.

```python
# CATH class name mapping for text-based matching
_CATH_CLASS_NAMES = {
    "1": "mainly alpha",
    "2": "mainly beta",
    "3": "alpha beta",       # alpha/beta (mixed)
    "4": "few secondary structures",
    "5": "special",          # coiled-coil, membrane, etc.
}

def compute_fold_classification_reward(
    generated_text: str,
    ground_truth: Union[str, Dict[str, Any]],
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
```

**Parsing logic**:
- Extract CATH code: regex `r"(\d+\.\d+\.\d+\.\d+)"` from generated text
- Fallback: match class names ("mainly alpha", "all beta", etc.)
- Compare level by level against ground truth CATH code
- Ground truth: parse from JSON metadata dict or dotted string

### System Prompt

```python
_FOLD_CLASSIFICATION_SYSTEM_PROMPT_THINK = (
    "You are a protein structure classification expert. Given a protein amino "
    "acid sequence, predict its CATH structural classification at four levels: "
    "Class (C), Architecture (A), Topology (T), and Homology (H).\n\n"
    "CATH classes: 1=Mainly Alpha, 2=Mainly Beta, 3=Alpha Beta, "
    "4=Few Secondary Structures.\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags about "
    "the protein's structural features, then give the CATH code inside "
    "<answer>...</answer> tags. Keep your thinking SHORT.\n\n"
    "Examples:\n"
    "<think>Helical bundle pattern, globin-like fold. Class 1, orthogonal bundle.</think>\n"
    "<answer>CATH: 1.10.490.10. Class: Mainly Alpha. Architecture: Orthogonal Bundle. "
    "Topology: Globin-like. Homology: Globin.</answer>\n\n"
    "<think>Beta sandwich with immunoglobulin topology. Class 2.</think>\n"
    "<answer>CATH: 2.60.40.10. Class: Mainly Beta. Architecture: Sandwich. "
    "Topology: Immunoglobulin-like. Homology: Immunoglobulin.</answer>"
)

_FOLD_CLASSIFICATION_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein structure classification expert. Given a protein amino "
    "acid sequence, predict its CATH structural classification at four levels: "
    "Class (C), Architecture (A), Topology (T), and Homology (H).\n\n"
    "CATH classes: 1=Mainly Alpha, 2=Mainly Beta, 3=Alpha Beta, "
    "4=Few Secondary Structures.\n\n"
    "First write a brief reasoning about the protein's structural features on a line "
    "starting with \"Reasoning:\", then give the CATH code inside "
    "<answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: Helical bundle pattern, globin-like fold. Class 1, orthogonal bundle.\n"
    "<answer>CATH: 1.10.490.10. Class: Mainly Alpha. Architecture: Orthogonal Bundle. "
    "Topology: Globin-like. Homology: Globin.</answer>\n\n"
    "Reasoning: Beta sandwich with immunoglobulin topology. Class 2.\n"
    "<answer>CATH: 2.60.40.10. Class: Mainly Beta. Architecture: Sandwich. "
    "Topology: Immunoglobulin-like. Homology: Immunoglobulin.</answer>"
)
```

---

## 4. Implementation Steps

### Step 1: Download Scripts (NEW files)

#### `scripts/data/download_solubility.py`

```
1. Load eSOL data from HuggingFace or direct download
   - Primary: https://www.tanpaku.org/tp-esol/ CSV
   - Alternative: HuggingFace dataset if available
2. Filter: require valid sequence + solubility score
3. Classify: soluble (>50%) / insoluble (≤50%)
4. Convert to instruction JSON format (see §2 JSON Format above)
5. Split: 90/5/5 train/val/test
6. Save to data/processed/solubility_dataset/solubility.json
7. Log statistics: class distribution, sequence length stats
```

**CLI**:
```bash
python scripts/data/download_solubility.py --max_samples 10000
python scripts/data/download_solubility.py --output_dir data/processed/solubility_dataset
```

#### `scripts/data/download_fold_classification.py`

```
1. Load CATH domain list from CATH API or flat file
   - http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/
   - cath-domain-list.txt: domain_id, class, arch, top, homol
   - Need to fetch sequences: PDB files or pre-mapped UniProt sequences
2. Filter: require valid sequence + complete CATH code (all 4 levels)
3. Map CATH codes to human-readable names (from cath-names.txt)
4. Convert to instruction JSON format (see §3 JSON Format above)
5. Split: 90/5/5 train/val/test (stratified by Class to ensure coverage)
6. Save to data/processed/fold_classification/fold_classification.json
7. Log statistics: class distribution across all 4 CATH levels
```

**CLI**:
```bash
python scripts/data/download_fold_classification.py --max_samples 10000
python scripts/data/download_fold_classification.py --output_dir data/processed/fold_classification
```

---

### Step 2: Reward Functions (`src/training/rewards.py`)

Add to file (follow existing pattern from `compute_stability_reward`):

1. **Constants** (top of file):
   ```python
   # Solubility class frequencies (approximate, from eSOL)
   _SOLUBILITY_CLASS_FREQ = {"soluble": 0.70, "insoluble": 0.30}
   ```

2. **`compute_solubility_reward()`** (~100 lines):
   - Parse ground truth (dict with solubility_score + solubility_class, or float, or string)
   - Parse prediction: extract "soluble"/"insoluble" keyword + numeric % from text
   - Compute 3 components: classification (0.5) + numerical (0.3) + format (0.2)
   - Optional focal weighting via `_focal_weight()`
   - Return `(reward, metrics_dict)` when `detailed=True`

3. **`compute_fold_classification_reward()`** (~100 lines):
   - Parse ground truth CATH code from dict or string (e.g., "1.10.490.10")
   - Parse prediction: extract CATH code via regex, or match class names
   - Compute hierarchical reward: 0.25 per matching level (C, A, T, H)
   - Return `(reward, metrics_dict)` with per-level match booleans

4. **Registry** (in `get_reward_function()`):
   ```python
   # Add to reward_functions dict:
   "solubility": compute_solubility_reward,
   "solubility_prediction": compute_solubility_reward,
   "fold_classification": compute_fold_classification_reward,
   "fold_class": compute_fold_classification_reward,
   "cath": compute_fold_classification_reward,
   "cath_classification": compute_fold_classification_reward,
   ```

---

### Step 3: GRPO Trainer Integration (`src/training/grpo_trainer.py`)

#### A. System Prompts (after line ~319)

Add the 4 prompt constants defined in §2 and §3 above.

#### B. Prompt Routing (in `_get_grpo_system_prompt()`, after line ~346)

```python
if task in ("solubility", "solubility_prediction"):
    return _SOLUBILITY_SYSTEM_PROMPT_THINK if enable_thinking else _SOLUBILITY_SYSTEM_PROMPT_NO_THINK
if task in ("fold_classification", "fold_class", "cath", "cath_classification"):
    return _FOLD_CLASSIFICATION_SYSTEM_PROMPT_THINK if enable_thinking else _FOLD_CLASSIFICATION_SYSTEM_PROMPT_NO_THINK
```

#### C. Reward Setup (in `_setup_reward_function()`, after line ~1508)

```python
solubility_tasks = {"solubility", "solubility_prediction"}
self._is_solubility_reward = task_normalized in solubility_tasks

fold_tasks = {"fold_classification", "fold_class", "cath", "cath_classification"}
self._is_fold_reward = task_normalized in fold_tasks
```

Add focal logging for solubility if enabled (same pattern as stability, line ~1530).

#### D. Ground Truth Extraction (in `_log_probe_completions()`, after line ~1270)

```python
elif task in ("solubility", "solubility_prediction") and isinstance(metadata, dict) and "solubility_score" in metadata:
    ground_truth = json.dumps({
        "solubility_score": metadata.get("solubility_score", 0),
        "solubility_class": metadata.get("solubility_class"),
    })
elif task in ("fold_classification", "fold_class", "cath") and isinstance(metadata, dict) and "cath_code" in metadata:
    ground_truth = json.dumps({
        "cath_code": metadata.get("cath_code"),
        "class_name": metadata.get("class_name"),
        "architecture_name": metadata.get("architecture_name"),
        "topology_name": metadata.get("topology_name"),
        "homology_name": metadata.get("homology_name"),
    })
```

#### E. Reward Computation (in `_compute_rewards()`, after line ~1983)

```python
# Solubility-specific kwargs
if self._is_solubility_reward and self._focal_gamma > 0:
    reward_kwargs["focal_gamma"] = self._focal_gamma
```

No special routing needed for solubility or fold — both use standard `ground_truth` (not `protein_sequences`).

---

### Step 4: Config Files (NEW files)

#### `configs/data/solubility_dataset.yaml`

```yaml
name: solubility_dataset
source: eSOL/custom
task: solubility

paths:
  raw: ${paths.raw_dir}/solubility_dataset
  processed: ${paths.processed_dir}/solubility_dataset

processing:
  max_seq_length: 2048
  include_structure: false

splits:
  train: 0.9
  validation: 0.05
  test: 0.05

limit: null
```

#### `configs/data/fold_classification.yaml`

```yaml
name: fold_classification
source: CATH
task: fold_classification

paths:
  raw: ${paths.raw_dir}/fold_classification
  processed: ${paths.processed_dir}/fold_classification

processing:
  max_seq_length: 2048
  include_structure: false

splits:
  train: 0.9
  validation: 0.05
  test: 0.05

limit: null
```

#### `configs/main_RL/grpo_solubility.yaml`

```yaml
# @package _global_

# GRPO: Solubility Prediction (Text SFT -> GRPO)
# Binary classification (soluble/insoluble) + regression (0-100%)
# Reward: classification (0.5) + numerical (0.3) + format (0.2)
#
# Usage:
#   python scripts/train.py main_RL=grpo_solubility

defaults:
  - override /model: qwen3_8b
  - override /encoder: esm3_small
  - override /data: solubility_dataset
  - override /training: grpo

approach: text  # Start with text for trial
experiment_name: grpo_solubility_${model.name}_${now:%m%d_%H%M%S}

# Chain from text SFT
parent_experiment: sft_text_combined_qwen3_8b_it_0307_190324

data:
  task: solubility
  processing:
    max_protein_length: 512

training:
  lr: 5e-6
  projector_lr: 2.5e-5
  batch_size: 2
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  epochs: 1
  warmup_steps: 10
  max_grad_norm: 1.0

  grpo:
    group_size: 8
    temperature: 0.9
    use_kl_penalty: false
    normalize_advantages: false
    focal_enabled: true
    focal_gamma: 2.0

  rollout:
    max_tokens: 128  # Solubility answers are short
    top_p: 0.95
    do_sample: true
    enable_thinking: false

  save_steps: 25
  eval_steps: 25
  logging_steps: 5

  fsdp:
    enabled: false

  wandb:
    tags:
      - grpo
      - solubility
      - text_sft
      - ${model.name}
```

#### `configs/main_RL/grpo_fold_classification.yaml`

```yaml
# @package _global_

# GRPO: Fold Classification via CATH (Text SFT -> GRPO)
# Hierarchical 4-level classification with partial credit
# Reward: 0.25 per correct CATH level (C, A, T, H)
#
# Usage:
#   python scripts/train.py main_RL=grpo_fold_classification

defaults:
  - override /model: qwen3_8b
  - override /encoder: esm3_small
  - override /data: fold_classification
  - override /training: grpo

approach: text  # Start with text for trial
experiment_name: grpo_fold_class_${model.name}_${now:%m%d_%H%M%S}

# Chain from text SFT
parent_experiment: sft_text_combined_qwen3_8b_it_0307_190324

data:
  task: fold_classification
  processing:
    max_protein_length: 512

training:
  lr: 5e-6
  projector_lr: 2.5e-5
  batch_size: 2
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  epochs: 1
  warmup_steps: 10
  max_grad_norm: 1.0

  grpo:
    group_size: 8
    temperature: 0.9
    use_kl_penalty: false
    normalize_advantages: false
    focal_enabled: false  # CATH classes are more balanced

  rollout:
    max_tokens: 256  # CATH classification needs more space
    top_p: 0.95
    do_sample: true
    enable_thinking: false

  save_steps: 25
  eval_steps: 25
  logging_steps: 5

  fsdp:
    enabled: false

  wandb:
    tags:
      - grpo
      - fold_classification
      - cath
      - text_sft
      - ${model.name}
```

---

### Step 5: Documentation Updates

#### `CLAUDE.md` — Add to Quick Reference:

```bash
# Downstream task data (add these lines)
python scripts/data/download_solubility.py --max_samples 10000        # Solubility
python scripts/data/download_fold_classification.py --max_samples 10000  # Fold (CATH)

# GRPO with new tasks (add these lines)
python scripts/train.py main_RL=grpo_solubility parent_experiment=my_sft
python scripts/train.py main_RL=grpo_fold_classification parent_experiment=my_sft
```

---

## 5. File Change Summary

| File | Action | Lines (est.) | Priority |
|------|--------|-------------|----------|
| `scripts/data/download_solubility.py` | **NEW** | ~250 | P0 |
| `scripts/data/download_fold_classification.py` | **NEW** | ~300 | P0 |
| `src/training/rewards.py` | **EDIT** | +200 (2 functions + registry) | P0 |
| `src/training/grpo_trainer.py` | **EDIT** | +80 (prompts + routing + setup) | P0 |
| `configs/data/solubility_dataset.yaml` | **NEW** | ~20 | P1 |
| `configs/data/fold_classification.yaml` | **NEW** | ~20 | P1 |
| `configs/main_RL/grpo_solubility.yaml` | **NEW** | ~55 | P1 |
| `configs/main_RL/grpo_fold_classification.yaml` | **NEW** | ~55 | P1 |
| `CLAUDE.md` | **EDIT** | +6 | P2 |

**Total**: 4 new files, 3 edited files, ~1000 new lines

---

## 6. Trial Run Plan

### Goal
Verify reward signal changes over 50 training steps with text SFT model.

### Setup
- **Base model**: `sft_text_combined_qwen3_8b_it_0307_190324` (text SFT)
- **Steps**: 50
- **Evaluations**: at step 25 and step 50 (eval_steps=25)
- **Logging**: every 5 steps

### Commands

```bash
# 1. Download data
python scripts/data/download_solubility.py --max_samples 10000
python scripts/data/download_fold_classification.py --max_samples 10000

# 2. Run trial: Solubility
python scripts/train.py main_RL=grpo_solubility \
  training.epochs=1 \
  training.save_steps=25 training.eval_steps=25 \
  experiment_name=grpo_solubility_trial_0316

# 3. Run trial: Fold Classification
python scripts/train.py main_RL=grpo_fold_classification \
  training.epochs=1 \
  training.save_steps=25 training.eval_steps=25 \
  experiment_name=grpo_fold_class_trial_0316
```

### Success Criteria

| Metric | Step 0 (baseline) | Step 50 (target) | Pass? |
|--------|-------------------|------------------|-------|
| **Solubility reward** | ~0.2-0.3 (random) | >0.4 (some learning) | Reward increases |
| **Fold class reward** | ~0.0-0.1 (random) | >0.15 (at least class-level) | Reward increases |
| **Format compliance** | ~0% (no \<answer\> tags) | >50% | Model learns format |
| **No NaN/divergence** | — | No NaN loss | Training stable |
| **Eval reward** | — | Moves in same direction as train | Not overfitting |

### What to Monitor in wandb
- `reward` (mean per step) — should trend upward
- `format_bonus` — should increase (model learns \<answer\> tags)
- `grad_norm` — should stay bounded (no explosions)
- `eval_reward` — should also increase (generalization)
- For solubility: `class_correct`, `mae`
- For fold: `class_match`, `architecture_match`, `topology_match`, `homology_match`

---

## 7. Acceptance Criteria

### Must Have
- [ ] Download scripts produce valid JSON in correct format
- [ ] Reward functions return values in [0, 1] range
- [ ] Reward functions handle edge cases (empty text, malformed output, missing fields)
- [ ] System prompts include examples matching expected output format
- [ ] GRPO configs chain correctly from text SFT checkpoint
- [ ] Trial run completes 50 steps without errors
- [ ] Reward signal shows improvement over 50 steps

### Should Have
- [ ] Focal weighting for imbalanced solubility classes
- [ ] Hierarchical partial credit for fold classification
- [ ] `detailed=True` returns per-component metrics for wandb logging
- [ ] Class distribution statistics logged at startup

### Nice to Have
- [ ] Standalone evaluation modules in `src/evaluation/`
- [ ] ESM3+MLP configs (in addition to text-only trial)
- [ ] Data quality validation (sequence length distribution, label distribution)
