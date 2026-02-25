# Downstream Tasks Implementation Plan

> **Purpose:** Actionable blueprint for Claude Code agent teams to implement RL reward
> functions AND evaluation benchmarks for each downstream task.
>
> **Context:** The GRPO trainer infrastructure (`src/training/grpo_trainer.py`) is complete.
> The reward functions exist but have **no matching datasets**. This plan fills that gap.

---

## Overview

| Task | Stage | Status | Priority |
|------|-------|--------|----------|
| **GO Term Prediction** | RL + Eval | To implement | P0 |
| **PPI Prediction** | RL + Eval | Future TODO | P2 |
| **Protein Stability (ddG)** | RL + Eval | To implement | P0 |
| **Structural Quality (ESMFold)** | RL + Eval | To implement | P1 |

---

## Task 1: GO Term Prediction

### 1.1 Data Sources (pick one primary, one eval-only)

| Resource | Use For | Proteins | GO Terms | Access |
|----------|---------|----------|----------|--------|
| **CAFA 5/6** | **Primary training + RL** | ~142K (CAFA5), ~82K (CAFA6) | 31K+ | Kaggle: `cafa-6-protein-function-prediction`, HuggingFace: `AmelieSchreiber/cafa_5_protein_function_prediction` |
| **GO Bench** | **Standard eval benchmark** | 567K | Configurable | gobench.org, cluster50 split (prevents homology leakage) |
| **DeepGO2 data** | Alternative training | Large | Curated | github.com/bio-ontology-research-group/deepgo2 |
| PAD | Large-scale pretraining (if needed) | 36M raw | 13.9K labels | kornmann.bioch.ox.ac.uk/jang/services/pad/ |

**Decision: Use CAFA 5 data for RL training, GO Bench for evaluation.**

### 1.2 Data Format

CAFA provides two files:
```
train_sequences.fasta    # >protein_id \n SEQUENCE
train_terms.tsv          # protein_id \t GO:XXXXXXX  (one row per annotation)
```

We need to convert to our JSON format:
```json
{
  "instruction": "Predict the Gene Ontology (GO) terms for this protein. List molecular functions (MF), biological processes (BP), and cellular components (CC). Format: GO:XXXXXXX",
  "input": "MKTAYIAKQRQISFVKSH...",
  "output": "GO:0005524, GO:0004713, GO:0016740, GO:0005886",
  "metadata": {
    "protein_id": "P00533",
    "go_aspect": ["MF", "MF", "MF", "CC"],
    "source": "cafa5"
  }
}
```

### 1.3 Metrics (CAFA Standard)

| Metric | Description | Use |
|--------|-------------|-----|
| **Fmax** | Max protein-centric F1 over thresholds [0, 1] | Primary metric (CAFA standard) |
| **Smin** | Min semantic distance (information-content-weighted) | Secondary metric |
| **AUPR** | Area under precision-recall curve | Per-category (MF/BP/CC) |
| **F1 (micro/macro)** | Multi-label F1 | Quick sanity check |

**NOTE:** The existing `compute_go_reward()` uses simple set-based F1. For RL reward, this
is fine (fast, differentiable-friendly). For evaluation, implement Fmax with threshold sweep.

### 1.4 Implementation Steps

```
Step 1: Data Pipeline
  File: scripts/data/download_cafa.py
  - Download CAFA 5 from HuggingFace (or Kaggle)
  - Parse FASTA + TSV into our JSON instruction format
  - GO DAG propagation: propagate annotations up the ontology hierarchy
    (a protein annotated with GO:0004713 also has GO:0016740 via is_a)
  - Save to data/processed/cafa5_go/go_prediction.json
  - Stats: log per-aspect (MF/BP/CC) term counts and protein counts

Step 2: Data Config
  File: configs/data/cafa5_go.yaml
  - name: cafa5_go
  - source: CAFA5
  - paths.processed: ${paths.processed_dir}/cafa5_go
  - Add field: task: go_prediction   # <-- this is what GRPOTrainer reads

Step 3: Update Reward Function (minor)
  File: src/training/grpo_trainer.py
  - compute_go_reward() is already correct
  - Add: handle metadata.go_aspect for per-category reward breakdown in detailed mode

Step 4: Evaluation Upgrade
  File: src/evaluation/go_prediction.py
  - Add Fmax computation: sweep threshold τ ∈ [0.01, 1.0], step=0.01
    For each τ: precision(τ), recall(τ) → F1(τ). Fmax = max F1(τ).
  - Add Smin computation: requires GO DAG + information content weights
    (use goatools library: pip install goatools)
  - Add per-aspect (MF/BP/CC) Fmax and AUPR
  - Replace demo dataset with real CAFA test set

Step 5: Experiment Config
  File: configs/experiment/grpo_go_prediction.yaml
  - defaults: /data: cafa5_go, /training: grpo
  - data.task: go_prediction
  - training.grpo.group_size: 4
  - training.rollout.max_tokens: 256  # GO terms are short
```

### 1.5 Validation Criteria
- [ ] `compute_go_reward("GO:0005524, GO:0004713", "GO:0005524, GO:0016740")` returns correct F1
- [ ] Fmax on CAFA5 test set with untrained model > 0.0 (sanity)
- [ ] GRPO training loop runs 10 steps without crash
- [ ] Rewards increase over 100 steps (learning signal exists)

---

## Task 2: PPI Prediction

### Status: FUTURE TODO

> **Reason:** PPI requires a fundamentally different data format (protein **pairs** + binary
> label), which breaks the single-sequence assumption in `MolInstructionsDataset`,
> `ProteinLLM.generate()`, and the ESM-3 encoder pipeline. Implementing this properly
> requires:
>
> 1. A new `PPIPairDataset` class that returns two sequences per sample
> 2. Modification to `ProteinLLM.prepare_inputs()` to encode two proteins
>    (either concatenate embeddings or dual-encoder architecture)
> 3. A PPI benchmark dataset (BioSNAP, STRING, or HuRI)
> 4. Careful handling of negative sampling (random pairs vs hard negatives)
>
> The reward function `compute_ppi_reward()` and evaluation code
> `src/evaluation/ppi_prediction.py` are already implemented and correct.
> The blocker is the data pipeline and dual-protein encoding.
>
> **Estimated effort:** 2-3 days for a senior engineer.
>
> **Candidate datasets:**
> - SHS27k / SHS148k (sequence-based PPI, commonly benchmarked)
> - BioSNAP (gold standard)
> - HuRI (Human Reference Protein Interactome)

---

## Task 3: Protein Stability (ddG) Prediction

### 3.1 Data Sources

| Resource | Use For | Entries | Format | Access |
|----------|---------|--------|--------|--------|
| **Mega-Scale** (Tsuboyama 2023) | **Primary RL training** | 776K curated | CSV w/ sequences + ddG | HuggingFace: `RosettaCommons/MegaScale` |
| **FireProtDB 2.0** | Diverse eval | 12.9M measurements | CSV/JSON | loschmidt.chemi.muni.cz/fireprotdb/ |
| **S669** | **Standard eval benchmark** | 669 mutations | From ProtDDG-Bench | protddg-bench.github.io |
| ThermoMutDB | Supplementary | 14.7K | CSV/JSON | biosig.lab.uq.edu.au/thermomutdb/ |
| ProThermDB | Legacy (needs cleaning) | 31.6K | Web download | web.iitm.ac.in/bioinfo2/prothermdb/ |

**Decision: Mega-Scale for RL training (largest, cleanest, HuggingFace-ready), S669 for eval.**

### 3.2 Data Format

Mega-Scale provides:
```csv
aa_seq, WT_name, mut_type, deltaG, ddG_ML, Stabilizing_mut, ...
```

Convert to our JSON format:
```json
{
  "instruction": "Predict the change in protein stability (ddG in kcal/mol) for this mutation. Classify as stabilizing (ddG > 1.0), neutral (-1.0 to 1.0), or destabilizing (ddG < -1.0).",
  "input": "Wild-type: MKTAYIAK... Mutation: A42G",
  "output": "ddG = -2.3 kcal/mol. This mutation is destabilizing.",
  "metadata": {
    "ddG": -2.3,
    "wt_name": "1BNZ",
    "mutation": "A42G",
    "source": "megascale"
  }
}
```

**Key:** The `metadata.ddG` field is what the reward function reads — NOT the text output.
The text output is what the model generates; the reward checks if the predicted ddG in the
generated text is close to `metadata.ddG`.

### 3.3 Metrics

| Metric | Description | Use |
|--------|-------------|-----|
| **Spearman ρ** | Rank correlation (primary, robust to outliers) | Primary metric |
| **Pearson r** | Linear correlation | Secondary |
| **RMSE** (kcal/mol) | Root mean squared error | Regression quality |
| **MAE** (kcal/mol) | Mean absolute error | Interpretable error |
| **3-class accuracy** | Stabilizing/Neutral/Destabilizing | Classification |

**Classification thresholds:** Stabilizing: ddG > 1.0, Neutral: -1.0 ≤ ddG ≤ 1.0, Destabilizing: ddG < -1.0

### 3.4 Implementation Steps

```
Step 1: Data Pipeline
  File: scripts/data/download_megascale.py
  - Load from HuggingFace: datasets.load_dataset("RosettaCommons/MegaScale", "dataset3_single")
  - Use pre-built train/val/test splits (1.5M / 164K / 169K)
  - Convert to JSON instruction format with metadata.ddG
  - Save to data/processed/megascale_stability/stability.json
  - Sign convention: CHECK and document. Mega-Scale uses positive dG = stable.
    Our reward function expects ddG (change upon mutation). Verify alignment.

Step 2: Data Config
  File: configs/data/megascale_stability.yaml
  - name: megascale_stability
  - source: RosettaCommons/MegaScale
  - task: stability   # <-- what GRPOTrainer._setup_reward_function() reads
  - paths.processed: ${paths.processed_dir}/megascale_stability

Step 3: Update Reward Function (minor)
  File: src/training/grpo_trainer.py
  - compute_stability_reward() is already correct
  - Ensure _compute_rewards() passes metadata.ddG (not text output) as ground_truth
  - Currently (line 1544): ground_truths = batch.get("response", batch.get("output", []))
    This passes the TEXT response. Need to pass metadata.ddG instead.
    FIX: Add a ground_truth_key to the reward config, or parse it from metadata.

Step 4: Evaluation
  File: src/evaluation/stability.py
  - Add Spearman correlation (primary metric, currently missing)
  - Add 3-class confusion matrix
  - Load S669 as the standard benchmark test set
  - Replace demo dataset with real data

Step 5: Experiment Config
  File: configs/experiment/grpo_stability.yaml
  - defaults: /data: megascale_stability, /training: grpo
  - data.task: stability
  - training.rollout.max_tokens: 128  # ddG predictions are short
```

### 3.5 Validation Criteria
- [ ] `compute_stability_reward("ddG = -2.3 kcal/mol", -2.3)` → ~1.0
- [ ] `compute_stability_reward("ddG = 5.0 kcal/mol", -2.3)` → ~0.0
- [ ] Spearman ρ computable on S669 test set
- [ ] GRPO training shows increasing mean reward over 100 steps

---

## Task 4: Structural Quality (ESMFold Reward)

### 4.1 Strategy: Pre-computed pLDDT Lookup + Offline ESMFold

**Problem:** Running ESMFold live during GRPO training is too slow (~14-21s per protein)
and too VRAM-heavy (~16-20 GB on top of ESM-3 + Qwen3-4B).

**Solution:** Two-phase approach:

**Phase A — Offline pre-computation (one-time):**
1. Download AlphaFold DB pLDDT scores for proteins in our training set
   (AlphaFold DB has 214M structures with pLDDT in B-factor field, CC-BY-4.0)
2. OR run ESMFold offline on our protein dataset, store `{sequence_hash: (pLDDT, pTM)}`
3. Save as a Parquet/SQLite lookup table

**Phase B — Training-time reward:**
1. For sequences in the lookup table: instant reward (0ms)
2. For novel generated sequences (shouldn't happen in GRPO — we generate TEXT about
   proteins, not new sequences): use pLDDT-Predictor proxy model (0.007s/protein,
   minimal VRAM). See: github.com/jw-chae/pLDDT_Predictor

### 4.2 Data Sources

| Resource | Use For | Entries | Access |
|----------|---------|--------|--------|
| **AlphaFold DB** | Pre-computed pLDDT/pTM | 214M structures | alphafold.ebi.ac.uk (FTP/API) |
| **ESM Metagenomic Atlas** | Pre-computed ESMFold scores | 617M | esmatlas.com (Parquet metadata) |
| **DisProt** | Disorder annotations (ordered vs disordered) | 2.3K proteins | disprot.org |
| **CATH 4.3** | Structural classification | Standard set | cathdb.info |

**Decision: AlphaFold DB for pLDDT lookup, DisProt for ordered/disordered classification.**

### 4.3 Data Format

```json
{
  "instruction": "Assess the structural quality of this protein. Describe whether it is well-folded or disordered, predict the approximate pLDDT confidence score (0-100), and classify the fold quality as high (>80), medium (50-80), or low (<50).",
  "input": "MKTAYIAK...",
  "output": "This protein is well-folded with high structural confidence. Predicted pLDDT: 85. Fold quality: high.",
  "metadata": {
    "plddt": 85.3,
    "ptm": 0.82,
    "fold_category": "high",
    "source": "alphafold_db",
    "uniprot_id": "P00533"
  }
}
```

### 4.4 Metrics

| Metric | Description |
|--------|-------------|
| **Quality claim accuracy** | % correct "well-folded" vs "disordered" claims |
| **pLDDT MAE** | Mean absolute error of predicted pLDDT score |
| **Category accuracy** | 3-class: high/medium/low |
| **Composite reward** | Weighted sum (same as `compute_esmfold_reward()`) |

### 4.5 Implementation Steps

```
Step 1: Pre-compute pLDDT Lookup Table
  File: scripts/data/build_plddt_lookup.py
  - For each protein in our training data (Mol-Instructions, Swiss-Prot, etc.):
    - Look up UniProt ID → query AlphaFold DB REST API for pLDDT
    - OR batch-download AlphaFold proteome files, parse B-factor for pLDDT
  - Store as: data/processed/plddt_lookup.parquet
    Columns: [sequence_hash, uniprot_id, mean_plddt, ptm, fold_category]

Step 2: Structure Quality Dataset
  File: scripts/data/build_structure_quality.py
  - Combine pLDDT lookup with protein sequences
  - Generate instruction-format samples with pLDDT ground truth in metadata
  - Add DisProt entries for disorder examples (enriches the "low quality" category)
  - Save to data/processed/structure_quality/structure_quality.json

Step 3: Update Reward Function
  File: src/training/grpo_trainer.py
  - Modify compute_esmfold_reward() to accept pre-computed pLDDT/pTM from metadata
    instead of calling ESMFold live
  - Add a wrapper: if metadata has plddt/ptm, use those; else fall back to ESMFold
  - This makes it fast (lookup) during training but still works for live eval

Step 4: Data Config
  File: configs/data/structure_quality.yaml
  - task: esmfold
  - paths.processed: ${paths.processed_dir}/structure_quality

Step 5: Experiment Config
  File: configs/experiment/grpo_structure.yaml
  - defaults: /data: structure_quality, /training: grpo
  - data.task: esmfold
```

### 4.6 Validation Criteria
- [ ] pLDDT lookup table covers >90% of training proteins
- [ ] Reward computation < 1ms per sample (lookup mode)
- [ ] Reward = ~0.4 when model correctly says "well-folded" for pLDDT > 70 protein
- [ ] GRPO training runs without ESMFold loaded on GPU

---

## Shared Infrastructure Changes

### A. Add `data.task` field to config system

Currently missing. The GRPO trainer defaults to `"go_prediction"` but this field
doesn't exist in any data config.

```yaml
# Add to each RL data config:
task: go_prediction  # or: stability, esmfold, ppi
```

### B. Ground Truth Extraction in GRPOTrainer

The trainer currently reads `batch["response"]` as ground truth (line 1544). For RL tasks,
the ground truth is in metadata:
- GO: output text contains GO terms (OK as-is)
- Stability: `metadata.ddG` (float, NOT the text output)
- ESMFold: `metadata.plddt` + `metadata.ptm` (pre-computed)

**Fix in `_training_step()`:**
```python
# Current (broken for stability/esmfold):
ground_truths = batch.get("response", batch.get("output", []))

# Fix: task-aware ground truth extraction
task = self.cfg.data.get("task", "go_prediction")
if task in ("stability", "ddg"):
    ground_truths = [sample.get("metadata", {}).get("ddG", "") for sample in raw_batch]
elif task in ("esmfold", "structure"):
    # Pass protein_sequence; reward function uses pLDDT from metadata or live fold
    ground_truths = protein_sequences  # already handled by _is_esmfold_reward
else:
    ground_truths = batch.get("response", batch.get("output", []))
```

### C. MolInstructionsDataset: Expose metadata

Currently `__getitem__` does not return the `metadata` field from the JSON.
Add it:

```python
# In MolInstructionsDataset.__getitem__():
sample["metadata"] = item.get("metadata", {})
```

### D. Dependencies

```
# Add to requirements.txt or pyproject.toml:
goatools          # GO DAG traversal for Fmax/Smin computation
obonet            # Parse OBO format GO ontology files
scikit-learn      # Already in env (for Spearman, AUPR, etc.)
pyarrow           # For parquet lookup table
```

---

## File Manifest (New Files to Create)

```
scripts/data/
├── download_cafa.py              # Task 1: Download + convert CAFA GO data
├── download_megascale.py         # Task 3: Download + convert Mega-Scale stability data
├── build_plddt_lookup.py         # Task 4: Pre-compute AlphaFold DB pLDDT lookup
└── build_structure_quality.py    # Task 4: Build structure quality dataset

configs/data/
├── cafa5_go.yaml                 # Task 1: GO prediction data config
├── megascale_stability.yaml      # Task 3: Stability data config
└── structure_quality.yaml        # Task 4: Structure quality data config

configs/experiment/
├── grpo_go_prediction.yaml       # Task 1: GRPO + GO experiment preset
├── grpo_stability.yaml           # Task 3: GRPO + stability experiment preset
└── grpo_structure.yaml           # Task 4: GRPO + structure experiment preset
```

## Files to Modify

```
src/training/grpo_trainer.py      # Ground truth extraction, reward metadata support
src/data/mol_instructions.py      # Expose metadata field in __getitem__
src/evaluation/go_prediction.py   # Add Fmax, Smin, per-aspect metrics
src/evaluation/stability.py       # Add Spearman, real benchmark data
configs/config.yaml               # (no change needed — data.task is per-data-config)
```

---

## Execution Order for Agent Teams

```
Phase 1 (can be parallel):
  ├── Agent A: Task 1 Steps 1-2 (download CAFA, build data config)
  ├── Agent B: Task 3 Steps 1-2 (download Mega-Scale, build data config)
  └── Agent C: Task 4 Step 1 (build pLDDT lookup table)

Phase 2 (after Phase 1):
  ├── Agent A: Shared infra changes (B, C, D above)
  └── Agent B: Task 4 Steps 2-4 (structure quality dataset + config)

Phase 3 (after Phase 2):
  ├── Agent A: Task 1 Steps 3-5 (reward update, eval upgrade, experiment config)
  ├── Agent B: Task 3 Steps 3-5 (reward update, eval upgrade, experiment config)
  └── Agent C: Task 4 Step 5 (experiment config)

Phase 4 (validation):
  └── Single agent: Run each GRPO experiment for 10 steps, verify rewards > 0
```

---

## References

**GO Term:**
- CAFA 5: kaggle.com/competitions/cafa-5-protein-function-prediction
- CAFA 6: kaggle.com/competitions/cafa-6-protein-function-prediction
- GO Bench: gobench.org (Bioinformatics 2023)
- DeepGO2: github.com/bio-ontology-research-group/deepgo2

**Stability:**
- Mega-Scale: Tsuboyama et al., Nature 620 (2023). HuggingFace: `RosettaCommons/MegaScale`
- S669: ProtDDG-Bench (protddg-bench.github.io)
- FireProtDB 2.0: NAR 2025 (loschmidt.chemi.muni.cz/fireprotdb/)
- ThermoMutDB: NAR 2021 (biosig.lab.uq.edu.au/thermomutdb/)

**Structure:**
- AlphaFold DB: alphafold.ebi.ac.uk (214M structures, CC-BY-4.0)
- pLDDT-Predictor: github.com/jw-chae/pLDDT_Predictor (250,000x speedup)
- ProteinZero: arxiv.org/abs/2506.07459 (GRPO for protein design)
- DisProt: disprot.org (disorder annotations)
