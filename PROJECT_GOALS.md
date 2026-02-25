# Project Goals: Post-Training Protein LLM

> **Purpose**: High-level goals, current sprint focus, and backlog for the agent team.
> Team Lead should check this file to understand priorities and direction.

---

## Vision

Build a **modular, extensible multimodal LLM system** for protein understanding that:
1. Combines **ESM-3** protein embeddings with LLM capabilities
2. Supports **multiple encoding approaches** (text-based, embedding-based, TBD)
3. Supports flexible training pipelines (SFT → GRPO/DPO)
4. Evaluates on standard protein benchmarks (GO, PPI, Stability)
5. Maintains reproducibility through proper versioning and logging

---

## Core Configuration

### Base Models (Initial Development)

| Component | Model | Notes |
|-----------|-------|-------|
| **LLM** | `Qwen/Qwen3-4B-Instruct-2507` | Smaller model for fast iteration |
| **Protein Encoder** | ESM-3 small | Start with small, scale up later |

### Encoding Approaches

> **CRITICAL**: Each approach requires its own data processing + architecture config

| Approach | Config Key | File | Status | Description |
|----------|------------|------|--------|-------------|
| 1. Text-based | `approach: text` | src/models/protein_encoder.py | ✅ Implemented | Raw sequence as `<protein>MKTL...</protein>` |
| 2. ESM-3 + MLP | `approach: esm3` + `projector.type: mlp` | src/models/multimodal_llm.py | ✅ Implemented | ESM-3 → AttentionPooling → MLP → LLM |
| 3. ESM-3 + Perceiver | `approach: esm3` + `projector.type: perceiver` | src/models/perceiver.py | ✅ Implemented | ESM-3 → PerceiverResampler → LLM |

**Config structure needed**:
```yaml
# configs/config.yaml
approach: esm3  # or "text" or "tbd"

# Each approach has:
# - Data processing pipeline
# - Model architecture
# - Evaluation strategy
```

### wandb Project Separation

| Training Phase | wandb Project | Purpose |
|----------------|---------------|---------|
| **SFT** | `protein-llm-sft` | Supervised fine-tuning experiments |
| **RL (GRPO/DPO)** | `protein-llm-rl` | Reinforcement learning alignment |

---

## Training Data Strategy

### SFT Data Sources

| Dataset | Status | SFT Ready? | Conversion Needed |
|---------|--------|------------|-------------------|
| **Mol-Instructions** | ✅ Available | ✅ Yes | None - already instruction format |
| **Swiss-Prot** | ✅ Available | ❌ No | Convert to Mol-Instructions format |
| **IPD-PDB** | ✅ Available | ❌ No | Convert to Mol-Instructions format |

### Mol-Instructions Data Format

```
┌─────────────────────────────────────────────────────────────────┐
│  instruction: "Predict the GO terms for this protein"          │  ← Task
│  input: "MKTLLIAAAVAAGIATA..."                                  │  ← Protein Seq
│  output: "GO:0003674, GO:0005575, GO:0008150"                  │  ← Answer
└─────────────────────────────────────────────────────────────────┘
```

**Tasks in Mol-Instructions** (~505K total):
- Protein Design
- Catalytic Activity Prediction
- Protein Function Prediction
- Functional Description Generation
- Domain/Motif Prediction

### SFT Task Split Strategy

> **DECISION**: Random 90/5/5 split per task (all tasks in train AND test)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Per-Task Split (90/5/5)                      │
├─────────────────────────────────────────────────────────────────┤
│  Task Type                    │ Train (90%) │ Val (5%) │ Test  │
│  ─────────────────────────────┼─────────────┼──────────┼───────│
│  Protein Design               │     90%     │    5%    │   5%  │
│  Catalytic Activity           │     90%     │    5%    │   5%  │
│  Function Prediction          │     90%     │    5%    │   5%  │
│  Domain/Motif Prediction      │     90%     │    5%    │   5%  │
│  Functional Description       │     90%     │    5%    │   5%  │
└─────────────────────────────────────────────────────────────────┘
```

### Swiss-Prot / IPD-PDB Conversion

> **DECISION**: Match Mol-Instructions instruction format

**Target conversion** (use same instruction templates):
```python
# Swiss-Prot → SFT
{
    "instruction": "Predict the function of this protein",
    "input": sequence,
    "output": function_annotation
}

# IPD-PDB → SFT
{
    "instruction": "Predict the secondary structure of this protein",
    "input": sequence,
    "output": secondary_structure
}
```

### RL Training Strategy

> **DECISION**: Use all four reward functions for multi-task GRPO

| RL Task | Reward Function | Status | Description |
|---------|-----------------|--------|-------------|
| **GO Prediction** | `compute_go_reward` | ✅ Implemented | F1 score of predicted GO terms |
| **PPI Prediction** | `compute_ppi_reward` | ✅ Implemented | Binary Yes/No accuracy |
| **Stability (ddG)** | `compute_stability_reward` | ✅ Implemented | Gaussian reward on ddG error |
| **ESMFold Structure** | `compute_esmfold_reward` | ✅ Implemented | pLDDT/pTM structural quality alignment |

**RL Pipeline**:
```
SFT Checkpoint ──► GRPO Training ──► Final Model
                        │
                        ├─ GO term rewards (F1-based)
                        ├─ PPI interaction rewards (binary)
                        ├─ Stability rewards (regression)
                        └─ ESMFold rewards (structural quality)
```

**Note**: GRPO uses **verifiable rewards** (no separate reward model needed)

---

## Strategic Goals

### Goal 1: ESM-3 + Qwen3-4B Integration
- [x] Get ESM-3 small model running with Qwen3-4B
- [x] Efficient memory usage (both models on same GPU)
- [x] Validate forward pass works end-to-end
- [ ] Benchmark inference speed

**Status**: ✅ Complete (forward pass verified, 50K SFT run done, eval_loss 3.64)

### Goal 2: Approach-Based Architecture
- [x] Config-driven approach selection (`approach: text|esm3`)
- [x] Approach-specific data processing pipelines
- [x] Approach-specific model architectures (text / MLP / Perceiver)
- [x] Easy to add new approaches (projector factory + config override)

**Status**: ✅ Complete (three approaches: text, esm3+mlp, esm3+perceiver)

### Goal 3: Multi-Task Training/Testing Split
- [x] Define training tasks vs testing tasks (90/5/5 random split per task)
- [x] Implement task-based data filtering (Mol-Instructions pipeline)
- [ ] Evaluate on held-out tasks only
- [ ] Track per-task metrics

**Status**: 🟡 Partially Complete

### Goal 4: SFT Data Conversion
- [x] Convert Swiss-Prot to SFT format (swissprot_converter.py)
- [ ] Convert IPD-PDB to SFT format
- [x] Combine with Mol-Instructions (combined dataset)
- [ ] Balanced sampling across sources

**Status**: 🟡 Partially Complete

### Goal 5: RL Pipeline (After SFT)
- [x] Investigate current RL implementation (GRPO trainer complete)
- [x] Define RL tasks and rewards (GO, PPI, Stability, ESMFold)
- [ ] Separate wandb project for RL
- [ ] SFT → RL pipeline end-to-end run

**Status**: 🟡 Implementation Complete, Awaiting First Run

### Goal 6: Quality & Testing
- [x] Unit tests for Perceiver Resampler (19 tests)
- [ ] Integration tests for full pipelines
- [x] Critical rule enforcement (ESM frozen, LoRA k/v only)

**Status**: 🟡 In Progress

### Goal 7: Documentation & Tracking
- [x] Research log with all experiments
- [x] Architecture documentation
- [x] Training guide

**Status**: ✅ Complete

---

## Current Sprint

> **Sprint Focus**: Three-way comparison (text vs MLP vs Perceiver) + GRPO with ESMFold rewards

### Sprint Goals
1. ✅ ESM-3 + Qwen3-4B SFT working (50K run, eval_loss 3.64)
2. ✅ Three encoding approaches implemented (text, MLP, Perceiver)
3. ✅ GRPO trainer with 4 reward functions (GO, PPI, Stability, ESMFold)
4. ⬜ Run SFT with Perceiver Resampler (compare to MLP baseline)
5. ⬜ Run text-only SFT baseline for comparison
6. ⬜ Run first GRPO training (SFT checkpoint → GRPO)
7. ⬜ Evaluate all three approaches on GO/PPI/Stability benchmarks

### Active Tasks

| Task | Status | Notes |
|------|--------|-------|
| Perceiver SFT run | ⬜ Pending | `encoder.projector.type=perceiver` |
| Text-only SFT baseline | ⬜ Pending | `approach=text` |
| GRPO with ESMFold reward | ⬜ Pending | Needs SFT checkpoint first |
| Three-way evaluation comparison | ⬜ Pending | GO, PPI, Stability metrics |

### Sprint Blockers
- None — all implementation complete, ready for experiments

---

## Backlog

> Future work items, prioritized. Team Lead pulls from here for next sprint.

### High Priority

| Item | Description | Estimated Effort |
|------|-------------|------------------|
| Three-way comparison experiments | Run text/MLP/Perceiver SFT, evaluate all | Large |
| First GRPO run | SFT→GRPO with ESMFold rewards | Large |
| Checkpoint resume | `--resume` flag for training (training_state.pt exists) | Medium |
| wandb artifact logging | Save checkpoints as artifacts | Small |

### Medium Priority

| Item | Description | Estimated Effort |
|------|-------------|------------------|
| DPO trainer | Alternative to GRPO | Medium |
| Hyperparameter sweeps | Hydra multirun configs | Small |
| Scale to larger LLM | Qwen3-8B or 14B | Medium |
| IPD-PDB data conversion | Convert to SFT format | Medium |

### Low Priority / Future

| Item | Description | Estimated Effort |
|------|-------------|------------------|
| Protein generation (Path A) | Text-token AA generation pipeline | Large |
| Multi-task training | Train on multiple tasks jointly | Large |
| Model distillation | Smaller models for inference | Large |
| API server | Serve model for inference | Medium |

---

## Milestones

### Milestone 1: SFT Baseline ✅
- [x] SFT trainer working end-to-end
- [x] Checkpoint versioning implemented
- [x] GO and PPI evaluation working
- [x] First results logged (50K run, eval_loss 3.64)

**Completed**: 2026-02-19

### Milestone 2: Architecture Comparison ⬜
- [x] Text-based approach implemented
- [x] MLP projector approach implemented
- [x] Perceiver Resampler approach implemented
- [ ] Run all three and compare on benchmarks

**Target**: Current sprint

### Milestone 3: RL Alignment ⬜
- [x] GRPO trainer implemented (4 reward functions)
- [x] ESMFold reward function added
- [ ] SFT → GRPO pipeline working end-to-end
- [ ] Comparison: SFT vs SFT+GRPO

**Target**: Next sprint

### Milestone 4: Publication Ready ⬜
- [ ] All evaluations working with statistical significance
- [ ] Three-way comparison results table
- [ ] GRPO improvement results
- [ ] Reproducible experiments

**Target**: Sprint +2

---

## Key Metrics to Track

| Metric | Current (MLP) | Target | Notes |
|--------|---------------|--------|-------|
| SFT Eval Loss | 3.64 | < 3.0 | 50K run, step 200/7815 |
| GO F1 (macro) | TBD | 0.50+ | Primary metric |
| PPI Accuracy | TBD | 0.80+ | Binary classification |
| Stability MAE | TBD | < 1.0 | ddG prediction |
| Test Coverage | ~30% | 80%+ | 19 perceiver tests added |

---

## Notes

> Free-form notes, ideas, observations

### Technical Notes
- **ESM-3 small** for initial development
- **Qwen3-4B-Instruct-2507** for faster iteration (all Qwen configs use Instruct variants)
- Attention pooling with 32 tokens is a good default
- LoRA r=8 on all linear layers (q/k/v/o + gate/up/down) for expressiveness
- Training uses model's native chat template with protein-expert system prompt
- Separate wandb projects: `protein-llm-sft` and `protein-llm-rl` (tensorboard disabled)

### Resolved Questions ✅
- [x] Task split: Random 90/5/5 per task (all tasks in train AND test)
- [x] RL tasks: Use all three (GO, PPI, Stability)
- [x] Data conversion: Match Mol-Instructions format
- [x] Base models: ESM-3 small + Qwen3-4B

### Open Questions
- How to handle very long sequences (>1024)? (ESMFold wrapper truncates at 1024)
- Should we pre-compute ESM-3 embeddings for efficiency?
- Which GRPO reward function to use first for experiments?

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-18 | Initial PROJECT_GOALS.md created |
| | Agent team structure defined |
| | Updated base models: ESM-3 small + Qwen3-4B |
| | Resolved: task split (90/5/5 random), RL tasks (all three), data conversion |
| | Added approach-based architecture requirement |
| | Separated wandb projects for SFT vs RL |
| 2026-02-19 | First successful 50K SFT run (eval_loss 3.64) |
| | Checkpoint/eval pipeline standardized across SFT and GRPO |
| 2026-02-20 | Perceiver Resampler implemented (src/models/perceiver.py) |
| | ESMFold GRPO reward implemented (src/models/esmfold_wrapper.py) |
| | Three-way comparison thesis defined (text vs MLP vs Perceiver) |
| | Perceiver 2-layer default (130M params, +1GB, 12% slower than MLP) |
| | Unified experiment directory: all artifacts under results/{experiment_name}/ |
| | Lineage tracking (lineage.json) for base→SFT→GRPO pipeline |
| | Updated goals to reflect current completion status |
| 2026-02-22 | Model configs updated to Instruct variants (Qwen3-4B/8B/14B-Instruct-2507) |
| | LoRA expanded from k/v only to all linear layers (q/k/v/o + gate/up/down) |
| | Added protein-expert system prompt for consistent model identity |
| | Switched training from Alpaca format to model's native chat template |
| | Disabled tensorboard, wandb-only logging |
| | Updated all agent spawn prompts with correct context |
