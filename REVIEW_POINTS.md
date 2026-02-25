# Review Points & User Requirements

> **Purpose**: Central place for you to leave comments, requirements, and decisions for the agent team.
> Agents should check this file before starting major tasks.

---

## Current Priorities

1. [x] Checkpoint versioning with clear naming convention
2. [x] Proper wandb logging for all experiments
3. [x] Test coverage for critical components (Perceiver: 19 tests)
4. [x] Documentation stays in sync with code
5. [ ] Run three-way comparison experiments (text vs MLP vs Perceiver)
6. [ ] First GRPO training run with ESMFold rewards
7. [ ] Benchmark evaluation across all approaches

---

## Open Questions (Need Your Input)

> Agents will ask you about these before proceeding.

### Architecture
- [x] Three projector approaches implemented: text, MLP, Perceiver Resampler
- [x] Attention pooling uses 32 query tokens (default)
- [x] Projector is configurable via `encoder.projector.type` (mlp/perceiver)
- [ ] Perceiver Resampler is ~381M params — acceptable for comparison?

### Training
- [ ] Default batch size for H100? (4 / 8 / 16)
- [ ] Checkpoint save frequency? (every N steps / every epoch)
- [ ] Which optimizer? (AdamW 8-bit / AdamW / SGD)

### Evaluation
- [ ] Which GO categories to prioritize? (MF / BP / CC / all)
- [ ] Minimum test set size for significance?
- [ ] Should we track inference latency?

### Data
- [ ] Include Swiss-Prot in SFT training? (adds ~2M examples)
- [ ] Maximum sequence length filter? (512 / 1024 / 2048)

---

## Decisions Made

> Record your decisions here so agents don't ask again.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-18 | Use SafeTensors format | Faster loading, safer than .pt |
| 2026-02-18 | ~~LoRA r=8, targets k/v only~~ | Superseded 2026-02-22 |
| 2026-02-22 | LoRA r=8, all linear layers (q/k/v/o + gate/up/down) | More expressiveness, standard practice |
| 2026-02-22 | Use Instruct model variants (e.g., Qwen3-4B-Instruct-2507) | Non-instruct base models lack chat capability |
| 2026-02-22 | Chat template format with system prompt for training | Alpaca format caused train/inference mismatch |
| 2026-02-22 | Tensorboard disabled, wandb only | Single consistent logging backend |
| 2026-02-18 | ESM encoders always frozen | Critical rule, never modify |
| 2026-02-20 | Three-way comparison thesis | text vs MLP vs Perceiver Resampler |
| 2026-02-20 | Focus on understanding, not generation | Generation deferred to future work |
| 2026-02-20 | ESMFold rewards for GRPO | Structural quality as reward signal |
| 2026-02-20 | Differential LR essential | projector_lr=2e-3 vs base lr=2e-4 |
| 2026-02-20 | Unified experiment directory | All artifacts under results/{experiment_name}/ |
| 2026-02-20 | Lineage tracking | lineage.json chains SFT→GRPO via parent_experiment |
| 2026-02-20 | Perceiver 2-layer default | 6-layer too expensive (382M params); 2-layer is +1GB, 12% slower |

---

## Specific Requirements

### Experiment Directory Convention
All training/eval artifacts go under `results/{experiment_name}/`:
```
results/{experiment_name}/
├── config.yaml            # Full resolved Hydra config
├── lineage.json           # Stage, parent experiment, timestamps
├── training_args.json     # Hyperparameters
├── metrics.json           # Final train/eval metrics
├── checkpoints/
│   └── protein_llm/       # ProteinLLM save (config, pooling, projector, adapter)
├── logs/                  # Hydra + TensorBoard logs
└── eval/                  # Evaluation metrics
```

`experiment_name` is auto-generated (`{method}_{approach}_{model}_{MMDD_HHMMSS}`)
or set manually: `python scripts/train.py experiment_name=my_run`

GRPO chains from SFT via: `parent_experiment=<sft_experiment_name>`

### Logging Requirements
- All training runs MUST log to wandb
- Include these tags: `model`, `dataset`, `method`, `lr`, `epochs`
- Save checkpoints as wandb artifacts

### Code Quality
- All new functions need type hints
- Google-style docstrings for public APIs
- Tests required before merge
- Ruff must pass

---

## Feedback on Current Work

> Leave comments here for specific agents.

### For Architect
```
-
```

### For Trainer
```
-
```

### For Evaluator
```
-
```

### For QA-Engineer
```
-
```

### For Doc-Tracker
```
-
```

---

## Blocked / Waiting On

| Item | Blocked By | Status |
|------|------------|--------|
| | | |

---

## Notes

> Free-form notes, ideas, or thoughts for the team.

```
-
```
