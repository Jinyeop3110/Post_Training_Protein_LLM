# Post-Training Protein LLM: Improvement Plan

**Date**: 2026-02-16
**Objective**: Restructure project for optimal Claude Code integration and LLM post-training workflows

---

## Executive Summary

This plan restructures the project based on 2025-2026 best practices for:
1. **Claude Code optimization** — Skills, commands, hooks, progressive disclosure
2. **LLM post-training workflows** — Hydra configs, modular training pipelines
3. **Python project standards** — pyproject.toml, proper package structure, testing

---

## Part 1: Key Research Findings

### 1.1 Claude Code Best Practices

| Practice | Rationale | Source |
|----------|-----------|--------|
| Keep CLAUDE.md < 60-300 lines | Loaded into every session; irrelevant content wastes tokens | [HumanLayer](https://www.humanlayer.dev/blog/writing-a-good-claude-md) |
| Use `.claude/` directory | Organize skills, agents, commands, hooks | [Claude Code Docs](https://code.claude.com/docs/en/best-practices) |
| Progressive disclosure | Put detailed docs in separate files, reference from CLAUDE.md | [Dometrain](https://dometrain.com/blog/creating-the-perfect-claudemd-for-claude-code/) |
| Pointers over copies | Reference `file:line` instead of embedding code | [HumanLayer](https://www.humanlayer.dev/blog/writing-a-good-claude-md) |
| Let linters handle style | Use deterministic tools via hooks, not LLM instructions | [Claude Best Practices](https://code.claude.com/docs/en/best-practices) |

### 1.2 LLM Post-Training Standards

| Practice | Rationale | Source |
|----------|-----------|--------|
| SFT → RL pipeline | Standard two-stage alignment (SFT then GRPO/DPO) | [PyTorch Blog](https://pytorch.org/blog/a-primer-on-llm-post-training/) |
| Data quality is ceiling | Worst examples limit model performance | [Red Hat](https://developers.redhat.com/articles/2025/11/04/post-training-methods-language-models) |
| GRPO over PPO | 50% less memory, better for reasoning | [Awesome LLM Post-Training](https://github.com/mbzuai-oryx/Awesome-LLM-Post-training) |
| Hydra for configs | Hierarchical YAML with CLI overrides and sweeps | [Hydra Docs](https://hydra.cc/docs/intro/) |

### 1.3 Python Project Standards

| Practice | Rationale | Source |
|----------|-----------|--------|
| Single pyproject.toml | Unified dependency and build configuration | [Python Packaging Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) |
| Modular package structure | Clear separation of concerns, easier testing | [Tweag](https://www.tweag.io/blog/2023-04-04-python-monorepo-1/) |
| Tests mirror src/ | Consistent test organization | Standard practice |

---

## Part 2: Current vs Proposed Structure

### 2.1 Current Structure (Problems)

```
Post_Training_Protein_LLM/
├── CLAUDE.md                    # ❌ 80 lines, too detailed
├── README.md                    # ✓ OK
├── agents.md                    # ❌ Should be in docs/
├── LLM_Post_Training_Methods... # ❌ Should be in docs/
├── setup_env.sh                 # ✓ OK
├── research/                    # ❌ Should be in docs/
├── plan/                        # ❌ Empty, unused
├── pdb_2021aug02_sample/        # ❌ Should be in data/raw/
└── src/
    ├── models/                  # ✓ OK but incomplete
    ├── data/                    # ✓ OK
    ├── training/                # ❌ Empty skeleton
    └── evaluation/              # ❌ Empty skeleton
```

**Issues:**
- No `.claude/` directory for Claude Code features
- No configuration management (Hydra)
- No entry point scripts
- No tests
- Documentation scattered at root level
- CLAUDE.md too verbose for every-session loading

### 2.2 Proposed Structure

```
Post_Training_Protein_LLM/
│
├── CLAUDE.md                          # Concise (<60 lines)
├── pyproject.toml                     # Dependencies & build
├── README.md                          # User documentation
├── setup_env.sh                       # Environment setup
│
├── .claude/                           # 🆕 Claude Code configuration
│   ├── settings.json                  # Hooks, permissions
│   ├── agents/
│   │   ├── research.md                # Research assistant
│   │   ├── code-reviewer.md           # Code review
│   │   └── experiment-runner.md       # Training launcher
│   ├── commands/
│   │   ├── train.md                   # /train command
│   │   ├── eval.md                    # /eval command
│   │   ├── data-prep.md               # /data-prep command
│   │   └── debug.md                   # /debug command
│   └── skills/
│       ├── protein-encoding/SKILL.md  # ESM-2 knowledge
│       ├── rl-training/SKILL.md       # veRL/GRPO knowledge
│       └── hydra-configs/SKILL.md     # Config patterns
│
├── docs/                              # 🆕 Detailed documentation
│   ├── architecture.md
│   ├── training_guide.md
│   ├── evaluation_guide.md
│   ├── datasets.md
│   ├── troubleshooting.md
│   └── research/
│       ├── protein_datasets_and_benchmarks.md
│       ├── LLM_Post_Training_Methods_Summary.md
│       └── agents_research_log.md
│
├── configs/                           # 🆕 Hydra configurations
│   ├── config.yaml                    # Main config
│   ├── model/
│   │   ├── qwen2_7b.yaml
│   │   ├── llama3_8b.yaml
│   │   └── default.yaml
│   ├── encoder/
│   │   ├── esm2_650m.yaml
│   │   ├── esm2_3b.yaml
│   │   └── default.yaml
│   ├── data/
│   │   ├── mol_instructions.yaml
│   │   ├── ipd_pdb.yaml
│   │   ├── swissprot.yaml
│   │   └── default.yaml
│   ├── training/
│   │   ├── sft_qlora.yaml
│   │   ├── sft_lora.yaml
│   │   ├── grpo.yaml
│   │   ├── dpo.yaml
│   │   └── default.yaml
│   ├── evaluation/
│   │   ├── go_prediction.yaml
│   │   ├── ppi.yaml
│   │   ├── stability.yaml
│   │   └── default.yaml
│   └── experiment/
│       ├── baseline_sft.yaml
│       ├── full_pipeline.yaml
│       └── ablation_pooling.yaml
│
├── scripts/                           # 🆕 Entry points
│   ├── train.py                       # Hydra training entry
│   ├── evaluate.py                    # Evaluation entry
│   ├── inference.py                   # Demo/inference
│   └── prepare_data.py                # Data preparation
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── protein_encoder.py         # Existing
│   │   ├── projector.py               # 🆕 MLP projector
│   │   ├── multimodal_llm.py          # 🆕 Combined model
│   │   └── pooling.py                 # 🆕 Attention pooling
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pdb_dataset.py             # Existing
│   │   ├── rcsb_dataset.py            # Existing
│   │   ├── download.py                # Existing
│   │   ├── instruction_dataset.py     # 🆕 Instruction format
│   │   ├── collators.py               # 🆕 Data collation
│   │   └── tokenization.py            # 🆕 Tokenization
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py             # 🆕 SFT implementation
│   │   ├── grpo_trainer.py            # 🆕 GRPO implementation
│   │   ├── dpo_trainer.py             # 🆕 DPO implementation
│   │   ├── callbacks.py               # 🆕 Training callbacks
│   │   └── utils.py                   # 🆕 Utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── go_prediction.py           # 🆕 GO term eval
│   │   ├── ppi_prediction.py          # 🆕 PPI eval
│   │   ├── stability.py               # 🆕 Stability eval
│   │   ├── metrics.py                 # 🆕 Metrics
│   │   └── benchmarks.py              # 🆕 Benchmark runners
│   └── utils/                         # 🆕 Shared utilities
│       ├── __init__.py
│       ├── logging.py
│       ├── checkpoint.py
│       └── distributed.py
│
├── tests/                             # 🆕 Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── models/
│   │   └── test_protein_encoder.py
│   ├── data/
│   │   └── test_datasets.py
│   └── training/
│       └── test_trainers.py
│
├── notebooks/                         # 🆕 Exploration notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_architecture.ipynb
│   └── 03_results_analysis.ipynb
│
└── data/                              # Data directory (gitignored)
    ├── raw/
    │   ├── pdb_2021aug02_sample/
    │   ├── mol_instructions/
    │   └── swissprot/
    ├── processed/
    └── checkpoints/
```

---

## Part 3: Detailed Implementation Plan

### Phase 1: Project Restructuring (Foundation)

**Goal**: Reorganize files and create new directory structure

#### 1.1 Create Directory Structure
```bash
# Create new directories
mkdir -p .claude/{agents,commands,skills/protein-encoding,skills/rl-training,skills/hydra-configs}
mkdir -p docs/research
mkdir -p configs/{model,encoder,data,training,evaluation,experiment}
mkdir -p scripts
mkdir -p src/utils
mkdir -p tests/{models,data,training}
mkdir -p notebooks
mkdir -p data/{raw,processed,checkpoints}
```

#### 1.2 Move Existing Files
```bash
# Move documentation to docs/
mv agents.md docs/research/agents_research_log.md
mv LLM_Post_Training_Methods_Summary.md docs/research/
mv research/protein_datasets_and_benchmarks.md docs/research/

# Move dataset to data/raw/
mv pdb_2021aug02_sample data/raw/

# Remove empty plan/ directory
rm -rf plan/
```

#### 1.3 Create pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "protein-llm"
version = "0.1.0"
description = "Post-training LLMs for protein understanding"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.5.0",
    "transformers>=4.51.0",
    "peft>=0.10.0",
    "bitsandbytes>=0.43.0",
    "fair-esm>=2.0.0",
    "biopython>=1.86",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "wandb",
    "tensorboard",
    "datasets>=2.18.0",
    "scipy",
    "pandas",
    "numpy<2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov",
    "ruff",
    "mypy",
]
training = [
    "deepspeed>=0.14.0",
    "ray[default]>=2.10.0",
]

[project.scripts]
protein-train = "scripts.train:main"
protein-eval = "scripts.evaluate:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
```

---

### Phase 2: Claude Code Integration

**Goal**: Set up `.claude/` directory for optimal Claude Code workflows

#### 2.1 Create .claude/settings.json
```json
{
  "permissions": {
    "allow": [
      "Bash(python scripts/*)",
      "Bash(pytest*)",
      "Bash(pip install*)",
      "Bash(ruff*)",
      "Read(*)",
      "Edit(src/**)",
      "Edit(configs/**)",
      "Edit(tests/**)",
      "Edit(scripts/**)"
    ]
  },
  "env": {
    "INSIDE_CLAUDE_CODE": "1",
    "PYTHONPATH": "${workspaceFolder}/src"
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "ruff check --fix $FILE 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

#### 2.2 Create Skills

**`.claude/skills/protein-encoding/SKILL.md`**:
```markdown
---
name: protein-encoding
description: ESM-2 protein embeddings, encoder integration, pooling strategies
allowed-tools: [Read, Edit, Grep, Glob, Bash]
---

# Protein Encoding Skill

## Critical Rules
1. **NEVER modify ESM-2 weights** - always keep frozen during training
2. **Use attention pooling** (BoM-Pooling, window=80), NOT mean pooling
3. **LoRA on k/v matrices only** for protein tasks (differs from NLP)

## ESM-2 Models
| Model | Parameters | Embedding Dim | Recommended |
|-------|------------|---------------|-------------|
| esm2_t33_650M_UR50D | 650M | 1,280 | ✓ Best efficiency |
| esm2_t36_3B_UR50D | 3B | 2,560 | More capacity |

## Key Files
- src/models/protein_encoder.py - Encoder implementations
- src/models/pooling.py - Pooling strategies
- configs/encoder/ - Encoder configurations

## Integration Pattern
```
ESM-2 (frozen) → Per-residue [L, 1280]
    ↓
Attention Pooling → [1, 1280]
    ↓
MLP Projector → [1, LLM_dim]
    ↓
LLM (with LoRA)
```
```

**`.claude/skills/rl-training/SKILL.md`**:
```markdown
---
name: rl-training
description: GRPO, DPO, veRL framework, reinforcement learning for LLMs
allowed-tools: [Read, Edit, Grep, Glob, Bash]
---

# RL Training Skill

## Recommended Methods
| Method | Memory | Complexity | Best For |
|--------|--------|------------|----------|
| GRPO | 50% less than PPO | Medium | Reasoning tasks |
| DPO | Low | Low | Simple preference learning |
| PPO | High | High | Avoid unless necessary |

## Key Hyperparameters
- **RL Learning Rate**: 5e-6 (much lower than SFT!)
- **SFT Learning Rate**: 2e-4
- **LoRA Rank**: r=8 (minimum r=4)

## veRL Configuration
See configs/training/grpo.yaml for full config.

## Training Pipeline
1. SFT with QLoRA (Phase 1)
2. GRPO/DPO alignment (Phase 2)

## Key Files
- src/training/grpo_trainer.py
- src/training/dpo_trainer.py
- configs/training/
```

#### 2.3 Create Commands

**`.claude/commands/train.md`**:
```markdown
---
description: Launch training run with Hydra config
---

Launch training:
```bash
python scripts/train.py $ARGUMENTS
```

## Examples
```bash
# Basic SFT
python scripts/train.py experiment=baseline_sft

# Override learning rate
python scripts/train.py training.lr=1e-4

# Different model
python scripts/train.py model=llama3_8b

# Hyperparameter sweep
python scripts/train.py --multirun training.lr=1e-4,2e-4,5e-4

# Full pipeline
python scripts/train.py experiment=full_pipeline
```
```

**`.claude/commands/eval.md`**:
```markdown
---
description: Run evaluation on benchmarks
---

Run evaluation:
```bash
python scripts/evaluate.py $ARGUMENTS
```

## Examples
```bash
# GO term prediction
python scripts/evaluate.py evaluation=go_prediction

# All benchmarks
python scripts/evaluate.py evaluation=all

# Specific checkpoint
python scripts/evaluate.py checkpoint_path=/path/to/checkpoint
```
```

#### 2.4 Create Agents

**`.claude/agents/experiment-runner.md`**:
```markdown
---
name: experiment-runner
description: Launch and monitor training experiments
---

# Experiment Runner Agent

You are an agent specialized in launching and monitoring protein-LLM training experiments.

## Capabilities
1. Validate configurations before launch
2. Start training with appropriate configs
3. Monitor training progress via logs
4. Report metrics and checkpoints

## Workflow
1. Check GPU availability: `nvidia-smi`
2. Validate config: `python scripts/train.py --cfg job`
3. Launch training: `python scripts/train.py experiment=...`
4. Monitor: `tail -f logs/*/train.log`

## Environment Check
Before any experiment:
```bash
source /home/yeopjin/orcd/pool/init_protein_llm.sh
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```
```

---

### Phase 3: Hydra Configuration System

**Goal**: Implement hierarchical configuration management

#### 3.1 Main Config (configs/config.yaml)
```yaml
defaults:
  - model: qwen2_7b
  - encoder: esm2_650m
  - data: mol_instructions
  - training: sft_qlora
  - _self_

# Project metadata
project_name: protein_llm
experiment_name: ${now:%Y-%m-%d}_${model.name}_${training.method}

# Paths
paths:
  data_dir: ${oc.env:DATA_DIR,./data}
  raw_dir: ${paths.data_dir}/raw
  processed_dir: ${paths.data_dir}/processed
  checkpoint_dir: ${paths.data_dir}/checkpoints/${experiment_name}
  log_dir: ./logs/${experiment_name}

# Hardware
hardware:
  n_gpus: 8
  precision: bf16

# Logging
logging:
  wandb:
    enabled: true
    project: ${project_name}
    name: ${experiment_name}
  tensorboard:
    enabled: true
    log_dir: ${paths.log_dir}/tensorboard

# Hydra settings
hydra:
  run:
    dir: ${paths.log_dir}
  sweep:
    dir: ${paths.log_dir}/multirun
  job:
    chdir: false
```

#### 3.2 Model Configs

**configs/model/qwen2_7b.yaml**:
```yaml
name: qwen2_7b
path: Qwen/Qwen2.5-7B-Instruct
type: causal_lm

architecture:
  hidden_size: 4096
  num_attention_heads: 32
  num_layers: 32
  vocab_size: 152064

generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true
```

**configs/model/llama3_8b.yaml**:
```yaml
name: llama3_8b
path: meta-llama/Llama-3.1-8B-Instruct
type: causal_lm

architecture:
  hidden_size: 4096
  num_attention_heads: 32
  num_layers: 32
  vocab_size: 128256

generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true
```

#### 3.3 Encoder Configs

**configs/encoder/esm2_650m.yaml**:
```yaml
name: esm2_650m
model_name: esm2_t33_650M_UR50D
embedding_dim: 1280
num_layers: 33

# Always frozen
freeze: true

# Pooling strategy
pooling:
  method: attention  # attention, mean, cls, last
  window_size: 80    # For BoM-Pooling

# Projector settings
projector:
  type: mlp
  hidden_dim: 2048
  output_dim: ${model.architecture.hidden_size}
  num_layers: 2
  activation: gelu
  dropout: 0.1
```

#### 3.4 Training Configs

**configs/training/sft_qlora.yaml**:
```yaml
method: sft_qlora
framework: trl  # trl or verl

# QLoRA quantization
quantization:
  enabled: true
  bits: 4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

# LoRA configuration
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - k_proj
    - v_proj
  bias: none
  task_type: CAUSAL_LM

# Training parameters
lr: 2e-4
epochs: 3
batch_size: 8
gradient_accumulation_steps: 4
max_seq_length: 2048
warmup_ratio: 0.03
max_grad_norm: 1.0
weight_decay: 0.01

# Optimizer
optimizer:
  type: adamw_8bit
  betas: [0.9, 0.999]
  eps: 1e-8

lr_scheduler:
  type: cosine
  num_warmup_steps: 100

# Checkpointing
save_strategy: steps
save_steps: 500
save_total_limit: 3

# Logging
logging_steps: 10
eval_steps: 100
```

**configs/training/grpo.yaml**:
```yaml
method: grpo
framework: verl

# GRPO-specific settings
grpo:
  group_size: 4          # Number of completions per prompt
  temperature: 1.0
  use_kl_penalty: false  # DAPO improvement
  normalize_advantages: false  # Dr. GRPO improvement

# Training parameters
lr: 5e-6  # Much lower than SFT!
epochs: 1
batch_size: 4
gradient_accumulation_steps: 8
max_seq_length: 2048
max_grad_norm: 1.0

# LoRA (same as SFT)
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - k_proj
    - v_proj

# Rollout settings
rollout:
  engine: vllm
  temperature: 1.0
  top_p: 0.95
  max_tokens: 512

# Checkpointing
save_steps: 100
eval_steps: 50
```

#### 3.5 Experiment Configs

**configs/experiment/baseline_sft.yaml**:
```yaml
# @package _global_

defaults:
  - override /model: qwen2_7b
  - override /encoder: esm2_650m
  - override /data: mol_instructions
  - override /training: sft_qlora

experiment_name: baseline_sft_${now:%Y%m%d_%H%M%S}

# Override specific settings
training:
  epochs: 3
  lr: 2e-4
```

**configs/experiment/full_pipeline.yaml**:
```yaml
# @package _global_

# Phase 1: SFT
phase1:
  defaults:
    - override /training: sft_qlora
  training:
    epochs: 3

# Phase 2: GRPO
phase2:
  defaults:
    - override /training: grpo
  training:
    epochs: 1
    checkpoint: ${paths.checkpoint_dir}/phase1/best

experiment_name: full_pipeline_${now:%Y%m%d_%H%M%S}
```

---

### Phase 4: Slim CLAUDE.md

**Goal**: Create concise CLAUDE.md that points to detailed docs

```markdown
# Post-Training Protein LLM

## Quick Reference
```bash
# Environment
source /home/yeopjin/orcd/pool/init_protein_llm.sh

# Train
python scripts/train.py experiment=baseline_sft

# Evaluate
python scripts/evaluate.py evaluation=go_prediction

# Tests
pytest tests/ -v
```

## Architecture
ESM-2 650M (frozen) → Attention Pooling → MLP Projector → LLM (LoRA)

## Critical Rules
- NEVER modify ESM-2 weights
- LoRA on k/v matrices ONLY
- Use attention pooling, NOT mean
- TRITON_CACHE_DIR must be local

## Config Override Examples
```bash
python scripts/train.py model=llama3_8b training.lr=1e-4
python scripts/train.py --multirun training.lr=1e-4,2e-4
```

## Documentation
- [docs/architecture.md](docs/architecture.md) - Full architecture
- [docs/training_guide.md](docs/training_guide.md) - Training details
- [docs/troubleshooting.md](docs/troubleshooting.md) - Common issues

## Hardware
8x H100 80GB | CUDA 12.4 | Python 3.11
```

---

### Phase 5: Implementation Priority

#### Week 1: Foundation
- [ ] Create directory structure
- [ ] Move files to new locations
- [ ] Create pyproject.toml
- [ ] Slim down CLAUDE.md
- [ ] Create docs/ with moved content

#### Week 2: Claude Integration
- [ ] Set up .claude/settings.json
- [ ] Create protein-encoding skill
- [ ] Create rl-training skill
- [ ] Create /train and /eval commands
- [ ] Create experiment-runner agent

#### Week 3: Configuration System
- [ ] Install and configure Hydra
- [ ] Create config.yaml (main)
- [ ] Create model configs
- [ ] Create encoder configs
- [ ] Create training configs
- [ ] Create experiment presets

#### Week 4: Entry Points & Testing
- [ ] Create scripts/train.py
- [ ] Create scripts/evaluate.py
- [ ] Set up pytest with conftest.py
- [ ] Write initial tests for encoders
- [ ] Write initial tests for datasets

#### Week 5+: Core Implementation
- [ ] Implement src/models/projector.py
- [ ] Implement src/models/pooling.py
- [ ] Implement src/training/sft_trainer.py
- [ ] Implement src/training/grpo_trainer.py
- [ ] Implement evaluation benchmarks

---

## Part 4: Benefits Summary

| Improvement | Before | After | Impact |
|-------------|--------|-------|--------|
| CLAUDE.md size | 80 lines | <60 lines | Better context efficiency |
| Configuration | None | Hydra hierarchy | CLI overrides, sweeps, reproducibility |
| Documentation | Scattered | docs/ directory | Progressive disclosure |
| Claude features | None | Skills, commands, agents | Automated workflows |
| Testing | None | pytest suite | Quality assurance |
| Entry points | None | scripts/ directory | Clear execution paths |
| Package structure | Flat | Modular | Easier navigation |

---

## Part 5: References

### Claude Code
- [Best Practices for Claude Code](https://code.claude.com/docs/en/best-practices)
- [Writing a Good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Claude Code Showcase](https://github.com/ChrisWiles/claude-code-showcase)
- [Using CLAUDE.md Files](https://claude.com/blog/using-claude-md-files)

### LLM Post-Training
- [A Primer on LLM Post-Training (PyTorch)](https://pytorch.org/blog/a-primer-on-llm-post-training/)
- [Awesome LLM Post-Training](https://github.com/mbzuai-oryx/Awesome-LLM-Post-training)
- [Post-Training Methods (Red Hat)](https://developers.redhat.com/articles/2025/11/04/post-training-methods-language-models)

### Configuration & Project Structure
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [veRL Config Documentation](https://verl.readthedocs.io/en/latest/examples/config.html)
- [Python Monorepo Best Practices](https://www.tweag.io/blog/2023-04-04-python-monorepo-1/)
- [pyproject.toml Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

---

## Part 6: SFT Data Preparation

### 6.1 Current SFT Data Status

| Dataset | Format | Size | SFT Ready? |
|---------|--------|------|------------|
| **Mol-Instructions** | `{instruction, input, output}` | ~505K pairs | ✅ Ready |
| **Swiss-Prot** | `{sequence, annotations, GO_terms, function}` | ~570K sequences | ❌ Needs conversion |
| **IPD-PDB** | `{sequence, 3D_coords, structure_features}` | ~556K chains | ❌ Needs conversion |

### 6.2 Mol-Instructions (Ready to Use)

**Source**: `zjunlp/Mol-Instructions` (HuggingFace)
**Subset**: `Protein-oriented Instructions`
**Paper**: [ICLR 2024](https://arxiv.org/abs/2306.08018)

**Tasks covered**:
- Protein Design
- Catalytic Activity Prediction
- Protein Function Prediction
- Functional Description Generation
- Domain/Motif Prediction

**Data format (already SFT-ready)**:
```python
{
    "instruction": "Predict the function of this protein.",
    "input": "MKTLLIAAAVAA...",  # protein sequence
    "output": "This protein is a kinase that phosphorylates..."
}
```

**Loader**: `src/data/mol_instructions.py` ✅ Implemented

---

### 6.3 Swiss-Prot → SFT Conversion

**Raw Format**:
```python
{
    "accession": "P12345",
    "sequence": "MKTLLIAAAVAA...",
    "protein_name": "Kinase ABC",
    "organism": "Homo sapiens",
    "go_terms": ["GO:0005524", "GO:0016301", "GO:0004672"],
    "function": "Catalyzes the phosphorylation of serine residues...",
    "subcellular_location": "Cytoplasm",
    "keywords": ["Kinase", "ATP-binding", "Transferase"]
}
```

**SFT Conversion Templates**:

| Task | Instruction Template | Output |
|------|---------------------|--------|
| Function Prediction | "What is the function of this protein?" | `{function}` |
| GO Term Prediction | "Predict the Gene Ontology terms for this protein." | `{go_terms}` formatted |
| Subcellular Localization | "Where is this protein located in the cell?" | `{subcellular_location}` |
| Protein Naming | "What is this protein called and what organism is it from?" | `{protein_name} from {organism}` |
| Keyword Extraction | "List the functional keywords for this protein." | `{keywords}` formatted |

**Conversion Script**: `scripts/convert_swissprot_to_sft.py` (to be implemented)

```python
# Example conversion logic
def convert_swissprot_to_sft(entry: dict) -> List[dict]:
    """Convert a Swiss-Prot entry to multiple SFT instruction pairs."""
    sft_samples = []

    # Task 1: Function prediction
    if entry.get("function"):
        sft_samples.append({
            "instruction": "What is the function of this protein?",
            "input": entry["sequence"],
            "output": entry["function"]
        })

    # Task 2: GO term prediction
    if entry.get("go_terms"):
        go_str = ", ".join(entry["go_terms"])
        sft_samples.append({
            "instruction": "Predict the Gene Ontology (GO) terms for this protein.",
            "input": entry["sequence"],
            "output": f"The GO terms for this protein are: {go_str}"
        })

    # Task 3: Subcellular localization
    if entry.get("subcellular_location"):
        sft_samples.append({
            "instruction": "Where is this protein located in the cell?",
            "input": entry["sequence"],
            "output": f"This protein is located in the {entry['subcellular_location']}."
        })

    # Task 4: Multi-task combined
    sft_samples.append({
        "instruction": "Describe this protein's function, location, and key features.",
        "input": entry["sequence"],
        "output": f"{entry['protein_name']} is a protein from {entry['organism']}. "
                  f"Function: {entry.get('function', 'Unknown')}. "
                  f"Location: {entry.get('subcellular_location', 'Unknown')}. "
                  f"GO terms: {', '.join(entry.get('go_terms', []))}."
    })

    return sft_samples
```

**Expected Output**: ~2-3M SFT pairs (4-5 tasks × 570K proteins)

---

### 6.4 IPD-PDB → SFT Conversion

**Raw Format**:
```python
{
    "chain_id": "1ABC_A",
    "sequence": "MKTLLIAAAVAA...",
    "coordinates": torch.Tensor([L, 14, 3]),  # 3D atom coordinates
    "resolution": 2.1,  # Ångströms
    "secondary_structure": "HHHHHCCCCEEEEE...",  # H=helix, E=sheet, C=coil
    "chain_length": 256
}
```

**SFT Conversion Templates**:

| Task | Instruction Template | Output |
|------|---------------------|--------|
| Secondary Structure | "Predict the secondary structure of this protein." | `{secondary_structure}` |
| Structure Quality | "What is the resolution of this protein's structure?" | `{resolution}Å` |
| Length Estimation | "How many residues are in this protein?" | `{chain_length} residues` |
| Structure Description | "Describe the structural properties of this protein." | Combined description |

**Conversion Script**: `scripts/convert_pdb_to_sft.py` (to be implemented)

```python
def convert_pdb_to_sft(entry: dict) -> List[dict]:
    """Convert an IPD-PDB entry to SFT instruction pairs."""
    sft_samples = []

    # Task 1: Secondary structure prediction
    if entry.get("secondary_structure"):
        # Convert to human-readable format
        ss = entry["secondary_structure"]
        helix_pct = ss.count('H') / len(ss) * 100
        sheet_pct = ss.count('E') / len(ss) * 100
        coil_pct = ss.count('C') / len(ss) * 100

        sft_samples.append({
            "instruction": "Predict the secondary structure composition of this protein.",
            "input": entry["sequence"],
            "output": f"This protein contains approximately {helix_pct:.1f}% alpha-helix, "
                      f"{sheet_pct:.1f}% beta-sheet, and {coil_pct:.1f}% coil/loop regions."
        })

    # Task 2: Structure quality assessment
    if entry.get("resolution"):
        quality = "high" if entry["resolution"] < 2.0 else "medium" if entry["resolution"] < 3.0 else "low"
        sft_samples.append({
            "instruction": "Assess the quality of this protein's experimental structure.",
            "input": entry["sequence"],
            "output": f"This protein has a {quality}-resolution structure at {entry['resolution']}Å."
        })

    # Task 3: Length-based properties
    length = entry.get("chain_length", len(entry["sequence"]))
    size_class = "small" if length < 100 else "medium" if length < 300 else "large"
    sft_samples.append({
        "instruction": "Describe the size of this protein.",
        "input": entry["sequence"],
        "output": f"This is a {size_class} protein with {length} amino acid residues."
    })

    return sft_samples
```

**Expected Output**: ~1-2M SFT pairs (3-4 tasks × 556K chains)

---

### 6.5 Combined SFT Dataset Strategy

**Phase 1 - Initial Training**:
- Use Mol-Instructions only (~505K pairs)
- Validates pipeline works end-to-end

**Phase 2 - Extended Training**:
- Convert Swiss-Prot → SFT format (~2M pairs)
- Combine with Mol-Instructions
- Focus on function/GO prediction tasks

**Phase 3 - Structure-Aware Training**:
- Convert IPD-PDB → SFT format (~1.5M pairs)
- Add structure-related tasks
- Total: ~4M SFT pairs

---

### 6.6 Data Conversion Implementation Plan

#### Files to Create

```
src/data/
├── swissprot_converter.py    # Swiss-Prot → SFT conversion
├── pdb_converter.py          # IPD-PDB → SFT conversion
└── combined_dataset.py       # Unified SFT dataset loader

scripts/
├── convert_swissprot_to_sft.py   # CLI for Swiss-Prot conversion
├── convert_pdb_to_sft.py         # CLI for PDB conversion
└── merge_sft_datasets.py         # Merge multiple SFT sources

configs/data/
├── swissprot_sft.yaml        # Swiss-Prot SFT config
├── pdb_sft.yaml              # PDB SFT config
└── combined_sft.yaml         # Combined dataset config
```

#### Config Example: `configs/data/swissprot_sft.yaml`

```yaml
name: swissprot_sft
source: converted  # Pre-converted SFT format

paths:
  raw: ${paths.raw_dir}/swissprot
  processed: ${paths.processed_dir}/swissprot_sft

# Conversion settings
conversion:
  tasks:
    - function_prediction
    - go_term_prediction
    - subcellular_localization
    - keyword_extraction

  # Instruction variations for diversity
  instruction_variations: true
  num_variations_per_task: 3

# Filtering
filters:
  min_sequence_length: 50
  max_sequence_length: 1000
  require_function: true
  require_go_terms: false

# Splits
splits:
  train: 0.9
  validation: 0.05
  test: 0.05
```

#### Config Example: `configs/data/combined_sft.yaml`

```yaml
name: combined_sft
source: multiple

datasets:
  - mol_instructions:
      weight: 1.0  # Original weight
  - swissprot_sft:
      weight: 0.5  # Downweight to balance
  - pdb_sft:
      weight: 0.3  # Structure tasks less common

# Sampling strategy
sampling:
  strategy: weighted  # weighted, uniform, or temperature
  temperature: 1.0
  shuffle: true

paths:
  processed: ${paths.processed_dir}/combined_sft

splits:
  train: 0.9
  validation: 0.05
  test: 0.05
```

---

### 6.7 Instruction Diversity

To improve model generalization, use multiple instruction phrasings per task:

**Function Prediction Variations**:
```python
FUNCTION_INSTRUCTIONS = [
    "What is the function of this protein?",
    "Describe the biological function of this protein.",
    "What does this protein do?",
    "Explain the role of this protein in the cell.",
    "Predict the molecular function of this protein sequence.",
]
```

**GO Term Prediction Variations**:
```python
GO_INSTRUCTIONS = [
    "Predict the Gene Ontology terms for this protein.",
    "What GO terms are associated with this protein?",
    "List the GO annotations for this protein sequence.",
    "Identify the Gene Ontology classifications for this protein.",
    "What molecular functions, biological processes, and cellular components are associated with this protein?",
]
```

**Subcellular Localization Variations**:
```python
LOCATION_INSTRUCTIONS = [
    "Where is this protein located in the cell?",
    "Predict the subcellular localization of this protein.",
    "In which cellular compartment is this protein found?",
    "Identify the cellular location of this protein.",
    "Where in the cell does this protein function?",
]
```

---

### 6.8 Data Quality Considerations

| Consideration | Strategy |
|---------------|----------|
| **Sequence length** | Filter to 50-1000 aa for training stability |
| **Annotation completeness** | Require at least one annotation type |
| **Duplicate removal** | Cluster at 90% identity, keep representatives |
| **Data leakage** | Ensure train/val/test splits by protein family |
| **Label noise** | Prefer reviewed (Swiss-Prot) over unreviewed (TrEMBL) |
| **Class imbalance** | Oversample rare GO terms/functions |

---

*Updated: 2026-02-18*
*Added: SFT Data Preparation section*
*Status: Ready for implementation*
