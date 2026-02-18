---
name: hydra-configs
description: Hydra configuration patterns, CLI overrides, experiment sweeps
allowed-tools: [Read, Edit, Grep, Glob, Bash]
---

# Hydra Configuration Skill

## Directory Structure
```
configs/
├── config.yaml          # Main config with defaults
├── model/               # LLM configurations
├── encoder/             # Protein encoder configs
├── data/                # Dataset configurations
├── training/            # Training method configs
├── evaluation/          # Evaluation configs
└── experiment/          # Preset experiments
```

## Basic Usage
```bash
# Run with defaults
python scripts/train.py

# Override single value
python scripts/train.py training.lr=1e-4

# Override multiple values
python scripts/train.py model=llama3_8b training.lr=1e-4 training.epochs=5

# Use experiment preset
python scripts/train.py experiment=baseline_sft

# Hyperparameter sweep
python scripts/train.py --multirun training.lr=1e-4,2e-4,5e-4
```

## Config Composition
```yaml
# config.yaml
defaults:
  - model: qwen2_7b
  - encoder: esm2_650m
  - data: mol_instructions
  - training: sft_qlora
  - _self_
```

## Variable Interpolation
```yaml
paths:
  data_dir: ${oc.env:DATA_DIR,./data}
  checkpoint_dir: ${paths.data_dir}/checkpoints/${experiment_name}

experiment_name: ${now:%Y-%m-%d}_${model.name}
```

## Package Directives
```yaml
# In experiment/baseline_sft.yaml
# @package _global_

defaults:
  - override /model: qwen2_7b
  - override /training: sft_qlora
```

## Debugging
```bash
# Print resolved config
python scripts/train.py --cfg job

# Print config tree
python scripts/train.py --info defaults
```

## Key Files
- configs/config.yaml - Main entry point
- configs/experiment/*.yaml - Preset experiments
