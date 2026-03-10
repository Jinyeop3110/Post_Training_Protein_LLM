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
├── main_SFT/            # SFT experiment presets
└── main_RL/             # RL experiment presets
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
python scripts/train.py main_SFT=sft_esm3_mlp_combined

# Hyperparameter sweep
python scripts/train.py --multirun training.lr=1e-4,2e-4,5e-4
```

## Config Composition
```yaml
# config.yaml
defaults:
  - model: qwen2_7b
  - encoder: esm3_small
  - data: mol_instructions
  - training: sft_lora
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
# In main_SFT/sft_esm3_mlp_combined.yaml
# @package _global_

defaults:
  - override /model: qwen3_8b
  - override /training: sft_lora
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
- configs/main_SFT/*.yaml - SFT experiment presets
- configs/main_RL/*.yaml - RL experiment presets
