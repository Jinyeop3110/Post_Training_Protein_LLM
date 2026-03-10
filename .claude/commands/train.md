---
description: Launch training run with Hydra config
---

Launch training:
```bash
python scripts/train.py $ARGUMENTS
```

## Examples
```bash
# SFT with MLP projector
python scripts/train.py main_SFT=sft_esm3_mlp_combined

# Override learning rate
python scripts/train.py training.lr=1e-4

# Different model
python scripts/train.py model=qwen3_8b

# Hyperparameter sweep
python scripts/train.py --multirun training.lr=1e-4,2e-4,5e-4

# GRPO (chain from SFT)
python scripts/train.py main_RL=grpo_go_prediction parent_experiment=my_sft

# Resume from checkpoint
python scripts/train.py training.resume_from=/path/to/checkpoint
```

## Pre-flight Check
Before training, verify:
1. Environment: `source /home/yeopjin/orcd/pool/init_protein_llm.sh`
2. GPUs: `nvidia-smi`
3. Config: `python scripts/train.py --cfg job`
