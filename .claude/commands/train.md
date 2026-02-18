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

# Full pipeline (SFT + GRPO)
python scripts/train.py experiment=full_pipeline

# Resume from checkpoint
python scripts/train.py training.resume_from=/path/to/checkpoint
```

## Pre-flight Check
Before training, verify:
1. Environment: `source /home/yeopjin/orcd/pool/init_protein_llm.sh`
2. GPUs: `nvidia-smi`
3. Config: `python scripts/train.py --cfg job`
