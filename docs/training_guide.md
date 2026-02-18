# Training Guide

## Prerequisites

1. Activate environment:
```bash
source /home/yeopjin/orcd/pool/init_protein_llm.sh
```

2. Verify setup:
```bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
nvidia-smi
```

## Quick Start

### Basic SFT Training
```bash
python scripts/train.py experiment=baseline_sft
```

### With Custom Settings
```bash
python scripts/train.py \
    experiment=baseline_sft \
    model=llama3_8b \
    training.lr=1e-4 \
    training.epochs=5
```

## Configuration System

### Hydra Basics
```bash
# Print resolved config
python scripts/train.py --cfg job

# Override single value
python scripts/train.py training.lr=1e-4

# Override nested value
python scripts/train.py model.architecture.hidden_size=4096

# Use different config file
python scripts/train.py model=llama3_8b

# Hyperparameter sweep
python scripts/train.py --multirun training.lr=1e-4,2e-4,5e-4
```

### Config Files
```
configs/
├── config.yaml          # Main config
├── model/               # LLM configs
│   ├── qwen2_7b.yaml
│   └── llama3_8b.yaml
├── encoder/             # Protein encoder configs
│   ├── esm2_650m.yaml
│   └── esm2_3b.yaml
├── training/            # Training method configs
│   ├── sft_qlora.yaml
│   ├── grpo.yaml
│   └── dpo.yaml
└── experiment/          # Preset experiments
    ├── baseline_sft.yaml
    └── full_pipeline.yaml
```

## Training Phases

### Phase 1: SFT with QLoRA

```bash
python scripts/train.py training=sft_qlora
```

Key settings:
- `training.lr`: 2e-4 (default)
- `training.epochs`: 3
- `training.batch_size`: 8
- `training.lora.r`: 8

### Phase 2: GRPO Alignment

```bash
python scripts/train.py \
    training=grpo \
    training.resume_from=/path/to/sft/checkpoint
```

Key settings:
- `training.lr`: 5e-6 (much lower than SFT!)
- `training.grpo.group_size`: 4
- `training.epochs`: 1

## Monitoring

### Weights & Biases
```bash
# View logs
wandb sync --view

# Dashboard
open https://wandb.ai/your-project
```

### TensorBoard
```bash
tensorboard --logdir=./logs
```

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Common Issues

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train.py training.batch_size=4

# Enable gradient checkpointing
python scripts/train.py training.gradient_checkpointing=true
```

### Slow Triton Compilation
```bash
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
```

### Flash Attention Issues
```bash
pip install flash-attn --no-build-isolation
```

## Best Practices

1. **Always validate config before training**:
   ```bash
   python scripts/train.py --cfg job
   ```

2. **Start with small experiments**:
   ```bash
   python scripts/train.py training.epochs=1 data.subset=0.1
   ```

3. **Save checkpoints frequently**:
   ```bash
   python scripts/train.py training.save_steps=100
   ```

4. **Monitor GPU memory**:
   ```bash
   watch -n 1 nvidia-smi
   ```
