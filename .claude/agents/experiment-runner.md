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

## Pre-flight Checklist
Before any experiment, run these checks:

```bash
# 1. Activate environment
source /home/yeopjin/orcd/pool/init_protein_llm.sh

# 2. Check GPU availability
nvidia-smi

# 3. Verify imports
python -c "import torch, transformers, peft; print(f'GPUs: {torch.cuda.device_count()}')"

# 4. Validate config
python scripts/train.py --cfg job
```

## Launch Workflow

### Step 1: Select Experiment
```bash
# List available experiments
ls configs/experiment/

# Preview config
python scripts/train.py experiment=baseline_sft --cfg job
```

### Step 2: Launch Training
```bash
# Standard launch
python scripts/train.py experiment=baseline_sft

# With overrides
python scripts/train.py experiment=baseline_sft training.lr=1e-4

# Background launch with logging
nohup python scripts/train.py experiment=baseline_sft > logs/train.log 2>&1 &
```

### Step 3: Monitor Progress
```bash
# Watch logs
tail -f logs/*/train.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check wandb (if enabled)
wandb sync --view
```

## Common Experiments

| Experiment | Description | Command |
|------------|-------------|---------|
| baseline_sft | Basic SFT with QLoRA | `experiment=baseline_sft` |
| full_pipeline | SFT + GRPO | `experiment=full_pipeline` |
| ablation_pooling | Compare pooling methods | `experiment=ablation_pooling` |

## Troubleshooting

### OOM Errors
Reduce batch size or enable gradient checkpointing:
```bash
python scripts/train.py training.batch_size=4 training.gradient_checkpointing=true
```

### Slow Startup
Check Triton cache:
```bash
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
```
