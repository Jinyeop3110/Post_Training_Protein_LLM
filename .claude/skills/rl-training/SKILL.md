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

### SFT Phase
- Learning Rate: 2e-4
- Epochs: 1-3
- LoRA Rank: r=8

### RL Phase
- Learning Rate: 5e-6 (much lower than SFT!)
- Epochs: 1
- LoRA Rank: r=8

## GRPO Configuration
```yaml
grpo:
  group_size: 4          # Completions per prompt
  temperature: 1.0
  use_kl_penalty: false  # DAPO improvement
  normalize_advantages: false  # Dr. GRPO
```

## veRL Setup
```bash
# Launch GRPO training
python scripts/train.py training=grpo

# With custom settings
python scripts/train.py training=grpo training.lr=1e-5
```

## Training Pipeline
1. **Phase 1: SFT with QLoRA**
   - 4-bit quantization
   - Train projector + LLM (LoRA)
   - Freeze ESM-3

2. **Phase 2: GRPO Alignment**
   - Load SFT checkpoint
   - Lower learning rate
   - Reward model or rule-based rewards

## Key Files
- src/training/sft_trainer.py - SFT implementation
- src/training/grpo_trainer.py - GRPO implementation
- configs/training/grpo.yaml - GRPO config
- configs/training/sft_qlora.yaml - SFT config
