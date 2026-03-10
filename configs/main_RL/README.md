# GRPO RL Experiment Configs

Optimized for **4x NVIDIA L40S 46GB** with FSDP sharding across all GPUs.

## Configs

| Config | Task | Approach | Parent | Effective Batch | Epochs |
|--------|------|----------|--------|-----------------|--------|
| `grpo_structure_base` | Structure Quality | esm3 | none (fresh LoRA) | 64 | 1 |
| `grpo_structure_text_sft` | Structure Quality | text | text SFT | 64 | 1 |
| `grpo_structure_mlp_sft` | Structure Quality | esm3 | MLP SFT | 64 | 1 |
| `grpo_proteinlm_base` | ProteinLM Bench | text | none (fresh LoRA) | 32 | 5 |
| `grpo_proteinlm_text_sft` | ProteinLM Bench | text | text SFT | 32 | 3 |
| `grpo_proteinlm_mlp_sft` | ProteinLM Bench | esm3 | MLP SFT | 32 | 3 |

## Usage

```bash
# Structure Quality experiments
python scripts/train.py main_RL=grpo_structure_base
python scripts/train.py main_RL=grpo_structure_text_sft
python scripts/train.py main_RL=grpo_structure_mlp_sft

# ProteinLM Bench experiments
python scripts/train.py main_RL=grpo_proteinlm_base
python scripts/train.py main_RL=grpo_proteinlm_text_sft
python scripts/train.py main_RL=grpo_proteinlm_mlp_sft

# Override any parameter
python scripts/train.py main_RL=grpo_structure_base training.lr=1e-5
```

## Memory Budget (per GPU with FSDP/4)

| Component | Size |
|-----------|------|
| Qwen3-8B bf16 (FSDP sharded) | 4.2 GB |
| LoRA adapters (~100M params) | 0.2 GB |
| AdamW optimizer (FSDP sharded) | 0.2 GB |
| ESM-3 small (frozen, fp32) | 1.6 GB |
| Pooling + Projector (~30M) | 0.3 GB |
| ESMFold (structure quality only) | 3.0 GB |

**Peak per GPU:**

| Config Type | Static | Peak (gen+grad) | Headroom |
|------------|--------|-----------------|----------|
| Text-only | 4.6 GB | ~10 GB | **36 GB** |
| ESM3+MLP | 6.5 GB | ~12 GB | **34 GB** |
| ESM3+MLP+ESMFold | 9.4 GB | ~16 GB | **30 GB** |

FSDP gives massive headroom on L40S 46GB. The bottleneck is generation throughput, not memory.

## Hyperparameter Rationale

### Learning Rate
- **Base (no parent)**: `lr=5e-6` — standard GRPO rate for fresh LoRA
- **SFT parent**: `lr=2e-6` (ProteinLM) or `5e-6` (Structure) — lower for already-trained models
- **Projector LR**: 10x for random-init (`5e-5`), 5x for trained (`2.5e-5` or `1e-5`)
- Range for sweeps: `[1e-6, 2e-6, 5e-6, 1e-5]`

### Batch Size & Accumulation
- `batch_size=4`: 4 prompts per micro-batch per GPU
- Structure Quality: `grad_accum=4` → effective batch = 4×4×4 = **64 prompts**
- ProteinLM Bench: `grad_accum=2` → effective batch = 4×2×4 = **32 prompts** (small dataset)
- Each prompt generates `group_size=8` completions

### Group Size
- **8** (up from default 4): more completions per prompt gives better advantage estimates
- With FSDP headroom, group_size=8 costs only ~1.5 GB extra KV cache per GPU
- For sweeps: `[4, 8, 16]`

### Generation
- Structure Quality: `max_tokens=256` (descriptions are 1-2 sentences)
- ProteinLM Bench: `max_tokens=64` (MC answers: "option N" + brief reasoning)
- `temperature=1.0` for structure (diverse descriptions), `0.8` for MC (more focused)

### Training Duration
- Structure Quality: **1 epoch** (10K samples is enough for RL signal)
- ProteinLM Bench base: **5 epochs** (only 944 questions, needs more passes)
- ProteinLM Bench SFT: **3 epochs** (already has protein knowledge from SFT)

### GRPO Variants
- `use_kl_penalty: false` — DAPO improvement, no reference model needed
- `normalize_advantages: false` — Dr. GRPO improvement, prevents reward hacking
- KL penalty can be enabled for SFT-chained experiments: `training.grpo.use_kl_penalty=true training.grpo.kl_coef=0.05`

## Extension to Other Tasks

To add Stability or GO Prediction tasks:
1. Create `grpo_{task}_{model_form}.yaml` following this pattern
2. Set `data.task` to match the reward function key
3. Adjust `rollout.max_tokens` for expected output length
4. Use the same batch/lr settings as Structure Quality (similar dataset sizes)
