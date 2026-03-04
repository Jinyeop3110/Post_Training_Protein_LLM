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
# Auto-named experiment (e.g., sft_qlora_esm3_qwen3_4b_0220_153000)
python scripts/train.py

# Custom experiment name
python scripts/train.py experiment_name=my_sft_run

# With custom settings
python scripts/train.py experiment_name=my_sft_run \
    model=qwen3_4b training.lr=1e-4 training.epochs=5
```

### GRPO After SFT
```bash
python scripts/train.py training=grpo experiment_name=my_grpo \
    parent_experiment=my_sft_run
```

### Evaluate
```bash
# By experiment name (auto-detects checkpoint)
python scripts/evaluate.py experiment_name=my_sft_run evaluation.name=all

# By explicit checkpoint path
python scripts/evaluate.py checkpoint_path=results/.../checkpoints/protein_llm
```

## Experiment Directory

All artifacts for a training run go under `results/{experiment_name}/`:

```
results/{experiment_name}/
├── config.yaml            # Full resolved Hydra config
├── lineage.json           # Stage, parent experiment, timestamps
├── training_args.json     # Hyperparameters
├── metrics.json           # Final train/eval metrics
├── checkpoints/
│   └── protein_llm/       # ProteinLLM save (config, pooling, projector, adapter)
├── logs/
│   └── .hydra/            # Hydra config snapshots
└── eval/
    └── {task}_metrics.json
```

### Lineage Tracking

`lineage.json` records the base→SFT→GRPO pipeline flow so you can trace
which SFT checkpoint a GRPO run was built from:

```bash
# View lineage
cat results/my_grpo/lineage.json
# → "parent_experiment": "my_sft_run", "stage": "grpo"

# List all experiments
python -c "from src.utils.experiment import list_experiments; \
  [print(e['name'], e['stage'], e.get('parent_experiment', '')) \
   for e in list_experiments('./results')]"
```

## Configuration System

### Hydra Basics
```bash
# Print resolved config
python scripts/train.py --cfg job

# Override single value
python scripts/train.py training.lr=1e-4

# Use different config file
python scripts/train.py model=llama3_8b

# Hyperparameter sweep
python scripts/train.py --multirun training.lr=1e-4,2e-4,5e-4
```

### Config Files
```
configs/
├── config.yaml          # Main config (experiment paths, approach, etc.)
├── model/               # LLM configs
│   ├── qwen3_4b.yaml
│   ├── qwen3_8b.yaml
│   └── qwen3_14b.yaml
├── encoder/             # Protein encoder configs
│   └── esm3_small.yaml  # Default — ESM-3 with pooling/projector settings
├── data/                # Dataset configs
│   ├── mol_instructions.yaml
│   ├── combined.yaml
│   ├── cafa5_go.yaml           # GO prediction (GRPO)
│   ├── megascale_stability.yaml # Stability/ddG (GRPO)
│   └── structure_quality.yaml  # Structure quality (GRPO)
├── experiment/          # Experiment presets
│   ├── grpo_go_prediction.yaml
│   ├── grpo_stability.yaml
│   └── grpo_structure.yaml
└── training/            # Training method configs
    ├── sft_qlora.yaml   # Default — SFT with 4-bit quantization
    ├── sft_lora.yaml    # SFT without quantization
    ├── grpo.yaml        # GRPO alignment
    └── dpo.yaml         # DPO alignment
```

### Key Overrides
```bash
# Approach (text / esm3)
python scripts/train.py approach=text

# Projector type (mlp / perceiver)
python scripts/train.py encoder.projector.type=perceiver

# Limit dataset for testing
python scripts/train.py data.limit=100

# Disable wandb
python scripts/train.py logging.wandb.enabled=false

# Single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train.py
```

## Training Phases

### Phase 1: SFT with LoRA (FSDP)

```bash
python scripts/train.py training=sft_lora experiment_name=sft_esm3_50k
```

Key settings:
- `training.lr`: 2e-4 (default)
- `training.projector_lr`: 2e-3 (10x base — essential for projector convergence)
- `training.epochs`: 3
- `training.batch_size`: 16
- `training.lora.r`: 8 (targets: all linear layers)
- `training.fsdp.enabled`: true (default — shards LLM across GPUs)

### Phase 2: GRPO Alignment

```bash
python scripts/train.py training=grpo experiment_name=grpo_esmfold \
    parent_experiment=sft_esm3_50k
```

Key settings:
- `training.lr`: 5e-6 (much lower than SFT!)
- `training.projector_lr`: 2e-5 (10x base)
- `training.grpo.group_size`: 4 (completions per prompt)
- `training.epochs`: 1
- Reward functions: GO (F1), PPI (accuracy), Stability (Gaussian), ESMFold (pLDDT)

## Monitoring

### Weights & Biases
```bash
# View logs
wandb sync --view

# Dashboard
open https://wandb.ai/your-project
```

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

## FSDP (Fully Sharded Data Parallel)

FSDP is **enabled by default** in `configs/training/sft_lora.yaml`. It shards LLM weights across all GPUs, saving ~14 GB/GPU for 8B models.

```bash
# Default: FSDP enabled
python scripts/train.py experiment_name=my_run

# Disable FSDP (single-GPU or debugging)
python scripts/train.py training.fsdp.enabled=false
```

Key details:
- Strategy: `full_shard` (ZeRO-3) — shards params, gradients, and optimizer states
- ESM-3 encoder stays replicated (not FSDP-sharded) — it's frozen and needs no gradient sharding
- `embed_tokens` weights are cached before FSDP shards the model (used for multimodal input preparation)
- Activation checkpointing replaces gradient checkpointing when FSDP is active (avoids redundant AllGather ops)
- `torch_compile=true` is compatible with FSDP and enabled by default
- Requires `torchrun` launch (used by `scripts/launch_train.sh`)

## Common Issues

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train.py training.batch_size=4

# FSDP activation checkpointing is already enabled by default
```

### Slow Triton Compilation
```bash
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
```

### Flash Attention Issues
```bash
pip install flash-attn --no-build-isolation
```

### Multi-GPU: Missing eval_loss
When running without `torchrun` on a multi-GPU node, HF Trainer uses DataParallel
which multiplies effective batch size. With `drop_last=True` and small eval sets,
this can result in 0 eval batches. Fix: use `CUDA_VISIBLE_DEVICES=0` for single-GPU runs.

## Downstream Tasks

### Available Tasks

| Task | Dataset | Reward | Config | Metric |
|------|---------|--------|--------|--------|
| GO Prediction | CAFA 5 (10K) | Set-F1 on GO terms | `data=cafa5_go` | Fmax |
| Stability/ddG | Mega-Scale (10K) | Gaussian on MAE | `data=megascale_stability` | Spearman ρ |
| Structure Quality | AlphaFold DB (10K) | pLDDT alignment | `data=structure_quality` | pLDDT MAE |

### Downloading Task Data

```bash
# Download all downstream task datasets (10K samples each)
python scripts/data/download_cafa.py --max_samples 10000
python scripts/data/download_megascale.py --max_samples 10000
python scripts/data/download_structure_quality.py --source alphafold --max_samples 10000

# Structure quality with ESMFold (requires GPU, more accurate)
python scripts/data/download_structure_quality.py --source esmfold --max_samples 10000
```

Data is saved to `data/processed/{task_name}/`.

### GRPO Training per Task

Chain from a trained SFT checkpoint:

```bash
# GO term prediction
python scripts/train.py experiment=grpo_go_prediction \
    parent_experiment=my_sft_run

# Stability prediction (ddG)
python scripts/train.py experiment=grpo_stability \
    parent_experiment=my_sft_run

# Structure quality (pLDDT)
python scripts/train.py experiment=grpo_structure \
    parent_experiment=my_sft_run
```

### Evaluating Downstream Tasks

```bash
# GO prediction (Fmax metric)
python scripts/evaluate.py experiment_name=grpo_go_run evaluation.name=go_prediction

# Stability (Spearman correlation)
python scripts/evaluate.py experiment_name=grpo_stability_run evaluation.name=stability

# All evaluations
python scripts/evaluate.py experiment_name=my_run evaluation.name=all
```

## Best Practices

1. **Always validate config before training**:
   ```bash
   python scripts/train.py --cfg job
   ```

2. **Start with small experiments**:
   ```bash
   python scripts/train.py data.limit=100 training.epochs=1 \
       logging.wandb.enabled=false
   ```

3. **Use custom experiment names** for important runs:
   ```bash
   python scripts/train.py experiment_name=sft_esm3_mlp_50k_v2
   ```

4. **Monitor GPU memory**:
   ```bash
   watch -n 1 nvidia-smi
   ```

5. **Check lineage** before GRPO:
   ```bash
   cat results/{parent_experiment}/lineage.json
   ```
