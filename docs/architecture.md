# Architecture

## Overview

The system supports two encoding approaches, selected via `approach: text|esm3` in `configs/config.yaml`.

```
                        Protein Sequence
                              |
              ┌───────────────┼───────────────┐
              |                               |
      approach: text                   approach: esm3
              |                               |
              ▼                               ▼
     ┌────────────┐                    ┌──────────────┐
     │  Raw text   │                    │  ESM-3 Small │
     │  <protein>  │                    │   (frozen)   │
     │  MKTL...    │                    │  1536-dim    │
     │  </protein> │                    │  esm3-sm-    │
     │             │                    │  open-v1     │
     └──────┬─────┘                    └──────┬───────┘
            |                                 |
            |                          ┌──────┴───────┐
            |                          │ Pooling +    │
            |                          │ Projector    │
            |                          │ (see below)  │
            |                          └──────┬───────┘
            |                                 |
            └─────────────────────────────────┘
                            |
                            ▼
                   ┌─────────────────┐
                   │ LLM + LoRA r=8  │  ◄── LoRA on k/v matrices ONLY
                   │ (Qwen3-4B /     │
                   │  Qwen-2.5-7B /  │
                   │  Llama-3.1-8B)  │
                   └────────┬────────┘
                            |
                            ▼
                        Response

  D = LLM hidden_size (2560 for Qwen3-4B, 4096 for Qwen-2.5-7B/Llama)
```

#### ESM-3 Projector Types

The ESM-3 path supports two projector types (`encoder.projector.type`):

**MLP (default)**: AttentionPooling (32 tokens) → 2-layer MLP
- ~20M trainable params (pooling ~9.5M + projector ~8.4M + LoRA ~2M)
- 18.3 GB peak GPU, 34 MB checkpoint

**Perceiver Resampler**: Replaces both pooling and projector in a single module
- Self-Attention → Cross-Attention → FFN per layer (2 layers default)
- ~132M trainable params (Perceiver ~130M + LoRA ~2M)
- 19.4 GB peak GPU, 520 MB checkpoint, 12% slower than MLP
- Config: `encoder.projector.type=perceiver encoder.projector.perceiver_layers=2`

### Approach Config Switching

```yaml
# configs/config.yaml
approach: esm3   # text | esm3

# text  -> No encoder. Protein fed as raw text tokens: <protein>MKTL...</protein>
# esm3  -> Frozen ESM-3 encoder -> attention pooling -> MLP projector -> LLM
```

Override from CLI:
```bash
python scripts/train.py approach=text          # Text-only, no encoder
python scripts/train.py approach=esm3          # ESM-3 path (default)
```

## Component Details

### 1. Protein Encoders

All encoders are **always frozen** -- never modify their weights during training.

#### ESM-3

| Model | Parameters | Embedding Dim | VRAM |
|-------|------------|---------------|------|
| esm3-sm-open-v1 | 1.4B | 1,536 | ~6GB |

ESM-3 is a multimodal generative model trained on 2.78B proteins with sequence, structure, and function tokens. We use the small open variant for embedding extraction.

Config: `configs/encoder/esm3_small.yaml`

**Critical**: Always keep the ESM-3 encoder frozen. Do not update weights during training.

### 2. Attention Pooling

Uses BoM-Pooling (Bag of Motifs) with learned attention:
- 32 output tokens
- Learns which residues are most important
- Better than mean pooling which treats all residues equally
- Number of attention heads: 8
- Dropout: 0.1, layer norm enabled

### 3. Projector (MLP or Perceiver Resampler)

The projector maps encoder output to the LLM embedding space. Two types are supported:

#### MLP Projector (default, `encoder.projector.type: mlp`)

Two-layer MLP with GELU activation, preceded by AttentionPooling:

**ESM-3 (small) -> Qwen3-4B** (hidden_size=2560):
```python
Projector:
  Linear(1536, 2048) -> GELU -> Dropout(0.1)
  Linear(2048, 2560) -> GELU -> Dropout(0.1)
```

#### Perceiver Resampler (`encoder.projector.type: perceiver`)

Replaces both pooling AND projector as a single module. Uses learned query tokens
with cross-attention to attend to encoder output:

```
Per layer (default 2 layers):
  LayerNorm + Self-Attention (queries attend to queries)
  LayerNorm + Cross-Attention (queries attend to encoder output)
  LayerNorm + FFN (feed-forward network)
```

- Input projection: `Linear(encoder_dim, output_dim)` if dims differ
- Learned query tokens: `[num_queries, output_dim]` (32 × 2560 for Qwen3-4B)
- Source: `src/models/perceiver.py`

Projector dimensions are derived from config interpolation:
- `input_dim`: from `encoder.embedding_dim`
- `output_dim`: from `model.architecture.hidden_size`

### 4. LLM with LoRA

| LLM Option | Parameters | Hidden Size | Notes |
|------------|------------|-------------|-------|
| Qwen3-4B-Instruct-2507 | 4B | 2,560 | Default for fast iteration |
| Qwen-2.5-7B-Instruct | 7B | 4,096 | Larger option |
| Llama-3.1-8B-Instruct | 8B | 4,096 | Alternative |

LoRA Configuration:
- Rank: r=8 (minimum r=4)
- Alpha: 16
- Target: k_proj, v_proj ONLY
- Dropout: 0.05

## Training Stages

### Stage 1: SFT with QLoRA
- 4-bit quantization
- Train: Projector + LLM (LoRA)
- Freeze: Protein encoder (ESM-3)
- LR: 2e-4 (projector_lr: 2e-3 — 10x higher for projector)
- Epochs: 1-3
- wandb project: `protein-llm-sft`

### Stage 2: GRPO Alignment
- Load SFT checkpoint via `parent_experiment=<sft_experiment_name>`
- LR: 5e-6 (much lower!)
- Group size: 4 completions per prompt
- Epochs: 1
- Reward functions: GO (F1), PPI (accuracy), Stability (Gaussian), ESMFold (pLDDT)
- wandb project: `protein-llm-rl`

## Experiment Pipeline

All experiment artifacts are stored under a unified directory:

```
results/{experiment_name}/
├── config.yaml            # Full resolved Hydra config
├── lineage.json           # Stage, parent experiment, timestamps
├── training_args.json     # All hyperparameters
├── metrics.json           # Final train/eval metrics
├── checkpoints/
│   └── protein_llm/       # ProteinLLM save (config, pooling, projector, adapter)
├── logs/
│   ├── .hydra/            # Hydra config snapshots
│   └── tensorboard/       # TensorBoard events
└── eval/
    └── {task}_metrics.json
```

### Lineage Tracking

`lineage.json` records the base→SFT→GRPO pipeline flow:
```json
{
  "experiment_name": "grpo_esm3_mlp_esmfold",
  "stage": "grpo",
  "parent_experiment": "sft_esm3_mlp_50k",
  "parent_checkpoint": "results/sft_esm3_mlp_50k/checkpoints/protein_llm",
  "base_model": "Qwen/Qwen3-4B",
  "encoder": "esm3-sm-open-v1",
  "approach": "esm3",
  "projector_type": "mlp",
  "created_at": "...",
  "completed_at": "..."
}
```

### Usage Examples

```bash
# SFT with custom experiment name
python scripts/train.py experiment_name=sft_esm3_mlp_50k

# GRPO chaining from SFT
python scripts/train.py training=grpo experiment_name=grpo_esmfold \
  parent_experiment=sft_esm3_mlp_50k

# Evaluate (auto-detects checkpoint from experiment)
python scripts/evaluate.py experiment_name=sft_esm3_mlp_50k evaluation.name=all

# List experiments
python -c "from src.utils.experiment import list_experiments; \
  [print(e['name'], e['stage']) for e in list_experiments('./results')]"
```

Utilities: `src/utils/experiment.py` (write/read/complete lineage, resolve parent, list experiments)

## Downstream Task Pipeline

The system supports task-specific GRPO training with verifiable rewards:

```
Download Scripts              Configs                     GRPO Reward
─────────────                 ───────                     ───────────
download_cafa.py        →  cafa5_go.yaml            →  GO set-F1 score
download_megascale.py   →  megascale_stability.yaml →  Gaussian on ddG MAE
download_structure_quality.py → structure_quality.yaml → pLDDT alignment
```

### Data Flow

```
1. Download: scripts/data/download_*.py --max_samples 10000
       ↓
2. JSON instruction format: data/processed/{task}/
       ↓
3. GRPO Training: experiment=grpo_{task} parent_experiment=<sft>
       ↓
4. Evaluation: evaluation.name={task}
```

### Reward Functions (src/training/grpo_trainer.py)

| Task | Reward | Components |
|------|--------|------------|
| `go_prediction` | `compute_go_reward` | F1 on predicted vs ground truth GO terms |
| `stability` | `compute_stability_reward` | Gaussian decay on ddG MAE |
| `esmfold` | `compute_esmfold_reward` | Quality alignment (0.4) + pLDDT accuracy (0.3) + category match (0.3) |
| `ppi` | `compute_ppi_reward` | Binary accuracy on interaction prediction |

The ESMFold reward supports both live folding (protein sequence → ESMFold → pLDDT)
and pre-computed metrics (JSON with pLDDT/pTM from AlphaFold DB).

## Memory Budget (8x H100 80GB)

### ESM-3 + Qwen3-4B (Default Configuration)

| Component | VRAM per GPU |
|-----------|-------------|
| ESM-3 small (1.4B, frozen) | ~6GB |
| Qwen3-4B (4-bit) | ~3GB |
| LoRA adapters (r=8) | ~0.3GB |
| Activations | ~20GB |
| Gradients | ~15GB |
| **Total** | ~45GB |

