# Post-Training Protein LLM

Multimodal LLM for protein understanding: ESM-3 protein embeddings + LLM via SFT and GRPO reinforcement learning.

## Architecture

Three encoding approaches for comparison:

| Approach | Config | Description |
|----------|--------|-------------|
| **ESM-3 + MLP** | `approach=esm3 encoder.projector.type=mlp` | ESM-3 (frozen) → AttentionPooling (32 tokens) → MLP → LLM |
| **ESM-3 + Perceiver** | `approach=esm3 encoder.projector.type=perceiver` | ESM-3 (frozen) → PerceiverResampler → LLM |
| **Text-only** | `approach=text` | Raw sequence as `<protein>MKTL...</protein>` tokens |

**Default**: ESM-3 small (1536-dim, frozen) + Qwen3-8B (LoRA r=8 on all linear layers)

## Quick Start

```bash
# Activate environment
source /home/yeopjin/orcd/pool/init_protein_llm.sh

# Download data
python src/data/download.py --dataset mol_instructions

# Train (single GPU)
python scripts/train.py experiment=sft_esm3_mlp

# Train (8 GPU DDP)
bash scripts/launch_train.sh experiment=sft_esm3_mlp

# Evaluate
python scripts/evaluate.py experiment_name=<experiment> evaluation.name=all

# Inference
python scripts/inference.py --checkpoint results/.../checkpoints/protein_llm \
    --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ

# Tests
pytest tests/ -v
```

## Training Pipeline

```
SFT (Mol-Instructions, 445K samples)  →  GRPO (GO/Stability/ESMFold rewards)
         ↓                                          ↓
  results/{experiment}/                   results/{experiment}/
  ├── checkpoints/protein_llm/            ├── checkpoints/protein_llm/
  ├── config.yaml                         ├── lineage.json (parent_experiment)
  └── metrics.json                        └── metrics.json
```

### SFT Experiments

```bash
# MLP projector (default)
python scripts/train.py experiment=sft_esm3_mlp

# Perceiver Resampler
python scripts/train.py experiment=sft_esm3_perceiver

# Text-only baseline
python scripts/train.py experiment=sft_text

# Custom overrides
python scripts/train.py experiment=sft_esm3_mlp training.lr=1e-4 model=qwen3_4b
```

### GRPO (chains from SFT checkpoint)

```bash
python scripts/train.py experiment=grpo_go_prediction parent_experiment=<sft_experiment>
python scripts/train.py experiment=grpo_stability parent_experiment=<sft_experiment>
```

### Downstream Task Data

```bash
python scripts/data/download_cafa.py --max_samples 10000          # GO prediction
python scripts/data/download_megascale.py --max_samples 10000     # Stability/ddG
python scripts/data/download_structure_quality.py --max_samples 10000  # Structure quality
```

## Evaluation

```bash
# All benchmarks
python scripts/evaluate.py experiment_name=<experiment> evaluation.name=all

# Specific task
python scripts/evaluate.py experiment_name=<experiment> evaluation.name=go_prediction

# Vanilla LLM baseline
python scripts/evaluate.py eval_mode=vanilla model=qwen3_8b evaluation.name=all
```

Tasks: GO prediction (F1), PPI prediction (accuracy), Stability (MAE), SFT generation quality.

## Project Structure

```
├── configs/                  # Hydra configurations
│   ├── experiment/           #   Experiment presets (sft_esm3_mlp, etc.)
│   ├── model/                #   LLM configs (qwen3_4b/8b/14b, llama3_8b)
│   ├── training/             #   Training configs (sft_lora, grpo, dpo)
│   ├── data/                 #   Dataset configs
│   └── encoder/              #   ESM-3 encoder config
├── src/
│   ├── models/               # ProteinLLM, ESM-3 encoder, pooling, projector, perceiver
│   ├── training/             # SFT trainer, GRPO trainer
│   ├── data/                 # Mol-Instructions dataset, converters
│   └── evaluation/           # GO, PPI, stability benchmarks
├── scripts/                  # Entry points (train, evaluate, inference)
├── tests/                    # Unit tests (457 passing)
├── results/                  # Experiment outputs (gitignored)
└── docs/                     # Architecture, training guide, research log
```

## Hardware

8x NVIDIA H100 80GB | CUDA 12.4 | Python 3.11 | PyTorch 2.6.0

## Key Design Decisions

- ESM-3 encoder **always frozen** (float32 weights, bf16 autocast forward)
- LoRA r=8 on **all linear layers** (q/k/v/o + gate/up/down)
- Chat template format with protein-expert system prompt
- Token-budget dynamic batching (`max_tokens_per_batch=6144`)
- Differential LR: `projector_lr=5e-4` (5x base `lr=1e-4`) for random-init projector
- LoRA freeze for first 500 steps (projector alignment phase)
- wandb-only logging (tensorboard disabled)

## Documentation

- [Architecture](docs/architecture.md) — Full system design
- [Training Guide](docs/training_guide.md) — Detailed training instructions
- [Research Log](docs/research/agents_research_log.md) — Experiment results and decisions
- [CLAUDE.md](CLAUDE.md) — Agent instructions and quick reference
