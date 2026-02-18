# Post-Training Protein LLM

## Quick Reference
```bash
# Environment
source /home/yeopjin/orcd/pool/init_protein_llm.sh

# Data pipeline: download → prepare → train
python src/data/download.py --dataset list              # List datasets
python src/data/download.py --dataset ipd_pdb_sample    # Download raw
python scripts/prepare_data.py data=mol_instructions    # Preprocess

# Train
python scripts/train.py experiment=baseline_sft

# Evaluate
python scripts/evaluate.py evaluation=go_prediction

# Tests
pytest tests/ -v
```

## Architecture
ESM-2 650M (frozen) → Attention Pooling → MLP Projector → LLM (LoRA k/v only)

## Critical Rules
- NEVER modify ESM-2 weights
- LoRA on k/v matrices ONLY
- Use attention pooling, NOT mean
- TRITON_CACHE_DIR must be local (`/tmp/triton_cache_$USER`)

## Config Overrides
```bash
python scripts/train.py model=llama3_8b training.lr=1e-4
python scripts/train.py --multirun training.lr=1e-4,2e-4
python scripts/train.py --cfg job  # Print config
```

## Key Paths
| Path | Purpose |
|------|---------|
| configs/ | Hydra configurations |
| scripts/ | Entry points (train, eval, inference) |
| src/ | Core implementation |
| docs/ | Detailed documentation |
| .claude/ | Skills, commands, agents |

## Documentation
- [docs/research/agents_research_log.md](docs/research/agents_research_log.md) - **Project Log** (progress, results, decisions)
- [docs/architecture.md](docs/architecture.md) - Full architecture
- [docs/training_guide.md](docs/training_guide.md) - Training details

## Hardware
8x H100 80GB | CUDA 12.4 | Python 3.11
