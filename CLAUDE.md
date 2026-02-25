# Post-Training Protein LLM

## Quick Reference
```bash
# Environment
source /home/yeopjin/orcd/pool/init_protein_llm.sh

# Data pipeline: download → prepare → train
python src/data/download.py --dataset list              # List datasets
python src/data/download.py --dataset ipd_pdb_sample    # Download raw
python scripts/prepare_data.py data=mol_instructions    # Preprocess

# Train (approach: text | esm3)
python scripts/train.py experiment_name=my_sft_run      # Custom experiment name
python scripts/train.py approach=esm3 model=qwen3_4b    # ESM-3 + Qwen3-4B
python scripts/train.py approach=text                    # Text-only baseline

# GRPO (chain from SFT)
python scripts/train.py training=grpo experiment_name=my_grpo \
  parent_experiment=my_sft_run

# Downstream task data (10K samples each)
python scripts/data/download_cafa.py --max_samples 10000          # GO prediction
python scripts/data/download_megascale.py --max_samples 10000     # Stability/ddG
python scripts/data/download_structure_quality.py --max_samples 10000  # Structure quality

# GRPO with downstream tasks
python scripts/train.py experiment=grpo_go_prediction parent_experiment=my_sft
python scripts/train.py experiment=grpo_stability parent_experiment=my_sft
python scripts/train.py experiment=grpo_structure parent_experiment=my_sft

# Evaluate (auto-detects checkpoint from experiment)
python scripts/evaluate.py experiment_name=my_sft_run evaluation.name=all
python scripts/evaluate.py checkpoint_path=results/.../checkpoints/protein_llm

# Tests
pytest tests/ -v
```

## Architecture
Approach-based: `approach: text | esm3` (set in `configs/config.yaml`)
- **text**: Raw sequence as `<protein>MKTL...</protein>` tokens to LLM
- **esm3** (default): ESM-3 small (frozen, 1536-dim) -> Pooling/Projector -> LLM

Projector types: `encoder.projector.type=mlp` (default) or `perceiver`
- **MLP**: AttentionPooling (32 tokens) + 2-layer MLP (~20M params)
- **Perceiver**: PerceiverResampler replaces both pooling+projector (~130M params, 2 layers)

Default: ESM-3 small + Qwen3-4B-Instruct-2507 (projector: 1536->2048->2560)
LLM options: Qwen3-4B-Instruct-2507 (default), Qwen3-8B-Instruct-2507, Qwen3-14B-Instruct-2507, Llama-3.1-8B-Instruct
LoRA r=8 on all linear layers (q/k/v/o + gate/up/down)
Training uses model's native chat template with protein-expert system prompt

## Experiment Pipeline
All artifacts are stored under `results/{experiment_name}/`:
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

Pipeline lineage: `parent_experiment` in config chains SFT → GRPO experiments.
`lineage.json` tracks stage, parent, encoder, approach, and timestamps.

## Critical Rules
- NEVER modify ESM-3 encoder weights (always frozen)
- ESM-3 weights stay float32 but runs under `torch.amp.autocast("cuda", dtype=bfloat16)` for inference (halves activation memory). Sub-batched independently via `encoder_batch_size` (default 4).
- LoRA on all linear layers (q/k/v/o + gate/up/down projections), r=8
- Use attention pooling, NOT mean (for MLP path)
- TRITON_CACHE_DIR must be local (`/tmp/triton_cache_$USER`)
- Always use Instruct model variants (e.g., Qwen3-4B-Instruct-2507)
- Use chat template format with system prompt for training (not Alpaca ### format)
- Log to wandb only (tensorboard disabled)

## Config Overrides
```bash
python scripts/train.py model=llama3_8b training.lr=1e-4
python scripts/train.py encoder.projector.type=perceiver  # Perceiver Resampler
python scripts/train.py experiment_name=my_run            # Custom experiment name
python scripts/train.py --multirun training.lr=1e-4,2e-4
python scripts/train.py --cfg job  # Print config
```

## Key Paths
| Path | Purpose |
|------|---------|
| results/ | **Experiment outputs** (one dir per experiment) |
| configs/ | Hydra configurations |
| scripts/ | Entry points (train, eval, inference) |
| src/ | Core implementation |
| src/utils/experiment.py | Lineage tracking utilities |
| docs/ | Detailed documentation |
| .claude/ | Skills, commands, agents |

## Documentation
- [docs/research/agents_research_log.md](docs/research/agents_research_log.md) - **Project Log** (progress, results, decisions)
- [docs/architecture.md](docs/architecture.md) - Full architecture
- [docs/training_guide.md](docs/training_guide.md) - Training details
- [SWE_AGENT_TEAM.md](SWE_AGENT_TEAM.md) - **Agent Team Guide** (multi-agent development)
- [REVIEW_POINTS.md](REVIEW_POINTS.md) - **User Requirements** (check before major tasks)
- [PROJECT_GOALS.md](PROJECT_GOALS.md) - **Goals & Backlog** (strategic direction)

## Hardware
8x H100 80GB | CUDA 12.4 | Python 3.11
