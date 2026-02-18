# Multi-Agent Research & Development Log

## Project: Post-Training Protein LLM

**Goal**: Train/post-train existing LLMs to understand proteins as multimodal embeddings using SFT (mid-training) and RL (problem-specific training).

**Repository**: `/home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM`

---

## Quick Links

| Section | Description |
|---------|-------------|
| [Development Log](#development-log) | Implementation progress & milestones |
| [Results & Metrics](#results--metrics) | Training results & benchmarks |
| [Decisions](#decision-log) | Key decisions & rationale |
| [Research Findings](#agent-interactions-log) | Literature review |
| [Action Items](#action-items) | TODO list |

---

---

## Development Log

### 2026-02-16: Model Loading Test Script

**Milestone**: Created model loading test script for Qwen 3 1.5B verification

**Created**:
- `scripts/test_model_loading.py` - Tests model loading, tokenizer, and inference

**Script Features**:
- Environment check (PyTorch, CUDA, Transformers versions)
- GPU detection with helpful error messages
- Model loading with configurable dtype
- Tokenizer verification
- Inference sanity check
- Memory usage reporting

**Usage** (requires GPU compute node):
```bash
# On compute node with GPU
srun --gres=gpu:1 --mem=32G --time=00:30:00 \
    bash -c "source /home/yeopjin/orcd/pool/init_protein_llm.sh && \
    python scripts/test_model_loading.py --model Qwen/Qwen3-1.5B"

# List available models
python scripts/test_model_loading.py --list-models

# Check environment only (no GPU needed)
python scripts/test_model_loading.py --check-only
```

**Models to Test**:
| Model | Size | Use Case |
|-------|------|----------|
| Qwen/Qwen3-1.5B | 1.5B | Quick testing |
| Qwen/Qwen2.5-7B-Instruct | 7B | Production training |
| meta-llama/Llama-3.1-8B-Instruct | 8B | Alternative |

**Note**: Script requires GPU access. Login nodes don't have GPUs - use `srun` or submit a job.

**Next Steps**:
- [ ] Run test on compute node with GPU
- [ ] Verify Qwen 3 1.5B loads correctly
- [ ] Test with ESM-2 encoder integration

---

### 2026-02-16: Dataset Download & Exploration

**Milestone**: Downloaded all training datasets and created exploration notebook

**Datasets Downloaded**:

| Dataset | Location | Size | Status |
|---------|----------|------|--------|
| IPD PDB Sample | `data/raw/pdb_2021aug02_sample/` | 870 `.pt` files | ✅ Ready |
| Swiss-Prot | `data/raw/swissprot/uniprot_sprot.fasta.gz` | ~90MB (~570K sequences) | ✅ Ready |
| Mol-Instructions | `data/raw/mol_instructions/data/Protein-oriented_Instructions/` | 5 JSON files (~678MB total) | ✅ Ready |

**Mol-Instructions Files**:
- `protein_design.json` (281 MB)
- `protein_function.json` (155 MB)
- `general_function.json` (127 MB)
- `catalytic_activity.json` (65 MB)
- `domain_motif.json` (50 MB)

**IPD PDB Sample Structure** (per `.pt` file):
- `seq`: Amino acid sequence (string)
- `xyz`: Atomic coordinates `[L, 14, 3]` tensor
- `mask`: Boolean mask `[L, 14]`
- `bfac`: Temperature factors `[L, 14]`
- `occ`: Occupancy `[L, 14]`

**Created**:
- `scripts/explore_datasets.ipynb` - Jupyter notebook for dataset exploration
- Download utilities in `src/data/download.py`

**Usage**:
```bash
# Download datasets
python src/data/download.py --dataset ipd_pdb_sample --output_dir ./data/raw
python src/data/download.py --dataset swissprot --output_dir ./data/raw/swissprot
python src/data/download.py --dataset mol_instructions --output_dir ./data/raw

# Explore datasets
jupyter notebook scripts/explore_datasets.ipynb
```

**Next Steps**:
- [ ] Preprocess datasets into training format
- [ ] Create train/val/test splits
- [ ] Build instruction dataset loader

---

### 2026-02-16: Project Restructuring

**Milestone**: Complete project restructuring for Claude Code integration and Hydra configs

**Changes Made**:
- Created `.claude/` directory with skills, commands, and agents
- Implemented Hydra configuration system in `configs/`
- Created entry point scripts in `scripts/`
- Moved documentation to `docs/`
- Slimmed CLAUDE.md from 80 to ~50 lines
- Set up pytest test structure in `tests/`
- Created `pyproject.toml` for package management

**New Structure**:
```
Post_Training_Protein_LLM/
├── .claude/           # Claude Code integration
│   ├── agents/        # experiment-runner, research, code-reviewer
│   ├── commands/      # /train, /eval, /data-prep, /debug
│   └── skills/        # protein-encoding, rl-training, hydra-configs
├── configs/           # Hydra hierarchical configs
│   ├── model/         # qwen2_7b, llama3_8b
│   ├── encoder/       # esm2_650m, esm2_3b
│   ├── training/      # sft_qlora, grpo, dpo
│   └── experiment/    # baseline_sft, full_pipeline
├── scripts/           # Entry points
├── src/               # Core implementation
├── tests/             # pytest suite
└── docs/              # Documentation
```

**Next Steps**:
- [ ] Implement SFT trainer in `src/training/sft_trainer.py`
- [ ] Implement GRPO trainer in `src/training/grpo_trainer.py`
- [ ] Add attention pooling in `src/models/pooling.py`
- [ ] Create MLP projector in `src/models/projector.py`

---

### Template: New Entry
```markdown
### YYYY-MM-DD: Title

**Milestone**: What was achieved

**Changes Made**:
- Item 1
- Item 2

**Results** (if applicable):
| Metric | Value |
|--------|-------|
| ... | ... |

**Issues Encountered**:
- Issue and resolution

**Next Steps**:
- [ ] Task 1
- [ ] Task 2
```

---

## Results & Metrics

### Experiments Summary

| Date | Experiment | Model | Dataset | Key Metric | Value | Notes |
|------|------------|-------|---------|------------|-------|-------|
| - | - | - | - | - | - | No experiments run yet |

### Best Configurations

*To be updated after experiments*

---

## Decision Log

### 2026-02-16: Project Structure Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Config system | Hydra | CLI overrides, sweeps, hierarchical composition |
| Claude integration | .claude/ directory | Skills, commands, agents for automation |
| Package manager | pyproject.toml | Modern Python standard |
| Test framework | pytest | Industry standard, good fixtures |

### 2026-02-14: Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Protein encoder | ESM-2 650M | Best efficiency/performance ratio |
| Keep encoder frozen | Yes | Preserve protein knowledge |
| LoRA targets | k/v matrices only | Protein-specific finding |
| Pooling method | Attention pooling | 4% better than mean pooling |
| RL method | GRPO | 50% less memory than PPO |

---

## Agent Rules & Protocols

### 1. Research Agent Protocols
- Each agent focuses on a specific topic area
- Findings must include: paper title, authors, year, key contributions, relevance to our project
- Agents should identify: datasets, methods, evaluation metrics, open questions
- All interactions and findings are logged in this document

### 2. Topic Assignments
| Agent ID | Topic Focus | Status |
|----------|-------------|--------|
| Agent-1 | Protein Language Models & Embeddings | Completed |
| Agent-2 | LLM Post-Training Methods (SFT/RL) | Completed |
| Agent-3 | Multimodal LLM Integration | Completed |
| Agent-4 | Protein-Related Datasets & Benchmarks | Completed |

### 3. Documentation Standards
- Each finding should be timestamped
- Cross-references between agents encouraged
- Questions for human input marked with `[QUESTION]`
- Action items marked with `[ACTION]`

### 4. Decision Log
All major decisions and their rationale are recorded here.

---

## Agent Interactions Log

### Session: Initial Research Phase
**Date**: 2026-02-14
**Objective**: Conduct literature review for protein-LLM post-training

---

## Agent-1: Protein Language Models & Embeddings

**Focus Areas**:
- Existing protein language models (ESM, ProtTrans, etc.)
- Protein embedding representations
- How to integrate protein embeddings into LLMs

### Findings Summary:

#### Key Protein Language Models

| Model | Parameters | Embedding Dim | Training Data |
|-------|------------|---------------|---------------|
| ESM-2 650M | 650M | 1,280 | UniProt 2021_04 |
| ESM-2 3B | 3B | 2,560 | UniProt 2021_04 |
| ESM-3 | 98B | Multimodal | 2.78B proteins |
| ProtT5-XL | 3B | 1,024 | UniRef50 |
| Ankh | 1.15B | ~1,536 | Efficient training |

#### Key Insights:
1. **ESM-2 650M recommended** for best performance/efficiency trade-off
2. Performance levels off around 650M parameters - larger models don't significantly outperform
3. **ESM-3** is the first multimodal generative model with sequence, structure, AND function tokens
4. **ProtT5** encoder-only version fits on 8GB GPU in half-precision

#### Integration Patterns:
- **Pattern A**: Linear Projection (ProteinGPT) - simple alignment
- **Pattern B**: Cross-Attention Adapter (Prot2Chat) - early fusion
- **Pattern C**: Knowledge Graph Instruction (InstructProtein) - bidirectional generation

**Full report**: See detailed research in agent output logs

---

## Agent-2: LLM Post-Training Methods (SFT/RL)

**Focus Areas**:
- Supervised Fine-Tuning (SFT) techniques
- Reinforcement Learning methods (RLHF, DPO, PPO)
- Training frameworks (TRL, veRL, OpenRLHF)

### Findings Summary:

**Full report saved to**: `LLM_Post_Training_Methods_Summary.md`

#### Framework Comparison

| Framework | Key Strength | Best For |
|-----------|--------------|----------|
| TRL | HuggingFace integration | Beginners, prototyping |
| veRL | Scalability (671B models) | Large-scale research |
| OpenRLHF | Easy + high-performance | Production, multimodal |

#### Recommended Training Strategy:

**Phase 1 - SFT with QLoRA:**
- 4-bit quantization: 70B model → ~46GB VRAM
- LoRA rank: r=8 (minimum r=4 for proteins)
- Learning rate: 2e-4
- Epochs: 1-3

**Phase 2 - Preference Alignment:**
- **DPO**: Simple, no reward model needed
- **GRPO**: 50% less memory than PPO, better for reasoning
- **Avoid PPO** unless significant compute available

#### Protein-Specific Insights:
- Apply LoRA to key/value matrices only (differs from NLP!)
- Fine-tuned ESM2-150M can outperform ESM2-650M/3B without fine-tuning
- LoRA reduces ProtT5 (1.2B) trainable params to ~3.5M

---

## Agent-3: Multimodal LLM Integration

**Focus Areas**:
- How to incorporate non-text modalities into LLMs
- Projection layers, adapters, cross-attention mechanisms
- Examples from vision-language models

### Findings Summary:

#### Projection Methods Comparison

| Method | Pros | Cons |
|--------|------|------|
| MLP Projector | Simple, preserves spatial info | High token count |
| Q-Former | Fixed output tokens | Information loss risk |
| Perceiver Resampler | Variable input, fixed output | Struggles with spatial |
| Linear | Simplest | Limited expressiveness |

#### Key VLM Architectures for Reference:
- **LLaVA**: CLIP + 2-layer MLP + Vicuna (two-stage training)
- **InternVL 2.5**: Large ViT-6B + pixel unshuffle → 256 tokens
- **Qwen2-VL**: Dynamic resolution with 2D-RoPE

#### Recommended Architecture for Proteins:

```
ESM-2 650M (frozen) → Per-residue [L, 1280]
       ↓
Mean Pooling OR Attention Pooling
       ↓
MLP Projector (1280 → LLM_dim)
       ↓
[protein_tokens] + [text_tokens] → LLM (with LoRA)
```

#### Training Recipe:
| Stage | Frozen | Trainable | Data |
|-------|--------|-----------|------|
| 1. Alignment | ESM-2 + LLM | Projector only | Protein-text pairs |
| 2. Instruction | ESM-2 | Projector + LLM (LoRA) | QA + text-only |

#### Variable Length Handling:
- **Direct mapping**: Each residue → one token (preserves info)
- **Attention pooling**: BoM-Pooling with window 80 outperforms mean pooling by 4%
- **Avoid mean pooling**: Treats all residues equally (biologically incorrect)

---

## Agent-4: Protein Datasets & Benchmarks

**Focus Areas**:
- Existing protein-text datasets
- Evaluation benchmarks for protein understanding
- Task definitions (function prediction, interaction, etc.)

### Findings Summary:

**Full report saved to**: `research/protein_datasets_and_benchmarks.md`

#### Training Datasets

| Dataset | Size | Format |
|---------|------|--------|
| Mol-Instructions (protein) | 505K | Instruction format |
| ProteinChat | 1.5M+ triplets | (protein, prompt, answer) |
| Swiss-Prot | 570K+ proteins | Curated annotations |
| UniRef50 | 30M sequences | Pre-training |

#### Evaluation Benchmarks

| Task | Benchmark | Metrics |
|------|-----------|---------|
| Function Prediction | CAFA5, PROBE | Fmax, AUPR |
| PPI Prediction | BioSNAP, STRING | AUC, MCC |
| Stability | S669, Megascale | Pearson, Spearman |
| Subcellular Localization | PEER | Accuracy, F1 |
| Drug-Target Interaction | Davis, KIBA | MSE, CI |

#### Recommended Evaluation Plan:
1. **Core tasks**: GO term prediction, PPI, stability, localization
2. **Baseline comparisons**: ESM2-650M/3B, ProtT5-XL, ProstT5
3. **Phases**: Zero-shot → task-specific → cross-domain transfer

---

## Open Questions for Discussion

1. [QUESTION] Which base LLM should we use? (LLaMA-3, Mistral, Qwen, etc.)
   - **Agent-2 suggests**: Qwen-2.5 7B or Llama-3 8B as starting points

2. [QUESTION] What specific downstream tasks are highest priority?
   - **Agent-4 suggests**: GO term prediction, PPI, stability (fundamental biology)

3. [QUESTION] GPU resources available for training?
   - **RESOLVED**: 8x NVIDIA H100 80GB available (CUDA 13.0)

4. [QUESTION] Should we focus on single proteins or protein-protein interactions?
   - **Recommendation**: Start with single protein understanding, add PPI later

5. [QUESTION] Third protein incorporation method (TBD)?
   - **Options identified**:
     - Structure-aware (ESM-IF1, ProteinMPNN)
     - Graph Neural Networks (GearNet, GVP)
     - Hybrid text + embedding approach

---

## Action Items

### Completed
- [x] Complete literature review with 4 agents
- [x] Set up conda environment (`setup_env.sh`)
- [x] Create base project structure (`src/`)
- [x] Project restructuring for Claude Code & Hydra (2026-02-16)
- [x] Create Hydra configuration system
- [x] Set up `.claude/` with skills, commands, agents
- [x] Create entry point scripts (`scripts/`)
- [x] Set up test structure (`tests/`)

### In Progress
- [ ] Select training framework (TRL vs veRL vs other)
  - **Current thinking**: TRL for SFT, veRL for GRPO
- [ ] Select base LLM (Qwen-2.5 7B vs Llama-3 8B)

### TODO: Model Testing
- [ ] Run `scripts/test_model_loading.py` on compute node with GPU
- [ ] Verify Qwen/Qwen3-1.5B loads and runs inference
- [ ] Test ESM-2 encoder loading
- [ ] Test combined encoder + LLM memory usage

### TODO: Implementation
- [ ] Implement attention pooling (`src/models/pooling.py`)
- [ ] Implement MLP projector (`src/models/projector.py`)
- [ ] Implement multimodal LLM wrapper (`src/models/multimodal_llm.py`)
- [ ] Implement SFT trainer (`src/training/sft_trainer.py`)
- [ ] Implement GRPO trainer (`src/training/grpo_trainer.py`)
- [ ] Create instruction dataset loader (`src/data/instruction_dataset.py`)
- [ ] Implement evaluation benchmarks (`src/evaluation/`)

### TODO: Data
- [x] Download Mol-Instructions dataset
- [x] Download IPD PDB Sample dataset
- [x] Download Swiss-Prot sequences
- [x] Create dataset exploration notebook (`scripts/explore_datasets.ipynb`)
- [ ] Preprocess protein-text pairs
- [ ] Create train/val/test splits

### TODO: Training
- [ ] Run baseline SFT experiment
- [ ] Run GRPO alignment
- [ ] Evaluate on GO prediction, PPI, stability

---

## Project Structure (Updated 2026-02-16)

```
Post_Training_Protein_LLM/
├── CLAUDE.md                     # Concise project overview (~50 lines)
├── pyproject.toml                # Package configuration
├── setup_env.sh                  # Conda environment setup
│
├── .claude/                      # Claude Code integration
│   ├── settings.json             # Permissions, hooks
│   ├── agents/                   # experiment-runner, research, code-reviewer
│   ├── commands/                 # /train, /eval, /data-prep, /debug
│   └── skills/                   # protein-encoding, rl-training, hydra-configs
│
├── configs/                      # Hydra configuration
│   ├── config.yaml               # Main config
│   ├── model/                    # qwen2_7b, llama3_8b
│   ├── encoder/                  # esm2_650m, esm2_3b
│   ├── data/                     # mol_instructions, ipd_pdb
│   ├── training/                 # sft_qlora, grpo, dpo
│   ├── evaluation/               # go_prediction, ppi, stability
│   └── experiment/               # baseline_sft, full_pipeline
│
├── scripts/                      # Entry points
│   ├── train.py                  # python scripts/train.py experiment=...
│   ├── evaluate.py
│   ├── prepare_data.py
│   └── inference.py
│
├── src/                          # Core implementation
│   ├── models/                   # protein_encoder.py, pooling.py, projector.py
│   ├── data/                     # pdb_dataset.py, instruction_dataset.py
│   ├── training/                 # sft_trainer.py, grpo_trainer.py
│   ├── evaluation/               # go_prediction.py, ppi_prediction.py
│   └── utils/                    # logging.py, checkpoint.py, distributed.py
│
├── tests/                        # pytest test suite
│   ├── models/
│   ├── data/
│   └── training/
│
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── training_guide.md
│   ├── troubleshooting.md
│   └── research/                 # This file + research reports
│
├── data/                         # Data directory (gitignored)
│   ├── raw/
│   ├── processed/
│   └── checkpoints/
│
└── notebooks/                    # Jupyter notebooks
```

---

## Environment Info

- **Hardware**: 8x NVIDIA H100 80GB HBM3
- **CUDA Version**: 13.0
- **Driver**: 580.105.08
- **Conda**: miniforge/25.11.0-0 (via module)

---

## Next Steps

### Immediate (This Week)
1. **[ACTION]** Decide on base LLM: Qwen-2.5 7B vs Llama-3 8B
2. **[ACTION]** Implement attention pooling (`src/models/pooling.py`)
3. **[ACTION]** Implement MLP projector (`src/models/projector.py`)
4. ~~**[ACTION]** Download and preprocess Mol-Instructions dataset~~ ✅ Downloaded
5. **[ACTION]** Preprocess downloaded datasets into training format

### Short-term (Next 2 Weeks)
5. **[ACTION]** Implement SFT trainer with TRL
6. **[ACTION]** Run first training experiment
7. **[ACTION]** Evaluate on GO prediction benchmark

### Medium-term
8. **[ACTION]** Implement GRPO with veRL
9. **[ACTION]** Run full SFT → GRPO pipeline
10. **[ACTION]** Compare against baselines (ESM2-650M, ProtT5)

---

## How to Update This Log

1. **Add new entries** under [Development Log](#development-log) with date
2. **Record results** in [Results & Metrics](#results--metrics) table
3. **Document decisions** in [Decision Log](#decision-log)
4. **Update action items** in [Action Items](#action-items)

**Quick commands**:
```bash
# Open this file
code docs/research/agents_research_log.md

# Run training
python scripts/train.py experiment=baseline_sft

# Run evaluation
python scripts/evaluate.py evaluation=go_prediction
```
