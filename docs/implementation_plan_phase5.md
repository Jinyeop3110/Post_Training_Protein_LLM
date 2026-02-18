# Phase 5: Core Implementation Plan

**Date**: 2026-02-18
**Scope**: Minimal Viable Pipeline (MVP)
**Goal**: End-to-end flow: Load model → Train on Mol-Instructions → Evaluate GO prediction

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SFT Framework | **TRL** | HuggingFace standard, well-documented, native QLoRA support |
| Eval Priority | **GO, PPI, Stability** | Core protein understanding tasks |
| Protein-LLM Integration | **Prefix Tokens** | Project ESM-2 → LLM dim, prepend as soft tokens |
| Dataset | **Mol-Instructions** | 505K ready-to-use instructions |
| RL Phase | **Defer** | SFT first, GRPO later |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input                                     │
│  Protein Sequence: "MKTLLILAVVAAALA..."                         │
│  Instruction: "What is the function of this protein?"           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ESM-2 Encoder (Frozen)                        │
│  esm2_t33_650M_UR50D → Per-residue [L, 1280]                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Attention Pooling (BoM)                       │
│  Per-residue [L, 1280] → Pooled [N_tokens, 1280]                │
│  Window size: 80, learnable                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLP Projector                                 │
│  [N_tokens, 1280] → [N_tokens, 4096]                            │
│  2-layer MLP with GELU, trainable                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM (Qwen-2.5-7B + LoRA)                      │
│  Prefix: [protein_embed_1, ..., protein_embed_N]                │
│  Text: [instruction tokens...]                                   │
│  LoRA on k_proj, v_proj only                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Output                                    │
│  "This protein is involved in cellular respiration..."          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### Task 1: MLP Projector (`src/models/projector.py`)

**Purpose**: Map ESM-2 embeddings (1280-dim) to LLM hidden size (4096-dim)

```python
# Key components:
class MLPProjector(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,      # ESM-2 embedding dim
        hidden_dim: int = 2048,     # Intermediate dim
        output_dim: int = 4096,     # LLM hidden size
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        ...
```

**Deliverables**:
- [ ] `MLPProjector` class with configurable layers
- [ ] Forward pass: `[B, N, 1280] → [B, N, 4096]`
- [ ] Unit tests in `tests/models/test_projector.py`

---

### Task 2: Attention Pooling (`src/models/pooling.py`)

**Purpose**: Pool per-residue ESM-2 embeddings into fixed number of tokens

```python
# Key components:
class AttentionPooling(nn.Module):
    """BoM-Pooling (Bag of Motifs) with learned attention."""

    def __init__(
        self,
        embed_dim: int = 1280,
        num_output_tokens: int = 32,  # Number of prefix tokens
        window_size: int = 80,        # Local attention window
    ):
        ...
```

**Deliverables**:
- [ ] `AttentionPooling` class (BoM-style)
- [ ] `MeanPooling` class (simple fallback)
- [ ] Forward pass: `[B, L, 1280] → [B, N, 1280]`
- [ ] Unit tests in `tests/models/test_pooling.py`

---

### Task 3: Multimodal Model (`src/models/multimodal_llm.py`)

**Purpose**: Combine ESM-2 encoder + Projector + LLM

```python
# Key components:
class ProteinLLM(nn.Module):
    """Multimodal protein-language model."""

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2.5-7B-Instruct",
        encoder_name: str = "esm2_t33_650M_UR50D",
        pooling: str = "attention",
        num_prefix_tokens: int = 32,
        freeze_encoder: bool = True,  # ALWAYS True
        use_qlora: bool = True,
    ):
        self.encoder = ESMProteinEncoder(encoder_name)
        self.pooling = AttentionPooling(...)
        self.projector = MLPProjector(...)
        self.llm = load_llm_with_qlora(llm_name)
```

**Deliverables**:
- [ ] `ProteinLLM` class combining all components
- [ ] Method to generate prefix embeddings from protein
- [ ] Method to prepare inputs for LLM (prefix + text tokens)
- [ ] Integration tests

---

### Task 4: Mol-Instructions Dataset (`src/data/mol_instructions.py`)

**Purpose**: Load and format Mol-Instructions for training

```python
# Key components:
class MolInstructionsDataset(Dataset):
    """Dataset for Mol-Instructions protein subset."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        split: str = "train",
    ):
        # Load from HuggingFace: zjunlp/Mol-Instructions
        ...

    def __getitem__(self, idx):
        # Return: {
        #   "protein_sequence": str,
        #   "instruction": str,
        #   "response": str,
        #   "input_ids": tensor,
        #   "labels": tensor,
        # }
```

**Deliverables**:
- [ ] `MolInstructionsDataset` class
- [ ] Data collator for batching
- [ ] Proper train/val/test splits
- [ ] Unit tests in `tests/data/test_mol_instructions.py`

---

### Task 5: SFT Trainer (`src/training/sft_trainer.py`)

**Purpose**: Full SFT implementation using TRL

```python
# Key components:
def run_sft_qlora(cfg: DictConfig) -> None:
    """Run SFT with QLoRA using TRL."""

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)

    # 2. Load model with quantization
    model = ProteinLLM(
        llm_name=cfg.model.path,
        encoder_name=cfg.encoder.model_name,
        use_qlora=True,
    )

    # 3. Load dataset
    dataset = MolInstructionsDataset(tokenizer, split="train")

    # 4. Configure TRL SFTTrainer
    training_args = SFTConfig(
        output_dir=cfg.paths.checkpoint_dir,
        learning_rate=cfg.training.lr,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        ...
    )

    # 5. Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
```

**Deliverables**:
- [ ] Complete `run_sft_qlora()` function
- [ ] QLoRA configuration from Hydra config
- [ ] Checkpoint saving/loading
- [ ] Wandb/Tensorboard logging integration
- [ ] Integration test with small model

---

### Task 6: GO Prediction Evaluation (`src/evaluation/go_prediction.py`)

**Purpose**: Evaluate GO term prediction capability

```python
# Key components:
def evaluate_go(
    cfg: DictConfig,
    checkpoint_path: str,
) -> Dict[str, float]:
    """Evaluate GO term prediction."""

    # 1. Load model from checkpoint
    model = ProteinLLM.from_checkpoint(checkpoint_path)

    # 2. Load test dataset (GO terms)
    test_data = load_go_test_set()

    # 3. Generate predictions
    predictions = []
    for sample in test_data:
        prompt = f"Predict GO terms for: {sample['sequence']}"
        output = model.generate(prompt)
        predictions.append(parse_go_terms(output))

    # 4. Compute metrics
    return {
        "accuracy": compute_accuracy(predictions, test_data),
        "f1_macro": compute_f1(predictions, test_data),
        "aupr": compute_aupr(predictions, test_data),
    }
```

**Deliverables**:
- [ ] GO term test dataset loading
- [ ] Prediction generation
- [ ] Metrics: Accuracy, F1, AUPR
- [ ] Results logging

---

## File Structure After Implementation

```
src/
├── models/
│   ├── __init__.py
│   ├── protein_encoder.py     # Existing
│   ├── pooling.py             # NEW: Attention/Mean pooling
│   ├── projector.py           # NEW: MLP projector
│   └── multimodal_llm.py      # NEW: Combined model
├── data/
│   ├── __init__.py
│   ├── pdb_dataset.py         # Existing
│   ├── rcsb_dataset.py        # Existing
│   ├── download.py            # Existing
│   └── mol_instructions.py    # NEW: Mol-Instructions loader
├── training/
│   ├── __init__.py
│   ├── sft_trainer.py         # UPDATE: Full implementation
│   ├── grpo_trainer.py        # Keep as stub for now
│   └── dpo_trainer.py         # Keep as stub for now
└── evaluation/
    ├── __init__.py
    ├── go_prediction.py       # UPDATE: Full implementation
    ├── ppi_prediction.py      # Keep as stub for now
    ├── stability.py           # Keep as stub for now
    └── benchmarks.py          # Existing
```

---

## Implementation Order

```
Week 1: Foundation
├── Task 1: MLPProjector
├── Task 2: AttentionPooling
└── Tests for both

Week 2: Model Integration
├── Task 3: ProteinLLM (multimodal)
├── Task 4: MolInstructionsDataset
└── Integration tests

Week 3: Training
├── Task 5: SFT Trainer
├── End-to-end training test
└── Checkpoint management

Week 4: Evaluation
├── Task 6: GO Prediction
├── Full pipeline test
└── Documentation
```

---

## Dependencies to Install

```bash
# Core (already in pyproject.toml)
pip install torch transformers peft bitsandbytes fair-esm

# TRL for SFT
pip install trl>=0.8.0

# Evaluation metrics
pip install scikit-learn

# Optional: Flash Attention
pip install flash-attn --no-build-isolation
```

---

## Success Criteria

### MVP Complete When:

1. **Training works**:
   ```bash
   python scripts/train.py experiment=baseline_sft
   # Completes without error, saves checkpoint
   ```

2. **Evaluation works**:
   ```bash
   python scripts/evaluate.py evaluation=go_prediction
   # Returns metrics: accuracy, F1, AUPR
   ```

3. **Tests pass**:
   ```bash
   pytest tests/ -v
   # All tests green
   ```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Memory OOM on H100 | Start with smaller batch size, gradient checkpointing |
| ESM-2 + LLM too slow | Use vLLM for inference, cache ESM embeddings |
| Mol-Instructions format issues | Add robust parsing, fallback to subset |
| GO evaluation metric complexity | Start with simple accuracy, add AUPR later |

---

## Next Steps

1. **Approve this plan** or request changes
2. **Start with Task 1** (MLPProjector) - simplest, no dependencies
3. **Iterate** through tasks in order

---

*Plan created: 2026-02-18*
*Status: Awaiting approval*
