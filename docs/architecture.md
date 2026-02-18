# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Protein-LLM Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Protein Sequence                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │   ESM-2 650M    │  ◄── FROZEN (never modify weights)         │
│  │   (Encoder)     │                                            │
│  └────────┬────────┘                                            │
│           │ Per-residue embeddings [L, 1280]                    │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │ Attention Pool  │  ◄── BoM-Pooling (window=80)               │
│  └────────┬────────┘                                            │
│           │ Fixed-size embedding [1, 1280]                      │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  MLP Projector  │  ◄── Trainable                             │
│  │  [1280 → 4096]  │                                            │
│  └────────┬────────┘                                            │
│           │ LLM-compatible embedding [1, 4096]                  │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │ LLM (Qwen/Llama)│  ◄── LoRA on k/v matrices only             │
│  │   + LoRA (r=8)  │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│      Response                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Protein Encoder (ESM-2)

| Model | Parameters | Embedding Dim | VRAM |
|-------|------------|---------------|------|
| esm2_t33_650M_UR50D | 650M | 1,280 | ~3GB |
| esm2_t36_3B_UR50D | 3B | 2,560 | ~12GB |

**Critical**: Always keep frozen. Do not update weights during training.

### 2. Attention Pooling

Uses BoM-Pooling (Bag of Motifs) with learned attention:
- Window size: 80 residues
- Learns which residues are most important
- Better than mean pooling which treats all residues equally

### 3. MLP Projector

```python
Projector:
  Linear(1280, 2048) → GELU → Dropout(0.1)
  Linear(2048, 4096) → GELU → Dropout(0.1)
```

### 4. LLM with LoRA

| LLM Option | Parameters | Hidden Size |
|------------|------------|-------------|
| Qwen-2.5-7B | 7B | 4,096 |
| Llama-3.1-8B | 8B | 4,096 |

LoRA Configuration:
- Rank: r=8 (minimum r=4)
- Alpha: 16
- Target: k_proj, v_proj ONLY
- Dropout: 0.05

## Training Stages

### Stage 1: SFT with QLoRA
- 4-bit quantization
- Train: Projector + LLM (LoRA)
- Freeze: ESM-2
- LR: 2e-4
- Epochs: 1-3

### Stage 2: GRPO Alignment
- Load SFT checkpoint
- LR: 5e-6 (much lower!)
- Group size: 4 completions per prompt
- Epochs: 1

## Memory Budget (8x H100 80GB)

| Component | VRAM per GPU |
|-----------|-------------|
| ESM-2 650M | ~3GB |
| LLM 7B (4-bit) | ~6GB |
| LoRA adapters | ~0.5GB |
| Activations | ~30GB |
| Gradients | ~20GB |
| **Total** | ~60GB |
