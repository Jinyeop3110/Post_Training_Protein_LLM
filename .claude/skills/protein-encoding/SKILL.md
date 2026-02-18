---
name: protein-encoding
description: ESM-2 protein embeddings, encoder integration, pooling strategies
allowed-tools: [Read, Edit, Grep, Glob, Bash]
---

# Protein Encoding Skill

## Critical Rules
1. **NEVER modify ESM-2 weights** - always keep frozen during training
2. **Use attention pooling** (BoM-Pooling, window=80), NOT mean pooling
3. **LoRA on k/v matrices only** for protein tasks (differs from NLP)

## ESM-2 Models
| Model | Parameters | Embedding Dim | Recommended |
|-------|------------|---------------|-------------|
| esm2_t33_650M_UR50D | 650M | 1,280 | Best efficiency |
| esm2_t36_3B_UR50D | 3B | 2,560 | More capacity |

## Key Files
- src/models/protein_encoder.py - Encoder implementations
- src/models/pooling.py - Pooling strategies
- configs/encoder/ - Encoder configurations

## Integration Pattern
```
ESM-2 (frozen) → Per-residue [L, 1280]
    ↓
Attention Pooling → [1, 1280]
    ↓
MLP Projector → [1, LLM_dim]
    ↓
LLM (with LoRA)
```

## Usage Examples
```python
from src.models.protein_encoder import ESM2Encoder

encoder = ESM2Encoder(
    model_name="esm2_t33_650M_UR50D",
    pooling="attention",
    freeze=True  # ALWAYS True
)

# Get embeddings
embeddings = encoder(sequences)  # [batch, hidden_dim]
```

## Pooling Comparison
| Method | Performance | Memory | Use When |
|--------|-------------|--------|----------|
| Attention | Best | Higher | Default choice |
| Mean | Good | Lower | Memory constrained |
| CLS | Okay | Lowest | Quick experiments |
