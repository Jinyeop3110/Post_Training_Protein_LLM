---
name: protein-encoding
description: ESM-3 protein embeddings, encoder integration, pooling strategies
allowed-tools: [Read, Edit, Grep, Glob, Bash]
---

# Protein Encoding Skill

## Critical Rules
1. **NEVER modify ESM-3 encoder weights** - always keep frozen during training
2. **Use attention pooling** (32 output tokens), NOT mean pooling
3. **LoRA on k/v matrices only** for protein tasks (differs from NLP)

## ESM-3 Model
| Model | Parameters | Embedding Dim | Recommended |
|-------|------------|---------------|-------------|
| esm3-sm-open-v1 | 1.4B | 1,536 | Default |

## Key Files
- src/models/protein_encoder.py - Encoder implementations
- src/models/pooling.py - Pooling strategies
- configs/encoder/ - Encoder configurations

## Integration Pattern
```
ESM-3 (frozen) → Per-residue [L, 1536]
    ↓
Attention Pooling → [32, 1536]
    ↓
MLP Projector → [32, LLM_dim]
    ↓
LLM (with LoRA)
```

## Usage Examples
```python
from src.models.protein_encoder import ESM3Encoder

encoder = ESM3Encoder(
    model_name="esm3-sm-open-v1",
    freeze=True  # ALWAYS True
)

# Get embeddings
embeddings = encoder(sequences)  # [batch, seq_len, 1536]
```

## Pooling Comparison
| Method | Performance | Memory | Use When |
|--------|-------------|--------|----------|
| Attention | Best | Higher | Default choice |
| Mean | Good | Lower | Memory constrained |
| CLS | Okay | Lowest | Quick experiments |
