# Datasets

## Overview

| Dataset | Type | Size | Use Case |
|---------|------|------|----------|
| Mol-Instructions | Instruction pairs | ~2M | SFT training |
| IPD PDB | Structures | Sample | Structure-aware training |
| Swiss-Prot | Sequences + annotations | ~500K | Pre-training / evaluation |

## Mol-Instructions

**Source**: [zjunlp/Mol-Instructions](https://huggingface.co/datasets/zjunlp/Mol-Instructions)

Contains protein-oriented instruction-response pairs for:
- Function prediction
- Property description
- Structure explanation

**Usage**:
```bash
python scripts/prepare_data.py data=mol_instructions
```

**Config**: `configs/data/mol_instructions.yaml`

## IPD PDB Sample

**Location**: `data/raw/pdb_2021aug02_sample/`

Pre-computed structure features from the IPD PDB database.

**Contents**:
- `.pt` files with structure embeddings
- Cluster splits (train/valid/test)

**Usage**:
```bash
python scripts/prepare_data.py data=ipd_pdb
```

**Config**: `configs/data/ipd_pdb.yaml`

## Swiss-Prot

**Source**: UniProt (reviewed sequences)

High-quality protein sequences with:
- GO term annotations
- Function descriptions
- Structure information

**Usage**:
```bash
python scripts/prepare_data.py data=swissprot
```

**Config**: `configs/data/swissprot.yaml`

## Data Preparation

### Download All Data
```bash
python scripts/prepare_data.py data=all
```

### Verify Data
```bash
ls -la data/raw/
ls -la data/processed/
```

## Data Format

### Instruction Format
```json
{
  "instruction": "Predict the function of this protein.",
  "input": "MKTAYIAKQRQISFVKSHFSRQ...",
  "output": "This protein is involved in..."
}
```

### Structure Format
```python
# .pt files contain:
{
  "sequence": str,
  "coords": torch.Tensor,  # [L, 3] or [L, 37, 3]
  "features": torch.Tensor,
}
```

## Adding New Datasets

1. Create config in `configs/data/your_dataset.yaml`
2. Implement loader in `src/data/`
3. Add to `scripts/prepare_data.py`
