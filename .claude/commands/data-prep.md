---
description: Prepare and process datasets
---

Prepare data:
```bash
python scripts/prepare_data.py $ARGUMENTS
```

## Examples
```bash
# Process Mol-Instructions
python scripts/prepare_data.py data=mol_instructions

# Process Swiss-Prot
python scripts/prepare_data.py data=swissprot

# Process IPD PDB structures
python scripts/prepare_data.py data=ipd_pdb

# All datasets
python scripts/prepare_data.py data=all

# Custom output directory
python scripts/prepare_data.py output_dir=./data/processed/custom
```

## Data Locations
- Raw: `data/raw/`
- Processed: `data/processed/`
- Checkpoints: `data/checkpoints/`
