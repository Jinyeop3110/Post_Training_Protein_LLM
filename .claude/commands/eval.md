---
description: Run evaluation on benchmarks
---

Run evaluation:
```bash
python scripts/evaluate.py $ARGUMENTS
```

## Examples
```bash
# GO term prediction
python scripts/evaluate.py evaluation=go_prediction

# Protein-protein interaction
python scripts/evaluate.py evaluation=ppi

# Stability prediction
python scripts/evaluate.py evaluation=stability

# All benchmarks
python scripts/evaluate.py evaluation=all

# Specific checkpoint
python scripts/evaluate.py checkpoint_path=/path/to/checkpoint

# Custom output directory
python scripts/evaluate.py output_dir=./results/my_eval
```

## Available Benchmarks
- `go_prediction` - Gene Ontology term prediction
- `ppi` - Protein-protein interaction prediction
- `stability` - Protein stability prediction
- `all` - Run all benchmarks
