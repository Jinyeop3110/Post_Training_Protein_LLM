---
name: data-prep
description: Prepare protein datasets for training
---
Prepare datasets for protein-LLM training:

1. Verify environment:
   ```bash
   source /home/yeopjin/orcd/pool/init_protein_llm.sh
   ```

2. List available datasets:
   ```bash
   python src/data/download.py --dataset list
   ```

3. Download datasets (choose as needed):
   ```bash
   # IPD PDB sample (recommended for training)
   python src/data/download.py --dataset ipd_pdb_sample --output_dir ./data

   # Swiss-Prot sequences
   python src/data/download.py --dataset swissprot --output_dir ./data

   # Mol-Instructions (HuggingFace)
   python -c "from datasets import load_dataset; load_dataset('zjunlp/Mol-Instructions', 'Protein')"
   ```

4. Verify downloads:
   ```bash
   ls -lh ./data/
   ```

5. Test dataloaders:
   ```python
   from src.data import get_pdb_dataloader
   dl = get_pdb_dataloader("./data/pdb_2021aug02_sample", batch_size=4)
   batch = next(iter(dl))
   print(f"Sequences: {len(batch['sequence'])}")
   print(f"Coords shape: {batch['coords'].shape}")
   ```

6. Preprocess for training (if needed):
   ```bash
   python scripts/preprocess_data.py \
       --input_dir ./data/pdb_2021aug02_sample \
       --output_dir ./data/processed \
       --max_length 512 \
       --max_resolution 3.0
   ```
