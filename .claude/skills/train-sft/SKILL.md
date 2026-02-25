---
name: train-sft
description: Run supervised fine-tuning with veRL
---
Run supervised fine-tuning for protein-LLM:

1. Verify environment:
   ```bash
   source /home/yeopjin/orcd/pool/init_protein_llm.sh
   python -c "import torch, verl, flash_attn; print(f'GPUs: {torch.cuda.device_count()}')"
   ```

2. Check configuration:
   - Read `configs/sft_config.yaml` (if exists)
   - Verify LoRA settings: r=8, applied to k/v matrices
   - Confirm ESM-3 is frozen

3. Pre-flight checks:
   - GPU memory: `nvidia-smi`
   - Data exists: `ls ./data/pdb_2021aug02_sample/`
   - Wandb configured: `echo $WANDB_DIR`

4. Run training:
   ```bash
   python scripts/train_sft.py --config configs/sft_config.yaml
   ```

5. Monitor:
   - Check wandb dashboard for loss curves
   - Watch for OOM errors
   - Verify gradient norms are stable

6. Post-training validation:
   - Run evaluation on validation set
   - Check checkpoint saved: `ls outputs/checkpoints/`
