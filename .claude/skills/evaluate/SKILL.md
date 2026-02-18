---
name: evaluate
description: Evaluate protein-LLM on benchmarks
---
Evaluate the protein-LLM model:

1. Verify environment:
   ```bash
   source /home/yeopjin/orcd/pool/init_protein_llm.sh
   ```

2. Select evaluation tasks:
   - GO term prediction (Fmax, AUPR metrics)
   - Protein-protein interaction (AUC, MCC metrics)
   - Stability prediction (Pearson, Spearman)
   - Subcellular localization (Accuracy, F1)

3. Load checkpoint:
   - Find latest: `ls -lt outputs/checkpoints/`
   - Or specify: `--checkpoint_path <path>`

4. Run evaluation:
   ```bash
   python scripts/evaluate.py \
       --checkpoint_path outputs/checkpoints/latest \
       --tasks go_term,ppi,stability \
       --output_dir eval_results/
   ```

5. Compare baselines:
   - ESM2-650M (no fine-tuning)
   - ESM2-3B (no fine-tuning)
   - ProtT5-XL

6. Generate report:
   - Results in `eval_results/metrics.json`
   - Tables in `eval_results/summary.md`
