---
description: Debug training issues
---

## Environment Check
```bash
source /home/yeopjin/orcd/pool/init_protein_llm.sh
python -c "import torch, transformers, peft; print('Core libs OK')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## Memory Check
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

## Config Validation
```bash
python scripts/train.py --cfg job
python scripts/train.py --info defaults
```

## Common Issues

### CUDA OOM
- Reduce batch_size
- Enable gradient checkpointing
- Use 4-bit quantization (QLoRA)

### Triton Cache Slow
```bash
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
```

### Import Errors
```bash
pip install -e ".[dev,training]"
```

### Flash Attention
```bash
pip install flash-attn --no-build-isolation
```
