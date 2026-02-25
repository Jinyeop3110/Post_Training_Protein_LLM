# Troubleshooting Guide

## Common Issues

### 1. CUDA Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error

**Solutions**:
```bash
# Reduce batch size
python scripts/train.py training.batch_size=4

# Enable gradient checkpointing
python scripts/train.py training.gradient_checkpointing=true

# Use 4-bit quantization
python scripts/train.py training=sft_qlora

# Reduce sequence length
python scripts/train.py training.max_seq_length=1024
```

### 2. Slow Startup / Triton Compilation

**Symptoms**: Training takes 10+ minutes to start

**Solution**:
```bash
# Use local storage for Triton cache
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER

# Clear existing cache
rm -rf $TRITON_CACHE_DIR
```

### 3. Flash Attention Not Found

**Symptoms**: `No module named 'flash_attn'`

**Solution**:
```bash
pip install flash-attn --no-build-isolation
```

If still failing:
```bash
pip install flash-attn==2.5.8 --no-build-isolation
```

### 4. Import Errors

**Symptoms**: `ModuleNotFoundError`

**Solutions**:
```bash
# Install package in editable mode
pip install -e ".[dev,training]"

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 5. Hydra Config Errors

**Symptoms**: `ConfigCompositionException` or `MissingMandatoryValue`

**Solutions**:
```bash
# Print resolved config to debug
python scripts/train.py --cfg job

# Check config structure
python scripts/train.py --info defaults
```

### 6. NCCL Timeout (Multi-GPU)

**Symptoms**: `NCCL timeout` or training hangs

**Solutions**:
```bash
# Increase timeout
export NCCL_TIMEOUT=3600

# Debug NCCL
export NCCL_DEBUG=INFO
```

### 7. WandB Issues

**Symptoms**: `wandb: Error` or authentication issues

**Solutions**:
```bash
# Login
wandb login

# Disable if not needed
python scripts/train.py logging.wandb.enabled=false

# Offline mode
export WANDB_MODE=offline
```

### 8. ESM-3 Loading Issues

**Symptoms**: ESM-3 model fails to load

**Solutions**:
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/hub/models--esm*

# Check available memory
nvidia-smi

# Verify encoder config
python scripts/train.py encoder=esm3_small --cfg job
```

## Debugging Checklist

1. **Verify environment**:
   ```bash
   source /home/yeopjin/orcd/pool/init_protein_llm.sh
   python -c "import torch, transformers, peft; print('OK')"
   ```

2. **Check GPU availability**:
   ```bash
   nvidia-smi
   python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
   ```

3. **Validate config**:
   ```bash
   python scripts/train.py --cfg job
   ```

4. **Run minimal test**:
   ```bash
   pytest tests/ -v -x
   ```

5. **Check disk space**:
   ```bash
   df -h
   ```

## Getting Help

1. Check existing documentation in `docs/`
2. Search research logs in `docs/research/`
3. Review similar issues in the codebase
