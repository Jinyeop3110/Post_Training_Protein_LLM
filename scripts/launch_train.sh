#!/bin/bash
# DDP training launcher for multi-GPU training with torchrun.
#
# Usage:
#   bash scripts/launch_train.sh training=sft_lora
#   bash scripts/launch_train.sh model=llama3_8b training.batch_size=8 training.gradient_accumulation_steps=8
#   NUM_GPUS=4 bash scripts/launch_train.sh training=sft_lora
#   MASTER_PORT=29501 bash scripts/launch_train.sh training=sft_lora
set -euo pipefail

# Source environment
source /home/yeopjin/orcd/pool/init_protein_llm.sh

# Triton cache must be on local disk (not NFS)
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
mkdir -p "${TRITON_CACHE_DIR}"

# Detect GPU count, allow override
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

# Random master port to avoid collisions between concurrent jobs
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 100))}"

echo "DDP Training: ${NUM_GPUS} GPUs, port ${MASTER_PORT}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    scripts/train.py "$@"
