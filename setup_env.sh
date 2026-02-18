#!/bin/bash
# Environment Setup Script for Post-Training Protein LLM with veRL + Flash Attention
# System: 8x NVIDIA H100 80GB, CUDA 13.0, Driver 580.105.08
#
# Tested Configuration:
#   - Python 3.11
#   - PyTorch 2.5.1+cu124
#   - Flash Attention 2.7.4.post1
#   - vLLM 0.6.6.post1
#   - veRL 0.8.0.dev

set -e

echo "============================================================"
echo "  Post-Training Protein LLM Environment Setup"
echo "  (veRL + Flash Attention)"
echo "============================================================"
echo ""

# Load conda and CUDA
echo "[1/8] Loading conda and CUDA..."
source /home/yeopjin/orcd/pool/conda_install/bin/activate
module load cuda/12.4.0
export CUDA_HOME=$CUDA_ROOT

# Environment configuration
ENV_NAME="protein_llm_flash"
PYTHON_VERSION="3.11"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} already exists."
    read -p "Do you want to remove and recreate it? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Activating existing environment..."
        conda activate ${ENV_NAME}
        echo "Environment activated. Exiting setup."
        exit 0
    fi
fi

# Create new conda environment
echo ""
echo "[2/8] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
conda activate ${ENV_NAME}

echo ""
echo "[3/8] Installing vLLM 0.6.6.post1 (includes PyTorch 2.5.1)..."
pip install vllm==0.6.6.post1

echo ""
echo "[4/8] Installing Flash Attention 2.7.4.post1..."
# Download and install the correct wheel (CXX11_ABI=FALSE)
cd /tmp
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install /tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
rm -f /tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo ""
echo "[5/8] Installing veRL and RL training dependencies..."
# Clone and install veRL
VERL_DIR="${HOME}/.verl_src"
if [ -d "${VERL_DIR}" ]; then
    echo "Updating existing veRL source..."
    cd ${VERL_DIR} && git pull
else
    echo "Cloning veRL repository..."
    git clone https://github.com/volcengine/verl.git ${VERL_DIR}
fi
cd ${VERL_DIR}
pip install --no-deps -e .

# Install veRL dependencies
pip install "peft>=0.10.0"
pip install "deepspeed>=0.14.0"
pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
pip install "numpy<2.0.0"
pip install hydra-core codetiming
pip install "optree>=0.13.0"
pip install datasets torchdata pybind11 pylatexenc

echo ""
echo "[6/8] Installing protein-specific libraries..."
pip install fair-esm biopython scipy pandas

echo ""
echo "[7/8] Installing experiment tracking and utilities..."
pip install wandb tensorboard matplotlib seaborn tqdm dill "bitsandbytes>=0.43.0"

echo ""
echo "[8/8] Verifying installation..."
echo ""
echo "============================================================"
echo "  Installation Verification"
echo "============================================================"

python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.cuda.is_available()} ({torch.version.cuda})')
print(f'GPUs:          {torch.cuda.device_count()}')

import flash_attn
print(f'Flash Attn:    {flash_attn.__version__}')

# Test flash attention
from flash_attn import flash_attn_func
q = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v)
print(f'Flash Compute: PASSED')

import vllm
print(f'vLLM:          {vllm.__version__}')

import peft
print(f'PEFT:          {peft.__version__}')

import deepspeed
print(f'DeepSpeed:     {deepspeed.__version__}')

import verl
print(f'veRL:          installed')

import esm
print(f'ESM:           installed')
"

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment, run:"
echo "  source /home/yeopjin/orcd/pool/init_protein_llm.sh"
echo ""
echo "veRL source location: ${VERL_DIR}"
echo ""
