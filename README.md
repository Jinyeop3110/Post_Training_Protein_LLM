# Post-Training Protein LLM

Train and post-train LLMs to understand proteins as multimodal embeddings using Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).

## Overview

This project integrates protein language model embeddings (ESM-2) with large language models to enable protein understanding tasks such as:
- GO term prediction
- Protein-protein interaction prediction
- Stability prediction
- Subcellular localization

## System Requirements

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA H100 80GB (8x available) |
| CUDA | 12.4 (for Flash Attention compatibility) |
| Driver | 580.105.08 |
| Python | 3.11 |

## Environment Setup

### Quick Start

```bash
# Clone the repository (if not already done)
cd /home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM

# Make setup script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

### Manual Installation

If you prefer to install manually or need to customize the setup:

#### 1. Activate Conda and Load CUDA

```bash
source /home/yeopjin/orcd/pool/conda_install/bin/activate
module load cuda/12.4.0
export CUDA_HOME=$CUDA_ROOT
```

#### 2. Create Conda Environment

```bash
conda create -n protein_llm_flash python=3.11 -y
conda activate protein_llm_flash
```

#### 3. Install vLLM (will install compatible PyTorch)

```bash
# vLLM 0.6.6.post1 includes PyTorch 2.5.1+cu124
pip install vllm==0.6.6.post1
```

#### 4. Install Flash Attention

```bash
# Download pre-built wheel (CXX11_ABI=FALSE for pip PyTorch)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

#### 5. Install veRL (Training Framework)

```bash
# Clone veRL repository
git clone https://github.com/volcengine/verl.git ~/.verl_src
cd ~/.verl_src
pip install --no-deps -e .

# Install veRL dependencies
pip install "transformers>=4.51.0" "accelerate>=0.30.0" "datasets>=2.18.0"
pip install "peft>=0.10.0" "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
pip install "numpy<2.0.0" "pyarrow>=15.0.0" "ray[default]>=2.10.0"
pip install hydra-core codetiming "optree>=0.13.0" "pydantic>=2.9"
pip install "grpcio>=1.62.1" "fastapi>=0.115.0" "nvidia-ml-py>=12.560.30"
pip install "deepspeed>=0.14.0"
```

#### 6. Install Protein Libraries

```bash
pip install fair-esm biopython scipy pandas
```

#### 7. Install Utilities

```bash
pip install wandb tensorboard matplotlib seaborn tqdm dill "bitsandbytes>=0.43.0"
```

## Package Versions (Tested)

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Runtime |
| PyTorch | 2.5.1+cu124 | Deep learning framework |
| CUDA (PyTorch) | 12.4 | GPU acceleration |
| **Flash Attention** | **2.7.4.post1** | **Optimized attention for H100** |
| vLLM | 0.6.6.post1 | Fast inference engine |
| veRL | 0.8.0.dev | RL training framework |
| Transformers | 5.1.0 | Model architectures |
| PEFT | 0.18.1 | LoRA/QLoRA fine-tuning |
| DeepSpeed | 0.18.6 | Distributed training |
| Ray | 2.53.0 | Distributed computing |
| ESM | 2.0.0 | Protein language models |
| BioPython | 1.86 | Protein sequence handling |

**Note**: Flash Attention 2.7.4.post1 is installed and working. This required using PyTorch 2.5.1 and vLLM 0.6.6.post1 due to GLIBC 2.28 compatibility on this system.

## Why veRL?

veRL (Volcano Engine RL) was chosen as the training framework for this project because:

1. **Scalability**: Supports models up to 671B parameters across hundreds of GPUs
2. **Performance**: Optimized actor-rollout weight sharing eliminates CUDA IPC overhead
3. **Flexibility**: Supports multiple backends (FSDP, DeepSpeed, Megatron-LM)
4. **RL Methods**: PPO, GRPO, DAPO, REINFORCE++, and more
5. **Production-Ready**: Used by ByteDance, achieving 50 points on AIME 2024 with DAPO

### veRL vs Alternatives

| Feature | veRL | TRL | OpenRLHF |
|---------|------|-----|----------|
| Max Model Size | 671B | ~70B | 70B+ |
| RL Algorithms | PPO, GRPO, DAPO, GSPO | PPO, DPO | PPO, DPO |
| Backend | FSDP/DeepSpeed/Megatron | HF Trainer | Ray + vLLM |
| Learning Curve | Medium | Easy | Medium |

## Activation

After installation, activate the environment:

```bash
source /home/yeopjin/orcd/pool/init_protein_llm.sh
```

This init script will:
- Activate the `protein_llm_flash` conda environment
- Load CUDA 12.4.0 and set `CUDA_HOME`
- Set `TRITON_CACHE_DIR` to local storage (avoids NFS slowdown)
- Login to HuggingFace and Wandb
- Change to the project directory

## Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')

import vllm
print(f'vLLM: {vllm.__version__}')

import verl
print('veRL: installed')
"
```

## Project Structure

```
Post_Training_Protein_LLM/
├── README.md                 # This file
├── setup_env.sh              # Environment setup script
├── agents.md                 # Multi-agent research log
├── LLM_Post_Training_Methods_Summary.md
├── plan/
│   ├── literature/
│   ├── datasets/
│   ├── evaluation/
│   └── methods/
├── research/
│   └── protein_datasets_and_benchmarks.md
├── data/                     # Downloaded datasets (gitignored)
│   └── pdb_2021aug02_sample/ # IPD PDB sample dataset
└── src/
    ├── models/
    │   └── protein_encoder.py
    ├── training/
    ├── data/
    │   ├── pdb_dataset.py    # IPD PDB dataloader
    │   ├── rcsb_dataset.py   # RCSB PDB/mmCIF dataloader
    │   └── download.py       # Dataset download utilities
    └── evaluation/
```

## Data Pipeline

```
1. Download (raw)     2. Prepare (processed)     3. Train
   download.py    →      prepare_data.py      →   train.py
       ↓                       ↓                     ↓
   data/raw/             data/processed/        checkpoints/
```

### Available Datasets

| Dataset | Description | Size | Format |
|---------|-------------|------|--------|
| IPD PDB Sample | RoseTTAFold/ProteinMPNN training set | ~47MB, 556K chains | PyTorch .pt |
| RCSB PDB | Standard PDB structures | Per structure | PDB/mmCIF |
| Mol-Instructions | Protein instruction pairs | 505K pairs | JSON |
| Swiss-Prot | Curated protein sequences | ~90MB, 570K seqs | FASTA |
| AlphaFold | Predicted structures | Per structure | PDB |

### Step 1: Download Datasets

```bash
# List available datasets
python src/data/download.py --dataset list

# Download IPD PDB sample (recommended for training)
python src/data/download.py --dataset ipd_pdb_sample --output_dir ./data

# Download Swiss-Prot sequences
python src/data/download.py --dataset swissprot --output_dir ./data

# Or download manually:
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz
tar xvf pdb_2021aug02_sample.tar.gz
rm pdb_2021aug02_sample.tar.gz
```

### Using the Dataloaders

#### IPD PDB Dataset (Pre-processed .pt files)

```python
from src.data import get_pdb_dataloader

# Load IPD PDB sample dataset
dataloader = get_pdb_dataloader(
    data_dir="./data/pdb_2021aug02_sample",
    batch_size=32,
    max_length=512,       # Filter by sequence length
    max_resolution=3.0,   # Filter by structure resolution
)

for batch in dataloader:
    sequences = batch['sequence']      # List of AA strings
    coords = batch['coords']           # [B, L, 14, 3] atom coordinates
    lengths = batch['length']          # [B] sequence lengths
    mask = batch['mask']               # [B, L, 14] valid atom mask
```

#### RCSB PDB Dataset (Standard PDB/mmCIF files)

```python
from src.data import get_rcsb_dataloader

# Download and load specific PDB structures
dataloader = get_rcsb_dataloader(
    pdb_ids=["1crn", "1ubq", "2gb1", "1tim"],
    pdb_dir="./data/pdb_files",
    batch_size=4,
    download=True,        # Auto-download from RCSB
)

for batch in dataloader:
    pdb_ids = batch['pdb_id']          # List of PDB IDs
    chains = batch['chain_id']         # List of chain IDs
    sequences = batch['sequence']      # List of AA strings
    coords = batch['coords']           # [B, L, 14, 3]
```

#### Download Utilities

```python
from src.data import (
    download_ipd_pdb_sample,
    download_rcsb_structures,
    download_swissprot_sequences,
    download_alphafold_structures,
)

# Download IPD PDB sample
data_dir = download_ipd_pdb_sample("./data")

# Download specific PDB structures
files = download_rcsb_structures(
    pdb_ids=["1crn", "1ubq", "2gb1"],
    output_dir="./data/pdb_files",
    file_format="pdb",  # or "cif"
)

# Download Swiss-Prot sequences
fasta_file = download_swissprot_sequences("./data")

# Download AlphaFold structures
files = download_alphafold_structures(
    uniprot_ids=["P00533", "P04637"],
    output_dir="./data/alphafold",
)
```

### Step 2: Prepare Data (Preprocessing)

After downloading, preprocess the raw data into training format:

```bash
# Prepare Mol-Instructions dataset
python scripts/prepare_data.py data=mol_instructions

# Prepare IPD PDB dataset
python scripts/prepare_data.py data=ipd_pdb

# Prepare Swiss-Prot dataset
python scripts/prepare_data.py data=swissprot
```

This creates processed data in `data/processed/` with train/val/test splits.

### Step 3: Explore Datasets

Use the exploration notebook to inspect dataset contents:

```bash
jupyter notebook scripts/explore_datasets.ipynb
```

## Training Strategy

### Phase 1: Supervised Fine-Tuning (SFT)

- **Method**: QLoRA (4-bit quantization)
- **LoRA Rank**: r=8 (minimum r=4 for protein models)
- **Learning Rate**: 2e-4
- **Epochs**: 1-3

### Phase 2: Reinforcement Learning

- **Framework**: veRL with FSDP backend
- **Method**: GRPO (recommended) or DPO
- **Learning Rate**: 5e-6

## Troubleshooting

### Common Issues

1. **vLLM import errors**
   - Ensure PyTorch is installed first before vLLM
   - Check CUDA compatibility: `python -c "import torch; print(torch.version.cuda)"`

2. **Flash Attention issues**
   - Flash Attention 2.8+ requires GLIBC 2.32, but this system has GLIBC 2.28
   - Solution: Use flash-attn 2.7.4.post1 with PyTorch 2.5.1 and the correct CXX11_ABI wheel

3. **veRL import errors**
   - Verify editable install: `pip show verl`
   - Check source exists: `ls ~/.verl_src`

4. **Out of Memory (OOM)**
   - Use FSDP sharding: Set `training.fsdp.enabled=true` in config
   - Enable gradient checkpointing
   - Reduce batch size

### Environment Variables

```bash
# Required for DeepSpeed (set by init script)
export CUDA_HOME=$CUDA_ROOT

# Triton cache (recommended for NFS systems)

# Triton cache on local storage (avoids NFS slowdown)
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER

# For multi-node training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## References

- [veRL Documentation](https://verl.readthedocs.io/)
- [veRL GitHub](https://github.com/volcengine/verl)
- [ESM GitHub](https://github.com/facebookresearch/esm)
- [vLLM Documentation](https://docs.vllm.ai/)

## License

This project is for research purposes.
