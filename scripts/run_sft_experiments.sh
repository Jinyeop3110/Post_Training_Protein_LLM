#!/bin/bash
# Three-way SFT comparison: text vs MLP vs Perceiver
#
# Dataset: Combined (Mol-Instructions + Swiss-Prot + Wikipedia Protein)
#   - 1.93M samples after α=0.5 temperature rebalancing
#   - Sources: mol (38%), sp (56%), wp (6%)
#
# Hardware: 8x H100 80GB, DDP via torchrun
# Training: QLoRA (4-bit), 3 epochs, batch_size=8, grad_accum=4
#   - Effective batch: 8 GPUs × 8 × 4 = 256
#   - Total steps: ~22,600 per experiment
#
# Usage:
#   bash scripts/run_sft_experiments.sh           # Run all three
#   bash scripts/run_sft_experiments.sh text      # Run text only
#   bash scripts/run_sft_experiments.sh mlp       # Run MLP only
#   bash scripts/run_sft_experiments.sh perceiver # Run Perceiver only

set -euo pipefail

# Source environment
source /home/yeopjin/orcd/pool/init_protein_llm.sh

# Triton cache on local disk
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
mkdir -p "${TRITON_CACHE_DIR}"

# DDP settings
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 100))}"

# Common settings for all experiments
COMMON_ARGS=(
    data=combined
    training=sft_qlora
    training.epochs=3
    training.batch_size=8
    training.gradient_accumulation_steps=4
    training.lr=2e-4
    training.projector_lr=2e-3
    training.save_steps=500
    training.eval_steps=200
    training.logging_steps=10
)

run_experiment() {
    local name="$1"
    shift
    echo ""
    echo "=============================================="
    echo "  Experiment: ${name}"
    echo "  GPUs: ${NUM_GPUS}, Port: ${MASTER_PORT}"
    echo "=============================================="
    echo ""

    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        scripts/train.py \
        experiment_name="${name}" \
        "${COMMON_ARGS[@]}" \
        "$@"

    # Increment port for next experiment
    MASTER_PORT=$((MASTER_PORT + 1))
}

# Determine which experiments to run
EXPERIMENT="${1:-all}"

# ----- Experiment 1: Text-only baseline -----
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "text" ]]; then
    run_experiment "sft_text_combined" \
        approach=text
fi

# ----- Experiment 2: ESM-3 + MLP Projector -----
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "mlp" ]]; then
    run_experiment "sft_esm3_mlp_combined" \
        approach=esm3 \
        encoder.projector.type=mlp
fi

# ----- Experiment 3: ESM-3 + Perceiver Resampler -----
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "perceiver" ]]; then
    run_experiment "sft_esm3_perceiver_combined" \
        approach=esm3 \
        encoder.projector.type=perceiver
fi

echo ""
echo "=============================================="
echo "  All requested experiments complete!"
echo "  Results in: results/"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  # Evaluate all three"
echo "  python scripts/evaluate.py experiment_name=sft_text_combined evaluation.name=all"
echo "  python scripts/evaluate.py experiment_name=sft_esm3_mlp_combined evaluation.name=all"
echo "  python scripts/evaluate.py experiment_name=sft_esm3_perceiver_combined evaluation.name=all"
echo ""
echo "  # Compare results"
echo "  ls results/*/eval/"
