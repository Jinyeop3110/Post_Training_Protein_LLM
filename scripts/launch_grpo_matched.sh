#!/bin/bash
# Launch matched GRPO structure quality runs: MLP SFT → GRPO, then Text SFT → GRPO
# Both use enable_thinking=false, epochs=1, DDP on 8×H100
# Uses config defaults (no LR/epoch overrides — values set in yaml)
set -eo pipefail

source /home/yeopjin/orcd/pool/init_protein_llm.sh
set -u  # Enable after sourcing init script (which has unbound vars)
cd /orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

run_grpo() {
    local config="$1"
    local logfile="$2"
    shift 2
    local PORT=$((29500 + RANDOM % 100))
    echo ""
    echo "============================================================"
    echo "Launching: ${config} (port ${PORT})"
    echo "  Overrides: $@"
    echo "  Log: ${logfile}"
    echo "  Time: $(date)"
    echo "============================================================"
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${PORT}" \
        scripts/train.py \
        main_RL="${config}" \
        "$@" \
        2>&1 | tee "${logfile}"
    echo "Finished: ${config} at $(date)"
    echo ""
}

echo "Starting matched GRPO structure quality runs at $(date)"

# 1. ESM3+MLP SFT → GRPO (structure quality)
run_grpo grpo_structure_mlp_sft \
    /tmp/grpo_structure_mlp_sft.log

# 2. Text SFT → GRPO (structure quality)
run_grpo grpo_structure_text_sft \
    /tmp/grpo_structure_text_sft.log

echo "All matched GRPO runs complete at $(date)"
