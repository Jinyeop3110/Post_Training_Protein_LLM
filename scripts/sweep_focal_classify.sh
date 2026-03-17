#!/bin/bash
# Sweep focal gamma for classification-only GRPO: TEXT only
# Runs sequentially on 8×H100 GPUs.
#
# Gamma values: 1, 3, 5, 8, 12
# (gamma=0 skipped — no focal baseline already known from prior runs)
# (gamma=2 already done: grpo_structure_text_classify)
#
# Usage:
#   nohup bash scripts/sweep_focal_classify.sh > /tmp/grpo_sweep_focal.log 2>&1 &

set -euo pipefail
source /home/yeopjin/orcd/pool/init_protein_llm.sh
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
export NCCL_TIMEOUT=1800

cd /home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM

GAMMAS=(1 3 5 8 12)
NUM_GPUS=8

run_experiment() {
    local gamma=$1
    local port=$((29500 + RANDOM % 200))

    local focal_args="training.grpo.classification_only=true"
    focal_args="${focal_args} training.grpo.focal_enabled=true training.grpo.focal_gamma=${gamma}"

    echo ""
    echo "============================================================"
    echo "$(date): Starting TEXT gamma=${gamma}"
    echo "============================================================"

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${port} \
        scripts/train.py \
        main_RL=grpo_structure_text_sft \
        training.batch_size=2 \
        training.gradient_accumulation_steps=4 \
        ${focal_args} \
        "experiment_name=sweep_classify_text_gamma${gamma}_\${model.name}_\${now:%m%d_%H%M%S}"

    echo "$(date): Finished TEXT gamma=${gamma}"
    echo ""
}

echo "============================================================"
echo "FOCAL GAMMA SWEEP: Classification-Only GRPO (Text)"
echo "Gammas: ${GAMMAS[*]}"
echo "Started: $(date)"
echo "============================================================"

for gamma in "${GAMMAS[@]}"; do
    run_experiment "$gamma"
done

echo ""
echo "============================================================"
echo "SWEEP COMPLETE: $(date)"
echo "============================================================"
