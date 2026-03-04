#!/bin/bash
# Profile batch configurations to find optimal throughput on 8×H100 with FSDP.
#
# Prerequisites: Environment must already be activated:
#   source /home/yeopjin/orcd/pool/init_protein_llm.sh
#
# Usage: bash scripts/analysis/profile_batch.sh
#
# Each config runs 5 training steps (no eval, no save, no compile) to measure
# peak GPU memory and throughput.

cd /orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM

export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
mkdir -p "${TRITON_CACHE_DIR}"

NUM_GPUS=$(nvidia-smi -L | wc -l)
RESULTS_DIR="results/profiling_$(date +%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

echo "=== Batch Profiling: ${NUM_GPUS} GPUs ===" | tee "${RESULTS_DIR}/profile.log"

# CSV header
echo "name,peak_gpu_mb,samples_per_sec,train_loss,oom,elapsed_sec" > "${RESULTS_DIR}/results.csv"

run_profile() {
    local name="$1"
    shift
    local all_args=("$@")

    echo "" | tee -a "${RESULTS_DIR}/profile.log"
    echo "--- ${name} ---" | tee -a "${RESULTS_DIR}/profile.log"
    echo "Args: ${all_args[*]}" | tee -a "${RESULTS_DIR}/profile.log"

    local logfile="${RESULTS_DIR}/${name}.log"
    local port=$((29500 + RANDOM % 100))
    local start_time=$(date +%s)

    # Run training with full output capture
    torchrun --nproc_per_node="${NUM_GPUS}" \
        --master_port="${port}" \
        scripts/train.py \
        "${all_args[@]}" \
        > "${logfile}" 2>&1
    local exit_code=$?

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Capture peak GPU memory (right after training, before it drops)
    local peak_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)

    # Extract metrics from log
    local train_samples_sec=$(grep -oP "'train_samples_per_second': [\d.]+" "${logfile}" | tail -1 | grep -oP '[\d.]+$') || true
    local train_loss=$(grep -oP "'train_loss': [\d.]+" "${logfile}" | tail -1 | grep -oP '[\d.]+$') || true

    # Check for OOM
    local oom="no"
    if grep -qi "out of memory\|CUDA OOM" "${logfile}"; then
        oom="YES"
    fi

    # Default to N/A
    train_samples_sec="${train_samples_sec:-N/A}"
    train_loss="${train_loss:-N/A}"

    echo "  Exit: ${exit_code} | Peak mem: ${peak_mem} MB | Samples/sec: ${train_samples_sec} | Loss: ${train_loss} | OOM: ${oom} | Time: ${elapsed}s" | tee -a "${RESULTS_DIR}/profile.log"
    echo "${name},${peak_mem},${train_samples_sec},${train_loss},${oom},${elapsed}" >> "${RESULTS_DIR}/results.csv"

    # Clean up experiment directory
    local exp_name=$(grep -oP 'experiment_name=\K\S+' <<< "${all_args[*]}") || true
    if [ -n "${exp_name}" ]; then
        rm -rf "results/${exp_name}" 2>/dev/null || true
    fi
}

# Common args for all ESM-3+MLP runs
MLP_BASE=(
    "experiment=sft_esm3_mlp_combined"
    "training.max_steps=5"
    "training.eval_steps=999"
    "training.save_steps=999"
    "training.torch_compile=false"
    "training.logging_steps=1"
    "data.limit=2000"
)

# Common args for text-only runs
TEXT_BASE=(
    "experiment=sft_text_combined"
    "training.max_steps=5"
    "training.eval_steps=999"
    "training.save_steps=999"
    "training.torch_compile=false"
    "training.logging_steps=1"
    "data.limit=2000"
)

echo ""
echo "========================================" | tee -a "${RESULTS_DIR}/profile.log"
echo "ESM-3 + MLP Projector Profiling" | tee -a "${RESULTS_DIR}/profile.log"
echo "========================================" | tee -a "${RESULTS_DIR}/profile.log"

# 1. Baseline (current config)
run_profile "esm3_mlp_baseline" \
    "${MLP_BASE[@]}" "experiment_name=prof_mlp_base" \
    "training.max_tokens_per_batch=8192" "training.max_batch_size=20" "training.gradient_accumulation_steps=4"

# 2. 12K tokens
run_profile "esm3_mlp_tokens12k" \
    "${MLP_BASE[@]}" "experiment_name=prof_mlp_12k" \
    "training.max_tokens_per_batch=12288" "training.max_batch_size=20" "training.gradient_accumulation_steps=4"

# 3. 16K tokens
run_profile "esm3_mlp_tokens16k" \
    "${MLP_BASE[@]}" "experiment_name=prof_mlp_16k" \
    "training.max_tokens_per_batch=16384" "training.max_batch_size=24" "training.gradient_accumulation_steps=3"

# 4. 20K tokens
run_profile "esm3_mlp_tokens20k" \
    "${MLP_BASE[@]}" "experiment_name=prof_mlp_20k" \
    "training.max_tokens_per_batch=20480" "training.max_batch_size=24" "training.gradient_accumulation_steps=2"

# 5. 24K tokens
run_profile "esm3_mlp_tokens24k" \
    "${MLP_BASE[@]}" "experiment_name=prof_mlp_24k" \
    "training.max_tokens_per_batch=24576" "training.max_batch_size=32" "training.gradient_accumulation_steps=2"

# 6. 16K tokens with lower grad accum
run_profile "esm3_mlp_tokens16k_accum2" \
    "${MLP_BASE[@]}" "experiment_name=prof_mlp_16k_a2" \
    "training.max_tokens_per_batch=16384" "training.max_batch_size=24" "training.gradient_accumulation_steps=2"

# 7. Encoder batch 6
run_profile "esm3_mlp_encbatch6" \
    "${MLP_BASE[@]}" "experiment_name=prof_mlp_enc6" \
    "training.max_tokens_per_batch=8192" "training.max_batch_size=20" "training.gradient_accumulation_steps=4" \
    "encoder.encoder_batch_size=6"

# 8. Encoder batch 8
run_profile "esm3_mlp_encbatch8" \
    "${MLP_BASE[@]}" "experiment_name=prof_mlp_enc8" \
    "training.max_tokens_per_batch=8192" "training.max_batch_size=20" "training.gradient_accumulation_steps=4" \
    "encoder.encoder_batch_size=8"

echo ""
echo "========================================" | tee -a "${RESULTS_DIR}/profile.log"
echo "Text-only Profiling" | tee -a "${RESULTS_DIR}/profile.log"
echo "========================================" | tee -a "${RESULTS_DIR}/profile.log"

# 9. Text baseline
run_profile "text_baseline" \
    "${TEXT_BASE[@]}" "experiment_name=prof_text_base" \
    "training.max_tokens_per_batch=8192" "training.max_batch_size=16" "training.gradient_accumulation_steps=4"

# 10. Text 16K
run_profile "text_tokens16k" \
    "${TEXT_BASE[@]}" "experiment_name=prof_text_16k" \
    "training.max_tokens_per_batch=16384" "training.max_batch_size=24" "training.gradient_accumulation_steps=3"

# 11. Text 24K
run_profile "text_tokens24k" \
    "${TEXT_BASE[@]}" "experiment_name=prof_text_24k" \
    "training.max_tokens_per_batch=24576" "training.max_batch_size=32" "training.gradient_accumulation_steps=2"

# 12. Text 32K
run_profile "text_tokens32k" \
    "${TEXT_BASE[@]}" "experiment_name=prof_text_32k" \
    "training.max_tokens_per_batch=32768" "training.max_batch_size=48" "training.gradient_accumulation_steps=2"

echo ""
echo "========================================" | tee -a "${RESULTS_DIR}/profile.log"
echo "Profiling complete!" | tee -a "${RESULTS_DIR}/profile.log"
echo "Results: ${RESULTS_DIR}/results.csv" | tee -a "${RESULTS_DIR}/profile.log"
echo "========================================" | tee -a "${RESULTS_DIR}/profile.log"

echo ""
echo "=== SUMMARY ==="
column -t -s',' "${RESULTS_DIR}/results.csv" 2>/dev/null || cat "${RESULTS_DIR}/results.csv"
