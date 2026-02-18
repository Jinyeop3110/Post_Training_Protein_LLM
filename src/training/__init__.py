"""Training modules for SFT and RL.

This package provides training implementations for:
- SFT (Supervised Fine-Tuning) with QLoRA/LoRA
- GRPO (Group Relative Policy Optimization) - Coming Soon
- DPO (Direct Preference Optimization) - Coming Soon
"""

from .sft_trainer import (
    SFTTrainer,
    ProteinLLMTrainer,
    ProteinLLMDataCollator,
    GPUMemoryCallback,
    get_qlora_config,
    get_quantization_config,
    get_training_arguments,
    run_sft_qlora,
    run_sft_lora,
    run_sft_with_trl,
)

__all__ = [
    # SFT Trainers
    "SFTTrainer",
    "ProteinLLMTrainer",
    "ProteinLLMDataCollator",
    "GPUMemoryCallback",
    # Config functions
    "get_qlora_config",
    "get_quantization_config",
    "get_training_arguments",
    # Training functions
    "run_sft_qlora",
    "run_sft_lora",
    "run_sft_with_trl",
]
