"""Training modules for SFT and RL.

This package provides training implementations for:
- SFT (Supervised Fine-Tuning) with QLoRA/LoRA
- GRPO (Group Relative Policy Optimization) with verifiable rewards
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

from .grpo_trainer import (
    GRPOTrainer,
    get_grpo_config,
    get_reward_function,
    compute_go_reward,
    compute_ppi_reward,
    compute_stability_reward,
    compute_generic_reward,
    run_grpo,
    run_grpo_with_trl,
    create_reward_dataset,
)

__all__ = [
    # SFT Trainers
    "SFTTrainer",
    "ProteinLLMTrainer",
    "ProteinLLMDataCollator",
    "GPUMemoryCallback",
    # GRPO Trainer
    "GRPOTrainer",
    # Config functions
    "get_qlora_config",
    "get_quantization_config",
    "get_training_arguments",
    "get_grpo_config",
    "get_reward_function",
    # Reward functions
    "compute_go_reward",
    "compute_ppi_reward",
    "compute_stability_reward",
    "compute_generic_reward",
    # Training functions
    "run_sft_qlora",
    "run_sft_lora",
    "run_sft_with_trl",
    "run_grpo",
    "run_grpo_with_trl",
    # Utilities
    "create_reward_dataset",
]
