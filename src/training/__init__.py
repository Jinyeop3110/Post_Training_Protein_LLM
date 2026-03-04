"""Training modules for SFT and RL.

This package provides training implementations for:
- SFT (Supervised Fine-Tuning) with QLoRA/LoRA
- GRPO (Group Relative Policy Optimization) with verifiable rewards
- DPO (Direct Preference Optimization) - Coming Soon
"""

from .callbacks import GenerationSamplesCallback, GPUMemoryCallback
from .collators import PackedDataCollator, PackedDataset, ProteinLLMDataCollator
from .config_utils import get_qlora_config, get_quantization_config, get_training_arguments
from .grpo_trainer import (
    GRPOTrainer,
    create_reward_dataset,
    get_grpo_config,
    run_grpo,
    run_grpo_with_trl,
)
from .rewards import (
    compute_esmfold_reward,
    compute_generic_reward,
    compute_go_reward,
    compute_ppi_reward,
    compute_stability_reward,
    get_reward_function,
)
from .sft_trainer import (
    ProteinLLMTrainer,
    SFTTrainer,
    run_sft,
    run_sft_lora,
    run_sft_qlora,
)
from .token_budget_sampler import TokenBudgetBatchSampler

__all__ = [
    # SFT Trainers
    "SFTTrainer",
    "ProteinLLMTrainer",
    "ProteinLLMDataCollator",
    "PackedDataset",
    "PackedDataCollator",
    "GPUMemoryCallback",
    "GenerationSamplesCallback",
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
    "compute_esmfold_reward",
    "compute_generic_reward",
    # Training functions
    "run_sft",
    "run_sft_qlora",
    "run_sft_lora",
    "run_grpo",
    "run_grpo_with_trl",
    # Utilities
    "create_reward_dataset",
    "TokenBudgetBatchSampler",
]
