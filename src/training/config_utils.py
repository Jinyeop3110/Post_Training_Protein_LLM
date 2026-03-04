"""Training configuration utilities.

Extracted from sft_trainer.py — pure functions that build LoRA, quantization,
and TrainingArguments configs from Hydra DictConfig.
"""

import logging
from typing import Optional

import torch
from omegaconf import DictConfig

try:
    from transformers import BitsAndBytesConfig, TrainingArguments
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


log = logging.getLogger(__name__)


def get_qlora_config(cfg: DictConfig) -> LoraConfig:
    """Get QLoRA/LoRA configuration from Hydra config.

    Args:
        cfg: Hydra configuration containing training.lora settings.

    Returns:
        LoraConfig object for PEFT.

    Raises:
        ImportError: If PEFT is not installed.
    """
    if not HAS_PEFT:
        raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

    lora_cfg = cfg.training.lora if hasattr(cfg.training, "lora") else cfg.training.get("lora", {})

    # Handle OmegaConf objects
    if hasattr(lora_cfg, "get"):
        r = lora_cfg.get("r", 8)
        alpha = lora_cfg.get("alpha", 16)
        dropout = lora_cfg.get("dropout", 0.05)
        target_modules = list(lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]))
        bias = lora_cfg.get("bias", "none")
        task_type_str = lora_cfg.get("task_type", "CAUSAL_LM")
    else:
        r = getattr(lora_cfg, "r", 8)
        alpha = getattr(lora_cfg, "alpha", 16)
        dropout = getattr(lora_cfg, "dropout", 0.05)
        target_modules = list(getattr(lora_cfg, "target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]))
        bias = getattr(lora_cfg, "bias", "none")
        task_type_str = getattr(lora_cfg, "task_type", "CAUSAL_LM")

    # Map task type string to TaskType enum
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
    }
    task_type = task_type_map.get(task_type_str, TaskType.CAUSAL_LM)

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )


def get_quantization_config(cfg: DictConfig) -> Optional[BitsAndBytesConfig]:
    """Get quantization configuration from Hydra config.

    Args:
        cfg: Hydra configuration containing training.quantization settings.

    Returns:
        BitsAndBytesConfig for 4-bit quantization or None if disabled.
    """
    quant_cfg = cfg.training.get("quantization", {})

    if not quant_cfg.get("enabled", True):
        return None

    bits = quant_cfg.get("bits", 4)
    compute_dtype_str = quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    quant_type = quant_cfg.get("bnb_4bit_quant_type", "nf4")
    double_quant = quant_cfg.get("bnb_4bit_use_double_quant", True)

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(compute_dtype_str, torch.bfloat16)

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=double_quant,
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}. Use 4 or 8.")


def get_training_arguments(cfg: DictConfig) -> TrainingArguments:
    """Create TrainingArguments from Hydra config.

    Args:
        cfg: Hydra configuration.

    Returns:
        TrainingArguments for HuggingFace Trainer.
    """
    training_cfg = cfg.training
    paths_cfg = cfg.get("paths", {})
    logging_cfg = cfg.get("logging", {})

    # Get output directory
    output_dir = paths_cfg.get("checkpoint_dir", "./checkpoints")
    log_dir = paths_cfg.get("log_dir", "./logs")

    # Determine reporting backends
    report_to = []
    if logging_cfg.get("wandb", {}).get("enabled", False) and HAS_WANDB:
        report_to.append("wandb")
    if logging_cfg.get("tensorboard", {}).get("enabled", False):
        report_to.append("tensorboard")
    if not report_to:
        report_to = ["none"]

    # Get optimizer type
    optimizer_cfg = training_cfg.get("optimizer", {})
    optim_type = optimizer_cfg.get("type", "adamw_torch")

    # Map optimizer names
    optim_map = {
        "adamw_8bit": "adamw_bnb_8bit",
        "adamw": "adamw_torch",
        "adam": "adam",
        "sgd": "sgd",
        "adafactor": "adafactor",
    }
    optim = optim_map.get(optim_type, optim_type)

    # Get learning rate scheduler
    lr_scheduler_cfg = training_cfg.get("lr_scheduler", {})
    lr_scheduler_type = lr_scheduler_cfg.get("type", "cosine")
    warmup_steps = lr_scheduler_cfg.get("num_warmup_steps", 0)

    # FSDP configuration
    fsdp_cfg = training_cfg.get("fsdp", {})
    fsdp_option = ""
    fsdp_config_dict = None
    use_fsdp = fsdp_cfg.get("enabled", False)
    if use_fsdp:
        strategy = fsdp_cfg.get("strategy", "full_shard")
        fsdp_options = [strategy]
        if fsdp_cfg.get("auto_wrap", True):
            fsdp_options.append("auto_wrap")
        fsdp_option = " ".join(fsdp_options)

        fsdp_config_dict = {
            "min_num_params": int(fsdp_cfg.get("min_num_params", 1e8)),
            "backward_prefetch": fsdp_cfg.get("backward_prefetch", "backward_pre"),
            "forward_prefetch": fsdp_cfg.get("forward_prefetch", True),
            "use_orig_params": fsdp_cfg.get("use_orig_params", True),
        }
        if fsdp_cfg.get("cpu_offload", False):
            fsdp_config_dict["cpu_offload"] = True
        # NOTE: FSDP-native activation_checkpointing (fsdp_config key) is
        # incompatible with Qwen3's causal mask computation — recomputed
        # tensors differ between forward passes, causing CheckpointError.
        # We use HF Trainer's gradient_checkpointing instead, which works
        # but incurs a minor redundant AllGather in backward under full_shard.

        log.info(f"FSDP enabled: strategy={strategy}, config={fsdp_config_dict}")

    # max_steps overrides num_train_epochs when set (HF Trainer convention).
    # -1 means "use epochs instead" (TrainingArguments default).
    max_steps = training_cfg.get("max_steps", -1)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("epochs", 3),
        max_steps=max_steps,
        per_device_train_batch_size=training_cfg.get("batch_size", 8),
        per_device_eval_batch_size=training_cfg.get("eval_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("lr", 2e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        logging_dir=log_dir,
        logging_steps=training_cfg.get("logging_steps", 10),
        eval_strategy="steps" if training_cfg.get("eval_steps") else "epoch",
        eval_steps=training_cfg.get("eval_steps", 100),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 500),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        bf16=cfg.get("hardware", {}).get("precision", "bf16") == "bf16",
        fp16=cfg.get("hardware", {}).get("precision", "bf16") == "fp16",
        fsdp=fsdp_option if fsdp_option else "",
        fsdp_config=fsdp_config_dict,
        report_to=report_to,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,  # Reuse workers across epochs (skip fork overhead)
        dataloader_prefetch_factor=4,  # Prefetch 4 batches per worker (default=2)
        remove_unused_columns=False,  # Important for custom collator
        group_by_length=True,  # Groups similar-length sequences to reduce padding waste
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_drop_last=True,  # Prevent DDP hang on uneven last batch
        torch_compile=training_cfg.get("torch_compile", False),
        bf16_full_eval=True,  # Run eval in bf16 for speed
    )
