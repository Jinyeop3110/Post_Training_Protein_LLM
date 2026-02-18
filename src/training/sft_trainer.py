"""
SFT Trainer Implementation

This module provides Supervised Fine-Tuning (SFT) functionality using TRL's SFTTrainer
with QLoRA/LoRA for efficient training of the multimodal ProteinLLM.

Main components:
- get_qlora_config: Extract LoRA configuration from Hydra config
- run_sft_qlora: Run SFT training with QLoRA (4-bit quantization)
- run_sft_lora: Run SFT training with LoRA (no quantization)
- SFTTrainerWrapper: Wrapper class for training with custom ProteinLLM forward pass
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

try:
    from dataclasses import dataclass, field
except ImportError:
    # Python < 3.7 compatibility
    dataclass = None
    field = None

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        PreTrainedTokenizer,
        PreTrainedModel,
        TrainingArguments,
        Trainer,
        TrainerCallback,
        TrainerState,
        TrainerControl,
    )
    from transformers.trainer_utils import EvalPrediction
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from trl import SFTTrainer as TRLSFTTrainer, SFTConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False

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
        target_modules = list(lora_cfg.get("target_modules", ["k_proj", "v_proj"]))
        bias = lora_cfg.get("bias", "none")
        task_type_str = lora_cfg.get("task_type", "CAUSAL_LM")
    else:
        r = getattr(lora_cfg, "r", 8)
        alpha = getattr(lora_cfg, "alpha", 16)
        dropout = getattr(lora_cfg, "dropout", 0.05)
        target_modules = list(getattr(lora_cfg, "target_modules", ["k_proj", "v_proj"]))
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

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("epochs", 3),
        per_device_train_batch_size=training_cfg.get("batch_size", 8),
        per_device_eval_batch_size=training_cfg.get("batch_size", 8),
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
        report_to=report_to,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important for custom collator
        group_by_length=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


class ProteinLLMDataCollator:
    """
    Custom data collator for ProteinLLM training.

    Handles tokenization while preserving protein sequences for the
    multimodal forward pass.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        padding: str = "longest",
        label_pad_token_id: int = -100,
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            label_pad_token_id: Token ID for padding labels.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.

        Args:
            batch: List of samples from MolInstructionsDataset.

        Returns:
            Dict containing tokenized inputs and protein sequences.
        """
        # Extract formatted prompts and protein sequences
        prompts = [item["formatted_prompt"] for item in batch]
        protein_sequences = [item["protein_sequence"] for item in batch]

        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        # Create labels (same as input_ids, with padding masked)
        labels = encoded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "protein_sequences": protein_sequences,
        }


class GPUMemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage during training."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            # Get current GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

            if logs is not None:
                logs["gpu_memory_allocated_gb"] = round(allocated, 2)
                logs["gpu_memory_reserved_gb"] = round(reserved, 2)
                logs["gpu_memory_max_allocated_gb"] = round(max_allocated, 2)

            log.debug(
                f"GPU Memory - Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB"
            )


class ProteinLLMTrainer(Trainer):
    """
    Custom Trainer for ProteinLLM that handles the multimodal forward pass.

    This trainer overrides compute_loss to properly handle protein sequence
    encoding and the custom forward pass of ProteinLLM.
    """

    def __init__(
        self,
        protein_llm: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Initialize the trainer.

        Args:
            protein_llm: The full ProteinLLM model (with encoder, pooling, projector).
            **kwargs: Arguments passed to the base Trainer.
        """
        super().__init__(**kwargs)
        self.protein_llm = protein_llm

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """
        Compute training loss with custom ProteinLLM forward pass.

        Args:
            model: The model (LLM with LoRA adapters).
            inputs: Batch inputs including protein_sequences.
            return_outputs: Whether to return model outputs.
            num_items_in_batch: Number of items in batch (unused).

        Returns:
            Loss tensor or (loss, outputs) tuple.
        """
        # Extract protein sequences (not tokenized, just raw strings)
        protein_sequences = inputs.pop("protein_sequences", None)

        # If we have a full ProteinLLM model with encoder, use custom forward
        if self.protein_llm is not None and protein_sequences is not None:
            # Use the full ProteinLLM forward pass
            outputs = self.protein_llm(
                protein_sequences=protein_sequences,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs.get("labels"),
            )
            loss = outputs["loss"]

            if return_outputs:
                return loss, outputs
            return loss

        # Fallback to standard forward pass (text-only)
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)

        if labels is not None:
            # Compute language modeling loss
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss


class SFTTrainer:
    """
    SFT trainer class for ProteinLLM.

    This class wraps the training setup and execution for supervised fine-tuning
    of the multimodal protein-language model.

    Args:
        cfg: Hydra configuration.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the SFT trainer."""
        self.cfg = cfg
        self.model = None
        self.protein_llm = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.data_collator = None

        # Validate dependencies
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers is required. Install with: pip install transformers"
            )
        if not HAS_PEFT:
            raise ImportError(
                "PEFT is required for LoRA. Install with: pip install peft"
            )

    def setup(self) -> None:
        """Set up model, tokenizer, and dataset."""
        log.info("Setting up SFT trainer...")

        # Initialize logging
        self._setup_logging()

        # Load tokenizer
        self._load_tokenizer()

        # Load model
        self._load_model()

        # Load datasets
        self._load_datasets()

        # Create data collator
        self._create_collator()

        # Create trainer
        self._create_trainer()

        log.info("SFT trainer setup complete")

    def _setup_logging(self) -> None:
        """Set up wandb and other logging."""
        logging_cfg = self.cfg.get("logging", {})

        # Initialize wandb if enabled
        if logging_cfg.get("wandb", {}).get("enabled", False) and HAS_WANDB:
            wandb.init(
                project=logging_cfg.wandb.get("project", "protein_llm"),
                name=logging_cfg.wandb.get("name", self.cfg.get("experiment_name", "sft")),
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
            log.info("Wandb logging initialized")

    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        model_path = self.cfg.model.path
        log.info(f"Loading tokenizer from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

    def _load_model(self) -> None:
        """Load the model with LoRA configuration."""
        model_path = self.cfg.model.path
        use_quantization = self.cfg.training.get("quantization", {}).get("enabled", True)

        log.info(f"Loading model from: {model_path}")
        log.info(f"Using quantization: {use_quantization}")

        # Get quantization config
        quantization_config = get_quantization_config(self.cfg) if use_quantization else None

        # Load base model
        if quantization_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Apply LoRA
        lora_config = get_qlora_config(self.cfg)
        self.model = get_peft_model(self.model, lora_config)

        log.info("LoRA configuration applied")
        self.model.print_trainable_parameters()

        # Optionally load full ProteinLLM for multimodal training
        if self.cfg.get("use_multimodal", False):
            self._load_protein_llm()

    def _load_protein_llm(self) -> None:
        """Load the full ProteinLLM model for multimodal training."""
        try:
            from src.models.multimodal_llm import ProteinLLM

            log.info("Loading ProteinLLM for multimodal training...")

            self.protein_llm = ProteinLLM.from_config(self.cfg)

            # Replace the LLM component with our LoRA model
            self.protein_llm.llm = self.model
            self.protein_llm.tokenizer = self.tokenizer

            log.info("ProteinLLM loaded successfully")
            self.protein_llm.print_trainable_parameters()

        except ImportError as e:
            log.warning(f"Could not load ProteinLLM: {e}. Using text-only training.")
            self.protein_llm = None

    def _load_datasets(self) -> None:
        """Load training and validation datasets."""
        from src.data.mol_instructions import MolInstructionsDataset

        data_cfg = self.cfg.data

        log.info("Loading training dataset...")
        self.train_dataset = MolInstructionsDataset(
            split="train",
            dataset_name=data_cfg.get("source", "zjunlp/Mol-Instructions"),
            subset=data_cfg.get("subset", "Protein-oriented Instructions"),
            cache_dir=data_cfg.get("paths", {}).get("raw"),
            max_seq_length=self.cfg.training.get("max_seq_length", 2048),
        )
        log.info(f"Training dataset loaded: {len(self.train_dataset)} samples")

        log.info("Loading validation dataset...")
        self.eval_dataset = MolInstructionsDataset(
            split="validation",
            dataset_name=data_cfg.get("source", "zjunlp/Mol-Instructions"),
            subset=data_cfg.get("subset", "Protein-oriented Instructions"),
            cache_dir=data_cfg.get("paths", {}).get("raw"),
            max_seq_length=self.cfg.training.get("max_seq_length", 2048),
        )
        log.info(f"Validation dataset loaded: {len(self.eval_dataset)} samples")

    def _create_collator(self) -> None:
        """Create the data collator."""
        self.data_collator = ProteinLLMDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.cfg.training.get("max_seq_length", 2048),
            padding="longest",
        )

    def _create_trainer(self) -> None:
        """Create the HuggingFace Trainer."""
        training_args = get_training_arguments(self.cfg)

        # Add callbacks
        callbacks = [GPUMemoryCallback()]

        # Use custom trainer for multimodal or standard trainer for text-only
        if self.protein_llm is not None:
            self.trainer = ProteinLLMTrainer(
                protein_llm=self.protein_llm,
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=self.data_collator,
                callbacks=callbacks,
            )
        else:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=self.data_collator,
                callbacks=callbacks,
            )

        log.info("Trainer created successfully")

    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Training metrics dictionary.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        log.info("Starting training...")
        log.info(f"  Epochs: {self.cfg.training.epochs}")
        log.info(f"  Batch size: {self.cfg.training.batch_size}")
        log.info(f"  Learning rate: {self.cfg.training.lr}")
        log.info(f"  Gradient accumulation steps: {self.cfg.training.gradient_accumulation_steps}")

        # Train
        train_result = self.trainer.train()

        # Log final metrics
        metrics = train_result.metrics
        log.info(f"Training completed. Final loss: {metrics.get('train_loss', 'N/A')}")

        # Save final model
        self.save_checkpoint(self.trainer.args.output_dir)

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Evaluation metrics dictionary.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        log.info("Running evaluation...")
        metrics = self.trainer.evaluate()

        log.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Directory path to save the checkpoint.
        """
        log.info(f"Saving checkpoint to: {path}")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the model (LoRA adapters)
        self.model.save_pretrained(path / "adapter")

        # Save tokenizer
        self.tokenizer.save_pretrained(path / "tokenizer")

        # Save trainer state
        if self.trainer is not None:
            self.trainer.save_state()

        # Save ProteinLLM components if available
        if self.protein_llm is not None:
            self.protein_llm.save_pretrained(path / "protein_llm")

        log.info(f"Checkpoint saved to {path}")


def run_sft_qlora(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run SFT training with QLoRA.

    This function orchestrates the full SFT training pipeline with
    4-bit quantization (QLoRA) for memory-efficient fine-tuning.

    Args:
        cfg: Hydra configuration.

    Returns:
        Training metrics dictionary.
    """
    log.info("=" * 60)
    log.info("Starting SFT with QLoRA")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model.path}")
    log.info(f"Encoder: {cfg.encoder.model_name}")
    log.info(f"Learning rate: {cfg.training.lr}")
    log.info(f"Batch size: {cfg.training.batch_size}")
    log.info(f"Epochs: {cfg.training.epochs}")

    # Ensure quantization is enabled
    if not cfg.training.get("quantization", {}).get("enabled", True):
        log.warning("QLoRA requested but quantization disabled in config. Enabling...")
        # Note: We should ideally modify cfg here, but OmegaConf may be frozen

    # Create and run trainer
    trainer = SFTTrainer(cfg)
    trainer.setup()

    # Run training
    metrics = trainer.train()

    # Run final evaluation
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)

    log.info("=" * 60)
    log.info("SFT with QLoRA completed")
    log.info("=" * 60)

    return metrics


def run_sft_lora(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run SFT training with LoRA (no quantization).

    This function orchestrates the full SFT training pipeline with
    standard LoRA (no 4-bit quantization) for higher precision training.

    Args:
        cfg: Hydra configuration.

    Returns:
        Training metrics dictionary.
    """
    log.info("=" * 60)
    log.info("Starting SFT with LoRA (no quantization)")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model.path}")
    log.info(f"Encoder: {cfg.encoder.model_name}")
    log.info(f"Learning rate: {cfg.training.lr}")
    log.info(f"Batch size: {cfg.training.batch_size}")
    log.info(f"Epochs: {cfg.training.epochs}")

    # Ensure quantization is disabled
    # Note: The config should already have quantization.enabled = false

    # Create and run trainer
    trainer = SFTTrainer(cfg)
    trainer.setup()

    # Run training
    metrics = trainer.train()

    # Run final evaluation
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)

    log.info("=" * 60)
    log.info("SFT with LoRA completed")
    log.info("=" * 60)

    return metrics


def run_sft_with_trl(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run SFT training using TRL's SFTTrainer directly.

    This is an alternative implementation using TRL's native SFTTrainer
    for text-only training (without multimodal protein encoding).

    Args:
        cfg: Hydra configuration.

    Returns:
        Training metrics dictionary.
    """
    if not HAS_TRL:
        raise ImportError("TRL is required. Install with: pip install trl")

    log.info("Starting SFT with TRL's native SFTTrainer...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get quantization config
    use_quantization = cfg.training.get("quantization", {}).get("enabled", True)
    quantization_config = get_quantization_config(cfg) if use_quantization else None

    # Load model
    if quantization_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # Apply LoRA
    lora_config = get_qlora_config(cfg)
    model = get_peft_model(model, lora_config)

    # Load dataset
    from src.data.mol_instructions import MolInstructionsDataset

    train_dataset = MolInstructionsDataset(
        split="train",
        dataset_name=cfg.data.get("source", "zjunlp/Mol-Instructions"),
        subset=cfg.data.get("subset", "Protein-oriented Instructions"),
    )

    eval_dataset = MolInstructionsDataset(
        split="validation",
        dataset_name=cfg.data.get("source", "zjunlp/Mol-Instructions"),
        subset=cfg.data.get("subset", "Protein-oriented Instructions"),
    )

    # Create SFT config
    training_args = get_training_arguments(cfg)

    # Create TRL SFTTrainer
    trainer = TRLSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="formatted_prompt",
        max_seq_length=cfg.training.get("max_seq_length", 2048),
    )

    # Train
    train_result = trainer.train()

    # Save
    trainer.save_model(training_args.output_dir)

    return train_result.metrics
