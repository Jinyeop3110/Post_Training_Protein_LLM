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

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    from transformers.trainer_utils import EvalPrediction
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from trl import SFTConfig
    from trl import SFTTrainer as TRLSFTTrainer
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

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("epochs", 3),
        per_device_train_batch_size=training_cfg.get("batch_size", 8),
        per_device_eval_batch_size=training_cfg.get("eval_batch_size", 16),
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
        dataloader_persistent_workers=True,  # Reuse workers across epochs (skip fork overhead)
        dataloader_prefetch_factor=4,  # Prefetch 4 batches per worker (default=2)
        remove_unused_columns=False,  # Important for custom collator
        group_by_length=True,  # Groups similar-length sequences to reduce padding waste
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_drop_last=True,  # Prevent DDP hang on uneven last batch
        torch_compile=True,  # Compile LLM into fused CUDA kernels (~10-20% speedup)
        bf16_full_eval=True,  # Run eval in bf16 for speed
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

        Masks labels so the loss is only computed on response tokens
        (everything after "### Response:\\n"). Instruction and input
        tokens are set to -100 in labels.

        Args:
            batch: List of samples from MolInstructionsDataset.

        Returns:
            Dict containing tokenized inputs and protein sequences.
        """
        # Extract formatted prompts and protein sequences
        prompts = [item["formatted_prompt"] for item in batch]
        protein_sequences = [item["protein_sequence"] for item in batch]

        # Tokenize the inference prompt (instruction + input, no response)
        # to determine where the response starts
        inference_prompts = [item["inference_prompt"] for item in batch]

        # Tokenize full prompts
        encoded = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize inference prompts (without padding) to get prompt lengths
        prompt_lengths = []
        for inf_prompt in inference_prompts:
            inf_ids = self.tokenizer.encode(
                inf_prompt, add_special_tokens=False, truncation=True,
                max_length=self.max_length,
            )
            prompt_lengths.append(len(inf_ids))

        # Create labels: mask prompt tokens with -100, keep response tokens
        labels = encoded["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = self.label_pad_token_id
        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "protein_sequences": protein_sequences,
        }


class PackedDataset(torch.utils.data.Dataset):
    """
    Concatenation+packing dataset for efficient SFT training.

    Pre-tokenizes all examples, concatenates them with EOS separators,
    and chunks into fixed-length blocks. Eliminates padding waste for
    variable-length protein instruction data.

    Each packed block contains multiple concatenated examples:
        [tokens_1] [EOS] [tokens_2] [EOS] ... [tokens_k] [EOS] [PAD...]

    Labels mask the EOS separators between documents with -100 so the
    model only learns to predict within each document.

    For multimodal (ESM-3) training, protein_sequences are stored per-block
    as a list of sequences from the constituent examples.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the packed dataset.

        Args:
            dataset: Source dataset (e.g., MolInstructionsDataset).
            tokenizer: HuggingFace tokenizer.
            max_length: Fixed block length for packed sequences.
            shuffle: Whether to shuffle examples before packing.
            seed: Random seed for shuffling.
        """
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.blocks: List[Dict[str, Any]] = []
        self._pack(dataset, shuffle=shuffle, seed=seed)

    def _pack(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Tokenize all examples and pack into fixed-length blocks."""
        import random

        log.info(f"Packing {len(dataset)} examples into blocks of {self.max_length} tokens...")

        # 1. Pre-tokenize all examples
        tokenized_examples = []
        for i in range(len(dataset)):
            item = dataset[i]
            tokens = self.tokenizer.encode(
                item["formatted_prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 1,  # Reserve space for EOS
            )
            # Determine prompt length (instruction+input) for label masking
            prompt_tokens = self.tokenizer.encode(
                item["inference_prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 1,
            )
            prompt_len = len(prompt_tokens)
            # Append EOS as document boundary
            tokens.append(self.eos_token_id)
            tokenized_examples.append({
                "token_ids": tokens,
                "prompt_len": prompt_len,
                "protein_sequence": item.get("protein_sequence", ""),
            })

        # 2. Shuffle before concatenation (so adjacent docs are random)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(tokenized_examples)

        # 3. Concatenate into a single stream and chunk into blocks
        token_stream = []
        protein_stream = []  # Track which protein maps to which token position

        # Pack into blocks
        current_tokens = []
        current_labels = []
        current_proteins = []
        current_protein_set = set()  # Track unique proteins in current block

        for ex in tokenized_examples:
            token_ids = ex["token_ids"]
            prompt_len = ex["prompt_len"]
            protein_seq = ex["protein_sequence"]

            # If adding this example would exceed block size, finalize current block
            if len(current_tokens) + len(token_ids) > self.max_length:
                # If current block has content, pad and save it
                if current_tokens:
                    self._finalize_block(
                        current_tokens, current_labels,
                        list(current_proteins),
                    )

                # Start new block — if example itself is too long, it gets its own block
                current_tokens = []
                current_labels = []
                current_proteins = []
                current_protein_set = set()

            # Build labels: mask prompt tokens + boundary EOS, keep response tokens
            doc_labels = [-100] * prompt_len + list(token_ids[prompt_len:])
            # The last token (EOS) is a separator — mask it in labels
            doc_labels[-1] = -100

            current_tokens.extend(token_ids)
            current_labels.extend(doc_labels)

            # Track protein sequence (deduplicate within block)
            if protein_seq and protein_seq not in current_protein_set:
                current_proteins.append(protein_seq)
                current_protein_set.add(protein_seq)

        # Finalize last block
        if current_tokens:
            self._finalize_block(
                current_tokens, current_labels,
                list(current_proteins),
            )

        log.info(
            f"Packed into {len(self.blocks)} blocks "
            f"(from {len(tokenized_examples)} examples, "
            f"{len(self.blocks) * self.max_length:,} total tokens)"
        )

    def _finalize_block(
        self,
        tokens: List[int],
        labels: List[int],
        proteins: List[str],
    ) -> None:
        """Pad a block to max_length and add to self.blocks."""
        pad_len = self.max_length - len(tokens)

        input_ids = tokens + [self.pad_token_id] * pad_len
        attention_mask = [1] * len(tokens) + [0] * pad_len
        block_labels = labels + [-100] * pad_len

        self.blocks.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(block_labels, dtype=torch.long),
            "protein_sequences": proteins,
        })

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.blocks[idx]


class PackedDataCollator:
    """Simple collator for PackedDataset — just stacks pre-tokenized blocks."""

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "protein_sequences": [
                seq for b in batch for seq in b["protein_sequences"]
            ],
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


class GenerationSamplesCallback(TrainerCallback):
    """Callback to generate and log sample outputs during evaluation.

    Samples a few examples per category from the eval dataset, runs
    model.generate(), and logs results to console and wandb table.
    Only runs on rank 0.
    """

    # Keyword patterns to infer task category from instruction text
    CATEGORY_PATTERNS = [
        ("catalytic", ["catalytic activity", "catalytic", "enzyme commission", "ec number"]),
        ("domain", ["domain", "motif", "structural domain"]),
        ("design", ["design", "generate a protein", "create a protein"]),
        ("function", ["function", "functional description", "what does", "predict the function"]),
    ]

    def __init__(
        self,
        protein_llm: Optional[nn.Module],
        eval_dataset,
        tokenizer,
        num_samples_per_category: int = 2,
        max_new_tokens: int = 256,
    ):
        super().__init__()
        self.protein_llm = protein_llm
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples_per_category = num_samples_per_category
        self.max_new_tokens = max_new_tokens
        # Pre-select sample indices grouped by category
        self._sample_indices = self._select_samples()

    def _infer_category(self, instruction: str) -> str:
        """Infer task category from instruction text."""
        instruction_lower = instruction.lower()
        for category, keywords in self.CATEGORY_PATTERNS:
            if any(kw in instruction_lower for kw in keywords):
                return category
        return "general"

    def _select_samples(self) -> Dict[str, List[int]]:
        """Pre-select sample indices grouped by category."""
        category_indices: Dict[str, List[int]] = {}
        for i in range(len(self.eval_dataset)):
            item = self.eval_dataset[i]
            category = self._infer_category(item.get("instruction", ""))
            category_indices.setdefault(category, [])
            if len(category_indices[category]) < self.num_samples_per_category:
                category_indices[category].append(i)

            # Early exit if we have enough samples for all seen categories
            if all(
                len(v) >= self.num_samples_per_category
                for v in category_indices.values()
            ):
                # Check if we've seen at least 100 items (to discover categories)
                if i >= min(100, len(self.eval_dataset) - 1):
                    break

        total = sum(len(v) for v in category_indices.values())
        log.info(
            f"GenerationSamplesCallback: selected {total} samples "
            f"across {len(category_indices)} categories: "
            f"{', '.join(f'{k}({len(v)})' for k, v in category_indices.items())}"
        )
        return category_indices

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Generate sample outputs after each evaluation step."""
        # Only run on rank 0
        if int(os.environ.get("RANK", 0)) != 0:
            return

        log.info("=" * 60)
        log.info("Generation Samples (eval step %d)", state.global_step)
        log.info("=" * 60)

        table_rows = []

        for category, indices in sorted(self._sample_indices.items()):
            for idx in indices:
                item = self.eval_dataset[idx]
                protein_seq = item.get("protein_sequence", "")
                instruction = item.get("instruction", "")
                expected = item.get("response", "")
                inference_prompt = item.get("inference_prompt", "")

                # Generate
                try:
                    if self.protein_llm is not None:
                        # Multimodal path: use ProteinLLM.generate()
                        self.protein_llm.eval()
                        with torch.no_grad():
                            generated_texts = self.protein_llm.generate(
                                protein_sequences=[protein_seq],
                                prompt=[inference_prompt],
                                max_new_tokens=self.max_new_tokens,
                                min_new_tokens=10,
                                do_sample=False,
                                repetition_penalty=1.2,
                            )
                        generated = generated_texts[0] if generated_texts else ""
                        # Debug: if output is empty, decode without stripping special tokens
                        if not generated.strip():
                            raw_texts, raw_ids, inp_len = self.protein_llm.generate(
                                protein_sequences=[protein_seq],
                                prompt=[inference_prompt],
                                max_new_tokens=20,
                                min_new_tokens=10,
                                do_sample=False,
                                repetition_penalty=1.2,
                                return_token_ids=True,
                            )
                            raw_decode = self.tokenizer.decode(raw_ids[0], skip_special_tokens=False)
                            raw_id_list = raw_ids[0].tolist()[:20]
                            log.info(f"  [DEBUG] Empty output. Raw token IDs: {raw_id_list}")
                            log.info(f"  [DEBUG] Raw decode: {raw_decode[:200]}")
                    else:
                        # Text-only path: use LLM directly
                        model = kwargs.get("model")
                        if model is None:
                            continue
                        model.eval()
                        inputs = self.tokenizer(
                            inference_prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=2048,
                        ).to(model.device)
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs,
                                max_new_tokens=self.max_new_tokens,
                                min_new_tokens=10,
                                do_sample=False,
                                repetition_penalty=1.2,
                            )
                        # Slice off the input prompt tokens
                        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
                        generated = self.tokenizer.decode(
                            gen_ids, skip_special_tokens=True
                        )
                except Exception as e:
                    log.warning(f"Generation failed for sample {idx}: {e}")
                    generated = f"[ERROR: {e}]"

                # Truncate for console display (500 chars to see more context)
                gen_display = generated[:500] + ("..." if len(generated) > 500 else "")
                exp_display = expected[:500] + ("..." if len(expected) > 500 else "")
                seq_preview = protein_seq[:30] + "..." if len(protein_seq) > 30 else protein_seq

                # Console log
                log.info(f"[{category}] seq={seq_preview}")
                log.info(f"  Instruction: {instruction[:100]}...")
                log.info(f"  Expected:    {exp_display}")
                log.info(f"  Generated:   {gen_display}")
                log.info("")

                table_rows.append([
                    category,
                    instruction[:100],
                    exp_display,
                    gen_display,
                    seq_preview,
                ])

        # Restore train mode — HF Trainer only restores model.train() (the LLM),
        # NOT protein_llm. Without this, pooling/projector stay in eval mode
        # (no dropout), causing gradient distribution shift → NaN explosion.
        # ProteinLLM.train() already keeps the ESM-3 encoder frozen.
        if self.protein_llm is not None:
            self.protein_llm.train()

        # Log to wandb if available
        if HAS_WANDB and wandb.run is not None and table_rows:
            try:
                table = wandb.Table(
                    columns=[
                        "category", "instruction", "expected",
                        "generated", "protein_seq",
                    ],
                    data=table_rows,
                )
                wandb.log(
                    {f"generation_samples/step_{state.global_step}": table},
                    step=state.global_step,
                )
            except Exception as e:
                log.warning(f"Failed to log generation samples to wandb: {e}")

        log.info("=" * 60)


class ProteinLLMTrainer(Trainer):
    """
    Custom Trainer for ProteinLLM that handles the multimodal forward pass.

    This trainer overrides compute_loss to properly handle protein sequence
    encoding and the custom forward pass of ProteinLLM.
    """

    def __init__(
        self,
        protein_llm: Optional[nn.Module] = None,
        projector_lr: Optional[float] = None,
        max_tokens_per_batch: Optional[int] = None,
        max_batch_size: int = 16,
        freeze_lora_steps: int = 0,
        **kwargs,
    ):
        """
        Initialize the trainer.

        Args:
            protein_llm: The full ProteinLLM model (with encoder, pooling, projector).
            projector_lr: Learning rate for pooling+projector params. If None,
                defaults to 10x the base learning rate.
            max_tokens_per_batch: Token budget per micro-batch. When set,
                replaces fixed batch_size with dynamic batching. None = disabled.
            max_batch_size: Cap on samples per micro-batch when using token
                budget (prevents OOM on many short sequences). Default 16.
            freeze_lora_steps: Number of initial steps to freeze LoRA params,
                training only pooling+projector. 0 = disabled (default).
            **kwargs: Arguments passed to the base Trainer.
        """
        super().__init__(**kwargs)
        self.protein_llm = protein_llm
        self.projector_lr = projector_lr or (self.args.learning_rate * 10)
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_batch_size_cap = max_batch_size
        self.freeze_lora_steps = freeze_lora_steps
        self._lora_frozen = False

        # Token-level loss tracking for fair train/eval comparison.
        # HF Trainer reports "average of per-batch means" which is biased
        # when batch sizes vary (token-budget batching). We track
        # (sum_of_CE, num_tokens) to compute a true per-token average.
        self._token_loss_sum: float = 0.0
        self._token_count: int = 0

        # Freeze LoRA for initial alignment phase
        if self.freeze_lora_steps > 0 and self.protein_llm is not None:
            self._freeze_lora()
            log.info(
                f"Projector alignment phase: LoRA frozen for first "
                f"{self.freeze_lora_steps} steps"
            )

    def _freeze_lora(self):
        """Freeze all LoRA parameters."""
        count = 0
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False
                count += 1
        self._lora_frozen = True
        log.info(f"Froze {count} LoRA parameters")

    def _unfreeze_lora(self):
        """Unfreeze all LoRA parameters."""
        count = 0
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                count += 1
        self._lora_frozen = False
        log.info(f"Unfroze {count} LoRA parameters — LoRA training begins")

    def _get_train_sampler(self, train_dataset=None):
        """Override to pass pre-computed lengths for group_by_length.

        Our dataset exposes a ``lengths`` property (approx token counts)
        so HF's LengthGroupedSampler can group similar-length sequences
        and reduce padding waste.

        When ``max_tokens_per_batch`` is set, always returns a non-distributed
        ``LengthGroupedSampler``. DDP sharding is handled downstream by
        ``BatchSamplerShard`` wrapping ``TokenBudgetBatchSampler``.
        """
        if train_dataset is None:
            train_dataset = self.train_dataset

        if self.args.group_by_length and hasattr(train_dataset, "lengths"):
            from transformers.trainer_pt_utils import (
                DistributedLengthGroupedSampler,
                LengthGroupedSampler,
            )

            lengths = train_dataset.lengths

            # Token-budget path: always non-distributed (DDP via BatchSamplerShard)
            if self.max_tokens_per_batch is not None:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    lengths=lengths,
                )

            if self.args.parallel_mode.value == "distributed":
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    lengths=lengths,
                )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
            )

        return super()._get_train_sampler(train_dataset)

    def _get_eval_sampler(self, eval_dataset=None):
        """Override to pass pre-computed lengths for group_by_length on eval.

        Same logic as ``_get_train_sampler`` but for eval — prevents HF
        Trainer from failing when it can't auto-infer lengths from dataset
        items (our items have ``formatted_prompt``, not ``input_ids``).
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if self.args.group_by_length and hasattr(eval_dataset, "lengths"):
            from transformers.trainer_pt_utils import LengthGroupedSampler

            return LengthGroupedSampler(
                self.args.eval_batch_size,
                dataset=eval_dataset,
                lengths=eval_dataset.lengths,
            )

        return super()._get_eval_sampler(eval_dataset)

    def get_train_dataloader(self):
        """Override to use TokenBudgetBatchSampler when max_tokens_per_batch is set.

        Builds a DataLoader with a token-budget batch sampler instead of fixed
        batch_size. DDP distribution is handled by Accelerate's
        ``BatchSamplerShard`` via ``accelerator.prepare()``.

        Falls back to the default Trainer implementation when
        ``max_tokens_per_batch`` is None.
        """
        if self.max_tokens_per_batch is None:
            return super().get_train_dataloader()

        from torch.utils.data import DataLoader

        from src.training.token_budget_sampler import TokenBudgetBatchSampler

        train_dataset = self.train_dataset

        # Get lengths for token budgeting
        if not hasattr(train_dataset, "lengths"):
            log.warning(
                "Dataset has no 'lengths' attribute; "
                "falling back to fixed batch_size dataloader"
            )
            return super().get_train_dataloader()

        lengths = train_dataset.lengths

        # Get base sampler (non-distributed LengthGroupedSampler)
        base_sampler = self._get_train_sampler(train_dataset)

        # Number of DDP processes for batch-count alignment
        num_processes = self.args.world_size if self.args.world_size else 1

        batch_sampler = TokenBudgetBatchSampler(
            sampler=base_sampler,
            lengths=lengths,
            max_tokens=self.max_tokens_per_batch,
            max_batch_size=self.max_batch_size_cap,
            num_processes=num_processes,
        )

        # Build DataLoader with batch_sampler (mutually exclusive with
        # batch_size, sampler, shuffle, and drop_last)
        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_persistent_workers and self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = True
        if hasattr(self.args, "dataloader_prefetch_factor") and self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(train_dataset, **dataloader_params)

        # Token-budget batches have variable size; tell Accelerate not to
        # enforce even batch counts across DDP ranks (requires even_batches=False).
        self.accelerator.even_batches = False
        # Let Accelerate wrap with BatchSamplerShard for DDP distribution
        return self.accelerator.prepare(dataloader)

    def create_optimizer(self):
        """Create optimizer that includes multimodal parameters.

        The default Trainer optimizer only includes parameters from
        ``self.model`` (the LoRA-adapted LLM). For multimodal training,
        we also need to optimize the attention pooling and MLP projector
        weights from the ProteinLLM pipeline.

        Uses ``training.projector_lr`` if set (default: 10x base LR) to
        give randomly-initialized pooling+projector a higher learning rate
        than the pretrained LoRA adapters, following LLaVA-style training.
        """
        optimizer = super().create_optimizer()

        if self.protein_llm is not None:
            extra_params = []
            if self.protein_llm.pooling is not None:
                extra_params.extend(
                    p for p in self.protein_llm.pooling.parameters()
                    if p.requires_grad
                )
            if self.protein_llm.projector is not None:
                extra_params.extend(
                    p for p in self.protein_llm.projector.parameters()
                    if p.requires_grad
                )
            if extra_params:
                # Use dedicated projector_lr (default: 10x base LR)
                projector_lr = self.projector_lr
                optimizer.add_param_group({
                    "params": extra_params,
                    "lr": projector_lr,
                    "weight_decay": self.args.weight_decay,
                })
                num_extra = sum(p.numel() for p in extra_params)
                log.info(
                    f"Added {num_extra:,} multimodal parameters "
                    f"(pooling + projector) to optimizer "
                    f"with lr={projector_lr}"
                )

        return optimizer

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
            labels = inputs.get("labels")
        else:
            # Fallback to standard forward pass (text-only)
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)

            if labels is not None:
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

        # Track token-level stats: loss * num_tokens recovers the CE sum
        # since the model uses reduction='mean' over non-ignored tokens.
        if labels is not None and loss is not None:
            num_tokens = (labels != -100).sum().item()
            if num_tokens > 0:
                self._token_loss_sum += loss.detach().item() * num_tokens
                self._token_count += num_tokens

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Inject token-level average loss before logging.

        HF Trainer's default 'loss' is the mean of per-batch means —
        biased when batch sizes vary (token-budget batching). We add
        'token_avg_loss' = total_CE_sum / total_tokens for a true
        per-token average that is directly comparable between train and eval.
        """
        if self._token_count > 0:
            token_avg = self._token_loss_sum / self._token_count
            logs["token_avg_loss"] = round(token_avg, 4)
            logs["total_tokens"] = self._token_count
            # Reset for next logging interval
            self._token_loss_sum = 0.0
            self._token_count = 0
        super().log(logs, start_time)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Override to inject token-level eval loss.

        Resets token counters before eval, runs HF Trainer's evaluate,
        then adds ``eval_token_avg_loss`` — a true per-token average
        comparable to the training ``token_avg_loss``.
        """
        # Save training-phase counters, start fresh for eval
        train_sum, train_count = self._token_loss_sum, self._token_count
        self._token_loss_sum = 0.0
        self._token_count = 0

        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Inject token-level eval loss
        if self._token_count > 0:
            metrics[f"{metric_key_prefix}_token_avg_loss"] = round(
                self._token_loss_sum / self._token_count, 4
            )

        # Restore training-phase counters
        self._token_loss_sum = train_sum
        self._token_count = train_count

        return metrics

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to sync and clip pooling/projector gradients.

        HF Trainer only DDP-wraps and clips self.model (LLM). Pooling and
        projector in self.protein_llm need:
        1. Manual gradient sync across DDP ranks (~38M params, <1ms overhead)
        2. Explicit gradient clipping (HF Trainer's clip_grad_norm_ only covers
           model.parameters(), missing the multimodal params)
        3. NaN detection — skip corrupted gradients instead of poisoning weights
        """
        # Unfreeze LoRA after alignment phase
        if (
            self._lora_frozen
            and self.state.global_step >= self.freeze_lora_steps
        ):
            self._unfreeze_lora()

        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.protein_llm is not None:
            # Sync multimodal gradients across ranks
            if self.args.parallel_mode.value == "distributed":
                self._sync_multimodal_gradients()

            # Clip multimodal gradients (not covered by HF Trainer's clip)
            self._clip_multimodal_gradients()

        return loss

    def _sync_multimodal_gradients(self):
        """All-reduce pooling+projector gradients across DDP ranks."""
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        for module in [self.protein_llm.pooling, self.protein_llm.projector]:
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad and p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _clip_multimodal_gradients(self):
        """Clip pooling+projector gradients and zero NaN grads.

        HF Trainer's ``clip_grad_norm_`` only covers ``model.parameters()``
        (the LoRA LLM). Multimodal params (pooling + projector) live on
        ``self.protein_llm`` and receive 10x higher LR, making them
        especially vulnerable to gradient explosions from noisy batches.
        """
        mm_params = []
        for module in [self.protein_llm.pooling, self.protein_llm.projector]:
            if module is not None:
                mm_params.extend(
                    p for p in module.parameters()
                    if p.requires_grad and p.grad is not None
                )
        if not mm_params:
            return

        # Zero out NaN/Inf gradients to prevent weight corruption
        has_nan = False
        for p in mm_params:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                has_nan = True
                p.grad.zero_()
        if has_nan:
            log.warning(
                "NaN/Inf detected in multimodal gradients — zeroed to prevent "
                "weight corruption. Consider lowering projector_lr."
            )

        # Clip to same max_grad_norm as LLM params
        max_grad_norm = self.args.max_grad_norm
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(mm_params, max_grad_norm)


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
        """Set up wandb and other logging.

        Uses the training-specific wandb project (protein-llm-sft for SFT methods)
        and includes tags for method, model, dataset, lr, and epochs.
        Only runs on rank 0 in DDP to avoid duplicate logging.
        """
        if int(os.environ.get("RANK", 0)) != 0:
            return

        logging_cfg = self.cfg.get("logging", {})

        # Initialize wandb if enabled (skip if already initialized by scripts/train.py)
        if logging_cfg.get("wandb", {}).get("enabled", False) and HAS_WANDB:
            if wandb.run is not None:
                log.info("Wandb already initialized, skipping re-initialization")
                return

            # Get project from training config, fall back to logging config
            project = self.cfg.training.get("wandb", {}).get(
                "project",
                logging_cfg.wandb.get("project", "protein-llm-sft"),
            )

            # Build tags: method, model, dataset, lr, epochs
            tags = list(self.cfg.training.get("wandb", {}).get("tags", []))
            method = self.cfg.training.get("method", "sft")
            model_name = self.cfg.model.get("name", "unknown")
            tags.extend([
                f"method:{method}",
                f"model:{model_name}",
                f"lr:{self.cfg.training.get('lr', 'unknown')}",
                f"epochs:{self.cfg.training.get('epochs', 'unknown')}",
            ])

            wandb.init(
                project=project,
                name=logging_cfg.wandb.get("name", self.cfg.get("experiment_name", "sft")),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=tags,
            )
            log.info(f"Wandb logging initialized: project={project}, tags={tags}")

    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        from src.models.multimodal_llm import PROTEIN_SPECIAL_TOKENS

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

        # Add protein special tokens for ESM-3 approach:
        # <|protein_start|>, <|protein_embed|>, <|protein_end|>
        approach = self.cfg.get("approach", "text")
        if approach in ("esm3",):
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": PROTEIN_SPECIAL_TOKENS}
            )
            if num_added > 0:
                log.info(
                    f"Added {num_added} protein special tokens: {PROTEIN_SPECIAL_TOKENS} "
                    f"(vocab size: {len(self.tokenizer)})"
                )

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
        # For quantized models, use single-device placement (required by accelerate)
        if quantization_config is not None:
            device_map = {"": torch.cuda.current_device()}
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": torch.cuda.current_device()},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Resize embeddings if special tokens were added (e.g., <|protein_embed|>)
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            log.info(
                f"Resized model embeddings: "
                f"{self.model.config.vocab_size} -> {len(self.tokenizer)}"
            )

        # Apply LoRA
        lora_config = get_qlora_config(self.cfg)
        self.model = get_peft_model(self.model, lora_config)

        log.info("LoRA configuration applied")
        self.model.print_trainable_parameters()

        # Load full ProteinLLM for multimodal training when using the esm3
        # approach. Can also be forced with use_multimodal=True.
        approach = self.cfg.get("approach", "text")
        use_multimodal = self.cfg.get(
            "use_multimodal", approach == "esm3"
        )
        if use_multimodal:
            self._load_protein_llm()

        # Step-based LoRA freeze is handled in ProteinLLMTrainer via
        # freeze_lora_steps param (replaces old boolean freeze_lora)

    def _load_protein_llm(self) -> None:
        """Load the full ProteinLLM model for multimodal training.

        Creates ProteinLLM with encoder, pooling, and projector but
        reuses the already-loaded LoRA model instead of loading a second
        copy of the LLM.
        """
        try:
            from src.models.multimodal_llm import EMBEDDING_APPROACHES, ProteinLLM

            approach = self.cfg.get("approach", "esm3")
            if approach not in EMBEDDING_APPROACHES:
                log.info(
                    f"Approach '{approach}' doesn't need multimodal "
                    f"components, skipping ProteinLLM setup"
                )
                return

            log.info("Loading ProteinLLM for multimodal training...")

            # Extract config sections
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            encoder_cfg = cfg_dict.get("encoder", {})
            model_cfg = cfg_dict.get("model", {})
            training_cfg = cfg_dict.get("training", {})
            pooling_cfg = encoder_cfg.get("pooling", {})
            projector_cfg = encoder_cfg.get("projector", {})
            lora_cfg = training_cfg.get("lora", {})

            # Determine if quantization is used (for correct save/load config)
            use_qlora = training_cfg.get("quantization", {}).get("enabled", False)

            # DDP: ensure encoder/pooling/projector go to the correct GPU per rank
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

            # Create ProteinLLM WITHOUT loading LLM (we already have it)
            self.protein_llm = ProteinLLM(
                approach=approach,
                llm_name=model_cfg.get("path", "Qwen/Qwen3-4B"),
                encoder_name=encoder_cfg.get("model_name", "esm3-sm-open-v1"),
                encoder_embed_dim=encoder_cfg.get("embedding_dim"),
                num_prefix_tokens=pooling_cfg.get("num_output_tokens", 32),
                pooling_type=pooling_cfg.get("method", "attention"),
                projector_type=projector_cfg.get("type", "mlp"),
                projector_hidden_dim=projector_cfg.get("hidden_dim", 2048),
                projector_num_layers=projector_cfg.get("num_layers", 2),
                projector_dropout=projector_cfg.get("dropout", 0.1),
                perceiver_layers=projector_cfg.get("perceiver_layers", 2),
                perceiver_heads=projector_cfg.get("perceiver_heads", 8),
                perceiver_ffn_dim=projector_cfg.get("perceiver_ffn_dim", 2048),
                perceiver_latent_dim=projector_cfg.get("perceiver_latent_dim", None),
                use_qlora=use_qlora,
                lora_r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                lora_target_modules=lora_cfg.get(
                    "target_modules", [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ]
                ),
                load_llm=False,  # Don't load LLM again
                load_encoder=True,
                device=device,  # DDP: correct GPU per rank
                encoder_dtype=encoder_cfg.get("dtype", "bfloat16"),
                encoder_batch_size=encoder_cfg.get("encoder_batch_size", 4),
            )

            # Assign our already-loaded LoRA model and tokenizer
            self.protein_llm.llm = self.model
            self.protein_llm.tokenizer = self.tokenizer

            # Set hidden size from the loaded model's config
            self.protein_llm.llm_hidden_size = self.model.config.hidden_size

            # Build projector now that we know the LLM hidden size
            self.protein_llm._build_projector()

            log.info("ProteinLLM loaded successfully")
            self.protein_llm.print_trainable_parameters()

        except ImportError as e:
            log.warning(f"Could not load ProteinLLM: {e}. Using text-only training.")
            self.protein_llm = None

    def _load_datasets(self) -> None:
        """Load training and validation datasets.

        When ``training.packing_sequences=True``, wraps the training set in a
        ``PackedDataset`` that concatenates tokenized examples into
        fixed-length blocks (eliminates padding waste).  Validation
        is kept unpacked for clean per-example eval loss.
        """
        from src.data.mol_instructions import MolInstructionsDataset
        from src.models.multimodal_llm import PROTEIN_PLACEHOLDER

        data_cfg = self.cfg.data
        limit = data_cfg.get("limit", None)

        sampling_temp = data_cfg.get("sampling_temperature", 1.0)
        exclude_files = list(data_cfg.get("exclude_files", []) or [])

        # For ESM-3 approach, replace protein text with placeholder token
        approach = self.cfg.get("approach", "text")
        placeholder = PROTEIN_PLACEHOLDER if approach in ("esm3",) else ""

        max_protein_length = data_cfg.get("processing", {}).get("max_protein_length", None)

        log.info("Loading training dataset...")
        raw_train = MolInstructionsDataset(
            split="train",
            dataset_name=data_cfg.get("source", "zjunlp/Mol-Instructions"),
            subset=data_cfg.get("subset", "Protein-oriented Instructions"),
            cache_dir=data_cfg.get("paths", {}).get("raw"),
            max_seq_length=self.cfg.training.get("max_seq_length", 2048),
            max_protein_length=max_protein_length,
            limit=limit,
            sampling_temperature=sampling_temp,
            exclude_files=exclude_files,
            protein_placeholder=placeholder,
            tokenizer=self.tokenizer,
        )
        log.info(f"Training dataset loaded: {len(raw_train)} samples")

        # Optionally pack training set
        self.use_packing = self.cfg.training.get("packing_sequences", False)
        if self.use_packing:
            max_length = self.cfg.training.get("max_seq_length", 2048)
            self.train_dataset = PackedDataset(
                dataset=raw_train,
                tokenizer=self.tokenizer,
                max_length=max_length,
                shuffle=True,
                seed=data_cfg.get("seed", 42),
            )
            log.info(
                f"Packing enabled: {len(raw_train)} examples -> "
                f"{len(self.train_dataset)} packed blocks"
            )
        else:
            self.train_dataset = raw_train

        # Use a proportionally smaller validation set when limit is set
        val_limit = max(1, limit // 10) if limit else None
        log.info("Loading validation dataset...")
        self.eval_dataset = MolInstructionsDataset(
            split="validation",
            dataset_name=data_cfg.get("source", "zjunlp/Mol-Instructions"),
            subset=data_cfg.get("subset", "Protein-oriented Instructions"),
            cache_dir=data_cfg.get("paths", {}).get("raw"),
            max_seq_length=self.cfg.training.get("max_seq_length", 2048),
            max_protein_length=max_protein_length,
            limit=val_limit,
            sampling_temperature=sampling_temp,
            exclude_files=exclude_files,
            protein_placeholder=placeholder,
            tokenizer=self.tokenizer,
        )
        log.info(f"Validation dataset loaded: {len(self.eval_dataset)} samples")

    def _create_collator(self) -> None:
        """Create the data collator.

        Uses ``PackedDataCollator`` when packing_sequences is enabled (blocks
        are already tokenized), otherwise the standard ``ProteinLLMDataCollator``.
        """
        if self.use_packing:
            self.data_collator = PackedDataCollator()
        else:
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

        # Generation samples callback (logs model outputs during eval)
        gen_callback = GenerationSamplesCallback(
            protein_llm=self.protein_llm,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            num_samples_per_category=5,
            max_new_tokens=self.cfg.get("evaluation", {}).get(
                "sft_gen_max_tokens", 256
            ),
        )
        callbacks.append(gen_callback)

        # Always use ProteinLLMTrainer — handles both multimodal (protein_llm set)
        # and text-only (protein_llm=None) modes. Also provides _get_train_sampler
        # override for group_by_length with our dataset's pre-computed lengths.
        projector_lr = self.cfg.training.get("projector_lr", None)

        # Token-budget dynamic batching
        max_tokens_per_batch = self.cfg.training.get("max_tokens_per_batch", None)
        max_batch_size = self.cfg.training.get("max_batch_size", 16)

        # Packing conflict: token-budget is useless with fixed-length packed blocks
        if max_tokens_per_batch is not None and self.use_packing:
            log.warning(
                "max_tokens_per_batch is incompatible with packing_sequences=True "
                "(all blocks are fixed-length). Disabling token-budget batching."
            )
            max_tokens_per_batch = None

        freeze_lora_steps = self.cfg.training.get("freeze_lora_steps", 0)

        self.trainer = ProteinLLMTrainer(
            protein_llm=self.protein_llm,
            projector_lr=projector_lr,
            max_tokens_per_batch=max_tokens_per_batch,
            max_batch_size=max_batch_size,
            freeze_lora_steps=freeze_lora_steps,
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            callbacks=callbacks,
        )

        if max_tokens_per_batch is not None:
            log.info(
                f"Token-budget batching enabled: "
                f"max_tokens={max_tokens_per_batch}, max_batch_size={max_batch_size}"
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

        # Save final model with metrics using naming convention
        self.save_checkpoint(metrics=metrics)

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

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save model checkpoint to experiment_dir/checkpoints/.

        Saves all artifacts under the unified experiment directory:
        - checkpoints/protein_llm/: ProteinLLM checkpoint (or adapter at root)
        - checkpoints/tokenizer/: Tokenizer files
        - training_args.json: All hyperparameters (at experiment root)
        - metrics.json: Final train/val loss and eval metrics (at experiment root)

        Only saves on rank 0 in DDP to avoid duplicate checkpoints.

        Args:
            path: Checkpoint directory path. If None, uses experiment_dir/checkpoints/.
            metrics: Training/eval metrics to save. If None, empty dict is saved.

        Returns:
            Path to the saved checkpoint directory, or None on non-rank-0 processes.
        """
        if int(os.environ.get("RANK", 0)) != 0:
            return None

        if path is None:
            path = Path(
                self.cfg.get("paths", {}).get("checkpoint_dir", "./checkpoints")
            )
        else:
            path = Path(path)

        log.info(f"Saving checkpoint to: {path}")
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter at root ONLY when there is no ProteinLLM
        # (ProteinLLM.save_pretrained saves adapter inside protein_llm/adapter/)
        if self.protein_llm is None:
            self.model.save_pretrained(path)

        # Save tokenizer
        self.tokenizer.save_pretrained(path / "tokenizer")

        # Save training_args.json at experiment root level
        experiment_dir = Path(
            self.cfg.get("paths", {}).get("experiment_dir", path.parent)
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        training_args = {
            "method": self.cfg.training.get("method", "sft"),
            "approach": self.cfg.get("approach", "text"),
            "projector_lr": self.cfg.training.get("projector_lr", None),
            "model": self.cfg.model.get("name", "unknown"),
            "model_path": self.cfg.model.get("path", "unknown"),
            "dataset": self.cfg.data.get("name", "unknown"),
            "lr": self.cfg.training.get("lr", None),
            "epochs": self.cfg.training.get("epochs", None),
            "batch_size": self.cfg.training.get("batch_size", None),
            "gradient_accumulation_steps": self.cfg.training.get(
                "gradient_accumulation_steps", None
            ),
            "max_seq_length": self.cfg.training.get("max_seq_length", None),
            "warmup_ratio": self.cfg.training.get("warmup_ratio", None),
            "max_grad_norm": self.cfg.training.get("max_grad_norm", None),
            "weight_decay": self.cfg.training.get("weight_decay", None),
            "lora": {
                "r": self.cfg.training.get("lora", {}).get("r", None),
                "alpha": self.cfg.training.get("lora", {}).get("alpha", None),
                "dropout": self.cfg.training.get("lora", {}).get("dropout", None),
                "target_modules": list(
                    self.cfg.training.get("lora", {}).get(
                        "target_modules", [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ]
                    )
                ),
            },
            "quantization": {
                "enabled": self.cfg.training.get("quantization", {}).get(
                    "enabled", False
                ),
                "bits": self.cfg.training.get("quantization", {}).get("bits", None),
            },
            "optimizer": OmegaConf.to_container(
                self.cfg.training.get("optimizer", {}), resolve=True
            ),
            "timestamp": datetime.now().isoformat(),
        }
        with open(experiment_dir / "training_args.json", "w") as f:
            json.dump(training_args, f, indent=2, default=str)

        # Save metrics.json at experiment root level
        metrics_to_save = metrics if metrics is not None else {}
        with open(experiment_dir / "metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2, default=str)

        # Save trainer state
        if self.trainer is not None:
            self.trainer.save_state()

        # Save ProteinLLM components if available
        if self.protein_llm is not None:
            self.protein_llm.save_pretrained(path / "protein_llm")

        log.info(f"Checkpoint saved to {path}")
        return path


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

    # Load model (single-device placement for DDP compatibility)
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "auto"
    if quantization_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path,
            device_map=device_map,
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
        tokenizer=tokenizer,
    )

    eval_dataset = MolInstructionsDataset(
        split="validation",
        dataset_name=cfg.data.get("source", "zjunlp/Mol-Instructions"),
        subset=cfg.data.get("subset", "Protein-oriented Instructions"),
        tokenizer=tokenizer,
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
