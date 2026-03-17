"""
SSL Trainer — Continued pre-training on biology literature.

A simplified trainer for causal language modelling with LoRA on a BASE model.
No ProteinLLM, no encoder, no chat template — loss on ALL tokens.

Pipeline: SSL (BASE + LoRA) → merge → SFT (Instruct + LoRA) → GRPO

Main components:
- SSLTrainer: Setup and orchestration
- run_ssl: Entry point called from scripts/train.py
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


log = logging.getLogger(__name__)


class SSLTrainer:
    """SSL trainer for continued pre-training with LoRA.

    Simplified variant of SFTTrainer:
    - No ProteinLLM / encoder / projector
    - No special tokens (no <|protein_embed|>)
    - No chat template — plain causal LM
    - Uses SSLDataset + SSLDataCollator
    - Supports packing via SSLPackedDataset
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.data_collator = None
        self.use_packing = False

        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers required: pip install transformers")
        if not HAS_PEFT:
            raise ImportError("PEFT required: pip install peft")

    def setup(self) -> None:
        """Set up model, tokenizer, datasets, and trainer."""
        log.info("Setting up SSL trainer...")
        self._load_tokenizer()
        self._load_model()
        self._load_datasets()
        self._create_collator()
        self._create_trainer()
        log.info("SSL trainer setup complete")

    def _load_tokenizer(self) -> None:
        """Load tokenizer from BASE model (no special tokens added)."""
        model_path = self.cfg.model.path
        log.info(f"Loading tokenizer from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",  # Causal LM: right-padding
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

    def _load_model(self) -> None:
        """Load BASE model with LoRA."""
        model_path = self.cfg.model.path
        use_fsdp = self.cfg.training.get("fsdp", {}).get("enabled", False)

        log.info(f"Loading BASE model from: {model_path}")

        if use_fsdp:
            # FSDP: load on CPU, FSDP handles device placement
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            log.info("FSDP: loaded model on CPU (FSDP will shard to GPUs)")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": torch.cuda.current_device()},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Apply LoRA
        from .config_utils import get_qlora_config

        lora_config = get_qlora_config(self.cfg)
        self.model = get_peft_model(self.model, lora_config)
        log.info("LoRA applied to BASE model")
        self.model.print_trainable_parameters()

    def _load_datasets(self) -> None:
        """Load SSL train and validation datasets."""
        from src.data.ssl_dataset import SSLDataset

        data_cfg = self.cfg.data
        limit = data_cfg.get("limit", None)

        log.info("Loading SSL training dataset...")
        self.train_dataset = SSLDataset(
            split="train",
            cache_dir=data_cfg.get("paths", {}).get("raw"),
            max_seq_length=self.cfg.training.get("max_seq_length", 4096),
            limit=limit,
            sampling_temperature=data_cfg.get("sampling_temperature", 1.0),
            exclude_files=list(data_cfg.get("exclude_files", []) or []),
            seed=data_cfg.get("splits", {}).get("seed", 42),
        )
        log.info(f"SSL training dataset: {len(self.train_dataset)} samples")

        # Packing: pre-tokenize and concatenate into fixed-length blocks
        self.use_packing = self.cfg.training.get("packing_sequences", False)
        if self.use_packing:
            max_length = self.cfg.training.get("max_seq_length", 4096)
            self.train_dataset = SSLPackedDataset(
                dataset=self.train_dataset,
                tokenizer=self.tokenizer,
                max_length=max_length,
                shuffle=True,
                seed=data_cfg.get("splits", {}).get("seed", 42),
            )
            log.info(
                f"Packing enabled: {len(self.train_dataset)} packed blocks"
            )

        # Validation
        val_limit = max(1, limit // 10) if limit else None
        log.info("Loading SSL validation dataset...")
        self.eval_dataset = SSLDataset(
            split="validation",
            cache_dir=data_cfg.get("paths", {}).get("raw"),
            max_seq_length=self.cfg.training.get("max_seq_length", 4096),
            limit=val_limit,
            sampling_temperature=data_cfg.get("sampling_temperature", 1.0),
            exclude_files=list(data_cfg.get("exclude_files", []) or []),
            seed=data_cfg.get("splits", {}).get("seed", 42),
        )
        log.info(f"SSL validation dataset: {len(self.eval_dataset)} samples")

        # Cap eval dataset
        max_eval = self.cfg.training.get("max_eval_samples", None)
        if max_eval and len(self.eval_dataset) > max_eval:
            import random

            rng = random.Random(42)
            indices = sorted(rng.sample(range(len(self.eval_dataset)), max_eval))
            self.eval_dataset.data = self.eval_dataset.data.select(
                indices
            ).flatten_indices()
            if hasattr(self.eval_dataset, "_lengths"):
                del self.eval_dataset._lengths
            log.info(f"Eval capped to {len(self.eval_dataset)} samples")

    def _create_collator(self) -> None:
        """Create the data collator."""
        if self.use_packing:
            from .collators import PackedDataCollator

            self.data_collator = PackedDataCollator()
        else:
            from src.data.ssl_dataset import SSLDataCollator

            self.data_collator = SSLDataCollator(
                tokenizer=self.tokenizer,
                max_length=self.cfg.training.get("max_seq_length", 4096),
                padding="longest",
            )

    def _create_trainer(self) -> None:
        """Create HuggingFace Trainer."""
        from .callbacks import GPUMemoryCallback
        from .config_utils import get_training_arguments

        training_args = get_training_arguments(self.cfg)
        callbacks = [GPUMemoryCallback()]

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
        """Run training loop with auto-resume."""
        if self.trainer is None:
            raise RuntimeError("Call setup() first")

        log.info("Starting SSL training...")
        log.info(f"  Epochs: {self.cfg.training.epochs}")
        log.info(f"  Batch size: {self.cfg.training.batch_size}")
        log.info(f"  Learning rate: {self.cfg.training.lr}")
        log.info(
            f"  Gradient accumulation: {self.cfg.training.gradient_accumulation_steps}"
        )

        # Auto-resume from latest checkpoint
        resume_from = None
        ckpt_dir = self.cfg.paths.checkpoint_dir
        if os.path.isdir(ckpt_dir):
            ckpts = sorted(
                [
                    d
                    for d in Path(ckpt_dir).iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")
                ],
                key=lambda x: int(x.name.rsplit("-", 1)[-1]),
            )
            if ckpts:
                resume_from = str(ckpts[-1])
                log.info(f"Auto-resuming from: {resume_from}")

        train_result = self.trainer.train(resume_from_checkpoint=resume_from)
        metrics = train_result.metrics
        log.info(f"Training completed. Final loss: {metrics.get('train_loss', 'N/A')}")

        self.save_checkpoint(metrics=metrics)
        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Run validation."""
        if self.trainer is None:
            raise RuntimeError("Call setup() first")

        log.info("Running evaluation...")
        metrics = self.trainer.evaluate()
        log.info(f"Eval metrics: {metrics}")
        return metrics

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save LoRA adapter and metadata."""
        if int(os.environ.get("RANK", 0)) != 0:
            return None

        if path is None:
            path = Path(self.cfg.get("paths", {}).get("checkpoint_dir", "./checkpoints"))
        else:
            path = Path(path)

        log.info(f"Saving SSL checkpoint to: {path}")
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.model.save_pretrained(path / "lora_adapter")

        # Save tokenizer
        self.tokenizer.save_pretrained(path / "tokenizer")

        # Save training_args.json at experiment root
        experiment_dir = Path(
            self.cfg.get("paths", {}).get("experiment_dir", path.parent)
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        training_args = {
            "method": "ssl_lora",
            "approach": "text",
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
            "lora": {
                "r": self.cfg.training.get("lora", {}).get("r", None),
                "alpha": self.cfg.training.get("lora", {}).get("alpha", None),
                "target_modules": list(
                    self.cfg.training.get("lora", {}).get("target_modules", [])
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }
        with open(experiment_dir / "training_args.json", "w") as f:
            json.dump(training_args, f, indent=2, default=str)

        # Save metrics.json
        metrics_to_save = metrics if metrics is not None else {}
        with open(experiment_dir / "metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2, default=str)

        # Save trainer state
        if self.trainer is not None:
            self.trainer.save_state()

        log.info(f"SSL checkpoint saved to {path}")
        return path


class SSLPackedDataset(torch.utils.data.Dataset):
    """Concatenation+packing for SSL causal LM training.

    Simpler than the SFT PackedDataset: no prompt/response masking,
    loss on ALL tokens (only EOS separators between documents are masked).
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        tokenizer,
        max_length: int = 4096,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.blocks = []
        self._pack(dataset, shuffle=shuffle, seed=seed)

    def _pack(self, dataset, shuffle: bool = True, seed: int = 42) -> None:
        """Tokenize all documents and pack into fixed-length blocks."""
        import random

        log.info(
            f"Packing {len(dataset)} SSL documents into {self.max_length}-token blocks..."
        )

        # Pre-tokenize
        tokenized = []
        for i in range(len(dataset)):
            item = dataset[i]
            tokens = self.tokenizer.encode(
                item["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length - 1,  # Reserve for EOS
            )
            tokens.append(self.eos_token_id)
            tokenized.append(tokens)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(tokenized)

        # Concatenate and chunk
        current_tokens = []
        current_labels = []

        for token_ids in tokenized:
            if len(current_tokens) + len(token_ids) > self.max_length:
                if current_tokens:
                    self._finalize_block(current_tokens, current_labels)
                current_tokens = []
                current_labels = []

            # Labels = all tokens, but mask the boundary EOS
            doc_labels = list(token_ids)
            doc_labels[-1] = -100  # Mask EOS separator

            current_tokens.extend(token_ids)
            current_labels.extend(doc_labels)

        if current_tokens:
            self._finalize_block(current_tokens, current_labels)

        log.info(
            f"Packed into {len(self.blocks)} blocks "
            f"({len(self.blocks) * self.max_length:,} total tokens)"
        )

    def _finalize_block(self, tokens, labels) -> None:
        """Pad block to max_length and store."""
        pad_len = self.max_length - len(tokens)
        self.blocks.append({
            "input_ids": torch.tensor(
                tokens + [self.pad_token_id] * pad_len, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [1] * len(tokens) + [0] * pad_len, dtype=torch.long
            ),
            "labels": torch.tensor(
                labels + [-100] * pad_len, dtype=torch.long
            ),
            # Empty protein_sequences for compatibility with PackedDataCollator
            "protein_sequences": [],
        })

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.blocks[idx]


# =========================================================================
# Entry point
# =========================================================================


def run_ssl(cfg: DictConfig) -> Dict[str, Any]:
    """Run SSL continued pre-training.

    Args:
        cfg: Hydra configuration.

    Returns:
        Training metrics dictionary.
    """
    log.info("=" * 60)
    log.info("Starting SSL continued pre-training")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model.path}")
    log.info(f"Learning rate: {cfg.training.lr}")
    log.info(f"Batch size: {cfg.training.batch_size}")
    log.info(f"Epochs: {cfg.training.epochs}")

    trainer = SSLTrainer(cfg)
    trainer.setup()

    metrics = trainer.train()

    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)

    log.info("=" * 60)
    log.info("SSL continued pre-training completed")
    log.info("=" * 60)

    return metrics
