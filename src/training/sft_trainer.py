"""
SFT Trainer Implementation

This module provides Supervised Fine-Tuning (SFT) functionality using
QLoRA/LoRA for efficient training of the multimodal ProteinLLM.

Main components:
- SFTTrainer: High-level trainer class that orchestrates setup and training
- ProteinLLMTrainer: Custom HF Trainer for multimodal forward pass
- run_sft: Unified entry point for SFT training
- run_sft_qlora / run_sft_lora: Backward-compatible aliases for run_sft
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizer,
        Trainer,
        TrainingArguments,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports for backward compatibility.
# External code (grpo_trainer, scripts/train.py, tests) can still do:
#   from src.training.sft_trainer import get_qlora_config, ...
# ---------------------------------------------------------------------------
from .callbacks import GenerationSamplesCallback, GPUMemoryCallback  # noqa: F401, E402
from .collators import PackedDataCollator, PackedDataset, ProteinLLMDataCollator  # noqa: F401, E402
from .config_utils import (  # noqa: F401, E402
    get_qlora_config,
    get_quantization_config,
    get_training_arguments,
)

# =========================================================================
# ProteinLLMTrainer — Custom HF Trainer
# =========================================================================


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

        self._has_mm_param_group = False
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
            # Gated XATTN params (flamingo approach — inside LLM layers)
            if self.protein_llm.gated_xattn_blocks is not None:
                for block in self.protein_llm.gated_xattn_blocks:
                    extra_params.extend(
                        p for p in block.parameters()
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
                self._has_mm_param_group = True
                num_extra = sum(p.numel() for p in extra_params)
                log.info(
                    f"Added {num_extra:,} multimodal parameters "
                    f"(pooling + projector) to optimizer "
                    f"with lr={projector_lr}"
                )

        return optimizer

    def _save_optimizer_and_scheduler(self, output_dir):
        """Override: save FSDP optimizer WITHOUT save_fsdp_model.

        The default super() call triggers both save_fsdp_model (14 GB full
        state dict) and save_fsdp_optimizer. For LoRA training the full model
        is redundant — adapter weights are already saved by save_model().
        We call save_fsdp_optimizer directly, skipping the 14 GB file.

        Multimodal params (pooling + projector) are NOT FSDP-wrapped, so we
        pop them before the FSDP optimizer save, then restore and save
        separately.
        """
        if not (self.is_fsdp_enabled and self._has_mm_param_group):
            return super()._save_optimizer_and_scheduler(output_dir)

        from accelerate.utils import save_fsdp_optimizer

        # Pop the multimodal param group (always the last one, added by create_optimizer)
        mm_group = self.optimizer.param_groups.pop()
        mm_state = {}
        for p in mm_group["params"]:
            if p in self.optimizer.state:
                mm_state[id(p)] = self.optimizer.state.pop(p)

        try:
            # Save FSDP optimizer ONLY (skip save_fsdp_model — no 14 GB file)
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.optimizer,
                self.model,
                output_dir,
            )
            # Save scheduler
            torch.save(
                self.lr_scheduler.state_dict(),
                os.path.join(output_dir, "scheduler.pt"),
            )
        finally:
            # Restore multimodal param group and state
            self.optimizer.param_groups.append(mm_group)
            for p in mm_group["params"]:
                state = mm_state.get(id(p))
                if state is not None:
                    self.optimizer.state[p] = state

        # Save multimodal optimizer state separately (rank 0 only)
        if output_dir and int(os.environ.get("RANK", 0)) == 0:
            mm_save = {
                "config": {k: v for k, v in mm_group.items() if k != "params"},
                "states": list(mm_state.values()),
            }
            save_path = os.path.join(output_dir, "mm_optimizer.pt")
            torch.save(mm_save, save_path)
            log.info(f"Saved multimodal optimizer state to {save_path}")

    def _save_checkpoint(self, model, trial):
        """Override: also save multimodal weights in intermediate checkpoints.

        The base Trainer saves adapter weights via save_model(). We add
        pooling.pt and projector.pt so intermediate checkpoints are fully
        resumable without the 14 GB pytorch_model_fsdp.bin.
        """
        super()._save_checkpoint(model, trial)

        if self.protein_llm is not None and self.args.should_save:
            ckpt_dir = os.path.join(
                self.args.output_dir,
                f"checkpoint-{self.state.global_step}",
            )
            if self.protein_llm.pooling is not None:
                torch.save(
                    self.protein_llm.pooling.state_dict(),
                    os.path.join(ckpt_dir, "pooling.pt"),
                )
            if self.protein_llm.projector is not None:
                torch.save(
                    self.protein_llm.projector.state_dict(),
                    os.path.join(ckpt_dir, "projector.pt"),
                )
            if self.protein_llm.gated_xattn_blocks is not None:
                torch.save(
                    self.protein_llm.gated_xattn_blocks.state_dict(),
                    os.path.join(ckpt_dir, "xattn.pt"),
                )
            log.info(f"Saved multimodal weights to {ckpt_dir}")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Override: load minimal checkpoint (no pytorch_model_fsdp.bin).

        Old-format checkpoints with pytorch_model_fsdp.bin fall through to
        super(). New minimal checkpoints load adapter weights via FSDP
        full-state-dict API and multimodal weights directly.
        """
        if model is None:
            model = self.model

        fsdp_path = os.path.join(resume_from_checkpoint, "pytorch_model_fsdp.bin")
        if os.path.exists(fsdp_path) or not self.is_fsdp_enabled:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

        # New minimal checkpoint: load adapter via PeftModel.load_adapter
        # under FSDP.summon_full_params (handles key mapping natively).
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )

        adapter_cfg = os.path.join(
            resume_from_checkpoint, "adapter_config.json"
        )
        if not os.path.exists(adapter_cfg):
            log.warning(
                f"No adapter_config.json in {resume_from_checkpoint}, "
                "skipping adapter load"
            )
            return

        # Gather all params, load adapter, then re-shard
        with FSDP.summon_full_params(model, writeback=True):
            # Unwrap FSDP → OptimizedModule → PeftModel
            inner = model.module
            while hasattr(inner, "_orig_mod"):
                inner = inner._orig_mod

            if hasattr(inner, "load_adapter"):
                active = (
                    inner.active_adapters[0]
                    if hasattr(inner, "active_adapters")
                    else inner.active_adapter
                )
                inner.load_adapter(
                    resume_from_checkpoint, active, is_trainable=True
                )
                log.info(
                    f"Loaded adapter '{active}' from {resume_from_checkpoint} "
                    f"via PeftModel.load_adapter"
                )
            else:
                log.warning(
                    "Model does not have load_adapter method, "
                    "skipping adapter load"
                )

        # Load multimodal weights (not FSDP-wrapped)
        if self.protein_llm is not None:
            for name, module in [
                ("pooling", self.protein_llm.pooling),
                ("projector", self.protein_llm.projector),
                ("xattn", self.protein_llm.gated_xattn_blocks),
            ]:
                path = os.path.join(resume_from_checkpoint, f"{name}.pt")
                if os.path.exists(path) and module is not None:
                    module.load_state_dict(
                        torch.load(path, map_location="cpu", weights_only=True)
                    )
                    log.info(f"Loaded {name} from {path}")

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Override: restore multimodal optimizer state on resume.

        Mirror of _save_optimizer_and_scheduler: pop multimodal group before
        loading FSDP optimizer state, then restore and load mm state.
        """
        if not (self.is_fsdp_enabled and self._has_mm_param_group):
            return super()._load_optimizer_and_scheduler(checkpoint)

        # Pop multimodal group (not in FSDP optimizer state)
        mm_group = self.optimizer.param_groups.pop()
        mm_state = {}
        for p in mm_group["params"]:
            if p in self.optimizer.state:
                mm_state[id(p)] = self.optimizer.state.pop(p)

        try:
            super()._load_optimizer_and_scheduler(checkpoint)
        finally:
            self.optimizer.param_groups.append(mm_group)
            for p in mm_group["params"]:
                state = mm_state.get(id(p))
                if state is not None:
                    self.optimizer.state[p] = state

        # Load multimodal optimizer state
        if checkpoint:
            mm_path = os.path.join(checkpoint, "mm_optimizer.pt")
            if os.path.exists(mm_path):
                mm_data = torch.load(
                    mm_path, map_location="cpu", weights_only=True
                )
                for p, state_data in zip(
                    mm_group["params"], mm_data["states"]
                ):
                    self.optimizer.state[p] = {
                        k: v.to(p.device) if isinstance(v, torch.Tensor) else v
                        for k, v in state_data.items()
                    }
                log.info(f"Loaded multimodal optimizer state from {mm_path}")

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
            # FSDP: protein_llm.llm must point to the FSDP-wrapped model
            # so that self.llm(...) triggers parameter all-gather.
            # DDP: also safe — DDP forward registers gradient hooks.
            if self.protein_llm.llm is not model:
                self.protein_llm.llm = model
                log.info(
                    f"Synced protein_llm.llm to {type(model).__name__}"
                )

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


# =========================================================================
# SFTTrainer — High-level orchestrator
# =========================================================================


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
        # Flamingo uses boundary tokens only (no <|protein_embed|>)
        approach = self.cfg.get("approach", "text")
        if approach == "flamingo":
            from src.models.multimodal_llm import PROTEIN_END_TOKEN, PROTEIN_START_TOKEN
            flamingo_tokens = [PROTEIN_START_TOKEN, PROTEIN_END_TOKEN]
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": flamingo_tokens}
            )
            if num_added > 0:
                log.info(
                    f"Added {num_added} flamingo boundary tokens: {flamingo_tokens} "
                    f"(vocab size: {len(self.tokenizer)})"
                )
        elif approach in ("esm3",):
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
        use_fsdp = self.cfg.training.get("fsdp", {}).get("enabled", False)

        # Load base model
        if quantization_config is not None:
            # QLoRA: explicit device placement (required by accelerate)
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
        elif use_fsdp:
            # FSDP: load on CPU — FSDP handles device placement and sharding
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            log.info("FSDP: loaded model on CPU (FSDP will shard to GPUs)")
        else:
            # DDP: explicit device placement
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

        # Apply LoRA (skip if explicitly disabled, e.g. pure Flamingo)
        approach = self.cfg.get("approach", "text")
        lora_enabled = self.cfg.training.get("lora", {}).get("enabled", True)
        if lora_enabled:
            lora_config = get_qlora_config(self.cfg)
            self.model = get_peft_model(self.model, lora_config)
            log.info("LoRA configuration applied")
            self.model.print_trainable_parameters()
        else:
            # Freeze all LLM parameters — only Flamingo components are trainable
            for param in self.model.parameters():
                param.requires_grad = False
            total = sum(p.numel() for p in self.model.parameters())
            log.info(f"LoRA disabled: all {total:,} LLM parameters frozen")

        # Load full ProteinLLM for multimodal training
        use_multimodal = self.cfg.get(
            "use_multimodal", approach in ("esm3", "flamingo")
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
            flamingo_cfg = encoder_cfg.get("flamingo", {})

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
                # Flamingo-specific params
                xattn_every=flamingo_cfg.get("xattn_every", 4),
                xattn_dim_head=flamingo_cfg.get("xattn_dim_head", 64),
                xattn_heads=flamingo_cfg.get("xattn_heads", 8),
                xattn_ff_mult=flamingo_cfg.get("xattn_ff_mult", 4),
                flamingo_num_queries=projector_cfg.get("num_queries", 64),
                flamingo_perceiver_layers=projector_cfg.get("perceiver_layers", 6),
                flamingo_ff_mult=projector_cfg.get("ff_mult", 4),
                flamingo_max_seq_len=projector_cfg.get("max_seq_len", 2048),
            )

            # Assign our already-loaded LoRA model and tokenizer
            self.protein_llm.llm = self.model
            self.protein_llm.tokenizer = self.tokenizer

            # Set hidden size from the loaded model's config
            self.protein_llm.llm_hidden_size = self.model.config.hidden_size

            # Build projector now that we know the LLM hidden size
            self.protein_llm._build_projector()

            # FSDP: cache embed_tokens weights before FSDP shards them.
            # embed_tokens is NOT LoRA-targeted, so weights are frozen and
            # a cached copy stays correct throughout training.
            # This avoids needing summon_full_params() in prepare_inputs().
            # Skip for flamingo — it doesn't call embed_tokens manually.
            use_fsdp = self.cfg.training.get("fsdp", {}).get("enabled", False)
            if use_fsdp and approach != "flamingo":
                base = (
                    self.model.get_base_model()
                    if hasattr(self.model, "get_base_model")
                    else self.model
                )
                embed_weight = base.model.embed_tokens.weight.data
                self.protein_llm._fsdp_embed_cache = embed_weight.clone().to(device)
                cache_gb = embed_weight.numel() * embed_weight.element_size() / 1024**3
                log.info(
                    f"FSDP: cached embed_tokens {tuple(embed_weight.shape)} "
                    f"on {device} ({cache_gb:.2f} GB)"
                )

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

        # For ESM-3 approach, replace protein text with placeholder token.
        # For flamingo, use boundary tokens only (no embed placeholder —
        # protein info comes via cross-attention, not inline embeddings).
        approach = self.cfg.get("approach", "text")
        if approach == "flamingo":
            from src.models.multimodal_llm import PROTEIN_END_TOKEN, PROTEIN_START_TOKEN
            placeholder = f"{PROTEIN_START_TOKEN}{PROTEIN_END_TOKEN}"
        elif approach in ("esm3",):
            placeholder = PROTEIN_PLACEHOLDER
        else:
            placeholder = ""

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

        # Cap eval dataset to avoid burning time on huge validation sets
        # (e.g., combined_sft_260225 has ~245K val samples)
        max_eval = self.cfg.training.get("max_eval_samples", None)
        if max_eval and len(self.eval_dataset) > max_eval:
            import random
            rng = random.Random(42)  # Fixed seed for reproducible eval subsets
            indices = sorted(rng.sample(range(len(self.eval_dataset)), max_eval))
            # flatten_indices() materializes the subset into a new PyArrow table
            # so .data.data.column("__length__") returns only the selected rows.
            # Without it, the raw PyArrow table still has all 271K rows, causing
            # LengthGroupedSampler to generate out-of-bounds indices.
            self.eval_dataset.data = self.eval_dataset.data.select(indices).flatten_indices()
            # Invalidate cached lengths so they're recomputed for the subset
            if hasattr(self.eval_dataset, "_lengths"):
                del self.eval_dataset._lengths
            log.info(
                f"Eval dataset capped: {len(self.eval_dataset)} samples "
                f"(max_eval_samples={max_eval})"
            )

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

        # Generation samples callback disabled — causes FSDP crashes due to
        # generate() bypassing FSDP's all-gather hooks (sharded params appear
        # as empty tensors).  Train/eval loss is sufficient for monitoring.
        # The summon_full_params fix exists in ProteinLLM.generate() but is
        # not yet validated in multi-GPU FSDP runs.
        # eval_cfg = self.cfg.get("evaluation", {})
        # gen_callback = GenerationSamplesCallback(
        #     protein_llm=self.protein_llm,
        #     eval_dataset=self.eval_dataset,
        #     tokenizer=self.tokenizer,
        #     num_samples_per_category=5,
        #     max_new_tokens=eval_cfg.get("sft_gen_max_tokens", 256),
        #     generation_temperature=float(eval_cfg.get("generation_temperature", 0.0)),
        # )
        # callbacks.append(gen_callback)

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

        # Auto-resume from latest checkpoint if any exist in output_dir
        resume_from = None
        ckpt_dir = self.cfg.paths.checkpoint_dir
        if os.path.isdir(ckpt_dir):
            ckpts = sorted(
                [d for d in Path(ckpt_dir).iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.rsplit("-", 1)[-1]),
            )
            if ckpts:
                resume_from = str(ckpts[-1])
                log.info(f"Auto-resuming from checkpoint: {resume_from}")

        # Train
        train_result = self.trainer.train(
            resume_from_checkpoint=resume_from,
        )

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
            "generation_temperature": self.cfg.get("evaluation", {}).get(
                "generation_temperature", 0.0
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

            # Under FSDP/torch.compile, protein_llm.llm is wrapped and its
            # save_pretrained doesn't emit adapter files. The HF Trainer
            # already saved them in the latest checkpoint-{step}/ directory.
            # Copy them into protein_llm/adapter/ so eval can find them.
            adapter_dir = path / "protein_llm" / "adapter"
            if not (adapter_dir / "adapter_config.json").exists():
                import shutil
                # Find the latest HF Trainer checkpoint
                ckpt_dirs = sorted(path.glob("checkpoint-*"), key=os.path.getmtime)
                if ckpt_dirs:
                    latest_ckpt = ckpt_dirs[-1]
                    src_config = latest_ckpt / "adapter_config.json"
                    if src_config.exists():
                        adapter_dir.mkdir(parents=True, exist_ok=True)
                        for fn in ["adapter_config.json", "adapter_model.safetensors",
                                    "adapter_model.bin"]:
                            src = latest_ckpt / fn
                            if src.exists():
                                shutil.copy2(src, adapter_dir / fn)
                        log.info(f"Copied LoRA adapter from {latest_ckpt} to {adapter_dir}")

        log.info(f"Checkpoint saved to {path}")
        return path


# =========================================================================
# Entry points
# =========================================================================


def run_sft(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run SFT training (unified entry point).

    Handles both QLoRA (4-bit quantization) and LoRA (no quantization)
    based on cfg.training.quantization.enabled.

    Args:
        cfg: Hydra configuration.

    Returns:
        Training metrics dictionary.
    """
    use_quant = cfg.training.get("quantization", {}).get("enabled", False)
    mode = "QLoRA" if use_quant else "LoRA"

    log.info("=" * 60)
    log.info(f"Starting SFT with {mode}")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model.path}")
    log.info(f"Encoder: {cfg.encoder.model_name}")
    log.info(f"Learning rate: {cfg.training.lr}")
    log.info(f"Batch size: {cfg.training.batch_size}")
    log.info(f"Epochs: {cfg.training.epochs}")

    # Create and run trainer
    trainer = SFTTrainer(cfg)
    trainer.setup()

    # Run training
    metrics = trainer.train()

    # Run final evaluation
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)

    log.info("=" * 60)
    log.info(f"SFT with {mode} completed")
    log.info("=" * 60)

    return metrics


def run_sft_qlora(cfg: DictConfig) -> Dict[str, Any]:
    """Run SFT with QLoRA. Backward-compatible alias for run_sft()."""
    return run_sft(cfg)


def run_sft_lora(cfg: DictConfig) -> Dict[str, Any]:
    """Run SFT with LoRA (no quantization). Backward-compatible alias for run_sft()."""
    return run_sft(cfg)
