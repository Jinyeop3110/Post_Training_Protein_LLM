"""Training callbacks for ProteinLLM.

Extracted from sft_trainer.py — stateless callback classes.

Classes:
    GPUMemoryCallback: Logs GPU memory usage during training.
    GenerationSamplesCallback: Thin wrapper around GenerationEvaluator.
"""

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from transformers import (
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


log = logging.getLogger(__name__)


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
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3

            if logs is not None:
                logs["gpu_memory_allocated_gb"] = round(allocated, 2)
                logs["gpu_memory_reserved_gb"] = round(reserved, 2)
                logs["gpu_memory_max_allocated_gb"] = round(max_allocated, 2)

            log.debug(
                f"GPU Memory - Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB"
            )


class GenerationSamplesCallback(TrainerCallback):
    """Generate and log sample outputs during evaluation.

    Thin wrapper around ``GenerationEvaluator`` from
    ``src.evaluation.generation``.  Delegates all generation, scoring,
    JSON saving, and wandb logging to the evaluator.

    After evaluation, restores ``protein_llm.train()`` so that
    pooling/projector dropout resumes (HF Trainer only calls
    ``model.train()`` on the LLM wrapper).
    """

    def __init__(
        self,
        protein_llm: Optional[nn.Module],
        eval_dataset,
        tokenizer,
        num_samples_per_category: int = 2,
        max_new_tokens: int = 256,
        generation_temperature: float = 0.0,
        output_dir: Optional[str] = None,
    ):
        super().__init__()
        from src.evaluation.generation import GenerationEvaluator

        self.protein_llm = protein_llm
        self._evaluator = GenerationEvaluator(
            protein_llm=protein_llm,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            num_samples_per_category=num_samples_per_category,
            max_new_tokens=max_new_tokens,
            generation_temperature=generation_temperature,
            output_dir=output_dir,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Generate sample outputs after each evaluation step."""
        use_fsdp = bool(args.fsdp)
        is_rank_0 = int(os.environ.get("RANK", 0)) == 0

        # In non-FSDP mode, only rank 0 generates.
        # In FSDP mode, all ranks must participate for NCCL collectives.
        if not is_rank_0 and not use_fsdp:
            return

        self._evaluator.evaluate(
            step=state.global_step,
            model=kwargs.get("model"),
            use_fsdp=use_fsdp,
            log_to_wandb=True,
        )

        # Restore train mode for pooling/projector
        if self.protein_llm is not None:
            self.protein_llm.train()
