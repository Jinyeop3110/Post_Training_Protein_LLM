"""Training callbacks for ProteinLLM.

Extracted from sft_trainer.py — stateless callback classes.

Classes:
    GPUMemoryCallback: Logs GPU memory usage during training.
    GenerationSamplesCallback: Generates and logs sample outputs during evaluation.
"""

import logging
import os
from typing import Dict, List, Optional

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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


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
        generation_temperature: float = 0.0,
    ):
        super().__init__()
        self.protein_llm = protein_llm
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples_per_category = num_samples_per_category
        self.max_new_tokens = max_new_tokens
        self.generation_temperature = generation_temperature
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
        """Generate sample outputs after each evaluation step.

        FSDP note: when FSDP is active, ALL ranks must call generate()
        because FSDP triggers all-gather during forward pass. Only rank 0
        collects and logs results; other ranks discard them.
        """
        is_rank_0 = int(os.environ.get("RANK", 0)) == 0
        use_fsdp = bool(args.fsdp)

        # In non-FSDP mode, only rank 0 generates (DDP has full model copy).
        # In FSDP mode, all ranks must participate for NCCL collective ops.
        if not is_rank_0 and not use_fsdp:
            return

        if is_rank_0:
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

                # Generate — all ranks must call this when FSDP is active
                try:
                    if self.protein_llm is not None:
                        # Multimodal path: use ProteinLLM.generate()
                        self.protein_llm.eval()
                        gen_kwargs = dict(
                            protein_sequences=[protein_seq],
                            prompt=[inference_prompt],
                            max_new_tokens=self.max_new_tokens,
                            min_new_tokens=10,
                            repetition_penalty=1.2,
                        )
                        if self.generation_temperature > 0:
                            gen_kwargs.update(do_sample=True, temperature=self.generation_temperature)
                        else:
                            gen_kwargs["do_sample"] = False
                        # autocast required: pooling/projector weights are float32
                        # but ESM-3 encoder outputs bfloat16 under its own autocast.
                        # During training HF Trainer's autocast handles this; during
                        # callback generation we must provide the context explicitly.
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            generated_texts = self.protein_llm.generate(**gen_kwargs)
                        generated = generated_texts[0] if generated_texts else ""
                        # Debug: if output is empty, decode without stripping special tokens.
                        # Skip in FSDP mode — debug generate would deadlock other ranks.
                        if is_rank_0 and not use_fsdp and not generated.strip():
                            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
                        # FSDP: model.device may be unreliable; use local CUDA device
                        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
                        inputs = self.tokenizer(
                            inference_prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=2048,
                        ).to(device)
                        text_gen_kwargs = dict(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            min_new_tokens=10,
                            repetition_penalty=1.2,
                        )
                        if self.generation_temperature > 0:
                            text_gen_kwargs.update(do_sample=True, temperature=self.generation_temperature)
                        else:
                            text_gen_kwargs["do_sample"] = False
                        # autocast for FSDP consistency (matches multimodal path)
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            output_ids = model.generate(**text_gen_kwargs)
                        # Slice off the input prompt tokens
                        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
                        generated = self.tokenizer.decode(
                            gen_ids, skip_special_tokens=True
                        )
                except Exception as e:
                    if is_rank_0:
                        log.warning(f"Generation failed for sample {idx}: {e}")
                    generated = f"[ERROR: {e}]"

                # Only rank 0 collects and logs results
                if is_rank_0:
                    gen_display = generated[:500] + ("..." if len(generated) > 500 else "")
                    exp_display = expected[:500] + ("..." if len(expected) > 500 else "")
                    seq_preview = protein_seq[:30] + "..." if len(protein_seq) > 30 else protein_seq

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

        # Log to wandb if available (rank 0 only)
        if is_rank_0 and HAS_WANDB and wandb.run is not None and table_rows:
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

        if is_rank_0:
            log.info("=" * 60)
