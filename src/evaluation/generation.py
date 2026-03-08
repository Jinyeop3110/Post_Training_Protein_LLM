"""Unified generation evaluation for ProteinLLM.

Used both during training (via GenerationSamplesCallback) and standalone
(via scripts/evaluate.py evaluation.name=generation).

Classes:
    GenerationEvaluator: Core generation + scoring engine.
"""

import gc
import json
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Keyword patterns to infer task category from instruction text
CATEGORY_PATTERNS = [
    ("catalytic", ["catalytic activity", "catalytic", "enzyme commission", "ec number"]),
    ("domain", ["domain", "motif", "structural domain"]),
    ("design", ["design", "generate a protein", "create a protein"]),
    ("function", ["function", "functional description", "what does", "predict the function"]),
]


def _init_scorers():
    """Lazily initialize BLEU and ROUGE scorers.

    Returns (sentence_bleu_fn, smoothing, rouge_scorer_obj) — any may be None.
    """
    sentence_bleu = None
    smoothing = None
    try:
        from nltk.translate.bleu_score import SmoothingFunction
        from nltk.translate.bleu_score import sentence_bleu as _sb
        sentence_bleu = _sb
        smoothing = SmoothingFunction().method1
    except ImportError:
        pass

    rouge_scorer_obj = None
    try:
        from rouge_score import rouge_scorer
        rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        pass

    return sentence_bleu, smoothing, rouge_scorer_obj


def _compute_sample_scores(reference, generated, sentence_bleu, smoothing, rouge_scorer_obj):
    """Compute BLEU-4 and ROUGE-L for a single (reference, generated) pair."""
    bleu = None
    rouge_l = None

    if sentence_bleu is not None:
        ref_tokens = reference.split()
        hyp_tokens = generated.split()
        if ref_tokens and hyp_tokens:
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

    if rouge_scorer_obj is not None:
        result = rouge_scorer_obj.score(reference, generated)
        rouge_l = result["rougeL"].fmeasure

    return bleu, rouge_l


def infer_category(instruction: str) -> str:
    """Infer task category from instruction text."""
    instruction_lower = instruction.lower()
    for category, keywords in CATEGORY_PATTERNS:
        if any(kw in instruction_lower for kw in keywords):
            return category
    return "general"


def select_samples(dataset, num_per_category: int = 2) -> Dict[str, List[int]]:
    """Select diverse sample indices grouped by category."""
    category_indices: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        cat = infer_category(item.get("instruction", ""))
        category_indices.setdefault(cat, [])
        if len(category_indices[cat]) < num_per_category:
            category_indices[cat].append(i)

        if all(len(v) >= num_per_category for v in category_indices.values()):
            if i >= min(100, len(dataset) - 1):
                break

    total = sum(len(v) for v in category_indices.values())
    log.info(
        f"Selected {total} generation samples across "
        f"{len(category_indices)} categories: "
        f"{', '.join(f'{k}({len(v)})' for k, v in category_indices.items())}"
    )
    return category_indices


class GenerationEvaluator:
    """Core generation evaluation engine.

    Works in two modes:
    - **Multimodal**: protein_llm is set, uses ProteinLLM.generate()
    - **Text-only**: protein_llm is None, uses raw LLM model.generate()

    Handles scoring (BLEU/ROUGE), JSON saving, and wandb logging.
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
        self.protein_llm = protein_llm
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples_per_category = num_samples_per_category
        self.max_new_tokens = max_new_tokens
        self.generation_temperature = generation_temperature
        self.output_dir = output_dir

        self._sample_indices = select_samples(eval_dataset, num_samples_per_category)
        self._sentence_bleu, self._smoothing, self._rouge_scorer = _init_scorers()
        if self._sentence_bleu is None and self._rouge_scorer is None:
            log.warning("Neither nltk nor rouge_score installed — scoring disabled")

    def _generate_one(
        self,
        item: Dict[str, Any],
        model: Optional[nn.Module] = None,
        use_fsdp: bool = False,
    ) -> str:
        """Generate output for a single sample.

        Args:
            item: Dataset item with protein_sequence, inference_prompt, etc.
            model: Raw LLM model for text-only path (unused in multimodal).
            use_fsdp: Whether FSDP is active (text-only needs summon_full_params).
        """
        protein_seq = item.get("protein_sequence", "")
        inference_prompt = item.get("inference_prompt", "")

        if self.protein_llm is not None:
            self.protein_llm.eval()
            gen_kwargs = dict(
                protein_sequences=[protein_seq],
                prompt=[inference_prompt],
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=10,
                repetition_penalty=1.2,
                wrap_chat_template=False,
            )
            if self.generation_temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=self.generation_temperature)
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                generated_texts = self.protein_llm.generate(**gen_kwargs)
            return generated_texts[0] if generated_texts else ""
        else:
            if model is None:
                return "[ERROR: no model provided for text-only generation]"
            model.eval()
            import os
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

            # FSDP guard: embed_tokens weight is flattened by FSDP, causing
            # "'weight' must be 2-D" during generate's autoregressive loop.
            # summon_full_params materializes full weights for generation.
            fsdp_ctx = None
            if use_fsdp:
                try:
                    from torch.distributed.fsdp import (
                        FullyShardedDataParallel as FSDP,
                    )
                    if isinstance(model, FSDP):
                        fsdp_ctx = FSDP.summon_full_params(
                            model, writeback=False
                        )
                except ImportError:
                    pass

            with torch.no_grad():
                if fsdp_ctx is not None:
                    torch.cuda.empty_cache()
                    with fsdp_ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        output_ids = model.generate(**text_gen_kwargs)
                else:
                    output_ids = model.generate(**text_gen_kwargs)
            gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def evaluate(
        self,
        step: Optional[int] = None,
        model: Optional[nn.Module] = None,
        use_fsdp: bool = False,
        log_to_wandb: bool = False,
    ) -> Dict[str, Any]:
        """Run generation evaluation on pre-selected samples.

        Args:
            step: Training step (for labeling output files).
            model: Raw LLM model for text-only path.
            use_fsdp: Whether FSDP is active (enables bail-out on failures).
            log_to_wandb: Whether to log results to wandb.

        Returns:
            Dict with 'predictions' list and aggregate 'bleu'/'rouge_l' scores.
        """
        import os
        is_rank_0 = int(os.environ.get("RANK", 0)) == 0

        if not is_rank_0 and not use_fsdp:
            return {}

        gc.collect()
        torch.cuda.empty_cache()

        if is_rank_0:
            log.info("=" * 60)
            log.info("Generation Evaluation%s", f" (step {step})" if step is not None else "")
            log.info("=" * 60)

        consecutive_failures = 0
        max_consecutive_failures = 3
        predictions: List[Dict[str, Any]] = []

        for category, indices in sorted(self._sample_indices.items()):
            for idx in indices:
                if use_fsdp and consecutive_failures >= max_consecutive_failures:
                    if is_rank_0:
                        log.warning(
                            f"Skipping remaining samples after "
                            f"{max_consecutive_failures} consecutive failures"
                        )
                    break

                item = self.eval_dataset[idx]
                try:
                    generated = self._generate_one(item, model=model, use_fsdp=use_fsdp)
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    if is_rank_0:
                        log.warning(f"Generation failed for sample {idx}: {e}")
                        log.warning(f"  {traceback.format_exc()}")
                    generated = f"[ERROR: {e}]"

                torch.cuda.empty_cache()

                if is_rank_0:
                    expected = item.get("response", "")
                    instruction = item.get("instruction", "")
                    protein_seq = item.get("protein_sequence", "")
                    bleu, rouge_l = _compute_sample_scores(
                        expected, generated,
                        self._sentence_bleu, self._smoothing, self._rouge_scorer,
                    )

                    seq_preview = protein_seq[:30] + "..." if len(protein_seq) > 30 else protein_seq
                    score_str = ""
                    if bleu is not None:
                        score_str += f" BLEU={bleu:.3f}"
                    if rouge_l is not None:
                        score_str += f" ROUGE-L={rouge_l:.3f}"

                    log.info(f"[{category}] seq={seq_preview}{score_str}")
                    log.info(f"  Instruction: {instruction[:100]}...")
                    log.info(f"  Expected:    {expected[:500]}")
                    log.info(f"  Generated:   {generated[:500]}")
                    log.info("")

                    predictions.append({
                        "category": category,
                        "instruction": instruction[:200],
                        "reference": expected,
                        "generated": generated,
                        "protein_sequence": protein_seq,
                        "protein_length": len(protein_seq),
                        "bleu": bleu,
                        "rouge_l": rouge_l,
                    })
            else:
                continue
            break  # FSDP bail-out broke inner loop

        if not is_rank_0 or not predictions:
            return {}

        # Aggregate scores
        cat_bleu: Dict[str, List[float]] = defaultdict(list)
        cat_rouge: Dict[str, List[float]] = defaultdict(list)
        all_bleu: List[float] = []
        all_rouge: List[float] = []

        for pred in predictions:
            cat = pred["category"]
            if pred["bleu"] is not None:
                cat_bleu[cat].append(pred["bleu"])
                all_bleu.append(pred["bleu"])
            if pred["rouge_l"] is not None:
                cat_rouge[cat].append(pred["rouge_l"])
                all_rouge.append(pred["rouge_l"])

        avg_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else None
        avg_rouge = sum(all_rouge) / len(all_rouge) if all_rouge else None

        if avg_bleu is not None or avg_rouge is not None:
            bleu_str = f"{avg_bleu:.4f}" if avg_bleu is not None else "N/A"
            rouge_str = f"{avg_rouge:.4f}" if avg_rouge is not None else "N/A"
            log.info(f"Generation scores: BLEU={bleu_str}, ROUGE-L={rouge_str}")

        result = {
            "step": step,
            "bleu": avg_bleu,
            "rouge_l": avg_rouge,
            "num_samples": len(predictions),
            "predictions": predictions,
        }

        # Per-category metrics
        metrics: Dict[str, float] = {}
        if avg_bleu is not None:
            metrics["generation/bleu"] = avg_bleu
        if avg_rouge is not None:
            metrics["generation/rouge_l"] = avg_rouge
        for cat, scores in cat_bleu.items():
            metrics[f"generation/bleu_{cat}"] = sum(scores) / len(scores)
        for cat, scores in cat_rouge.items():
            metrics[f"generation/rouge_l_{cat}"] = sum(scores) / len(scores)
        result["metrics"] = metrics

        # Save JSON
        if self.output_dir:
            label = f"step_{step}" if step is not None else "standalone"
            out_path = Path(self.output_dir) / f"generation_{label}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                log.info(f"Generation results saved to {out_path}")
            except Exception as e:
                log.warning(f"Failed to save generation results: {e}")

        # Log to wandb
        if log_to_wandb and metrics:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=step)
                    table = wandb.Table(
                        columns=["category", "instruction", "expected",
                                 "generated", "protein_seq", "bleu", "rouge_l"],
                        data=[
                            [p["category"], p["instruction"][:100],
                             p["reference"][:500], p["generated"][:500],
                             p["protein_sequence"], p["bleu"], p["rouge_l"]]
                            for p in predictions
                        ],
                    )
                    wandb.log(
                        {f"generation_samples/step_{step}": table},
                        step=step,
                    )
            except Exception as e:
                log.warning(f"Failed to log to wandb: {e}")

        log.info("=" * 60)
        return result


def evaluate_generation(
    cfg,
    checkpoint_path: Optional[str] = None,
    model=None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Standalone generation evaluation entry point.

    Compatible with scripts/evaluate.py dispatch interface.
    Loads eval dataset, creates GenerationEvaluator, runs evaluation.
    """

    from src.data.mol_instructions import MolInstructionsDataset

    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})

    # Load tokenizer
    if model is not None and hasattr(model, "tokenizer") and model.tokenizer is not None:
        tokenizer = model.tokenizer
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)

    # Determine protein_llm
    protein_llm = model if hasattr(model, "encode_protein") else None

    # Load eval dataset
    max_protein_length = data_cfg.get("processing", {}).get("max_protein_length")
    placeholder = "<|protein_embed|>" if cfg.get("approach") == "esm3" else ""

    dataset = MolInstructionsDataset(
        split="validation",
        dataset_name=data_cfg.get("source", "zjunlp/Mol-Instructions"),
        subset=data_cfg.get("subset", "Protein-oriented Instructions"),
        cache_dir=data_cfg.get("paths", {}).get("raw"),
        max_seq_length=cfg.get("training", {}).get("max_seq_length", 2048),
        max_protein_length=max_protein_length,
        limit=eval_cfg.get("max_samples"),
        protein_placeholder=placeholder,
        tokenizer=tokenizer,
    )

    num_per_cat = eval_cfg.get("sft_combined_gen_per_file", 3)
    max_tokens = eval_cfg.get("max_new_tokens", 256)
    temperature = float(eval_cfg.get("generation_temperature", 0.0))

    evaluator = GenerationEvaluator(
        protein_llm=protein_llm,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        num_samples_per_category=num_per_cat,
        max_new_tokens=max_tokens,
        generation_temperature=temperature,
        output_dir=output_dir,
    )

    result = evaluator.evaluate(log_to_wandb=False)
    return result.get("metrics", {})
