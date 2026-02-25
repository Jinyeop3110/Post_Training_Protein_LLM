"""
SFT (Supervised Fine-Tuning) Evaluation

Evaluates model quality on the Mol-Instructions test split with:
- **Perplexity**: exp(avg cross-entropy loss) over the test set
- **BLEU-4**: Token overlap via nltk.translate.bleu_score
- **ROUGE-L**: Longest common subsequence via rouge_score

Generation metrics (BLEU/ROUGE) run on a configurable subset since they
require autoregressive decoding.

Output keys: ``perplexity``, ``bleu``, ``rouge_l``, ``num_samples``,
plus per-task variants like ``bleu_protein_function``.
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@dataclass
class SFTPredictionResult:
    """A single SFT generation prediction with scores."""

    instruction: str
    reference: str
    generated_text: str
    task_type: str
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None


# Task-type keywords found in Mol-Instructions instruction fields
TASK_KEYWORDS = {
    "protein_function": ["function", "functional"],
    "protein_design": ["design", "generate a protein", "create a protein"],
    "catalytic_activity": ["catalytic", "enzyme", "catalysis"],
    "domain_motif": ["domain", "motif"],
    "description": ["describe", "description"],
}


def _classify_task(instruction: str) -> str:
    """Classify a sample into a task type based on instruction keywords."""
    instruction_lower = instruction.lower()
    for task_name, keywords in TASK_KEYWORDS.items():
        if any(kw in instruction_lower for kw in keywords):
            return task_name
    return "other"


def _load_test_dataset(cfg: DictConfig, max_samples: Optional[int] = None):
    """Load Mol-Instructions test split."""
    from src.data.mol_instructions import MolInstructionsDataset

    # Build a config dict suitable for MolInstructionsDataset.from_config
    data_cfg = cfg.get("data", {})

    ds_cfg = {
        "source": data_cfg.get("source", "zjunlp/Mol-Instructions"),
        "subset": data_cfg.get("subset", "Protein-oriented Instructions"),
        "split": "test",
        "limit": max_samples,
    }

    # Propagate paths/processing if present
    if "paths" in data_cfg:
        ds_cfg["paths"] = data_cfg["paths"]
    if "processing" in data_cfg:
        ds_cfg["processing"] = data_cfg["processing"]
    if "splits" in data_cfg:
        ds_cfg["splits"] = data_cfg["splits"]

    dataset = MolInstructionsDataset.from_config(ds_cfg)
    log.info(f"Loaded {len(dataset)} test samples for SFT evaluation")
    return dataset


def _compute_perplexity(model, dataset, tokenizer, batch_size: int = 4) -> float:
    """Compute perplexity over dataset using teacher-forced forward passes."""
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(dataset), batch_size):
        batch_samples = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        prompts = [s["formatted_prompt"] for s in batch_samples]

        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Labels = input_ids shifted; pad tokens masked with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Use model's compute_loss if available, otherwise do manual forward
        if hasattr(model, "compute_loss"):
            loss_val = model.compute_loss(input_ids, attention_mask, labels)
        else:
            device = next(model.llm.parameters()).device
            with torch.no_grad():
                outputs = model.llm(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=labels.to(device),
                )
            loss_val = outputs.loss.item()

        # Count non-padding tokens in this batch
        n_tokens = (labels != -100).sum().item()
        total_loss += loss_val * n_tokens
        total_tokens += n_tokens

        if (i // batch_size) % 20 == 0:
            log.info(f"  Perplexity: processed {min(i + batch_size, len(dataset))}/{len(dataset)} samples")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def _compute_generation_metrics(
    model, dataset, max_gen_samples: int = 200,
) -> Tuple[Dict[str, Any], List[SFTPredictionResult]]:
    """Compute BLEU-4 and ROUGE-L on a subset via generation.

    Returns:
        Tuple of (metrics dict, list of per-sample predictions).
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        log.warning("nltk not installed — skipping BLEU computation")
        sentence_bleu = None

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        log.warning("rouge_score not installed — skipping ROUGE computation")
        scorer = None

    if sentence_bleu is None and scorer is None:
        return {}, []

    n_samples = min(len(dataset), max_gen_samples)
    bleu_scores: List[float] = []
    rouge_scores: List[float] = []
    predictions: List[SFTPredictionResult] = []

    # Per-task accumulators
    task_bleu: Dict[str, List[float]] = defaultdict(list)
    task_rouge: Dict[str, List[float]] = defaultdict(list)

    smoothie = SmoothingFunction().method1 if sentence_bleu else None

    for i in range(n_samples):
        sample = dataset[i]
        instruction = sample["instruction"]
        input_text = sample["input_text"]
        reference = sample["response"]
        protein_seq = sample.get("protein_sequence", "")

        # Build inference prompt (instruction + input, no response)
        prompt = sample.get("inference_prompt", f"{instruction}\n{input_text}")

        try:
            generated = model.generate(
                protein_sequences=[protein_seq] if protein_seq else [""],
                prompt=[prompt],
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
            )[0]
        except Exception as e:
            log.warning(f"Generation failed for sample {i}: {e}")
            continue

        task_type = _classify_task(instruction)
        sample_bleu: Optional[float] = None
        sample_rouge: Optional[float] = None

        # BLEU-4
        if sentence_bleu is not None:
            ref_tokens = reference.split()
            hyp_tokens = generated.split()
            if ref_tokens and hyp_tokens:
                score = sentence_bleu(
                    [ref_tokens], hyp_tokens, smoothing_function=smoothie,
                )
                bleu_scores.append(score)
                task_bleu[task_type].append(score)
                sample_bleu = score

        # ROUGE-L
        if scorer is not None:
            rouge_result = scorer.score(reference, generated)
            rl = rouge_result["rougeL"].fmeasure
            rouge_scores.append(rl)
            task_rouge[task_type].append(rl)
            sample_rouge = rl

        predictions.append(SFTPredictionResult(
            instruction=instruction,
            reference=reference,
            generated_text=generated,
            task_type=task_type,
            bleu=sample_bleu,
            rouge_l=sample_rouge,
        ))

        if (i + 1) % 50 == 0:
            log.info(f"  Generation metrics: {i + 1}/{n_samples} samples")

    metrics: Dict[str, Any] = {}
    metrics["gen_samples"] = n_samples

    if bleu_scores:
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores)
        for task, scores in task_bleu.items():
            metrics[f"bleu_{task}"] = sum(scores) / len(scores)

    if rouge_scores:
        metrics["rouge_l"] = sum(rouge_scores) / len(rouge_scores)
        for task, scores in task_rouge.items():
            metrics[f"rouge_l_{task}"] = sum(scores) / len(scores)

    return metrics, predictions


def _save_predictions(
    predictions: List[SFTPredictionResult],
    output_dir: str,
    task_name: str,
) -> None:
    """Save individual predictions to JSON file."""
    output_path = Path(output_dir) / f"{task_name}_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "instruction": p.instruction,
            "reference": p.reference,
            "generated_text": p.generated_text,
            "task_type": p.task_type,
            "bleu": p.bleu,
            "rouge_l": p.rouge_l,
        }
        for p in predictions
    ]

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    log.info(f"Predictions saved to {output_path}")


def _log_example_generations(
    model,
    dataset,
    n_per_task: int = 5,
    output_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generate and log example outputs for qualitative inspection.

    Selects up to ``n_per_task`` samples per task type using greedy decoding
    (temperature=0) and logs them.  Also saves to ``examples.json`` in
    *output_dir* if provided.

    Args:
        model: Model with a ``generate()`` method.
        dataset: Test dataset.
        n_per_task: Max examples per task type.
        output_dir: Optional directory for JSON output.

    Returns:
        List of example dicts (task_type, instruction, reference, generated).
    """
    # Collect indices per task type
    task_indices: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        task = _classify_task(sample["instruction"])
        if len(task_indices[task]) < n_per_task:
            task_indices[task].append(i)

    examples: List[Dict[str, str]] = []

    for task, indices in sorted(task_indices.items()):
        for idx in indices:
            sample = dataset[idx]
            protein_seq = sample.get("protein_sequence", "")
            prompt = sample.get("inference_prompt", f"{sample['instruction']}\n{sample['input_text']}")

            try:
                generated = model.generate(
                    protein_sequences=[protein_seq] if protein_seq else [""],
                    prompt=[prompt],
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            except Exception as e:
                log.warning(f"Example generation failed (idx={idx}): {e}")
                generated = f"[ERROR: {e}]"

            examples.append({
                "task_type": task,
                "instruction": sample["instruction"],
                "input_preview": sample["input_text"][:120],
                "reference": sample["response"],
                "generated": generated,
            })

    # Log examples
    log.info("=" * 70)
    log.info("EXAMPLE GENERATIONS (greedy, temperature=0)")
    log.info("=" * 70)
    for i, ex in enumerate(examples):
        log.info(f"\n--- Example {i+1}/{len(examples)} [{ex['task_type']}] ---")
        log.info(f"Instruction: {ex['instruction'][:120]}")
        log.info(f"Input:       {ex['input_preview']}")
        log.info(f"Reference:   {ex['reference'][:200]}")
        log.info(f"Generated:   {ex['generated'][:200]}")

    log.info("=" * 70)
    log.info(f"Logged {len(examples)} examples across {len(task_indices)} task types")

    # Save to file
    if output_dir:
        out_path = Path(output_dir) / "sft_examples.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(examples, f, indent=2)
        log.info(f"Examples saved to {out_path}")

    return examples


def evaluate_sft(
    cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
    model=None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate SFT quality on Mol-Instructions test split.

    Args:
        cfg: Hydra configuration.
        checkpoint_path: Path to ProteinLLM checkpoint (unused if model given).
        model: Pre-loaded model (VanillaLLMWrapper or ProteinLLM).
        output_dir: Directory to save prediction files.

    Returns:
        Dictionary of metric names to values.
    """
    log.info("Evaluating SFT quality...")

    # Load model if not provided
    if model is None:
        from src.models.multimodal_llm import ProteinLLM

        if checkpoint_path:
            model = ProteinLLM.from_pretrained(checkpoint_path)
        else:
            model = ProteinLLM.from_config(cfg)
        model.eval()

    # Resolve tokenizer — VanillaLLMWrapper and ProteinLLM both expose one
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer
        model_path = cfg.get("model", {}).get("path", "Qwen/Qwen3-4B")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Eval settings
    eval_cfg = cfg.get("evaluation", {})
    max_samples = eval_cfg.get("max_samples", None)
    max_gen_samples = eval_cfg.get("sft_gen_samples", 200)
    batch_size = eval_cfg.get("batch_size", 4)

    # Load test data
    dataset = _load_test_dataset(cfg, max_samples=max_samples)
    if len(dataset) == 0:
        log.error("No test samples loaded for SFT evaluation")
        return {"error": "no_test_samples"}

    metrics: Dict[str, Any] = {"num_samples": len(dataset)}

    # 1. Perplexity
    try:
        ppl = _compute_perplexity(model, dataset, tokenizer, batch_size=batch_size)
        metrics["perplexity"] = ppl
        log.info(f"Perplexity: {ppl:.2f}")
    except Exception as e:
        log.error(f"Perplexity computation failed: {e}", exc_info=True)
        metrics["perplexity"] = float("nan")

    # 2. Generation metrics (BLEU / ROUGE)
    predictions: List[SFTPredictionResult] = []
    try:
        gen_metrics, predictions = _compute_generation_metrics(
            model, dataset, max_gen_samples=max_gen_samples,
        )
        metrics.update(gen_metrics)
        if "bleu" in gen_metrics:
            log.info(f"BLEU-4: {gen_metrics['bleu']:.4f}")
        if "rouge_l" in gen_metrics:
            log.info(f"ROUGE-L: {gen_metrics['rouge_l']:.4f}")
    except Exception as e:
        log.error(f"Generation metrics failed: {e}", exc_info=True)

    # 3. Example generations for qualitative inspection
    n_per_task = eval_cfg.get("examples_per_task", 5)
    try:
        _log_example_generations(
            model, dataset, n_per_task=n_per_task, output_dir=output_dir,
        )
    except Exception as e:
        log.error(f"Example generation failed: {e}", exc_info=True)

    # Save predictions to output_dir if provided
    if output_dir and predictions:
        _save_predictions(predictions, output_dir, "sft")

    log.info(f"SFT evaluation complete: {len(metrics)} metrics")
    return metrics
