"""ProteinLMBench Evaluation Module

Multiple-choice benchmark for protein understanding (944 questions).
Each question has 2-10 options with one correct answer.

Source: https://huggingface.co/datasets/tsynbio/ProteinLMBench
Paper: https://arxiv.org/abs/2406.05540

Covers protein knowledge across:
  - Enzyme catalysis and reaction mechanisms
  - Protein function and binding
  - Post-translational modifications
  - Subunit structure and interactions
  - Disease involvement
  - Tissue specificity and expression

Primary metric: accuracy (overall and per-option-count).
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProteinLMBenchSample:
    """A single multiple-choice question from ProteinLMBench."""
    sample_id: str
    question: str
    options: List[str]
    correct_answer: str      # e.g. "option 3"
    correct_index: int       # 0-based index
    explanation: str

@dataclass
class ProteinLMBenchResult:
    """Prediction result for a single question."""
    sample_id: str
    predicted_answer: str    # Parsed answer string (e.g. "option 2")
    predicted_index: int     # 0-based, -1 if unparsed
    correct_answer: str
    correct_index: int
    is_correct: bool
    generated_text: str
    num_options: int


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_mc_answer(
    generated_text: str,
    num_options: int,
) -> Tuple[str, int]:
    """Extract the selected option from generated text.

    Tries multiple strategies:
    1. Match "option N" pattern (matches dataset format)
    2. Match standalone number at start of response
    3. Match letter (A-J) and convert to option number
    4. Match "answer is N" / "answer: N" patterns

    Args:
        generated_text: Model's generated response.
        num_options: Number of options for this question.

    Returns:
        (answer_string, 0-based index). Index is -1 if unparsed.
    """
    text = generated_text.strip()

    # Strategy 1: "option N" pattern (exact match to dataset format)
    match = re.search(r'\boption\s+(\d+)\b', text, re.IGNORECASE)
    if match:
        n = int(match.group(1))
        if 1 <= n <= num_options:
            return f"option {n}", n - 1

    # Strategy 2: Starts with a number (e.g. "3", "3.", "3)")
    match = re.match(r'^\s*(\d+)\s*[.):\s]', text)
    if match:
        n = int(match.group(1))
        if 1 <= n <= num_options:
            return f"option {n}", n - 1

    # Strategy 3: Letter answer (A=1, B=2, ...) — common LLM response format
    match = re.match(r'^\s*\(?([A-Ja-j])\)?[.):\s]', text)
    if not match:
        match = re.search(r'\b(?:answer|correct)\s*(?:is|:)\s*\(?([A-Ja-j])\)?', text, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        n = ord(letter) - ord('A') + 1
        if 1 <= n <= num_options:
            return f"option {n}", n - 1

    # Strategy 4: "answer is N" / "answer: N"
    match = re.search(r'\b(?:answer|correct)\s*(?:is|:)\s*(?:option\s+)?(\d+)', text, re.IGNORECASE)
    if match:
        n = int(match.group(1))
        if 1 <= n <= num_options:
            return f"option {n}", n - 1

    # Strategy 5: Single number anywhere in short response (< 20 chars)
    if len(text) < 20:
        match = re.search(r'(\d+)', text)
        if match:
            n = int(match.group(1))
            if 1 <= n <= num_options:
                return f"option {n}", n - 1

    return "", -1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_proteinlm_bench_metrics(
    results: List[ProteinLMBenchResult],
) -> Dict[str, float]:
    """Compute accuracy metrics from prediction results.

    Returns:
        Dict with keys:
        - accuracy: Overall accuracy
        - total: Total questions
        - correct: Number correct
        - parsed: Number with successfully parsed answers
        - unparsed: Number where answer could not be extracted
    """
    if not results:
        return {"accuracy": 0.0, "total": 0, "correct": 0, "parsed": 0, "unparsed": 0}

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    parsed = sum(1 for r in results if r.predicted_index >= 0)
    unparsed = total - parsed

    metrics: Dict[str, Any] = {
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "parsed": parsed,
        "unparsed": unparsed,
        "parse_rate": parsed / total if total > 0 else 0.0,
    }

    # Per-option-count accuracy (group by number of options)
    from collections import defaultdict
    by_count: Dict[int, List[bool]] = defaultdict(list)
    for r in results:
        by_count[r.num_options].append(r.is_correct)

    for n_opts, corrects in sorted(by_count.items()):
        acc = sum(corrects) / len(corrects) if corrects else 0.0
        metrics[f"accuracy_{n_opts}opt"] = acc
        metrics[f"count_{n_opts}opt"] = len(corrects)

    return metrics


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_proteinlm_bench_dataset(
    cfg: DictConfig,
    max_samples: Optional[int] = None,
) -> List[ProteinLMBenchSample]:
    """Load ProteinLMBench evaluation dataset.

    Tries local processed JSON first, then falls back to HuggingFace download.

    Args:
        cfg: Hydra config (uses cfg.data.paths.processed or cfg.data.paths.raw).
        max_samples: Limit number of samples.

    Returns:
        List of ProteinLMBenchSample objects.
    """
    samples = []

    # Try local processed file
    data_cfg = cfg.get("data", {})
    processed_dir = data_cfg.get("paths", {}).get("processed", None)
    raw_dir = data_cfg.get("paths", {}).get("raw", None)

    loaded_records = None

    # Check processed dir
    if processed_dir:
        processed_path = Path(processed_dir) / "proteinlm_bench.json"
        if processed_path.exists():
            log.info(f"Loading ProteinLMBench from: {processed_path}")
            with open(processed_path) as f:
                loaded_records = json.load(f)

    # Check raw dir
    if loaded_records is None and raw_dir:
        raw_path = Path(raw_dir) / "evaluation.json"
        if raw_path.exists():
            log.info(f"Loading ProteinLMBench from raw: {raw_path}")
            with open(raw_path) as f:
                loaded_records = json.load(f)

    # Fall back to HuggingFace
    if loaded_records is None:
        log.info("Downloading ProteinLMBench from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset("tsynbio/ProteinLMBench", "evaluation", split="train")
            loaded_records = [dict(ds[i]) for i in range(len(ds))]
            log.info(f"Downloaded {len(loaded_records)} records from HuggingFace")
        except Exception as e:
            log.error(f"Failed to load ProteinLMBench: {e}")
            return []

    # Convert to ProteinLMBenchSample objects
    for i, record in enumerate(loaded_records):
        answer_str = record.get("answer", "")
        options = record.get("options", [])

        # Parse correct index from "option N" format
        match = re.match(r'option\s+(\d+)', answer_str)
        correct_idx = int(match.group(1)) - 1 if match else -1

        samples.append(ProteinLMBenchSample(
            sample_id=f"plb_{i:04d}",
            question=record.get("question", ""),
            options=options,
            correct_answer=answer_str,
            correct_index=correct_idx,
            explanation=record.get("explanation", ""),
        ))

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    log.info(f"Loaded {len(samples)} ProteinLMBench samples")
    return samples


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def create_proteinlm_bench_prompt(
    sample: ProteinLMBenchSample,
    prompt_template: Optional[str] = None,
) -> str:
    """Format a multiple-choice question as a prompt.

    Args:
        sample: ProteinLMBenchSample with question and options.
        prompt_template: Optional custom template.

    Returns:
        Formatted prompt string.
    """
    # Build option list with numbers (matching dataset convention)
    option_lines = []
    for i, opt in enumerate(sample.options):
        # Options already have "option N: " prefix — strip it for clean display
        clean = re.sub(r'^option\s+\d+\s*:\s*', '', opt)
        option_lines.append(f"{i + 1}) {clean}")

    options_text = "\n".join(option_lines)

    if prompt_template:
        return prompt_template.format(
            question=sample.question,
            options=options_text,
        )

    return f"""{sample.question}

{options_text}

Answer with just the option number (e.g., "option 1")."""


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_proteinlm_bench(
    cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
    model=None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate protein understanding via ProteinLMBench multiple-choice.

    Args:
        cfg: Hydra configuration.
        checkpoint_path: Path to model checkpoint.
        model: Pre-loaded model instance (skips loading if provided).
        output_dir: Directory to save prediction results.

    Returns:
        Dictionary of metric names to values.
    """
    log.info("Evaluating ProteinLMBench (multiple-choice)...")

    if model is None:
        from src.models.multimodal_llm import ProteinLLM
        if checkpoint_path:
            log.info(f"Loading model from checkpoint: {checkpoint_path}")
            model = ProteinLLM.from_pretrained(checkpoint_path)
        else:
            log.info("Creating model from config (no checkpoint provided)")
            model = ProteinLLM.from_config(cfg)
        model.eval()

    eval_cfg = cfg.get("evaluation", {})
    batch_size = eval_cfg.get("batch_size", 1)
    max_new_tokens = eval_cfg.get("max_new_tokens", 64)
    max_samples = eval_cfg.get("max_samples", None)

    samples = load_proteinlm_bench_dataset(cfg, max_samples=max_samples)
    if not samples:
        log.error("No ProteinLMBench samples loaded")
        return {"error": "no_samples"}

    log.info(f"Evaluating on {len(samples)} questions")

    results: List[ProteinLMBenchResult] = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        prompts = [create_proteinlm_bench_prompt(s) for s in batch]

        try:
            generated_texts = model.generate(
                prompt=prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        except Exception as e:
            log.error(f"Generation failed for batch {i}: {e}")
            continue

        for sample, gen_text in zip(batch, generated_texts):
            pred_answer, pred_idx = parse_mc_answer(gen_text, len(sample.options))

            is_correct = (pred_idx == sample.correct_index) if pred_idx >= 0 else False

            results.append(ProteinLMBenchResult(
                sample_id=sample.sample_id,
                predicted_answer=pred_answer,
                predicted_index=pred_idx,
                correct_answer=sample.correct_answer,
                correct_index=sample.correct_index,
                is_correct=is_correct,
                generated_text=gen_text,
                num_options=len(sample.options),
            ))

        if (i + batch_size) % 50 == 0 or (i + batch_size) >= len(samples):
            log.info(f"Processed {min(i + batch_size, len(samples))}/{len(samples)} questions")

    metrics = compute_proteinlm_bench_metrics(results)

    log.info("ProteinLMBench Results:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            log.info(f"  {key}: {value:.4f}")
        else:
            log.info(f"  {key}: {value}")

    if output_dir:
        _save_predictions(results, output_dir)

    return metrics


def _save_predictions(
    results: List[ProteinLMBenchResult],
    output_dir: str,
) -> None:
    """Save prediction results to JSON."""
    output_path = Path(output_dir) / "proteinlm_bench_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for r in results:
        records.append({
            "sample_id": r.sample_id,
            "predicted_answer": r.predicted_answer,
            "predicted_index": r.predicted_index,
            "correct_answer": r.correct_answer,
            "correct_index": r.correct_index,
            "is_correct": r.is_correct,
            "generated_text": r.generated_text,
            "num_options": r.num_options,
        })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    log.info(f"Predictions saved to {output_path}")
