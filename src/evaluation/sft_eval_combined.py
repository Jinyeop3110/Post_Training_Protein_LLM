"""
Per-Source / Per-File SFT Evaluation for Combined Dataset

Evaluates model quality on the combined SFT test split with granular
breakdowns by source (mol, sp, plm, pd, clap, p2t) and by individual
task file (sp_gene_prediction, plm_disease, etc.).

Reuses the scoring helpers from sft_eval.py but organises results by
the ``__source__`` and ``__filename__`` provenance metadata injected
during data loading.

Output keys: overall BLEU/ROUGE plus ``source/<name>/bleu``,
``file/<name>/bleu``, etc.

Usage::

    python scripts/evaluate.py experiment_name=<run> evaluation.name=sft_combined
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_combined_test_dataset(cfg: DictConfig, max_samples: Optional[int] = None):
    """Load combined test split with source/task_file metadata."""
    from src.data.mol_instructions import MolInstructionsDataset

    data_cfg = cfg.get("data", {})

    ds_cfg = {
        "source": data_cfg.get("source", "zjunlp/Mol-Instructions"),
        "subset": data_cfg.get("subset", "Protein-oriented Instructions"),
        "split": "test",
        "limit": max_samples,
    }

    if "paths" in data_cfg:
        ds_cfg["paths"] = data_cfg["paths"]
    if "processing" in data_cfg:
        ds_cfg["processing"] = data_cfg["processing"]
    if "splits" in data_cfg:
        ds_cfg["splits"] = data_cfg["splits"]

    dataset = MolInstructionsDataset.from_config(ds_cfg)
    log.info(f"Loaded {len(dataset)} test samples for combined SFT evaluation")
    return dataset


def _group_indices_by(dataset, key: str) -> Dict[str, List[int]]:
    """Group dataset indices by a sample field (``source`` or ``task_file``)."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        groups[sample.get(key, "unknown")].append(i)
    return dict(groups)


def _compute_bleu_rouge(reference: str, generated: str, sentence_bleu, smoothie, scorer):
    """Compute BLEU-4 and ROUGE-L for a single pair."""
    bleu_val = None
    rouge_val = None

    if sentence_bleu is not None:
        ref_tokens = reference.split()
        hyp_tokens = generated.split()
        if ref_tokens and hyp_tokens:
            bleu_val = sentence_bleu(
                [ref_tokens], hyp_tokens, smoothing_function=smoothie,
            )

    if scorer is not None:
        rouge_result = scorer.score(reference, generated)
        rouge_val = rouge_result["rougeL"].fmeasure

    return bleu_val, rouge_val


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stratified_sample_indices(
    group_indices: Dict[str, List[int]],
    n_per_group: int,
    seed: int = 42,
) -> List[int]:
    """Sample up to ``n_per_group`` indices from each group."""
    import random
    rng = random.Random(seed)
    selected: List[int] = []
    for group_name in sorted(group_indices):
        indices = group_indices[group_name]
        if len(indices) <= n_per_group:
            selected.extend(indices)
        else:
            selected.extend(rng.sample(indices, n_per_group))
    return selected


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_sft_combined(
    cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
    model=None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate SFT quality with per-source and per-file breakdowns.

    Args:
        cfg: Hydra configuration.
        checkpoint_path: Path to ProteinLLM checkpoint (unused if model given).
        model: Pre-loaded model (VanillaLLMWrapper or ProteinLLM).
        output_dir: Directory to save result files.

    Returns:
        Flat dictionary of metric names to values.
    """
    log.info("Evaluating combined SFT quality (per-source / per-file)...")

    # Load model if not provided
    if model is None:
        from src.models.multimodal_llm import ProteinLLM

        if checkpoint_path:
            model = ProteinLLM.from_pretrained(checkpoint_path)
        else:
            model = ProteinLLM.from_config(cfg)
        model.eval()

    # Eval settings
    eval_cfg = cfg.get("evaluation", {})
    max_samples = eval_cfg.get("max_samples", None)
    n_per_file = eval_cfg.get("sft_combined_gen_per_file", 20)
    n_examples_per_file = eval_cfg.get("sft_combined_examples_per_file", 2)

    # Load test data
    dataset = _load_combined_test_dataset(cfg, max_samples=max_samples)
    if len(dataset) == 0:
        log.error("No test samples loaded for combined SFT evaluation")
        return {"error": "no_test_samples"}

    # Group by source and file
    source_groups = _group_indices_by(dataset, "source")
    file_groups = _group_indices_by(dataset, "task_file")

    log.info(f"Sources ({len(source_groups)}): {sorted(source_groups.keys())}")
    log.info(f"Task files ({len(file_groups)}): {sorted(file_groups.keys())}")
    for src, idxs in sorted(source_groups.items()):
        log.info(f"  source={src}: {len(idxs)} test samples")
    for fn, idxs in sorted(file_groups.items()):
        log.info(f"  file={fn}: {len(idxs)} test samples")

    # Stratified sampling: n_per_file from each task file
    gen_indices = _stratified_sample_indices(file_groups, n_per_file)
    log.info(f"Stratified sample: {len(gen_indices)} indices across {len(file_groups)} files")

    # Setup scoring
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        smoothie = SmoothingFunction().method1
    except ImportError:
        log.warning("nltk not installed — skipping BLEU computation")
        sentence_bleu = None
        smoothie = None

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        log.warning("rouge_score not installed — skipping ROUGE computation")
        scorer = None

    if sentence_bleu is None and scorer is None:
        log.error("Neither nltk nor rouge_score installed; cannot compute metrics")
        return {"error": "missing_scoring_libs"}

    # Generate and score
    all_bleu: List[float] = []
    all_rouge: List[float] = []
    source_bleu: Dict[str, List[float]] = defaultdict(list)
    source_rouge: Dict[str, List[float]] = defaultdict(list)
    file_bleu: Dict[str, List[float]] = defaultdict(list)
    file_rouge: Dict[str, List[float]] = defaultdict(list)

    predictions: List[Dict[str, Any]] = []

    for count, idx in enumerate(gen_indices):
        sample = dataset[idx]
        source = sample.get("source", "unknown")
        task_file = sample.get("task_file", "unknown")
        protein_seq = sample.get("protein_sequence", "")
        prompt = sample.get("inference_prompt", f"{sample['instruction']}\n{sample['input_text']}")
        reference = sample["response"]

        try:
            generated = model.generate(
                protein_sequences=[protein_seq] if protein_seq else [""],
                prompt=[prompt],
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
            )[0]
        except Exception as e:
            log.warning(f"Generation failed for idx={idx}: {e}")
            continue

        bleu_val, rouge_val = _compute_bleu_rouge(
            reference, generated, sentence_bleu, smoothie, scorer,
        )

        if bleu_val is not None:
            all_bleu.append(bleu_val)
            source_bleu[source].append(bleu_val)
            file_bleu[task_file].append(bleu_val)

        if rouge_val is not None:
            all_rouge.append(rouge_val)
            source_rouge[source].append(rouge_val)
            file_rouge[task_file].append(rouge_val)

        predictions.append({
            "index": idx,
            "source": source,
            "task_file": task_file,
            "instruction": sample["instruction"],
            "input_preview": sample["input_text"][:120],
            "reference": reference,
            "generated": generated,
            "bleu": bleu_val,
            "rouge_l": rouge_val,
        })

        if (count + 1) % 50 == 0:
            log.info(f"  Combined eval: {count + 1}/{len(gen_indices)} samples generated")

    # Assemble metrics
    metrics: Dict[str, Any] = {
        "num_test_samples": len(dataset),
        "num_generated": len(predictions),
        "num_sources": len(source_groups),
        "num_files": len(file_groups),
    }

    if all_bleu:
        metrics["bleu"] = _avg(all_bleu)
    if all_rouge:
        metrics["rouge_l"] = _avg(all_rouge)

    # Per-source metrics
    per_source_metrics: Dict[str, Dict[str, Any]] = {}
    for src in sorted(source_groups):
        src_metrics: Dict[str, Any] = {
            "num_test": len(source_groups[src]),
            "num_generated": len(source_bleu.get(src, [])),
        }
        if src in source_bleu and source_bleu[src]:
            src_metrics["bleu"] = _avg(source_bleu[src])
            metrics[f"source/{src}/bleu"] = src_metrics["bleu"]
        if src in source_rouge and source_rouge[src]:
            src_metrics["rouge_l"] = _avg(source_rouge[src])
            metrics[f"source/{src}/rouge_l"] = src_metrics["rouge_l"]
        per_source_metrics[src] = src_metrics

    # Per-file metrics
    per_file_metrics: Dict[str, Dict[str, Any]] = {}
    for fn in sorted(file_groups):
        fn_metrics: Dict[str, Any] = {
            "num_test": len(file_groups[fn]),
            "num_generated": len(file_bleu.get(fn, [])),
        }
        if fn in file_bleu and file_bleu[fn]:
            fn_metrics["bleu"] = _avg(file_bleu[fn])
            metrics[f"file/{fn}/bleu"] = fn_metrics["bleu"]
        if fn in file_rouge and file_rouge[fn]:
            fn_metrics["rouge_l"] = _avg(file_rouge[fn])
            metrics[f"file/{fn}/rouge_l"] = fn_metrics["rouge_l"]
        per_file_metrics[fn] = fn_metrics

    # Log summary
    log.info("--- Per-Source Metrics ---")
    for src, m in sorted(per_source_metrics.items()):
        bleu_str = f"BLEU={m.get('bleu', 'N/A'):.4f}" if isinstance(m.get("bleu"), float) else "BLEU=N/A"
        rouge_str = f"ROUGE-L={m.get('rouge_l', 'N/A'):.4f}" if isinstance(m.get("rouge_l"), float) else "ROUGE-L=N/A"
        log.info(f"  {src}: {bleu_str}, {rouge_str} (n_test={m['num_test']}, n_gen={m['num_generated']})")

    log.info("--- Per-File Metrics ---")
    for fn, m in sorted(per_file_metrics.items()):
        bleu_str = f"BLEU={m.get('bleu', 'N/A'):.4f}" if isinstance(m.get("bleu"), float) else "BLEU=N/A"
        rouge_str = f"ROUGE-L={m.get('rouge_l', 'N/A'):.4f}" if isinstance(m.get("rouge_l"), float) else "ROUGE-L=N/A"
        log.info(f"  {fn}: {bleu_str}, {rouge_str} (n_test={m['num_test']}, n_gen={m['num_generated']})")

    # Collect qualitative examples (first n_examples_per_file from each file)
    examples: List[Dict[str, Any]] = []
    file_example_counts: Dict[str, int] = defaultdict(int)
    for pred in predictions:
        fn = pred["task_file"]
        if file_example_counts[fn] < n_examples_per_file:
            examples.append(pred)
            file_example_counts[fn] += 1

    # Save outputs
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Combined metrics (overall + per-source + per-file in one file)
        combined_metrics = {
            "overall": metrics,
            "per_source": per_source_metrics,
            "per_file": per_file_metrics,
        }
        with open(out / "sft_combined_metrics.json", "w") as f:
            json.dump(combined_metrics, f, indent=2, default=str)
        log.info(f"Metrics saved to {out / 'sft_combined_metrics.json'}")

        # All predictions
        with open(out / "sft_combined_predictions.json", "w") as f:
            json.dump(predictions, f, indent=2, default=str)
        log.info(f"Predictions saved to {out / 'sft_combined_predictions.json'} ({len(predictions)} records)")

        # Qualitative examples
        with open(out / "sft_combined_examples.json", "w") as f:
            json.dump(examples, f, indent=2, default=str)
        log.info(f"Examples saved to {out / 'sft_combined_examples.json'} ({len(examples)} records)")

    log.info(f"Combined SFT evaluation complete: {len(metrics)} metrics")
    return metrics
