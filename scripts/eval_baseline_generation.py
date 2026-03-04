#!/usr/bin/env python3
"""Baseline generation evaluation for vanilla Qwen3-8B (no fine-tuning).

Generates outputs from the pre-trained LLM on the SFT evaluation dataset,
computes quality metrics, and saves results matching the format of
generation_quality_analysis.json.

Usage:
    source /home/yeopjin/orcd/pool/init_protein_llm.sh
    python scripts/eval_baseline_generation.py
    python scripts/eval_baseline_generation.py --model Qwen/Qwen3-8B --num-per-category 5
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mol_instructions import (
    DEFAULT_SYSTEM_PROMPT,
    MolInstructionsConfig,
    MolInstructionsDataset,
)
from src.models.vanilla_llm import VanillaLLMWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category inference (same patterns as GenerationSamplesCallback)
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS = [
    ("catalytic", ["catalytic activity", "catalytic", "enzyme commission", "ec number"]),
    ("domain", ["domain", "motif", "structural domain"]),
    ("design", ["design", "generate a protein", "create a protein"]),
    ("function", ["function", "functional description", "what does", "predict the function"]),
]


def infer_category(instruction: str) -> str:
    """Infer task category from instruction text (same as GenerationSamplesCallback)."""
    instruction_lower = instruction.lower()
    for category, keywords in CATEGORY_PATTERNS:
        if any(kw in instruction_lower for kw in keywords):
            return category
    return "general"


# ---------------------------------------------------------------------------
# Sample selection (same logic as GenerationSamplesCallback._select_samples)
# ---------------------------------------------------------------------------

def select_samples(
    dataset: MolInstructionsDataset,
    num_per_category: int = 5,
) -> Dict[str, List[int]]:
    """Select sample indices grouped by category.

    Iterates through the dataset collecting up to ``num_per_category``
    indices per category, using the same early-exit logic as
    GenerationSamplesCallback._select_samples: checks only *seen*
    categories (not a fixed target list) so that excluded categories
    (e.g. design) don't block the exit condition.
    """
    category_indices: Dict[str, List[int]] = {}

    for i in range(len(dataset)):
        item = dataset[i]
        category = infer_category(item.get("instruction", ""))
        category_indices.setdefault(category, [])
        if len(category_indices[category]) < num_per_category:
            category_indices[category].append(i)

        # Early exit: all *seen* categories have enough samples
        # (same as GenerationSamplesCallback — checks category_indices.values())
        if all(
            len(v) >= num_per_category for v in category_indices.values()
        ):
            # Wait until we've seen at least 100 items to discover categories
            if i >= min(100, len(dataset) - 1):
                break

    total = sum(len(v) for v in category_indices.values())
    log.info(
        "Selected %d samples across %d categories: %s",
        total,
        len(category_indices),
        ", ".join(f"{k}({len(v)})" for k, v in sorted(category_indices.items())),
    )
    return category_indices


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def word_overlap(generated: str, expected: str) -> float:
    """Compute word-level Jaccard similarity between generated and expected."""
    words_gen = set(generated.lower().split())
    words_exp = set(expected.lower().split())
    union = words_gen | words_exp
    if not union:
        return 0.0
    return len(words_gen & words_exp) / len(union)


def is_degenerate(text: str) -> bool:
    """Detect degenerate outputs with excessive repetition.

    Catches patterns seen in SFT run outputs:
    - Single char: "SSSSS...", "EEEEE..." (char dominance > 0.7)
    - Alternating: "DEDEDE...", "SDSDSD..." (periodic substring with period 1-4)
    - Trigram: "NNSNNSNNS..." (periodic or n-gram dominance)
    """
    text = text.strip()
    if len(text) < 20:
        return False

    # Single-char dominance (> 70% one character)
    char_counts: Dict[str, int] = defaultdict(int)
    for c in text:
        char_counts[c] += 1
    if max(char_counts.values()) / len(text) > 0.7:
        return True

    # Periodic substring detection for periods 1-4.
    # If a short motif repeats to cover most of the text, it's degenerate.
    # "DEDEDE..." has period 2; "NNSNNSNNS..." has period 3.
    for period in range(1, 5):
        if len(text) < period * 4:
            continue
        motif = text[:period]
        # Count how many period-aligned positions match the motif
        matches = sum(
            1 for i in range(0, len(text) - period + 1, period)
            if text[i : i + period] == motif
        )
        total_chunks = (len(text) - period + 1 + period - 1) // period  # ceil division
        if total_chunks > 0 and matches / total_chunks > 0.5:
            return True

    return False


def compute_aggregate_metrics(
    samples: List[Dict],
    high_sim_threshold: float = 0.5,
) -> Dict:
    """Compute aggregate metrics matching generation_quality_analysis.json format."""
    if not samples:
        return {}
    overlaps = [s["word_overlap"] for s in samples]
    lengths = [s["generation_length"] for s in samples]
    return {
        "avg_word_overlap": round(sum(overlaps) / len(overlaps), 3),
        "high_similarity_count": sum(1 for o in overlaps if o > high_sim_threshold),
        "degenerate_count": sum(1 for s in samples if s["is_degenerate"]),
        "avg_generation_length": round(sum(lengths) / len(lengths)),
        "total_samples": len(samples),
    }


def compute_per_category_metrics(
    samples: List[Dict],
    high_sim_threshold: float = 0.5,
    start_chars: int = 50,
) -> Dict[str, Dict]:
    """Compute per-category diversity and mode-collapse metrics."""
    categories: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        categories[s["category"]].append(s)

    result = {}
    for cat, cat_samples in sorted(categories.items()):
        starts = [s["generated"][:start_chars] for s in cat_samples]
        fulls = [s["generated"] for s in cat_samples]
        total = len(cat_samples)
        unique_fulls = len(set(fulls))

        result[cat] = {
            "high_similarity": sum(
                1 for s in cat_samples if s["word_overlap"] > high_sim_threshold
            ),
            "degenerate": sum(1 for s in cat_samples if s["is_degenerate"]),
            "total": total,
            "unique_full_generations": unique_fulls,
            "unique_generation_starts": len(set(starts)),
            "mode_collapse": unique_fulls < total,
        }
    return result


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def generate_baseline_samples(
    wrapper: VanillaLLMWrapper,
    dataset: MolInstructionsDataset,
    sample_indices: Dict[str, List[int]],
    max_new_tokens: int = 256,
    min_new_tokens: int = 10,
    repetition_penalty: float = 1.2,
) -> List[Dict]:
    """Generate outputs for selected samples using the vanilla LLM.

    Builds prompts with the same chat template as VanillaLLMWrapper, then
    calls wrapper.llm.generate() directly so we can pass min_new_tokens
    and repetition_penalty (which VanillaLLMWrapper.generate() does not
    forward).
    """
    all_samples = []
    total = sum(len(v) for v in sample_indices.values())
    done = 0

    for category, indices in sorted(sample_indices.items()):
        for idx in indices:
            item = dataset[idx]
            instruction = item.get("instruction", "")
            input_text = item.get("input_text", "")
            expected = item.get("response", "")
            protein_seq = item.get("protein_sequence", "")

            done += 1
            log.info(
                "[%d/%d] Generating for %s sample (idx=%d)...", done, total, category, idx
            )

            # Build prompt: same approach as VanillaLLMWrapper.generate()
            user_content = f"{instruction.strip()}\n\n{input_text.strip()}"
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            try:
                text = wrapper.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = wrapper.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = wrapper.tokenizer(text, return_tensors="pt").to(wrapper.device)
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = wrapper.llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                )

            generated_ids = outputs[0][input_length:]
            generated = wrapper.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            overlap = word_overlap(generated, expected)
            degenerate = is_degenerate(generated)

            seq_preview = (
                protein_seq[:30] + "..." if len(protein_seq) > 30 else protein_seq
            )

            log.info("  [%s] seq=%s", category, seq_preview)
            log.info("  Instruction: %s...", instruction[:100])
            log.info("  Expected:    %s...", expected[:200])
            log.info("  Generated:   %s...", generated[:200])
            log.info("  Word overlap: %.3f, Degenerate: %s", overlap, degenerate)

            all_samples.append(
                {
                    "category": category,
                    "instruction": instruction[:100],
                    "expected": expected,
                    "generated": generated,
                    "protein_seq": seq_preview,
                    "word_overlap": overlap,
                    "is_degenerate": degenerate,
                    "generation_length": len(generated),
                }
            )

    return all_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline generation evaluation for vanilla LLM",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B, unified base+instruct)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/combined_sft_260225",
        help="Path to dataset directory (Arrow or JSON)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--num-per-category",
        type=int,
        default=5,
        help="Number of samples per category (default: 5)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--output-dir",
        default="blog/data/03-02",
        help="Output directory for results (default: blog/data/03-02)",
    )
    parser.add_argument(
        "--high-sim-threshold",
        type=float,
        default=0.5,
        help="Threshold for high-similarity count (default: 0.5)",
    )
    args = parser.parse_args()

    # Triton cache must be local
    os.environ.setdefault(
        "TRITON_CACHE_DIR", f"/tmp/triton_cache_{os.environ.get('USER', 'unknown')}"
    )

    log.info("Model: %s", args.model)
    log.info("Data dir: %s", args.data_dir)
    log.info("Split: %s", args.split)

    # ------------------------------------------------------------------
    # 1. Load vanilla LLM
    # ------------------------------------------------------------------
    log.info("Loading vanilla LLM via VanillaLLMWrapper...")
    wrapper = VanillaLLMWrapper(model_name=args.model)

    # ------------------------------------------------------------------
    # 2. Load eval dataset (same config as SFT training)
    # ------------------------------------------------------------------
    log.info("Loading dataset from %s (split=%s)...", args.data_dir, args.split)
    config = MolInstructionsConfig(
        cache_dir=args.data_dir,
        # Match combined_sft_260225 settings
        sampling_temperature=0.7,
        exclude_files=["mol_protein_design.json"],
        max_protein_length=1024,
        train_split=0.9,
        val_split=0.05,
        test_split=0.05,
    )
    dataset = MolInstructionsDataset(split=args.split, config=config)
    log.info("Dataset loaded: %d samples", len(dataset))

    # ------------------------------------------------------------------
    # 3. Select 25 samples (5 per category)
    # ------------------------------------------------------------------
    sample_indices = select_samples(dataset, num_per_category=args.num_per_category)

    found_categories = set(sample_indices.keys())
    expected_categories = {"catalytic", "domain", "design", "function", "general"}
    missing = expected_categories - found_categories
    if missing:
        log.warning(
            "Missing categories in dataset: %s (data may exclude design tasks)", missing
        )

    # ------------------------------------------------------------------
    # 4. Generate outputs
    # ------------------------------------------------------------------
    samples = generate_baseline_samples(
        wrapper=wrapper,
        dataset=dataset,
        sample_indices=sample_indices,
        max_new_tokens=args.max_new_tokens,
    )

    # ------------------------------------------------------------------
    # 5. Compute quality metrics
    # ------------------------------------------------------------------
    agg_metrics = compute_aggregate_metrics(samples, args.high_sim_threshold)
    cat_metrics = compute_per_category_metrics(samples, args.high_sim_threshold)

    log.info("=" * 60)
    log.info("Aggregate Metrics:")
    for k, v in agg_metrics.items():
        log.info("  %s: %s", k, v)
    log.info("Per-Category Metrics:")
    for cat, m in cat_metrics.items():
        log.info("  %s: %s", cat, m)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 10,
        "do_sample": False,
        "repetition_penalty": 1.2,
    }

    # 6a. Raw generation samples (matching generation_samples.json schema)
    samples_for_json = []
    for s in samples:
        samples_for_json.append(
            {
                "category": s["category"],
                "instruction": s["instruction"],
                "expected": s["expected"][:500]
                + ("..." if len(s["expected"]) > 500 else ""),
                "generated": s["generated"][:500]
                + ("..." if len(s["generated"]) > 500 else ""),
                "protein_seq": s["protein_seq"],
            }
        )

    samples_output = {
        "model": args.model,
        "approach": "vanilla",
        "split": args.split,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generation_kwargs": gen_kwargs,
        "total_samples": len(samples_for_json),
        "samples": samples_for_json,
    }
    samples_path = output_dir / "baseline_generation_samples.json"
    with open(samples_path, "w") as f:
        json.dump(samples_output, f, indent=2)
    log.info("Saved generation samples to %s", samples_path)

    # 6b. Quality metrics (matching generation_quality_analysis.json per-run schema)
    quality_output = {
        "description": f"Baseline generation quality for {args.model} (no fine-tuning)",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": args.model,
        "approach": "vanilla",
        "projector_type": "none",
        "generation_kwargs": gen_kwargs,
        "per_step_metrics": {
            "step_0": agg_metrics,
        },
        "final_step_per_category": cat_metrics,
        "mode_collapse_analysis": {},
    }
    quality_path = output_dir / "baseline_generation_quality.json"
    with open(quality_path, "w") as f:
        json.dump(quality_output, f, indent=2)
    log.info("Saved quality metrics to %s", quality_path)

    log.info("Done!")


if __name__ == "__main__":
    main()
