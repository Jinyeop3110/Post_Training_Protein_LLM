"""Reward functions for GRPO verifiable tasks.

Extracted from grpo_trainer.py — pure functions (text in → float out).

Functions:
    get_reward_function: Registry lookup for task-specific reward functions.
    compute_go_reward: F1 score of predicted GO terms.
    compute_ppi_reward: Binary PPI prediction accuracy.
    compute_stability_reward: ddG prediction + classification accuracy.
    compute_esmfold_reward: ESMFold structural quality assessment.
    compute_generic_reward: Fallback text-similarity reward.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# ESMFold category frequencies from the structure_quality dataset (10K samples).
# Used by focal reward weighting to combat category imbalance.
_ESMFOLD_CATEGORY_FREQ = {"high": 0.511, "medium": 0.454, "low": 0.035}

# Stability class frequencies from MegaScale dataset (10K samples).
# neutral: 67.4%, stabilizing: 31.8%, destabilizing: 0.8%
_STABILITY_CLASS_FREQ = {"neutral": 0.674, "stabilizing": 0.318, "destabilizing": 0.008}

# Solubility class frequencies (approximate, from eSOL)
_SOLUBILITY_CLASS_FREQ = {"soluble": 0.70, "insoluble": 0.30}

# CATH class name mapping for text-based matching
_CATH_CLASS_NAMES = {
    "1": "mainly alpha",
    "2": "mainly beta",
    "3": "alpha beta",
    "4": "few secondary structures",
    "5": "special",
}

# Instruction appended to GRPO prompts to encourage <answer> tag usage
ANSWER_TAG_INSTRUCTION = (
    "Always wrap your final answer in <answer> and </answer> tags."
)

# Format reward for using <answer> tags correctly
FORMAT_REWARD_BONUS = 0.1


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output before reward parsing."""
    return _THINK_RE.sub("", text).strip()


def extract_answer_content(text: str) -> Tuple[str, bool]:
    """Extract content from <answer>...</answer> tags.

    Args:
        text: Raw model output (may contain <think> and <answer> blocks).

    Returns:
        Tuple of (extracted_text, has_answer_tags).
        If <answer> tags found, returns content inside them.
        Otherwise returns text with <think> tags stripped (fallback).
    """
    # First strip think tags
    clean = _strip_think_tags(text)
    match = _ANSWER_RE.search(clean)
    if match:
        return match.group(1).strip(), True
    return clean, False


def get_reward_function(task: str) -> Callable[[str, Any], float]:
    """Get verifiable reward function for a specific protein task.

    Returns the appropriate reward function based on task type. These reward
    functions compute scores by comparing model predictions to ground truth,
    enabling verifiable rewards without a separate reward model.

    Args:
        task: Task type identifier. Supported values:
            - "go_prediction": Gene Ontology term prediction
            - "go_terms": Alias for go_prediction
            - "ppi": Protein-protein interaction prediction
            - "stability": Protein stability (ddG) prediction
            - "function": Generic function prediction (uses GO reward)

    Returns:
        Reward function that takes (generated_text, ground_truth) and returns float.

    Raises:
        ValueError: If task type is not supported.
    """
    task_lower = task.lower().replace("-", "_").replace(" ", "_")

    reward_functions = {
        "go_prediction": compute_go_reward,
        "go_terms": compute_go_reward,
        "go": compute_go_reward,
        "ppi": compute_ppi_reward,
        "ppi_prediction": compute_ppi_reward,
        "protein_protein_interaction": compute_ppi_reward,
        "stability": compute_stability_reward,
        "stability_prediction": compute_stability_reward,
        "ddg": compute_stability_reward,
        "function": compute_go_reward,
        "function_prediction": compute_go_reward,
        "esmfold": compute_esmfold_reward,
        "structure": compute_esmfold_reward,
        "structure_prediction": compute_esmfold_reward,
        "fold_quality": compute_esmfold_reward,
        "ss_composition": compute_ss_composition_reward,
        "structure_properties_a": compute_ss_composition_reward,
        "ss_sequence": compute_ss_sequence_reward,
        "ss_per_residue": compute_ss_sequence_reward,
        "structure_properties_b": compute_ss_sequence_reward,
        "structure_composite": compute_structure_composite_reward,
        "structure_properties": compute_structure_composite_reward,
        "structure_properties_c": compute_structure_composite_reward,
        "solubility": compute_solubility_reward,
        "solubility_prediction": compute_solubility_reward,
        "fold_classification": compute_fold_classification_reward,
        "fold_class": compute_fold_classification_reward,
        "cath": compute_fold_classification_reward,
        "cath_classification": compute_fold_classification_reward,
    }

    if task_lower not in reward_functions:
        raise ValueError(
            f"Unsupported task type: {task}. "
            f"Supported tasks: go_prediction, ppi, stability, "
            f"ss_composition, ss_sequence, structure_composite"
        )

    return reward_functions[task_lower]


def compute_go_reward(
    generated_text: str,
    ground_truth_go_terms: Union[str, List[str]],
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """Compute reward based on F1 score of predicted GO terms.

    Extracts GO terms from generated text and computes F1 score against
    ground truth terms. GO terms are expected in format GO:XXXXXXX.

    Args:
        generated_text: Model-generated text containing GO term predictions.
        ground_truth_go_terms: Ground truth GO terms as string (comma/space
            separated) or list of GO term strings.
        detailed: If True, return (reward, metrics_dict) tuple.

    Returns:
        F1 score between 0 and 1, or (f1, metrics) if detailed=True.

    Example:
        >>> compute_go_reward("The protein has GO:0003674 and GO:0005575",
        ...                   ["GO:0003674", "GO:0008150"])
        0.5  # Precision: 0.5, Recall: 0.5, F1: 0.5
    """
    generated_text = _strip_think_tags(generated_text)
    # Extract GO terms from generated text (format: GO:XXXXXXX)
    go_pattern = r"GO:\d{7}"
    predicted_terms = set(re.findall(go_pattern, generated_text.upper()))

    # Normalize ground truth
    if isinstance(ground_truth_go_terms, str):
        ground_truth_terms = set(re.findall(go_pattern, ground_truth_go_terms.upper()))
    else:
        ground_truth_terms = set()
        for term in ground_truth_go_terms:
            matches = re.findall(go_pattern, str(term).upper())
            ground_truth_terms.update(matches)

    # Handle edge cases
    if not ground_truth_terms:
        f1 = 1.0 if not predicted_terms else 0.0
        if detailed:
            return f1, {"precision": f1, "recall": f1, "f1": f1,
                        "num_predicted": len(predicted_terms),
                        "num_ground_truth": 0, "num_correct": 0}
        return f1

    if not predicted_terms:
        if detailed:
            return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                         "num_predicted": 0,
                         "num_ground_truth": len(ground_truth_terms),
                         "num_correct": 0}
        return 0.0

    # Compute F1 score
    true_positives = len(predicted_terms & ground_truth_terms)
    precision = true_positives / len(predicted_terms) if predicted_terms else 0.0
    recall = true_positives / len(ground_truth_terms) if ground_truth_terms else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    if detailed:
        return f1, {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_predicted": len(predicted_terms),
            "num_ground_truth": len(ground_truth_terms),
            "num_correct": true_positives,
        }
    return f1


def compute_ppi_reward(
    generated_text: str,
    ground_truth_label: Union[str, int, bool],
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute reward based on correct PPI (protein-protein interaction) prediction.

    Determines if the model correctly predicted whether two proteins interact.
    Searches for positive/negative indicators in the generated text.

    Args:
        generated_text: Model-generated text containing interaction prediction.
        ground_truth_label: True interaction label. Can be:
            - Boolean: True (interacts) or False (does not interact)
            - Integer: 1 (interacts) or 0 (does not interact)
            - String: "yes"/"interact"/"positive" vs "no"/"not"/"negative"
        detailed: If True, return (reward, metrics_dict) tuple.

    Returns:
        1.0 if prediction matches ground truth, 0.0 otherwise.
        If detailed=True, returns (reward, metrics) with accuracy, prediction, ground_truth.

    Example:
        >>> compute_ppi_reward("Yes, these proteins interact strongly.", True)
        1.0
        >>> compute_ppi_reward("The proteins do not interact.", True)
        0.0
    """
    # Normalize ground truth to boolean
    if isinstance(ground_truth_label, bool):
        gt_interacts = ground_truth_label
    elif isinstance(ground_truth_label, int):
        gt_interacts = ground_truth_label == 1
    elif isinstance(ground_truth_label, str):
        gt_label_lower = ground_truth_label.lower().strip()
        positive_indicators = {"yes", "true", "1", "interact", "interacts", "positive", "binding"}
        gt_interacts = any(ind in gt_label_lower for ind in positive_indicators)
    else:
        # Try to convert to int
        try:
            gt_interacts = int(ground_truth_label) == 1
        except (ValueError, TypeError):
            log.warning(f"Could not parse ground truth label: {ground_truth_label}")
            if detailed:
                return 0.0, {"accuracy": 0.0, "predicted": None,
                             "ground_truth": str(ground_truth_label), "ambiguous": True}
            return 0.0

    # Parse generated text for prediction
    generated_text = _strip_think_tags(generated_text)
    text_lower = generated_text.lower()

    # Check for explicit positive indicators
    positive_patterns = [
        r"\byes\b",
        r"\binteract[s]?\b",
        r"\bbind[s]?\b",
        r"\bpositive\b",
        r"\btrue\b",
        r"\bwill interact\b",
        r"\bdo interact\b",
        r"\blikely to interact\b",
    ]

    # Check for explicit negative indicators
    negative_patterns = [
        r"\bno\b",
        r"\bnot interact\b",
        r"\bdon't interact\b",
        r"\bdo not interact\b",
        r"\bnegative\b",
        r"\bfalse\b",
        r"\bunlikely\b",
        r"\bwill not\b",
        r"\bwon't\b",
    ]

    # Count matches
    positive_score = sum(1 for p in positive_patterns if re.search(p, text_lower))
    negative_score = sum(1 for p in negative_patterns if re.search(p, text_lower))

    # Determine prediction
    if positive_score > negative_score:
        pred_interacts = True
    elif negative_score > positive_score:
        pred_interacts = False
    else:
        # Ambiguous - check for any yes/no at the start
        if text_lower.strip().startswith("yes"):
            pred_interacts = True
        elif text_lower.strip().startswith("no"):
            pred_interacts = False
        else:
            # Cannot determine, give partial credit based on any match
            pred_interacts = None

    if pred_interacts is None:
        reward = 0.5  # Uncertain prediction gets partial credit
    else:
        reward = 1.0 if pred_interacts == gt_interacts else 0.0

    if detailed:
        return reward, {
            "accuracy": reward,
            "predicted": pred_interacts,
            "ground_truth": gt_interacts,
            "ambiguous": pred_interacts is None,
            "positive_score": positive_score,
            "negative_score": negative_score,
        }
    return reward


def _classify_ddg(ddg: float) -> str:
    """Classify a ddG value into stabilizing/neutral/destabilizing."""
    if ddg < -1.0:
        return "stabilizing"
    elif ddg > 1.0:
        return "destabilizing"
    return "neutral"


def compute_stability_reward(
    generated_text: str,
    ground_truth_ddg: Union[str, float, Dict[str, Any]],
    tolerance: float = 1.0,
    detailed: bool = False,
    focal_gamma: float = 0.0,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute reward based on ddG prediction and stability classification.

    Three reward components:
    1. Classification accuracy (0.4): correct stabilizing/neutral/destabilizing
       - stabilizing: ddG < -1.0
       - neutral: -1.0 <= ddG <= 1.0
       - destabilizing: ddG > 1.0
    2. Numerical accuracy (0.3): Gaussian decay on |predicted - true| ddG,
       sigma = ``tolerance`` (default 1.0 kcal/mol)
    3. Direction accuracy (0.3): correct sign of ddG (negative = stabilizing,
       positive = destabilizing). Neutral ground truth gets 0.3 if predicted
       magnitude is <= 1.5.

    Optional focal weighting combats class imbalance (67% neutral, 32%
    stabilizing, 0.8% destabilizing).

    Args:
        generated_text: Model-generated text containing stability prediction.
        ground_truth_ddg: True ddG value. Can be:
            - float or numeric string
            - JSON string with {"ddG": X, "stability_class": "...", "mutation": "..."}
        tolerance: Gaussian sigma in kcal/mol for numerical accuracy. Default 1.0.
        detailed: If True, return (reward, metrics_dict) tuple.
        focal_gamma: Focal weighting exponent (0 = disabled).

    Returns:
        Reward between 0 and 1 (scaled by focal weight), or
        (reward, metrics) if detailed=True.

    Example:
        >>> compute_stability_reward("ddG = -2.1 kcal/mol. Stabilizing.", -2.0)
        0.99  # correct class + direction + close numerically
    """
    # ── Parse ground truth ──
    gt_value = None
    gt_class = None
    gt_mutation = None

    if isinstance(ground_truth_ddg, dict):
        gt_value = float(ground_truth_ddg.get("ddG", 0))
        gt_class = ground_truth_ddg.get("stability_class")
        gt_mutation = ground_truth_ddg.get("mutation")
    elif isinstance(ground_truth_ddg, str):
        # Try JSON first
        if ground_truth_ddg.strip().startswith("{"):
            try:
                parsed = json.loads(ground_truth_ddg)
                gt_value = float(parsed.get("ddG", 0))
                gt_class = parsed.get("stability_class")
                gt_mutation = parsed.get("mutation")
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Fallback: extract number from string
        if gt_value is None:
            numbers = re.findall(r"-?\d+\.?\d*", ground_truth_ddg)
            if numbers:
                gt_value = float(numbers[0])
    elif isinstance(ground_truth_ddg, (int, float)):
        gt_value = float(ground_truth_ddg)

    if gt_value is None:
        if detailed:
            return 0.0, {"mae": float("nan"), "predicted": None,
                         "ground_truth": str(ground_truth_ddg), "parsed": False}
        return 0.0

    # Derive true class from ddG if not provided
    if gt_class is None:
        gt_class = _classify_ddg(gt_value)

    # ── Parse prediction ──
    generated_text = _strip_think_tags(generated_text)
    text_lower = generated_text.lower()

    # Extract predicted ddG number
    ddg_patterns = [
        r"ddG\s*[=:]\s*(-?\d+\.?\d*)",
        r"ΔΔG\s*[=:]\s*(-?\d+\.?\d*)",
        r"delta\s*G\s*[=:]\s*(-?\d+\.?\d*)",
        r"(-?\d+\.?\d*)\s*kcal\s*/?\s*mol",
        r"stability[:\s]+(-?\d+\.?\d*)",
        r"change[:\s]+(-?\d+\.?\d*)",
        r"predicted[:\s]+(-?\d+\.?\d*)",
    ]

    pred_value = None
    for pattern in ddg_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            try:
                pred_value = float(match.group(1))
                break
            except ValueError:
                continue

    # Fallback: any plausible number in ddG range
    if pred_value is None:
        numbers = re.findall(r"-?\d+\.?\d+", generated_text)
        for num_str in numbers:
            num = float(num_str)
            if -20 <= num <= 20:
                pred_value = num
                break

    # Extract predicted class from text
    stabilizing_kw = ["stabilizing", "stabilizes", "more stable", "increased stability"]
    destabilizing_kw = ["destabilizing", "destabilizes", "less stable", "decreased stability",
                        "unfavorable", "detrimental"]
    neutral_kw = ["neutral", "no significant", "negligible", "minimal effect",
                  "no effect", "marginal"]

    pred_class = None
    # Check explicit "classification: X" pattern
    class_match = re.search(
        r"(?:classification|class|effect)[:\s]+(\w+)", text_lower
    )
    if class_match:
        word = class_match.group(1)
        if word in ("stabilizing", "stabilizes"):
            pred_class = "stabilizing"
        elif word in ("destabilizing", "destabilizes"):
            pred_class = "destabilizing"
        elif word in ("neutral", "negligible"):
            pred_class = "neutral"

    # Fallback: keyword matching
    if pred_class is None:
        if any(kw in text_lower for kw in destabilizing_kw):
            pred_class = "destabilizing"
        elif any(kw in text_lower for kw in stabilizing_kw):
            pred_class = "stabilizing"
        elif any(kw in text_lower for kw in neutral_kw):
            pred_class = "neutral"

    # If still None, infer from predicted ddG value
    if pred_class is None and pred_value is not None:
        pred_class = _classify_ddg(pred_value)

    # ── Component 1: Classification accuracy (0.4) ──
    class_reward = 0.4 if pred_class == gt_class else 0.0

    # ── Component 2: Numerical accuracy (0.3) ──
    numerical_reward = 0.0
    if pred_value is not None:
        error = abs(pred_value - gt_value)
        numerical_reward = 0.3 * float(
            torch.exp(torch.tensor(-error**2 / (2 * tolerance**2)))
        )
    else:
        error = float("nan")

    # ── Component 3: Direction accuracy (0.3) ──
    direction_reward = 0.0
    if pred_value is not None:
        gt_sign = (gt_value < 0)  # True = stabilizing direction
        pred_sign = (pred_value < 0)
        if abs(gt_value) <= 1.0:
            # Neutral ground truth: reward if predicted magnitude is small
            direction_reward = 0.3 if abs(pred_value) <= 1.5 else 0.0
        elif gt_sign == pred_sign:
            direction_reward = 0.3

    reward = class_reward + numerical_reward + direction_reward

    # ── Focal weighting ──
    focal_weight = 1.0
    if focal_gamma > 0:
        focal_weight = _focal_weight(gt_class, focal_gamma, _STABILITY_CLASS_FREQ)
    reward *= focal_weight

    if detailed:
        return reward, {
            "mae": error if pred_value is not None else float("nan"),
            "squared_error": error ** 2 if pred_value is not None else float("nan"),
            "predicted": pred_value,
            "ground_truth": gt_value,
            "true_class": gt_class,
            "pred_class": pred_class,
            "class_correct": pred_class == gt_class,
            "direction_correct": direction_reward > 0,
            "classification_reward": class_reward,
            "numerical_reward": numerical_reward,
            "direction_reward": direction_reward,
            "focal_weight": focal_weight,
            "parsed": pred_value is not None,
            "mutation": gt_mutation,
        }
    return reward


def _focal_weight(
    category: str,
    gamma: float,
    freq_table: Optional[Dict[str, float]] = None,
) -> float:
    """Compute normalized focal weight for a category.

    Weight = (1 - freq)^gamma, normalized so the weighted mean across
    the dataset equals 1.0 (preserves overall reward scale).

    Args:
        category: Category name (e.g. "high", "stabilizing").
        gamma: Focal exponent. 0 = all weights 1.0.
        freq_table: Category-to-frequency mapping. Defaults to
            ``_ESMFOLD_CATEGORY_FREQ`` for backward compatibility.

    Returns:
        Normalized focal weight for the given category.
    """
    if freq_table is None:
        freq_table = _ESMFOLD_CATEGORY_FREQ
    raw = {
        cat: (1.0 - freq) ** gamma
        for cat, freq in freq_table.items()
    }
    # Normalize: mean weight across dataset = 1.0
    mean_weight = sum(
        raw[cat] * freq for cat, freq in freq_table.items()
    )
    if mean_weight > 0:
        return raw.get(category, 1.0) / mean_weight
    return 1.0


def compute_esmfold_reward(
    generated_text: str,
    protein_sequence_or_metrics: str,
    detailed: bool = False,
    focal_gamma: float = 0.0,
    binary_alignment: bool = False,
    classification_only: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute reward based on ESMFold structural quality assessment.

    Supports two modes:
    1. Live ESMFold: pass a protein sequence string, folds with ESMFold
    2. Pre-computed: pass a JSON string with {"plddt": X, "ptm": Y} from
       metadata (e.g., from AlphaFold DB download). Instant, no GPU needed.

    When classification_only=True, reward is purely binary:
    - Correct category (high/medium/low) = 1.0, wrong = 0.0
    - No partial credit, no pLDDT numerical accuracy, no quality claims
    - Categories: high (pLDDT>80), medium (50-80), low (<50)
    - Optional focal weighting still applies
    - Format bonus (+0.1) still applies in the trainer

    When classification_only=False (default), reward has 3 components:
    1. Quality claim alignment (0.4): correct "well-folded"/"disordered" claim
    2. Numerical pLDDT prediction accuracy (0.3): Gaussian decay on error
    3. Fold quality category match (0.3): high/medium/low correct

    After computing the base reward, an optional focal weight is applied based
    on the true pLDDT category frequency.  This combats category imbalance
    (51% high, 45% medium, 3.5% low) by up-weighting rare categories:
        weight = (1 - freq) ^ gamma, normalized so mean weight = 1.0.
    Set focal_gamma=0 (default) to disable.

    When binary_alignment=True, quality alignment scoring is strict:
    - No partial credit (0.2) for medium-range proteins
    - Adds medium_terms detection ("moderate", "medium quality", etc.)
    - Exact match required: well-folded claims must match pLDDT>70,
      disordered claims must match pLDDT<50, medium claims must match 50-70

    Args:
        generated_text: Model-generated text about the protein.
        protein_sequence_or_metrics: Either a protein sequence (for live
            ESMFold) or a JSON string with pre-computed {"plddt", "ptm"}.
        detailed: If True, return (reward, metrics) tuple.
        focal_gamma: Focal weighting exponent (0 = disabled).
        binary_alignment: If True, use strict binary alignment scoring
            (no partial credit, adds medium claims detection).
        classification_only: If True, pure binary classification reward
            (correct category = 1.0, wrong = 0.0).

    Returns:
        Reward between 0 and 1 (scaled by focal weight), or
        (reward, metrics) if detailed=True.
    """
    # Check if input is pre-computed metrics (JSON string)
    plddt = None
    ptm = None
    if protein_sequence_or_metrics.strip().startswith("{"):
        try:
            metrics = json.loads(protein_sequence_or_metrics)
            plddt = float(metrics.get("plddt", 0))
            ptm = float(metrics.get("ptm", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass  # Fall through to ESMFold path

    # Live ESMFold path
    if plddt is None:
        from src.models.esmfold_wrapper import get_esmfold_predictor

        predictor = get_esmfold_predictor()
        fold_result = predictor.predict(protein_sequence_or_metrics)
        plddt = fold_result["plddt"]
        ptm = fold_result["ptm"]

    generated_text = _strip_think_tags(generated_text)
    text_lower = generated_text.lower()

    # Determine true category from pLDDT
    if plddt > 80:
        true_category = "high"
    elif plddt > 50:
        true_category = "medium"
    else:
        true_category = "low"

    # ── Classification-only mode: binary reward ──
    if classification_only:
        # Detect predicted category from text
        high_kw = ["high", "excellent", "very confident", "well-folded",
                    "well folded", "reliable structure"]
        medium_kw = ["medium", "moderate", "intermediate", "partially",
                     "mixed", "reasonable"]
        low_kw = ["low", "poor", "disordered", "unfolded", "unstructured",
                  "unreliable", "intrinsically disordered"]

        pred_category = None
        # Check for explicit "fold quality: X" or "quality: X" patterns first
        cat_match = re.search(
            r"(?:fold\s+)?quality[:\s]+(\w+)", text_lower
        )
        if cat_match:
            cat_word = cat_match.group(1)
            if cat_word in ("high", "excellent"):
                pred_category = "high"
            elif cat_word in ("medium", "moderate", "intermediate"):
                pred_category = "medium"
            elif cat_word in ("low", "poor"):
                pred_category = "low"

        # Fallback: keyword matching
        if pred_category is None:
            if any(kw in text_lower for kw in low_kw):
                pred_category = "low"
            elif any(kw in text_lower for kw in medium_kw):
                pred_category = "medium"
            elif any(kw in text_lower for kw in high_kw):
                pred_category = "high"

        correct = pred_category == true_category
        reward = 1.0 if correct else 0.0

        # Focal weighting
        focal_weight = 1.0
        if focal_gamma > 0:
            focal_weight = _focal_weight(true_category, focal_gamma)
        reward *= focal_weight

        if detailed:
            return reward, {
                "plddt": plddt,
                "ptm": ptm,
                "true_category": true_category,
                "pred_category": pred_category,
                "correct": correct,
                "focal_weight": focal_weight,
                "classification_only": True,
            }
        return reward

    # ── Multi-component mode (original) ──
    reward = 0.0
    component_scores = {}

    # Component 1: Structural quality claim alignment (0.4 max)
    well_folded_terms = [
        "well-folded", "well folded", "stable", "ordered",
        "structured", "high confidence", "reliable structure",
    ]
    disordered_terms = [
        "disordered", "unfolded", "unstructured", "intrinsically disordered",
        "low confidence", "unreliable",
    ]

    medium_terms = [
        "moderate", "medium quality", "medium confidence",
        "intermediate", "partially folded", "mixed",
    ]

    claims_well_folded = any(term in text_lower for term in well_folded_terms)
    claims_disordered = any(term in text_lower for term in disordered_terms)
    claims_medium = any(term in text_lower for term in medium_terms)

    quality_reward = 0.0
    if binary_alignment:
        # Strict binary scoring: exact match or zero, no partial credit.
        # Medium claims also recognized for proteins in the 50-70 range.
        if claims_well_folded and plddt > 70:
            quality_reward = 0.4
        elif claims_disordered and plddt < 50:
            quality_reward = 0.4
        elif claims_medium and 50 <= plddt <= 70:
            quality_reward = 0.4
        # Everything else is 0.0 (wrong claim or no claim)
    else:
        # Original scoring with partial credit for medium-range
        if claims_well_folded and plddt > 70:
            quality_reward = 0.4
        elif claims_disordered and plddt < 50:
            quality_reward = 0.4
        elif claims_well_folded and plddt < 50:
            quality_reward = 0.0  # Wrong claim
        elif claims_disordered and plddt > 70:
            quality_reward = 0.0  # Wrong claim
        elif claims_well_folded or claims_disordered:
            quality_reward = 0.2  # Partially correct range

    reward += quality_reward
    component_scores["quality_alignment"] = quality_reward

    # Component 2: Numerical pLDDT prediction accuracy (0.3 max)
    plddt_patterns = [
        r"plddt[:\s]+(\d+\.?\d*)",
        r"confidence[:\s]+(\d+\.?\d*)",
        r"plDDT[:\s]+(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*%?\s*(?:plddt|confidence)",
    ]

    pred_plddt = None
    for pattern in plddt_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 0 <= val <= 100:
                    pred_plddt = val
                    break
            except ValueError:
                continue

    numerical_reward = 0.0
    if pred_plddt is not None:
        error = abs(pred_plddt - plddt)
        numerical_reward = 0.3 * float(
            torch.exp(torch.tensor(-error**2 / (2 * 10**2)))
        )

    reward += numerical_reward
    component_scores["numerical_accuracy"] = numerical_reward

    # Component 3: Fold quality category (0.3 max)
    # true_category already computed above

    high_terms = ["high quality", "high confidence", "very confident", "excellent"]
    medium_terms = ["moderate", "medium confidence", "reasonable"]
    low_terms = ["low quality", "low confidence", "poor", "unreliable"]

    category_reward = 0.0
    if true_category == "high" and any(t in text_lower for t in high_terms):
        category_reward = 0.3
    elif true_category == "medium" and any(t in text_lower for t in medium_terms):
        category_reward = 0.3
    elif true_category == "low" and any(t in text_lower for t in low_terms):
        category_reward = 0.3

    reward += category_reward
    component_scores["category_match"] = category_reward

    # Focal weighting: up-weight rare categories to combat imbalance
    focal_weight = 1.0
    if focal_gamma > 0:
        focal_weight = _focal_weight(true_category, focal_gamma)
    reward *= focal_weight

    if detailed:
        return reward, {
            "plddt": plddt,
            "ptm": ptm,
            "predicted_plddt": pred_plddt,
            "true_category": true_category,
            "claims_well_folded": claims_well_folded,
            "claims_disordered": claims_disordered,
            "claims_medium": claims_medium,
            "focal_weight": focal_weight,
            **component_scores,
        }
    return reward



# ── Dominant SS category frequencies (estimated from typical PDB distributions)
_SS_DOMINANT_FREQ = {
    "helix-rich": 0.35, "sheet-rich": 0.25, "mixed": 0.25, "coil-rich": 0.15,
}


def _parse_ss_fractions(text: str) -> Dict[str, Optional[float]]:
    """Parse helix/sheet/coil percentages from model text.

    Looks for patterns like "45.2% helix" or "helix: 45.2%".
    """
    result = {"helix": None, "sheet": None, "coil": None}
    for ss_type in result:
        # Pattern: "X% helix" or "helix: X%" or "helix X%"
        patterns = [
            rf"(\d+\.?\d*)\s*%?\s*{ss_type}",
            rf"{ss_type}\s*[:\s]\s*(\d+\.?\d*)\s*%",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result[ss_type] = float(m.group(1)) / 100.0
                break
    return result


def _parse_rsa(text: str) -> Optional[float]:
    """Parse mean RSA value from model text."""
    patterns = [
        r"(?:mean\s+)?RSA[:\s]+(\d+\.?\d*)",
        r"RSA\s*=\s*(\d+\.?\d*)",
        r"solvent\s+accessibility[:\s]+(\d+\.?\d*)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return val if val <= 1.0 else val / 100.0  # Handle % format
    return None


def _parse_ss3_string(text: str) -> Optional[str]:
    """Extract per-residue SS3 string (H/E/C) from model text.

    Looks for patterns like "SS3: HHHEEEECCCC" or a long run of H/E/C chars.
    """
    # Try explicit SS3 tag first
    m = re.search(r"SS3[:\s]+([HEC]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Try <answer>SS3: ...</answer> pattern
    answer, has_tag = extract_answer_content(text)
    if has_tag:
        m = re.search(r"SS3[:\s]+([HEC]+)", answer, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # If answer is just H/E/C string
        clean = answer.strip()
        if clean and all(c in "HEC" for c in clean.upper()) and len(clean) >= 5:
            return clean.upper()

    # Fallback: find longest run of H/E/C characters
    runs = re.findall(r"[HEC]{10,}", text.upper())
    if runs:
        return max(runs, key=len)

    return None


def compute_ss_composition_reward(
    generated_text: str,
    ground_truth: Union[str, Dict[str, Any]],
    detailed: bool = False,
    focal_gamma: float = 0.0,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute reward for secondary structure composition prediction (Option A).

    Three components:
    1. SS fraction accuracy (0.4): Gaussian decay on absolute error of
       helix%, sheet%, coil% predictions.
    2. Dominant SS type match (0.3): correct dominant type
       (helix-rich/sheet-rich/mixed/coil-rich).
    3. RSA accuracy (0.3): Gaussian decay on |predicted - true| mean RSA.

    Args:
        generated_text: Model-generated text with SS percentages.
        ground_truth: JSON string or dict with helix_fraction, sheet_fraction,
            coil_fraction, mean_rsa, dominant_ss.
        detailed: If True, return (reward, metrics) tuple.
        focal_gamma: Focal weighting exponent (0 = disabled).

    Returns:
        Reward between 0 and 1.
    """
    # Parse ground truth
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except (json.JSONDecodeError, ValueError):
            gt = {}
    else:
        gt = ground_truth

    gt_h = float(gt.get("helix_fraction", 0))
    gt_e = float(gt.get("sheet_fraction", 0))
    gt_c = float(gt.get("coil_fraction", 0))
    gt_rsa = float(gt.get("mean_rsa", 0.5))
    gt_dominant = gt.get("dominant_ss", "")

    # Parse predictions
    generated_text = _strip_think_tags(generated_text)
    pred_fracs = _parse_ss_fractions(generated_text)
    pred_rsa = _parse_rsa(generated_text)

    # Component 1: SS fraction accuracy (0.4)
    frac_reward = 0.0
    parsed_fracs = 0
    frac_errors = {}
    for ss_type, gt_val in [("helix", gt_h), ("sheet", gt_e), ("coil", gt_c)]:
        pred_val = pred_fracs[ss_type]
        if pred_val is not None:
            error = abs(pred_val - gt_val)
            frac_errors[ss_type] = error
            # Gaussian with sigma=0.15 (15% tolerance)
            frac_reward += float(torch.exp(torch.tensor(-error**2 / (2 * 0.15**2))))
            parsed_fracs += 1
    if parsed_fracs > 0:
        frac_reward = 0.4 * (frac_reward / 3.0)  # Normalize by 3 types
    else:
        frac_reward = 0.0

    # Component 2: Dominant SS type (0.3)
    dominant_reward = 0.0
    pred_h = pred_fracs["helix"]
    pred_e = pred_fracs["sheet"]
    if pred_h is not None and pred_e is not None:
        if pred_h > 0.4:
            pred_dominant = "helix-rich"
        elif pred_e > 0.3:
            pred_dominant = "sheet-rich"
        elif pred_h > 0.15 and pred_e > 0.15:
            pred_dominant = "mixed"
        else:
            pred_dominant = "coil-rich"
        if pred_dominant == gt_dominant:
            dominant_reward = 0.3
    else:
        pred_dominant = None
        # Check text keywords
        text_lower = generated_text.lower()
        if gt_dominant and gt_dominant.replace("-", " ") in text_lower:
            dominant_reward = 0.3

    # Component 3: RSA accuracy (0.3)
    rsa_reward = 0.0
    if pred_rsa is not None:
        rsa_error = abs(pred_rsa - gt_rsa)
        rsa_reward = 0.3 * float(torch.exp(torch.tensor(-rsa_error**2 / (2 * 0.1**2))))

    reward = frac_reward + dominant_reward + rsa_reward

    # Focal weighting
    focal_weight = 1.0
    if focal_gamma > 0 and gt_dominant:
        focal_weight = _focal_weight(gt_dominant, focal_gamma, _SS_DOMINANT_FREQ)
    reward *= focal_weight

    if detailed:
        return reward, {
            "fraction_reward": frac_reward,
            "dominant_reward": dominant_reward,
            "rsa_reward": rsa_reward,
            "pred_fracs": pred_fracs,
            "gt_fracs": {"helix": gt_h, "sheet": gt_e, "coil": gt_c},
            "frac_errors": frac_errors,
            "pred_dominant": pred_dominant,
            "gt_dominant": gt_dominant,
            "pred_rsa": pred_rsa,
            "gt_rsa": gt_rsa,
            "focal_weight": focal_weight,
            "parsed_fracs": parsed_fracs,
        }
    return reward


def compute_ss_sequence_reward(
    generated_text: str,
    ground_truth: Union[str, Dict[str, Any]],
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute reward for per-residue SS3 string prediction (Option B).

    Two components:
    1. Per-residue accuracy (0.6): fraction of positions where predicted SS
       matches ground truth. Handles length mismatch by truncating to shorter.
    2. Segment-level F1 (0.4): F1 score on SS segments (contiguous runs of
       same type). Rewards correct segment boundaries, not just individual
       residues.

    Args:
        generated_text: Model text containing SS3 string (H/E/C).
        ground_truth: JSON string or dict with "ss3_string" key, or just
            the SS3 string directly.
        detailed: If True, return (reward, metrics) tuple.

    Returns:
        Reward between 0 and 1.
    """
    # Parse ground truth
    if isinstance(ground_truth, dict):
        gt_ss3 = ground_truth.get("ss3_string", "")
    elif isinstance(ground_truth, str):
        if ground_truth.strip().startswith("{"):
            try:
                gt = json.loads(ground_truth)
                gt_ss3 = gt.get("ss3_string", "")
            except (json.JSONDecodeError, ValueError):
                gt_ss3 = ground_truth.strip().upper()
        else:
            gt_ss3 = ground_truth.strip().upper()
    else:
        gt_ss3 = ""

    if not gt_ss3 or not all(c in "HEC" for c in gt_ss3):
        if detailed:
            return 0.0, {"error": "invalid ground truth SS3"}
        return 0.0

    # Parse prediction
    generated_text = _strip_think_tags(generated_text)
    pred_ss3 = _parse_ss3_string(generated_text)

    if not pred_ss3:
        if detailed:
            return 0.0, {"error": "no SS3 string found in prediction",
                         "gt_length": len(gt_ss3)}
        return 0.0

    # Component 1: Per-residue accuracy (0.6)
    min_len = min(len(pred_ss3), len(gt_ss3))
    matches = sum(1 for i in range(min_len) if pred_ss3[i] == gt_ss3[i])

    # Penalize length mismatch
    max_len = max(len(pred_ss3), len(gt_ss3))
    per_residue_acc = matches / max_len if max_len > 0 else 0.0
    acc_reward = 0.6 * per_residue_acc

    # Component 2: Segment-level F1 (0.4)
    def _get_segments(ss_str: str) -> List[Tuple[str, int, int]]:
        """Extract segments as (type, start, end) tuples."""
        if not ss_str:
            return []
        segments = []
        start = 0
        for i in range(1, len(ss_str)):
            if ss_str[i] != ss_str[start]:
                segments.append((ss_str[start], start, i))
                start = i
        segments.append((ss_str[start], start, len(ss_str)))
        return segments

    gt_segments = _get_segments(gt_ss3)
    pred_segments = _get_segments(pred_ss3)

    # Match segments: a predicted segment matches a GT segment if they have
    # the same type and overlap > 50% of the GT segment length.
    matched_gt = set()
    matched_pred = set()
    for pi, (pt, ps, pe) in enumerate(pred_segments):
        for gi, (gt, gs, ge) in enumerate(gt_segments):
            if pt != gt or gi in matched_gt:
                continue
            overlap_start = max(ps, gs)
            overlap_end = min(pe, ge)
            overlap = max(0, overlap_end - overlap_start)
            gt_len = ge - gs
            if gt_len > 0 and overlap / gt_len > 0.5:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    seg_precision = len(matched_pred) / len(pred_segments) if pred_segments else 0
    seg_recall = len(matched_gt) / len(gt_segments) if gt_segments else 0
    if seg_precision + seg_recall > 0:
        seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall)
    else:
        seg_f1 = 0.0
    seg_reward = 0.4 * seg_f1

    reward = acc_reward + seg_reward

    if detailed:
        return reward, {
            "per_residue_accuracy": per_residue_acc,
            "accuracy_reward": acc_reward,
            "segment_f1": seg_f1,
            "segment_precision": seg_precision,
            "segment_recall": seg_recall,
            "segment_reward": seg_reward,
            "pred_length": len(pred_ss3),
            "gt_length": len(gt_ss3),
            "num_gt_segments": len(gt_segments),
            "num_pred_segments": len(pred_segments),
            "matched_segments": len(matched_gt),
        }
    return reward


def compute_structure_composite_reward(
    generated_text: str,
    ground_truth: Union[str, Dict[str, Any]],
    detailed: bool = False,
    focal_gamma: float = 0.0,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute composite structural property reward (Option C).

    Four weighted components:
    1. SS3 accuracy (0.30): Per-residue SS3 accuracy (from Option B).
    2. SS composition (0.25): Gaussian accuracy on helix/sheet/coil fractions.
    3. Ramachandran region match (0.20): Correct alpha/beta/other fractions.
    4. RSA + contact prediction (0.25): RSA accuracy + contact density.

    Args:
        generated_text: Model text with comprehensive structural analysis.
        ground_truth: JSON string or dict with full structural properties.
        detailed: If True, return (reward, metrics) tuple.
        focal_gamma: Focal weighting exponent (0 = disabled).

    Returns:
        Reward between 0 and 1.
    """
    # Parse ground truth
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except (json.JSONDecodeError, ValueError):
            gt = {}
    else:
        gt = ground_truth if isinstance(ground_truth, dict) else {}

    generated_text = _strip_think_tags(generated_text)
    text_lower = generated_text.lower()

    # ── Component 1: SS3 per-residue accuracy (0.30) ──
    ss3_reward = 0.0
    gt_ss3 = gt.get("ss3_string", "")
    if gt_ss3:
        pred_ss3 = _parse_ss3_string(generated_text)
        if pred_ss3:
            min_len = min(len(pred_ss3), len(gt_ss3))
            max_len = max(len(pred_ss3), len(gt_ss3))
            matches = sum(1 for i in range(min_len) if pred_ss3[i] == gt_ss3[i])
            ss3_reward = 0.30 * (matches / max_len if max_len > 0 else 0)

    # ── Component 2: SS composition accuracy (0.25) ──
    comp_reward = 0.0
    pred_fracs = _parse_ss_fractions(generated_text)
    gt_h = float(gt.get("helix_fraction", 0))
    gt_e = float(gt.get("sheet_fraction", 0))
    gt_c = float(gt.get("coil_fraction", 0))

    frac_scores = []
    for ss_type, gt_val in [("helix", gt_h), ("sheet", gt_e), ("coil", gt_c)]:
        pred_val = pred_fracs[ss_type]
        if pred_val is not None:
            error = abs(pred_val - gt_val)
            frac_scores.append(
                float(torch.exp(torch.tensor(-error**2 / (2 * 0.15**2))))
            )
    if frac_scores:
        comp_reward = 0.25 * (sum(frac_scores) / 3.0)

    # ── Component 3: Ramachandran region match (0.20) ──
    rama_reward = 0.0
    gt_rama = gt.get("ramachandran", {})
    if gt_rama:
        # Parse Ramachandran from text
        rama_patterns = {
            "alpha": [r"(\d+\.?\d*)\s*%?\s*alpha", r"alpha[:\s]+(\d+\.?\d*)\s*%"],
            "beta": [r"(\d+\.?\d*)\s*%?\s*beta", r"beta[:\s]+(\d+\.?\d*)\s*%"],
        }
        pred_rama = {}
        for region, pats in rama_patterns.items():
            for pat in pats:
                m = re.search(pat, text_lower)
                if m:
                    pred_rama[region] = float(m.group(1)) / 100.0
                    break

        if pred_rama:
            rama_scores = []
            for region in ["alpha", "beta"]:
                if region in pred_rama and region in gt_rama:
                    error = abs(pred_rama[region] - float(gt_rama[region]))
                    rama_scores.append(
                        float(torch.exp(torch.tensor(-error**2 / (2 * 0.15**2))))
                    )
            if rama_scores:
                rama_reward = 0.20 * (sum(rama_scores) / len(rama_scores))

    # ── Component 4: RSA + contact density (0.25) ──
    rsa_contact_reward = 0.0

    # RSA sub-component (0.15 of 0.25)
    pred_rsa = _parse_rsa(generated_text)
    gt_rsa = float(gt.get("mean_rsa", 0.5))
    rsa_sub = 0.0
    if pred_rsa is not None:
        rsa_error = abs(pred_rsa - gt_rsa)
        rsa_sub = 0.15 * float(torch.exp(torch.tensor(-rsa_error**2 / (2 * 0.1**2))))

    # Contact density sub-component (0.10 of 0.25)
    contact_sub = 0.0
    gt_contacts = gt.get("num_contacts", 0)
    gt_density = float(gt.get("contact_density", 0))
    # Parse contact count or density
    contact_pats = [
        r"(\d+)\s*(?:long[- ]range\s+)?contacts",
        r"contacts[:\s]+(\d+)",
        r"contact\s+density[:\s]+(\d+\.?\d*)",
    ]
    for pat in contact_pats:
        m = re.search(pat, text_lower)
        if m:
            pred_val = float(m.group(1))
            if "density" in pat:
                error = abs(pred_val - gt_density)
                contact_sub = 0.10 * float(
                    torch.exp(torch.tensor(-error**2 / (2 * 0.01**2)))
                )
            else:
                # Contact count: Gaussian with sigma = max(10, 20% of GT)
                sigma = max(10, gt_contacts * 0.2)
                error = abs(pred_val - gt_contacts)
                contact_sub = 0.10 * float(
                    torch.exp(torch.tensor(-error**2 / (2 * sigma**2)))
                )
            break

    rsa_contact_reward = rsa_sub + contact_sub

    reward = ss3_reward + comp_reward + rama_reward + rsa_contact_reward

    # Focal weighting
    focal_weight = 1.0
    gt_dominant = gt.get("dominant_ss", "")
    if focal_gamma > 0 and gt_dominant:
        focal_weight = _focal_weight(gt_dominant, focal_gamma, _SS_DOMINANT_FREQ)
    reward *= focal_weight

    if detailed:
        return reward, {
            "ss3_reward": ss3_reward,
            "composition_reward": comp_reward,
            "ramachandran_reward": rama_reward,
            "rsa_contact_reward": rsa_contact_reward,
            "rsa_sub": rsa_sub,
            "contact_sub": contact_sub,
            "pred_fracs": pred_fracs,
            "gt_fracs": {"helix": gt_h, "sheet": gt_e, "coil": gt_c},
            "pred_rsa": pred_rsa,
            "gt_rsa": gt_rsa,
            "focal_weight": focal_weight,
        }
    return reward


def compute_solubility_reward(
    generated_text: str,
    ground_truth: Union[str, float, Dict[str, Any]],
    tolerance: float = 15.0,
    detailed: bool = False,
    focal_gamma: float = 0.0,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute reward for protein solubility prediction.

    Three components:
    - Classification (0.5): exact match soluble/insoluble
    - Numerical accuracy (0.3): Gaussian decay on |predicted% - true%|, σ=tolerance
    - Format compliance (0.2): correct output structure

    Args:
        generated_text: Model output text.
        ground_truth: Dict with solubility_score/solubility_class, float (score),
            or JSON string.
        tolerance: σ for Gaussian decay on numerical error (default 15%).
        detailed: If True, return (reward, metrics_dict).
        focal_gamma: Focal weighting gamma (0 = disabled).

    Returns:
        Reward in [0, 1], or (reward, metrics) if detailed=True.
    """
    import math

    # Parse ground truth
    true_score = None
    true_class = None

    if isinstance(ground_truth, dict):
        true_score = ground_truth.get("solubility_score")
        true_class = ground_truth.get("solubility_class")
    elif isinstance(ground_truth, (int, float)):
        true_score = float(ground_truth)
    elif isinstance(ground_truth, str):
        try:
            gt_dict = json.loads(ground_truth)
            true_score = gt_dict.get("solubility_score")
            true_class = gt_dict.get("solubility_class")
        except (json.JSONDecodeError, TypeError):
            try:
                true_score = float(ground_truth)
            except ValueError:
                pass

    if true_score is not None and true_class is None:
        true_class = "soluble" if true_score > 50 else "insoluble"

    if true_score is None and true_class is None:
        if detailed:
            return 0.0, {"error": "no_ground_truth"}
        return 0.0

    # Parse prediction
    text = generated_text.strip().lower()

    # Extract class — careful: "insoluble" contains "soluble"
    pred_class = None
    if "insoluble" in text:
        pred_class = "insoluble"
    elif "soluble" in text:
        pred_class = "soluble"

    # Extract numerical score (0-100)
    pred_score = None
    # Try patterns like "solubility: 72.3%", "estimated solubility: 72%"
    score_patterns = [
        r"(?:estimated\s+)?solubility[:\s]+(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%",
        r"solubility[:\s]+(\d+(?:\.\d+)?)",
    ]
    for pattern in score_patterns:
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                if 0 <= val <= 100:
                    pred_score = val
                    break
            except ValueError:
                continue

    # Format check
    has_format = pred_class is not None and pred_score is not None
    format_reward = 1.0 if has_format else (0.5 if pred_class is not None else 0.0)

    # Classification reward
    class_reward = 0.0
    if pred_class is not None and true_class is not None:
        class_reward = 1.0 if pred_class == true_class else 0.0

    # Numerical reward (Gaussian decay)
    numerical_reward = 0.0
    if pred_score is not None and true_score is not None:
        error = abs(pred_score - true_score)
        numerical_reward = math.exp(-0.5 * (error / tolerance) ** 2)

    # Weighted total
    reward = 0.5 * class_reward + 0.3 * numerical_reward + 0.2 * format_reward

    # Focal weighting
    focal_weight = 1.0
    if focal_gamma > 0 and true_class:
        focal_weight = _focal_weight(true_class, focal_gamma, _SOLUBILITY_CLASS_FREQ)
        reward *= focal_weight

    if detailed:
        metrics = {
            "classification_reward": class_reward,
            "numerical_reward": numerical_reward,
            "format_reward": format_reward,
            "predicted_class": pred_class or "none",
            "predicted_score": pred_score,
            "true_class": true_class,
            "true_score": true_score,
            "focal_weight": focal_weight,
            "mae": (
                abs(pred_score - true_score)
                if pred_score is not None and true_score is not None
                else None
            ),
        }
        return reward, metrics
    return reward


def compute_fold_classification_reward(
    generated_text: str,
    ground_truth: Union[str, Dict[str, Any]],
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute reward for CATH fold classification with hierarchical partial credit.

    Reward = 0.25 per matching CATH level (Class, Architecture, Topology, Homology).

    Args:
        generated_text: Model output text.
        ground_truth: Dict with cath_code and level names, or JSON string,
            or dotted CATH code string (e.g., "1.10.490.10").
        detailed: If True, return (reward, metrics_dict).

    Returns:
        Reward in [0, 1], or (reward, metrics) if detailed=True.
    """
    # Parse ground truth
    true_code = None

    if isinstance(ground_truth, dict):
        true_code = ground_truth.get("cath_code")
    elif isinstance(ground_truth, str):
        try:
            gt_dict = json.loads(ground_truth)
            true_code = gt_dict.get("cath_code")
        except (json.JSONDecodeError, TypeError):
            # Maybe it's a dotted CATH code directly
            if re.match(r"^\d+\.\d+\.\d+\.\d+$", ground_truth.strip()):
                true_code = ground_truth.strip()

    if true_code is None:
        if detailed:
            return 0.0, {"error": "no_ground_truth"}
        return 0.0

    true_parts = true_code.split(".")
    if len(true_parts) != 4:
        if detailed:
            return 0.0, {"error": "invalid_cath_code"}
        return 0.0

    # Parse prediction — extract CATH code from generated text
    text = generated_text.strip()
    pred_code = None
    pred_parts = None

    # Try exact CATH code pattern
    cath_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", text)
    if cath_match:
        pred_code = cath_match.group(1)
        pred_parts = pred_code.split(".")

    # Fallback: try to match class by name
    pred_class_num = None
    if pred_parts is None:
        text_lower = text.lower()
        for num, name in _CATH_CLASS_NAMES.items():
            if name in text_lower:
                pred_class_num = num
                break
        # Also try common aliases
        if pred_class_num is None:
            if "all alpha" in text_lower or "all-alpha" in text_lower:
                pred_class_num = "1"
            elif "all beta" in text_lower or "all-beta" in text_lower:
                pred_class_num = "2"
            elif "alpha/beta" in text_lower or "alpha-beta" in text_lower:
                pred_class_num = "3"

    # Compute hierarchical reward
    level_names = ["class_match", "architecture_match", "topology_match", "homology_match"]
    level_matches = [False, False, False, False]

    if pred_parts is not None and len(pred_parts) == 4:
        # Full CATH code extracted — compare level by level
        for i in range(4):
            if pred_parts[i] == true_parts[i]:
                level_matches[i] = True
            else:
                break  # Stop at first mismatch (hierarchy)
    elif pred_class_num is not None:
        # Only class-level match possible
        if pred_class_num == true_parts[0]:
            level_matches[0] = True

    reward = sum(0.25 for m in level_matches if m)

    if detailed:
        metrics = {
            level_names[i]: level_matches[i]
            for i in range(4)
        }
        metrics["predicted_code"] = pred_code or "none"
        metrics["true_code"] = true_code
        metrics["predicted_class_num"] = pred_class_num
        metrics["levels_matched"] = sum(level_matches)
        return reward, metrics
    return reward


def compute_generic_reward(
    generated_text: str,
    ground_truth: str,
    task_type: Optional[str] = None,
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute a generic reward by attempting to match task type or using text similarity.

    This is a fallback reward function when the task type is not explicitly known.
    It attempts to detect the task type from the content and use the appropriate
    reward function.

    Args:
        generated_text: Model-generated text.
        ground_truth: Expected output/ground truth.
        task_type: Optional task type hint.
        detailed: If True, return (reward, metrics_dict) tuple.

    Returns:
        Reward value between 0 and 1, or (reward, metrics) if detailed=True.
    """
    # Try to detect task type from ground truth content
    if re.search(r"GO:\d{7}", ground_truth):
        return compute_go_reward(generated_text, ground_truth, detailed=detailed)

    if any(kw in ground_truth.lower() for kw in ["interact", "binding", "yes", "no"]):
        return compute_ppi_reward(generated_text, ground_truth, detailed=detailed)

    if any(kw in ground_truth.lower() for kw in ["kcal", "ddg", "stability"]):
        return compute_stability_reward(generated_text, ground_truth, detailed=detailed)

    # Fallback: simple text matching
    generated_text = _strip_think_tags(generated_text)
    gen_lower = generated_text.lower().strip()
    gt_lower = ground_truth.lower().strip()

    if gen_lower == gt_lower:
        reward = 1.0
    else:
        gen_words = set(gen_lower.split())
        gt_words = set(gt_lower.split())
        if not gt_words:
            reward = 0.0
        else:
            overlap = len(gen_words & gt_words)
            reward = overlap / len(gen_words | gt_words) if gen_words else 0.0

    if detailed:
        return reward, {"jaccard": reward, "task_type": "generic"}
    return reward
