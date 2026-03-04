"""Reward functions for GRPO verifiable tasks.

Extracted from grpo_trainer.py — pure functions (text in → float out).

Functions:
    get_reward_function: Registry lookup for task-specific reward functions.
    compute_go_reward: F1 score of predicted GO terms.
    compute_ppi_reward: Binary PPI prediction accuracy.
    compute_stability_reward: ddG prediction accuracy (Gaussian decay).
    compute_esmfold_reward: ESMFold structural quality assessment.
    compute_proteinlm_bench_reward: Multiple-choice accuracy.
    compute_generic_reward: Fallback text-similarity reward.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

log = logging.getLogger(__name__)


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
        "proteinlm_bench": compute_proteinlm_bench_reward,
        "protein_lm_bench": compute_proteinlm_bench_reward,
        "multiple_choice": compute_proteinlm_bench_reward,
    }

    if task_lower not in reward_functions:
        supported = list(set(reward_functions.values()))
        raise ValueError(
            f"Unsupported task type: {task}. "
            f"Supported tasks: go_prediction, ppi, stability"
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


def compute_stability_reward(
    generated_text: str,
    ground_truth_ddg: Union[str, float],
    tolerance: float = 1.0,
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """Compute reward based on ddG prediction accuracy.

    Extracts predicted ddG (change in Gibbs free energy) from generated text
    and computes reward based on how close it is to ground truth. Uses a
    smooth reward function based on the error magnitude.

    Args:
        generated_text: Model-generated text containing stability prediction.
        ground_truth_ddg: True ddG value in kcal/mol. Can be string or float.
        tolerance: Error tolerance in kcal/mol for reward scaling. Default 1.0.
        detailed: If True, return (reward, metrics_dict) tuple.

    Returns:
        Reward between 0 and 1 based on prediction accuracy.
        If detailed=True, returns (reward, metrics) with MAE, RMSE, predicted, ground_truth.

    Example:
        >>> compute_stability_reward("The predicted ddG is 2.5 kcal/mol", 2.3)
        0.96  # Small error, high reward
        >>> compute_stability_reward("ddG = -1.0", 3.0)
        0.02  # Large error, low reward
    """
    # Parse ground truth
    if isinstance(ground_truth_ddg, str):
        try:
            numbers = re.findall(r"-?\d+\.?\d*", ground_truth_ddg)
            if numbers:
                gt_value = float(numbers[0])
            else:
                log.warning(f"Could not parse ground truth ddG: {ground_truth_ddg}")
                if detailed:
                    return 0.0, {"mae": float("nan"), "predicted": None,
                                 "ground_truth": ground_truth_ddg, "parsed": False}
                return 0.0
        except ValueError:
            log.warning(f"Could not parse ground truth ddG: {ground_truth_ddg}")
            if detailed:
                return 0.0, {"mae": float("nan"), "predicted": None,
                             "ground_truth": ground_truth_ddg, "parsed": False}
            return 0.0
    else:
        gt_value = float(ground_truth_ddg)

    # Extract predicted ddG from generated text
    patterns = [
        r"ddG\s*[=:]\s*(-?\d+\.?\d*)",
        r"ΔΔG\s*[=:]\s*(-?\d+\.?\d*)",
        r"delta\s*G\s*[=:]\s*(-?\d+\.?\d*)",
        r"(-?\d+\.?\d*)\s*kcal\s*/?\s*mol",
        r"stability[:\s]+(-?\d+\.?\d*)",
        r"change[:\s]+(-?\d+\.?\d*)",
        r"predicted[:\s]+(-?\d+\.?\d*)",
    ]

    pred_value = None
    for pattern in patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            try:
                pred_value = float(match.group(1))
                break
            except ValueError:
                continue

    # If no pattern matched, try to find any number
    if pred_value is None:
        numbers = re.findall(r"-?\d+\.?\d+", generated_text)
        if numbers:
            for num_str in numbers:
                num = float(num_str)
                if -20 <= num <= 20:
                    pred_value = num
                    break

    if pred_value is None:
        if detailed:
            return 0.0, {"mae": float("nan"), "predicted": None,
                         "ground_truth": gt_value, "parsed": False}
        return 0.0

    # Compute reward based on error
    error = abs(pred_value - gt_value)
    reward = float(torch.exp(torch.tensor(-error**2 / (2 * tolerance**2))))

    if detailed:
        return reward, {
            "mae": error,
            "squared_error": error ** 2,
            "predicted": pred_value,
            "ground_truth": gt_value,
            "parsed": True,
        }
    return reward


def compute_esmfold_reward(
    generated_text: str,
    protein_sequence_or_metrics: str,
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute reward based on ESMFold structural quality assessment.

    Supports two modes:
    1. Live ESMFold: pass a protein sequence string, folds with ESMFold
    2. Pre-computed: pass a JSON string with {"plddt": X, "ptm": Y} from
       metadata (e.g., from AlphaFold DB download). Instant, no GPU needed.

    Reward components (sum to max 1.0):
    1. Quality claim alignment (0.4): correct "well-folded"/"disordered" claim
    2. Numerical pLDDT prediction accuracy (0.3): Gaussian decay on error
    3. Fold quality category match (0.3): high/medium/low correct

    Args:
        generated_text: Model-generated text about the protein.
        protein_sequence_or_metrics: Either a protein sequence (for live
            ESMFold) or a JSON string with pre-computed {"plddt", "ptm"}.
        detailed: If True, return (reward, metrics) tuple.

    Returns:
        Reward between 0 and 1, or (reward, metrics) if detailed=True.
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

    text_lower = generated_text.lower()
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

    claims_well_folded = any(term in text_lower for term in well_folded_terms)
    claims_disordered = any(term in text_lower for term in disordered_terms)

    quality_reward = 0.0
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
    if plddt > 80:
        true_category = "high"
    elif plddt > 50:
        true_category = "medium"
    else:
        true_category = "low"

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

    if detailed:
        return reward, {
            "plddt": plddt,
            "ptm": ptm,
            "predicted_plddt": pred_plddt,
            "true_category": true_category,
            "claims_well_folded": claims_well_folded,
            "claims_disordered": claims_disordered,
            **component_scores,
        }
    return reward


def compute_proteinlm_bench_reward(
    generated_text: str,
    ground_truth: str,
    detailed: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Compute multiple-choice accuracy reward for ProteinLMBench.

    Returns 1.0 if the predicted option matches the correct answer, 0.0 otherwise.
    The ground truth should be in "option N" format (e.g., "option 3").

    Args:
        generated_text: Model-generated text.
        ground_truth: Correct answer in "option N" format, or just "N".
        detailed: If True, return (reward, metrics_dict) tuple.

    Returns:
        1.0 for correct, 0.0 for incorrect.
    """
    from src.evaluation.proteinlm_bench import parse_mc_answer

    # Parse correct answer index from ground truth
    gt = ground_truth.strip()
    gt_match = re.match(r'option\s+(\d+)', gt, re.IGNORECASE)
    if gt_match:
        correct_idx = int(gt_match.group(1)) - 1
    elif gt.isdigit():
        correct_idx = int(gt) - 1
    else:
        correct_idx = -1

    # Parse predicted answer (assume up to 10 options)
    _, pred_idx = parse_mc_answer(generated_text, num_options=10)

    is_correct = (pred_idx >= 0 and pred_idx == correct_idx)
    reward = 1.0 if is_correct else 0.0

    if detailed:
        return reward, {
            "predicted_index": pred_idx,
            "correct_index": correct_idx,
            "is_correct": is_correct,
            "parsed": pred_idx >= 0,
        }
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
