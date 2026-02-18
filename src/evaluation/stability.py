"""
Protein Stability Prediction Evaluation

This module evaluates the model's ability to predict protein thermostability,
specifically the change in folding free energy (ddG) caused by point mutations.

ddG (Delta-Delta G) represents:
- Negative values: Stabilizing mutations (protein becomes more stable)
- Positive values: Destabilizing mutations (protein becomes less stable)
- Near zero: Neutral mutations (little effect on stability)

Standard classification thresholds:
- Stabilizing: ddG < -1.0 kcal/mol
- Neutral: -1.0 <= ddG <= 1.0 kcal/mol
- Destabilizing: ddG > 1.0 kcal/mol

Metrics computed:
- Regression metrics: Pearson, Spearman, RMSE, MAE, R^2
- Classification metrics: Accuracy, F1 (macro/per-class), Confusion Matrix

Supported data formats:
- JSON: List of mutation records with ddG values
- CSV: Comma-separated values with mutation information
"""

import csv
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

log = logging.getLogger(__name__)

# Regex patterns for parsing stability predictions
DDG_PATTERN = re.compile(
    r"(?:ddG|delta\s*delta\s*G|ΔΔG|\u0394\u0394G|change\s+in\s+(?:free\s+)?energy)"
    r"[:\s]*([+-]?\d+\.?\d*)",
    re.IGNORECASE
)
KCAL_PATTERN = re.compile(r"([+-]?\d+\.?\d*)\s*(?:kcal/?mol|kcal)", re.IGNORECASE)
NUMERIC_PATTERN = re.compile(r"([+-]?\d+\.?\d*)\s*(?:kcal|kJ)?/?(?:mol)?", re.IGNORECASE)

# Classification patterns
STABILIZING_PATTERN = re.compile(r"\b(stabiliz(?:ing|es?|ation)|more\s+stable|increases?\s+stability)\b", re.IGNORECASE)
DESTABILIZING_PATTERN = re.compile(r"\b(destabiliz(?:ing|es?|ation)|less\s+stable|decreases?\s+stability|unfold(?:ing|s)?)\b", re.IGNORECASE)
NEUTRAL_PATTERN = re.compile(r"\b(neutral|no\s+(?:significant\s+)?effect|negligible|minimal)\b", re.IGNORECASE)

# Stability classification thresholds (kcal/mol)
STABILIZING_THRESHOLD = -1.0
DESTABILIZING_THRESHOLD = 1.0

# Stability class names
STABILITY_CLASSES = ["stabilizing", "neutral", "destabilizing"]


@dataclass
class StabilityTestSample:
    """A test sample for protein stability prediction.

    Attributes:
        protein_id: Unique identifier for the protein.
        sequence: Mutant protein sequence.
        wild_type_sequence: Original wild-type sequence (optional).
        mutation: Mutation notation (e.g., "A123G" means Alanine at position 123 to Glycine).
        ddg_value: Experimentally measured ddG value in kcal/mol.
        stability_class: Optional classification ("stabilizing", "neutral", "destabilizing").
        pdb_id: PDB identifier if available.
        chain: Chain identifier if from PDB.
        description: Additional description or notes.
        source: Data source (e.g., "Megascale", "S669", "FireProtDB").
    """

    protein_id: str
    sequence: str
    wild_type_sequence: Optional[str]
    mutation: str
    ddg_value: float
    stability_class: Optional[str] = None
    pdb_id: Optional[str] = None
    chain: Optional[str] = None
    description: str = ""
    source: str = ""

    def __post_init__(self):
        """Automatically assign stability class if not provided."""
        if self.stability_class is None:
            self.stability_class = classify_ddg(self.ddg_value)
        # Validate stability class
        if self.stability_class not in STABILITY_CLASSES:
            raise ValueError(
                f"stability_class must be one of {STABILITY_CLASSES}, got {self.stability_class}"
            )


@dataclass
class StabilityPredictionResult:
    """Result of stability prediction for a single sample.

    Attributes:
        predicted_ddg: Predicted ddG value in kcal/mol (None if not extractable).
        predicted_class: Predicted stability class.
        ground_truth_ddg: Ground truth ddG value.
        ground_truth_class: Ground truth stability class.
        generated_text: Raw generated text from the model.
        protein_id: Protein identifier.
        mutation: Mutation notation.
        parse_success: Whether ddG value was successfully parsed from text.
    """

    predicted_ddg: Optional[float]
    predicted_class: str
    ground_truth_ddg: float
    ground_truth_class: str
    generated_text: str
    protein_id: str = ""
    mutation: str = ""
    parse_success: bool = True


def classify_ddg(ddg_value: float) -> str:
    """
    Classify ddG value into stability category.

    Args:
        ddg_value: ddG value in kcal/mol.

    Returns:
        Stability class: "stabilizing", "neutral", or "destabilizing".

    Examples:
        >>> classify_ddg(-2.5)
        'stabilizing'
        >>> classify_ddg(0.5)
        'neutral'
        >>> classify_ddg(3.0)
        'destabilizing'
    """
    if ddg_value < STABILIZING_THRESHOLD:
        return "stabilizing"
    elif ddg_value > DESTABILIZING_THRESHOLD:
        return "destabilizing"
    else:
        return "neutral"


def parse_stability_prediction(text: str) -> Tuple[Optional[float], str]:
    """
    Extract ddG value and stability class from generated text.

    The function looks for:
    1. Explicit ddG values with units
    2. Classification keywords (stabilizing/destabilizing/neutral)
    3. Numeric values in context

    Args:
        text: Generated text that may contain stability prediction.

    Returns:
        Tuple of (predicted_ddg, predicted_class).
        - predicted_ddg: Extracted ddG value or None if not found
        - predicted_class: Inferred stability class

    Examples:
        >>> parse_stability_prediction("The ddG is -2.5 kcal/mol, making this stabilizing.")
        (-2.5, 'stabilizing')
        >>> parse_stability_prediction("This mutation is destabilizing with ddG = 3.2.")
        (3.2, 'destabilizing')
        >>> parse_stability_prediction("Neutral effect on stability.")
        (None, 'neutral')
    """
    if not text:
        return None, "neutral"

    text = text.strip()
    predicted_ddg = None

    # Try to extract ddG value using various patterns
    # Pattern 1: Explicit "ddG" or "delta delta G" notation
    ddg_match = DDG_PATTERN.search(text)
    if ddg_match:
        try:
            predicted_ddg = float(ddg_match.group(1))
        except ValueError:
            pass

    # Pattern 2: Value with kcal/mol units
    if predicted_ddg is None:
        kcal_match = KCAL_PATTERN.search(text)
        if kcal_match:
            try:
                predicted_ddg = float(kcal_match.group(1))
            except ValueError:
                pass

    # Pattern 3: Look for numeric values in common contexts
    if predicted_ddg is None:
        # Look for patterns like "value of X" or "approximately X" or "is X"
        value_patterns = [
            re.compile(r"(?:value|prediction|result|estimate)[^0-9]*([+-]?\d+\.?\d*)", re.IGNORECASE),
            re.compile(r"(?:is|equals?|=)[^0-9]*([+-]?\d+\.?\d*)", re.IGNORECASE),
            re.compile(r"approximately[^0-9]*([+-]?\d+\.?\d*)", re.IGNORECASE),
        ]
        for pattern in value_patterns:
            match = pattern.search(text)
            if match:
                try:
                    val = float(match.group(1))
                    # Sanity check: ddG values typically range from -10 to +15 kcal/mol
                    if -15 <= val <= 20:
                        predicted_ddg = val
                        break
                except ValueError:
                    pass

    # Determine stability class
    # First check if we have a ddG value
    if predicted_ddg is not None:
        predicted_class = classify_ddg(predicted_ddg)
    else:
        # Try to infer from keywords
        predicted_class = _infer_class_from_text(text)

    return predicted_ddg, predicted_class


def _infer_class_from_text(text: str) -> str:
    """Infer stability class from text keywords."""
    # Check for explicit class keywords
    stabilizing_match = STABILIZING_PATTERN.search(text)
    destabilizing_match = DESTABILIZING_PATTERN.search(text)
    neutral_match = NEUTRAL_PATTERN.search(text)

    # Count matches and positions
    matches = []
    if stabilizing_match:
        matches.append(("stabilizing", stabilizing_match.start()))
    if destabilizing_match:
        matches.append(("destabilizing", destabilizing_match.start()))
    if neutral_match:
        matches.append(("neutral", neutral_match.start()))

    if not matches:
        return "neutral"  # Default

    # If multiple matches, prefer the first one in the text
    matches.sort(key=lambda x: x[1])
    return matches[0][0]


def load_stability_test_dataset(
    cfg: DictConfig,
    max_samples: Optional[int] = None,
) -> List[StabilityTestSample]:
    """
    Load stability test dataset.

    This function supports multiple data sources:
    1. Pre-processed JSON file with stability data
    2. CSV file with mutation information and ddG values
    3. Demo dataset for testing

    Args:
        cfg: Configuration with dataset settings.
        max_samples: Maximum number of samples to load (for testing).

    Returns:
        List of test samples with stability annotations.
    """
    # Extract dataset config
    dataset_cfg = cfg.get("dataset", {})
    data_path = dataset_cfg.get("path", None)
    data_format = dataset_cfg.get("format", "json")

    # Check for max_samples in config
    if max_samples is None:
        max_samples = cfg.get("evaluation", {}).get("max_samples", None)

    samples = []

    if data_path and Path(data_path).exists():
        # Load from local file
        if data_format == "json":
            samples = _load_json_dataset(data_path, max_samples)
        elif data_format == "csv":
            samples = _load_csv_dataset(data_path, max_samples)
        else:
            log.warning(f"Unknown data format: {data_format}, using demo data")
            samples = _create_demo_dataset(max_samples or 20)
    else:
        # Use demo dataset for testing
        log.info("No dataset path provided, using demo dataset")
        samples = _create_demo_dataset(max_samples or 20)

    log.info(f"Loaded {len(samples)} stability test samples")

    # Log class distribution
    class_counts = {c: 0 for c in STABILITY_CLASSES}
    for s in samples:
        class_counts[s.stability_class] += 1
    log.info(f"Class distribution: {class_counts}")

    # Log ddG statistics
    ddg_values = [s.ddg_value for s in samples]
    log.info(
        f"ddG statistics: min={min(ddg_values):.2f}, max={max(ddg_values):.2f}, "
        f"mean={np.mean(ddg_values):.2f}, std={np.std(ddg_values):.2f}"
    )

    return samples


def _load_json_dataset(path: str, max_samples: Optional[int]) -> List[StabilityTestSample]:
    """Load stability dataset from JSON file."""
    samples = []

    with open(path, "r") as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "mutations" in data:
        entries = data["mutations"]
    elif isinstance(data, dict) and "samples" in data:
        entries = data["samples"]
    else:
        entries = [data]

    for entry in entries:
        if max_samples and len(samples) >= max_samples:
            break

        # Extract fields with fallbacks
        protein_id = entry.get("protein_id", entry.get("id", entry.get("pdb_id", f"protein_{len(samples)}")))
        sequence = entry.get("sequence", entry.get("mutant_sequence", entry.get("seq", "")))
        wild_type_sequence = entry.get("wild_type_sequence", entry.get("wt_sequence", entry.get("wt_seq", None)))
        mutation = entry.get("mutation", entry.get("variant", entry.get("substitution", "")))

        # Get ddG value
        ddg_value = entry.get("ddg_value", entry.get("ddg", entry.get("ddG", entry.get("delta_delta_g", None))))
        if ddg_value is None:
            log.warning(f"Skipping entry without ddG value: {protein_id}")
            continue
        ddg_value = float(ddg_value)

        stability_class = entry.get("stability_class", entry.get("class", None))
        if stability_class and stability_class not in STABILITY_CLASSES:
            stability_class = None  # Will be auto-assigned

        if sequence and mutation:
            samples.append(StabilityTestSample(
                protein_id=protein_id,
                sequence=sequence,
                wild_type_sequence=wild_type_sequence,
                mutation=mutation,
                ddg_value=ddg_value,
                stability_class=stability_class,
                pdb_id=entry.get("pdb_id", None),
                chain=entry.get("chain", None),
                description=entry.get("description", ""),
                source=entry.get("source", "json"),
            ))

    return samples


def _load_csv_dataset(path: str, max_samples: Optional[int]) -> List[StabilityTestSample]:
    """Load stability dataset from CSV file."""
    samples = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if max_samples and len(samples) >= max_samples:
                break

            # Map common column names
            protein_id = row.get("protein_id", row.get("id", row.get("pdb_id", f"protein_{len(samples)}")))
            sequence = row.get("sequence", row.get("mutant_sequence", row.get("seq", "")))
            wild_type_sequence = row.get("wild_type_sequence", row.get("wt_sequence", None))
            mutation = row.get("mutation", row.get("variant", row.get("substitution", "")))

            # Get ddG value
            ddg_str = row.get("ddg_value", row.get("ddg", row.get("ddG", row.get("delta_delta_g", ""))))
            if not ddg_str:
                continue
            try:
                ddg_value = float(ddg_str)
            except ValueError:
                log.warning(f"Could not parse ddG value: {ddg_str}")
                continue

            stability_class = row.get("stability_class", row.get("class", None))
            if stability_class and stability_class not in STABILITY_CLASSES:
                stability_class = None

            if sequence and mutation:
                samples.append(StabilityTestSample(
                    protein_id=protein_id,
                    sequence=sequence,
                    wild_type_sequence=wild_type_sequence,
                    mutation=mutation,
                    ddg_value=ddg_value,
                    stability_class=stability_class,
                    pdb_id=row.get("pdb_id", None),
                    chain=row.get("chain", None),
                    description=row.get("description", ""),
                    source="csv",
                ))

    return samples


def _create_demo_dataset(num_samples: int = 20) -> List[StabilityTestSample]:
    """
    Create a demo dataset for testing purposes.

    Contains mutations with known stability effects from well-characterized proteins:
    - Barnase (1BNI): Well-studied ribonuclease
    - T4 Lysozyme (2LZM): Extensively characterized for stability
    - Staphylococcal Nuclease (1STN): Classic stability studies
    - Myoglobin (1MBN): Hemoglobin family protein

    ddG values are based on experimental data from ProTherm and similar databases.
    """
    # Demo protein sequences (shortened for brevity)
    demo_proteins = {
        "barnase": {
            "pdb_id": "1BNI",
            "wt_seq": "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR",
        },
        "t4_lysozyme": {
            "pdb_id": "2LZM",
            "wt_seq": "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
        },
        "sn": {
            "pdb_id": "1STN",
            "wt_seq": "ATSTKKLHKEPATLIKAIDGDTVKLMYKGQPMTFRLLLVDTPETKHPKKGVEKYGPEASAFTKKMVENAKKIEVEFDKGQRTDKYGRGLAYIYADGKMVNEALVRQGLAKVAYVYKPNNTHEQHLRKSEAQAKKEKLNIWSEDNADSGQ",
        },
        "myoglobin": {
            "pdb_id": "1MBN",
            "wt_seq": "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG",
        },
    }

    # Known mutations and their approximate ddG values (from literature)
    # Format: (protein_key, mutation, ddg_value, description)
    demo_mutations = [
        # Barnase mutations
        ("barnase", "I88V", -0.8, "Hydrophobic core, slightly stabilizing"),
        ("barnase", "I88A", 2.3, "Hydrophobic core disruption, destabilizing"),
        ("barnase", "Y17A", 4.1, "Aromatic residue removal, strongly destabilizing"),
        ("barnase", "H102A", 3.5, "Active site, destabilizing"),
        ("barnase", "K27A", 0.5, "Surface charge removal, neutral"),
        # T4 Lysozyme mutations
        ("t4_lysozyme", "L99A", 5.1, "Cavity creation, destabilizing"),
        ("t4_lysozyme", "A98V", -1.5, "Cavity filling, stabilizing"),
        ("t4_lysozyme", "T152S", 0.3, "Conservative, neutral"),
        ("t4_lysozyme", "M102L", -0.6, "Hydrophobic, slightly stabilizing"),
        ("t4_lysozyme", "F153A", 3.8, "Aromatic removal, destabilizing"),
        ("t4_lysozyme", "I3A", 2.8, "N-terminal destabilization"),
        # Staphylococcal Nuclease mutations
        ("sn", "V66L", -1.2, "Improved packing, stabilizing"),
        ("sn", "V66A", 2.1, "Cavity creation, destabilizing"),
        ("sn", "G88V", -0.9, "Cavity filling, stabilizing"),
        ("sn", "P117G", 1.5, "Flexibility change, destabilizing"),
        ("sn", "A58G", 0.2, "Surface, neutral"),
        # Myoglobin mutations
        ("myoglobin", "V68A", 1.8, "Heme pocket, destabilizing"),
        ("myoglobin", "L29A", 2.5, "Hydrophobic core, destabilizing"),
        ("myoglobin", "H64A", 1.2, "Distal histidine, destabilizing"),
        ("myoglobin", "F46A", 3.2, "Aromatic packing, destabilizing"),
        ("myoglobin", "I107V", 0.4, "Conservative, neutral"),
    ]

    samples = []
    for protein_key, mutation, ddg_value, description in demo_mutations[:num_samples]:
        protein = demo_proteins[protein_key]
        wt_seq = protein["wt_seq"]

        # Apply mutation to create mutant sequence
        mutant_seq = _apply_mutation(wt_seq, mutation)

        samples.append(StabilityTestSample(
            protein_id=f"{protein['pdb_id']}_{mutation}",
            sequence=mutant_seq,
            wild_type_sequence=wt_seq,
            mutation=mutation,
            ddg_value=ddg_value,
            pdb_id=protein["pdb_id"],
            description=description,
            source="demo",
        ))

    return samples


def _apply_mutation(sequence: str, mutation: str) -> str:
    """
    Apply a point mutation to a sequence.

    Args:
        sequence: Original amino acid sequence.
        mutation: Mutation in format like "A123G" (original AA, position, new AA).

    Returns:
        Mutated sequence.
    """
    # Parse mutation format: e.g., "A123G"
    match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
    if not match:
        return sequence  # Return unchanged if can't parse

    original_aa = match.group(1)
    position = int(match.group(2))
    new_aa = match.group(3)

    # Convert to 0-indexed
    idx = position - 1

    # Validate
    if idx < 0 or idx >= len(sequence):
        log.warning(f"Position {position} out of range for sequence length {len(sequence)}")
        return sequence

    if sequence[idx] != original_aa:
        log.warning(f"Expected {original_aa} at position {position}, found {sequence[idx]}")

    # Apply mutation
    return sequence[:idx] + new_aa + sequence[idx + 1:]


def create_stability_prompt(
    sequence: str,
    mutation: str,
    wild_type_sequence: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> str:
    """
    Create a prompt for stability prediction.

    Args:
        sequence: Mutant protein sequence.
        mutation: Mutation notation (e.g., "A123G").
        wild_type_sequence: Optional wild-type sequence.
        prompt_template: Optional custom prompt template with placeholders.

    Returns:
        Formatted prompt string.
    """
    if prompt_template:
        return prompt_template.format(
            sequence=sequence,
            mutation=mutation,
            wild_type_sequence=wild_type_sequence or sequence,
        )

    # Parse mutation for readable format
    aa_names = {
        'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartate',
        'C': 'Cysteine', 'E': 'Glutamate', 'Q': 'Glutamine', 'G': 'Glycine',
        'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
        'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
        'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine',
    }

    mutation_match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
    if mutation_match:
        orig_aa = mutation_match.group(1)
        pos = mutation_match.group(2)
        new_aa = mutation_match.group(3)
        orig_name = aa_names.get(orig_aa, orig_aa)
        new_name = aa_names.get(new_aa, new_aa)
        mutation_description = f"{mutation} ({orig_name} to {new_name} at position {pos})"
    else:
        mutation_description = mutation

    # Build prompt
    if wild_type_sequence:
        prompt = f"""Wild-type protein sequence: {wild_type_sequence}

Mutation: {mutation_description}

Predict the change in protein stability (ddG) caused by this mutation.
Provide the ddG value in kcal/mol and classify as stabilizing (ddG < -1), neutral (-1 <= ddG <= 1), or destabilizing (ddG > 1).
"""
    else:
        prompt = f"""Protein sequence: {sequence}

Mutation: {mutation_description}

Predict the change in protein stability (ddG) caused by this mutation.
Provide the ddG value in kcal/mol and classify as stabilizing (ddG < -1), neutral (-1 <= ddG <= 1), or destabilizing (ddG > 1).
"""

    return prompt


def compute_stability_metrics(
    predictions: List[StabilityPredictionResult],
    include_per_class: bool = True,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for stability predictions.

    Regression metrics (for samples with predicted ddG values):
    - pearson: Pearson correlation coefficient
    - spearman: Spearman rank correlation
    - rmse: Root mean squared error
    - mae: Mean absolute error
    - r2: Coefficient of determination

    Classification metrics (for all samples):
    - accuracy: Overall classification accuracy
    - f1_macro: Macro-averaged F1 score
    - f1_stabilizing/f1_neutral/f1_destabilizing: Per-class F1
    - confusion_matrix: 3x3 confusion matrix

    Args:
        predictions: List of prediction results.
        include_per_class: Whether to include per-class metrics.

    Returns:
        Dictionary of metric names to values.
    """
    if not predictions:
        return {"error": "no_predictions"}

    metrics = {}

    # Separate predictions with and without ddG values
    predictions_with_ddg = [p for p in predictions if p.predicted_ddg is not None]

    # Classification metrics (for all predictions)
    y_true_class = [p.ground_truth_class for p in predictions]
    y_pred_class = [p.predicted_class for p in predictions]

    # Compute regression metrics if we have ddG predictions
    if predictions_with_ddg:
        y_true_ddg = np.array([p.ground_truth_ddg for p in predictions_with_ddg])
        y_pred_ddg = np.array([p.predicted_ddg for p in predictions_with_ddg])

        # Pearson correlation
        if SCIPY_AVAILABLE and len(y_true_ddg) >= 2:
            try:
                pearson_r, pearson_p = pearsonr(y_true_ddg, y_pred_ddg)
                metrics["pearson"] = pearson_r
                metrics["pearson_pvalue"] = pearson_p
            except Exception as e:
                log.warning(f"Could not compute Pearson correlation: {e}")
                metrics["pearson"] = 0.0
        else:
            metrics["pearson"] = _compute_pearson_basic(y_true_ddg, y_pred_ddg)

        # Spearman correlation
        if SCIPY_AVAILABLE and len(y_true_ddg) >= 2:
            try:
                spearman_r, spearman_p = spearmanr(y_true_ddg, y_pred_ddg)
                metrics["spearman"] = spearman_r
                metrics["spearman_pvalue"] = spearman_p
            except Exception as e:
                log.warning(f"Could not compute Spearman correlation: {e}")
                metrics["spearman"] = 0.0
        else:
            metrics["spearman"] = _compute_spearman_basic(y_true_ddg, y_pred_ddg)

        # RMSE
        if SKLEARN_AVAILABLE:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true_ddg, y_pred_ddg))
        else:
            metrics["rmse"] = np.sqrt(np.mean((y_true_ddg - y_pred_ddg) ** 2))

        # MAE
        if SKLEARN_AVAILABLE:
            metrics["mae"] = mean_absolute_error(y_true_ddg, y_pred_ddg)
        else:
            metrics["mae"] = np.mean(np.abs(y_true_ddg - y_pred_ddg))

        # R^2
        if SKLEARN_AVAILABLE:
            metrics["r2"] = r2_score(y_true_ddg, y_pred_ddg)
        else:
            metrics["r2"] = _compute_r2_basic(y_true_ddg, y_pred_ddg)

        metrics["num_samples_with_ddg"] = len(predictions_with_ddg)
        metrics["ddg_parse_success_rate"] = len(predictions_with_ddg) / len(predictions)
    else:
        metrics["pearson"] = float("nan")
        metrics["spearman"] = float("nan")
        metrics["rmse"] = float("nan")
        metrics["mae"] = float("nan")
        metrics["r2"] = float("nan")
        metrics["num_samples_with_ddg"] = 0
        metrics["ddg_parse_success_rate"] = 0.0

    # Classification metrics
    if SKLEARN_AVAILABLE:
        # Convert to numeric labels for sklearn
        class_to_idx = {c: i for i, c in enumerate(STABILITY_CLASSES)}
        y_true_idx = [class_to_idx[c] for c in y_true_class]
        y_pred_idx = [class_to_idx[c] for c in y_pred_class]

        metrics["accuracy"] = accuracy_score(y_true_idx, y_pred_idx)
        metrics["f1_macro"] = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
        metrics["f1_micro"] = f1_score(y_true_idx, y_pred_idx, average="micro", zero_division=0)

        # Per-class F1
        if include_per_class:
            f1_per_class = f1_score(y_true_idx, y_pred_idx, average=None, labels=[0, 1, 2], zero_division=0)
            for cls_name, f1_val in zip(STABILITY_CLASSES, f1_per_class):
                metrics[f"f1_{cls_name}"] = f1_val

        # Confusion matrix
        try:
            cm = confusion_matrix(y_true_idx, y_pred_idx, labels=[0, 1, 2])
            # Store as flattened for easy access
            metrics["confusion_matrix"] = cm.tolist()
            # Also store individual elements for logging
            for i, true_class in enumerate(STABILITY_CLASSES):
                for j, pred_class in enumerate(STABILITY_CLASSES):
                    metrics[f"cm_{true_class}_pred_{pred_class}"] = int(cm[i, j])
        except Exception as e:
            log.warning(f"Could not compute confusion matrix: {e}")
    else:
        metrics.update(_compute_classification_basic(y_true_class, y_pred_class))

    # Sample statistics
    metrics["num_samples"] = len(predictions)
    class_distribution = {}
    for cls in STABILITY_CLASSES:
        class_distribution[f"num_{cls}"] = sum(1 for p in predictions if p.ground_truth_class == cls)
    metrics.update(class_distribution)

    # Parse success rate
    metrics["num_parse_failures"] = sum(1 for p in predictions if not p.parse_success)

    return metrics


def _compute_pearson_basic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation without scipy."""
    if len(y_true) < 2:
        return 0.0

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    denominator = np.sqrt(np.sum((y_true - mean_true) ** 2) * np.sum((y_pred - mean_pred) ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _compute_spearman_basic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation without scipy."""
    if len(y_true) < 2:
        return 0.0

    # Convert to ranks
    def rankdata(x):
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        return ranks + 1  # 1-indexed ranks

    rank_true = rankdata(y_true)
    rank_pred = rankdata(y_pred)

    return _compute_pearson_basic(rank_true, rank_pred)


def _compute_r2_basic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 without sklearn."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def _compute_classification_basic(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute basic classification metrics without sklearn."""
    metrics = {}

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    metrics["accuracy"] = correct / len(y_true) if y_true else 0.0

    # Per-class precision, recall, F1
    f1_scores = []
    for cls in STABILITY_CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"f1_{cls}"] = f1
        f1_scores.append(f1)

    metrics["f1_macro"] = np.mean(f1_scores)

    return metrics


def evaluate_stability(
    cfg: DictConfig,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate protein stability prediction.

    This function:
    1. Loads the model from checkpoint
    2. Loads the stability test dataset
    3. Generates predictions for each mutation
    4. Parses ddG values and classes from generated text
    5. Computes evaluation metrics

    Args:
        cfg: Hydra configuration containing:
            - model: Model configuration
            - dataset: Dataset configuration (path, format, max_samples)
            - evaluation: Evaluation settings (batch_size, max_new_tokens)
            - logging: Logging settings (wandb, tensorboard, save_results)

        checkpoint_path: Path to model checkpoint.

    Returns:
        Dictionary of metric names to values.
    """
    log.info("Evaluating stability prediction...")

    # Import model here to avoid circular imports
    from src.models.multimodal_llm import ProteinLLM

    # Load model
    if checkpoint_path:
        log.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = ProteinLLM.from_pretrained(checkpoint_path)
    else:
        log.info("Creating model from config (no checkpoint provided)")
        model = ProteinLLM.from_config(cfg)

    model.eval()

    # Get evaluation settings
    eval_cfg = cfg.get("evaluation", {})
    batch_size = eval_cfg.get("batch_size", 1)
    max_new_tokens = eval_cfg.get("max_new_tokens", 256)
    max_samples = eval_cfg.get("max_samples", None)
    prompt_template = eval_cfg.get("prompt_template", None)

    # Load test dataset
    test_samples = load_stability_test_dataset(cfg, max_samples=max_samples)

    if not test_samples:
        log.error("No test samples loaded")
        return {"error": "no_test_samples"}

    log.info(f"Evaluating on {len(test_samples)} samples")

    # Generate predictions
    predictions = []

    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i + batch_size]

        # Prepare prompts
        prompts = [
            create_stability_prompt(
                sample.sequence,
                sample.mutation,
                sample.wild_type_sequence,
                prompt_template,
            )
            for sample in batch
        ]

        # Prepare sequences
        sequences = [sample.wild_type_sequence or sample.sequence for sample in batch]

        # Generate responses
        try:
            generated_texts = model.generate(
                protein_sequences=sequences,
                prompt=prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for evaluation
                temperature=1.0,
            )
        except Exception as e:
            log.error(f"Generation failed for batch {i}: {e}")
            continue

        # Process results
        for sample, generated_text in zip(batch, generated_texts):
            predicted_ddg, predicted_class = parse_stability_prediction(generated_text)
            parse_success = predicted_ddg is not None

            result = StabilityPredictionResult(
                predicted_ddg=predicted_ddg,
                predicted_class=predicted_class,
                ground_truth_ddg=sample.ddg_value,
                ground_truth_class=sample.stability_class,
                generated_text=generated_text,
                protein_id=sample.protein_id,
                mutation=sample.mutation,
                parse_success=parse_success,
            )
            predictions.append(result)

        if (i + batch_size) % 10 == 0 or (i + batch_size) >= len(test_samples):
            log.info(f"Processed {min(i + batch_size, len(test_samples))}/{len(test_samples)} samples")

    # Compute metrics
    metrics = compute_stability_metrics(predictions)

    log.info("Stability Prediction Evaluation Results:")
    for key, value in sorted(metrics.items()):
        if key == "confusion_matrix":
            log.info(f"  {key}: {value}")
        elif isinstance(value, float):
            if math.isnan(value):
                log.info(f"  {key}: NaN")
            else:
                log.info(f"  {key}: {value:.4f}")
        else:
            log.info(f"  {key}: {value}")

    # Save results if configured
    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("save_results", False):
        _save_results(predictions, metrics, cfg)

    # Log to wandb if configured
    if logging_cfg.get("wandb", {}).get("enabled", False):
        _log_to_wandb(metrics, predictions, cfg)

    # Log to tensorboard if configured
    if logging_cfg.get("tensorboard", {}).get("enabled", False):
        _log_to_tensorboard(metrics, cfg)

    return metrics


def _save_results(
    predictions: List[StabilityPredictionResult],
    metrics: Dict[str, float],
    cfg: DictConfig,
) -> None:
    """Save evaluation results to JSON file."""
    output_dir = cfg.get("logging", {}).get("output_dir", "./outputs")
    output_path = Path(output_dir) / "stability_prediction_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": {k: v if not isinstance(v, float) or not math.isnan(v) else None
                    for k, v in metrics.items()},
        "predictions": [
            {
                "protein_id": p.protein_id,
                "mutation": p.mutation,
                "predicted_ddg": p.predicted_ddg,
                "predicted_class": p.predicted_class,
                "ground_truth_ddg": p.ground_truth_ddg,
                "ground_truth_class": p.ground_truth_class,
                "parse_success": p.parse_success,
                "generated_text": p.generated_text,
            }
            for p in predictions
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved to {output_path}")


def _log_to_wandb(
    metrics: Dict[str, float],
    predictions: List[StabilityPredictionResult],
    cfg: DictConfig,
) -> None:
    """Log metrics to Weights & Biases."""
    try:
        import wandb

        # Filter out NaN values for wandb
        wandb_metrics = {
            k: v for k, v in metrics.items()
            if not (isinstance(v, float) and math.isnan(v)) and k != "confusion_matrix"
        }

        # Log metrics
        wandb.log({"stability_prediction": wandb_metrics})

        # Log a sample table
        table_data = []
        for p in predictions[:30]:  # Limit to 30 samples
            table_data.append([
                p.protein_id,
                p.mutation,
                p.predicted_ddg,
                p.ground_truth_ddg,
                p.predicted_class,
                p.ground_truth_class,
                "Correct" if p.predicted_class == p.ground_truth_class else "Wrong",
                p.generated_text[:200],
            ])

        table = wandb.Table(
            columns=[
                "protein_id", "mutation", "predicted_ddg", "ground_truth_ddg",
                "predicted_class", "ground_truth_class", "class_status", "generated_text"
            ],
            data=table_data,
        )
        wandb.log({"stability_prediction_samples": table})

        # Log confusion matrix as heatmap
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            wandb.log({
                "stability_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=[p.ground_truth_class for p in predictions],
                    preds=[p.predicted_class for p in predictions],
                    class_names=STABILITY_CLASSES,
                )
            })

    except ImportError:
        log.warning("wandb not installed, skipping wandb logging")
    except Exception as e:
        log.warning(f"Failed to log to wandb: {e}")


def _log_to_tensorboard(
    metrics: Dict[str, float],
    cfg: DictConfig,
) -> None:
    """Log metrics to TensorBoard."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = cfg.get("logging", {}).get("tensorboard", {}).get("log_dir", "./runs")
        writer = SummaryWriter(log_dir=log_dir)

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                writer.add_scalar(f"stability_prediction/{key}", value)

        writer.close()

    except ImportError:
        log.warning("tensorboard not installed, skipping tensorboard logging")
    except Exception as e:
        log.warning(f"Failed to log to tensorboard: {e}")


# Utility functions for external use

def evaluate_stability_from_predictions(
    predictions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate stability predictions from a list of pre-computed predictions.

    Useful for evaluating predictions without loading the model.

    Args:
        predictions: List of dicts with keys:
            - predicted_ddg: Predicted ddG value (can be None)
            - predicted_class: Predicted stability class
            - ground_truth_ddg: Ground truth ddG value
            - ground_truth_class: Ground truth stability class (optional)

    Returns:
        Dictionary of metric names to values.
    """
    results = []
    for pred in predictions:
        ground_truth_ddg = pred.get("ground_truth_ddg", 0.0)
        ground_truth_class = pred.get("ground_truth_class", classify_ddg(ground_truth_ddg))

        results.append(StabilityPredictionResult(
            predicted_ddg=pred.get("predicted_ddg"),
            predicted_class=pred.get("predicted_class", "neutral"),
            ground_truth_ddg=ground_truth_ddg,
            ground_truth_class=ground_truth_class,
            generated_text=pred.get("generated_text", ""),
            protein_id=pred.get("protein_id", ""),
            mutation=pred.get("mutation", ""),
            parse_success=pred.get("predicted_ddg") is not None,
        ))

    return compute_stability_metrics(results)
