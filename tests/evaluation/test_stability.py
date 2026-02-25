"""
Tests for Protein Stability Prediction Evaluation Module.

Tests cover:
- parse_stability_prediction: Extracting ddG values and stability class from text
- classify_ddg: Classifying ddG values into stability categories
- StabilityTestSample: Data class for test samples
- StabilityPredictionResult: Data class for prediction results
- compute_stability_metrics: Computing evaluation metrics
- load_stability_test_dataset: Loading test datasets
- create_stability_prompt: Creating prompts for stability prediction
- evaluate_stability: End-to-end evaluation with mocked model
"""

import csv
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.evaluation.stability import (
    STABILITY_CLASSES,
    StabilityPredictionResult,
    StabilityTestSample,
    _apply_mutation,
    _compute_classification_basic,
    _compute_pearson_basic,
    _compute_r2_basic,
    _compute_spearman_basic,
    _create_demo_dataset,
    _infer_class_from_text,
    _load_csv_dataset,
    classify_ddg,
    compute_stability_metrics,
    create_stability_prompt,
    evaluate_stability,
    evaluate_stability_from_predictions,
    load_stability_test_dataset,
    parse_stability_prediction,
)


class TestClassifyDDG:
    """Tests for classify_ddg function."""

    def test_classify_stabilizing(self):
        """Test classifying stabilizing mutations (ddG < -1)."""
        assert classify_ddg(-2.5) == "stabilizing"
        assert classify_ddg(-1.5) == "stabilizing"
        assert classify_ddg(-1.1) == "stabilizing"

    def test_classify_destabilizing(self):
        """Test classifying destabilizing mutations (ddG > 1)."""
        assert classify_ddg(2.5) == "destabilizing"
        assert classify_ddg(1.5) == "destabilizing"
        assert classify_ddg(1.1) == "destabilizing"

    def test_classify_neutral(self):
        """Test classifying neutral mutations (-1 <= ddG <= 1)."""
        assert classify_ddg(0.0) == "neutral"
        assert classify_ddg(0.5) == "neutral"
        assert classify_ddg(-0.5) == "neutral"
        assert classify_ddg(1.0) == "neutral"
        assert classify_ddg(-1.0) == "neutral"

    def test_classify_boundary_values(self):
        """Test classification at exact boundary values."""
        # At exactly -1.0, should be neutral (not < -1)
        assert classify_ddg(-1.0) == "neutral"
        # At exactly 1.0, should be neutral (not > 1)
        assert classify_ddg(1.0) == "neutral"

    def test_classify_extreme_values(self):
        """Test classification with extreme ddG values."""
        assert classify_ddg(-10.0) == "stabilizing"
        assert classify_ddg(15.0) == "destabilizing"


class TestParseStabilityPrediction:
    """Tests for parse_stability_prediction function."""

    def test_parse_explicit_ddg_value(self):
        """Test parsing text with explicit ddG value."""
        text = "The ddG value is -2.5 kcal/mol. This is a stabilizing mutation."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == -2.5
        assert cls == "stabilizing"

    def test_parse_ddg_with_positive_sign(self):
        """Test parsing ddG with explicit positive sign."""
        text = "The ddG is +3.2 kcal/mol, indicating destabilization."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == 3.2
        assert cls == "destabilizing"

    def test_parse_delta_delta_g_notation(self):
        """Test parsing text with 'delta delta G' notation."""
        text = "The delta delta G is approximately 1.8 kcal/mol."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == 1.8
        assert cls == "destabilizing"

    def test_parse_kcal_mol_units(self):
        """Test parsing values with kcal/mol units."""
        text = "The stability change is approximately -0.5 kcal/mol."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == -0.5
        assert cls == "neutral"

    def test_parse_stabilizing_keyword_only(self):
        """Test parsing with stabilizing keyword but no numeric value."""
        text = "This mutation is stabilizing due to improved hydrophobic packing."
        ddg, cls = parse_stability_prediction(text)
        assert ddg is None
        assert cls == "stabilizing"

    def test_parse_destabilizing_keyword_only(self):
        """Test parsing with destabilizing keyword but no numeric value."""
        text = "The mutation causes destabilization by disrupting the hydrophobic core."
        ddg, cls = parse_stability_prediction(text)
        assert ddg is None
        assert cls == "destabilizing"

    def test_parse_neutral_keyword(self):
        """Test parsing with neutral keyword."""
        text = "This mutation has a neutral effect on stability."
        ddg, cls = parse_stability_prediction(text)
        assert ddg is None
        assert cls == "neutral"

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        ddg, cls = parse_stability_prediction("")
        assert ddg is None
        assert cls == "neutral"

    def test_parse_none_text(self):
        """Test parsing None input (should handle gracefully)."""
        ddg, cls = parse_stability_prediction(None)
        assert ddg is None
        assert cls == "neutral"

    def test_parse_ambiguous_text(self):
        """Test parsing ambiguous text without clear indicators."""
        text = "The protein structure changes slightly with this mutation."
        ddg, cls = parse_stability_prediction(text)
        assert cls == "neutral"  # Default

    def test_parse_value_equals_pattern(self):
        """Test parsing 'value = X' pattern."""
        text = "The predicted ddG value equals -1.5 kcal/mol."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == -1.5
        assert cls == "stabilizing"

    def test_parse_prediction_result_pattern(self):
        """Test parsing 'prediction result' pattern."""
        text = "The prediction result is 2.1 for this mutation."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == 2.1
        assert cls == "destabilizing"

    def test_parse_approximately_pattern(self):
        """Test parsing 'approximately X' pattern."""
        text = "The stability change is approximately 0.3 kcal/mol."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == 0.3
        assert cls == "neutral"

    def test_parse_conflicting_class_and_value(self):
        """Test that value takes precedence over class keyword."""
        # Even though "stabilizing" is mentioned, the value indicates destabilizing
        text = "The ddG is 3.0 kcal/mol. This could be stabilizing in some contexts."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == 3.0
        assert cls == "destabilizing"  # Value takes precedence

    def test_parse_more_stable_keyword(self):
        """Test parsing 'more stable' keyword."""
        text = "This mutation makes the protein more stable."
        ddg, cls = parse_stability_prediction(text)
        assert cls == "stabilizing"

    def test_parse_less_stable_keyword(self):
        """Test parsing 'less stable' keyword."""
        text = "This mutation makes the protein less stable."
        ddg, cls = parse_stability_prediction(text)
        assert cls == "destabilizing"

    def test_parse_increases_stability_keyword(self):
        """Test parsing 'increases stability' keyword."""
        text = "This mutation increases stability significantly."
        ddg, cls = parse_stability_prediction(text)
        assert cls == "stabilizing"

    def test_parse_decreases_stability_keyword(self):
        """Test parsing 'decreases stability' keyword."""
        text = "This mutation decreases stability of the protein."
        ddg, cls = parse_stability_prediction(text)
        assert cls == "destabilizing"

    def test_parse_no_effect_keyword(self):
        """Test parsing 'no effect' keyword."""
        text = "This mutation has no significant effect on protein stability."
        ddg, cls = parse_stability_prediction(text)
        assert cls == "neutral"

    def test_parse_unicode_delta(self):
        """Test parsing with Unicode delta symbol."""
        text = "The DDG is 2.0 kcal/mol."
        ddg, cls = parse_stability_prediction(text)
        # Should still parse based on value if found
        assert cls in STABILITY_CLASSES


class TestInferClassFromText:
    """Tests for _infer_class_from_text function."""

    def test_infer_stabilizing_first(self):
        """Test inferring when stabilizing appears first."""
        text = "Stabilizing mutation, not destabilizing."
        cls = _infer_class_from_text(text)
        assert cls == "stabilizing"

    def test_infer_destabilizing_first(self):
        """Test inferring when destabilizing appears first."""
        text = "Destabilizing, definitely not stabilizing."
        cls = _infer_class_from_text(text)
        assert cls == "destabilizing"

    def test_infer_neutral_only(self):
        """Test inferring with only neutral keyword."""
        text = "The effect is neutral."
        cls = _infer_class_from_text(text)
        assert cls == "neutral"

    def test_infer_no_keywords(self):
        """Test inferring when no keywords present."""
        text = "The protein has a mutation at position 45."
        cls = _infer_class_from_text(text)
        assert cls == "neutral"  # Default


class TestStabilityTestSample:
    """Tests for StabilityTestSample dataclass."""

    def test_sample_creation_basic(self):
        """Test creating a basic stability test sample."""
        sample = StabilityTestSample(
            protein_id="1BNI_A123G",
            sequence="MKTAYIAK",
            wild_type_sequence="MKTAYIAK",
            mutation="A123G",
            ddg_value=2.5,
        )
        assert sample.protein_id == "1BNI_A123G"
        assert sample.sequence == "MKTAYIAK"
        assert sample.wild_type_sequence == "MKTAYIAK"
        assert sample.mutation == "A123G"
        assert sample.ddg_value == 2.5
        assert sample.stability_class == "destabilizing"  # Auto-assigned

    def test_sample_auto_classification(self):
        """Test that stability class is auto-assigned from ddG."""
        sample_stab = StabilityTestSample(
            protein_id="test",
            sequence="MKTAY",
            wild_type_sequence=None,
            mutation="A1G",
            ddg_value=-2.0,
        )
        assert sample_stab.stability_class == "stabilizing"

        sample_neutral = StabilityTestSample(
            protein_id="test",
            sequence="MKTAY",
            wild_type_sequence=None,
            mutation="A1G",
            ddg_value=0.5,
        )
        assert sample_neutral.stability_class == "neutral"

        sample_destab = StabilityTestSample(
            protein_id="test",
            sequence="MKTAY",
            wild_type_sequence=None,
            mutation="A1G",
            ddg_value=3.0,
        )
        assert sample_destab.stability_class == "destabilizing"

    def test_sample_with_explicit_class(self):
        """Test sample with explicitly provided stability class."""
        sample = StabilityTestSample(
            protein_id="test",
            sequence="MKTAY",
            wild_type_sequence=None,
            mutation="A1G",
            ddg_value=0.5,
            stability_class="neutral",
        )
        assert sample.stability_class == "neutral"

    def test_sample_invalid_class(self):
        """Test that invalid stability class raises error."""
        with pytest.raises(ValueError, match="stability_class must be one of"):
            StabilityTestSample(
                protein_id="test",
                sequence="MKTAY",
                wild_type_sequence=None,
                mutation="A1G",
                ddg_value=0.5,
                stability_class="invalid",
            )

    def test_sample_with_optional_fields(self):
        """Test sample with all optional fields."""
        sample = StabilityTestSample(
            protein_id="1BNI_A123G",
            sequence="MKTAYIAK",
            wild_type_sequence="MKTAYIAK",
            mutation="A123G",
            ddg_value=2.5,
            pdb_id="1BNI",
            chain="A",
            description="Test mutation",
            source="ProTherm",
        )
        assert sample.pdb_id == "1BNI"
        assert sample.chain == "A"
        assert sample.description == "Test mutation"
        assert sample.source == "ProTherm"


class TestStabilityPredictionResult:
    """Tests for StabilityPredictionResult dataclass."""

    def test_result_creation(self):
        """Test creating a prediction result."""
        result = StabilityPredictionResult(
            predicted_ddg=-2.5,
            predicted_class="stabilizing",
            ground_truth_ddg=-2.0,
            ground_truth_class="stabilizing",
            generated_text="The ddG is -2.5 kcal/mol. Stabilizing.",
        )
        assert result.predicted_ddg == -2.5
        assert result.predicted_class == "stabilizing"
        assert result.ground_truth_ddg == -2.0
        assert result.ground_truth_class == "stabilizing"

    def test_result_with_none_ddg(self):
        """Test result with None predicted ddG."""
        result = StabilityPredictionResult(
            predicted_ddg=None,
            predicted_class="destabilizing",
            ground_truth_ddg=3.0,
            ground_truth_class="destabilizing",
            generated_text="This mutation is destabilizing.",
            parse_success=False,
        )
        assert result.predicted_ddg is None
        assert result.parse_success is False

    def test_result_with_metadata(self):
        """Test result with protein ID and mutation."""
        result = StabilityPredictionResult(
            predicted_ddg=1.5,
            predicted_class="destabilizing",
            ground_truth_ddg=1.8,
            ground_truth_class="destabilizing",
            generated_text="ddG = 1.5",
            protein_id="1BNI_A123G",
            mutation="A123G",
        )
        assert result.protein_id == "1BNI_A123G"
        assert result.mutation == "A123G"


class TestApplyMutation:
    """Tests for _apply_mutation function."""

    def test_apply_simple_mutation(self):
        """Test applying a simple mutation."""
        seq = "MKTAYIAK"
        mutant = _apply_mutation(seq, "A3G")
        assert mutant == "MKGAYIAK"  # A at position 3 (1-indexed) -> G

    def test_apply_mutation_first_position(self):
        """Test applying mutation at first position."""
        seq = "MKTAYIAK"
        mutant = _apply_mutation(seq, "M1A")
        assert mutant == "AKTAYIAK"

    def test_apply_mutation_last_position(self):
        """Test applying mutation at last position."""
        seq = "MKTAYIAK"
        mutant = _apply_mutation(seq, "K8L")
        assert mutant == "MKTAYIAL"

    def test_apply_invalid_mutation_format(self):
        """Test that invalid mutation format returns unchanged sequence."""
        seq = "MKTAYIAK"
        mutant = _apply_mutation(seq, "invalid")
        assert mutant == seq

    def test_apply_mutation_position_out_of_range(self):
        """Test mutation with position out of range."""
        seq = "MKTAYIAK"
        mutant = _apply_mutation(seq, "A100G")
        assert mutant == seq  # Should return unchanged


class TestComputeStabilityMetrics:
    """Tests for compute_stability_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=-2.0,
                predicted_class="stabilizing",
                ground_truth_ddg=-2.0,
                ground_truth_class="stabilizing",
                generated_text="ddG = -2.0",
            ),
            StabilityPredictionResult(
                predicted_ddg=0.5,
                predicted_class="neutral",
                ground_truth_ddg=0.5,
                ground_truth_class="neutral",
                generated_text="ddG = 0.5",
            ),
            StabilityPredictionResult(
                predicted_ddg=3.0,
                predicted_class="destabilizing",
                ground_truth_ddg=3.0,
                ground_truth_class="destabilizing",
                generated_text="ddG = 3.0",
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        assert metrics["accuracy"] == 1.0
        assert metrics["pearson"] == 1.0 or abs(metrics["pearson"] - 1.0) < 0.01
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0

    def test_no_predictions(self):
        """Test metrics with no predictions."""
        metrics = compute_stability_metrics([])
        assert "error" in metrics

    def test_predictions_without_ddg(self):
        """Test metrics when no ddG values were parsed."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=None,
                predicted_class="stabilizing",
                ground_truth_ddg=-2.0,
                ground_truth_class="stabilizing",
                generated_text="Stabilizing",
                parse_success=False,
            ),
            StabilityPredictionResult(
                predicted_ddg=None,
                predicted_class="destabilizing",
                ground_truth_ddg=3.0,
                ground_truth_class="destabilizing",
                generated_text="Destabilizing",
                parse_success=False,
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        # Classification metrics should still work
        assert metrics["accuracy"] == 1.0
        # Regression metrics should be NaN
        assert math.isnan(metrics["pearson"])
        assert math.isnan(metrics["rmse"])
        assert metrics["ddg_parse_success_rate"] == 0.0

    def test_partial_ddg_predictions(self):
        """Test metrics with some predictions having ddG values."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=-2.0,
                predicted_class="stabilizing",
                ground_truth_ddg=-2.0,
                ground_truth_class="stabilizing",
                generated_text="ddG = -2.0",
            ),
            StabilityPredictionResult(
                predicted_ddg=None,
                predicted_class="destabilizing",
                ground_truth_ddg=3.0,
                ground_truth_class="destabilizing",
                generated_text="Destabilizing",
                parse_success=False,
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        assert metrics["ddg_parse_success_rate"] == 0.5
        assert metrics["num_samples_with_ddg"] == 1
        assert metrics["accuracy"] == 1.0

    def test_classification_metrics(self):
        """Test that all classification metrics are computed."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=-2.0,
                predicted_class="stabilizing",
                ground_truth_ddg=-2.0,
                ground_truth_class="stabilizing",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=0.5,
                predicted_class="neutral",
                ground_truth_ddg=0.0,
                ground_truth_class="neutral",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=3.0,
                predicted_class="destabilizing",
                ground_truth_ddg=3.5,
                ground_truth_class="destabilizing",
                generated_text="",
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_stabilizing" in metrics
        assert "f1_neutral" in metrics
        assert "f1_destabilizing" in metrics

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=None,
                predicted_class="stabilizing",
                ground_truth_ddg=-2.0,
                ground_truth_class="stabilizing",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=None,
                predicted_class="neutral",
                ground_truth_ddg=0.5,
                ground_truth_class="neutral",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=None,
                predicted_class="destabilizing",
                ground_truth_ddg=-1.5,
                ground_truth_class="stabilizing",  # Misclassified
                generated_text="",
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            # Should be a 3x3 matrix
            assert len(cm) == 3
            assert all(len(row) == 3 for row in cm)

    def test_regression_metrics(self):
        """Test regression metrics computation."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=2.0,
                predicted_class="destabilizing",
                ground_truth_ddg=2.5,
                ground_truth_class="destabilizing",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=-1.0,
                predicted_class="neutral",
                ground_truth_ddg=-1.5,
                ground_truth_class="stabilizing",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=0.5,
                predicted_class="neutral",
                ground_truth_ddg=0.0,
                ground_truth_class="neutral",
                generated_text="",
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        assert "pearson" in metrics
        assert "spearman" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

        # Check reasonable ranges
        assert -1 <= metrics["pearson"] <= 1
        assert -1 <= metrics["spearman"] <= 1
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_sample_counts(self):
        """Test that sample counts are correct."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=-2.0,
                predicted_class="stabilizing",
                ground_truth_ddg=-2.0,
                ground_truth_class="stabilizing",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=0.5,
                predicted_class="neutral",
                ground_truth_ddg=0.5,
                ground_truth_class="neutral",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=3.0,
                predicted_class="destabilizing",
                ground_truth_ddg=3.0,
                ground_truth_class="destabilizing",
                generated_text="",
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        assert metrics["num_samples"] == 3
        assert metrics["num_stabilizing"] == 1
        assert metrics["num_neutral"] == 1
        assert metrics["num_destabilizing"] == 1


class TestBasicStatistics:
    """Tests for basic statistics functions."""

    def test_pearson_basic_perfect(self):
        """Test basic Pearson correlation with perfect correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = _compute_pearson_basic(y_true, y_pred)
        assert abs(r - 1.0) < 0.001

    def test_pearson_basic_negative(self):
        """Test basic Pearson correlation with negative correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        r = _compute_pearson_basic(y_true, y_pred)
        assert abs(r - (-1.0)) < 0.001

    def test_pearson_basic_single_value(self):
        """Test basic Pearson with single value."""
        y_true = np.array([1.0])
        y_pred = np.array([2.0])
        r = _compute_pearson_basic(y_true, y_pred)
        assert r == 0.0

    def test_spearman_basic_perfect(self):
        """Test basic Spearman correlation with perfect correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = _compute_spearman_basic(y_true, y_pred)
        assert abs(r - 1.0) < 0.001

    def test_r2_basic_perfect(self):
        """Test basic R^2 with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = _compute_r2_basic(y_true, y_pred)
        assert abs(r2 - 1.0) < 0.001

    def test_r2_basic_poor(self):
        """Test basic R^2 with poor predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        r2 = _compute_r2_basic(y_true, y_pred)
        assert r2 < 0  # Negative R^2 for worse than mean prediction


class TestClassificationBasic:
    """Tests for _compute_classification_basic function."""

    def test_perfect_classification(self):
        """Test basic classification metrics with perfect predictions."""
        y_true = ["stabilizing", "neutral", "destabilizing"]
        y_pred = ["stabilizing", "neutral", "destabilizing"]
        metrics = _compute_classification_basic(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_stabilizing"] == 1.0
        assert metrics["f1_neutral"] == 1.0
        assert metrics["f1_destabilizing"] == 1.0

    def test_all_wrong_classification(self):
        """Test basic classification metrics with all wrong predictions."""
        y_true = ["stabilizing", "neutral", "destabilizing"]
        y_pred = ["neutral", "destabilizing", "stabilizing"]
        metrics = _compute_classification_basic(y_true, y_pred)

        assert metrics["accuracy"] == 0.0


class TestCreateStabilityPrompt:
    """Tests for create_stability_prompt function."""

    def test_default_prompt_with_wt(self):
        """Test creating prompt with wild-type sequence."""
        seq = "MKTAYIAK"
        mutation = "A3G"
        wt_seq = "MKAAYIAK"
        prompt = create_stability_prompt(seq, mutation, wild_type_sequence=wt_seq)

        assert wt_seq in prompt
        assert "A3G" in prompt
        assert "ddG" in prompt.lower() or "stability" in prompt.lower()

    def test_default_prompt_without_wt(self):
        """Test creating prompt without wild-type sequence."""
        seq = "MKTAYIAK"
        mutation = "A3G"
        prompt = create_stability_prompt(seq, mutation)

        assert seq in prompt
        assert "A3G" in prompt

    def test_prompt_mutation_expansion(self):
        """Test that mutation is expanded to readable form."""
        prompt = create_stability_prompt("MKTAY", "A3G")
        # Should contain expanded form like "Alanine to Glycine"
        assert "3" in prompt
        assert ("Alanine" in prompt or "A3G" in prompt)

    def test_custom_prompt_template(self):
        """Test using custom prompt template."""
        template = "Sequence: {sequence}\nMutation: {mutation}\nWT: {wild_type_sequence}"
        prompt = create_stability_prompt(
            "MKTAY", "A3G", wild_type_sequence="MKGAY", prompt_template=template
        )

        assert prompt == "Sequence: MKTAY\nMutation: A3G\nWT: MKGAY"


class TestLoadStabilityTestDataset:
    """Tests for load_stability_test_dataset function."""

    def test_load_demo_dataset(self):
        """Test loading demo dataset when no path provided."""
        cfg = OmegaConf.create({})
        samples = load_stability_test_dataset(cfg, max_samples=10)

        assert len(samples) == 10
        assert all(isinstance(s, StabilityTestSample) for s in samples)
        assert all(s.stability_class in STABILITY_CLASSES for s in samples)

    def test_load_with_max_samples(self):
        """Test loading with max_samples limit."""
        cfg = OmegaConf.create({
            "evaluation": {"max_samples": 5}
        })
        samples = load_stability_test_dataset(cfg)

        assert len(samples) == 5

    def test_load_json_dataset(self):
        """Test loading from JSON file."""
        test_data = [
            {
                "protein_id": "test1",
                "sequence": "MKTAYIAK",
                "wild_type_sequence": "MKAAYIAK",
                "mutation": "A3T",
                "ddg_value": 2.5,
            },
            {
                "protein_id": "test2",
                "sequence": "MKGAYIAK",
                "mutation": "A3G",
                "ddg_value": -1.5,
            },
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            cfg = OmegaConf.create({
                "dataset": {
                    "path": temp_path,
                    "format": "json",
                }
            })
            samples = load_stability_test_dataset(cfg)

            assert len(samples) == 2
            assert samples[0].protein_id == "test1"
            assert samples[0].ddg_value == 2.5
            assert samples[0].stability_class == "destabilizing"
            assert samples[1].ddg_value == -1.5
            assert samples[1].stability_class == "stabilizing"
        finally:
            Path(temp_path).unlink()

    def test_load_csv_dataset(self):
        """Test loading from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=[
                "protein_id", "sequence", "mutation", "ddg_value"
            ])
            writer.writeheader()
            writer.writerow({
                "protein_id": "test1",
                "sequence": "MKTAYIAK",
                "mutation": "A3T",
                "ddg_value": "2.5",
            })
            writer.writerow({
                "protein_id": "test2",
                "sequence": "MKGAYIAK",
                "mutation": "A3G",
                "ddg_value": "-1.5",
            })
            temp_path = f.name

        try:
            samples = _load_csv_dataset(temp_path, max_samples=None)

            assert len(samples) == 2
            assert samples[0].ddg_value == 2.5
            assert samples[1].ddg_value == -1.5
        finally:
            Path(temp_path).unlink()


class TestCreateDemoDataset:
    """Tests for _create_demo_dataset function."""

    def test_create_demo_dataset_default(self):
        """Test creating demo dataset with default size."""
        samples = _create_demo_dataset()
        assert len(samples) == 20

    def test_create_demo_dataset_custom_size(self):
        """Test creating demo dataset with custom size."""
        samples = _create_demo_dataset(num_samples=5)
        assert len(samples) == 5

    def test_demo_dataset_has_valid_samples(self):
        """Test that demo dataset samples are valid."""
        samples = _create_demo_dataset(num_samples=10)

        for sample in samples:
            assert sample.protein_id is not None
            assert len(sample.sequence) > 0
            assert sample.mutation is not None
            assert sample.ddg_value is not None
            assert sample.stability_class in STABILITY_CLASSES

    def test_demo_dataset_has_all_classes(self):
        """Test that demo dataset has all stability classes."""
        samples = _create_demo_dataset(num_samples=20)

        classes = {s.stability_class for s in samples}
        assert "stabilizing" in classes
        assert "neutral" in classes
        assert "destabilizing" in classes


class TestEvaluateStabilityFromPredictions:
    """Tests for evaluate_stability_from_predictions function."""

    def test_evaluate_from_dict_predictions(self):
        """Test evaluating from dictionary predictions."""
        predictions = [
            {
                "predicted_ddg": -2.0,
                "predicted_class": "stabilizing",
                "ground_truth_ddg": -2.5,
                "ground_truth_class": "stabilizing",
                "protein_id": "test1",
                "mutation": "A1G",
            },
            {
                "predicted_ddg": 3.0,
                "predicted_class": "destabilizing",
                "ground_truth_ddg": 2.8,
                "ground_truth_class": "destabilizing",
                "protein_id": "test2",
                "mutation": "G2A",
            },
        ]

        metrics = evaluate_stability_from_predictions(predictions)

        assert "accuracy" in metrics
        assert "pearson" in metrics
        assert "rmse" in metrics
        assert metrics["num_samples"] == 2
        assert metrics["accuracy"] == 1.0

    def test_evaluate_auto_class_assignment(self):
        """Test that ground truth class is auto-assigned if not provided."""
        predictions = [
            {
                "predicted_ddg": -2.0,
                "predicted_class": "stabilizing",
                "ground_truth_ddg": -2.5,
                # No ground_truth_class - should be auto-assigned
            },
        ]

        metrics = evaluate_stability_from_predictions(predictions)

        assert metrics["accuracy"] == 1.0  # Should correctly classify as stabilizing

    def test_evaluate_empty_predictions(self):
        """Test evaluating empty predictions."""
        metrics = evaluate_stability_from_predictions([])
        assert "error" in metrics


class TestEvaluateStability:
    """Tests for evaluate_stability function with mocked model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock ProteinLLM model."""
        model = MagicMock()
        model.eval = MagicMock(return_value=None)
        # Mock generate to return text with ddG predictions
        model.generate = MagicMock(return_value=[
            "The ddG is approximately 2.5 kcal/mol. This mutation is destabilizing.",
        ])
        return model

    @pytest.fixture
    def basic_config(self):
        """Create basic configuration for testing."""
        return OmegaConf.create({
            "model": {"path": "test/model"},
            "dataset": {},
            "evaluation": {
                "batch_size": 1,
                "max_new_tokens": 256,
                "max_samples": 2,
            },
            "logging": {},
        })

    def test_evaluate_stability_with_mock_model(self, mock_model, basic_config):
        """Test full evaluation with mocked model (passed directly)."""
        metrics = evaluate_stability(basic_config, model=mock_model)

        assert "accuracy" in metrics or "error" not in metrics
        assert mock_model.generate.called

    def test_evaluate_stability_with_checkpoint(self, mock_model, basic_config):
        """Test evaluation with checkpoint path."""
        with patch("src.models.multimodal_llm.ProteinLLM") as MockProteinLLM:
            MockProteinLLM.from_pretrained.return_value = mock_model

            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / "checkpoint"
                checkpoint_path.mkdir()
                (checkpoint_path / "config.json").write_text("{}")

                metrics = evaluate_stability(basic_config, checkpoint_path=str(checkpoint_path))

                MockProteinLLM.from_pretrained.assert_called_once()

    def test_evaluate_stability_handles_generation_error(self, mock_model, basic_config):
        """Test that evaluation handles generation errors gracefully."""
        mock_model.generate.side_effect = RuntimeError("Generation failed")

        # Pass model directly - should not raise, but return empty or minimal metrics
        metrics = evaluate_stability(basic_config, model=mock_model)
        assert isinstance(metrics, dict)


class TestSaveAndLogging:
    """Tests for result saving and logging functions."""

    def test_save_predictions(self):
        """Test saving predictions to JSON file."""
        from src.evaluation.stability import _save_predictions

        predictions = [
            StabilityPredictionResult(
                predicted_ddg=-2.0,
                predicted_class="stabilizing",
                ground_truth_ddg=-2.5,
                ground_truth_class="stabilizing",
                generated_text="ddG = -2.0",
                protein_id="test1",
                mutation="A1G",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_predictions(predictions, tmpdir, "stability")

            output_file = Path(tmpdir) / "stability_predictions.json"
            assert output_file.exists()

            with open(output_file) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["protein_id"] == "test1"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test metrics with single sample."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=2.0,
                predicted_class="destabilizing",
                ground_truth_ddg=2.5,
                ground_truth_class="destabilizing",
                generated_text="",
            )
        ]
        metrics = compute_stability_metrics(predictions)
        assert metrics["accuracy"] == 1.0
        assert metrics["num_samples"] == 1

    def test_all_same_class(self):
        """Test metrics when all samples are same class."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=2.0,
                predicted_class="destabilizing",
                ground_truth_ddg=2.5,
                ground_truth_class="destabilizing",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=3.0,
                predicted_class="destabilizing",
                ground_truth_ddg=3.5,
                ground_truth_class="destabilizing",
                generated_text="",
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        assert metrics["num_destabilizing"] == 2
        assert metrics["num_stabilizing"] == 0
        assert metrics["num_neutral"] == 0

    def test_extreme_ddg_values(self):
        """Test with extreme ddG values."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=-10.0,
                predicted_class="stabilizing",
                ground_truth_ddg=-8.0,
                ground_truth_class="stabilizing",
                generated_text="",
            ),
            StabilityPredictionResult(
                predicted_ddg=15.0,
                predicted_class="destabilizing",
                ground_truth_ddg=12.0,
                ground_truth_class="destabilizing",
                generated_text="",
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        # Should still compute metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert not math.isnan(metrics["rmse"])

    def test_very_long_generated_text(self):
        """Test parsing very long generated text."""
        long_text = "The ddG value is 2.5 kcal/mol. " + "Additional context. " * 100
        ddg, cls = parse_stability_prediction(long_text)
        assert ddg == 2.5
        assert cls == "destabilizing"

    def test_parse_text_with_special_characters(self):
        """Test parsing text with special characters."""
        text = "The ddG is approximately -1.5 kcal/mol (95% CI: -2.0 to -1.0)."
        ddg, cls = parse_stability_prediction(text)
        assert ddg == -1.5
        assert cls == "stabilizing"

    def test_nan_handling_in_metrics(self):
        """Test that NaN values are handled properly."""
        predictions = [
            StabilityPredictionResult(
                predicted_ddg=None,
                predicted_class="neutral",
                ground_truth_ddg=0.5,
                ground_truth_class="neutral",
                generated_text="",
            ),
        ]
        metrics = compute_stability_metrics(predictions)

        # Regression metrics should be NaN
        assert math.isnan(metrics["pearson"])
        # Classification metrics should still work
        assert metrics["accuracy"] == 1.0
