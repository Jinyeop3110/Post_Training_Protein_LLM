"""
Tests for GO (Gene Ontology) Term Prediction Evaluation Module.

Tests cover:
- parse_go_terms: Extracting GO terms from text
- categorize_go_terms: Categorizing GO terms into MF/BP/CC
- compute_go_metrics: Computing evaluation metrics
- load_go_test_dataset: Loading test datasets
- evaluate_go: End-to-end evaluation with mocked model
"""

import json
import tempfile
from pathlib import Path
from typing import Set
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from src.evaluation.go_prediction import (
    GOPredictionResult,
    GOTestSample,
    categorize_go_terms,
    compute_go_metrics,
    create_go_prompt,
    evaluate_go,
    evaluate_go_from_predictions,
    load_go_test_dataset,
    parse_go_terms,
    _create_demo_dataset,
    _compute_exact_match_accuracy,
    _compute_category_metrics,
)


class TestParseGOTerms:
    """Tests for parse_go_terms function."""

    def test_parse_single_go_term(self):
        """Test parsing a single GO term from text."""
        text = "The protein has GO:0003674 as its molecular function."
        result = parse_go_terms(text)
        assert result == ["GO:0003674"]

    def test_parse_multiple_go_terms(self):
        """Test parsing multiple GO terms from text."""
        text = "GO:0003674, GO:0008150, and GO:0005575 are the GO terms."
        result = parse_go_terms(text)
        assert result == ["GO:0003674", "GO:0008150", "GO:0005575"]

    def test_parse_duplicate_go_terms(self):
        """Test that duplicate GO terms are removed."""
        text = "GO:0005515 is mentioned twice: GO:0005515 and GO:0016020"
        result = parse_go_terms(text)
        assert result == ["GO:0005515", "GO:0016020"]

    def test_parse_no_go_terms(self):
        """Test parsing text with no GO terms."""
        text = "This protein binds to DNA and regulates transcription."
        result = parse_go_terms(text)
        assert result == []

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        result = parse_go_terms("")
        assert result == []

    def test_parse_none_text(self):
        """Test parsing None input."""
        result = parse_go_terms(None)
        assert result == []

    def test_parse_invalid_go_format(self):
        """Test that invalid GO formats are not parsed."""
        text = "GO:123456 is invalid (6 digits), GO:12345678 is too long (8 digits)"
        result = parse_go_terms(text)
        assert result == []

    def test_parse_go_terms_in_list_format(self):
        """Test parsing GO terms from a formatted list."""
        text = """
        Molecular Functions:
        - GO:0004713 (protein kinase activity)
        - GO:0005524 (ATP binding)

        Biological Processes:
        - GO:0006468 (protein phosphorylation)
        """
        result = parse_go_terms(text)
        assert "GO:0004713" in result
        assert "GO:0005524" in result
        assert "GO:0006468" in result
        assert len(result) == 3

    def test_parse_preserves_order(self):
        """Test that order is preserved when removing duplicates."""
        text = "First GO:0003674, then GO:0008150, then GO:0003674 again, finally GO:0005575"
        result = parse_go_terms(text)
        assert result == ["GO:0003674", "GO:0008150", "GO:0005575"]


class TestCategorizeGOTerms:
    """Tests for categorize_go_terms function."""

    def test_categorize_mf_terms(self):
        """Test categorizing molecular function terms."""
        go_terms = {"GO:0003674", "GO:0004713", "GO:0016740"}
        mf, bp, cc = categorize_go_terms(go_terms)
        assert len(mf) == 3
        assert len(bp) == 0
        assert len(cc) == 0

    def test_categorize_bp_terms(self):
        """Test categorizing biological process terms."""
        go_terms = {"GO:0006468", "GO:0007165", "GO:0008150"}
        mf, bp, cc = categorize_go_terms(go_terms)
        assert len(mf) == 0
        assert len(bp) == 3
        assert len(cc) == 0

    def test_categorize_cc_terms(self):
        """Test categorizing cellular component terms."""
        go_terms = {"GO:0005575", "GO:0005634", "GO:0005886"}
        mf, bp, cc = categorize_go_terms(go_terms)
        assert len(mf) == 0
        assert len(bp) == 0
        assert len(cc) == 3

    def test_categorize_mixed_terms(self):
        """Test categorizing a mix of GO terms."""
        go_terms = {"GO:0003674", "GO:0006468", "GO:0005634"}
        mf, bp, cc = categorize_go_terms(go_terms)
        assert "GO:0003674" in mf
        assert "GO:0006468" in bp
        assert "GO:0005634" in cc

    def test_categorize_empty_set(self):
        """Test categorizing empty set."""
        mf, bp, cc = categorize_go_terms(set())
        assert len(mf) == 0
        assert len(bp) == 0
        assert len(cc) == 0

    def test_categorize_list_input(self):
        """Test categorizing from list input."""
        go_terms = ["GO:0003674", "GO:0006468"]
        mf, bp, cc = categorize_go_terms(go_terms)
        assert "GO:0003674" in mf
        assert "GO:0006468" in bp


class TestGOTestSample:
    """Tests for GOTestSample dataclass."""

    def test_sample_creation(self):
        """Test creating a GO test sample."""
        sample = GOTestSample(
            protein_id="P00533",
            sequence="MKTAYIAKQRQISFVK",
            go_terms={"GO:0004713", "GO:0006468", "GO:0005634"},
        )
        assert sample.protein_id == "P00533"
        assert sample.sequence == "MKTAYIAKQRQISFVK"
        assert len(sample.go_terms) == 3

    def test_sample_auto_categorization(self):
        """Test that GO terms are auto-categorized on creation."""
        sample = GOTestSample(
            protein_id="test",
            sequence="MKTAYIAK",
            go_terms={"GO:0003674", "GO:0006468", "GO:0005634"},
        )
        # MF term should be categorized
        assert "GO:0003674" in sample.go_terms_mf
        # BP term should be categorized
        assert "GO:0006468" in sample.go_terms_bp
        # CC term should be categorized
        assert "GO:0005634" in sample.go_terms_cc


class TestGOPredictionResult:
    """Tests for GOPredictionResult dataclass."""

    def test_result_creation(self):
        """Test creating a prediction result."""
        result = GOPredictionResult(
            protein_id="P00533",
            predicted_terms={"GO:0004713", "GO:0005524"},
            ground_truth_terms={"GO:0004713", "GO:0006468"},
            generated_text="The protein has GO:0004713 and GO:0005524.",
        )
        assert result.protein_id == "P00533"
        assert len(result.predicted_terms) == 2
        assert len(result.ground_truth_terms) == 2


class TestComputeGOMetrics:
    """Tests for compute_go_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674", "GO:0005634"},
                ground_truth_terms={"GO:0003674", "GO:0005634"},
                generated_text="GO:0003674, GO:0005634",
                predicted_mf={"GO:0003674"},
                predicted_cc={"GO:0005634"},
                ground_truth_mf={"GO:0003674"},
                ground_truth_cc={"GO:0005634"},
            )
        ]
        metrics = compute_go_metrics(predictions)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_micro"] == 1.0
        assert metrics["precision_micro"] == 1.0
        assert metrics["recall_micro"] == 1.0

    def test_no_predictions(self):
        """Test metrics with no predictions."""
        predictions = []
        metrics = compute_go_metrics(predictions)
        assert "error" in metrics

    def test_partial_predictions(self):
        """Test metrics with partial predictions."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674"},
                ground_truth_terms={"GO:0003674", "GO:0005634"},
                generated_text="GO:0003674",
                predicted_mf={"GO:0003674"},
                ground_truth_mf={"GO:0003674"},
                ground_truth_cc={"GO:0005634"},
            )
        ]
        metrics = compute_go_metrics(predictions)

        assert metrics["accuracy"] == 0.0  # Not exact match
        assert 0 < metrics["recall_micro"] < 1.0  # Partial recall

    def test_no_overlap_predictions(self):
        """Test metrics with no overlap between predictions and ground truth."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0004713"},
                ground_truth_terms={"GO:0005634"},
                generated_text="GO:0004713",
                predicted_mf={"GO:0004713"},
                ground_truth_cc={"GO:0005634"},
            )
        ]
        metrics = compute_go_metrics(predictions)

        assert metrics["accuracy"] == 0.0
        # With no overlap, precision for the predicted label is 0

    def test_multiple_samples(self):
        """Test metrics with multiple samples."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674"},
                ground_truth_terms={"GO:0003674"},
                generated_text="GO:0003674",
            ),
            GOPredictionResult(
                protein_id="P2",
                predicted_terms={"GO:0005634"},
                ground_truth_terms={"GO:0005634"},
                generated_text="GO:0005634",
            ),
        ]
        metrics = compute_go_metrics(predictions)

        assert metrics["accuracy"] == 1.0  # Both exact matches
        assert metrics["num_samples"] == 2

    def test_per_category_metrics(self):
        """Test that per-category metrics are computed."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674", "GO:0006468", "GO:0005634"},
                ground_truth_terms={"GO:0003674", "GO:0006468", "GO:0005634"},
                generated_text="test",
                predicted_mf={"GO:0003674"},
                predicted_bp={"GO:0006468"},
                predicted_cc={"GO:0005634"},
                ground_truth_mf={"GO:0003674"},
                ground_truth_bp={"GO:0006468"},
                ground_truth_cc={"GO:0005634"},
            )
        ]
        metrics = compute_go_metrics(predictions, include_per_category=True)

        assert "mf_precision" in metrics
        assert "mf_recall" in metrics
        assert "mf_f1" in metrics
        assert "bp_precision" in metrics
        assert "cc_precision" in metrics


class TestComputeExactMatchAccuracy:
    """Tests for _compute_exact_match_accuracy function."""

    def test_all_exact_matches(self):
        """Test with all exact matches."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674"},
                ground_truth_terms={"GO:0003674"},
                generated_text="",
            ),
            GOPredictionResult(
                protein_id="P2",
                predicted_terms={"GO:0005634", "GO:0006468"},
                ground_truth_terms={"GO:0005634", "GO:0006468"},
                generated_text="",
            ),
        ]
        accuracy = _compute_exact_match_accuracy(predictions)
        assert accuracy == 1.0

    def test_no_exact_matches(self):
        """Test with no exact matches."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674"},
                ground_truth_terms={"GO:0003674", "GO:0005634"},
                generated_text="",
            ),
        ]
        accuracy = _compute_exact_match_accuracy(predictions)
        assert accuracy == 0.0

    def test_partial_exact_matches(self):
        """Test with partial exact matches."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674"},
                ground_truth_terms={"GO:0003674"},
                generated_text="",
            ),
            GOPredictionResult(
                protein_id="P2",
                predicted_terms={"GO:0005634"},
                ground_truth_terms={"GO:0003674"},
                generated_text="",
            ),
        ]
        accuracy = _compute_exact_match_accuracy(predictions)
        assert accuracy == 0.5

    def test_empty_predictions(self):
        """Test with empty predictions list."""
        accuracy = _compute_exact_match_accuracy([])
        assert accuracy == 0.0


class TestComputeCategoryMetrics:
    """Tests for _compute_category_metrics function."""

    def test_perfect_category_metrics(self):
        """Test category metrics with perfect predictions."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms=set(),
                ground_truth_terms=set(),
                generated_text="",
                predicted_mf={"GO:0003674"},
                ground_truth_mf={"GO:0003674"},
            ),
        ]

        get_mf = lambda p: (p.predicted_mf, p.ground_truth_mf)
        metrics = _compute_category_metrics(predictions, get_mf)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0


class TestCreateGOPrompt:
    """Tests for create_go_prompt function."""

    def test_default_prompt(self):
        """Test creating prompt with default template."""
        sequence = "MKTAYIAK"
        prompt = create_go_prompt(sequence)

        assert "MKTAYIAK" in prompt
        assert "Gene Ontology" in prompt
        assert "GO:XXXXXXX" in prompt or "GO:" in prompt

    def test_custom_prompt(self):
        """Test creating prompt with custom template."""
        sequence = "MKTAYIAK"
        template = "Predict GO terms for: {sequence}"
        prompt = create_go_prompt(sequence, prompt_template=template)

        assert prompt == "Predict GO terms for: MKTAYIAK"


class TestLoadGOTestDataset:
    """Tests for load_go_test_dataset function."""

    def test_load_demo_dataset(self):
        """Test loading demo dataset when no path provided."""
        cfg = OmegaConf.create({})
        samples = load_go_test_dataset(cfg, max_samples=5)

        assert len(samples) == 5
        assert all(isinstance(s, GOTestSample) for s in samples)
        assert all(len(s.go_terms) > 0 for s in samples)

    def test_load_with_max_samples(self):
        """Test loading with max_samples limit."""
        cfg = OmegaConf.create({
            "evaluation": {"max_samples": 3}
        })
        samples = load_go_test_dataset(cfg)

        assert len(samples) == 3

    def test_load_json_dataset(self):
        """Test loading from JSON file."""
        # Create temporary JSON file
        test_data = [
            {
                "id": "test_1",
                "sequence": "MKTAYIAK",
                "go_terms": ["GO:0003674", "GO:0005634"],
            },
            {
                "id": "test_2",
                "sequence": "MNIFEMLR",
                "go_terms": ["GO:0006468"],
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
            samples = load_go_test_dataset(cfg)

            assert len(samples) == 2
            assert samples[0].protein_id == "test_1"
            assert "GO:0003674" in samples[0].go_terms
        finally:
            Path(temp_path).unlink()


class TestCreateDemoDataset:
    """Tests for _create_demo_dataset function."""

    def test_create_demo_dataset_default(self):
        """Test creating demo dataset with default size."""
        samples = _create_demo_dataset()
        assert len(samples) == 10

    def test_create_demo_dataset_custom_size(self):
        """Test creating demo dataset with custom size."""
        samples = _create_demo_dataset(num_samples=5)
        assert len(samples) == 5

    def test_demo_dataset_has_valid_samples(self):
        """Test that demo dataset samples are valid."""
        samples = _create_demo_dataset(num_samples=3)

        for sample in samples:
            assert sample.protein_id is not None
            assert len(sample.sequence) > 0
            assert len(sample.go_terms) > 0
            # Check GO terms have valid format
            for term in sample.go_terms:
                assert term.startswith("GO:")
                assert len(term) == 10  # GO:XXXXXXX


class TestEvaluateGOFromPredictions:
    """Tests for evaluate_go_from_predictions function."""

    def test_evaluate_from_dict_predictions(self):
        """Test evaluating from dictionary predictions."""
        predictions = [
            {
                "protein_id": "P1",
                "predicted_terms": ["GO:0003674", "GO:0005634"],
                "ground_truth_terms": ["GO:0003674", "GO:0005634"],
            },
            {
                "protein_id": "P2",
                "predicted_terms": ["GO:0006468"],
                "ground_truth_terms": ["GO:0006468", "GO:0007165"],
            },
        ]

        metrics = evaluate_go_from_predictions(predictions)

        assert "accuracy" in metrics
        assert "f1_micro" in metrics
        assert metrics["num_samples"] == 2

    def test_evaluate_empty_predictions(self):
        """Test evaluating empty predictions."""
        metrics = evaluate_go_from_predictions([])
        assert "error" in metrics


class TestEvaluateGO:
    """Tests for evaluate_go function with mocked model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock ProteinLLM model."""
        model = MagicMock()
        model.eval = MagicMock(return_value=None)
        # Mock generate to return text with GO terms
        model.generate = MagicMock(return_value=[
            "The protein has molecular function GO:0004713 and GO:0005524.",
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

    def test_evaluate_go_with_mock_model(self, mock_model, basic_config):
        """Test full evaluation with mocked model."""
        with patch("src.evaluation.go_prediction.ProteinLLM") as MockProteinLLM:
            MockProteinLLM.from_config.return_value = mock_model

            metrics = evaluate_go(basic_config)

            assert "accuracy" in metrics or "error" not in metrics
            assert mock_model.eval.called
            assert mock_model.generate.called

    def test_evaluate_go_with_checkpoint(self, mock_model, basic_config):
        """Test evaluation with checkpoint path."""
        with patch("src.evaluation.go_prediction.ProteinLLM") as MockProteinLLM:
            MockProteinLLM.from_pretrained.return_value = mock_model

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create minimal checkpoint files
                checkpoint_path = Path(tmpdir) / "checkpoint"
                checkpoint_path.mkdir()
                (checkpoint_path / "config.json").write_text("{}")

                metrics = evaluate_go(basic_config, checkpoint_path=str(checkpoint_path))

                MockProteinLLM.from_pretrained.assert_called_once()

    def test_evaluate_go_handles_generation_error(self, mock_model, basic_config):
        """Test that evaluation handles generation errors gracefully."""
        mock_model.generate.side_effect = RuntimeError("Generation failed")

        with patch("src.evaluation.go_prediction.ProteinLLM") as MockProteinLLM:
            MockProteinLLM.from_config.return_value = mock_model

            # Should not raise, but return empty metrics
            metrics = evaluate_go(basic_config)
            # With all batches failing, metrics should have error or be minimal
            assert isinstance(metrics, dict)


class TestSaveAndLogging:
    """Tests for result saving and logging functions."""

    def test_save_results(self):
        """Test saving results to JSON file."""
        from src.evaluation.go_prediction import _save_results

        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674"},
                ground_truth_terms={"GO:0003674", "GO:0005634"},
                generated_text="test output",
            )
        ]
        metrics = {"accuracy": 0.5, "f1_micro": 0.6}

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({
                "logging": {"output_dir": tmpdir}
            })

            _save_results(predictions, metrics, cfg)

            output_file = Path(tmpdir) / "go_prediction_results.json"
            assert output_file.exists()

            with open(output_file) as f:
                data = json.load(f)

            assert "metrics" in data
            assert "predictions" in data
            assert data["metrics"]["accuracy"] == 0.5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_go_term_list(self):
        """Test handling of many GO terms."""
        go_terms = {f"GO:{str(i).zfill(7)}" for i in range(100)}
        mf, bp, cc = categorize_go_terms(go_terms)
        assert len(mf) + len(bp) + len(cc) == 100

    def test_parse_go_terms_with_special_characters(self):
        """Test parsing GO terms surrounded by special characters."""
        text = "[GO:0003674], (GO:0005634), 'GO:0006468'"
        result = parse_go_terms(text)
        assert len(result) == 3

    def test_empty_ground_truth(self):
        """Test metrics with empty ground truth."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms={"GO:0003674"},
                ground_truth_terms=set(),
                generated_text="GO:0003674",
            )
        ]
        # Should handle gracefully
        metrics = compute_go_metrics(predictions)
        assert isinstance(metrics, dict)

    def test_empty_predictions_set(self):
        """Test metrics with empty predictions set."""
        predictions = [
            GOPredictionResult(
                protein_id="P1",
                predicted_terms=set(),
                ground_truth_terms={"GO:0003674"},
                generated_text="No GO terms predicted",
            )
        ]
        metrics = compute_go_metrics(predictions)
        assert metrics["recall_micro"] == 0.0
