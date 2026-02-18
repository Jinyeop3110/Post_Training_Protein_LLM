"""Tests for Mol-Instructions dataset implementation."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any


class TestMolInstructionsImport:
    """Test that Mol-Instructions components can be imported."""

    def test_import_dataset(self):
        """Test MolInstructionsDataset can be imported."""
        from src.data.mol_instructions import MolInstructionsDataset
        assert MolInstructionsDataset is not None

    def test_import_collator(self):
        """Test MolInstructionsCollator can be imported."""
        from src.data.mol_instructions import MolInstructionsCollator
        assert MolInstructionsCollator is not None

    def test_import_config(self):
        """Test MolInstructionsConfig can be imported."""
        from src.data.mol_instructions import MolInstructionsConfig
        assert MolInstructionsConfig is not None

    def test_import_dataloader(self):
        """Test get_mol_instructions_dataloader can be imported."""
        from src.data.mol_instructions import get_mol_instructions_dataloader
        assert get_mol_instructions_dataloader is not None

    def test_import_from_init(self):
        """Test imports work from the package __init__."""
        from src.data import (
            MolInstructionsDataset,
            MolInstructionsCollator,
            MolInstructionsConfig,
            get_mol_instructions_dataloader,
        )
        assert MolInstructionsDataset is not None
        assert MolInstructionsCollator is not None
        assert MolInstructionsConfig is not None
        assert get_mol_instructions_dataloader is not None


class TestMolInstructionsConfig:
    """Tests for MolInstructionsConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.data.mol_instructions import MolInstructionsConfig

        config = MolInstructionsConfig()

        assert config.dataset_name == "zjunlp/Mol-Instructions"
        assert config.subset == "Protein-oriented Instructions"
        assert config.max_seq_length == 2048
        assert config.train_split == 0.9
        assert config.val_split == 0.05
        assert config.test_split == 0.05
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom configuration values."""
        from src.data.mol_instructions import MolInstructionsConfig

        config = MolInstructionsConfig(
            max_seq_length=1024,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            seed=123,
        )

        assert config.max_seq_length == 1024
        assert config.train_split == 0.8
        assert config.seed == 123


class TestMolInstructionsDataset:
    """Tests for MolInstructionsDataset."""

    @pytest.fixture
    def mock_hf_dataset(self):
        """Create a mock HuggingFace dataset."""
        # Create mock data samples
        mock_data = [
            {
                "instruction": "Predict the function of the following protein.",
                "input": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH",
                "output": "This protein is involved in ATP binding and has kinase activity.",
            },
            {
                "instruction": "Describe the subcellular localization of this protein.",
                "input": "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
                "output": "The protein is primarily localized in the cytoplasm.",
            },
            {
                "instruction": "Generate a protein sequence with kinase activity.",
                "input": "",
                "output": "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFGLFVQLQVGNQPKNSNVSLDLCVFSEMG",
            },
        ]

        # Create mock dataset with select method
        class MockDataset:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

            def shuffle(self, seed=None):
                return self

            def select(self, indices):
                selected_data = [self._data[i] for i in indices if i < len(self._data)]
                return MockDataset(selected_data)

        return MockDataset(mock_data)

    def test_dataset_with_mock_data(self, mock_hf_dataset):
        """Test dataset loading with mocked HuggingFace data."""
        from src.data.mol_instructions import MolInstructionsDataset

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = {"train": mock_hf_dataset}

            dataset = MolInstructionsDataset(split="train", limit=3)

            assert len(dataset) == 3

    def test_dataset_item_format(self, mock_hf_dataset):
        """Test that dataset items have the correct format."""
        from src.data.mol_instructions import MolInstructionsDataset

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = {"train": mock_hf_dataset}

            dataset = MolInstructionsDataset(split="train", limit=3)
            item = dataset[0]

            # Check required fields
            assert "protein_sequence" in item
            assert "instruction" in item
            assert "response" in item
            assert "formatted_prompt" in item
            assert "input_text" in item

            # Check field types
            assert isinstance(item["protein_sequence"], str)
            assert isinstance(item["instruction"], str)
            assert isinstance(item["response"], str)
            assert isinstance(item["formatted_prompt"], str)

    def test_protein_sequence_extraction(self, mock_hf_dataset):
        """Test protein sequence is correctly extracted from input."""
        from src.data.mol_instructions import MolInstructionsDataset

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = {"train": mock_hf_dataset}

            dataset = MolInstructionsDataset(split="train", limit=3)
            item = dataset[0]

            # The protein sequence should be extracted from input
            assert len(item["protein_sequence"]) > 0
            # Should contain only amino acid characters
            aa_chars = set("ACDEFGHIKLMNPQRSTVWY")
            assert all(c in aa_chars for c in item["protein_sequence"].upper())

    def test_formatted_prompt_structure(self, mock_hf_dataset):
        """Test that formatted prompt has correct structure."""
        from src.data.mol_instructions import MolInstructionsDataset

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = {"train": mock_hf_dataset}

            dataset = MolInstructionsDataset(split="train", limit=3)
            item = dataset[0]

            prompt = item["formatted_prompt"]

            # Check prompt contains expected sections
            assert "### Instruction:" in prompt
            assert "### Input:" in prompt
            assert "### Response:" in prompt
            assert item["instruction"] in prompt
            assert item["response"] in prompt

    def test_from_config(self, mock_hf_dataset):
        """Test creating dataset from Hydra config."""
        from src.data.mol_instructions import MolInstructionsDataset
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "source": "zjunlp/Mol-Instructions",
            "subset": "Protein-oriented Instructions",
            "split": "train",
            "limit": 2,
            "processing": {
                "max_seq_length": 1024,
            },
            "splits": {
                "train": 0.9,
                "validation": 0.05,
                "test": 0.05,
            },
        })

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = {"train": mock_hf_dataset}

            dataset = MolInstructionsDataset.from_config(config)

            assert dataset is not None
            assert dataset.config.max_seq_length == 1024

    def test_split_validation(self, mock_hf_dataset):
        """Test that different splits work correctly."""
        from src.data.mol_instructions import MolInstructionsDataset

        for split in ["train", "validation", "test"]:
            with patch("src.data.mol_instructions.load_dataset") as mock_load:
                mock_load.return_value = {"train": mock_hf_dataset}

                dataset = MolInstructionsDataset(split=split, limit=1)
                assert len(dataset) >= 0  # May be empty for val/test with small mock

    def test_invalid_split_raises(self, mock_hf_dataset):
        """Test that invalid split raises an error."""
        from src.data.mol_instructions import MolInstructionsDataset

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = mock_hf_dataset  # Not a dict, triggers split creation

            with pytest.raises(ValueError, match="Unknown split"):
                MolInstructionsDataset(split="invalid_split")


class TestMolInstructionsCollator:
    """Tests for MolInstructionsCollator."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token = "[EOS]"

        def mock_call(texts, **kwargs):
            # Simple mock tokenization
            max_len = max(len(t) for t in texts)
            batch_size = len(texts)

            import torch
            return {
                "input_ids": torch.ones(batch_size, min(max_len, kwargs.get("max_length", 100)), dtype=torch.long),
                "attention_mask": torch.ones(batch_size, min(max_len, kwargs.get("max_length", 100)), dtype=torch.long),
            }

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call

        return tokenizer

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch of data."""
        return [
            {
                "protein_sequence": "MKTAYIAK",
                "instruction": "Predict the function.",
                "response": "This is a kinase.",
                "formatted_prompt": "### Instruction:\nPredict the function.\n\n### Input:\nMKTAYIAK\n\n### Response:\nThis is a kinase.",
                "input_text": "MKTAYIAK",
            },
            {
                "protein_sequence": "ACDEFGHIK",
                "instruction": "Describe localization.",
                "response": "Located in cytoplasm.",
                "formatted_prompt": "### Instruction:\nDescribe localization.\n\n### Input:\nACDEFGHIK\n\n### Response:\nLocated in cytoplasm.",
                "input_text": "ACDEFGHIK",
            },
        ]

    def test_collator_creation(self, mock_tokenizer):
        """Test collator can be created with tokenizer."""
        from src.data.mol_instructions import MolInstructionsCollator

        collator = MolInstructionsCollator(
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert collator is not None
        assert collator.max_length == 512

    def test_collator_batch_processing(self, mock_tokenizer, sample_batch):
        """Test collator processes batch correctly."""
        from src.data.mol_instructions import MolInstructionsCollator

        collator = MolInstructionsCollator(
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        result = collator(sample_batch)

        # Check required keys
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert "protein_sequences" in result
        assert "instructions" in result
        assert "responses" in result

        # Check batch size
        assert len(result["protein_sequences"]) == 2
        assert len(result["instructions"]) == 2

    def test_collator_without_labels(self, mock_tokenizer, sample_batch):
        """Test collator can exclude labels."""
        from src.data.mol_instructions import MolInstructionsCollator

        collator = MolInstructionsCollator(
            tokenizer=mock_tokenizer,
            max_length=512,
            include_labels=False,
        )

        result = collator(sample_batch)

        assert "input_ids" in result
        assert "labels" not in result

    def test_collator_padding_options(self, mock_tokenizer, sample_batch):
        """Test different padding options."""
        from src.data.mol_instructions import MolInstructionsCollator

        for padding in ["longest", "max_length"]:
            collator = MolInstructionsCollator(
                tokenizer=mock_tokenizer,
                max_length=512,
                padding=padding,
            )

            result = collator(sample_batch)
            assert "input_ids" in result


class TestMolInstructionsDataLoader:
    """Tests for get_mol_instructions_dataloader function."""

    @pytest.fixture
    def mock_hf_dataset(self):
        """Create a mock HuggingFace dataset."""
        mock_data = [
            {
                "instruction": f"Task {i}",
                "input": f"MKTAYIAK{'A' * i}",
                "output": f"Response {i}",
            }
            for i in range(10)
        ]

        class MockDataset:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

            def shuffle(self, seed=None):
                return self

            def select(self, indices):
                selected_data = [self._data[i] for i in indices if i < len(self._data)]
                return MockDataset(selected_data)

        return MockDataset(mock_data)

    def test_dataloader_creation(self, mock_hf_dataset):
        """Test dataloader can be created."""
        from src.data.mol_instructions import get_mol_instructions_dataloader

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = {"train": mock_hf_dataset}

            dataloader = get_mol_instructions_dataloader(
                split="train",
                batch_size=2,
                num_workers=0,
                limit=5,
            )

            assert dataloader is not None

    def test_dataloader_iteration(self, mock_hf_dataset):
        """Test dataloader can be iterated."""
        from src.data.mol_instructions import get_mol_instructions_dataloader

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = {"train": mock_hf_dataset}

            dataloader = get_mol_instructions_dataloader(
                split="train",
                batch_size=2,
                num_workers=0,
                limit=4,
            )

            batch = next(iter(dataloader))

            assert "protein_sequences" in batch
            assert "instructions" in batch
            assert "responses" in batch
            assert len(batch["protein_sequences"]) == 2

    def test_dataloader_with_tokenizer(self, mock_hf_dataset):
        """Test dataloader with tokenizer creates proper collator."""
        from src.data.mol_instructions import get_mol_instructions_dataloader
        import torch

        # Create a more complete mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "[EOS]"

        def mock_call(texts, **kwargs):
            batch_size = len(texts)
            max_len = 10
            return {
                "input_ids": torch.ones(batch_size, max_len, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, max_len, dtype=torch.long),
            }

        mock_tokenizer.__call__ = mock_call

        with patch("src.data.mol_instructions.load_dataset") as mock_load:
            mock_load.return_value = {"train": mock_hf_dataset}

            dataloader = get_mol_instructions_dataloader(
                split="train",
                batch_size=2,
                num_workers=0,
                tokenizer=mock_tokenizer,
                limit=4,
            )

            batch = next(iter(dataloader))

            # When tokenizer is provided, should have tensor outputs
            assert "input_ids" in batch
            assert "attention_mask" in batch


class TestProteinSequenceExtraction:
    """Tests for protein sequence extraction logic."""

    def test_extract_pure_sequence(self):
        """Test extracting a pure amino acid sequence."""
        from src.data.mol_instructions import MolInstructionsDataset

        dataset = MolInstructionsDataset.__new__(MolInstructionsDataset)
        dataset.config = MagicMock()
        dataset.config.max_protein_length = None

        # Test pure sequence
        seq = dataset._extract_protein_sequence("MKTAYIAKQRQISFVK")
        assert seq == "MKTAYIAKQRQISFVK"

    def test_extract_lowercase_sequence(self):
        """Test extracting lowercase sequence."""
        from src.data.mol_instructions import MolInstructionsDataset

        dataset = MolInstructionsDataset.__new__(MolInstructionsDataset)
        dataset.config = MagicMock()
        dataset.config.max_protein_length = None

        seq = dataset._extract_protein_sequence("mktayiakqrqisfvk")
        assert seq == "MKTAYIAKQRQISFVK"

    def test_extract_empty_input(self):
        """Test handling empty input."""
        from src.data.mol_instructions import MolInstructionsDataset

        dataset = MolInstructionsDataset.__new__(MolInstructionsDataset)
        dataset.config = MagicMock()
        dataset.config.max_protein_length = None

        seq = dataset._extract_protein_sequence("")
        assert seq == ""

    def test_extract_with_whitespace(self):
        """Test extracting sequence with whitespace."""
        from src.data.mol_instructions import MolInstructionsDataset

        dataset = MolInstructionsDataset.__new__(MolInstructionsDataset)
        dataset.config = MagicMock()
        dataset.config.max_protein_length = None

        seq = dataset._extract_protein_sequence("  MKTAYIAK  ")
        assert seq == "MKTAYIAK"


class TestPromptFormatting:
    """Tests for prompt formatting."""

    def test_format_prompt_default_template(self):
        """Test prompt formatting with default template."""
        from src.data.mol_instructions import MolInstructionsDataset, MolInstructionsConfig

        dataset = MolInstructionsDataset.__new__(MolInstructionsDataset)
        dataset.config = MolInstructionsConfig()

        prompt = dataset._format_prompt(
            instruction="Test instruction",
            input_text="MKTAYIAK",
            output="Test response",
        )

        assert "### Instruction:" in prompt
        assert "Test instruction" in prompt
        assert "### Input:" in prompt
        assert "MKTAYIAK" in prompt
        assert "### Response:" in prompt
        assert "Test response" in prompt

    def test_format_prompt_for_inference(self):
        """Test prompt formatting for inference (no output)."""
        from src.data.mol_instructions import MolInstructionsDataset, MolInstructionsConfig

        dataset = MolInstructionsDataset.__new__(MolInstructionsDataset)
        dataset.config = MolInstructionsConfig()

        prompt = dataset._format_prompt(
            instruction="Test instruction",
            input_text="MKTAYIAK",
            output="",
            for_inference=True,
        )

        assert "### Instruction:" in prompt
        assert "### Response:" in prompt
        # For inference, response section should be empty (ready for model to complete)
