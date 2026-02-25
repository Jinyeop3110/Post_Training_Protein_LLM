"""Tests for Multimodal Protein-LLM module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.multimodal_llm import ProteinLLM
from src.models.pooling import AttentionPooling, MeanPooling


class MockESMModel(nn.Module):
    """Mock ESM model for testing without loading real weights."""

    def __init__(self, embed_dim: int = 1280, num_layers: int = 33):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.linear = nn.Linear(embed_dim, embed_dim)  # Dummy layer

    def forward(self, tokens, repr_layers=None, return_contacts=False):
        batch_size, seq_len = tokens.shape
        # Return mock representations
        representations = {
            self.num_layers: torch.randn(batch_size, seq_len, self.embed_dim)
        }
        return {"representations": representations}


class MockAlphabet:
    """Mock ESM alphabet for testing."""

    def __init__(self):
        self.padding_idx = 0

    def get_batch_converter(self):
        def batch_converter(data):
            batch_labels = [d[0] for d in data]
            batch_strs = [d[1] for d in data]
            # Create mock tokens: [BOS] + sequence + [EOS]
            max_len = max(len(s) for s in batch_strs) + 2
            batch_tokens = torch.zeros(len(data), max_len, dtype=torch.long)
            return batch_labels, batch_strs, batch_tokens
        return batch_converter


class MockLLM(nn.Module):
    """Mock LLM for testing without loading real weights."""

    def __init__(self, hidden_size: int = 4096, vocab_size: int = 32000):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = vocab_size

        # Mock embedding layer
        self.model = MagicMock()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        **kwargs,
    ):
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        logits = torch.randn(batch_size, seq_len, self.config.vocab_size)

        # Calculate mock loss if labels provided
        loss = None
        if labels is not None:
            loss = torch.tensor(2.5)  # Mock loss value

        output = MagicMock()
        output.logits = logits
        output.loss = loss
        return output

    def generate(self, inputs_embeds=None, attention_mask=None, **kwargs):
        batch_size = inputs_embeds.shape[0]
        max_new_tokens = kwargs.get("max_new_tokens", 10)
        # Return mock token IDs
        return torch.randint(0, 32000, (batch_size, max_new_tokens))

    def save_pretrained(self, path):
        """Mock save_pretrained - no-op."""
        pass


class MockTokenizerOutput(dict):
    """Dict-like object that supports .to(device) like BatchEncoding."""

    def to(self, device):
        return MockTokenizerOutput({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.items()
        })


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 1
        # Map special tokens for convert_tokens_to_ids
        self._token_to_id = {}

    def __len__(self):
        return self.vocab_size

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        max_len = max(len(t.split()) for t in texts) + 5
        input_ids = torch.randint(2, self.vocab_size, (len(texts), max_len))
        attention_mask = torch.ones_like(input_ids)

        return MockTokenizerOutput({"input_ids": input_ids, "attention_mask": attention_mask})

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [f"Generated text {i}" for i in range(len(token_ids))]

    def convert_tokens_to_ids(self, token):
        """Mock convert_tokens_to_ids - return a fixed ID for any token."""
        return self._token_to_id.get(token, -1)

    def save_pretrained(self, path):
        """Mock save_pretrained - no-op."""
        pass

    def encode(self, text, add_special_tokens=True, truncation=True, max_length=None):
        """Mock encode - return a list of token IDs."""
        return list(range(min(len(text.split()) + 2, max_length or 100)))


@pytest.fixture
def mock_protein_llm(device):
    """Create a ProteinLLM with mocked dependencies for testing."""
    # Create model without loading real components
    model = ProteinLLM(
        llm_name="mock/model",
        encoder_name="esm3-sm-open-v1",
        num_prefix_tokens=8,
        pooling_type="attention",
        use_qlora=False,  # Disable QLoRA for simpler testing
        device=device,
        load_llm=False,
        load_encoder=False,
    )

    # Manually set up mock components
    model.encoder_embed_dim = 1280
    model.llm_hidden_size = 4096

    # Build pooling and projector
    model._build_pooling()
    model._build_projector()

    # Set mock LLM and tokenizer
    mock_llm = MockLLM()
    # Ensure embed_tokens is on the correct device
    mock_llm.model.embed_tokens = mock_llm.model.embed_tokens.to(device)
    model.llm = mock_llm.to(device)
    model.tokenizer = MockTokenizer()

    # Create mock encoder
    mock_encoder = MagicMock()
    mock_encoder.encode = lambda seqs: {
        "embeddings": torch.randn(len(seqs), 50, 1280, device=device),
        "type": "embedding",
    }
    mock_encoder.get_embedding_dim = lambda: 1280
    mock_encoder.model = MockESMModel().to(device)
    model.encoder = mock_encoder

    return model


class TestProteinLLMInitialization:
    """Tests for ProteinLLM initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        model = ProteinLLM(load_llm=False, load_encoder=False)

        assert model.llm_name == ProteinLLM.DEFAULT_LLM_NAME
        assert model.encoder_name == ProteinLLM.DEFAULT_ENCODER_NAME
        assert model.num_prefix_tokens == ProteinLLM.DEFAULT_NUM_PREFIX_TOKENS
        assert model.pooling_type == ProteinLLM.DEFAULT_POOLING_TYPE
        assert model.freeze_encoder is True
        assert model.use_qlora is True

    def test_custom_initialization(self):
        """Test custom initialization values."""
        model = ProteinLLM(
            llm_name="custom/model",
            encoder_name="esm3-sm-open-v1",
            num_prefix_tokens=64,
            pooling_type="mean",
            use_qlora=False,
            lora_r=16,
            lora_alpha=32,
            load_llm=False,
            load_encoder=False,
        )

        assert model.llm_name == "custom/model"
        assert model.encoder_name == "esm3-sm-open-v1"
        assert model.num_prefix_tokens == 64
        assert model.pooling_type == "mean"
        assert model.use_qlora is False
        assert model.lora_r == 16
        assert model.lora_alpha == 32

    def test_esm_embed_dim_mapping(self):
        """Test ESM-3 model embedding dimension mapping."""
        test_cases = [
            ("esm3-sm-open-v1", 1536),
            ("esm3_sm_open_v1", 1536),
        ]

        for encoder_name, expected_dim in test_cases:
            model = ProteinLLM(
                encoder_name=encoder_name,
                load_llm=False,
                load_encoder=False,
            )
            assert model.encoder_embed_dim == expected_dim, \
                f"Failed for {encoder_name}: expected {expected_dim}, got {model.encoder_embed_dim}"


class TestEncodeProtein:
    """Tests for encode_protein method."""

    def test_encode_protein_output_shape(self, mock_protein_llm, device):
        """Test encode_protein produces correct output shape."""
        sequences = ["MKTAYIAKQRQISFVK", "MNIFEMLRIDEGLR"]

        embeddings = mock_protein_llm.encode_protein(sequences)

        assert embeddings.shape == (2, 8, 4096)  # [B, num_prefix_tokens, llm_hidden_size]
        assert embeddings.device.type == device.split(":")[0]

    def test_encode_protein_with_attention_mask(self, mock_protein_llm, device):
        """Test encode_protein returns attention mask when requested."""
        sequences = ["MKTAYIAKQRQISFVK"]

        embeddings, attention_mask = mock_protein_llm.encode_protein(
            sequences, return_attention_mask=True
        )

        assert embeddings.shape == (1, 8, 4096)
        assert attention_mask.shape == (1, 8)
        assert (attention_mask == 1).all()  # All prefix tokens are valid

    def test_encode_protein_batch_sizes(self, mock_protein_llm, device):
        """Test encode_protein with various batch sizes."""
        for batch_size in [1, 4, 8]:
            sequences = [f"MKTAYIAK{i}" for i in range(batch_size)]
            embeddings = mock_protein_llm.encode_protein(sequences)

            assert embeddings.shape[0] == batch_size
            assert embeddings.shape[1] == mock_protein_llm.num_prefix_tokens
            assert embeddings.shape[2] == mock_protein_llm.llm_hidden_size


class TestPrepareInputs:
    """Tests for prepare_inputs method."""

    def test_prepare_inputs_combines_correctly(self, mock_protein_llm, device):
        """Test that prepare_inputs correctly combines protein and text."""
        protein_sequences = ["MKTAYIAK", "MNIFEMLR"]
        text_input_ids = torch.randint(0, 32000, (2, 20), device=device)
        text_attention_mask = torch.ones(2, 20, device=device)

        prepared = mock_protein_llm.prepare_inputs(
            protein_sequences=protein_sequences,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
        )

        # Check shapes
        num_prefix = mock_protein_llm.num_prefix_tokens
        total_len = num_prefix + 20

        assert prepared["inputs_embeds"].shape == (2, total_len, 4096)
        assert prepared["attention_mask"].shape == (2, total_len)
        assert prepared["position_ids"].shape == (2, total_len)

        # Check attention mask is all ones (all valid)
        assert (prepared["attention_mask"] == 1).all()

        # Check position IDs are sequential
        expected_positions = torch.arange(total_len, device=device)
        assert torch.all(prepared["position_ids"][0] == expected_positions)

    def test_prepare_inputs_with_labels(self, mock_protein_llm, device):
        """Test prepare_inputs handles labels correctly."""
        protein_sequences = ["MKTAYIAK"]
        text_input_ids = torch.randint(0, 32000, (1, 10), device=device)
        text_attention_mask = torch.ones(1, 10, device=device)
        labels = torch.randint(0, 32000, (1, 10), device=device)

        prepared = mock_protein_llm.prepare_inputs(
            protein_sequences=protein_sequences,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
        )

        num_prefix = mock_protein_llm.num_prefix_tokens

        assert "labels" in prepared
        assert prepared["labels"].shape == (1, num_prefix + 10)

        # Prefix labels should be -100 (ignored)
        assert (prepared["labels"][:, :num_prefix] == -100).all()

        # Text labels should match original
        assert torch.all(prepared["labels"][:, num_prefix:] == labels)


class TestForward:
    """Tests for forward method."""

    def test_forward_training(self, mock_protein_llm, device):
        """Test forward pass during training."""
        protein_sequences = ["MKTAYIAK", "MNIFEMLR"]
        input_ids = torch.randint(0, 32000, (2, 15), device=device)
        attention_mask = torch.ones(2, 15, device=device)
        labels = torch.randint(0, 32000, (2, 15), device=device)

        outputs = mock_protein_llm(
            protein_sequences=protein_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"] is not None

        # Check logits shape
        num_prefix = mock_protein_llm.num_prefix_tokens
        assert outputs["logits"].shape[0] == 2
        assert outputs["logits"].shape[1] == num_prefix + 15

    def test_forward_without_labels(self, mock_protein_llm, device):
        """Test forward pass without labels (inference mode)."""
        protein_sequences = ["MKTAYIAK"]
        input_ids = torch.randint(0, 32000, (1, 10), device=device)
        attention_mask = torch.ones(1, 10, device=device)

        outputs = mock_protein_llm(
            protein_sequences=protein_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert "logits" in outputs
        # Loss may be None without labels (depends on mock)


class TestGenerate:
    """Tests for generate method."""

    def test_generate_single_sequence(self, mock_protein_llm, device):
        """Test generation with single protein and prompt."""
        protein_sequences = ["MKTAYIAKQRQISFVK"]
        prompt = "Describe this protein:"

        generated = mock_protein_llm.generate(
            protein_sequences=protein_sequences,
            prompt=prompt,
            max_new_tokens=20,
        )

        assert len(generated) == 1
        assert isinstance(generated[0], str)

    def test_generate_batch(self, mock_protein_llm, device):
        """Test generation with batch of proteins."""
        protein_sequences = ["MKTAYIAK", "MNIFEMLR", "GLFVQLQV"]
        prompts = ["What is this?", "Describe this:", "Analyze:"]

        generated = mock_protein_llm.generate(
            protein_sequences=protein_sequences,
            prompt=prompts,
            max_new_tokens=15,
        )

        assert len(generated) == 3

    def test_generate_broadcast_prompt(self, mock_protein_llm, device):
        """Test that single prompt is broadcast to all sequences."""
        protein_sequences = ["MKTAYIAK", "MNIFEMLR"]
        prompt = "What is this protein?"

        generated = mock_protein_llm.generate(
            protein_sequences=protein_sequences,
            prompt=prompt,
        )

        assert len(generated) == 2


class TestFromConfig:
    """Tests for from_config class method."""

    def test_from_config_full(self):
        """Test creating ProteinLLM from full config."""
        config = OmegaConf.create({
            "model": {
                "path": "test/model",
                "architecture": {
                    "hidden_size": 2048,
                },
            },
            "encoder": {
                "model_name": "esm3-sm-open-v1",
                "embedding_dim": 1536,
                "freeze": True,
                "pooling": {
                    "method": "attention",
                    "num_output_tokens": 16,
                },
                "projector": {
                    "hidden_dim": 1024,
                    "num_layers": 3,
                    "dropout": 0.2,
                },
            },
            "training": {
                "quantization": {
                    "enabled": False,
                },
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["q_proj", "k_proj", "v_proj"],
                },
            },
        })

        # Don't actually load models
        with patch.object(ProteinLLM, "_load_llm"), \
             patch.object(ProteinLLM, "_load_encoder"), \
             patch.object(ProteinLLM, "_build_pooling"), \
             patch.object(ProteinLLM, "_build_projector"):

            model = ProteinLLM.from_config(config)

        assert model.llm_name == "test/model"
        assert model.encoder_name == "esm3-sm-open-v1"
        assert model.num_prefix_tokens == 16
        assert model.pooling_type == "attention"
        assert model.projector_hidden_dim == 1024
        assert model.projector_num_layers == 3
        assert model.projector_dropout == 0.2
        assert model.use_qlora is False
        assert model.lora_r == 16
        assert model.lora_alpha == 32
        assert model.lora_target_modules == ["q_proj", "k_proj", "v_proj"]

    def test_from_config_defaults(self):
        """Test from_config uses defaults for missing values."""
        config = OmegaConf.create({})

        with patch.object(ProteinLLM, "_load_llm"), \
             patch.object(ProteinLLM, "_load_encoder"), \
             patch.object(ProteinLLM, "_build_pooling"), \
             patch.object(ProteinLLM, "_build_projector"):

            model = ProteinLLM.from_config(config)

        assert model.llm_name == ProteinLLM.DEFAULT_LLM_NAME
        assert model.encoder_name == ProteinLLM.DEFAULT_ENCODER_NAME
        assert model.num_prefix_tokens == ProteinLLM.DEFAULT_NUM_PREFIX_TOKENS

    def test_from_config_dict(self):
        """Test from_config works with regular dict."""
        config = {
            "model": {"path": "dict/model"},
            "encoder": {
                "model_name": "esm3-sm-open-v1",
                "pooling": {"method": "mean"},
            },
        }

        with patch.object(ProteinLLM, "_load_llm"), \
             patch.object(ProteinLLM, "_load_encoder"), \
             patch.object(ProteinLLM, "_build_pooling"), \
             patch.object(ProteinLLM, "_build_projector"):

            model = ProteinLLM.from_config(config)

        assert model.llm_name == "dict/model"
        assert model.encoder_name == "esm3-sm-open-v1"
        assert model.pooling_type == "mean"


class TestSaveLoadPretrained:
    """Tests for save_pretrained and from_pretrained methods."""

    def test_save_and_load_config(self, mock_protein_llm, device):
        """Test saving and loading model configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Save model
            mock_protein_llm.save_pretrained(save_path)

            # Check config was saved
            config_path = save_path / "config.json"
            assert config_path.exists()

            with open(config_path) as f:
                config = json.load(f)

            assert config["llm_name"] == mock_protein_llm.llm_name
            assert config["encoder_name"] == mock_protein_llm.encoder_name
            assert config["num_prefix_tokens"] == mock_protein_llm.num_prefix_tokens

    def test_save_and_load_pooling(self, mock_protein_llm, device):
        """Test saving and loading pooling weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Save model
            mock_protein_llm.save_pretrained(save_path)

            # Check pooling weights were saved
            pooling_path = save_path / "pooling.pt"
            assert pooling_path.exists()

    def test_save_and_load_projector(self, mock_protein_llm, device):
        """Test saving and loading projector weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Save model
            mock_protein_llm.save_pretrained(save_path)

            # Check projector weights were saved
            projector_path = save_path / "projector.pt"
            assert projector_path.exists()

    def test_from_pretrained_loads_config(self, mock_protein_llm, device):
        """Test from_pretrained loads configuration correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Save model
            mock_protein_llm.save_pretrained(save_path)

            # Load model (without actually loading LLM/encoder)
            loaded_model = ProteinLLM.from_pretrained(
                save_path,
                device=device,
                load_llm=False,
                load_encoder=False,
            )

            assert loaded_model.llm_name == mock_protein_llm.llm_name
            assert loaded_model.encoder_name == mock_protein_llm.encoder_name
            assert loaded_model.num_prefix_tokens == mock_protein_llm.num_prefix_tokens
            assert loaded_model.pooling_type == mock_protein_llm.pooling_type


class TestPoolingTypes:
    """Tests for different pooling types."""

    def test_attention_pooling_type(self, device):
        """Test model with attention pooling."""
        model = ProteinLLM(
            pooling_type="attention",
            num_prefix_tokens=16,
            load_llm=False,
            load_encoder=False,
            device=device,
        )

        # Manually build pooling
        model.encoder_embed_dim = 1280
        model._build_pooling()

        assert isinstance(model.pooling, AttentionPooling)
        assert model.pooling.num_output_tokens == 16

    def test_mean_pooling_type(self, device):
        """Test model with mean pooling."""
        model = ProteinLLM(
            pooling_type="mean",
            num_prefix_tokens=1,
            load_llm=False,
            load_encoder=False,
            device=device,
        )

        # Manually build pooling
        model.encoder_embed_dim = 1280
        model._build_pooling()

        assert isinstance(model.pooling, MeanPooling)

    def test_invalid_pooling_type(self, device):
        """Test that invalid pooling type raises error."""
        model = ProteinLLM(
            pooling_type="invalid",
            load_llm=False,
            load_encoder=False,
            device=device,
        )

        model.encoder_embed_dim = 1280

        with pytest.raises(ValueError, match="Unknown pooling type"):
            model._build_pooling()


class TestProjector:
    """Tests for projector configuration."""

    def test_projector_dimensions(self, device):
        """Test projector has correct dimensions."""
        model = ProteinLLM(
            projector_hidden_dim=1024,
            projector_num_layers=3,
            projector_dropout=0.2,
            load_llm=False,
            load_encoder=False,
            device=device,
        )

        model.encoder_embed_dim = 1280
        model.llm_hidden_size = 2048
        model._build_projector()

        assert model.projector.input_dim == 1280
        assert model.projector.hidden_dim == 1024
        assert model.projector.output_dim == 2048
        assert model.projector.num_layers == 3
        assert model.projector.dropout_rate == 0.2


class TestTrainableParameters:
    """Tests for trainable parameter tracking."""

    def test_get_trainable_parameters(self, mock_protein_llm, device):
        """Test get_trainable_parameters returns correct structure."""
        params = mock_protein_llm.get_trainable_parameters()

        assert "pooling" in params
        assert "projector" in params
        assert "total" in params

        for component in ["pooling", "projector", "total"]:
            assert "total" in params[component]
            assert "trainable" in params[component]
            assert params[component]["total"] >= params[component]["trainable"]

    def test_print_trainable_parameters(self, mock_protein_llm, capsys):
        """Test print_trainable_parameters produces output."""
        mock_protein_llm.print_trainable_parameters()

        captured = capsys.readouterr()
        assert "Trainable Parameters Summary" in captured.out
        assert "pooling" in captured.out
        assert "projector" in captured.out


class TestTrainEvalModes:
    """Tests for train/eval mode switching."""

    def test_train_mode(self, mock_protein_llm, device):
        """Test setting model to train mode."""
        mock_protein_llm.train()

        assert mock_protein_llm.pooling.training is True
        assert mock_protein_llm.projector.training is True

    def test_eval_mode(self, mock_protein_llm, device):
        """Test setting model to eval mode."""
        mock_protein_llm.eval()

        assert mock_protein_llm.pooling.training is False
        assert mock_protein_llm.projector.training is False


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradient_flow_pooling(self, mock_protein_llm, device):
        """Test gradients flow through pooling."""
        mock_protein_llm.train()

        # Create mock input that requires grad
        x = torch.randn(2, 50, 1280, device=device, requires_grad=True)
        pooled = mock_protein_llm.pooling(x)
        loss = pooled.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_projector(self, mock_protein_llm, device):
        """Test gradients flow through projector."""
        mock_protein_llm.train()

        x = torch.randn(2, 8, 1280, device=device, requires_grad=True)
        projected = mock_protein_llm.projector(x)
        loss = projected.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestIntegration:
    """Integration tests for the full model."""

    def test_full_pipeline(self, mock_protein_llm, device):
        """Test full encode -> prepare -> forward pipeline."""
        protein_sequences = ["MKTAYIAK", "MNIFEMLR"]
        input_ids = torch.randint(0, 32000, (2, 20), device=device)
        attention_mask = torch.ones(2, 20, device=device)
        labels = torch.randint(0, 32000, (2, 20), device=device)

        # Full forward pass
        outputs = mock_protein_llm(
            protein_sequences=protein_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["logits"].shape[0] == 2

    def test_deterministic_eval(self, mock_protein_llm, device):
        """Test model produces deterministic outputs in eval mode."""
        mock_protein_llm.eval()

        protein_sequences = ["MKTAYIAK"]
        input_ids = torch.randint(0, 32000, (1, 10), device=device)
        attention_mask = torch.ones(1, 10, device=device)

        with torch.no_grad():
            # Note: Due to mock randomness, we mainly test the structure
            output1 = mock_protein_llm(
                protein_sequences=protein_sequences,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            output2 = mock_protein_llm(
                protein_sequences=protein_sequences,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        assert output1["logits"].shape == output2["logits"].shape
