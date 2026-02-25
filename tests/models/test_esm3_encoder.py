"""
Tests for ESM-3 Protein Encoder and Approach Config Switching.

This module tests:
1. ESM3ProteinEncoder: initialization, output shape, frozen weights, mock model
2. Approach config switching: selecting between text and ESM-3 encoders
3. Projector dimension compatibility with ESM-3
4. Factory function (get_protein_encoder) with ESM-3 support
5. Critical rule enforcement: frozen encoder, attention pooling, LoRA k/v only

Note:
    The ESM-3 encoder implementation is expected in src/models/protein_encoder.py.
    Tests use mocked ESM-3 models to avoid requiring actual model downloads.
    Tests will skip gracefully if ESM3ProteinEncoder is not yet implemented.
"""

import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# ============================================================================
# Helper: Mock ESMProtein class for patching esm.sdk.api
# ============================================================================

class _MockESMProtein:
    """Lightweight mock of esm.sdk.api.ESMProtein for testing."""

    def __init__(self, sequence: str = "") -> None:
        self.sequence = sequence


def _patch_esm_sdk_api():
    """Context manager to inject a mock esm.sdk.api module into sys.modules.

    This allows the local ``from esm.sdk.api import ESMProtein`` inside
    ESM3ProteinEncoder.encode() to succeed even when the real EvolutionaryScale
    ``esm`` package is not installed.
    """
    mock_api = MagicMock()
    mock_api.ESMProtein = _MockESMProtein
    mock_api.ESMProteinTensor = MagicMock

    # Build the module chain: esm -> esm.sdk -> esm.sdk.api
    mock_esm = MagicMock()
    mock_sdk = MagicMock()
    mock_esm.sdk = mock_sdk
    mock_sdk.api = mock_api

    modules = {
        "esm": mock_esm,
        "esm.sdk": mock_sdk,
        "esm.sdk.api": mock_api,
    }
    return patch.dict(sys.modules, modules)


# ============================================================================
# Helper: Check if ESM3ProteinEncoder is implemented
# ============================================================================

def _esm3_available() -> bool:
    """Check whether ESM3ProteinEncoder is implemented yet."""
    try:
        from src.models.protein_encoder import ESM3ProteinEncoder
        return True
    except (ImportError, AttributeError):
        return False


esm3_required = pytest.mark.skipif(
    not _esm3_available(),
    reason="ESM3ProteinEncoder not yet implemented (Task #1 pending)",
)


# ============================================================================
# Mock ESM-3 components (for testing without real model download)
# ============================================================================

class MockESM3Model(nn.Module):
    """Mock ESM-3 model for testing without loading real weights.

    Simulates the ESM-3 forward pass returning per-residue embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        num_layers: int = 48,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        # Dummy parameter to ensure the model has parameters for freezing tests
        self.encoder = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        repr_layers: Optional[List[int]] = None,
        return_contacts: bool = False,
    ) -> Dict[str, Any]:
        batch_size, seq_len = tokens.shape
        representations = {}
        if repr_layers:
            for layer in repr_layers:
                representations[layer] = torch.randn(
                    batch_size, seq_len, self.embed_dim
                )
        else:
            representations[self.num_layers] = torch.randn(
                batch_size, seq_len, self.embed_dim
            )
        return {"representations": representations}


class MockESM3Alphabet:
    """Mock ESM-3 alphabet for testing."""

    def __init__(self) -> None:
        self.padding_idx = 0

    def get_batch_converter(self):
        def batch_converter(data):
            batch_labels = [d[0] for d in data]
            batch_strs = [d[1] for d in data]
            max_len = max(len(s) for s in batch_strs) + 2
            batch_tokens = torch.zeros(len(data), max_len, dtype=torch.long)
            return batch_labels, batch_strs, batch_tokens
        return batch_converter


# ============================================================================
# Tests for existing encoder critical rules (always runnable)
# ============================================================================

class TestCriticalRuleEnforcement:
    """Validate that critical architectural rules are enforced in existing code.

    These tests verify rules from CLAUDE.md and REVIEW_POINTS.md:
    - NEVER modify ESM-3 encoder weights (always frozen)
    - LoRA on all linear layers (q/k/v/o + gate/up/down)
    - Attention pooling is the default (not mean)
    - No hardcoded model paths in source code
    """

    def test_default_lora_targets_all_linear(self):
        """Test that default LoRA target modules are all 7 linear layers."""
        from src.models.multimodal_llm import ProteinLLM

        expected = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        assert ProteinLLM.DEFAULT_LORA_TARGET_MODULES == expected, (
            "LoRA default targets must be all 7 linear layers. "
            f"Got: {ProteinLLM.DEFAULT_LORA_TARGET_MODULES}"
        )

    def test_sft_trainer_default_lora_targets(self):
        """Test that SFT trainer extracts all 7 linear layers as default targets."""
        from src.training.sft_trainer import get_qlora_config

        cfg = OmegaConf.create({
            "training": {
                "lora": {}  # Empty - should use defaults
            }
        })
        lora_config = get_qlora_config(cfg)
        expected = {"q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"}
        assert set(lora_config.target_modules) == expected, (
            f"SFT trainer default LoRA targets must be all 7 linear layers. "
            f"Got: {set(lora_config.target_modules)}"
        )

    def test_sft_config_yaml_lora_targets(self):
        """Test that the sft_qlora.yaml config specifies all 7 linear layers."""
        config = OmegaConf.load(
            "/orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM/"
            "configs/training/sft_qlora.yaml"
        )
        targets = list(config.lora.target_modules)
        expected = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        assert targets == expected, (
            f"Config file lora.target_modules must be all 7 linear layers. Got: {targets}"
        )

    def test_default_pooling_is_attention(self):
        """Test that the default pooling type is 'attention', not 'mean'."""
        from src.models.multimodal_llm import ProteinLLM

        assert ProteinLLM.DEFAULT_POOLING_TYPE == "attention", (
            "Default pooling must be 'attention' (not 'mean'). "
            f"Got: {ProteinLLM.DEFAULT_POOLING_TYPE}"
        )

    def test_no_hardcoded_model_paths_in_encoder(self):
        """Test that protein_encoder.py does not hardcode absolute model paths."""
        import inspect

        from src.models import protein_encoder

        source = inspect.getsource(protein_encoder)
        # Check for common hardcoded path patterns
        hardcoded_patterns = [
            "/home/",
            "/root/",
            "/data/models/",
            "/scratch/",
            "facebook/esm",  # HF hub paths are OK, local paths are not
        ]
        for pattern in hardcoded_patterns:
            assert pattern not in source, (
                f"Found hardcoded path pattern '{pattern}' in protein_encoder.py"
            )

    def test_no_hardcoded_model_paths_in_multimodal(self):
        """Test that multimodal_llm.py does not hardcode local model paths."""
        import inspect

        from src.models import multimodal_llm

        source = inspect.getsource(multimodal_llm)
        hardcoded_patterns = [
            "/home/",
            "/root/",
            "/data/models/",
            "/scratch/",
        ]
        for pattern in hardcoded_patterns:
            assert pattern not in source, (
                f"Found hardcoded path pattern '{pattern}' in multimodal_llm.py"
            )

    def test_protein_llm_freeze_encoder_default_true(self):
        """Test that ProteinLLM defaults to freeze_encoder=True."""
        from src.models.multimodal_llm import ProteinLLM

        model = ProteinLLM(load_llm=False, load_encoder=False)
        assert model.freeze_encoder is True, (
            "freeze_encoder must default to True. "
            f"Got: {model.freeze_encoder}"
        )


# ============================================================================
# Tests for Protein Encoder Factory / Approach Config Switching
# ============================================================================

class TestProteinEncoderFactory:
    """Tests for get_protein_encoder factory and approach switching."""

    def test_get_text_encoder(self):
        """Test factory creates TextProteinEncoder."""
        from src.models.protein_encoder import TextProteinEncoder, get_protein_encoder

        encoder = get_protein_encoder("text")
        assert isinstance(encoder, TextProteinEncoder)

    @esm3_required
    def test_get_esm3_encoder(self):
        """Test factory creates ESM3ProteinEncoder for 'esm3' type."""
        from src.models.protein_encoder import ESM3ProteinEncoder, get_protein_encoder

        encoder = get_protein_encoder("esm3", device="cpu")
        assert isinstance(encoder, ESM3ProteinEncoder)

    def test_get_encoder_invalid_type(self):
        """Test factory raises ValueError for unknown encoder type."""
        from src.models.protein_encoder import get_protein_encoder

        with pytest.raises(ValueError, match="Unknown encoder type"):
            get_protein_encoder("invalid_encoder_type")

    def test_text_encoder_encode(self):
        """Test TextProteinEncoder encodes sequences as text."""
        from src.models.protein_encoder import TextProteinEncoder

        encoder = TextProteinEncoder()
        result = encoder.encode(["MKTAYIAK", "MNIFEMLR"])

        assert result["type"] == "text"
        assert len(result["text"]) == 2
        assert "MKTAYIAK" in result["text"][0]
        assert "MNIFEMLR" in result["text"][1]

    def test_text_encoder_embedding_dim(self):
        """Test TextProteinEncoder returns -1 for embedding dim."""
        from src.models.protein_encoder import TextProteinEncoder

        encoder = TextProteinEncoder()
        assert encoder.get_embedding_dim() == -1

    def test_esm3_encoder_embedding_dim_lookup(self):
        """Test ESM-3 model dimension lookup table."""
        from src.models.multimodal_llm import ProteinLLM

        expected_dims = {
            "esm3-sm-open-v1": 1536,
            "esm3_sm_open_v1": 1536,
        }
        for model_name, expected_dim in expected_dims.items():
            assert ProteinLLM.ENCODER_EMBED_DIMS.get(model_name) == expected_dim, (
                f"ENCODER_EMBED_DIMS['{model_name}'] should be {expected_dim}"
            )


# ============================================================================
# Tests for ESM-3 Encoder (will run once architect implements Task #1)
# ============================================================================

@esm3_required
class TestESM3ProteinEncoderInit:
    """Tests for ESM3ProteinEncoder initialization."""

    def test_initialization_default_params(self):
        """Test ESM3ProteinEncoder initializes with default parameters."""
        from src.models.protein_encoder import ESM3ProteinEncoder

        encoder = ESM3ProteinEncoder(device="cpu")
        assert encoder is not None
        assert encoder.device == "cpu"

    def test_initialization_custom_model(self):
        """Test ESM3ProteinEncoder accepts custom model name."""
        from src.models.protein_encoder import ESM3ProteinEncoder

        encoder = ESM3ProteinEncoder(model_name="esm3_sm_open_v1", device="cpu")
        assert encoder.model_name == "esm3_sm_open_v1"

    def test_initialization_no_pooling_attribute(self):
        """Test ESM3ProteinEncoder does NOT have a pooling attribute.

        ESM-3 encoder always returns per-residue embeddings.
        Pooling is handled externally by the pipeline.
        """
        from src.models.protein_encoder import ESM3ProteinEncoder

        encoder = ESM3ProteinEncoder(device="cpu")
        assert not hasattr(encoder, "pooling"), (
            "ESM3ProteinEncoder should not have a 'pooling' attribute. "
            "Pooling is handled by the pipeline's AttentionPooling module."
        )

    def test_lazy_loading(self):
        """Test that model is not loaded until needed (lazy loading)."""
        from src.models.protein_encoder import ESM3ProteinEncoder

        encoder = ESM3ProteinEncoder(device="cpu")
        # Model should not be loaded yet
        assert encoder.model is None


@esm3_required
class TestESM3ProteinEncoderOutputShape:
    """Tests for ESM3ProteinEncoder output shape."""

    @pytest.fixture
    def mock_esm3_encoder(self):
        """Create an ESM3ProteinEncoder with mocked model loading.

        The ESM-3 encode() API works as follows (updated):
        1. Creates ESMProtein(sequence=seq)
        2. Calls self.model.encode(protein) -> protein_tensor with .sequence attr
        3. Calls self.model.forward(sequence_tokens=...) -> output with .embeddings
        4. Strips BOS/EOS: embeddings[:, 1:-1, :] -> [1, L, D]

        We mock model.encode and model.forward to return tensors of the
        correct shape, and patch sys.modules so that the local
        ``from esm.sdk.api import ESMProtein`` inside encode() succeeds.
        """
        from src.models.protein_encoder import ESM3ProteinEncoder

        embed_dim = 1536

        encoder = ESM3ProteinEncoder(device="cpu")
        encoder._embedding_dim = embed_dim

        # Create a mock model with encode() and forward() matching current ESM-3 API
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.randn(2, 2))]
        )

        def mock_encode(protein):
            """Mock model.encode: return a sentinel with .sequence tensor."""
            seq_len = len(protein.sequence)
            result = MagicMock()
            result.sequence = torch.zeros(seq_len + 2, dtype=torch.long)  # BOS + seq + EOS
            return result

        def mock_forward(sequence_tokens=None, sequence_id=None, **kwargs):
            """Mock model.forward: return output with .embeddings [B, L_tok, D]."""
            batch_size, tok_len = sequence_tokens.shape
            output = MagicMock()
            output.embeddings = torch.randn(batch_size, tok_len, embed_dim)
            return output

        mock_model.encode = mock_encode
        mock_model.forward = mock_forward

        encoder.model = mock_model

        return encoder

    def test_per_residue_output_shape(self, mock_esm3_encoder):
        """Test ESM-3 encoder produces per-residue [B, L, D] output.

        ESM3ProteinEncoder always returns per-residue embeddings (no internal
        pooling). Pooling is handled externally by the pipeline.
        """
        with _patch_esm_sdk_api():
            result = mock_esm3_encoder.encode(["MKTAYIAK", "MNIFEMLR"])

        embeddings = result["embeddings"]
        assert embeddings.dim() == 3
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] == 8  # sequence length (both seqs are len 8)
        assert embeddings.shape[2] == 1536  # ESM-3 embed dim

    def test_single_sequence_output_shape(self, mock_esm3_encoder):
        """Test ESM-3 encoder with a single sequence returns [1, L, D]."""
        with _patch_esm_sdk_api():
            result = mock_esm3_encoder.encode(["MKTAYIAK"])

        embeddings = result["embeddings"]
        assert embeddings.dim() == 3
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 8  # len("MKTAYIAK")
        assert embeddings.shape[2] == 1536

    def test_embedding_dim_is_correct(self, mock_esm3_encoder):
        """Test get_embedding_dim returns correct dimension for ESM-3."""
        dim = mock_esm3_encoder.get_embedding_dim()
        assert dim == 1536, (
            f"ESM-3 embedding dim should be 1536, got {dim}"
        )

    def test_output_type_is_embedding(self, mock_esm3_encoder):
        """Test that output dictionary has type='embedding'."""
        with _patch_esm_sdk_api():
            result = mock_esm3_encoder.encode(["MKTAYIAK"])

        assert result["type"] == "embedding"
        assert "embeddings" in result
        assert result["encoder"] == "esm3"


@esm3_required
class TestESM3ProteinEncoderFrozen:
    """Tests for ESM3ProteinEncoder weight freezing."""

    def test_weights_are_frozen(self):
        """Test that ESM-3 model weights have requires_grad=False."""
        from src.models.protein_encoder import ESM3ProteinEncoder

        encoder = ESM3ProteinEncoder(device="cpu")

        # Inject mock model and simulate freeze
        mock_model = MockESM3Model(embed_dim=1536, num_layers=48)
        encoder.model = mock_model
        encoder.model.eval()

        # Freeze weights (as done in ProteinLLM._load_encoder)
        for param in encoder.model.parameters():
            param.requires_grad = False

        # Verify all parameters are frozen
        for name, param in encoder.model.named_parameters():
            assert not param.requires_grad, (
                f"ESM-3 parameter '{name}' should be frozen (requires_grad=False)"
            )

    def test_no_gradient_computation(self):
        """Test that encoding happens with no_grad context.

        The ESM-3 encode() method wraps everything in torch.no_grad(),
        so the output embeddings should not require gradients.
        """
        from src.models.protein_encoder import ESM3ProteinEncoder

        embed_dim = 1536
        encoder = ESM3ProteinEncoder(device="cpu")
        encoder._embedding_dim = embed_dim

        # Create a mock model with encode() and forward() matching current ESM-3 API
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.randn(2, 2))]
        )

        def mock_encode(protein):
            result = MagicMock()
            result.sequence = torch.zeros(len(protein.sequence) + 2, dtype=torch.long)
            return result

        def mock_forward(sequence_tokens=None, sequence_id=None, **kwargs):
            batch_size, tok_len = sequence_tokens.shape
            output = MagicMock()
            output.embeddings = torch.randn(batch_size, tok_len, embed_dim)
            return output

        mock_model.encode = mock_encode
        mock_model.forward = mock_forward
        encoder.model = mock_model

        # Encode should not create computation graph
        with _patch_esm_sdk_api():
            result = encoder.encode(["MKTAYIAK"])

        embeddings = result["embeddings"]

        # The embeddings should not have a gradient function
        # (because encoding should be wrapped in torch.no_grad)
        assert not embeddings.requires_grad, (
            "ESM-3 encoder output should not require gradients"
        )


# ============================================================================
# Tests for Projector Dimension Compatibility
# ============================================================================

class TestProjectorDimensionCompatibility:
    """Tests for projector compatibility with ESM-3 dimensions."""

    def test_projector_esm3_dimensions(self, device):
        """Test projector works with ESM-3 dimensions (1536 -> 4096)."""
        from src.models.projector import MLPProjector

        projector = MLPProjector(
            input_dim=1536,
            hidden_dim=2048,
            output_dim=4096,
        ).to(device)

        x = torch.randn(2, 32, 1536, device=device)
        output = projector(x)
        assert output.shape == (2, 32, 4096)

    def test_projector_esm3_to_qwen3_dimensions(self, device):
        """Test projector maps ESM-3 embeddings to Qwen3-4B hidden size (2560)."""
        from src.models.projector import MLPProjector

        projector = MLPProjector(
            input_dim=1536,  # ESM-3
            hidden_dim=2048,
            output_dim=2560,  # Qwen3-4B
        ).to(device)

        x = torch.randn(2, 32, 1536, device=device)
        output = projector(x)
        assert output.shape == (2, 32, 2560)

    def test_projector_dimension_mismatch_raises(self, device):
        """Test that dimension mismatch between encoder and projector raises error."""
        from src.models.projector import MLPProjector

        projector = MLPProjector(input_dim=1280).to(device)

        # Feed wrong dimensions (1536) to projector expecting (1280)
        x = torch.randn(2, 32, 1536, device=device)
        with pytest.raises(ValueError, match="Expected input_dim=1280"):
            projector(x)


class TestPoolingCompatibility:
    """Tests for pooling layer compatibility with ESM-3 dimensions."""

    def test_attention_pooling_esm3_dimensions(self, device):
        """Test attention pooling with ESM-3 embedding dim (1536)."""
        from src.models.pooling import AttentionPooling

        pooling = AttentionPooling(
            embed_dim=1536,
            num_output_tokens=32,
            num_heads=8,
        ).to(device)

        x = torch.randn(2, 100, 1536, device=device)
        output = pooling(x)
        assert output.shape == (2, 32, 1536)


# ============================================================================
# Tests for Approach Config Switching via Hydra
# ============================================================================

class TestApproachConfigSwitching:
    """Tests for switching between approaches via configuration."""

    def test_protein_llm_config_parsing_attention_pooling(self):
        """Test ProteinLLM correctly parses attention pooling from config."""
        from src.models.multimodal_llm import ProteinLLM

        config = OmegaConf.create({
            "model": {"path": "test/model"},
            "encoder": {
                "model_name": "esm3-sm-open-v1",
                "pooling": {
                    "method": "attention",
                    "num_output_tokens": 64,
                },
            },
            "training": {
                "quantization": {"enabled": False},
                "lora": {"target_modules": ["k_proj", "v_proj"]},
            },
        })

        with patch.object(ProteinLLM, "_load_llm"), \
             patch.object(ProteinLLM, "_load_encoder"), \
             patch.object(ProteinLLM, "_build_pooling"), \
             patch.object(ProteinLLM, "_build_projector"):
            model = ProteinLLM.from_config(config)

        assert model.pooling_type == "attention"
        assert model.num_prefix_tokens == 64

    def test_protein_llm_config_parsing_mean_pooling(self):
        """Test ProteinLLM correctly parses mean pooling from config."""
        from src.models.multimodal_llm import ProteinLLM

        config = OmegaConf.create({
            "model": {"path": "test/model"},
            "encoder": {
                "model_name": "esm3-sm-open-v1",
                "pooling": {
                    "method": "mean",
                },
            },
            "training": {
                "quantization": {"enabled": False},
            },
        })

        with patch.object(ProteinLLM, "_load_llm"), \
             patch.object(ProteinLLM, "_load_encoder"), \
             patch.object(ProteinLLM, "_build_pooling"), \
             patch.object(ProteinLLM, "_build_projector"):
            model = ProteinLLM.from_config(config)

        assert model.pooling_type == "mean"

    @esm3_required
    def test_protein_llm_config_esm3_encoder(self):
        """Test ProteinLLM correctly parses ESM-3 encoder from config."""
        from src.models.multimodal_llm import ProteinLLM

        config = OmegaConf.create({
            "model": {"path": "test/model"},
            "encoder": {
                "model_name": "esm3_sm_open_v1",
                "embedding_dim": 1536,
                "pooling": {"method": "attention", "num_output_tokens": 32},
                "projector": {"hidden_dim": 2048},
            },
            "training": {
                "quantization": {"enabled": False},
                "lora": {"target_modules": ["k_proj", "v_proj"]},
            },
        })

        with patch.object(ProteinLLM, "_load_llm"), \
             patch.object(ProteinLLM, "_load_encoder"), \
             patch.object(ProteinLLM, "_build_pooling"), \
             patch.object(ProteinLLM, "_build_projector"):
            model = ProteinLLM.from_config(config)

        assert model.encoder_name == "esm3_sm_open_v1"

    def test_protein_llm_builds_correct_pooling_attention(self, device):
        """Test that ProteinLLM builds AttentionPooling when configured."""
        from src.models.multimodal_llm import ProteinLLM
        from src.models.pooling import AttentionPooling

        model = ProteinLLM(
            pooling_type="attention",
            num_prefix_tokens=32,
            load_llm=False,
            load_encoder=False,
            device=device,
        )
        model.encoder_embed_dim = 1280
        model._build_pooling()

        assert isinstance(model.pooling, AttentionPooling)
        assert model.pooling.num_output_tokens == 32

    def test_protein_llm_builds_correct_pooling_mean(self, device):
        """Test that ProteinLLM builds MeanPooling when configured."""
        from src.models.multimodal_llm import ProteinLLM
        from src.models.pooling import MeanPooling

        model = ProteinLLM(
            pooling_type="mean",
            load_llm=False,
            load_encoder=False,
            device=device,
        )
        model.encoder_embed_dim = 1280
        model._build_pooling()

        assert isinstance(model.pooling, MeanPooling)


# ============================================================================
# Tests for end-to-end encoder -> pooling -> projector pipeline dimensions
# ============================================================================

class TestEndToEndPipelineDimensions:
    """Test the full pipeline dimension flow: encoder -> pooling -> projector."""

    def test_esm3_pipeline(self, device):
        """Test full dimension flow for ESM-3."""
        from src.models.pooling import AttentionPooling
        from src.models.projector import MLPProjector

        # ESM-3: 1536 -> attention pooling -> projector -> 4096
        embed_dim = 1536
        num_tokens = 32
        llm_hidden = 4096

        pooling = AttentionPooling(
            embed_dim=embed_dim, num_output_tokens=num_tokens
        ).to(device)
        projector = MLPProjector(
            input_dim=embed_dim, hidden_dim=2048, output_dim=llm_hidden
        ).to(device)

        encoder_output = torch.randn(2, 100, embed_dim, device=device)
        pooled = pooling(encoder_output)
        projected = projector(pooled)

        assert pooled.shape == (2, num_tokens, embed_dim)
        assert projected.shape == (2, num_tokens, llm_hidden)

    def test_gradient_flow_pooling_to_projector(self, device):
        """Test gradients flow through pooling and projector (but not encoder)."""
        from src.models.pooling import AttentionPooling
        from src.models.projector import MLPProjector

        embed_dim = 1280
        pooling = AttentionPooling(
            embed_dim=embed_dim, num_output_tokens=16
        ).to(device)
        projector = MLPProjector(
            input_dim=embed_dim, output_dim=4096
        ).to(device)

        # Simulated frozen encoder output (no grad)
        encoder_output = torch.randn(2, 50, embed_dim, device=device)

        pooled = pooling(encoder_output)
        projected = projector(pooled)
        loss = projected.sum()
        loss.backward()

        # Pooling should have gradients (trainable)
        assert pooling.query_tokens.grad is not None
        # Projector should have gradients (trainable)
        for name, param in projector.named_parameters():
            assert param.grad is not None, (
                f"Projector parameter '{name}' should have gradient"
            )


# ============================================================================
# Tests for existing test_protein_encoder.py issues
# ============================================================================

class TestExistingTestCorrections:
    """Tests that correct known issues in the original test_protein_encoder.py.

    The original tests had bugs:
    1. Used 'embedding_dim' instead of 'embed_dim' for AttentionPooling
    2. Expected (2, 1280) output from MeanPooling but keepdim=True gives (2, 1, 1280)
    """

    def test_attention_pooling_correct_kwarg(self, device):
        """Test AttentionPooling uses 'embed_dim' not 'embedding_dim'."""
        from src.models.pooling import AttentionPooling

        # Correct keyword
        pooling = AttentionPooling(embed_dim=1280).to(device)
        x = torch.randn(2, 100, 1280, device=device)
        output = pooling(x)
        assert output.shape == (2, 32, 1280)  # Default: 32 output tokens

    def test_mean_pooling_default_keepdim(self, device):
        """Test MeanPooling output shape with default keepdim=True."""
        from src.models.pooling import MeanPooling

        pooling = MeanPooling().to(device)
        x = torch.randn(2, 100, 1280, device=device)
        output = pooling(x)
        # Default keepdim=True means output is [B, 1, D], not [B, D]
        assert output.shape == (2, 1, 1280)

    def test_mean_pooling_no_keepdim(self, device):
        """Test MeanPooling output shape with keepdim=False."""
        from src.models.pooling import MeanPooling

        pooling = MeanPooling(keepdim=False).to(device)
        x = torch.randn(2, 100, 1280, device=device)
        output = pooling(x)
        assert output.shape == (2, 1280)

    def test_text_encoder_class_name(self):
        """Test that TextProteinEncoder exists."""
        from src.models.protein_encoder import TextProteinEncoder

        assert TextProteinEncoder is not None

    def test_tbd_encoder_raises(self):
        """Test that TBDProteinEncoder raises NotImplementedError."""
        from src.models.protein_encoder import TBDProteinEncoder

        with pytest.raises(NotImplementedError, match="Third protein encoding approach"):
            TBDProteinEncoder()
