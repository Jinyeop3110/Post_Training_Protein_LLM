"""Tests for MLP Projector module."""

import pytest
import torch
from omegaconf import OmegaConf

from src.models.projector import MLPProjector, get_projector


class TestMLPProjector:
    """Tests for MLPProjector implementation."""

    def test_default_initialization(self):
        """Test projector initializes with default parameters."""
        projector = MLPProjector()

        assert projector.input_dim == 1280
        assert projector.hidden_dim == 2048
        assert projector.output_dim == 4096
        assert projector.num_layers == 2
        assert projector.activation_name == "gelu"
        assert projector.dropout_rate == 0.1

    def test_custom_initialization(self):
        """Test projector initializes with custom parameters."""
        projector = MLPProjector(
            input_dim=768,
            hidden_dim=1024,
            output_dim=2048,
            num_layers=3,
            activation="relu",
            dropout=0.2,
        )

        assert projector.input_dim == 768
        assert projector.hidden_dim == 1024
        assert projector.output_dim == 2048
        assert projector.num_layers == 3
        assert projector.activation_name == "relu"
        assert projector.dropout_rate == 0.2

    def test_forward_pass_shape(self, device):
        """Test forward pass produces correct output shape."""
        projector = MLPProjector(
            input_dim=1280,
            output_dim=4096,
        ).to(device)

        # Input: [batch_size, seq_len, input_dim]
        x = torch.randn(2, 100, 1280, device=device)
        output = projector(x)

        # Output: [batch_size, seq_len, output_dim]
        assert output.shape == (2, 100, 4096)

    def test_forward_pass_batch_sizes(self, device):
        """Test forward pass with various batch sizes."""
        projector = MLPProjector().to(device)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 50, 1280, device=device)
            output = projector(x)
            assert output.shape == (batch_size, 50, 4096)

    def test_forward_pass_sequence_lengths(self, device):
        """Test forward pass with various sequence lengths."""
        projector = MLPProjector().to(device)

        for seq_len in [1, 10, 100, 500]:
            x = torch.randn(2, seq_len, 1280, device=device)
            output = projector(x)
            assert output.shape == (2, seq_len, 4096)

    def test_single_layer(self, device):
        """Test projector with single layer (direct projection)."""
        projector = MLPProjector(
            input_dim=1280,
            output_dim=4096,
            num_layers=1,
        ).to(device)

        x = torch.randn(2, 50, 1280, device=device)
        output = projector(x)

        assert output.shape == (2, 50, 4096)
        # Single layer should have only one linear layer
        assert len(projector.layers) == 1

    def test_multiple_layers(self, device):
        """Test projector with multiple layers."""
        projector = MLPProjector(
            input_dim=1280,
            hidden_dim=2048,
            output_dim=4096,
            num_layers=4,
            dropout=0.1,
        ).to(device)

        x = torch.randn(2, 50, 1280, device=device)
        output = projector(x)

        assert output.shape == (2, 50, 4096)

    def test_no_dropout(self, device):
        """Test projector with no dropout."""
        projector = MLPProjector(
            input_dim=1280,
            output_dim=4096,
            num_layers=2,
            dropout=0.0,
        ).to(device)

        x = torch.randn(2, 50, 1280, device=device)
        output = projector(x)

        assert output.shape == (2, 50, 4096)


class TestMLPProjectorActivations:
    """Tests for different activation functions."""

    @pytest.mark.parametrize("activation", ["gelu", "relu", "silu", "tanh"])
    def test_supported_activations(self, activation, device):
        """Test projector works with all supported activations."""
        projector = MLPProjector(
            activation=activation,
            num_layers=2,
        ).to(device)

        x = torch.randn(2, 50, 1280, device=device)
        output = projector(x)

        assert output.shape == (2, 50, 4096)

    def test_invalid_activation(self):
        """Test projector raises error for invalid activation."""
        with pytest.raises(ValueError, match="Unknown activation"):
            MLPProjector(activation="invalid_activation")


class TestMLPProjectorValidation:
    """Tests for parameter validation."""

    def test_invalid_input_dim(self):
        """Test projector raises error for invalid input_dim."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            MLPProjector(input_dim=0)

        with pytest.raises(ValueError, match="input_dim must be positive"):
            MLPProjector(input_dim=-1)

    def test_invalid_hidden_dim(self):
        """Test projector raises error for invalid hidden_dim."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            MLPProjector(hidden_dim=0)

    def test_invalid_output_dim(self):
        """Test projector raises error for invalid output_dim."""
        with pytest.raises(ValueError, match="output_dim must be positive"):
            MLPProjector(output_dim=-10)

    def test_invalid_num_layers(self):
        """Test projector raises error for invalid num_layers."""
        with pytest.raises(ValueError, match="num_layers must be at least 1"):
            MLPProjector(num_layers=0)

    def test_invalid_dropout(self):
        """Test projector raises error for invalid dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            MLPProjector(dropout=1.0)

        with pytest.raises(ValueError, match="dropout must be in"):
            MLPProjector(dropout=-0.1)

    def test_invalid_input_tensor_dims(self, device):
        """Test forward raises error for invalid input dimensions."""
        projector = MLPProjector().to(device)

        # 2D tensor (missing batch or sequence dimension)
        x_2d = torch.randn(100, 1280, device=device)
        with pytest.raises(ValueError, match="Expected 3D input tensor"):
            projector(x_2d)

        # 4D tensor
        x_4d = torch.randn(2, 10, 100, 1280, device=device)
        with pytest.raises(ValueError, match="Expected 3D input tensor"):
            projector(x_4d)

    def test_invalid_input_tensor_dim_size(self, device):
        """Test forward raises error for mismatched input dimension."""
        projector = MLPProjector(input_dim=1280).to(device)

        # Wrong embedding dimension
        x = torch.randn(2, 100, 768, device=device)
        with pytest.raises(ValueError, match="Expected input_dim=1280"):
            projector(x)


class TestMLPProjectorFromConfig:
    """Tests for from_config class method."""

    def test_from_config_omegaconf(self):
        """Test creating projector from OmegaConf config."""
        cfg = OmegaConf.create({
            "input_dim": 1280,
            "hidden_dim": 2048,
            "output_dim": 4096,
            "num_layers": 2,
            "activation": "gelu",
            "dropout": 0.1,
        })

        projector = MLPProjector.from_config(cfg)

        assert projector.input_dim == 1280
        assert projector.hidden_dim == 2048
        assert projector.output_dim == 4096
        assert projector.num_layers == 2
        assert projector.activation_name == "gelu"
        assert projector.dropout_rate == 0.1

    def test_from_config_dict(self):
        """Test creating projector from regular dict config."""
        cfg = {
            "input_dim": 768,
            "hidden_dim": 1024,
            "output_dim": 2048,
            "num_layers": 3,
            "activation": "relu",
            "dropout": 0.2,
        }

        projector = MLPProjector.from_config(cfg)

        assert projector.input_dim == 768
        assert projector.hidden_dim == 1024
        assert projector.output_dim == 2048
        assert projector.num_layers == 3
        assert projector.activation_name == "relu"
        assert projector.dropout_rate == 0.2

    def test_from_config_partial(self):
        """Test creating projector with partial config (uses defaults)."""
        cfg = OmegaConf.create({
            "input_dim": 1280,
            "output_dim": 4096,
        })

        projector = MLPProjector.from_config(cfg)

        # Specified values
        assert projector.input_dim == 1280
        assert projector.output_dim == 4096

        # Default values
        assert projector.hidden_dim == 2048
        assert projector.num_layers == 2
        assert projector.activation_name == "gelu"
        assert projector.dropout_rate == 0.1

    def test_from_config_empty(self):
        """Test creating projector with empty config (all defaults)."""
        cfg = OmegaConf.create({})

        projector = MLPProjector.from_config(cfg)

        assert projector.input_dim == 1280
        assert projector.hidden_dim == 2048
        assert projector.output_dim == 4096
        assert projector.num_layers == 2
        assert projector.activation_name == "gelu"
        assert projector.dropout_rate == 0.1

    def test_from_config_forward_pass(self, device):
        """Test projector created from config works correctly."""
        cfg = OmegaConf.create({
            "input_dim": 1280,
            "hidden_dim": 2048,
            "output_dim": 4096,
            "num_layers": 2,
        })

        projector = MLPProjector.from_config(cfg).to(device)
        x = torch.randn(2, 100, 1280, device=device)
        output = projector(x)

        assert output.shape == (2, 100, 4096)


class TestGetProjector:
    """Tests for get_projector factory function."""

    def test_get_mlp_projector(self):
        """Test getting MLP projector via factory."""
        projector = get_projector("mlp", input_dim=1280, output_dim=4096)

        assert isinstance(projector, MLPProjector)
        assert projector.input_dim == 1280
        assert projector.output_dim == 4096

    def test_get_mlp_projector_case_insensitive(self):
        """Test factory is case insensitive."""
        projector_lower = get_projector("mlp")
        projector_upper = get_projector("MLP")
        projector_mixed = get_projector("Mlp")

        assert isinstance(projector_lower, MLPProjector)
        assert isinstance(projector_upper, MLPProjector)
        assert isinstance(projector_mixed, MLPProjector)

    def test_invalid_projector_type(self):
        """Test factory raises error for invalid type."""
        with pytest.raises(ValueError, match="Unknown projector type"):
            get_projector("invalid_type")


class TestMLPProjectorProperties:
    """Tests for projector properties and methods."""

    def test_get_output_dim(self):
        """Test get_output_dim returns correct value."""
        projector = MLPProjector(output_dim=4096)
        assert projector.get_output_dim() == 4096

    def test_get_input_dim(self):
        """Test get_input_dim returns correct value."""
        projector = MLPProjector(input_dim=1280)
        assert projector.get_input_dim() == 1280

    def test_extra_repr(self):
        """Test extra_repr returns informative string."""
        projector = MLPProjector(
            input_dim=1280,
            hidden_dim=2048,
            output_dim=4096,
            num_layers=2,
            activation="gelu",
            dropout=0.1,
        )

        repr_str = projector.extra_repr()

        assert "input_dim=1280" in repr_str
        assert "hidden_dim=2048" in repr_str
        assert "output_dim=4096" in repr_str
        assert "num_layers=2" in repr_str
        assert "activation=gelu" in repr_str
        assert "dropout=0.1" in repr_str


class TestMLPProjectorGradients:
    """Tests for gradient flow through the projector."""

    def test_gradient_flow(self, device):
        """Test gradients flow correctly through the projector."""
        projector = MLPProjector().to(device)
        projector.train()

        x = torch.randn(2, 50, 1280, device=device, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameters_have_gradients(self, device):
        """Test all parameters receive gradients."""
        projector = MLPProjector().to(device)
        projector.train()

        x = torch.randn(2, 50, 1280, device=device)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        for name, param in projector.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


class TestMLPProjectorModes:
    """Tests for train/eval modes."""

    def test_eval_mode_deterministic(self, device):
        """Test projector is deterministic in eval mode."""
        projector = MLPProjector(dropout=0.5).to(device)
        projector.eval()

        x = torch.randn(2, 50, 1280, device=device)

        with torch.no_grad():
            output1 = projector(x)
            output2 = projector(x)

        assert torch.allclose(output1, output2)

    def test_train_mode_with_dropout(self, device):
        """Test projector has stochastic behavior in train mode with dropout."""
        projector = MLPProjector(dropout=0.5, num_layers=2).to(device)
        projector.train()

        x = torch.randn(2, 50, 1280, device=device)

        # Run multiple times - with high dropout, outputs should differ
        outputs = [projector(x) for _ in range(5)]

        # At least some outputs should be different
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Train mode with dropout should produce different outputs"
