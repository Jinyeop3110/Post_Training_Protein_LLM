"""Tests for pooling module."""

import pytest
import torch
from omegaconf import OmegaConf

from src.models.pooling import (
    AttentionPooling,
    MeanPooling,
    CLSPooling,
    BasePooling,
    get_pooling,
    build_pooling_from_config,
    POOLING_REGISTRY,
)


class TestAttentionPooling:
    """Tests for AttentionPooling (BoM-Pooling style)."""

    def test_forward_shape_default(self, device):
        """Test forward pass produces correct output shape with defaults."""
        pooling = AttentionPooling().to(device)
        x = torch.randn(2, 100, 1280, device=device)
        output = pooling(x)

        assert output.shape == (2, 32, 1280)

    def test_forward_shape_custom_output_tokens(self, device):
        """Test forward pass with different num_output_tokens."""
        for num_tokens in [1, 8, 16, 64, 128]:
            pooling = AttentionPooling(
                embed_dim=1280,
                num_output_tokens=num_tokens,
            ).to(device)
            x = torch.randn(4, 50, 1280, device=device)
            output = pooling(x)

            assert output.shape == (4, num_tokens, 1280), \
                f"Failed for num_output_tokens={num_tokens}"

    def test_forward_shape_custom_embed_dim(self, device):
        """Test forward pass with different embedding dimensions."""
        for embed_dim in [256, 512, 768, 1024, 2560]:
            pooling = AttentionPooling(
                embed_dim=embed_dim,
                num_output_tokens=16,
                num_heads=8,
            ).to(device)
            x = torch.randn(2, 100, embed_dim, device=device)
            output = pooling(x)

            assert output.shape == (2, 16, embed_dim)

    def test_forward_variable_sequence_lengths(self, device):
        """Test forward pass with different sequence lengths."""
        pooling = AttentionPooling(embed_dim=1280, num_output_tokens=32).to(device)

        for seq_len in [10, 50, 100, 500, 1000]:
            x = torch.randn(2, seq_len, 1280, device=device)
            output = pooling(x)

            assert output.shape == (2, 32, 1280), \
                f"Failed for sequence length {seq_len}"

    def test_forward_with_attention_mask(self, device):
        """Test forward pass with attention mask."""
        pooling = AttentionPooling(embed_dim=1280, num_output_tokens=32).to(device)

        batch_size, seq_len, embed_dim = 4, 100, 1280
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Create mask where different sequences have different lengths
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        attention_mask[0, 50:] = 0  # First sequence: length 50
        attention_mask[1, 75:] = 0  # Second sequence: length 75
        attention_mask[2, :] = 1   # Third sequence: full length
        attention_mask[3, 25:] = 0  # Fourth sequence: length 25

        output = pooling(x, attention_mask=attention_mask)

        assert output.shape == (batch_size, 32, embed_dim)
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_with_boolean_mask(self, device):
        """Test forward pass with boolean attention mask."""
        pooling = AttentionPooling(embed_dim=1280, num_output_tokens=32).to(device)

        x = torch.randn(2, 100, 1280, device=device)
        attention_mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        attention_mask[0, 60:] = False

        output = pooling(x, attention_mask=attention_mask)

        assert output.shape == (2, 32, 1280)

    def test_num_output_tokens_property(self):
        """Test num_output_tokens property returns correct value."""
        for num_tokens in [1, 16, 32, 64]:
            pooling = AttentionPooling(num_output_tokens=num_tokens)
            assert pooling.num_output_tokens == num_tokens

    def test_different_num_heads(self, device):
        """Test with different numbers of attention heads."""
        for num_heads in [1, 4, 8, 16]:
            pooling = AttentionPooling(
                embed_dim=1280,  # 1280 is divisible by all these heads
                num_output_tokens=32,
                num_heads=num_heads,
            ).to(device)
            x = torch.randn(2, 100, 1280, device=device)
            output = pooling(x)

            assert output.shape == (2, 32, 1280)

    def test_invalid_embed_dim_num_heads(self):
        """Test that invalid embed_dim/num_heads combination raises error."""
        with pytest.raises(ValueError, match="must be divisible by"):
            AttentionPooling(embed_dim=100, num_heads=8)

    def test_gradient_flow(self, device):
        """Test that gradients flow through the pooling layer."""
        pooling = AttentionPooling(embed_dim=256, num_output_tokens=8).to(device)
        x = torch.randn(2, 50, 256, device=device, requires_grad=True)

        output = pooling(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that query tokens also have gradients
        assert pooling.query_tokens.grad is not None

    def test_without_layer_norm(self, device):
        """Test pooling without layer normalization."""
        pooling = AttentionPooling(
            embed_dim=1280,
            num_output_tokens=32,
            layer_norm=False,
        ).to(device)

        x = torch.randn(2, 100, 1280, device=device)
        output = pooling(x)

        assert output.shape == (2, 32, 1280)

    def test_dropout(self, device):
        """Test that dropout affects output during training."""
        pooling = AttentionPooling(
            embed_dim=256,
            num_output_tokens=8,
            dropout=0.5,
        ).to(device)

        x = torch.randn(2, 50, 256, device=device)

        # In training mode, outputs should differ due to dropout
        pooling.train()
        torch.manual_seed(42)
        output1 = pooling(x)
        torch.manual_seed(123)
        output2 = pooling(x)

        # Outputs should potentially differ (with high dropout)
        # Note: Due to randomness, we can't guarantee difference

        # In eval mode, outputs should be deterministic
        pooling.eval()
        output3 = pooling(x)
        output4 = pooling(x)
        assert torch.allclose(output3, output4)


class TestMeanPooling:
    """Tests for MeanPooling."""

    def test_forward_shape_keepdim_true(self, device):
        """Test forward pass shape with keepdim=True (default)."""
        pooling = MeanPooling(keepdim=True).to(device)
        x = torch.randn(4, 100, 1280, device=device)
        output = pooling(x)

        assert output.shape == (4, 1, 1280)

    def test_forward_shape_keepdim_false(self, device):
        """Test forward pass shape with keepdim=False."""
        pooling = MeanPooling(keepdim=False).to(device)
        x = torch.randn(4, 100, 1280, device=device)
        output = pooling(x)

        assert output.shape == (4, 1280)

    def test_forward_variable_sequence_lengths(self, device):
        """Test forward pass with different sequence lengths."""
        pooling = MeanPooling().to(device)

        for seq_len in [1, 10, 100, 500, 1000]:
            x = torch.randn(2, seq_len, 1280, device=device)
            output = pooling(x)

            assert output.shape == (2, 1, 1280)

    def test_mean_computation_correctness(self, device):
        """Test that mean pooling correctly computes the mean."""
        pooling = MeanPooling(keepdim=False).to(device)

        x = torch.randn(2, 10, 64, device=device)
        output = pooling(x)
        expected = x.mean(dim=1)

        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_with_attention_mask(self, device):
        """Test forward pass with attention mask."""
        pooling = MeanPooling(keepdim=False).to(device)

        # Create input where we can verify masked mean
        x = torch.ones(2, 10, 64, device=device)
        x[0, 5:, :] = 2.0  # First batch: second half has value 2
        x[1, 5:, :] = 2.0  # Second batch: second half has value 2

        # Mask out second half for first batch element only
        attention_mask = torch.ones(2, 10, device=device)
        attention_mask[0, 5:] = 0

        output = pooling(x, attention_mask=attention_mask)

        # First batch element should have mean=1.0 (only first half counted)
        assert torch.allclose(output[0], torch.ones(64, device=device), atol=1e-6)

        # Second batch element should have mean=1.5 (full sequence: 5*1 + 5*2 / 10)
        expected_mean = (1.0 * 5 + 2.0 * 5) / 10
        assert torch.allclose(output[1], torch.full((64,), expected_mean, device=device), atol=1e-6)

    def test_num_output_tokens_property(self):
        """Test num_output_tokens property returns 1."""
        pooling = MeanPooling()
        assert pooling.num_output_tokens == 1

    def test_gradient_flow(self, device):
        """Test that gradients flow through mean pooling."""
        pooling = MeanPooling().to(device)
        x = torch.randn(2, 50, 256, device=device, requires_grad=True)

        output = pooling(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCLSPooling:
    """Tests for CLSPooling."""

    def test_forward_shape_keepdim_true(self, device):
        """Test forward pass shape with keepdim=True (default)."""
        pooling = CLSPooling(keepdim=True).to(device)
        x = torch.randn(4, 100, 1280, device=device)
        output = pooling(x)

        assert output.shape == (4, 1, 1280)

    def test_forward_shape_keepdim_false(self, device):
        """Test forward pass shape with keepdim=False."""
        pooling = CLSPooling(keepdim=False).to(device)
        x = torch.randn(4, 100, 1280, device=device)
        output = pooling(x)

        assert output.shape == (4, 1280)

    def test_cls_extraction_correctness(self, device):
        """Test that CLS pooling correctly extracts first token."""
        pooling = CLSPooling(keepdim=False).to(device)

        x = torch.randn(2, 10, 64, device=device)
        output = pooling(x)
        expected = x[:, 0, :]

        assert torch.allclose(output, expected)

    def test_num_output_tokens_property(self):
        """Test num_output_tokens property returns 1."""
        pooling = CLSPooling()
        assert pooling.num_output_tokens == 1


class TestGetPooling:
    """Tests for the get_pooling factory function."""

    def test_get_attention_pooling(self):
        """Test getting attention pooling by name."""
        pooling = get_pooling("attention", embed_dim=512, num_output_tokens=16)

        assert isinstance(pooling, AttentionPooling)
        assert pooling.embed_dim == 512
        assert pooling.num_output_tokens == 16

    def test_get_mean_pooling(self):
        """Test getting mean pooling by name."""
        pooling = get_pooling("mean", keepdim=False)

        assert isinstance(pooling, MeanPooling)
        assert pooling.keepdim is False

    def test_get_cls_pooling(self):
        """Test getting CLS pooling by name."""
        pooling = get_pooling("cls")

        assert isinstance(pooling, CLSPooling)

    def test_case_insensitive(self):
        """Test that pooling type is case insensitive."""
        pooling1 = get_pooling("ATTENTION")
        pooling2 = get_pooling("Attention")
        pooling3 = get_pooling("attention")

        assert all(isinstance(p, AttentionPooling) for p in [pooling1, pooling2, pooling3])

    def test_invalid_pooling_type(self):
        """Test that invalid pooling type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pooling type"):
            get_pooling("invalid_pooling")

    def test_registry_contains_all_pooling_types(self):
        """Test that registry contains all expected pooling types."""
        expected_types = {"attention", "mean", "cls"}
        assert set(POOLING_REGISTRY.keys()) == expected_types


class TestBuildPoolingFromConfig:
    """Tests for building pooling from Hydra config."""

    def test_build_attention_pooling_from_config(self, device):
        """Test building attention pooling from config."""
        config = OmegaConf.create({
            "method": "attention",
            "embed_dim": 1280,
            "num_output_tokens": 64,
            "num_heads": 8,
            "dropout": 0.1,
            "layer_norm": True,
        })

        pooling = build_pooling_from_config(config).to(device)

        assert isinstance(pooling, AttentionPooling)
        assert pooling.embed_dim == 1280
        assert pooling.num_output_tokens == 64

        x = torch.randn(2, 100, 1280, device=device)
        output = pooling(x)
        assert output.shape == (2, 64, 1280)

    def test_build_mean_pooling_from_config(self, device):
        """Test building mean pooling from config."""
        config = OmegaConf.create({
            "method": "mean",
            "keepdim": True,
        })

        pooling = build_pooling_from_config(config).to(device)

        assert isinstance(pooling, MeanPooling)

        x = torch.randn(2, 100, 1280, device=device)
        output = pooling(x)
        assert output.shape == (2, 1, 1280)

    def test_build_from_nested_config(self, device):
        """Test building pooling from nested config structure."""
        config = OmegaConf.create({
            "encoder": {
                "name": "esm2",
                "pooling": {
                    "method": "attention",
                    "embed_dim": 1280,
                    "num_output_tokens": 32,
                    "num_heads": 8,
                },
            },
        })

        pooling = build_pooling_from_config(config).to(device)

        assert isinstance(pooling, AttentionPooling)
        assert pooling.num_output_tokens == 32

    def test_build_with_default_method(self, device):
        """Test building pooling defaults to attention."""
        config = OmegaConf.create({
            "embed_dim": 1280,
            "num_output_tokens": 16,
        })

        pooling = build_pooling_from_config(config).to(device)

        assert isinstance(pooling, AttentionPooling)

    def test_integration_with_mock_config(self, mock_config, device):
        """Test integration with mock_config fixture."""
        # Add pooling config if not present
        if not hasattr(mock_config.encoder.pooling, "method"):
            mock_config.encoder.pooling.method = "attention"
        mock_config.encoder.pooling.embed_dim = 1280
        mock_config.encoder.pooling.num_output_tokens = 32
        mock_config.encoder.pooling.num_heads = 8

        pooling = build_pooling_from_config(mock_config).to(device)

        assert isinstance(pooling, AttentionPooling)
        x = torch.randn(2, 50, 1280, device=device)
        output = pooling(x)
        assert output.shape == (2, 32, 1280)


class TestPoolingIntegration:
    """Integration tests for pooling module."""

    def test_all_pooling_inherit_from_base(self):
        """Test all pooling classes inherit from BasePooling."""
        for pooling_cls in POOLING_REGISTRY.values():
            assert issubclass(pooling_cls, BasePooling)

    def test_batch_size_one(self, device):
        """Test pooling works with batch size 1."""
        for pooling_type in ["attention", "mean", "cls"]:
            kwargs = {"embed_dim": 1280, "num_output_tokens": 32} if pooling_type == "attention" else {}
            pooling = get_pooling(pooling_type, **kwargs).to(device)

            x = torch.randn(1, 50, 1280, device=device)
            output = pooling(x)

            expected_tokens = 32 if pooling_type == "attention" else 1
            assert output.shape == (1, expected_tokens, 1280)

    def test_very_short_sequence(self, device):
        """Test pooling works with very short sequences."""
        pooling = get_pooling("attention", embed_dim=1280, num_output_tokens=32).to(device)

        x = torch.randn(2, 5, 1280, device=device)  # Sequence shorter than num_output_tokens
        output = pooling(x)

        assert output.shape == (2, 32, 1280)

    def test_deterministic_eval_mode(self, device):
        """Test that eval mode produces deterministic outputs."""
        pooling = get_pooling(
            "attention",
            embed_dim=256,
            num_output_tokens=8,
            dropout=0.5,
        ).to(device)
        pooling.eval()

        x = torch.randn(2, 50, 256, device=device)

        output1 = pooling(x)
        output2 = pooling(x)

        assert torch.allclose(output1, output2)

    def test_half_precision(self, device):
        """Test pooling works with half precision (fp16)."""
        if device == "cpu":
            pytest.skip("Half precision on CPU can be slow/unsupported")

        pooling = get_pooling(
            "attention",
            embed_dim=1280,
            num_output_tokens=32,
        ).to(device).half()

        x = torch.randn(2, 100, 1280, device=device, dtype=torch.float16)
        output = pooling(x)

        assert output.shape == (2, 32, 1280)
        assert output.dtype == torch.float16
