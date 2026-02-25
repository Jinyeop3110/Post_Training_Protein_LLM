"""Tests for Perceiver Resampler module."""

import pytest
import torch

from src.models.perceiver import PerceiverResampler, PerceiverResamplerLayer


class TestPerceiverResamplerLayer:
    """Tests for individual Perceiver Resampler layer."""

    def test_forward_shape(self):
        layer = PerceiverResamplerLayer(dim=256, num_heads=8, ffn_dim=512)
        queries = torch.randn(2, 32, 256)
        encoder_output = torch.randn(2, 100, 256)
        out = layer(queries, encoder_output)
        assert out.shape == (2, 32, 256)

    def test_forward_with_mask(self):
        layer = PerceiverResamplerLayer(dim=256, num_heads=8, ffn_dim=512)
        queries = torch.randn(2, 32, 256)
        encoder_output = torch.randn(2, 100, 256)
        mask = torch.ones(2, 100, dtype=torch.bool)
        mask[0, 80:] = False  # Mask out last 20 positions in first batch
        out = layer(queries, encoder_output, encoder_mask=mask)
        assert out.shape == (2, 32, 256)

    def test_gradient_flow(self):
        layer = PerceiverResamplerLayer(dim=256, num_heads=8, ffn_dim=512)
        queries = torch.randn(2, 32, 256, requires_grad=True)
        encoder_output = torch.randn(2, 100, 256)
        out = layer(queries, encoder_output)
        loss = out.sum()
        loss.backward()
        assert queries.grad is not None
        assert queries.grad.shape == queries.shape


class TestPerceiverResampler:
    """Tests for the full Perceiver Resampler."""

    def test_default_initialization(self):
        model = PerceiverResampler()
        assert model.encoder_dim == 1536
        assert model.output_dim == 2560
        assert model.latent_dim == 2560  # defaults to output_dim
        assert model.num_queries == 32
        assert model.num_layers_count == 2
        assert len(model.layers) == 2
        assert model.query_tokens.shape == (32, 2560)

    def test_custom_initialization(self):
        model = PerceiverResampler(
            encoder_dim=1280,
            output_dim=4096,
            num_queries=64,
            num_layers=4,
            num_heads=16,
            ffn_dim=1024,
        )
        assert model.encoder_dim == 1280
        assert model.output_dim == 4096
        assert model.latent_dim == 4096  # defaults to output_dim
        assert model.num_queries == 64
        assert len(model.layers) == 4
        assert model.query_tokens.shape == (64, 4096)

    def test_latent_dim_initialization(self):
        model = PerceiverResampler(
            encoder_dim=1536,
            output_dim=2560,
            latent_dim=1024,
            num_queries=32,
            num_layers=2,
            num_heads=8,
        )
        assert model.latent_dim == 1024
        assert model.output_dim == 2560
        # Query tokens at latent_dim
        assert model.query_tokens.shape == (32, 1024)
        # input_proj: encoder_dim -> latent_dim
        assert isinstance(model.input_proj, torch.nn.Linear)
        assert model.input_proj.in_features == 1536
        assert model.input_proj.out_features == 1024
        # output_proj: latent_dim -> output_dim
        assert isinstance(model.output_proj, torch.nn.Linear)
        assert model.output_proj.in_features == 1024
        assert model.output_proj.out_features == 2560

    def test_latent_dim_forward_shape(self):
        model = PerceiverResampler(
            encoder_dim=1536, output_dim=2560, latent_dim=1024,
            num_queries=32, num_layers=2, num_heads=8,
        )
        x = torch.randn(2, 100, 1536)
        out = model(x)
        assert out.shape == (2, 32, 2560)

    def test_forward_shape_basic(self):
        model = PerceiverResampler(
            encoder_dim=1536, output_dim=2560, num_queries=32, num_layers=2
        )
        x = torch.randn(2, 100, 1536)
        out = model(x)
        assert out.shape == (2, 32, 2560)

    def test_forward_variable_seq_lengths(self):
        """Different sequence lengths should all produce same output shape."""
        model = PerceiverResampler(
            encoder_dim=256, output_dim=512, num_queries=16, num_layers=2,
            num_heads=8,
        )
        for seq_len in [10, 50, 200, 500]:
            x = torch.randn(1, seq_len, 256)
            out = model(x)
            assert out.shape == (1, 16, 512), f"Failed for seq_len={seq_len}"

    def test_forward_with_attention_mask(self):
        model = PerceiverResampler(
            encoder_dim=256, output_dim=512, num_queries=16, num_layers=2,
            num_heads=8,
        )
        x = torch.randn(3, 100, 256)
        mask = torch.ones(3, 100, dtype=torch.bool)
        mask[0, 50:] = False
        mask[1, 80:] = False
        out = model(x, attention_mask=mask)
        assert out.shape == (3, 16, 512)

    def test_input_projection_when_dims_differ(self):
        model = PerceiverResampler(
            encoder_dim=1536, output_dim=2560, num_queries=32, num_layers=2,
            num_heads=8,
        )
        assert isinstance(model.input_proj, torch.nn.Linear)
        assert model.input_proj.in_features == 1536
        # Without latent_dim, projects to output_dim
        assert model.input_proj.out_features == 2560

    def test_no_projection_when_dims_match(self):
        model = PerceiverResampler(
            encoder_dim=512, output_dim=512, num_queries=16, num_layers=2,
            num_heads=8,
        )
        assert isinstance(model.input_proj, torch.nn.Identity)
        assert isinstance(model.output_proj, torch.nn.Identity)

    def test_get_output_dim(self):
        model = PerceiverResampler(encoder_dim=1536, output_dim=2560)
        assert model.get_output_dim() == 2560

    def test_get_input_dim(self):
        model = PerceiverResampler(encoder_dim=1536, output_dim=2560)
        assert model.get_input_dim() == 1536

    def test_extra_repr(self):
        model = PerceiverResampler(
            encoder_dim=1536, output_dim=2560, num_queries=32, num_layers=6
        )
        repr_str = model.extra_repr()
        assert "encoder_dim=1536" in repr_str
        assert "latent_dim=2560" in repr_str
        assert "output_dim=2560" in repr_str
        assert "num_queries=32" in repr_str
        assert "num_layers=6" in repr_str

    def test_gradient_flow(self):
        model = PerceiverResampler(
            encoder_dim=256, output_dim=512, num_queries=16, num_layers=2,
            num_heads=8,
        )
        x = torch.randn(2, 50, 256)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that query tokens have gradients
        assert model.query_tokens.grad is not None
        # Check that input projection has gradients
        if isinstance(model.input_proj, torch.nn.Linear):
            assert model.input_proj.weight.grad is not None

    def test_invalid_input_dim(self):
        model = PerceiverResampler(
            encoder_dim=256, output_dim=512, num_queries=16, num_layers=2,
            num_heads=8,
        )
        with pytest.raises(ValueError, match="Expected 3D"):
            model(torch.randn(256))  # 1D input

    def test_head_dim_validation(self):
        # Without latent_dim, validates output_dim % num_heads
        with pytest.raises(ValueError, match="divisible"):
            PerceiverResampler(
                encoder_dim=256, output_dim=100, num_queries=16,
                num_layers=2, num_heads=8,  # 100 % 8 != 0
            )
        # With latent_dim, validates latent_dim % num_heads
        with pytest.raises(ValueError, match="divisible"):
            PerceiverResampler(
                encoder_dim=256, output_dim=512, latent_dim=100,
                num_queries=16, num_layers=2, num_heads=8,
            )

    def test_parameter_count(self):
        """Verify parameter count is in expected range."""
        model = PerceiverResampler(
            encoder_dim=256, output_dim=512, num_queries=32,
            num_layers=2, num_heads=8, ffn_dim=1024,
        )
        total = sum(p.numel() for p in model.parameters())
        # Should have meaningful params but not be absurdly large
        assert total > 100_000, f"Too few params: {total}"
        assert total < 50_000_000, f"Too many params: {total}"

    def test_eval_mode_deterministic(self):
        model = PerceiverResampler(
            encoder_dim=256, output_dim=512, num_queries=16, num_layers=2,
            num_heads=8, dropout=0.1,
        )
        model.eval()
        x = torch.randn(1, 50, 256)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_save_load_state_dict(self):
        model = PerceiverResampler(
            encoder_dim=256, output_dim=512, num_queries=16, num_layers=2,
            num_heads=8,
        )
        state = model.state_dict()
        model2 = PerceiverResampler(
            encoder_dim=256, output_dim=512, num_queries=16, num_layers=2,
            num_heads=8,
        )
        model2.load_state_dict(state)
        x = torch.randn(1, 50, 256)
        model.eval()
        model2.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), model2(x))
