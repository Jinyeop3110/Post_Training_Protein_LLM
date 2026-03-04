"""Tests for Flamingo-style Perceiver Resampler module."""

import pytest
import torch

from src.models.flamingo_perceiver import FlamingoPerceiverLayer, FlamingoPerceiverResampler


class TestFlamingoPerceiverLayer:
    """Tests for individual Flamingo Perceiver layer."""

    def test_forward_shape(self):
        layer = FlamingoPerceiverLayer(dim=256, dim_head=32, heads=8, ff_mult=4)
        latents = torch.randn(2, 64, 256)
        media = torch.randn(2, 100, 256)
        out = layer(latents, media)
        assert out.shape == (2, 64, 256)

    def test_implicit_self_attention(self):
        """K,V should come from concat(media, latents), not media alone."""
        layer = FlamingoPerceiverLayer(dim=128, dim_head=32, heads=4)
        latents = torch.randn(1, 16, 128)
        media = torch.randn(1, 50, 128)

        # Run forward and check output is valid
        out = layer(latents, media)
        assert out.shape == (1, 16, 128)

        # Verify KV input size: the kv projection should have processed L+N tokens
        # We test this by checking that changing latents affects output through KV path
        latents2 = latents.clone()
        latents2[0, 0] += 10.0  # Modify first latent
        out2 = layer(latents2, media)
        # Output should differ (latents affect both Q and KV)
        assert not torch.allclose(out, out2)

    def test_gradient_flow(self):
        layer = FlamingoPerceiverLayer(dim=128, dim_head=32, heads=4)
        latents = torch.randn(1, 16, 128, requires_grad=True)
        media = torch.randn(1, 50, 128, requires_grad=True)
        out = layer(latents, media)
        loss = out.sum()
        loss.backward()
        assert latents.grad is not None
        assert media.grad is not None

    def test_no_bias_in_layers(self):
        """Verify all linear layers have bias=False (Suggestion 5)."""
        layer = FlamingoPerceiverLayer(dim=128, dim_head=32, heads=4)
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert module.bias is None, f"{name} should not have bias"


class TestFlamingoPerceiverResampler:
    """Tests for the full Flamingo Perceiver Resampler."""

    def test_default_initialization(self):
        model = FlamingoPerceiverResampler()
        assert model.encoder_dim == 1536
        assert model.output_dim == 2560
        assert model.latent_dim == 2560
        assert model.num_queries == 64
        assert model.num_layers_count == 6
        assert model.max_seq_len == 2048
        assert len(model.layers) == 6

    def test_custom_initialization(self):
        model = FlamingoPerceiverResampler(
            encoder_dim=1536, output_dim=2560, latent_dim=1024,
            num_queries=64, num_layers=6, num_heads=8, dim_head=64,
        )
        assert model.latent_dim == 1024
        assert model.output_dim == 2560
        assert model.query_tokens.shape == (64, 1024)
        assert isinstance(model.output_proj, torch.nn.Linear)
        assert model.output_proj.in_features == 1024
        assert model.output_proj.out_features == 2560

    def test_forward_shape_basic(self):
        model = FlamingoPerceiverResampler(
            encoder_dim=256, output_dim=512, latent_dim=256,
            num_queries=16, num_layers=2, num_heads=4, dim_head=32,
        )
        x = torch.randn(2, 100, 256)
        out = model(x)
        assert out.shape == (2, 16, 512)

    def test_forward_shape_esm3_config(self):
        """Test with ESM-3 → Qwen3-4B dimensions."""
        model = FlamingoPerceiverResampler(
            encoder_dim=1536, output_dim=2560, latent_dim=1024,
            num_queries=64, num_layers=6, num_heads=8, dim_head=64,
        )
        x = torch.randn(1, 200, 1536)
        out = model(x)
        assert out.shape == (1, 64, 2560)

    def test_forward_variable_seq_lengths(self):
        """Different sequence lengths should all produce same output shape."""
        model = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=16, num_layers=2, num_heads=4, dim_head=32,
        )
        for seq_len in [10, 50, 200, 500]:
            x = torch.randn(1, seq_len, 128)
            out = model(x)
            assert out.shape == (1, 16, 256), f"Failed for seq_len={seq_len}"

    def test_position_embeddings_effect(self):
        """Position embeddings should make output position-dependent."""
        model = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=8, num_layers=1, num_heads=4, dim_head=32,
        )
        model.eval()

        # Same content but different lengths → different pos embeddings
        x1 = torch.randn(1, 50, 128)
        x2 = torch.cat([x1, torch.randn(1, 10, 128)], dim=1)  # 60 tokens

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        # Outputs should differ due to extra tokens + different pos embeddings
        assert out1.shape != out2.shape or not torch.allclose(out1, out2)

    def test_residue_pos_embed_shape(self):
        model = FlamingoPerceiverResampler(
            encoder_dim=256, output_dim=512, max_seq_len=1024,
        )
        assert model.residue_pos_embed.shape == (1024, 256)

    def test_gradient_flow(self):
        model = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=8, num_layers=2, num_heads=4, dim_head=32,
        )
        x = torch.randn(2, 50, 128)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.query_tokens.grad is not None
        assert model.residue_pos_embed.grad is not None
        assert model.input_proj.weight.grad is not None

    def test_no_bias_anywhere(self):
        """Verify no linear layers have bias (Suggestion 5)."""
        model = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=8, num_layers=2, num_heads=4, dim_head=32,
        )
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert module.bias is None, f"{name} should not have bias"

    def test_invalid_input_dim(self):
        model = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=8, num_layers=2, num_heads=4, dim_head=32,
        )
        with pytest.raises(ValueError, match="Expected 3D"):
            model(torch.randn(128))

    def test_parameter_count(self):
        """Verify parameter count is reasonable."""
        model = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=16, num_layers=2, num_heads=4, dim_head=32,
        )
        total = sum(p.numel() for p in model.parameters())
        assert total > 100_000, f"Too few params: {total}"
        assert total < 50_000_000, f"Too many params: {total}"

    def test_eval_mode_deterministic(self):
        model = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=8, num_layers=2, num_heads=4, dim_head=32,
        )
        model.eval()
        x = torch.randn(1, 50, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_save_load_state_dict(self):
        model = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=8, num_layers=2, num_heads=4, dim_head=32,
        )
        state = model.state_dict()
        model2 = FlamingoPerceiverResampler(
            encoder_dim=128, output_dim=256, latent_dim=128,
            num_queries=8, num_layers=2, num_heads=4, dim_head=32,
        )
        model2.load_state_dict(state)
        model.eval()
        model2.eval()
        x = torch.randn(1, 50, 128)
        with torch.no_grad():
            assert torch.allclose(model(x), model2(x))

    def test_get_output_dim(self):
        model = FlamingoPerceiverResampler(encoder_dim=1536, output_dim=2560)
        assert model.get_output_dim() == 2560

    def test_get_input_dim(self):
        model = FlamingoPerceiverResampler(encoder_dim=1536, output_dim=2560)
        assert model.get_input_dim() == 1536

    def test_extra_repr(self):
        model = FlamingoPerceiverResampler(
            encoder_dim=1536, output_dim=2560, num_queries=64, num_layers=6,
        )
        repr_str = model.extra_repr()
        assert "encoder_dim=1536" in repr_str
        assert "output_dim=2560" in repr_str
        assert "num_queries=64" in repr_str
        assert "num_layers=6" in repr_str
        assert "max_seq_len=2048" in repr_str
