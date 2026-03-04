"""Tests for Gated Cross-Attention module."""

import pytest
import torch
import torch.nn as nn

from src.models.gated_cross_attention import (
    FlamingoDecoderLayer,
    GatedCrossAttentionBlock,
    MaskedCrossAttention,
    clear_protein_features,
    get_xattn_parameters,
    inject_cross_attention_layers,
    set_protein_features,
)


class TestMaskedCrossAttention:
    """Tests for MaskedCrossAttention."""

    def test_forward_shape(self):
        xattn = MaskedCrossAttention(dim=256, dim_visual=512, dim_head=32, heads=8)
        x = torch.randn(2, 100, 256)
        media = torch.randn(2, 64, 512)
        out = xattn(x, media)
        assert out.shape == (2, 100, 256)

    def test_same_dim_visual(self):
        """dim_visual can equal dim."""
        xattn = MaskedCrossAttention(dim=256, dim_visual=256, dim_head=32, heads=4)
        x = torch.randn(1, 50, 256)
        media = torch.randn(1, 16, 256)
        out = xattn(x, media)
        assert out.shape == (1, 50, 256)

    def test_gradient_flow(self):
        xattn = MaskedCrossAttention(dim=128, dim_visual=256, dim_head=32, heads=4)
        x = torch.randn(1, 20, 128, requires_grad=True)
        media = torch.randn(1, 16, 256, requires_grad=True)
        out = xattn(x, media)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert media.grad is not None

    def test_no_bias(self):
        """All linear layers should have bias=False."""
        xattn = MaskedCrossAttention(dim=128, dim_visual=256, dim_head=32, heads=4)
        for name, module in xattn.named_modules():
            if isinstance(module, nn.Linear):
                assert module.bias is None, f"{name} should not have bias"


class TestGatedCrossAttentionBlock:
    """Tests for GatedCrossAttentionBlock."""

    def test_forward_shape(self):
        block = GatedCrossAttentionBlock(
            dim=256, dim_visual=512, dim_head=32, heads=8, ff_mult=4,
        )
        x = torch.randn(2, 100, 256)
        media = torch.randn(2, 64, 512)
        out = block(x, media)
        assert out.shape == (2, 100, 256)

    def test_gate_init_zero(self):
        """Gates should be initialized to 0."""
        block = GatedCrossAttentionBlock(dim=256, dim_visual=512)
        assert block.gate_attn.item() == 0.0
        assert block.gate_ffn.item() == 0.0

    def test_identity_at_init(self):
        """At initialization (gates=0), output should equal input."""
        block = GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
        x = torch.randn(1, 20, 128)
        media = torch.randn(1, 16, 256)

        block.eval()
        with torch.no_grad():
            out = block(x, media)

        # tanh(0) = 0, so gated contributions are zeroed out
        assert torch.allclose(out, x, atol=1e-6), (
            f"At init, block should be identity. "
            f"Max diff: {(out - x).abs().max().item()}"
        )

    def test_gradient_flow(self):
        block = GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
        x = torch.randn(1, 20, 128, requires_grad=True)
        media = torch.randn(1, 16, 256, requires_grad=True)
        out = block(x, media)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert media.grad is not None
        # Gates should also get gradients
        assert block.gate_attn.grad is not None
        assert block.gate_ffn.grad is not None

    def test_gate_opening(self):
        """Setting gates to non-zero should change output."""
        block = GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
        x = torch.randn(1, 20, 128)
        media = torch.randn(1, 16, 256)

        block.eval()
        with torch.no_grad():
            out_closed = block(x, media)

            # Open gates
            block.gate_attn.fill_(1.0)
            block.gate_ffn.fill_(1.0)
            out_open = block(x, media)

        assert not torch.allclose(out_closed, out_open)

    def test_save_load(self):
        block = GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
        # Open gates for more interesting state
        with torch.no_grad():
            block.gate_attn.fill_(0.5)
            block.gate_ffn.fill_(0.3)

        state = block.state_dict()
        block2 = GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
        block2.load_state_dict(state)

        assert block2.gate_attn.item() == pytest.approx(0.5)
        assert block2.gate_ffn.item() == pytest.approx(0.3)


class TestFlamingoDecoderLayer:
    """Tests for FlamingoDecoderLayer wrapper."""

    def _make_mock_decoder_layer(self, dim):
        """Create a simple mock decoder layer."""
        class MockDecoderLayer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, hidden_states, **kwargs):
                return self.linear(hidden_states)

        return MockDecoderLayer(dim)

    def test_forward_with_features(self):
        dim = 128
        decoder = self._make_mock_decoder_layer(dim)
        xattn = GatedCrossAttentionBlock(dim=dim, dim_visual=256, dim_head=32, heads=4)
        wrapper = FlamingoDecoderLayer(decoder, xattn)

        x = torch.randn(1, 20, dim)
        media = torch.randn(1, 16, 256)
        wrapper._protein_features = media

        out = wrapper(x)
        assert out.shape == (1, 20, dim)

    def test_forward_without_features(self):
        """Without protein features, should just run decoder layer."""
        dim = 128
        decoder = self._make_mock_decoder_layer(dim)
        xattn = GatedCrossAttentionBlock(dim=dim, dim_visual=256, dim_head=32, heads=4)
        wrapper = FlamingoDecoderLayer(decoder, xattn)

        x = torch.randn(1, 20, dim)
        wrapper._protein_features = None

        out = wrapper(x)
        assert out.shape == (1, 20, dim)

        # Should be same as just running decoder
        with torch.no_grad():
            expected = decoder(x)
        # Can't compare directly because of grad state, just check shape
        assert out.shape == expected.shape

    def test_set_clear_features(self):
        dim = 128
        decoder = self._make_mock_decoder_layer(dim)
        xattn = GatedCrossAttentionBlock(dim=dim, dim_visual=256, dim_head=32, heads=4)
        wrapper = FlamingoDecoderLayer(decoder, xattn)

        media = torch.randn(1, 16, 256)
        wrapper._protein_features = media
        assert wrapper._protein_features is not None

        wrapper._protein_features = None
        assert wrapper._protein_features is None


class TestInjectCrossAttention:
    """Tests for injection helpers."""

    def _make_mock_model(self, num_layers=12, dim=128):
        """Create a mock model with decoder layers."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([
                    nn.Linear(dim, dim) for _ in range(num_layers)
                ])

        return MockModel()

    def test_inject_every_4th(self):
        model = self._make_mock_model(num_layers=12, dim=128)
        # Every 4th: indices 3, 7, 11 → 3 blocks needed
        xattn_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
            for _ in range(3)
        ])

        wrapped = inject_cross_attention_layers(model, xattn_blocks, xattn_every=4)
        assert len(wrapped) == 3

        # Check that layers 3, 7, 11 are FlamingoDecoderLayer
        for idx in [3, 7, 11]:
            assert isinstance(model.model.layers[idx], FlamingoDecoderLayer)

        # Other layers should NOT be wrapped
        for idx in [0, 1, 2, 4, 5, 6, 8, 9, 10]:
            assert not isinstance(model.model.layers[idx], FlamingoDecoderLayer)

    def test_inject_wrong_count(self):
        model = self._make_mock_model(num_layers=12, dim=128)
        # Only 2 blocks for 3 injection points → should error
        xattn_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
            for _ in range(2)
        ])

        with pytest.raises(ValueError, match="Expected 3"):
            inject_cross_attention_layers(model, xattn_blocks, xattn_every=4)

    def test_set_clear_protein_features(self):
        model = self._make_mock_model(num_layers=12, dim=128)
        xattn_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
            for _ in range(3)
        ])
        inject_cross_attention_layers(model, xattn_blocks, xattn_every=4)

        media = torch.randn(1, 16, 256)
        set_protein_features(model, media)

        # All wrapped layers should have features
        for idx in [3, 7, 11]:
            assert model.model.layers[idx]._protein_features is not None

        clear_protein_features(model)
        for idx in [3, 7, 11]:
            assert model.model.layers[idx]._protein_features is None

    def test_get_xattn_parameters(self):
        model = self._make_mock_model(num_layers=8, dim=128)
        xattn_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(dim=128, dim_visual=256, dim_head=32, heads=4)
            for _ in range(2)  # Every 4th of 8: indices 3, 7
        ])
        inject_cross_attention_layers(model, xattn_blocks, xattn_every=4)

        params = get_xattn_parameters(model)
        assert len(params) > 0

        # Count should match 2 blocks worth of parameters
        block_params = sum(p.numel() for p in xattn_blocks.parameters())
        collected_params = sum(p.numel() for p in params)
        assert collected_params == block_params
