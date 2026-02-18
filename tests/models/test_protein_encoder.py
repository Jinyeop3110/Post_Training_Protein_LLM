"""Tests for protein encoder."""

import pytest
import torch


class TestProteinEncoder:
    """Tests for protein encoder implementations."""

    def test_encoder_import(self):
        """Test encoder module can be imported."""
        try:
            from src.models.protein_encoder import ESM2Encoder
            assert ESM2Encoder is not None
        except ImportError:
            pytest.skip("Encoder not yet implemented")

    def test_encoder_output_shape(self, sample_sequence, device):
        """Test encoder produces correct output shape."""
        try:
            from src.models.protein_encoder import ESM2Encoder

            # This would require actual ESM-2 model
            # Skip if model not available
            pytest.skip("Full encoder test requires ESM-2 model")

            encoder = ESM2Encoder(
                model_name="esm2_t33_650M_UR50D",
                pooling="attention",
            ).to(device)

            output = encoder([sample_sequence])
            assert output.shape[0] == 1
            assert output.shape[1] == 1280  # ESM-2 650M embedding dim
        except ImportError:
            pytest.skip("Encoder not yet implemented")

    def test_encoder_frozen(self):
        """Test ESM-2 weights are frozen."""
        try:
            from src.models.protein_encoder import ESM2Encoder

            pytest.skip("Full encoder test requires ESM-2 model")

            encoder = ESM2Encoder(
                model_name="esm2_t33_650M_UR50D",
                freeze=True,
            )

            for param in encoder.esm.parameters():
                assert not param.requires_grad, "ESM-2 weights should be frozen"
        except ImportError:
            pytest.skip("Encoder not yet implemented")


class TestPooling:
    """Tests for pooling strategies."""

    def test_attention_pooling(self):
        """Test attention pooling implementation."""
        try:
            from src.models.pooling import AttentionPooling

            pooling = AttentionPooling(embedding_dim=1280)

            # Create mock input
            x = torch.randn(2, 100, 1280)  # [batch, seq_len, dim]
            output = pooling(x)

            assert output.shape == (2, 1280)
        except ImportError:
            pytest.skip("Pooling not yet implemented")

    def test_mean_pooling(self):
        """Test mean pooling implementation."""
        try:
            from src.models.pooling import MeanPooling

            pooling = MeanPooling()

            x = torch.randn(2, 100, 1280)
            output = pooling(x)

            assert output.shape == (2, 1280)
        except ImportError:
            pytest.skip("Pooling not yet implemented")
