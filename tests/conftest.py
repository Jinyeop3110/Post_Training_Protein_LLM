"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_sequence():
    """Sample protein sequence for testing."""
    return "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH"


@pytest.fixture
def sample_sequences():
    """Multiple sample protein sequences."""
    return [
        "MKTAYIAKQRQISFVKSHFSRQ",
        "MNIFEMLRIDEGLRLKIYKDTEG",
        "GLFVQLQVGNQPKNSNVSLDLCVFSEMG",
    ]


@pytest.fixture
def mock_config():
    """Mock Hydra config for testing."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "model": {
            "name": "test_model",
            "path": "test",
            "architecture": {
                "hidden_size": 256,
            },
        },
        "encoder": {
            "name": "esm3_test",
            "embedding_dim": 1536,
            "freeze": True,
            "pooling": {
                "method": "attention",
            },
        },
        "training": {
            "lr": 1e-4,
            "batch_size": 2,
            "epochs": 1,
        },
    })
