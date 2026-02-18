# Model architectures for protein-LLM integration

from src.models.projector import MLPProjector, get_projector
from src.models.pooling import (
    AttentionPooling,
    MeanPooling,
    CLSPooling,
    BasePooling,
    get_pooling,
    build_pooling_from_config,
    POOLING_REGISTRY,
)
from src.models.multimodal_llm import ProteinLLM

__all__ = [
    # Multimodal
    "ProteinLLM",
    # Projector
    "MLPProjector",
    "get_projector",
    # Pooling
    "AttentionPooling",
    "MeanPooling",
    "CLSPooling",
    "BasePooling",
    "get_pooling",
    "build_pooling_from_config",
    "POOLING_REGISTRY",
]
