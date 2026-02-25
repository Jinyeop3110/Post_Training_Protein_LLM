"""
Pooling Module for Protein Embeddings

This module provides different pooling strategies to transform per-residue
ESM-3 embeddings [B, L, D] into fixed-size representations suitable for
use as LLM prefix tokens.

Pooling Strategies:
1. AttentionPooling (BoM-Pooling style): Uses learned query tokens with
   multi-head cross-attention to pool variable-length sequences into a
   fixed number of output tokens.
2. MeanPooling: Simple mean pooling over the sequence dimension as a
   lightweight fallback.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BasePooling(nn.Module, ABC):
    """Abstract base class for pooling strategies."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence embeddings.

        Args:
            x: Input tensor of shape [B, L, D] where B is batch size,
               L is sequence length, and D is embedding dimension.
            attention_mask: Optional mask of shape [B, L] where True/1
               indicates valid positions and False/0 indicates padding.

        Returns:
            Pooled tensor of shape [B, N, D] where N is the number of
            output tokens (1 for MeanPooling, num_output_tokens for
            AttentionPooling).
        """
        pass

    @property
    @abstractmethod
    def num_output_tokens(self) -> int:
        """Return the number of output tokens produced by this pooling."""
        pass


class AttentionPooling(BasePooling):
    """
    Attention-based pooling using learned query tokens (BoM-Pooling style).

    This pooling mechanism uses a set of learned query tokens that attend
    to the input sequence via multi-head cross-attention. This allows the
    model to learn which parts of the protein sequence are most relevant
    for different output tokens.

    Architecture:
        - Learned query tokens: [num_output_tokens, embed_dim]
        - Multi-head cross-attention: queries attend to sequence
        - Optional layer normalization on output

    Reference:
        Similar to Perceiver IO's cross-attention and Q-Former's approach.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        num_output_tokens: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        """
        Initialize AttentionPooling.

        Args:
            embed_dim: Embedding dimension (must match input embedding dim).
                      Default is 1280.
            num_output_tokens: Number of output tokens to produce.
            num_heads: Number of attention heads.
            dropout: Dropout probability for attention weights.
            layer_norm: Whether to apply layer normalization to output.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self._num_output_tokens = num_output_tokens
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learned query tokens that will attend to the sequence
        self.query_tokens = nn.Parameter(
            torch.randn(num_output_tokens, embed_dim) * 0.02
        )

        # Multi-head cross-attention
        # Queries: learned tokens, Keys/Values: input sequence
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()

        # Initialize attention weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        # MultiheadAttention is already initialized by PyTorch

    @property
    def num_output_tokens(self) -> int:
        """Return the number of output tokens."""
        return self._num_output_tokens

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence embeddings using cross-attention.

        Args:
            x: Input tensor of shape [B, L, D].
            attention_mask: Optional mask of shape [B, L] where True/1
               indicates valid positions and False/0 indicates padding.

        Returns:
            Pooled tensor of shape [B, num_output_tokens, D].
        """
        batch_size = x.shape[0]

        # Expand query tokens for batch: [N, D] -> [B, N, D]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Convert attention mask to key_padding_mask format if provided
        # MultiheadAttention expects True for positions to mask (ignore)
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: True/1 = valid, False/0 = padding
            # key_padding_mask: True = ignore (mask), False = attend
            key_padding_mask = ~attention_mask.bool()

        # Cross-attention: queries attend to sequence (keys/values)
        pooled, _ = self.cross_attention(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Apply layer normalization
        pooled = self.layer_norm(pooled)

        return pooled


class MeanPooling(BasePooling):
    """
    Simple mean pooling over the sequence dimension.

    This is a lightweight fallback pooling strategy that computes
    the mean of all token embeddings. It produces a single output
    token representing the entire sequence.
    """

    def __init__(self, keepdim: bool = True):
        """
        Initialize MeanPooling.

        Args:
            keepdim: If True, output shape is [B, 1, D]. If False, [B, D].
                    Default True for consistency with AttentionPooling.
        """
        super().__init__()
        self.keepdim = keepdim

    @property
    def num_output_tokens(self) -> int:
        """Return the number of output tokens (always 1 for mean pooling)."""
        return 1

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence embeddings using mean pooling.

        Args:
            x: Input tensor of shape [B, L, D].
            attention_mask: Optional mask of shape [B, L] where True/1
               indicates valid positions and False/0 indicates padding.

        Returns:
            Pooled tensor of shape [B, 1, D] if keepdim=True, else [B, D].
        """
        if attention_mask is not None:
            # Expand mask for broadcasting: [B, L] -> [B, L, 1]
            mask = attention_mask.unsqueeze(-1).float()
            # Masked mean: sum of valid tokens / count of valid tokens
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        if self.keepdim:
            pooled = pooled.unsqueeze(1)

        return pooled


class CLSPooling(BasePooling):
    """
    CLS token pooling - uses the first token as the sequence representation.

    Common in BERT-style models where the first token aggregates
    sequence information during pre-training.
    """

    def __init__(self, keepdim: bool = True):
        """
        Initialize CLSPooling.

        Args:
            keepdim: If True, output shape is [B, 1, D]. If False, [B, D].
        """
        super().__init__()
        self.keepdim = keepdim

    @property
    def num_output_tokens(self) -> int:
        """Return the number of output tokens (always 1 for CLS pooling)."""
        return 1

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract the CLS (first) token.

        Args:
            x: Input tensor of shape [B, L, D].
            attention_mask: Not used for CLS pooling, included for API
               consistency.

        Returns:
            CLS token of shape [B, 1, D] if keepdim=True, else [B, D].
        """
        pooled = x[:, 0, :]

        if self.keepdim:
            pooled = pooled.unsqueeze(1)

        return pooled


# Registry of available pooling methods
POOLING_REGISTRY: Dict[str, type] = {
    "attention": AttentionPooling,
    "mean": MeanPooling,
    "cls": CLSPooling,
}


def get_pooling(
    pooling_type: str,
    **kwargs: Any
) -> BasePooling:
    """
    Factory function to instantiate pooling by name.

    Args:
        pooling_type: Name of the pooling strategy. One of:
            - "attention": AttentionPooling (BoM-style with learned queries)
            - "mean": MeanPooling (simple mean over sequence)
            - "cls": CLSPooling (use first token)
        **kwargs: Additional arguments passed to the pooling constructor.

    Returns:
        Instantiated pooling module.

    Raises:
        ValueError: If pooling_type is not recognized.

    Example:
        >>> pooling = get_pooling("attention", embed_dim=1280, num_output_tokens=32)
        >>> output = pooling(x)  # x: [B, L, 1280] -> output: [B, 32, 1280]
    """
    pooling_type = pooling_type.lower()

    if pooling_type not in POOLING_REGISTRY:
        available = list(POOLING_REGISTRY.keys())
        raise ValueError(
            f"Unknown pooling type: '{pooling_type}'. "
            f"Available options: {available}"
        )

    return POOLING_REGISTRY[pooling_type](**kwargs)


def build_pooling_from_config(config: Any) -> BasePooling:
    """
    Build pooling module from Hydra/OmegaConf configuration.

    Expected config structure:
        pooling:
            method: "attention"  # or "mean", "cls"
            embed_dim: 1280
            num_output_tokens: 32
            num_heads: 8
            dropout: 0.1
            layer_norm: true

    Args:
        config: Hydra/OmegaConf configuration object with pooling settings.
               Can be the full config (with config.encoder.pooling) or
               just the pooling sub-config.

    Returns:
        Instantiated pooling module.

    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({
        ...     "method": "attention",
        ...     "embed_dim": 1280,
        ...     "num_output_tokens": 32,
        ... })
        >>> pooling = build_pooling_from_config(config)
    """
    # Handle nested config structure
    if hasattr(config, "encoder") and hasattr(config.encoder, "pooling"):
        pooling_config = config.encoder.pooling
    elif hasattr(config, "pooling"):
        pooling_config = config.pooling
    else:
        pooling_config = config

    # Extract pooling method
    method = getattr(pooling_config, "method", "attention")

    # Build kwargs from config, excluding 'method'
    kwargs = {}
    for key in ["embed_dim", "num_output_tokens", "num_heads", "dropout",
                "layer_norm", "keepdim"]:
        if hasattr(pooling_config, key):
            kwargs[key] = getattr(pooling_config, key)

    return get_pooling(method, **kwargs)


# Alias for backwards compatibility
PoolingFactory = get_pooling
