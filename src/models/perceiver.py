"""
Perceiver Resampler Module

Implements a Perceiver Resampler that replaces both AttentionPooling and
MLPProjector as a single module. Maps variable-length encoder output
[B, L, encoder_dim] to fixed-size LLM-ready output [B, num_queries, output_dim].

Architecture per layer:
  1. LayerNorm + Self-Attention (queries attend to queries)
  2. LayerNorm + Cross-Attention (queries attend to encoder output)
  3. LayerNorm + FFN (feed-forward network)

Reference: Flamingo (Alayrac et al., 2022) Perceiver Resampler,
           BLIP-2 (Li et al., 2023) Q-Former
"""

from typing import Optional

import torch
import torch.nn as nn


class PerceiverResamplerLayer(nn.Module):
    """Single Perceiver Resampler layer.

    Each layer performs:
      1. Pre-norm Self-Attention (queries attend to queries)
      2. Pre-norm Cross-Attention (queries attend to encoder output)
      3. Pre-norm FFN (feed-forward network)

    All sub-layers use residual connections.

    Args:
        dim: Working dimension (same as output_dim / query dim).
        num_heads: Number of attention heads.
        ffn_dim: Hidden dimension of the feed-forward network.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention on queries
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: queries attend to encoder output
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.encoder_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, num_queries, dim]
            encoder_output: [B, L, dim] (already projected to working dim)
            encoder_mask: [B, L] where True/1 = valid, False/0 = padding

        Returns:
            Updated queries: [B, num_queries, dim]
        """
        # Self-attention (pre-norm residual)
        residual = queries
        q = self.self_attn_norm(queries)
        q, _ = self.self_attn(q, q, q, need_weights=False)
        queries = residual + q

        # Cross-attention (pre-norm residual)
        residual = queries
        q = self.cross_attn_norm(queries)
        kv = self.encoder_norm(encoder_output)

        key_padding_mask = None
        if encoder_mask is not None:
            # MultiheadAttention expects True = ignore (mask), False = attend
            key_padding_mask = ~encoder_mask.bool()

        q, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = residual + q

        # FFN (pre-norm residual)
        residual = queries
        queries = residual + self.ffn(self.ffn_norm(queries))

        return queries


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler that replaces both pooling and projector.

    Maps variable-length encoder embeddings [B, L, encoder_dim] to fixed-size
    output [B, num_queries, output_dim] suitable for LLM prefix tokens.

    Architecture:
      1. Input projection: encoder_dim -> latent_dim
      2. Learned query tokens: [num_queries, latent_dim]
      3. N layers of: Self-Attn -> Cross-Attn -> FFN (all at latent_dim)
      4. Output projection: latent_dim -> output_dim
      5. Final LayerNorm

    Args:
        encoder_dim: Input dimension from protein encoder (e.g., 1536 for ESM-3).
        output_dim: Output dimension matching LLM hidden size (e.g., 2560).
        latent_dim: Internal working dimension for attention layers. If None,
                    defaults to output_dim (original behavior).
        num_queries: Number of output tokens (default 32).
        num_layers: Number of Perceiver layers (default 2).
        num_heads: Number of attention heads (default 8).
        ffn_dim: FFN hidden dimension (default 2048).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        encoder_dim: int = 1536,
        output_dim: int = 2560,
        latent_dim: Optional[int] = None,
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # latent_dim defaults to output_dim for backward compatibility
        working_dim = latent_dim if latent_dim is not None else output_dim

        if working_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({working_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.encoder_dim = encoder_dim
        self.output_dim = output_dim
        self.latent_dim = working_dim
        self.num_queries = num_queries
        self.num_layers_count = num_layers

        # Input projection: encoder_dim -> latent_dim
        if encoder_dim != working_dim:
            self.input_proj = nn.Linear(encoder_dim, working_dim)
        else:
            self.input_proj = nn.Identity()

        # Learned query tokens at latent_dim
        self.query_tokens = nn.Parameter(
            torch.randn(num_queries, working_dim) * 0.02
        )

        # Perceiver layers (all at latent_dim)
        self.layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(
                    dim=working_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm at latent_dim
        self.output_norm = nn.LayerNorm(working_dim)

        # Output projection: latent_dim -> output_dim (if they differ)
        if working_dim != output_dim:
            self.output_proj = nn.Linear(working_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Encoder output [B, L, encoder_dim].
            attention_mask: Optional mask [B, L], True/1=valid, False/0=padding.

        Returns:
            Resampled output [B, num_queries, output_dim].
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor [B, L, D], got {x.dim()}D tensor"
            )

        batch_size = x.shape[0]

        # Project encoder output to latent dimension
        encoder_output = self.input_proj(x)  # [B, L, latent_dim]

        # Expand query tokens for batch
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Pass through Perceiver layers (all at latent_dim)
        for layer in self.layers:
            queries = layer(queries, encoder_output, encoder_mask=attention_mask)

        # Final norm + project to output_dim
        queries = self.output_norm(queries)
        queries = self.output_proj(queries)  # [B, num_queries, output_dim]

        return queries

    def get_output_dim(self) -> int:
        """Return the output dimension."""
        return self.output_dim

    def get_input_dim(self) -> int:
        """Return the input dimension."""
        return self.encoder_dim

    def extra_repr(self) -> str:
        return (
            f"encoder_dim={self.encoder_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_queries={self.num_queries}, "
            f"num_layers={self.num_layers_count}"
        )
