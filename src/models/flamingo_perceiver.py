"""
Flamingo-style Perceiver Resampler

Differs from the standard PerceiverResampler (src/models/perceiver.py) in:
  1. Implicit self-attention: K,V from concat(media, latents) so latents
     attend to both visual features AND each other in one attention op.
  2. No dropout / no bias (Flamingo paper ablation, Suggestion 5).
  3. 64 queries, 6 layers, 4x FFN expansion (Suggestion 3).
  4. Residue position embeddings added to encoder output (Suggestion 6).
  5. SiLU activation in FFN (matches Qwen3/LLaMA FFN style).

Reference: Flamingo (Alayrac et al., 2022) Section 3.2
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlamingoPerceiverLayer(nn.Module):
    """Single Flamingo-style Perceiver layer with implicit self-attention.

    K,V are formed from concat(media, latents), so latents attend to
    both the encoder output AND each other in a single attention op.
    This is more efficient than separate self-attn + cross-attn.

    Args:
        dim: Working dimension for queries/latents.
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
        ff_mult: FFN expansion factor.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        # Pre-norms
        self.norm_latents = nn.LayerNorm(dim)
        self.norm_media = nn.LayerNorm(dim)

        # Q from latents only, K/V from concat(media, latents)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # FFN: pre-norm + SiLU gated (matches LLM style)
        self.ffn_norm = nn.LayerNorm(dim)
        ff_inner = dim * ff_mult
        self.ffn_up = nn.Linear(dim, ff_inner, bias=False)
        self.ffn_gate = nn.Linear(dim, ff_inner, bias=False)
        self.ffn_down = nn.Linear(ff_inner, dim, bias=False)

    def forward(
        self,
        latents: torch.Tensor,
        media: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, num_queries, dim]
            media: [B, L, dim] (already projected to working dim)

        Returns:
            Updated latents: [B, num_queries, dim]
        """
        B, N, D = latents.shape
        H = self.heads

        # Pre-norm
        latents_normed = self.norm_latents(latents)
        media_normed = self.norm_media(media)

        # Q from latents, K/V from concat(media, latents) — implicit self-attn
        q = self.to_q(latents_normed)  # [B, N, inner]
        kv_input = torch.cat([media_normed, latents_normed], dim=1)  # [B, L+N, dim]
        kv = self.to_kv(kv_input)  # [B, L+N, 2*inner]
        k, v = kv.chunk(2, dim=-1)  # each [B, L+N, inner]

        # Reshape for multi-head attention
        q = q.view(B, N, H, self.dim_head).transpose(1, 2)      # [B, H, N, dh]
        k = k.view(B, -1, H, self.dim_head).transpose(1, 2)     # [B, H, L+N, dh]
        v = v.view(B, -1, H, self.dim_head).transpose(1, 2)     # [B, H, L+N, dh]

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, H, N, dh]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, inner]
        attn_out = self.to_out(attn_out)

        # Residual
        latents = latents + attn_out

        # FFN with SiLU gating (residual)
        residual = latents
        x = self.ffn_norm(latents)
        latents = residual + self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))

        return latents


class FlamingoPerceiverResampler(nn.Module):
    """Flamingo-style Perceiver Resampler with residue position embeddings.

    Maps variable-length encoder embeddings [B, L, encoder_dim] to fixed-size
    output [B, num_queries, output_dim] suitable for cross-attention in LLM.

    Key design choices (from Flamingo paper):
      - 64 learned query tokens (Suggestion 3)
      - 6 layers with implicit self-attention (Suggestion 4)
      - No dropout, no bias in linear layers (Suggestion 5)
      - Residue position embeddings for sequence position (Suggestion 6)
      - 4x FFN expansion, SiLU activation

    Args:
        encoder_dim: Input dimension from protein encoder (e.g., 1536 for ESM-3).
        output_dim: Output dimension matching LLM hidden size (e.g., 2560).
        latent_dim: Internal working dimension. Defaults to output_dim.
        num_queries: Number of output query tokens (default 64).
        num_layers: Number of Perceiver layers (default 6).
        max_seq_len: Maximum sequence length for position embeddings (default 2048).
        num_heads: Number of attention heads (default 8).
        dim_head: Dimension per attention head (default 64).
        ff_mult: FFN expansion factor (default 4).
    """

    def __init__(
        self,
        encoder_dim: int = 1536,
        output_dim: int = 2560,
        latent_dim: Optional[int] = None,
        num_queries: int = 64,
        num_layers: int = 6,
        max_seq_len: int = 2048,
        num_heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        working_dim = latent_dim if latent_dim is not None else output_dim

        if working_dim % num_heads != 0:
            # Check dim_head * num_heads matches working_dim, or allow flexible
            pass  # We use dim_head directly, not working_dim // num_heads

        self.encoder_dim = encoder_dim
        self.output_dim = output_dim
        self.latent_dim = working_dim
        self.num_queries = num_queries
        self.num_layers_count = num_layers
        self.max_seq_len = max_seq_len

        # Input projection: encoder_dim -> working_dim (bias=False)
        self.input_proj = nn.Linear(encoder_dim, working_dim, bias=False)

        # Residue position embeddings (Suggestion 6)
        # Added to encoder output BEFORE input projection
        self.residue_pos_embed = nn.Parameter(
            torch.randn(max_seq_len, encoder_dim) * 0.02
        )

        # Learned query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_queries, working_dim))

        # Perceiver layers (all at working_dim)
        self.layers = nn.ModuleList([
            FlamingoPerceiverLayer(
                dim=working_dim,
                dim_head=dim_head,
                heads=num_heads,
                ff_mult=ff_mult,
            )
            for _ in range(num_layers)
        ])

        # Output norm + projection
        self.output_norm = nn.LayerNorm(working_dim)
        if working_dim != output_dim:
            self.output_proj = nn.Linear(working_dim, output_dim, bias=False)
        else:
            self.output_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        # Xavier for projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)

        # Normal init for queries (standard in Perceiver/Flamingo)
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

        # Small init for position embeddings
        nn.init.normal_(self.residue_pos_embed, mean=0.0, std=0.02)

        # Initialize attention/FFN layers
        for layer in self.layers:
            # Scale output projections by 1/sqrt(2*num_layers)
            scale = 1.0 / math.sqrt(2 * self.num_layers_count)
            nn.init.xavier_uniform_(layer.to_out.weight)
            layer.to_out.weight.data *= scale
            nn.init.xavier_uniform_(layer.ffn_down.weight)
            layer.ffn_down.weight.data *= scale

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Encoder output [B, L, encoder_dim].
            attention_mask: Optional mask [B, L] (currently unused, kept
                for API compatibility with PerceiverResampler).

        Returns:
            Resampled output [B, num_queries, output_dim].
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor [B, L, D], got {x.dim()}D tensor"
            )

        batch_size, seq_len, _ = x.shape

        # Add residue position embeddings (Suggestion 6)
        # Truncate or pad position embeddings to match sequence length
        pos_embed = self.residue_pos_embed[:seq_len]  # [L, encoder_dim]
        x = x + pos_embed.unsqueeze(0)  # [B, L, encoder_dim]

        # Project encoder output to working dimension
        media = self.input_proj(x)  # [B, L, working_dim]

        # Expand query tokens for batch
        latents = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Pass through Perceiver layers
        for layer in self.layers:
            latents = layer(latents, media)

        # Output norm + projection
        latents = self.output_norm(latents)
        latents = self.output_proj(latents)  # [B, num_queries, output_dim]

        return latents

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
            f"num_layers={self.num_layers_count}, "
            f"max_seq_len={self.max_seq_len}"
        )
