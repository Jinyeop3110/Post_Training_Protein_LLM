"""
Gated Cross-Attention for Flamingo-style multimodal fusion.

Implements cross-attention layers that are injected into the LLM decoder,
allowing protein information to be fused at multiple layers rather than
only at the input (prefix) level.

Key design (from Flamingo paper, Alayrac et al., 2022):
  - tanh(alpha) gating: gates init to 0 so model starts as original LLM
  - Cross-attention: Q from text, K/V from protein perceiver output
  - NO RoPE on cross-attention (protein features don't have positions)
  - Injected every N-th LLM decoder layer (default: every 4th)

Suggestion 1: Gated cross-attention at every 4th layer (+4.2% from gating)
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MaskedCrossAttention(nn.Module):
    """Cross-attention: Q from LLM hidden states, K/V from perceiver output.

    Uses own Q/K/V linear projections (no RoPE) with pre-norm on both
    text and visual inputs.

    Args:
        dim: LLM hidden dimension (text side).
        dim_visual: Visual/protein feature dimension (from perceiver).
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
    ) -> None:
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        # Pre-norms
        self.norm_text = nn.LayerNorm(dim)
        self.norm_visual = nn.LayerNorm(dim_visual)

        # Q from text, K/V from visual — no bias (Flamingo style)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim_visual, inner_dim, bias=False)
        self.to_v = nn.Linear(dim_visual, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: LLM hidden states [B, T, dim].
            media: Protein features from perceiver [B, N, dim_visual].

        Returns:
            Cross-attended output [B, T, dim].
        """
        B, T, _ = x.shape
        H = self.heads

        # Pre-norm
        x_normed = self.norm_text(x)
        media_normed = self.norm_visual(media)

        # Project to Q, K, V
        q = self.to_q(x_normed)       # [B, T, inner]
        k = self.to_k(media_normed)   # [B, N, inner]
        v = self.to_v(media_normed)   # [B, N, inner]

        # Reshape for multi-head attention
        q = q.view(B, T, H, self.dim_head).transpose(1, 2)   # [B, H, T, dh]
        k = k.view(B, -1, H, self.dim_head).transpose(1, 2)  # [B, H, N, dh]
        v = v.view(B, -1, H, self.dim_head).transpose(1, 2)  # [B, H, N, dh]

        # Scaled dot-product attention (no RoPE — protein features have no positions)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, H, T, dh]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.to_out(attn_out)


class GatedCrossAttentionBlock(nn.Module):
    """Gated cross-attention block with tanh gates initialized to 0.

    At initialization, tanh(0) = 0, so the block acts as identity
    (model starts as the original LLM). During training, gates learn
    to open and incorporate protein information.

    Formula:
        y = x + tanh(alpha) * cross_attn(x, media)
        z = y + tanh(beta) * ffn(y)

    Args:
        dim: LLM hidden dimension.
        dim_visual: Visual/protein feature dimension.
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
        ff_mult: FFN expansion factor.
    """

    def __init__(
        self,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        # Cross-attention
        self.cross_attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
        )

        # tanh gates — init to 0 so block starts as identity
        self.gate_attn = nn.Parameter(torch.tensor(0.0))
        self.gate_ffn = nn.Parameter(torch.tensor(0.0))

        # FFN with SiLU gating
        self.ffn_norm = nn.LayerNorm(dim)
        ff_inner = dim * ff_mult
        self.ffn_up = nn.Linear(dim, ff_inner, bias=False)
        self.ffn_gate = nn.Linear(dim, ff_inner, bias=False)
        self.ffn_down = nn.Linear(ff_inner, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: LLM hidden states [B, T, dim].
            media: Protein features [B, N, dim_visual].

        Returns:
            Gated output [B, T, dim].
        """
        # Gated cross-attention
        x = x + torch.tanh(self.gate_attn) * self.cross_attn(x, media)

        # Gated FFN
        residual = x
        h = self.ffn_norm(x)
        x = residual + torch.tanh(self.gate_ffn) * self.ffn_down(
            F.silu(self.ffn_gate(h)) * self.ffn_up(h)
        )

        return x


class FlamingoDecoderLayer(nn.Module):
    """Wrapper that prepends a GatedCrossAttentionBlock before a decoder layer.

    Stores ``_protein_features`` which should be set before the LLM forward
    pass and cleared afterward.

    Args:
        decoder_layer: Original LLM decoder layer (e.g., Qwen3DecoderLayer).
        gated_xattn: GatedCrossAttentionBlock to apply before the decoder layer.
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        gated_xattn: GatedCrossAttentionBlock,
    ) -> None:
        super().__init__()
        self.decoder_layer = decoder_layer
        self.gated_xattn = gated_xattn
        self._protein_features: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ):
        """Forward: apply gated XATTN (if features set) then original decoder layer."""
        if self._protein_features is not None:
            hidden_states = self.gated_xattn(hidden_states, self._protein_features)

        return self.decoder_layer(hidden_states, **kwargs)


def inject_cross_attention_layers(
    model: nn.Module,
    xattn_blocks: nn.ModuleList,
    xattn_every: int = 4,
) -> List[FlamingoDecoderLayer]:
    """Inject gated cross-attention at every N-th LLM decoder layer.

    Wraps selected decoder layers with FlamingoDecoderLayer, replacing them
    in-place in the model's layer list.

    Args:
        model: The LLM model (e.g., Qwen3ForCausalLM or PeftModel).
        xattn_blocks: ModuleList of GatedCrossAttentionBlock instances.
        xattn_every: Inject cross-attention every N-th layer.

    Returns:
        List of FlamingoDecoderLayer wrappers that were injected.
    """
    # Get the decoder layers from the model
    base_model = model
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()

    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        layers = base_model.model.layers
    else:
        raise ValueError(
            f"Cannot find decoder layers in model of type {type(base_model)}. "
            "Expected model.model.layers attribute."
        )

    num_layers = len(layers)
    injection_indices = list(range(xattn_every - 1, num_layers, xattn_every))

    if len(xattn_blocks) != len(injection_indices):
        raise ValueError(
            f"Expected {len(injection_indices)} XATTN blocks for "
            f"{num_layers} layers (every {xattn_every}), "
            f"got {len(xattn_blocks)}"
        )

    wrapped_layers = []
    for block_idx, layer_idx in enumerate(injection_indices):
        original_layer = layers[layer_idx]
        wrapped = FlamingoDecoderLayer(original_layer, xattn_blocks[block_idx])
        layers[layer_idx] = wrapped
        wrapped_layers.append(wrapped)
        logger.debug(f"Injected gated XATTN at decoder layer {layer_idx}")

    logger.info(
        f"Injected {len(wrapped_layers)} gated cross-attention layers "
        f"at indices {injection_indices}"
    )
    return wrapped_layers


def set_protein_features(
    model: nn.Module,
    features: torch.Tensor,
) -> None:
    """Set protein features on all FlamingoDecoderLayers for the next forward pass.

    Args:
        model: The LLM model with injected FlamingoDecoderLayers.
        features: Protein features from perceiver [B, N, dim].
    """
    base_model = model
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()

    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        layers = base_model.model.layers
    else:
        return

    for layer in layers:
        if isinstance(layer, FlamingoDecoderLayer):
            layer._protein_features = features


def clear_protein_features(model: nn.Module) -> None:
    """Clear protein features from all FlamingoDecoderLayers after forward pass.

    Args:
        model: The LLM model with injected FlamingoDecoderLayers.
    """
    base_model = model
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()

    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        layers = base_model.model.layers
    else:
        return

    for layer in layers:
        if isinstance(layer, FlamingoDecoderLayer):
            layer._protein_features = None


def get_xattn_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Collect all parameters from injected GatedCrossAttentionBlocks.

    Args:
        model: The LLM model with injected FlamingoDecoderLayers.

    Returns:
        List of parameters from all gated cross-attention blocks.
    """
    params = []
    base_model = model
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()

    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        layers = base_model.model.layers
    else:
        return params

    for layer in layers:
        if isinstance(layer, FlamingoDecoderLayer):
            params.extend(layer.gated_xattn.parameters())

    return params
