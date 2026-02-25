"""
MLP Projector Module

This module provides the MLPProjector class for mapping protein embeddings
from the ESM-3 encoder (1536-dim) to LLM hidden size as prefix tokens.

The projector serves as a bridge between the protein encoder and the language model,
transforming protein representations into a format compatible with the LLM's
embedding space.
"""

from typing import Any, Callable, Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig


class MLPProjector(nn.Module):
    """
    Multi-Layer Perceptron Projector for mapping protein embeddings to LLM space.

    This module projects ESM-3 embeddings to the LLM's hidden dimension,
    enabling the use of protein representations as prefix tokens for the LLM.

    Args:
        input_dim: Dimension of input embeddings (ESM-3 output). Default: 1280.
        hidden_dim: Dimension of intermediate hidden layers. Default: 2048.
        output_dim: Dimension of output embeddings (LLM hidden size). Default: 4096.
        num_layers: Number of MLP layers. Default: 2.
        activation: Activation function name ("gelu", "relu", "silu"). Default: "gelu".
        dropout: Dropout rate applied after each layer except the last. Default: 0.1.

    Example:
        >>> projector = MLPProjector(input_dim=1280, output_dim=4096)
        >>> protein_embeddings = torch.randn(2, 100, 1280)  # [B, N, input_dim]
        >>> projected = projector(protein_embeddings)  # [B, N, output_dim]
        >>> print(projected.shape)
        torch.Size([2, 100, 4096])
    """

    # Supported activation functions
    ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 2048,
        output_dim: int = 4096,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation_name = activation.lower()
        self.dropout_rate = dropout

        # Validate parameters
        self._validate_parameters()

        # Build the MLP layers
        self.layers = self._build_layers()

    def _validate_parameters(self) -> None:
        """Validate constructor parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")
        if self.activation_name not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {self.activation_name}. "
                f"Available: {list(self.ACTIVATIONS.keys())}"
            )
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout_rate}"
            )

    def _build_layers(self) -> nn.Sequential:
        """
        Build the MLP layers.

        For num_layers=1: input_dim -> output_dim
        For num_layers=2: input_dim -> hidden_dim -> output_dim
        For num_layers>2: input_dim -> hidden_dim -> ... -> hidden_dim -> output_dim

        """
        layers = []
        activation_cls = self.ACTIVATIONS[self.activation_name]

        if self.num_layers == 1:
            # Single layer: direct projection
            layers.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            # First layer: input_dim -> hidden_dim
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(activation_cls())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            # Intermediate layers: hidden_dim -> hidden_dim
            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(activation_cls())
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))

            # Final layer: hidden_dim -> output_dim
            layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projector.

        Args:
            x: Input tensor of shape [B, N, input_dim] where:
               - B is the batch size
               - N is the sequence length (number of residues/tokens)
               - input_dim is the embedding dimension

        Returns:
            Projected tensor of shape [B, N, output_dim]

        Raises:
            ValueError: If input tensor has incorrect dimensions.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor [B, N, D], got {x.dim()}D tensor"
            )
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {x.size(-1)}"
            )

        return self.layers(x)

    def get_output_dim(self) -> int:
        """Return the output dimension of the projector."""
        return self.output_dim

    def get_input_dim(self) -> int:
        """Return the input dimension of the projector."""
        return self.input_dim

    @classmethod
    def from_config(cls, cfg: Union[DictConfig, Dict[str, Any]]) -> "MLPProjector":
        """
        Create MLPProjector from a Hydra/OmegaConf configuration.

        Args:
            cfg: Configuration object with projector parameters.
                 Expected keys (all optional with defaults):
                 - input_dim: int (default 1280)
                 - hidden_dim: int (default 2048)
                 - output_dim: int (default 4096)
                 - num_layers: int (default 2)
                 - activation: str (default "gelu")
                 - dropout: float (default 0.1)

        Returns:
            Configured MLPProjector instance.

        Example:
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({
            ...     "input_dim": 1280,
            ...     "hidden_dim": 2048,
            ...     "output_dim": 4096,
            ...     "num_layers": 2,
            ...     "activation": "gelu",
            ...     "dropout": 0.1,
            ... })
            >>> projector = MLPProjector.from_config(cfg)
        """
        # Convert DictConfig to dict if necessary for .get() access
        if isinstance(cfg, DictConfig):
            config_dict = dict(cfg)
        else:
            config_dict = cfg

        return cls(
            input_dim=config_dict.get("input_dim", 1280),
            hidden_dim=config_dict.get("hidden_dim", 2048),
            output_dim=config_dict.get("output_dim", 4096),
            num_layers=config_dict.get("num_layers", 2),
            activation=config_dict.get("activation", "gelu"),
            dropout=config_dict.get("dropout", 0.1),
        )

    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_layers={self.num_layers}, "
            f"activation={self.activation_name}, "
            f"dropout={self.dropout_rate}"
        )


def get_projector(projector_type: str = "mlp", **kwargs) -> nn.Module:
    """
    Factory function to get a projector by type.

    Args:
        projector_type: Type of projector. Supported:
            - "mlp": MLPProjector (default)
            - "perceiver": PerceiverResampler (replaces both pooling + projection)
        **kwargs: Arguments passed to the projector constructor.

    Returns:
        Configured projector instance.

    Raises:
        ValueError: If projector_type is not recognized.
    """
    if projector_type.lower() == "perceiver":
        from src.models.perceiver import PerceiverResampler
        return PerceiverResampler(**kwargs)

    projectors = {
        "mlp": MLPProjector,
    }

    if projector_type.lower() not in projectors:
        raise ValueError(
            f"Unknown projector type: {projector_type}. "
            f"Available: {list(projectors.keys()) + ['perceiver']}"
        )

    return projectors[projector_type.lower()](**kwargs)
