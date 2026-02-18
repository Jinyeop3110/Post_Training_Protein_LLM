"""
Protein Encoder Module

This module provides different strategies for encoding proteins:
1. Text-based: Raw protein sequence as text (e.g., "MKTLLILAVVAAALA...")
2. Embedding-based: Using pretrained protein language models (ESM-2, ProtTrans, etc.)
3. TBD: Third approach to be determined

"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn


class ProteinEncoder(ABC):
    """Abstract base class for protein encoders."""

    @abstractmethod
    def encode(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Encode protein sequences.

        Args:
            sequences: List of protein sequences (amino acid strings)

        Returns:
            Dictionary containing encoded representations
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of the output embeddings."""
        pass


class TextProteinEncoder(ProteinEncoder):
    """
    Approach 1: Encode proteins as raw text.

    Simply formats protein sequences for direct input to LLM tokenizer.
    No additional embeddings - relies on LLM's text understanding.
    """

    def __init__(self, format_template: str = "<protein>{sequence}</protein>"):
        self.format_template = format_template

    def encode(self, sequences: List[str]) -> Dict[str, Any]:
        """Format protein sequences as text strings."""
        formatted = [
            self.format_template.format(sequence=seq)
            for seq in sequences
        ]
        return {"text": formatted, "type": "text"}

    def get_embedding_dim(self) -> int:
        """Text encoding doesn't have fixed embedding dim."""
        return -1  # Variable, depends on tokenizer


class ESMProteinEncoder(ProteinEncoder):
    """
    Approach 2: Use ESM-2 pretrained protein embeddings.

    Extracts per-residue or pooled embeddings from ESM-2 model.
    Common choices: esm2_t33_650M_UR50D (1280-dim), esm2_t36_3B_UR50D
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        pooling: str = "mean",  # "mean", "cls", "last", "per_residue"
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.pooling = pooling
        self.device = device
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self._embedding_dim = None

    def _load_model(self):
        """Lazy load ESM model."""
        if self.model is None:
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
                self.model_name
            )
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model = self.model.to(self.device)
            self.model.eval()

            # Get embedding dimension from model config
            self._embedding_dim = self.model.embed_dim

    def encode(self, sequences: List[str]) -> Dict[str, Any]:
        """Extract embeddings using ESM-2."""
        self._load_model()

        # Prepare batch
        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.model.num_layers],
                return_contacts=False
            )

        # Extract representations from last layer
        token_representations = results["representations"][self.model.num_layers]

        # Apply pooling strategy
        if self.pooling == "per_residue":
            # Return full sequence embeddings (B, L, D)
            embeddings = token_representations[:, 1:-1, :]  # Remove BOS/EOS
        elif self.pooling == "mean":
            # Mean pooling over sequence length
            # Exclude BOS and EOS tokens
            embeddings = token_representations[:, 1:-1, :].mean(dim=1)
        elif self.pooling == "cls":
            # Use CLS token (first position)
            embeddings = token_representations[:, 0, :]
        elif self.pooling == "last":
            # Use last token before EOS
            embeddings = token_representations[:, -2, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return {
            "embeddings": embeddings,
            "type": "embedding",
            "pooling": self.pooling,
            "dim": self._embedding_dim
        }

    def get_embedding_dim(self) -> int:
        self._load_model()
        return self._embedding_dim


class TBDProteinEncoder(ProteinEncoder):
    """
    Approach 3: To Be Determined

    Placeholder for third protein encoding approach.
    Potential options:
    - Structure-aware encodings (using AlphaFold embeddings)
    - Graph neural network representations
    - Hybrid text + embedding approach
    """

    def __init__(self):
        raise NotImplementedError(
            "Third protein encoding approach is TBD. "
            "Options: structure-aware, GNN, hybrid, etc."
        )

    def encode(self, sequences: List[str]) -> Dict[str, Any]:
        pass

    def get_embedding_dim(self) -> int:
        pass


def get_protein_encoder(
    encoder_type: str,
    **kwargs
) -> ProteinEncoder:
    """Factory function to get protein encoder by type."""

    encoders = {
        "text": TextProteinEncoder,
        "esm": ESMProteinEncoder,
        "esm2": ESMProteinEncoder,
        "tbd": TBDProteinEncoder,
    }

    if encoder_type.lower() not in encoders:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: {list(encoders.keys())}"
        )

    return encoders[encoder_type.lower()](**kwargs)
