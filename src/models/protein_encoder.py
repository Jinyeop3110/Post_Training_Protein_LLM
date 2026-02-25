"""
Protein Encoder Module

This module provides different strategies for encoding proteins:
1. Text-based: Raw protein sequence as text (e.g., "MKTLLILAVVAAALA...")
2. ESM-3 Embedding-based: Using ESM-3 pretrained multimodal protein model

Supported encoders:
- TextProteinEncoder: Raw sequence as formatted text for LLM tokenizer
- ESM3ProteinEncoder: ESM-3 sequence track embeddings (1536-dim for small)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


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


class ESM3ProteinEncoder(ProteinEncoder):
    """
    ESM-3 pretrained protein encoder (sequence track only).

    ESM-3 is a multimodal generative model from EvolutionaryScale that jointly
    models sequence, structure, and function. We use only the sequence track
    embeddings for protein representation.

    Model variants:
        - esm3-sm-open-v1: 1.4B parameters, 1536-dim embeddings

    The model is always frozen (requires_grad=False) to preserve pretrained
    protein knowledge.

    Requires the EvolutionaryScale ESM package:
        pip install esm

    Args:
        model_name: ESM-3 model identifier. Default: "esm3-sm-open-v1".
        device: Device to load the model on. Default: "cuda".
    """

    # Known ESM-3 model dimensions
    ESM3_EMBED_DIMS: Dict[str, int] = {
        "esm3-sm-open-v1": 1536,
        "esm3_sm_open_v1": 1536,
    }

    def __init__(
        self,
        model_name: str = "esm3-sm-open-v1",
        device: str = "cuda",
        dtype: str = "bfloat16",
        encoder_batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.encoder_batch_size = encoder_batch_size
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._embedding_dim: Optional[int] = self.ESM3_EMBED_DIMS.get(
            model_name, None
        )

        # Resolve autocast dtype for forward pass (halves activation memory).
        # Model weights stay float32 (ESM-3's internal RBF functions create
        # float32 intermediates), but matmuls/activations run in this dtype.
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.autocast_dtype = dtype_map.get(dtype, torch.bfloat16)

    def _load_model(self) -> None:
        """Lazy load ESM-3 model and tokenizer.

        Uses the EvolutionaryScale ESM package to load the pretrained model.
        The model is set to eval mode and all parameters are frozen.

        Raises:
            ImportError: If the EvolutionaryScale ESM package is not installed.
            RuntimeError: If model loading fails.
        """
        if self.model is not None:
            return

        try:
            from esm.models.esm3 import ESM3
        except ImportError:
            raise ImportError(
                "ESM-3 requires the EvolutionaryScale ESM package. "
                "Install with: pip install esm\n"
                "See https://github.com/evolutionaryscale/esm"
            )

        logger.info(f"Loading ESM-3 model: {self.model_name}")

        # Load ESM-3 in float32 (internal RBF functions create float32 tensors).
        # Autocast to bf16 during forward pass to halve activation memory.
        self.model = ESM3.from_pretrained(self.model_name).to(self.device).float()
        self.model.eval()

        # Freeze all parameters - CRITICAL: ESM-3 must be frozen
        for param in self.model.parameters():
            param.requires_grad = False

        # Get tokenizers for batched sequence encoding
        try:
            from esm.tokenization import get_esm3_model_tokenizers
            self.tokenizer = get_esm3_model_tokenizers(self.model_name)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load ESM-3 tokenizers: {e}")
            self.tokenizer = None

        # Determine embedding dimension from model config
        if self._embedding_dim is None:
            # Fallback: inspect model hidden size
            if hasattr(self.model, "embed_dim"):
                self._embedding_dim = self.model.embed_dim
            elif hasattr(self.model, "config") and hasattr(
                self.model.config, "hidden_size"
            ):
                self._embedding_dim = self.model.config.hidden_size
            else:
                raise RuntimeError(
                    f"Cannot determine embedding dimension for {self.model_name}. "
                    "Please specify it in ESM3_EMBED_DIMS."
                )

        logger.info(
            f"ESM-3 loaded: {self.model_name} "
            f"(embedding_dim={self._embedding_dim}, frozen=True)"
        )

    def encode(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Extract per-residue sequence embeddings from ESM-3.

        Tokenizes all sequences, pads to max length, and runs a single
        batched forward pass through ESM-3. Falls back to sequential
        encoding if the tokenizer is unavailable.

        Args:
            sequences: List of protein sequences (amino acid strings).

        Returns:
            Dictionary containing:
                - embeddings: Tensor of shape [B, L, D] with per-residue
                  embeddings where D is the embedding dimension (1536 for small).
                  Padding positions are zeroed.
                - type: "embedding"
                - encoder: "esm3"
                - dim: Embedding dimension
        """
        self._load_model()

        if self.tokenizer is not None:
            embeddings = self._encode_batched(sequences)
        else:
            embeddings = self._encode_sequential(sequences)

        return {
            "embeddings": embeddings,
            "type": "embedding",
            "encoder": "esm3",
            "dim": self._embedding_dim,
        }

    def _encode_batched(self, sequences: List[str]) -> torch.Tensor:
        """Batched ESM-3 encoding with sub-batching for memory safety.

        Tokenizes all sequences with the ESM-3 sequence tokenizer, then
        processes them in sub-batches of ``encoder_batch_size`` to decouple
        ESM-3 memory from the LLM's (potentially larger) batch size.

        Args:
            sequences: List of protein sequences.

        Returns:
            Tensor of shape [B, L_max, D] with zero-padded embeddings.
        """
        import torch.nn.functional as F
        from esm.utils.encoding import tokenize_sequence

        seq_tokenizer = self.tokenizer.sequence
        pad_id = seq_tokenizer.pad_token_id

        # 1. Tokenize all sequences (each gets BOS + residues + EOS)
        token_lists = [
            tokenize_sequence(seq, seq_tokenizer, add_special_tokens=True)
            for seq in sequences
        ]
        seq_lengths = [len(t) - 2 for t in token_lists]  # AA count (no BOS/EOS)

        # 2. Process in sub-batches of encoder_batch_size
        all_embeddings = []
        B = len(sequences)
        bs = self.encoder_batch_size

        for start in range(0, B, bs):
            end = min(start + bs, B)
            chunk_tokens = token_lists[start:end]
            chunk_lengths = seq_lengths[start:end]

            # Pad within this sub-batch only (less padding waste)
            max_tok_len = max(len(t) for t in chunk_tokens)
            padded = [
                F.pad(t, (0, max_tok_len - len(t)), value=pad_id)
                for t in chunk_tokens
            ]
            batch_tokens = torch.stack(padded).to(self.device)

            # Sequence_id mask for attention
            sequence_id = (batch_tokens != pad_id)

            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=self.autocast_dtype
            ):
                output = self.model.forward(
                    sequence_tokens=batch_tokens, sequence_id=sequence_id
                )
            embeddings = output.embeddings  # [chunk_B, max_tok_len, D]

            # Strip BOS and EOS → per-residue embeddings
            embeddings = embeddings[:, 1:-1, :]

            # Zero out padding positions
            max_seq_len = max(chunk_lengths)
            for i, slen in enumerate(chunk_lengths):
                if slen < max_seq_len:
                    embeddings[i, slen:, :] = 0.0

            all_embeddings.append(embeddings)

        # 3. Pad sub-batch results to the global max length and concatenate
        global_max_len = max(e.shape[1] for e in all_embeddings)
        if any(e.shape[1] < global_max_len for e in all_embeddings):
            padded_embeddings = []
            for e in all_embeddings:
                if e.shape[1] < global_max_len:
                    pad = torch.zeros(
                        e.shape[0], global_max_len - e.shape[1], e.shape[2],
                        device=e.device, dtype=e.dtype,
                    )
                    e = torch.cat([e, pad], dim=1)
                padded_embeddings.append(e)
            return torch.cat(padded_embeddings, dim=0)

        return torch.cat(all_embeddings, dim=0)

    def _encode_sequential(self, sequences: List[str]) -> torch.Tensor:
        """Fallback: encode sequences one at a time (when tokenizer unavailable).

        Args:
            sequences: List of protein sequences.

        Returns:
            Tensor of shape [B, L_max, D] with zero-padded embeddings.
        """
        try:
            from esm.sdk.api import ESMProtein
        except ImportError:
            raise ImportError(
                "ESM-3 API not available. Install with: pip install esm"
            )

        all_embeddings = []

        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=self.autocast_dtype
        ):
            for seq in sequences:
                protein = ESMProtein(sequence=seq)
                protein_tensor = self.model.encode(protein)
                output = self.model.forward(
                    sequence_tokens=protein_tensor.sequence.unsqueeze(0).to(
                        self.device
                    )
                )
                embedding = output.embeddings[:, 1:-1, :]  # [1, L, D]
                all_embeddings.append(embedding)

        # Pad to max length and stack
        if len(all_embeddings) == 1:
            return all_embeddings[0]

        max_len = max(e.shape[1] for e in all_embeddings)
        padded = []
        for e in all_embeddings:
            if e.shape[1] < max_len:
                pad = torch.zeros(
                    1, max_len - e.shape[1], e.shape[2],
                    device=e.device, dtype=e.dtype,
                )
                e = torch.cat([e, pad], dim=1)
            padded.append(e)
        return torch.cat(padded, dim=0)

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension of ESM-3.

        Returns:
            Embedding dimension (1536 for esm3-sm-open-v1).
        """
        if self._embedding_dim is not None:
            return self._embedding_dim
        self._load_model()
        return self._embedding_dim


class TBDProteinEncoder(ProteinEncoder):
    """
    Approach TBD: Placeholder for future protein encoding approaches.

    Potential options:
    - Structure-aware encodings (using AlphaFold embeddings)
    - Graph neural network representations
    - Hybrid text + embedding approach
    """

    def __init__(self) -> None:
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
    **kwargs: Any,
) -> ProteinEncoder:
    """
    Factory function to get a protein encoder by type.

    Args:
        encoder_type: Type of encoder to instantiate. One of:
            - "text": TextProteinEncoder (raw sequence as text)
            - "esm3": ESM3ProteinEncoder (ESM-3 sequence embeddings)
            - "tbd": TBDProteinEncoder (placeholder)
        **kwargs: Additional arguments passed to the encoder constructor.

    Returns:
        Instantiated ProteinEncoder.

    Raises:
        ValueError: If encoder_type is not recognized.
    """
    encoders: Dict[str, type] = {
        "text": TextProteinEncoder,
        "esm3": ESM3ProteinEncoder,
        "tbd": TBDProteinEncoder,
    }

    if encoder_type.lower() not in encoders:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: {list(encoders.keys())}"
        )

    return encoders[encoder_type.lower()](**kwargs)
