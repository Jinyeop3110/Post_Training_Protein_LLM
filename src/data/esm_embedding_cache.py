"""
ESM Embedding Cache backed by LMDB.

Pre-computed ESM-3 embeddings are stored in an LMDB database, keyed by
``sha256(sequence).hexdigest()``.  Each value contains a variable-length
per-residue embedding tensor (shape ``[L, 1536]`` for ESM-3 small).

This eliminates:
- ESM-3 GPU memory (~6 GB) during SFT/GRPO training
- ~40% of forward pass time (ESM-3 encoding is the bottleneck)

The cache stores raw encoder output (before pooling/projection) because
pooling and projection have trainable parameters that change during training.

Usage:
    cache = ESMEmbeddingCache("data/esm_cache/combined.lmdb")
    embedding = cache.get("MKMRFLGLV...")  # -> Tensor [L, 1536] or None
    cache.put("MKMRFLGLV...", embedding_tensor)
    "MKMRFLGLV..." in cache  # -> True
"""

import hashlib
import logging
import struct
from pathlib import Path
from typing import Optional

import torch

try:
    import lmdb

    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False


logger = logging.getLogger(__name__)

# Header format: 2 int32 values (num_rows, num_cols)
_HEADER_FMT = "<ii"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _sequence_key(sequence: str) -> bytes:
    """Compute cache key from protein sequence."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest().encode("ascii")


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a 2D tensor to bytes: header (rows, cols) + bf16 raw data."""
    # Ensure contiguous bf16 on CPU
    t = tensor.detach().cpu().to(torch.bfloat16).contiguous()
    rows, cols = t.shape
    header = struct.pack(_HEADER_FMT, rows, cols)
    # bf16 doesn't support .numpy(), use untyped_storage for raw bytes
    raw = bytes(t.untyped_storage())
    return header + raw


def _deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserialize bytes back to a bf16 tensor."""
    rows, cols = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
    raw = data[_HEADER_SIZE:]
    # torch.frombuffer shares memory with the bytes object
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(rows, cols)
    return t


class ESMEmbeddingCache:
    """LMDB-backed cache for pre-computed ESM-3 embeddings.

    Thread-safe for concurrent reads (LMDB is memory-mapped).
    Writes should be serialized (single writer at a time).

    Args:
        path: Path to LMDB database directory.
        map_size: Maximum database size in bytes (default: 100 GB).
        readonly: Open in read-only mode (no writes allowed).
    """

    def __init__(
        self,
        path: str,
        map_size: int = 100 * 1024**3,  # 100 GB
        readonly: bool = False,
    ):
        if not HAS_LMDB:
            raise ImportError("lmdb required: pip install lmdb")

        self.path = Path(path)
        self.readonly = readonly
        self._env = None
        self._map_size = map_size

    def _get_env(self) -> "lmdb.Environment":
        """Lazy-open LMDB environment."""
        if self._env is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._env = lmdb.open(
                str(self.path),
                map_size=self._map_size,
                readonly=self.readonly,
                readahead=False,  # Better for random access
                meminit=False,
                max_readers=128,
                lock=not self.readonly,  # No lock needed for read-only
            )
        return self._env

    def get(self, sequence: str) -> Optional[torch.Tensor]:
        """Look up pre-computed embedding for a protein sequence.

        Args:
            sequence: Amino acid sequence string.

        Returns:
            Tensor of shape [L, embed_dim] in bf16, or None if not cached.
        """
        key = _sequence_key(sequence)
        env = self._get_env()
        with env.begin(write=False) as txn:
            data = txn.get(key)
        if data is None:
            return None
        return _deserialize_tensor(data)

    def get_batch(self, sequences: list[str]) -> list[Optional[torch.Tensor]]:
        """Look up embeddings for multiple sequences in a single transaction.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            List of tensors (or None for cache misses), same order as input.
        """
        keys = [_sequence_key(seq) for seq in sequences]
        env = self._get_env()
        results = []
        with env.begin(write=False) as txn:
            for key in keys:
                data = txn.get(key)
                if data is None:
                    results.append(None)
                else:
                    results.append(_deserialize_tensor(data))
        return results

    def put(self, sequence: str, embedding: torch.Tensor) -> None:
        """Store a pre-computed embedding.

        Args:
            sequence: Amino acid sequence string.
            embedding: Tensor of shape [L, embed_dim].
        """
        if self.readonly:
            raise RuntimeError("Cache opened in read-only mode")
        key = _sequence_key(sequence)
        value = _serialize_tensor(embedding)
        env = self._get_env()
        with env.begin(write=True) as txn:
            txn.put(key, value)

    def put_batch(
        self, sequences: list[str], embeddings: list[torch.Tensor]
    ) -> int:
        """Store multiple embeddings in a single transaction.

        Args:
            sequences: List of amino acid sequences.
            embeddings: List of tensors, one per sequence.

        Returns:
            Number of entries written.
        """
        if self.readonly:
            raise RuntimeError("Cache opened in read-only mode")
        env = self._get_env()
        count = 0
        with env.begin(write=True) as txn:
            for seq, emb in zip(sequences, embeddings):
                key = _sequence_key(seq)
                value = _serialize_tensor(emb)
                txn.put(key, value)
                count += 1
        return count

    def __contains__(self, sequence: str) -> bool:
        """Check if a sequence is in the cache."""
        key = _sequence_key(sequence)
        env = self._get_env()
        with env.begin(write=False) as txn:
            return txn.get(key) is not None

    def __len__(self) -> int:
        """Number of cached embeddings."""
        env = self._get_env()
        return env.stat()["entries"]

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        n = len(self) if self._env is not None else "?"
        return f"ESMEmbeddingCache(path={self.path!r}, entries={n})"
