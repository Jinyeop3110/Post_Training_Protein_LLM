"""Token-budget dynamic batch sampler.

Replaces fixed batch_size with a token budget: each micro-batch packs as many
examples as fit within ``max_tokens``. Short sequences get large batches (8-16),
long sequences get small batches (2-3). This maximizes GPU utilization while
keeping memory bounded.

Compatible with DDP via HF Accelerate's ``BatchSamplerShard``.
"""

import logging
from typing import Iterator, List, Sequence

from torch.utils.data import BatchSampler, Sampler

log = logging.getLogger(__name__)


class TokenBudgetBatchSampler(BatchSampler):
    """Batch sampler that groups indices by token budget rather than count.

    Takes a base sampler (e.g. LengthGroupedSampler) and greedily packs
    indices into batches until the token budget is exceeded, then starts
    a new batch. Single samples exceeding the budget get their own batch.

    For DDP, the total batch count is padded to a multiple of
    ``num_processes`` so ``BatchSamplerShard`` can split evenly.

    Args:
        sampler: Base sampler yielding dataset indices.
        lengths: Token length for each dataset index.
        max_tokens: Maximum total tokens per micro-batch.
        max_batch_size: Cap on samples per micro-batch (prevents OOM
            on many short sequences). Default 16.
        num_processes: Number of DDP processes for batch-count alignment.
            Default 1 (no alignment).
    """

    def __init__(
        self,
        sampler: Sampler,
        lengths: Sequence[int],
        max_tokens: int,
        max_batch_size: int = 16,
        num_processes: int = 1,
    ):
        # BatchSampler expects (sampler, batch_size, drop_last) but we
        # override __iter__ and __len__ entirely, so pass dummy values.
        self.base_sampler = sampler
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.num_processes = max(1, num_processes)
        # Expose batch_size for Accelerate's BatchSamplerShard introspection
        self.batch_size = max_batch_size

        # Pre-compute batches so __len__ is accurate
        self._batches: List[List[int]] = []
        self._build_batches()

    def _build_batches(self) -> None:
        """Greedily pack sampler indices into token-budgeted batches."""
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_tokens = 0

        for idx in self.base_sampler:
            length = self.lengths[idx]

            # Would this sample exceed the budget or batch-size cap?
            would_exceed_tokens = (current_tokens + length) > self.max_tokens
            would_exceed_size = len(current_batch) >= self.max_batch_size

            if current_batch and (would_exceed_tokens or would_exceed_size):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(idx)
            current_tokens += length

        # Flush remaining
        if current_batch:
            batches.append(current_batch)

        # Pad to multiple of num_processes for DDP alignment
        if self.num_processes > 1 and batches:
            remainder = len(batches) % self.num_processes
            if remainder != 0:
                pad_count = self.num_processes - remainder
                for i in range(pad_count):
                    # Repeat from the end (negligible gradient impact)
                    batches.append(batches[-(i + 1) % len(batches)])

        self._batches = batches

        # Log statistics
        if batches:
            sizes = [len(b) for b in batches]
            token_counts = [sum(self.lengths[i] for i in b) for b in batches]
            log.info(
                f"TokenBudgetBatchSampler: {len(batches)} batches, "
                f"batch_size range [{min(sizes)}, {max(sizes)}], "
                f"tokens/batch range [{min(token_counts)}, {max(token_counts)}], "
                f"budget={self.max_tokens}, cap={self.max_batch_size}"
            )

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)
