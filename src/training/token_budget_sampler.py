"""Token-budget dynamic batch sampler.

Replaces fixed batch_size with a token budget: each micro-batch packs as many
examples as fit within ``max_tokens``. Short sequences get large batches (8-16),
long sequences get small batches (2-3). This maximizes GPU utilization while
keeping memory bounded.

Compatible with DDP via HF Accelerate's ``BatchSamplerShard``.
"""

import logging
from typing import Any, Dict, Iterator, List, Sequence

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
        self._batch_idx: int = 0  # absolute index of next unprocessed batch
        self._batch_offset: int = 0  # offset from load_state_dict
        self._build_batches()

    def _build_batches(self) -> None:
        """Greedily pack sampler indices into token-budgeted batches.

        Uses *padded* token count (max_length * batch_size) rather than
        sum-of-lengths to budget memory, because the collator pads all
        sequences in a batch to the longest one.  This prevents OOM on
        batches with high length variance (e.g. one 4000-token sample
        padded across 16 samples = 64K tokens vs sum = 7K).
        """
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0

        for idx in self.base_sampler:
            length = self.lengths[idx]

            # Estimate padded token count if we add this sample
            new_max_len = max(current_max_len, length)
            new_batch_size = len(current_batch) + 1
            padded_tokens = new_max_len * new_batch_size

            # Would this sample exceed the budget or batch-size cap?
            would_exceed_tokens = current_batch and padded_tokens > self.max_tokens
            would_exceed_size = len(current_batch) >= self.max_batch_size

            if current_batch and (would_exceed_tokens or would_exceed_size):
                batches.append(current_batch)
                current_batch = []
                current_max_len = 0

            current_batch.append(idx)
            current_max_len = max(current_max_len, length)

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
            padded_counts = [
                max(self.lengths[i] for i in b) * len(b) for b in batches
            ]
            raw_counts = [sum(self.lengths[i] for i in b) for b in batches]
            log.info(
                f"TokenBudgetBatchSampler: {len(batches)} batches, "
                f"batch_size range [{min(sizes)}, {max(sizes)}], "
                f"padded tokens/batch range [{min(padded_counts)}, {max(padded_counts)}], "
                f"raw tokens/batch range [{min(raw_counts)}, {max(raw_counts)}], "
                f"budget={self.max_tokens}, cap={self.max_batch_size}"
            )

    def __iter__(self) -> Iterator[List[int]]:
        for i, batch in enumerate(self._batches):
            self._batch_idx = self._batch_offset + i + 1
            yield batch

    def __len__(self) -> int:
        return len(self._batches)

    # ------------------------------------------------------------------
    # Checkpoint save / restore
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Save sampler position for checkpoint resume.

        Returns dict with the batch index so the sampler can skip
        already-consumed batches on resume instead of replaying from
        the start.
        """
        return {
            "batch_idx": getattr(self, "_batch_idx", 0),
            "total_batches": len(self._batches),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore sampler position from checkpoint.

        Slices ``_batches`` so the next ``__iter__`` yields only the
        remaining batches.  Safe to call before the training loop starts.
        """
        batch_idx = state.get("batch_idx", 0)
        total = len(self._batches) + self._batch_offset  # original total
        if batch_idx > 0 and batch_idx <= total:
            # Slice off consumed batches; adjust offset for __iter__ tracking
            skip_from_current = batch_idx - self._batch_offset
            if skip_from_current > 0:
                self._batches = self._batches[skip_from_current:]
            self._batch_offset = batch_idx
            self._batch_idx = batch_idx
            log.info(
                f"TokenBudgetBatchSampler: resuming from batch {batch_idx}/{total} "
                f"(skipped {batch_idx}, {len(self._batches)} remaining)"
            )
        elif batch_idx > total:
            log.warning(
                f"TokenBudgetBatchSampler: saved batch_idx {batch_idx} > "
                f"total {total}; starting from beginning"
            )
