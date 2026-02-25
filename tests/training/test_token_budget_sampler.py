"""Tests for TokenBudgetBatchSampler."""


from src.training.token_budget_sampler import TokenBudgetBatchSampler


class ListSampler:
    """Simple sampler that yields indices from a list."""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TestTokenBudgetBatchSampler:
    def test_basic_batching(self):
        """All samples fit in one batch."""
        lengths = [100, 100, 100]
        sampler = ListSampler([0, 1, 2])
        bs = TokenBudgetBatchSampler(sampler, lengths, max_tokens=500)
        batches = list(bs)
        assert len(batches) == 1
        assert batches[0] == [0, 1, 2]

    def test_splits_on_budget(self):
        """Batch splits when token budget is exceeded."""
        lengths = [300, 300, 300, 300]
        sampler = ListSampler([0, 1, 2, 3])
        bs = TokenBudgetBatchSampler(sampler, lengths, max_tokens=500)
        batches = list(bs)
        # 300+300=600 > 500, so each batch gets at most 1 sample
        # Actually: first sample (300) fits, second (300+300=600) exceeds -> flush
        assert len(batches) == 4
        for b in batches:
            assert len(b) == 1

    def test_greedy_packing(self):
        """Greedy packing fills batches up to budget."""
        lengths = [200, 200, 200, 200]
        sampler = ListSampler([0, 1, 2, 3])
        bs = TokenBudgetBatchSampler(sampler, lengths, max_tokens=500)
        batches = list(bs)
        # 200+200=400 fits, 400+200=600 exceeds -> [0,1], then [2,3]
        assert len(batches) == 2
        assert batches[0] == [0, 1]
        assert batches[1] == [2, 3]

    def test_max_batch_size_cap(self):
        """Batch size is capped even if token budget allows more."""
        lengths = [10] * 20
        sampler = ListSampler(list(range(20)))
        bs = TokenBudgetBatchSampler(
            sampler, lengths, max_tokens=10000, max_batch_size=5
        )
        batches = list(bs)
        assert all(len(b) <= 5 for b in batches)
        assert len(batches) == 4  # 20 / 5

    def test_oversized_single_sample(self):
        """Single sample exceeding budget gets its own batch."""
        lengths = [100, 2000, 100]
        sampler = ListSampler([0, 1, 2])
        bs = TokenBudgetBatchSampler(sampler, lengths, max_tokens=500)
        batches = list(bs)
        # idx=0 (100 fits), idx=1 (100+2000=2100 > 500) -> flush [0], start [1]
        # idx=2 (2000+100=2100 > 500) -> flush [1], start [2]
        assert len(batches) == 3
        assert batches[0] == [0]
        assert batches[1] == [1]
        assert batches[2] == [2]

    def test_ddp_alignment(self):
        """Batch count padded to multiple of num_processes."""
        lengths = [100, 100, 100]
        sampler = ListSampler([0, 1, 2])
        bs = TokenBudgetBatchSampler(
            sampler, lengths, max_tokens=150, num_processes=4
        )
        batches = list(bs)
        # 3 batches of 1 each, padded to 4
        assert len(batches) % 4 == 0
        assert len(batches) == 4

    def test_no_padding_single_process(self):
        """No padding with single process."""
        lengths = [100, 100, 100]
        sampler = ListSampler([0, 1, 2])
        bs = TokenBudgetBatchSampler(
            sampler, lengths, max_tokens=150, num_processes=1
        )
        batches = list(bs)
        assert len(batches) == 3

    def test_all_indices_present(self):
        """All dataset indices appear in exactly one batch."""
        lengths = [50, 200, 150, 300, 100, 80]
        sampler = ListSampler(list(range(6)))
        bs = TokenBudgetBatchSampler(sampler, lengths, max_tokens=400)
        batches = list(bs)
        all_indices = [idx for batch in batches for idx in batch]
        assert sorted(all_indices) == list(range(6))

    def test_len_matches_iter(self):
        """__len__ matches number of batches from __iter__."""
        lengths = [100, 200, 300, 400, 500]
        sampler = ListSampler(list(range(5)))
        bs = TokenBudgetBatchSampler(sampler, lengths, max_tokens=600)
        assert len(bs) == len(list(bs))

    def test_empty_sampler(self):
        """Empty sampler produces no batches."""
        bs = TokenBudgetBatchSampler(
            ListSampler([]), lengths=[], max_tokens=1000
        )
        assert len(bs) == 0
        assert list(bs) == []
