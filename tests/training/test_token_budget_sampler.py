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

    # ------------------------------------------------------------------
    # state_dict / load_state_dict tests
    # ------------------------------------------------------------------

    def test_state_dict_initial(self):
        """Initial state_dict has batch_idx=0."""
        lengths = [100, 100, 100]
        bs = TokenBudgetBatchSampler(
            ListSampler([0, 1, 2]), lengths, max_tokens=150
        )
        state = bs.state_dict()
        assert state["batch_idx"] == 0
        assert state["total_batches"] == 3

    def test_state_dict_after_partial_iteration(self):
        """state_dict tracks batch index during iteration."""
        lengths = [100] * 8
        bs = TokenBudgetBatchSampler(
            ListSampler(list(range(8))), lengths, max_tokens=150
        )
        it = iter(bs)
        next(it)  # consume batch 0
        next(it)  # consume batch 1
        state = bs.state_dict()
        assert state["batch_idx"] == 2

    def test_state_dict_after_full_iteration(self):
        """state_dict returns total batch count after full iteration."""
        lengths = [100] * 4
        bs = TokenBudgetBatchSampler(
            ListSampler(list(range(4))), lengths, max_tokens=150
        )
        list(bs)  # consume all
        state = bs.state_dict()
        assert state["batch_idx"] == len(list(TokenBudgetBatchSampler(
            ListSampler(list(range(4))), lengths, max_tokens=150
        )))

    def test_load_state_dict_skips_batches(self):
        """load_state_dict slices off already-consumed batches."""
        lengths = [100] * 10
        bs = TokenBudgetBatchSampler(
            ListSampler(list(range(10))), lengths, max_tokens=150
        )
        original_batches = list(bs)
        total = len(original_batches)

        # Create a fresh sampler and restore from batch 3
        bs2 = TokenBudgetBatchSampler(
            ListSampler(list(range(10))), lengths, max_tokens=150
        )
        bs2.load_state_dict({"batch_idx": 3, "total_batches": total})
        remaining = list(bs2)
        assert remaining == original_batches[3:]

    def test_load_state_dict_preserves_tracking(self):
        """After load_state_dict, subsequent state_dict returns correct absolute position."""
        lengths = [100] * 10
        bs = TokenBudgetBatchSampler(
            ListSampler(list(range(10))), lengths, max_tokens=150
        )
        total_batches = len(bs)

        # Restore from batch 5, then consume 2 more
        bs.load_state_dict({"batch_idx": 5, "total_batches": total_batches})
        it = iter(bs)
        next(it)  # batch 5
        next(it)  # batch 6
        state = bs.state_dict()
        assert state["batch_idx"] == 7  # absolute: 5 + 2

    def test_load_state_dict_zero_is_noop(self):
        """load_state_dict with batch_idx=0 doesn't change anything."""
        lengths = [100] * 4
        bs = TokenBudgetBatchSampler(
            ListSampler(list(range(4))), lengths, max_tokens=150
        )
        original_len = len(bs)
        bs.load_state_dict({"batch_idx": 0, "total_batches": original_len})
        assert len(bs) == original_len

    def test_load_state_dict_exceeding_warns(self):
        """load_state_dict with batch_idx > total logs warning, doesn't crash."""
        lengths = [100] * 4
        bs = TokenBudgetBatchSampler(
            ListSampler(list(range(4))), lengths, max_tokens=150
        )
        original_len = len(bs)
        bs.load_state_dict({"batch_idx": 999, "total_batches": 999})
        # Should not crash; batches unchanged
        assert len(bs) == original_len

    def test_save_restore_roundtrip(self):
        """Full save→restore roundtrip produces correct remaining batches."""
        lengths = [50, 200, 150, 300, 100, 80, 120, 250]
        indices = list(range(8))

        # First pass: iterate halfway, save state
        bs1 = TokenBudgetBatchSampler(
            ListSampler(indices), lengths, max_tokens=400
        )
        all_batches = list(bs1)
        midpoint = len(all_batches) // 2

        # Simulate partial iteration
        bs1_partial = TokenBudgetBatchSampler(
            ListSampler(indices), lengths, max_tokens=400
        )
        it = iter(bs1_partial)
        for _ in range(midpoint):
            next(it)
        state = bs1_partial.state_dict()
        assert state["batch_idx"] == midpoint

        # Second pass: restore and get remaining
        bs2 = TokenBudgetBatchSampler(
            ListSampler(indices), lengths, max_tokens=400
        )
        bs2.load_state_dict(state)
        remaining = list(bs2)
        assert remaining == all_batches[midpoint:]
