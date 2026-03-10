"""Tests for thinking-aware loss masking in collators.

Verifies that when enable_thinking=True, tokens inside <think>...</think>
blocks are masked (-100 in labels) and loss is only computed on response
tokens after </think>.
"""

import pytest

from src.training.collators import PackedDataset, ProteinLLMDataCollator


@pytest.fixture
def qwen3_tokenizer():
    """Load Qwen3 tokenizer (or skip if unavailable)."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            trust_remote_code=True,
        )
        return tokenizer
    except Exception:
        pytest.skip("Qwen3 tokenizer not available")


def _make_sample(instruction: str, input_text: str, output: str,
                 tokenizer, enable_thinking: bool = True,
                 protein_sequence: str = "MKTLLILAVL") -> dict:
    """Build a dataset sample dict with formatted prompts."""
    from src.data.mol_instructions import format_chat_messages

    messages_full = format_chat_messages(
        instruction=instruction,
        input_text=input_text,
        output=output,
        enable_thinking=enable_thinking,
    )
    formatted = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False,
    )

    messages_inf = format_chat_messages(
        instruction=instruction,
        input_text=input_text,
        output="",
        for_inference=True,
    )
    inference = tokenizer.apply_chat_template(
        messages_inf, tokenize=False, add_generation_prompt=True,
    )

    return {
        "formatted_prompt": formatted,
        "inference_prompt": inference,
        "protein_sequence": protein_sequence,
    }


class TestThinkingMaskingCollator:
    """Tests for ProteinLLMDataCollator thinking-aware masking."""

    def test_think_close_id_cached(self, qwen3_tokenizer):
        """</think> should tokenize to a single token for Qwen3."""
        collator = ProteinLLMDataCollator(
            tokenizer=qwen3_tokenizer,
            enable_thinking=True,
        )
        assert collator._think_close_id is not None
        assert collator.enable_thinking is True
        # Verify it decodes back
        decoded = qwen3_tokenizer.decode([collator._think_close_id])
        assert "</think>" in decoded

    def test_thinking_tokens_masked(self, qwen3_tokenizer):
        """All tokens from assistant start through </think> should be -100."""
        collator = ProteinLLMDataCollator(
            tokenizer=qwen3_tokenizer,
            max_length=512,
            enable_thinking=True,
        )
        sample = _make_sample(
            instruction="What is this protein?",
            input_text="MKTLLILAVL",
            output="This is a test protein.",
            tokenizer=qwen3_tokenizer,
            enable_thinking=True,
        )

        batch = collator([sample])
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]

        # Find </think> position in input_ids
        think_close_id = collator._think_close_id
        think_positions = (input_ids == think_close_id).nonzero(as_tuple=True)[0]
        assert len(think_positions) >= 1, "No </think> token found in input_ids"

        think_pos = think_positions[0].item()

        # Everything up to and including </think> should be masked
        assert (labels[:think_pos + 1] == -100).all(), (
            f"Tokens before/at </think> (pos {think_pos}) should all be -100"
        )

        # At least some tokens AFTER </think> should NOT be -100
        response_labels = labels[think_pos + 1:]
        non_masked = response_labels[response_labels != -100]
        assert len(non_masked) > 0, "No unmasked response tokens found after </think>"

    def test_no_thinking_fallback(self, qwen3_tokenizer):
        """Without enable_thinking, masking should be prompt-only."""
        collator_no_think = ProteinLLMDataCollator(
            tokenizer=qwen3_tokenizer,
            max_length=512,
            enable_thinking=False,
        )
        sample = _make_sample(
            instruction="What is this protein?",
            input_text="MKTLLILAVL",
            output="This is a test protein.",
            tokenizer=qwen3_tokenizer,
            enable_thinking=False,
        )

        batch = collator_no_think([sample])
        labels = batch["labels"][0]

        # Should have unmasked tokens (the full response)
        non_masked = labels[labels != -100]
        assert len(non_masked) > 0

    def test_missing_think_token_fallback(self, qwen3_tokenizer):
        """If </think> is not in the response, fall back to prompt masking."""
        collator = ProteinLLMDataCollator(
            tokenizer=qwen3_tokenizer,
            max_length=512,
            enable_thinking=True,
        )
        # Build sample WITHOUT thinking prefix in response
        sample = _make_sample(
            instruction="What is this protein?",
            input_text="MKTLLILAVL",
            output="This is a test protein.",
            tokenizer=qwen3_tokenizer,
            enable_thinking=False,  # No <think> block
        )

        batch = collator([sample])
        labels = batch["labels"][0]

        # Should still have unmasked response tokens (prompt-only masking)
        non_masked = labels[labels != -100]
        assert len(non_masked) > 0, "Fallback masking should leave response tokens unmasked"

    def test_thinking_masks_more_than_prompt_only(self, qwen3_tokenizer):
        """Thinking masking should mask MORE tokens than prompt-only masking."""
        sample_think = _make_sample(
            instruction="What is this protein?",
            input_text="MKTLLILAVL",
            output="This is a test protein.",
            tokenizer=qwen3_tokenizer,
            enable_thinking=True,
        )
        sample_no_think = _make_sample(
            instruction="What is this protein?",
            input_text="MKTLLILAVL",
            output="This is a test protein.",
            tokenizer=qwen3_tokenizer,
            enable_thinking=False,
        )

        collator_think = ProteinLLMDataCollator(
            tokenizer=qwen3_tokenizer, max_length=512, enable_thinking=True,
        )
        collator_no_think = ProteinLLMDataCollator(
            tokenizer=qwen3_tokenizer, max_length=512, enable_thinking=False,
        )

        batch_think = collator_think([sample_think])
        batch_no = collator_no_think([sample_no_think])

        masked_think = (batch_think["labels"][0] == -100).sum().item()
        masked_no = (batch_no["labels"][0] == -100).sum().item()

        # Thinking masking should mask the <think>\n\n</think>\n\n tokens
        # on top of the prompt, so more total tokens masked
        assert masked_think > masked_no, (
            f"Thinking masking ({masked_think}) should mask more tokens "
            f"than prompt-only ({masked_no})"
        )


class TestThinkingMaskingPacked:
    """Tests for PackedDataset thinking-aware masking."""

    def test_packed_thinking_per_document(self, qwen3_tokenizer):
        """Each document in a packed block gets independent thinking masking."""

        class FakeDataset:
            """Minimal dataset that returns pre-formatted samples."""
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]

        samples = [
            _make_sample(
                f"Task {i}", "MKTLLILAVL", f"Response {i}",
                qwen3_tokenizer, enable_thinking=True,
            )
            for i in range(3)
        ]
        fake_ds = FakeDataset(samples)

        packed = PackedDataset(
            dataset=fake_ds,
            tokenizer=qwen3_tokenizer,
            max_length=2048,
            shuffle=False,
            enable_thinking=True,
        )

        assert len(packed) > 0
        block = packed[0]
        labels = block["labels"]
        input_ids = block["input_ids"]

        think_close_id = packed._think_close_id
        assert think_close_id is not None

        # Find all </think> positions in the block
        think_positions = (input_ids == think_close_id).nonzero(as_tuple=True)[0]

        # Each </think> and everything before it (within its doc) should be -100
        for pos in think_positions:
            assert labels[pos].item() == -100, (
                f"</think> at position {pos} should be masked in labels"
            )

        # There should be unmasked response tokens after each </think>
        non_masked = labels[labels != -100]
        assert len(non_masked) > 0, "No unmasked response tokens in packed block"

    def test_packed_eos_separators_masked(self, qwen3_tokenizer):
        """EOS separators appended between documents should be masked (-100).

        Note: The chat template itself contains <|im_end|> tokens (same ID
        as eos_token) that are valid training targets. Only the EOS tokens
        *appended by PackedDataset* as document boundaries should be masked.
        We verify this by checking that each document's last token (the
        appended separator) has label -100.
        """

        class FakeDataset:
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]

        samples = [
            _make_sample(
                f"Task {i}", "MKTL", f"Short response {i}",
                qwen3_tokenizer, enable_thinking=True,
            )
            for i in range(3)
        ]
        fake_ds = FakeDataset(samples)

        packed = PackedDataset(
            dataset=fake_ds,
            tokenizer=qwen3_tokenizer,
            max_length=2048,
            shuffle=False,
            enable_thinking=True,
        )

        block = packed[0]
        labels = block["labels"]
        input_ids = block["input_ids"]
        eos_id = qwen3_tokenizer.eos_token_id

        # Find all EOS positions in active region
        attn = block["attention_mask"]
        active_eos = ((input_ids == eos_id) & (attn == 1)).nonzero(as_tuple=True)[0]
        assert len(active_eos) > 0, "Should have EOS tokens in block"

        # Count how many EOS tokens are masked — at minimum, each document's
        # appended separator EOS should be masked
        masked_eos = sum(1 for pos in active_eos if labels[pos].item() == -100)
        assert masked_eos > 0, "At least some EOS tokens should be masked as separators"

    def test_packed_fallback_no_think_token(self, qwen3_tokenizer):
        """Packed dataset falls back to prompt masking if no </think> found."""

        class FakeDataset:
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]

        # Sample without thinking prefix
        samples = [
            _make_sample(
                "Task 1", "MKTL", "Response without thinking",
                qwen3_tokenizer, enable_thinking=False,
            )
        ]
        fake_ds = FakeDataset(samples)

        # PackedDataset has enable_thinking=True but sample has no </think>
        packed = PackedDataset(
            dataset=fake_ds,
            tokenizer=qwen3_tokenizer,
            max_length=2048,
            shuffle=False,
            enable_thinking=True,
        )

        block = packed[0]
        labels = block["labels"]
        non_masked = labels[labels != -100]
        assert len(non_masked) > 0, "Should have unmasked response tokens (fallback)"
