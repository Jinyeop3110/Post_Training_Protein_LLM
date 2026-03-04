"""Data collators for ProteinLLM training.

Extracted from sft_trainer.py — data-processing classes with no model state.

Classes:
    ProteinLLMDataCollator: Tokenizes + masks labels for multimodal training.
    PackedDataset: Concatenation+packing for efficient SFT training.
    PackedDataCollator: Stacks pre-tokenized packed blocks.
"""

import logging
from typing import Any, Dict, List

import torch
import torch.utils.data

try:
    from transformers import PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


log = logging.getLogger(__name__)


class ProteinLLMDataCollator:
    """
    Custom data collator for ProteinLLM training.

    Handles tokenization while preserving protein sequences for the
    multimodal forward pass.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        padding: str = "longest",
        label_pad_token_id: int = -100,
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            label_pad_token_id: Token ID for padding labels.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.

        Masks labels so the loss is only computed on response tokens
        (everything after "### Response:\\n"). Instruction and input
        tokens are set to -100 in labels.

        Args:
            batch: List of samples from MolInstructionsDataset.

        Returns:
            Dict containing tokenized inputs and protein sequences.
        """
        # Extract formatted prompts and protein sequences
        prompts = [item["formatted_prompt"] for item in batch]
        protein_sequences = [item["protein_sequence"] for item in batch]

        # Tokenize the inference prompt (instruction + input, no response)
        # to determine where the response starts
        inference_prompts = [item["inference_prompt"] for item in batch]

        # Tokenize full prompts
        encoded = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize inference prompts (without padding) to get prompt lengths
        prompt_lengths = []
        for inf_prompt in inference_prompts:
            inf_ids = self.tokenizer.encode(
                inf_prompt, add_special_tokens=False, truncation=True,
                max_length=self.max_length,
            )
            prompt_lengths.append(len(inf_ids))

        # Create labels: mask prompt tokens with -100, keep response tokens
        labels = encoded["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = self.label_pad_token_id
        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "protein_sequences": protein_sequences,
        }


class PackedDataset(torch.utils.data.Dataset):
    """
    Concatenation+packing dataset for efficient SFT training.

    Pre-tokenizes all examples, concatenates them with EOS separators,
    and chunks into fixed-length blocks. Eliminates padding waste for
    variable-length protein instruction data.

    Each packed block contains multiple concatenated examples:
        [tokens_1] [EOS] [tokens_2] [EOS] ... [tokens_k] [EOS] [PAD...]

    Labels mask the EOS separators between documents with -100 so the
    model only learns to predict within each document.

    For multimodal (ESM-3) training, protein_sequences are stored per-block
    as a list of sequences from the constituent examples.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the packed dataset.

        Args:
            dataset: Source dataset (e.g., MolInstructionsDataset).
            tokenizer: HuggingFace tokenizer.
            max_length: Fixed block length for packed sequences.
            shuffle: Whether to shuffle examples before packing.
            seed: Random seed for shuffling.
        """
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.blocks: List[Dict[str, Any]] = []
        self._pack(dataset, shuffle=shuffle, seed=seed)

    def _pack(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Tokenize all examples and pack into fixed-length blocks."""
        import random

        log.info(f"Packing {len(dataset)} examples into blocks of {self.max_length} tokens...")

        # 1. Pre-tokenize all examples
        tokenized_examples = []
        for i in range(len(dataset)):
            item = dataset[i]
            tokens = self.tokenizer.encode(
                item["formatted_prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 1,  # Reserve space for EOS
            )
            # Determine prompt length (instruction+input) for label masking
            prompt_tokens = self.tokenizer.encode(
                item["inference_prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 1,
            )
            prompt_len = len(prompt_tokens)
            # Append EOS as document boundary
            tokens.append(self.eos_token_id)
            tokenized_examples.append({
                "token_ids": tokens,
                "prompt_len": prompt_len,
                "protein_sequence": item.get("protein_sequence", ""),
            })

        # 2. Shuffle before concatenation (so adjacent docs are random)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(tokenized_examples)

        # 3. Concatenate into a single stream and chunk into blocks
        current_tokens = []
        current_labels = []
        current_proteins = []
        current_protein_set = set()  # Track unique proteins in current block

        for ex in tokenized_examples:
            token_ids = ex["token_ids"]
            prompt_len = ex["prompt_len"]
            protein_seq = ex["protein_sequence"]

            # If adding this example would exceed block size, finalize current block
            if len(current_tokens) + len(token_ids) > self.max_length:
                # If current block has content, pad and save it
                if current_tokens:
                    self._finalize_block(
                        current_tokens, current_labels,
                        list(current_proteins),
                    )

                # Start new block — if example itself is too long, it gets its own block
                current_tokens = []
                current_labels = []
                current_proteins = []
                current_protein_set = set()

            # Build labels: mask prompt tokens + boundary EOS, keep response tokens
            doc_labels = [-100] * prompt_len + list(token_ids[prompt_len:])
            # The last token (EOS) is a separator — mask it in labels
            doc_labels[-1] = -100

            current_tokens.extend(token_ids)
            current_labels.extend(doc_labels)

            # Track protein sequence (deduplicate within block)
            if protein_seq and protein_seq not in current_protein_set:
                current_proteins.append(protein_seq)
                current_protein_set.add(protein_seq)

        # Finalize last block
        if current_tokens:
            self._finalize_block(
                current_tokens, current_labels,
                list(current_proteins),
            )

        log.info(
            f"Packed into {len(self.blocks)} blocks "
            f"(from {len(tokenized_examples)} examples, "
            f"{len(self.blocks) * self.max_length:,} total tokens)"
        )

    def _finalize_block(
        self,
        tokens: List[int],
        labels: List[int],
        proteins: List[str],
    ) -> None:
        """Pad a block to max_length and add to self.blocks."""
        pad_len = self.max_length - len(tokens)

        input_ids = tokens + [self.pad_token_id] * pad_len
        attention_mask = [1] * len(tokens) + [0] * pad_len
        block_labels = labels + [-100] * pad_len

        self.blocks.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(block_labels, dtype=torch.long),
            "protein_sequences": proteins,
        })

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.blocks[idx]


class PackedDataCollator:
    """Simple collator for PackedDataset — just stacks pre-tokenized blocks."""

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "protein_sequences": [
                seq for b in batch for seq in b["protein_sequences"]
            ],
        }
