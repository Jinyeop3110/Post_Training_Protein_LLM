"""
Mol-Instructions Dataset Loader

Dataset loader for the Mol-Instructions protein-oriented subset from HuggingFace.
Contains ~505K instruction-following pairs for protein tasks including:
- Protein Design
- Catalytic Activity Prediction
- Protein Function Prediction
- Functional Description Generation
- Domain/Motif Prediction

Source: https://huggingface.co/datasets/zjunlp/Mol-Instructions
Paper: https://arxiv.org/abs/2306.08018 (ICLR 2024)

Data format:
    - instruction: Task description/prompt
    - input: Protein sequence or context
    - output: Expected response
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from datasets import load_dataset, Dataset as HFDataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


logger = logging.getLogger(__name__)


# Default prompt template for formatting instructions
DEFAULT_PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# Template without response (for inference)
INFERENCE_PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


@dataclass
class MolInstructionsConfig:
    """Configuration for Mol-Instructions dataset."""

    # Dataset source
    dataset_name: str = "zjunlp/Mol-Instructions"
    subset: str = "Protein-oriented Instructions"
    cache_dir: Optional[str] = None

    # Processing settings
    max_seq_length: int = 2048
    max_protein_length: Optional[int] = None

    # Split configuration
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    seed: int = 42

    # Prompt formatting
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    inference_template: str = INFERENCE_PROMPT_TEMPLATE

    # Filtering
    min_instruction_length: int = 10
    min_response_length: int = 1


class MolInstructionsDataset(Dataset):
    """
    PyTorch Dataset for Mol-Instructions protein-oriented subset.

    Each sample contains:
        - protein_sequence: The protein sequence from the input
        - instruction: The task instruction/prompt
        - response: The expected response/output
        - formatted_prompt: Full formatted prompt for LLM training
    """

    def __init__(
        self,
        split: str = "train",
        config: Optional[MolInstructionsConfig] = None,
        dataset_name: str = "zjunlp/Mol-Instructions",
        subset: str = "Protein-oriented Instructions",
        cache_dir: Optional[str] = None,
        max_seq_length: int = 2048,
        max_protein_length: Optional[int] = None,
        prompt_template: Optional[str] = None,
        transform: Optional[callable] = None,
        limit: Optional[int] = None,
    ):
        """
        Initialize the Mol-Instructions dataset.

        Args:
            split: Dataset split - "train", "validation", or "test"
            config: MolInstructionsConfig object (overrides other args if provided)
            dataset_name: HuggingFace dataset name
            subset: Dataset subset/configuration name
            cache_dir: Directory to cache the downloaded dataset
            max_seq_length: Maximum total sequence length for prompts
            max_protein_length: Maximum protein sequence length (None = no limit)
            prompt_template: Template for formatting prompts
            transform: Optional transform to apply to samples
            limit: Limit number of samples (for debugging/testing)
        """
        if not HAS_HF_DATASETS:
            raise ImportError(
                "HuggingFace datasets is required. Install with: pip install datasets"
            )

        # Use config if provided, otherwise use individual arguments
        if config is not None:
            self.config = config
        else:
            self.config = MolInstructionsConfig(
                dataset_name=dataset_name,
                subset=subset,
                cache_dir=cache_dir,
                max_seq_length=max_seq_length,
                max_protein_length=max_protein_length,
                prompt_template=prompt_template or DEFAULT_PROMPT_TEMPLATE,
            )

        self.split = split
        self.transform = transform
        self.limit = limit

        # Load and prepare dataset
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from HuggingFace and prepare splits."""
        logger.info(f"Loading Mol-Instructions dataset: {self.config.subset}")

        try:
            # Load the protein-oriented subset
            # The dataset uses a custom loading script, so we specify the name parameter
            full_dataset = load_dataset(
                self.config.dataset_name,
                name=self.config.subset,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
            )

            # The dataset might come with predefined splits or as a single dataset
            if isinstance(full_dataset, dict):
                if self.split in full_dataset:
                    self.data = full_dataset[self.split]
                elif "train" in full_dataset:
                    # Create splits from training data
                    self.data = self._create_splits(full_dataset["train"])
                else:
                    # Use the first available split
                    first_key = list(full_dataset.keys())[0]
                    self.data = self._create_splits(full_dataset[first_key])
            else:
                # Single dataset, create splits
                self.data = self._create_splits(full_dataset)

            # Apply limit if specified
            if self.limit is not None and self.limit < len(self.data):
                self.data = self.data.select(range(self.limit))

            logger.info(f"Loaded {len(self.data)} samples for split '{self.split}'")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _create_splits(self, dataset: HFDataset) -> HFDataset:
        """Create train/val/test splits from a single dataset."""
        # Calculate split sizes
        total = len(dataset)
        train_size = int(total * self.config.train_split)
        val_size = int(total * self.config.val_split)

        # Shuffle and split
        dataset = dataset.shuffle(seed=self.config.seed)

        if self.split == "train":
            return dataset.select(range(train_size))
        elif self.split in ("validation", "val"):
            return dataset.select(range(train_size, train_size + val_size))
        elif self.split == "test":
            return dataset.select(range(train_size + val_size, total))
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train', 'validation', or 'test'")

    def _extract_protein_sequence(self, input_text: str) -> str:
        """
        Extract protein sequence from input text.

        The input may contain the sequence directly or embedded in a longer context.
        We look for patterns that suggest amino acid sequences.
        """
        if not input_text:
            return ""

        # Common amino acid characters
        aa_chars = set("ACDEFGHIKLMNPQRSTVWY")

        # Check if the entire input is a protein sequence
        cleaned = input_text.strip().upper()
        if cleaned and all(c in aa_chars for c in cleaned):
            return cleaned

        # Try to extract sequence from structured input
        # Sometimes sequences are on their own line or after a colon
        lines = input_text.strip().split('\n')
        for line in lines:
            line = line.strip().upper()
            # Skip short lines and lines with too many non-AA characters
            if len(line) >= 10:
                aa_count = sum(1 for c in line if c in aa_chars)
                if aa_count / len(line) > 0.9:
                    return line

        # Return the original input if no clear sequence is found
        return input_text.strip()

    def _format_prompt(
        self,
        instruction: str,
        input_text: str,
        output: str,
        for_inference: bool = False,
    ) -> str:
        """Format instruction-input-output into a prompt string."""
        template = (
            self.config.inference_template if for_inference
            else self.config.prompt_template
        )

        formatted = template.format(
            instruction=instruction.strip(),
            input=input_text.strip(),
            output=output.strip() if output else "",
        )

        return formatted

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Returns:
            Dict containing:
                - protein_sequence: Extracted protein sequence
                - instruction: The task instruction
                - response: The expected output
                - formatted_prompt: Full formatted prompt for training
                - input_text: Original input field
        """
        item = self.data[idx]

        # Extract fields - handle both possible field name conventions
        instruction = item.get("instruction", item.get("Instruction", ""))
        input_text = item.get("input", item.get("Input", ""))
        output = item.get("output", item.get("Output", ""))

        # Extract protein sequence from input
        protein_sequence = self._extract_protein_sequence(input_text)

        # Apply protein length limit if specified
        if self.config.max_protein_length and len(protein_sequence) > self.config.max_protein_length:
            protein_sequence = protein_sequence[:self.config.max_protein_length]

        # Format the full prompt
        formatted_prompt = self._format_prompt(instruction, input_text, output)

        sample = {
            "protein_sequence": protein_sequence,
            "instruction": instruction,
            "response": output,
            "formatted_prompt": formatted_prompt,
            "input_text": input_text,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @classmethod
    def from_config(cls, cfg: Any) -> "MolInstructionsDataset":
        """
        Create dataset from Hydra/OmegaConf configuration.

        Args:
            cfg: Configuration object with dataset parameters

        Returns:
            MolInstructionsDataset instance
        """
        # Handle both dict and OmegaConf objects
        if hasattr(cfg, "to_container"):
            cfg = cfg.to_container(resolve=True)
        elif not isinstance(cfg, dict):
            cfg = dict(cfg)

        # Extract configuration values
        config = MolInstructionsConfig(
            dataset_name=cfg.get("source", cfg.get("dataset_name", "zjunlp/Mol-Instructions")),
            subset=cfg.get("subset", "Protein-oriented Instructions"),
            cache_dir=cfg.get("paths", {}).get("raw", cfg.get("cache_dir")),
            max_seq_length=cfg.get("processing", {}).get("max_seq_length", 2048),
            train_split=cfg.get("splits", {}).get("train", 0.9),
            val_split=cfg.get("splits", {}).get("validation", 0.05),
            test_split=cfg.get("splits", {}).get("test", 0.05),
        )

        split = cfg.get("split", "train")
        limit = cfg.get("limit", None)

        return cls(split=split, config=config, limit=limit)


class MolInstructionsCollator:
    """
    Data collator for Mol-Instructions dataset.

    Handles tokenization and padding for batching instruction-following samples.
    """

    def __init__(
        self,
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        max_length: int = 2048,
        padding: str = "longest",
        truncation: bool = True,
        return_tensors: str = "pt",
        label_pad_token_id: int = -100,
        include_labels: bool = True,
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy ("longest", "max_length", or "do_not_pad")
            truncation: Whether to truncate sequences
            return_tensors: Return type ("pt" for PyTorch)
            label_pad_token_id: Token ID to use for padding labels
            include_labels: Whether to include labels for training
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers is required for tokenization. Install with: pip install transformers"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.label_pad_token_id = label_pad_token_id
        self.include_labels = include_labels

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of samples from MolInstructionsDataset

        Returns:
            Dict containing:
                - input_ids: Tokenized input sequences
                - attention_mask: Attention mask
                - labels: Labels for training (if include_labels=True)
                - protein_sequences: List of protein sequences
        """
        # Extract formatted prompts
        prompts = [item["formatted_prompt"] for item in batch]

        # Tokenize
        encoded = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )

        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

        # Add labels for training (same as input_ids, but with padding masked)
        if self.include_labels:
            labels = encoded["input_ids"].clone()
            # Mask padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id
            result["labels"] = labels

        # Include metadata
        result["protein_sequences"] = [item["protein_sequence"] for item in batch]
        result["instructions"] = [item["instruction"] for item in batch]
        result["responses"] = [item["response"] for item in batch]

        return result


def get_mol_instructions_dataloader(
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    tokenizer: Optional[Any] = None,
    max_length: int = 2048,
    collator: Optional[MolInstructionsCollator] = None,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for Mol-Instructions dataset.

    Args:
        split: Dataset split ("train", "validation", "test")
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        tokenizer: HuggingFace tokenizer (required if collator not provided)
        max_length: Maximum sequence length for tokenization
        collator: Optional custom collator
        **dataset_kwargs: Additional arguments for MolInstructionsDataset

    Returns:
        DataLoader instance
    """
    dataset = MolInstructionsDataset(split=split, **dataset_kwargs)

    # Create collator if tokenizer provided
    if collator is None and tokenizer is not None:
        collator = MolInstructionsCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )

    # Default collate function if no tokenizer
    if collator is None:
        def default_collate(batch):
            return {
                "protein_sequences": [item["protein_sequence"] for item in batch],
                "instructions": [item["instruction"] for item in batch],
                "responses": [item["response"] for item in batch],
                "formatted_prompts": [item["formatted_prompt"] for item in batch],
            }
        collator = default_collate

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Example usage and testing
    import argparse

    parser = argparse.ArgumentParser(description="Test Mol-Instructions Dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=10, help="Limit samples for testing")
    parser.add_argument("--show_sample", action="store_true", help="Show sample data")
    args = parser.parse_args()

    print("Loading Mol-Instructions dataset...")

    try:
        dataset = MolInstructionsDataset(
            split=args.split,
            limit=args.limit,
        )

        print(f"Dataset size: {len(dataset)} samples")

        if args.show_sample and len(dataset) > 0:
            sample = dataset[0]
            print("\nSample 0:")
            print(f"  Instruction: {sample['instruction'][:100]}...")
            print(f"  Protein sequence: {sample['protein_sequence'][:50]}..."
                  if len(sample['protein_sequence']) > 50
                  else f"  Protein sequence: {sample['protein_sequence']}")
            print(f"  Response: {sample['response'][:100]}..."
                  if len(sample['response']) > 100
                  else f"  Response: {sample['response']}")
            print(f"\n  Formatted prompt:\n{sample['formatted_prompt'][:500]}...")

        # Test dataloader without tokenizer
        print("\nTesting DataLoader (no tokenizer)...")
        dataloader = get_mol_instructions_dataloader(
            split=args.split,
            batch_size=2,
            num_workers=0,
            limit=args.limit,
        )

        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch size: {len(batch['protein_sequences'])}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the datasets library installed:")
        print("  pip install datasets")
