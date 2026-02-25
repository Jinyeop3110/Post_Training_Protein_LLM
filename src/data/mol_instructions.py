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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


logger = logging.getLogger(__name__)


# Default system prompt for protein expert identity
DEFAULT_SYSTEM_PROMPT = (
    "You are a protein science expert. Given a protein amino acid sequence, "
    "you analyze its properties, predict its function, structure, and "
    "biological associations based on your knowledge of protein biology."
)

# Default prompt template for formatting instructions (Alpaca-style fallback)
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


def format_chat_messages(
    instruction: str,
    input_text: str,
    output: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    for_inference: bool = False,
) -> list:
    """Build chat messages list for tokenizer.apply_chat_template().

    Returns a list of message dicts suitable for apply_chat_template().
    This ensures training and inference use the same prompt format.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = f"{instruction.strip()}\n\n{input_text.strip()}"
    messages.append({"role": "user", "content": user_content})

    if not for_inference and output:
        messages.append({"role": "assistant", "content": output.strip()})

    return messages


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
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_chat_template: bool = True  # Use tokenizer.apply_chat_template() instead of Alpaca format
    # Qwen3-specific: False disables thinking mode (empty <think></think>),
    # True lets the model decide whether to reason.  Ignored for non-Qwen models.
    enable_thinking: bool = False

    # Filtering
    min_instruction_length: int = 10
    min_response_length: int = 1
    # Files to skip when loading from local JSON. Design tasks have text
    # descriptions as input (not protein sequences), which breaks ESM-3 encoding.
    exclude_files: Optional[List[str]] = None

    # Multi-source balancing (temperature-based upsampling)
    # Files are grouped by prefix (e.g., "mol_", "sp_", "wp_").
    # sampling_temperature < 1.0 upsamples smaller sources:
    #   weight_i ∝ n_i^α  (α = sampling_temperature)
    # 1.0 = no rebalancing, 0.5 = moderate, 0.0 = fully equalized
    sampling_temperature: float = 1.0

    # Protein placeholder for ESM-3 approach.  When set (non-empty), the
    # protein sequence in ``input_text`` is replaced with this token so the
    # model receives protein information only as learned embeddings, not raw
    # amino-acid text.  Leave empty for text-only approach.
    protein_placeholder: str = ""


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
        sampling_temperature: float = 1.0,
        exclude_files: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        protein_placeholder: str = "",
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
            sampling_temperature: Temperature for multi-source balancing (< 1.0 upsamples smaller sources)
            exclude_files: List of filenames to skip when loading local JSON files.
            tokenizer: HuggingFace tokenizer for apply_chat_template() formatting.
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
                sampling_temperature=sampling_temperature,
                exclude_files=exclude_files,
                protein_placeholder=protein_placeholder,
            )

        self.split = split
        self.transform = transform
        self.limit = limit
        self.tokenizer = tokenizer

        # Load and prepare dataset
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from local JSON files or HuggingFace.

        First attempts to load from local JSON files (which avoids compatibility
        issues with HuggingFace dataset loading scripts). Falls back to HuggingFace
        ``load_dataset`` if local files are not found.
        """
        logger.info(f"Loading Mol-Instructions dataset: {self.config.subset}")

        # Try loading from local JSON files first
        loaded = self._try_load_local_json()
        if loaded:
            self._filter_long_proteins()
            return

        # Fall back to HuggingFace load_dataset
        try:
            full_dataset = load_dataset(
                self.config.dataset_name,
                name=self.config.subset,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
            )

            if isinstance(full_dataset, dict):
                if self.split in full_dataset:
                    self.data = full_dataset[self.split]
                elif "train" in full_dataset:
                    self.data = self._create_splits(full_dataset["train"])
                else:
                    first_key = list(full_dataset.keys())[0]
                    self.data = self._create_splits(full_dataset[first_key])
            else:
                self.data = self._create_splits(full_dataset)

            if self.limit is not None and self.limit < len(self.data):
                self.data = self.data.select(range(self.limit))

            self._filter_long_proteins()

            logger.info(f"Loaded {len(self.data)} samples for split '{self.split}'")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _filter_long_proteins(self) -> None:
        """Drop samples whose protein sequence exceeds max_protein_length.

        Filtering at load time (instead of truncating in __getitem__) ensures
        __len__ is accurate and avoids training on incomplete sequences whose
        labels were written for the full protein.
        """
        max_len = self.config.max_protein_length
        if not max_len:
            return

        aa_chars = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
        before = len(self.data)

        def _seq_len(item):
            text = item.get("input", item.get("Input", ""))
            if not text:
                return 0
            cleaned = text.strip().upper()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
            if cleaned and all(c in aa_chars for c in cleaned):
                return len(cleaned)
            # Fall back to longest AA-like line
            for line in text.split("\n"):
                line = line.strip().upper()
                if len(line) >= 4:
                    aa_count = sum(1 for c in line if c in aa_chars)
                    if aa_count / len(line) > 0.9:
                        return len(line)
            return len(text)

        keep_idx = [i for i in range(before) if _seq_len(self.data[i]) <= max_len]

        if len(keep_idx) < before:
            self.data = self.data.select(keep_idx)
            dropped = before - len(keep_idx)
            logger.info(
                f"Filtered {dropped} samples with protein > {max_len} AA "
                f"({dropped/before*100:.1f}% dropped, {len(self.data)} remaining)"
            )

    def _try_load_local_json(self) -> bool:
        """Attempt to load dataset from local JSON files.

        Searches for JSON files under the cache directory following the
        Mol-Instructions layout (``data/Protein-oriented_Instructions/*.json``).

        Returns:
            True if data was successfully loaded, False otherwise.
        """
        import json as _json

        if self.config.cache_dir is None:
            return False

        # Search for the Protein-oriented Instructions JSON directory
        cache_path = Path(self.config.cache_dir)
        subset_dir_name = self.config.subset.replace(" ", "_").replace("-", "_")
        candidates = [
            cache_path / "data" / subset_dir_name,
            cache_path / subset_dir_name,
            cache_path / "data" / "Protein-oriented_Instructions",
            cache_path,
        ]

        json_dir = None
        for candidate in candidates:
            if candidate.is_dir():
                json_files = list(candidate.glob("*.json"))
                if json_files:
                    json_dir = candidate
                    break

        if json_dir is None:
            return False

        logger.info(f"Loading from local JSON files: {json_dir}")

        # Load all JSON files, grouped by source prefix for balancing
        import re

        exclude = set(self.config.exclude_files or [])

        source_groups: Dict[str, List] = {}
        for json_file in sorted(json_dir.glob("*.json")):
            if json_file.name in exclude:
                logger.info(f"  {json_file.name}: EXCLUDED by config")
                continue
            with open(json_file) as f:
                records = _json.load(f)
            if not isinstance(records, list):
                logger.info(f"  {json_file.name}: skipped (not a record list)")
                continue
            # Derive source group from filename prefix (e.g., "mol_", "sp_", "wp_")
            prefix_match = re.match(r"^([a-z]+)_", json_file.name)
            source = prefix_match.group(1) if prefix_match else "other"
            source_groups.setdefault(source, [])
            source_groups[source].extend(records)
            logger.info(f"  {json_file.name}: {len(records)} samples (source: {source})")

        if not source_groups:
            return False

        # Apply temperature-based upsampling if temperature < 1.0
        alpha = self.config.sampling_temperature
        all_records = []
        if alpha < 1.0 and len(source_groups) > 1:
            all_records = self._apply_temperature_sampling(source_groups, alpha)
        else:
            for records in source_groups.values():
                all_records.extend(records)

        logger.info(f"Total samples loaded: {len(all_records)}")

        # Convert to HuggingFace Dataset for consistent split handling
        from datasets import Dataset as HFDatasetCls

        full_dataset = HFDatasetCls.from_list(all_records)

        # Create splits
        self.data = self._create_splits(full_dataset)

        # Apply limit
        if self.limit is not None and self.limit < len(self.data):
            self.data = self.data.select(range(self.limit))

        logger.info(f"Loaded {len(self.data)} samples for split '{self.split}'")
        return True

    def _apply_temperature_sampling(
        self, source_groups: Dict[str, List], alpha: float
    ) -> List:
        """Upsample smaller source groups using temperature-based sampling.

        Computes target proportion for source *i* as  p_i ∝ n_i^α.
        Smaller sources get repeated more times to match the target proportion.
        The largest source is kept at 1x (no duplication).

        Args:
            source_groups: Mapping of source prefix → list of records.
            alpha: Sampling temperature (0 < α < 1). Lower = more balanced.

        Returns:
            Combined list of records after upsampling.
        """
        import math
        import random

        sizes = {src: len(recs) for src, recs in source_groups.items()}
        max_size = max(sizes.values())

        # Compute target proportions: p_i ∝ n_i^α
        raw_weights = {src: n ** alpha for src, n in sizes.items()}
        total_weight = sum(raw_weights.values())
        target_props = {src: w / total_weight for src, w in raw_weights.items()}

        # Compute upsample factor relative to the largest source (which stays at 1x)
        # For each source: target_n / actual_n, normalized so largest source = 1x
        max_source = max(sizes, key=lambda s: sizes[s])
        baseline_ratio = target_props[max_source] / sizes[max_source]

        all_records = []
        rng = random.Random(self.config.seed)
        for src, records in source_groups.items():
            factor = (target_props[src] / sizes[src]) / baseline_ratio
            full_repeats = int(factor)
            fractional = factor - full_repeats

            repeated = records * full_repeats
            # Fractional part: randomly sample that fraction of records
            if fractional > 0:
                extra_count = int(math.ceil(fractional * len(records)))
                extra = rng.sample(records, min(extra_count, len(records)))
                repeated.extend(extra)

            logger.info(
                f"  Source '{src}': {sizes[src]:,} → {len(repeated):,} "
                f"(×{factor:.1f}, target {target_props[src]:.1%})"
            )
            all_records.extend(repeated)

        return all_records

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

        Handles:
        - Pure AA sequences (standard 20 + ambiguous XBZUO codes)
        - Sequences wrapped in triple backticks (```...```)
        - Short sequences (< 10 AA)
        """
        if not input_text:
            return ""

        # Standard 20 + IUPAC ambiguous codes (X=unknown, B=D/N, Z=E/Q, U=Sec, O=Pyl)
        aa_chars = set("ACDEFGHIKLMNPQRSTVWYBXZUO")

        # Strip markdown code fences that wrap sequences in many datasets
        text = input_text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()

        # Check if the entire input is a protein sequence
        cleaned = text.strip().upper()
        if cleaned and all(c in aa_chars for c in cleaned):
            return cleaned

        # Try to extract sequence from structured input
        # Sometimes sequences are on their own line or after a colon
        lines = text.split('\n')
        for line in lines:
            line = line.strip().upper()
            if not line or line.startswith("```"):
                continue
            # Accept any line that's mostly AA characters (>= 4 chars to catch short seqs)
            if len(line) >= 4:
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
        """Format instruction-input-output into a prompt string.

        When use_chat_template is True and a tokenizer is available, uses
        tokenizer.apply_chat_template() to match the model's native format.
        Falls back to Alpaca-style template otherwise.
        """
        if self.config.use_chat_template and self.tokenizer is not None:
            messages = format_chat_messages(
                instruction=instruction,
                input_text=input_text,
                output=output,
                system_prompt=self.config.system_prompt,
                for_inference=for_inference,
            )
            # apply_chat_template() is model-agnostic: each tokenizer carries
            # its own Jinja2 template, so Qwen3 produces <|im_start|>/<|im_end|>
            # blocks while Llama produces <|begin_of_text|> style tokens, etc.
            #
            # enable_thinking=False is Qwen3-specific — it forces empty
            # <think></think> blocks so the model skips reasoning and outputs
            # the response directly.  Non-Qwen tokenizers don't accept this
            # kwarg, so we catch TypeError and fall back to the plain call.
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=for_inference,
                    enable_thinking=self.config.enable_thinking,
                )
            except TypeError:
                # Non-Qwen models (Llama, Mistral, etc.) — no enable_thinking
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=for_inference,
                )
            return formatted

        # Fallback: Alpaca-style template
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

    @property
    def lengths(self) -> List[int]:
        """Approximate token lengths for HF Trainer's LengthGroupedSampler.

        Uses character count / 4 as a rough proxy for token count (avoids
        expensive tokenization at dataset init). Good enough for grouping
        similar-length sequences to reduce padding waste.
        """
        if not hasattr(self, "_lengths"):
            self._lengths = []
            for i in range(len(self.data)):
                item = self.data[i]
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")
                # ~4 chars per token is a reasonable approximation
                approx_tokens = (len(instruction) + len(input_text) + len(output)) // 4
                self._lengths.append(approx_tokens)
        return self._lengths

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

        # For ESM-3 approach: replace protein text with placeholder token
        # so the model receives protein info only as learned embeddings.
        prompt_input = input_text
        if self.config.protein_placeholder and protein_sequence:
            prompt_input = self.config.protein_placeholder

        # Format the full prompt (training: includes response)
        formatted_prompt = self._format_prompt(instruction, prompt_input, output)
        # Format inference prompt (no response, for RL generation)
        inference_prompt = self._format_prompt(
            instruction, prompt_input, "", for_inference=True
        )

        sample = {
            "protein_sequence": protein_sequence,
            "instruction": instruction,
            "response": output,
            "formatted_prompt": formatted_prompt,
            "inference_prompt": inference_prompt,
            "input_text": input_text,
            "metadata": item.get("metadata", {}),
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
        result["metadata"] = [item.get("metadata", {}) for item in batch]

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
