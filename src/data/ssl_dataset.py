"""
SSL Dataset for continued pre-training on biology literature.

Loads conversation-format JSON from ProteinLMDataset (PubMed abstracts,
PMC full text, protein-sequence-in-text).  Each record is a plain text
document used for causal language modelling — loss on ALL tokens, no chat
template, no instruction/response split.

Protein sequences appear as ``<seq>M K M R F ...</seq>`` tags and are kept
as plain text tokens (no ESM-3 encoding at the SSL stage).

Data format (per JSON file):
    [{"conversation": [{"system": "", "input": "", "output": "..."}]}, ...]

The ``output`` field is the training text.

Supports:
- Direct JSON loading with source-grouped temperature sampling
- Pre-built Arrow datasets (via ``scripts/prepare_arrow.py``) for fast loading
- Train/val split with deterministic seed
"""

import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

try:
    from datasets import Dataset as HFDataset
    from datasets import load_from_disk

    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


logger = logging.getLogger(__name__)


@dataclass
class SSLDatasetConfig:
    """Configuration for SSL pre-training dataset."""

    # Data directory containing JSON files (or Arrow cache)
    cache_dir: Optional[str] = None

    # Processing
    max_seq_length: int = 2048

    # Split configuration (no test split for SSL — all data is unsupervised)
    train_split: float = 0.95
    val_split: float = 0.05
    seed: int = 42

    # Multi-source balancing (temperature-based upsampling)
    # Groups files by prefix (e.g., "pmc_", "pubmed_", "seq_").
    # alpha < 1.0 upsamples smaller sources relative to larger ones.
    sampling_temperature: float = 1.0

    # Files to exclude
    exclude_files: Optional[List[str]] = None

    # Limit samples (for debugging)
    limit: Optional[int] = None


class SSLDataset(Dataset):
    """
    PyTorch Dataset for SSL continued pre-training.

    Each sample is a plain text document for causal LM training.
    Returns the raw text; tokenization happens in the collator.
    """

    def __init__(
        self,
        split: str = "train",
        config: Optional[SSLDatasetConfig] = None,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 2048,
        limit: Optional[int] = None,
        sampling_temperature: float = 1.0,
        exclude_files: Optional[List[str]] = None,
        seed: int = 42,
    ):
        if not HAS_HF_DATASETS:
            raise ImportError("HuggingFace datasets required: pip install datasets")

        if config is not None:
            self.config = config
        else:
            self.config = SSLDatasetConfig(
                cache_dir=cache_dir,
                max_seq_length=max_seq_length,
                sampling_temperature=sampling_temperature,
                exclude_files=exclude_files,
                seed=seed,
                limit=limit,
            )

        self.split = split
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from Arrow or JSON, then split."""
        if self._try_load_arrow():
            return

        if not self._try_load_json():
            raise FileNotFoundError(
                f"No SSL data found at {self.config.cache_dir}. "
                "Expected JSON files with conversation format or Arrow dataset."
            )

    def _try_load_arrow(self) -> bool:
        """Load pre-built Arrow dataset from ``{cache_dir}_arrow/{split}/``."""
        if self.config.cache_dir is None:
            return False

        arrow_dir = Path(self.config.cache_dir + "_arrow") / self.split
        if not arrow_dir.exists():
            return False

        logger.info(f"Loading SSL data from Arrow: {arrow_dir}")
        self.data = load_from_disk(str(arrow_dir))

        if self.config.limit and self.config.limit < len(self.data):
            self.data = self.data.select(range(self.config.limit))

        logger.info(
            f"Loaded {len(self.data)} samples for split '{self.split}' from Arrow"
        )
        return True

    def _try_load_json(self) -> bool:
        """Load from conversation-format JSON files."""
        import json as _json

        if self.config.cache_dir is None:
            return False

        cache_path = Path(self.config.cache_dir)
        if not cache_path.is_dir():
            return False

        json_files = sorted(cache_path.glob("*.json"))
        if not json_files:
            return False

        exclude = set(self.config.exclude_files or [])

        # Load and group by source prefix for temperature sampling
        source_groups: Dict[str, List[Dict]] = {}
        for json_file in json_files:
            if json_file.name in exclude:
                logger.info(f"  {json_file.name}: EXCLUDED")
                continue

            logger.info(f"  Loading {json_file.name}...")
            with open(json_file) as f:
                records = _json.load(f)

            if not isinstance(records, list):
                logger.info(f"  {json_file.name}: skipped (not a list)")
                continue

            # Extract output text from conversation format
            texts = []
            for record in records:
                text = self._extract_text(record)
                if text:
                    texts.append({"text": text, "__source__": json_file.stem})

            # Group by prefix for temperature sampling
            prefix_match = re.match(r"^([a-z]+)_", json_file.name)
            source = prefix_match.group(1) if prefix_match else "other"
            source_groups.setdefault(source, [])
            source_groups[source].extend(texts)
            logger.info(
                f"  {json_file.name}: {len(texts):,} samples (source: {source})"
            )

        if not source_groups:
            return False

        # Apply temperature-based upsampling
        alpha = self.config.sampling_temperature
        if alpha < 1.0 and len(source_groups) > 1:
            all_records = self._apply_temperature_sampling(source_groups, alpha)
        else:
            all_records = []
            for records in source_groups.values():
                all_records.extend(records)

        logger.info(f"Total SSL samples: {len(all_records):,}")

        # Compute approximate token lengths for Arrow column
        for record in all_records:
            record["__length__"] = len(record["text"]) // 4

        # Convert to HF Dataset and split
        full_dataset = HFDataset.from_list(all_records)
        self.data = self._create_splits(full_dataset)

        if self.config.limit and self.config.limit < len(self.data):
            self.data = self.data.select(range(self.config.limit))

        logger.info(f"Loaded {len(self.data)} samples for split '{self.split}'")
        return True

    @staticmethod
    def _extract_text(record: dict) -> str:
        """Extract plain text from a conversation-format record.

        Expected format: {"conversation": [{"system": "", "input": "", "output": "..."}]}
        Falls back to direct "output" or "text" fields.
        """
        # Conversation format (ProteinLMDataset)
        conversation = record.get("conversation")
        if conversation and isinstance(conversation, list):
            # Concatenate all output turns (usually just one)
            parts = []
            for turn in conversation:
                output = turn.get("output", "")
                if output:
                    parts.append(output)
            if parts:
                return "\n\n".join(parts)

        # Fallback: direct fields
        if "output" in record:
            return record["output"]
        if "text" in record:
            return record["text"]

        return ""

    def _apply_temperature_sampling(
        self, source_groups: Dict[str, List], alpha: float
    ) -> List:
        """Upsample smaller sources using temperature-based sampling."""
        sizes = {src: len(recs) for src, recs in source_groups.items()}

        raw_weights = {src: n**alpha for src, n in sizes.items()}
        total_weight = sum(raw_weights.values())
        target_props = {src: w / total_weight for src, w in raw_weights.items()}

        max_source = max(sizes, key=lambda s: sizes[s])
        baseline_ratio = target_props[max_source] / sizes[max_source]

        all_records = []
        rng = random.Random(self.config.seed)
        for src, records in source_groups.items():
            factor = (target_props[src] / sizes[src]) / baseline_ratio
            full_repeats = int(factor)
            fractional = factor - full_repeats

            repeated = records * full_repeats
            if fractional > 0:
                extra_count = int(math.ceil(fractional * len(records)))
                extra = rng.sample(records, min(extra_count, len(records)))
                repeated.extend(extra)

            logger.info(
                f"  Source '{src}': {sizes[src]:,} -> {len(repeated):,} "
                f"(x{factor:.1f}, target {target_props[src]:.1%})"
            )
            all_records.extend(repeated)

        return all_records

    def _create_splits(self, dataset: HFDataset) -> HFDataset:
        """Create train/val splits from a single dataset."""
        total = len(dataset)
        train_size = int(total * self.config.train_split)

        dataset = dataset.shuffle(seed=self.config.seed)

        if self.split == "train":
            return dataset.select(range(train_size))
        elif self.split in ("validation", "val"):
            return dataset.select(range(train_size, total))
        else:
            raise ValueError(
                f"Unknown split: {self.split}. Use 'train' or 'validation'."
            )

    @property
    def lengths(self) -> List[int]:
        """Approximate token lengths for LengthGroupedSampler."""
        if not hasattr(self, "_lengths"):
            if "__length__" in self.data.column_names:
                self._lengths = self.data.data.column("__length__").to_pylist()
            else:
                self._lengths = [
                    len(self.data[i]["text"]) // 4 for i in range(len(self.data))
                ]
        return self._lengths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample with plain text for causal LM.

        Returns:
            Dict with:
                - text: Plain text document (causal LM target)
                - source: Source file identifier
        """
        item = self.data[idx]
        return {
            "text": item["text"],
            "source": item.get("__source__", "unknown"),
        }

    @classmethod
    def from_config(cls, cfg: Any) -> "SSLDataset":
        """Create dataset from Hydra/OmegaConf configuration.

        Args:
            cfg: Configuration object with dataset parameters.

        Returns:
            SSLDataset instance.
        """
        if hasattr(cfg, "to_container"):
            cfg = cfg.to_container(resolve=True)
        elif not isinstance(cfg, dict):
            cfg = dict(cfg)

        config = SSLDatasetConfig(
            cache_dir=cfg.get("paths", {}).get("raw", cfg.get("cache_dir")),
            max_seq_length=cfg.get("processing", {}).get("max_seq_length", 2048),
            train_split=cfg.get("splits", {}).get("train", 0.95),
            val_split=cfg.get("splits", {}).get("validation", 0.05),
            seed=cfg.get("seed", 42),
            sampling_temperature=cfg.get("sampling_temperature", 1.0),
            exclude_files=cfg.get("exclude_files", None),
            limit=cfg.get("limit", None),
        )

        split = cfg.get("split", "train")
        return cls(split=split, config=config)


class SSLDataCollator:
    """Data collator for SSL causal LM training.

    Tokenizes plain text, creates input_ids/attention_mask/labels.
    Loss is computed on ALL tokens (no prompt masking).
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        max_length: int = 2048,
        padding: str = "longest",
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers required: pip install transformers")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of SSL samples.

        Tokenizes text and creates labels with loss on all non-padding tokens.

        Returns:
            Dict with input_ids, attention_mask, labels.
        """
        texts = [item["text"] for item in batch]

        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        # Labels = input_ids, with padding masked to -100
        labels = encoded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
