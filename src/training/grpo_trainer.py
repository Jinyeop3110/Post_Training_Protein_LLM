"""
GRPO (Group Relative Policy Optimization) Trainer Implementation

This module provides GRPO training with verifiable rewards for protein tasks.
GRPO generates multiple completions per prompt and uses verifiable rewards
(e.g., GO term correctness, stability prediction accuracy) instead of a
separate reward model.

Key features:
- Verifiable rewards for protein tasks (GO terms, PPI, stability)
- Group-based advantage computation (no need for critic/value model)
- Support for DAPO (no KL penalty) and Dr. GRPO (no advantage normalization)
- Integration with TRL's GRPO trainer or custom implementation

Reference: https://arxiv.org/abs/2402.03300 (DeepSeekMath GRPO)
"""

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from trl import GRPOConfig
    from trl import GRPOTrainer as TRLGRPOTrainer
    HAS_TRL_GRPO = True
except ImportError:
    HAS_TRL_GRPO = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# FSDP2 composable API (PyTorch 2.4+)
try:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import DeviceMesh
    HAS_FSDP2 = True
except ImportError:
    HAS_FSDP2 = False


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports for backward compatibility.
# External code can still do:
#   from src.training.grpo_trainer import compute_go_reward, ...
# ---------------------------------------------------------------------------
from .rewards import (  # noqa: F401, E402
    compute_esmfold_reward,
    compute_generic_reward,
    compute_go_reward,
    compute_ppi_reward,
    compute_stability_reward,
    get_reward_function,
)

# =============================================================================
# GRPO Prompt Constants
# =============================================================================

# GRPO-specific system prompt: adds <answer> tag instruction to base prompt.
# This teaches the model to wrap final answers in <answer>...</answer> tags,
# enabling clean reward extraction (content inside tags) + format reward.
_GRPO_ANSWER_SUFFIX = " Always wrap your final answer in <answer> and </answer> tags."

# Task-specific system prompts for GRPO.
# Two variants per task: thinking-enabled and thinking-disabled.
#
# Thinking-enabled: model uses <think>...</think> to reason, then <answer>...</answer>.
#   Allows the model to process ESM-3 embeddings and reason before answering.
#   Requires more max_tokens but produces more diverse/accurate predictions.
#
# Thinking-disabled: empty <think></think> prefix, direct answer.
#   Faster but model can't reason, tends to copy examples verbatim.

# --- Structure Quality (ESMFold) ---
_ESMFOLD_SYSTEM_PROMPT_THINK = (
    "You are a protein structure expert. You have already analyzed the protein's "
    "3D structure using computational methods. Based on your analysis, report the "
    "structural quality assessment.\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags about the "
    "protein's structural features, then give your final answer inside "
    "<answer>...</answer> tags. Keep your thinking SHORT.\n\n"
    "Examples:\n"
    "<think>Strong hydrophobic core, regular secondary structure. High confidence ~88.</think>\n"
    "<answer>Fold quality: high. pLDDT: 88.4. This protein is well-folded with high confidence.</answer>\n\n"
    "<think>Mixed ordered/disordered regions. N-terminal structured, C-terminal flexible.</think>\n"
    "<answer>Fold quality: medium. pLDDT: 62.1. This protein is moderate confidence, partially structured.</answer>\n\n"
    "<think>Rich in charged/polar residues, lacks hydrophobic core. Likely disordered.</think>\n"
    "<answer>Fold quality: low. pLDDT: 35.7. This protein is likely disordered with low confidence.</answer>"
)

_ESMFOLD_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein structure expert. You have already analyzed the protein's "
    "3D structure using computational methods. Based on your analysis, report the "
    "structural quality assessment.\n\n"
    "First write a brief reasoning about the protein's structural features on a line "
    "starting with \"Reasoning:\", then give your final assessment inside "
    "<answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: Strong hydrophobic core, regular secondary structure. High confidence ~88.\n"
    "<answer>Fold quality: high. pLDDT: 88.4. This protein is well-folded with high confidence.</answer>\n\n"
    "Reasoning: Mixed ordered/disordered regions. N-terminal structured, C-terminal flexible.\n"
    "<answer>Fold quality: medium. pLDDT: 62.1. This protein is moderate confidence, partially structured.</answer>\n\n"
    "Reasoning: Rich in charged/polar residues, lacks hydrophobic core. Likely disordered.\n"
    "<answer>Fold quality: low. pLDDT: 35.7. This protein is likely disordered with low confidence.</answer>"
)

# --- GO Prediction ---
_GO_SYSTEM_PROMPT_THINK = (
    "You are a protein function expert. Given a protein amino acid sequence, "
    "predict its Gene Ontology (GO) terms covering molecular function (MF), "
    "biological process (BP), and cellular component (CC).\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags about "
    "the protein's likely function, then list the GO terms inside "
    "<answer>...</answer> tags. Keep your thinking SHORT.\n\n"
    "Examples:\n"
    "<think>Contains zinc-finger domain, likely DNA-binding transcription factor.</think>\n"
    "<answer>GO:0003700, GO:0005634, GO:0006355</answer>\n\n"
    "<think>Signal peptide + lectin fold suggests secreted carbohydrate binding.</think>\n"
    "<answer>GO:0030246, GO:0005576, GO:0005488</answer>\n\n"
    "<think>Kinase domain with ATP binding site, cytoplasmic localization.</think>\n"
    "<answer>GO:0004672, GO:0005524, GO:0005737, GO:0006468</answer>"
)

_GO_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein function expert. Given a protein amino acid sequence, "
    "predict its Gene Ontology (GO) terms covering molecular function (MF), "
    "biological process (BP), and cellular component (CC).\n\n"
    "First write a brief reasoning about the protein's features on a line "
    "starting with \"Reasoning:\", then list the GO terms inside "
    "<answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: Contains zinc-finger domain, likely DNA-binding transcription factor.\n"
    "<answer>GO:0003700, GO:0005634, GO:0006355</answer>\n\n"
    "Reasoning: Signal peptide + lectin fold suggests secreted carbohydrate binding.\n"
    "<answer>GO:0030246, GO:0005576, GO:0005488</answer>\n\n"
    "Reasoning: Kinase domain with ATP binding site, cytoplasmic localization.\n"
    "<answer>GO:0004672, GO:0005524, GO:0005737, GO:0006468</answer>"
)

# --- Stability Prediction ---
_STABILITY_SYSTEM_PROMPT_THINK = (
    "You are a protein stability expert. You are given a wild-type protein "
    "sequence, a mutant sequence, and the mutation notation. Compare the two "
    "sequences and predict the change in thermodynamic stability "
    "(ddG in kcal/mol). Classify the effect as stabilizing, neutral, or "
    "destabilizing.\n\n"
    "- Stabilizing: ddG < -1.0 kcal/mol\n"
    "- Neutral: -1.0 <= ddG <= 1.0 kcal/mol\n"
    "- Destabilizing: ddG > 1.0 kcal/mol\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags, then "
    "give your prediction inside <answer>...</answer> tags. Keep thinking SHORT.\n\n"
    "Examples:\n"
    "<think>Replacing a core hydrophobic with polar residue disrupts packing.</think>\n"
    "<answer>ddG = 2.3 kcal/mol. This mutation is destabilizing.</answer>\n\n"
    "<think>Conservative substitution at a solvent-exposed position.</think>\n"
    "<answer>ddG = -0.2 kcal/mol. This mutation is neutral.</answer>\n\n"
    "<think>Introducing a disulfide-compatible cysteine, improves core packing.</think>\n"
    "<answer>ddG = -1.8 kcal/mol. This mutation is stabilizing.</answer>"
)

_STABILITY_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein stability expert. You are given a wild-type protein "
    "sequence, a mutant sequence, and the mutation notation. Compare the two "
    "sequences and predict the change in thermodynamic stability "
    "(ddG in kcal/mol). Classify the effect as stabilizing, neutral, or "
    "destabilizing.\n\n"
    "- Stabilizing: ddG < -1.0 kcal/mol\n"
    "- Neutral: -1.0 <= ddG <= 1.0 kcal/mol\n"
    "- Destabilizing: ddG > 1.0 kcal/mol\n\n"
    "First write a brief reasoning on a line starting with \"Reasoning:\", "
    "then give your prediction inside <answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: Replacing a core hydrophobic with polar residue disrupts packing.\n"
    "<answer>ddG = 2.3 kcal/mol. This mutation is destabilizing.</answer>\n\n"
    "Reasoning: Conservative substitution at a solvent-exposed position.\n"
    "<answer>ddG = -0.2 kcal/mol. This mutation is neutral.</answer>\n\n"
    "Reasoning: Introducing a disulfide-compatible cysteine, improves core packing.\n"
    "<answer>ddG = -1.8 kcal/mol. This mutation is stabilizing.</answer>"
)

# --- SS Composition (Option A) ---
_SS_COMPOSITION_SYSTEM_PROMPT_THINK = (
    "You are a protein structure expert. Given a protein sequence, predict its "
    "secondary structure composition and solvent accessibility.\n\n"
    "Report the percentage of helix, sheet, and coil residues, plus the mean "
    "relative solvent accessibility (RSA, 0-1 scale).\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags, then "
    "give your prediction inside <answer>...</answer> tags. Keep thinking SHORT.\n\n"
    "Examples:\n"
    "<think>Leucine zipper motif, mostly alpha-helical with exposed charged residues.</think>\n"
    "<answer>45.2% helix, 5.1% sheet, 49.7% coil. Mean RSA: 0.42.</answer>\n\n"
    "<think>Immunoglobulin fold with beta sandwich architecture, compact core.</think>\n"
    "<answer>8.3% helix, 48.7% sheet, 43.0% coil. Mean RSA: 0.31.</answer>\n\n"
    "<think>Intrinsically disordered region, no stable secondary structure.</think>\n"
    "<answer>2.0% helix, 1.5% sheet, 96.5% coil. Mean RSA: 0.68.</answer>"
)

_SS_COMPOSITION_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein structure expert. Given a protein sequence, predict its "
    "secondary structure composition and solvent accessibility.\n\n"
    "Report the percentage of helix, sheet, and coil residues, plus the mean "
    "relative solvent accessibility (RSA, 0-1 scale).\n\n"
    "First write brief reasoning on a line starting with \"Reasoning:\", "
    "then give your prediction inside <answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: Leucine zipper motif, mostly alpha-helical with exposed charged residues.\n"
    "<answer>45.2% helix, 5.1% sheet, 49.7% coil. Mean RSA: 0.42.</answer>\n\n"
    "Reasoning: Immunoglobulin fold with beta sandwich architecture, compact core.\n"
    "<answer>8.3% helix, 48.7% sheet, 43.0% coil. Mean RSA: 0.31.</answer>\n\n"
    "Reasoning: Intrinsically disordered region, no stable secondary structure.\n"
    "<answer>2.0% helix, 1.5% sheet, 96.5% coil. Mean RSA: 0.68.</answer>"
)

# --- SS Per-Residue (Option B) ---
_SS_SEQUENCE_SYSTEM_PROMPT_THINK = (
    "You are a protein structure expert. Given a protein sequence, predict the "
    "per-residue secondary structure using H (helix), E (sheet), C (coil).\n\n"
    "The SS3 string must match the input sequence length exactly. Also report "
    "the overall composition percentages.\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags, then "
    "give your prediction inside <answer>...</answer> tags. Keep thinking SHORT.\n\n"
    "Examples:\n"
    "<think>N-terminal helix, central beta hairpin, C-terminal coil.</think>\n"
    "<answer>SS3: CHHHHHHHEEEEEECCCEEEEECCCC\n"
    "Composition: 29.6% helix, 37.0% sheet, 33.3% coil.</answer>\n\n"
    "<think>All-alpha bundle, short loops between helices.</think>\n"
    "<answer>SS3: CCHHHHHHHHCCHHHHHHHHHCC\n"
    "Composition: 78.3% helix, 0.0% sheet, 21.7% coil.</answer>"
)

_SS_SEQUENCE_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein structure expert. Given a protein sequence, predict the "
    "per-residue secondary structure using H (helix), E (sheet), C (coil).\n\n"
    "The SS3 string must match the input sequence length exactly. Also report "
    "the overall composition percentages.\n\n"
    "First write brief reasoning on a line starting with \"Reasoning:\", "
    "then give your prediction inside <answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: N-terminal helix, central beta hairpin, C-terminal coil.\n"
    "<answer>SS3: CHHHHHHHEEEEEECCCEEEEECCCC\n"
    "Composition: 29.6% helix, 37.0% sheet, 33.3% coil.</answer>\n\n"
    "Reasoning: All-alpha bundle, short loops between helices.\n"
    "<answer>SS3: CCHHHHHHHHCCHHHHHHHHHCC\n"
    "Composition: 78.3% helix, 0.0% sheet, 21.7% coil.</answer>"
)

# --- Structure Composite (Option C) ---
_STRUCTURE_COMPOSITE_SYSTEM_PROMPT_THINK = (
    "You are a protein structure expert. Given a protein sequence, provide a "
    "comprehensive structural analysis including secondary structure, backbone "
    "geometry, solvent accessibility, and long-range contacts.\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags, then "
    "give your analysis inside <answer>...</answer> tags. Keep thinking SHORT.\n\n"
    "Example:\n"
    "<think>TIM barrel fold — alternating alpha/beta with buried core.</think>\n"
    "<answer>SS3: CEEEECCHHHHHHCCEEEECCHHHHHHCCEEEECCHHHHHH\n"
    "Secondary structure: 42.9% helix, 28.6% sheet, 28.6% coil.\n"
    "Ramachandran: 40.5% alpha, 26.2% beta, 0.0% left-helix, 33.3% other.\n"
    "Mean RSA: 0.35. Buried: 62.0%.\n"
    "Long-range contacts: 58 (density 0.0085).\n"
    "pLDDT: 82.3.</answer>"
)

_STRUCTURE_COMPOSITE_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein structure expert. Given a protein sequence, provide a "
    "comprehensive structural analysis including secondary structure, backbone "
    "geometry, solvent accessibility, and long-range contacts.\n\n"
    "First write brief reasoning on a line starting with \"Reasoning:\", "
    "then give your analysis inside <answer>...</answer> tags.\n\n"
    "Example:\n"
    "Reasoning: TIM barrel fold — alternating alpha/beta with buried core.\n"
    "<answer>SS3: CEEEECCHHHHHHCCEEEECCHHHHHHCCEEEECCHHHHHH\n"
    "Secondary structure: 42.9% helix, 28.6% sheet, 28.6% coil.\n"
    "Ramachandran: 40.5% alpha, 26.2% beta, 0.0% left-helix, 33.3% other.\n"
    "Mean RSA: 0.35. Buried: 62.0%.\n"
    "Long-range contacts: 58 (density 0.0085).\n"
    "pLDDT: 82.3.</answer>"
)

# ── Solubility system prompts ──

_SOLUBILITY_SYSTEM_PROMPT_THINK = (
    "You are a protein solubility expert. Given a protein amino acid sequence, "
    "predict its solubility when expressed in E. coli.\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags about "
    "the protein's solubility-related features, then give your prediction inside "
    "<answer>...</answer> tags. Keep your thinking SHORT.\n\n"
    "Examples:\n"
    "<think>Small protein, no transmembrane domains, balanced charge. Likely soluble.</think>\n"
    "<answer>Solubility: soluble. Estimated solubility: 78.5%.</answer>\n\n"
    "<think>Large hydrophobic patches, many cysteines, membrane-associated. Likely insoluble.</think>\n"
    "<answer>Solubility: insoluble. Estimated solubility: 12.3%.</answer>"
)

_SOLUBILITY_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein solubility expert. Given a protein amino acid sequence, "
    "predict its solubility when expressed in E. coli.\n\n"
    "First write a brief reasoning about the protein's features on a line "
    "starting with \"Reasoning:\", then give your prediction inside "
    "<answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: Small protein, no transmembrane domains, balanced charge. Likely soluble.\n"
    "<answer>Solubility: soluble. Estimated solubility: 78.5%.</answer>\n\n"
    "Reasoning: Large hydrophobic patches, many cysteines, membrane-associated. Likely insoluble.\n"
    "<answer>Solubility: insoluble. Estimated solubility: 12.3%.</answer>"
)

# ── Fold Classification (CATH) system prompts ──

_FOLD_CLASSIFICATION_SYSTEM_PROMPT_THINK = (
    "You are a protein structure classification expert. Given a protein amino "
    "acid sequence, predict its CATH structural classification at four levels: "
    "Class (C), Architecture (A), Topology (T), and Homology (H).\n\n"
    "CATH classes: 1=Mainly Alpha, 2=Mainly Beta, 3=Alpha Beta, "
    "4=Few Secondary Structures.\n\n"
    "Think BRIEFLY (1-2 sentences only) inside <think>...</think> tags about "
    "the protein's structural features, then give the CATH code inside "
    "<answer>...</answer> tags. Keep your thinking SHORT.\n\n"
    "Examples:\n"
    "<think>Helical bundle pattern, globin-like fold. Class 1, orthogonal bundle.</think>\n"
    "<answer>CATH: 1.10.490.10. Class: Mainly Alpha. Architecture: Orthogonal Bundle. "
    "Topology: Globin-like. Homology: Globin.</answer>\n\n"
    "<think>Beta sandwich with immunoglobulin topology. Class 2.</think>\n"
    "<answer>CATH: 2.60.40.10. Class: Mainly Beta. Architecture: Sandwich. "
    "Topology: Immunoglobulin-like. Homology: Immunoglobulin.</answer>"
)

_FOLD_CLASSIFICATION_SYSTEM_PROMPT_NO_THINK = (
    "You are a protein structure classification expert. Given a protein amino "
    "acid sequence, predict its CATH structural classification at four levels: "
    "Class (C), Architecture (A), Topology (T), and Homology (H).\n\n"
    "CATH classes: 1=Mainly Alpha, 2=Mainly Beta, 3=Alpha Beta, "
    "4=Few Secondary Structures.\n\n"
    "First write a brief reasoning about the protein's structural features on a line "
    "starting with \"Reasoning:\", then give the CATH code inside "
    "<answer>...</answer> tags.\n\n"
    "Examples:\n"
    "Reasoning: Helical bundle pattern, globin-like fold. Class 1, orthogonal bundle.\n"
    "<answer>CATH: 1.10.490.10. Class: Mainly Alpha. Architecture: Orthogonal Bundle. "
    "Topology: Globin-like. Homology: Globin.</answer>\n\n"
    "Reasoning: Beta sandwich with immunoglobulin topology. Class 2.\n"
    "<answer>CATH: 2.60.40.10. Class: Mainly Beta. Architecture: Sandwich. "
    "Topology: Immunoglobulin-like. Homology: Immunoglobulin.</answer>"
)

# Empty thinking prefix appended after the generation prompt.
# With enable_thinking=False, the model generates directly after this,
# producing: <think>\n\n</think>\n\n<answer>...</answer>
_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def _get_grpo_system_prompt(task: str, enable_thinking: bool = True) -> str:
    """Return the task-specific GRPO system prompt.

    Args:
        task: Task name (esmfold, go_prediction, stability, etc.)
        enable_thinking: If True, use think+answer format. If False, answer-only.
    """
    task = task.lower()
    if task in ("esmfold", "structure", "structure_prediction", "fold_quality"):
        return _ESMFOLD_SYSTEM_PROMPT_THINK if enable_thinking else _ESMFOLD_SYSTEM_PROMPT_NO_THINK
    if task in ("go_prediction", "go_terms", "go", "function", "function_prediction"):
        return _GO_SYSTEM_PROMPT_THINK if enable_thinking else _GO_SYSTEM_PROMPT_NO_THINK
    if task in ("stability", "stability_prediction", "ddg"):
        return _STABILITY_SYSTEM_PROMPT_THINK if enable_thinking else _STABILITY_SYSTEM_PROMPT_NO_THINK
    if task in ("ss_composition", "structure_properties_a"):
        return _SS_COMPOSITION_SYSTEM_PROMPT_THINK if enable_thinking else _SS_COMPOSITION_SYSTEM_PROMPT_NO_THINK
    if task in ("ss_sequence", "ss_per_residue", "structure_properties_b"):
        return _SS_SEQUENCE_SYSTEM_PROMPT_THINK if enable_thinking else _SS_SEQUENCE_SYSTEM_PROMPT_NO_THINK
    if task in ("structure_composite", "structure_properties", "structure_properties_c"):
        return _STRUCTURE_COMPOSITE_SYSTEM_PROMPT_THINK if enable_thinking else _STRUCTURE_COMPOSITE_SYSTEM_PROMPT_NO_THINK
    if task in ("solubility", "solubility_prediction"):
        return _SOLUBILITY_SYSTEM_PROMPT_THINK if enable_thinking else _SOLUBILITY_SYSTEM_PROMPT_NO_THINK
    if task in ("fold_classification", "fold_class", "cath", "cath_classification"):
        return _FOLD_CLASSIFICATION_SYSTEM_PROMPT_THINK if enable_thinking else _FOLD_CLASSIFICATION_SYSTEM_PROMPT_NO_THINK
    # Fallback: generic
    from src.data.mol_instructions import DEFAULT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT + _GRPO_ANSWER_SUFFIX


# =============================================================================
# Configuration Functions
# =============================================================================


def get_grpo_config(cfg: DictConfig) -> Dict[str, Any]:
    """Get GRPO configuration from Hydra config.

    Extracts GRPO-specific settings from the configuration including
    group size, temperature, KL penalty, and advantage normalization settings.

    Args:
        cfg: Hydra configuration containing training.grpo settings.

    Returns:
        Dictionary with GRPO configuration parameters:
            - group_size: Number of completions per prompt
            - temperature: Sampling temperature for generation
            - use_kl_penalty: Whether to use KL divergence penalty (False for DAPO)
            - normalize_advantages: Whether to normalize advantages (False for Dr. GRPO)
            - max_new_tokens: Maximum tokens to generate per completion
            - top_p: Top-p sampling parameter
    """
    grpo_cfg = cfg.training.get("grpo", {})
    rollout_cfg = cfg.training.get("rollout", {})

    return {
        "group_size": grpo_cfg.get("group_size", 4),
        "temperature": grpo_cfg.get("temperature", 1.0),
        "use_kl_penalty": grpo_cfg.get("use_kl_penalty", False),
        "normalize_advantages": grpo_cfg.get("normalize_advantages", False),
        "kl_coef": grpo_cfg.get("kl_coef", 0.1),
        "clip_range": grpo_cfg.get("clip_range", 0.2),
        # Rollout settings
        "max_new_tokens": rollout_cfg.get("max_tokens", 512),
        "top_p": rollout_cfg.get("top_p", 0.95),
        "do_sample": rollout_cfg.get("do_sample", True),
        # Thinking mode: if True, model generates <think>...</think> freely;
        # if False, we prepend empty <think>\n\n</think>\n\n prefix.
        "enable_thinking": rollout_cfg.get("enable_thinking", False),
    }


# =============================================================================
# GRPO Trainer Class
# =============================================================================


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) trainer for protein LLMs.

    This trainer implements GRPO with verifiable rewards, which is particularly
    suited for protein tasks where rewards can be computed directly from
    predictions (e.g., GO term correctness, stability accuracy).

    Key features:
    - Generates multiple completions per prompt (group_size)
    - Computes verifiable rewards without a separate reward model
    - Uses group-relative advantages for policy updates
    - Supports DAPO (no KL penalty) and Dr. GRPO (no advantage normalization)

    Attributes:
        cfg: Hydra configuration object
        model: The policy model (LLM with optional LoRA adapters)
        ref_model: Reference model for KL penalty (if used)
        tokenizer: HuggingFace tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        grpo_config: GRPO-specific configuration
        reward_fn: Reward function for the task

    Example:
        >>> trainer = GRPOTrainer(cfg)
        >>> trainer.setup()
        >>> trainer.train()
        >>> trainer.save_checkpoint("./checkpoints/grpo_final")
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the GRPO trainer.

        Args:
            cfg: Hydra configuration containing model, training, and data settings.
        """
        self.cfg = cfg
        self.model = None
        self.ref_model = None
        self.protein_llm = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.optimizer = None
        self.scheduler = None
        self.grpo_config = None
        self.reward_fn = None
        self.device = None
        self.global_step = 0
        self.epoch = 0

        # Distributed training attributes
        self.local_rank = 0
        self.world_size = 1
        self.is_main_process = True
        self.use_fsdp = False
        self._grad_ckpt_available = False  # toggled gradient checkpointing
        self._probe_indices: List[int] = []  # fixed probe prompts for wandb logging

        # Validate dependencies
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers is required. Install with: pip install transformers"
            )

    def setup(self) -> None:
        """Set up model, tokenizer, dataset, and GRPO configuration.

        This method must be called before train(). It:
        1. Initializes distributed training (NCCL process group)
        2. Initializes logging (wandb if enabled, rank 0 only)
        3. Loads tokenizer and model (with ProteinLLM for esm3)
        4. Applies FSDP2 sharding for multi-GPU
        5. Creates reference model for KL penalty
        6. Loads datasets
        7. Configures optimizer (differential LR) and scheduler
        8. Sets up reward function
        """
        log.info("Setting up GRPO trainer...")

        # Initialize distributed training
        self._init_distributed()

        # Initialize logging (main process only)
        if self.is_main_process:
            self._setup_logging()

        # Load GRPO config
        self.grpo_config = get_grpo_config(self.cfg)
        if self.is_main_process:
            log.info(f"GRPO config: {self.grpo_config}")

        # Load tokenizer
        self._load_tokenizer()

        # Load model (+ ProteinLLM for embedding approaches)
        self._load_model()

        # Apply FSDP2 for multi-GPU training
        fsdp_enabled = self.cfg.training.get("fsdp", {}).get("enabled", True)
        if self.world_size > 1 and fsdp_enabled:
            self._apply_fsdp()
        elif self.world_size > 1:
            log.info("FSDP disabled — using DDP-style gradient sync")
            # Enable gradient checkpointing (saves ~40% memory during training fwd)
            # It's only active when model.training=True (HF checks both flags).
            # Generation switches to eval mode → grad ckpt off → KV cache works.
            if self.cfg.training.get("gradient_checkpointing", False):
                self._ensure_grad_ckpt_on()
                log.info("Gradient checkpointing enabled on LLM")

        # Create reference model for KL penalty (if enabled)
        if self.grpo_config["use_kl_penalty"]:
            self._create_reference_model()

        # Load ESM embedding cache (if configured)
        embedding_cache_path = self.cfg.get("encoder", {}).get(
            "embedding_cache_path", None
        )
        self._embedding_cache = None
        if embedding_cache_path is not None:
            try:
                from src.data.esm_embedding_cache import ESMEmbeddingCache
                self._embedding_cache = ESMEmbeddingCache(
                    embedding_cache_path, readonly=True
                )
                if self.is_main_process:
                    log.info(
                        f"ESM embedding cache loaded: {embedding_cache_path} "
                        f"({len(self._embedding_cache)} entries)"
                    )
            except Exception as e:
                log.warning(f"Failed to load ESM embedding cache: {e}")

        # Load datasets
        self._load_datasets()

        # Set up reward function (must be before probe selection, which
        # uses _is_esmfold_reward / _is_stability_reward for stratification)
        self._setup_reward_function()

        # Select fixed probe prompts for wandb completion logging.
        # Classification tasks: 8 per class = 24 total (ESMFold: 3 classes, stability: 3 classes).
        probe_count = 8 if (self._is_esmfold_reward or self._is_stability_reward or self._is_solubility_reward) else 5
        self._select_probe_prompts(num_probes=probe_count)

        # Optionally freeze multimodal head (pooling + projector) so only
        # LoRA adapters are optimized.  Eliminates joint gradient clipping
        # where 38M multimodal params with higher LR dominate the gradient
        # norm and starve LoRA of learning signal.
        if self.cfg.training.get("freeze_multimodal", False):
            self._freeze_multimodal()

        # Set up optimizer and scheduler (with differential LR)
        self._setup_optimizer()

        log.info("GRPO trainer setup complete")

    def _init_distributed(self) -> None:
        """Initialize distributed training (NCCL process group, device)."""
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.is_main_process = self.local_rank == 0

        if self.is_main_process:
            log.info(
                f"Distributed: world_size={self.world_size}, device={self.device}"
            )

    def _apply_fsdp(self) -> None:
        """Apply FSDP2 sharding to the LLM for multi-GPU training.

        Sharding plan:
        - ESM-3 encoder: excluded (replicated, frozen, fp32, ~1.2 GB)
        - AttentionPooling + MLPProjector: excluded (replicated, manual grad sync)
        - Qwen3-4B LLM: FSDP2 with reshard_after_forward=False (SHARD_GRAD_OP)
        """
        if not HAS_FSDP2:
            log.warning(
                "FSDP2 not available (requires PyTorch 2.4+). "
                "Multi-GPU gradient sync only applies to pooling/projector. "
                "LLM gradients will NOT be synchronized across GPUs."
            )
            return

        log.info("Applying FSDP2 to LLM...")

        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        # Get the base model (unwrap PeftModel if present)
        llm = self.protein_llm.llm if self.protein_llm is not None else self.model
        base_model = (
            llm.get_base_model() if hasattr(llm, "get_base_model") else llm
        )

        # Ensure all LLM parameters are uniform bfloat16 before FSDP2.
        # resize_token_embeddings() or LoRA adapter loading can introduce
        # float32 params, which causes FSDP2 AssertionError on lazy_init.
        dtypes = {p.dtype for p in base_model.parameters()}
        if len(dtypes) > 1:
            log.info(
                f"Mixed dtypes in LLM before FSDP: {dtypes}. "
                f"Casting all to bfloat16."
            )
            base_model.to(torch.bfloat16)

        # Cache embed_tokens weights BEFORE FSDP shards the model.
        # After sharding, calling embed_tokens directly on sharded params
        # produces garbage. ProteinLLM.prepare_inputs() uses this cache.
        if self.protein_llm is not None:
            embed_weight = base_model.get_input_embeddings().weight.data.clone()
            self.protein_llm._fsdp_embed_cache = embed_weight
            log.info(
                f"Cached embed_tokens for FSDP: {embed_weight.shape} "
                f"({embed_weight.nbytes / 1024**2:.1f} MB)"
            )

        # Wrap each decoder layer individually.
        # Use reshard_after_forward=True (FULL_SHARD) for GRPO because
        # generation + ESM-3 encoding + ESMFold rewards all compete for
        # GPU memory. FULL_SHARD reshards params after each layer's
        # forward pass, trading speed for ~14 GB less memory per GPU.
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            for layer in base_model.model.layers:
                fully_shard(
                    layer,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=True,
                )
            # Wrap root model
            fully_shard(
                base_model, mesh=mesh, mp_policy=mp_policy,
                reshard_after_forward=True,
            )
            self.use_fsdp = True
            log.info(
                f"FSDP2 applied to {len(base_model.model.layers)} decoder layers"
            )
        else:
            log.warning("Could not find decoder layers for FSDP2 wrapping")

        # Enable gradient checkpointing for the differentiable training forward.
        # This disables KV cache during generation (making it O(T²) instead of
        # O(T)), but without it the training forward OOMs. Generation speed is
        # the bottleneck — reduce group_size to compensate.
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            log.info("Gradient checkpointing enabled on LLM")

    def _setup_logging(self) -> None:
        """Set up wandb and other logging.

        Uses the training-specific wandb project (protein-llm-rl for GRPO/DPO)
        and includes tags for method, model, dataset, lr, and epochs.
        """
        logging_cfg = self.cfg.get("logging", {})

        if logging_cfg.get("wandb", {}).get("enabled", False) and HAS_WANDB:
            if wandb.run is not None:
                log.info("Wandb already initialized, skipping re-initialization")
                return

            # Get project from training config, fall back to logging config
            project = self.cfg.training.get("wandb", {}).get(
                "project",
                logging_cfg.wandb.get("project", "protein-llm-rl"),
            )

            # Build tags: method, model, dataset, lr, epochs
            tags = list(self.cfg.training.get("wandb", {}).get("tags", []))
            method = self.cfg.training.get("method", "grpo")
            model_name = self.cfg.model.get("name", "unknown")
            task_type = self.cfg.data.get("task", "go_prediction")
            tags.extend([
                f"method:{method}",
                f"model:{model_name}",
                f"task:{task_type}",
                f"lr:{self.cfg.training.get('lr', 'unknown')}",
                f"epochs:{self.cfg.training.get('epochs', 'unknown')}",
                f"group_size:{self.cfg.training.get('grpo', {}).get('group_size', 4)}",
            ])

            wandb.init(
                project=project,
                name=logging_cfg.wandb.get("name", f"grpo_{self.cfg.get('experiment_name', 'run')}"),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=tags,
            )
            log.info(f"Wandb logging initialized for GRPO: project={project}, tags={tags}")

    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        from src.models.multimodal_llm import PROTEIN_SPECIAL_TOKENS

        model_path = self.cfg.model.path
        log.info(f"Loading tokenizer from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add protein special tokens for ESM-3 approach:
        # <|protein_start|>, <|protein_embed|>, <|protein_end|>
        approach = self.cfg.get("approach", "text")
        if approach in ("esm3",):
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": PROTEIN_SPECIAL_TOKENS}
            )
            if num_added > 0:
                log.info(
                    f"Added {num_added} protein special tokens: {PROTEIN_SPECIAL_TOKENS} "
                    f"(vocab size: {len(self.tokenizer)})"
                )

        log.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

    def _load_model(self) -> None:
        """Load the model with optional quantization, LoRA, and ProteinLLM.

        Supports two modes:
        1. **From SFT checkpoint** (parent_checkpoint set): Loads the full
           ProteinLLM (encoder + pooling + projector + LoRA adapter) from a
           trained SFT checkpoint via ProteinLLM.from_pretrained().
        2. **Fresh base model** (no parent_checkpoint): Loads base LLM,
           applies fresh LoRA, and builds new ProteinLLM components.

        For the esm3 approach, creates a full ProteinLLM with encoder,
        pooling, and projector, reusing the already-loaded LoRA model
        (same pattern as sft_trainer._load_protein_llm).
        """
        parent_checkpoint = self.cfg.get("parent_checkpoint", None)

        # Auto-resolve parent checkpoint from parent_experiment
        if not parent_checkpoint:
            parent_experiment = self.cfg.get("parent_experiment", None)
            if parent_experiment:
                from src.utils.experiment import resolve_parent_checkpoint
                results_dir = Path(self.cfg.paths.results_dir)
                resolved = resolve_parent_checkpoint(results_dir, parent_experiment)
                if resolved:
                    parent_checkpoint = str(resolved)
                    log.info(
                        f"Resolved parent_experiment '{parent_experiment}' "
                        f"-> checkpoint: {parent_checkpoint}"
                    )
                else:
                    log.warning(
                        f"Could not resolve checkpoint for parent_experiment "
                        f"'{parent_experiment}'. Loading fresh base model."
                    )

        if parent_checkpoint:
            self._load_model_from_checkpoint(parent_checkpoint)
        else:
            self._load_model_fresh()

        self.model.train()

    def _load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from an SFT checkpoint (LoRA adapter + multimodal components).

        Supports two checkpoint formats:
        1. **ProteinLLM format** (has config.json, no adapter_config.json at root):
           Uses ProteinLLM.from_pretrained().
        2. **HF Trainer format** (has adapter_config.json at root, optional
           pooling.pt/projector.pt): Loads base LLM, applies LoRA adapter,
           and loads multimodal weights if present.

        Args:
            checkpoint_path: Path to the saved checkpoint directory.
        """
        checkpoint_dir = Path(checkpoint_path)
        approach = self.cfg.get("approach", "text")
        log.info(f"Loading model from SFT checkpoint: {checkpoint_path}")

        # Detect format: ProteinLLM (config.json without adapter_config.json)
        # vs HF Trainer (adapter_config.json at root)
        is_protein_llm_format = (
            (checkpoint_dir / "config.json").exists()
            and not (checkpoint_dir / "adapter_config.json").exists()
        )

        if is_protein_llm_format:
            self._load_from_protein_llm_checkpoint(checkpoint_dir, approach)
        else:
            self._load_from_hf_trainer_checkpoint(checkpoint_dir, approach)

    def _load_from_protein_llm_checkpoint(
        self, checkpoint_dir: Path, approach: str
    ) -> None:
        """Load from ProteinLLM.save_pretrained() format."""
        from src.models.multimodal_llm import EMBEDDING_APPROACHES, ProteinLLM

        if approach in EMBEDDING_APPROACHES:
            self.protein_llm = ProteinLLM.from_pretrained(
                checkpoint_dir,
                device=str(self.device),
                load_llm=True,
                load_encoder=True,
            )
            self.model = self.protein_llm.llm
            self.tokenizer = self.protein_llm.tokenizer

            if len(self.tokenizer) != self.model.config.vocab_size:
                base = (
                    self.model.get_base_model()
                    if hasattr(self.model, "get_base_model")
                    else self.model
                )
                base.resize_token_embeddings(len(self.tokenizer))

            log.info("ProteinLLM loaded from checkpoint")
            if self.is_main_process:
                self.protein_llm.print_trainable_parameters()
        else:
            # Text-only: adapter is in adapter/ subdir
            self._load_base_llm_with_adapter(checkpoint_dir / "adapter")

    def _load_from_hf_trainer_checkpoint(
        self, checkpoint_dir: Path, approach: str
    ) -> None:
        """Load from HF Trainer intermediate checkpoint format.

        HF Trainer saves adapter_config.json + adapter_model.safetensors at
        the checkpoint root, alongside optional pooling.pt and projector.pt.
        """
        from src.models.multimodal_llm import EMBEDDING_APPROACHES

        # Load base LLM + LoRA adapter (adapter_config.json at root)
        self._load_base_llm_with_adapter(checkpoint_dir)

        # For esm3 approach: build ProteinLLM and load pooling/projector weights
        if approach in EMBEDDING_APPROACHES:
            self._load_protein_llm()

            # Load trained pooling weights from checkpoint
            pooling_path = checkpoint_dir / "pooling.pt"
            if pooling_path.exists() and self.protein_llm.pooling is not None:
                self.protein_llm.pooling.load_state_dict(
                    torch.load(
                        pooling_path, map_location=self.device, weights_only=True
                    )
                )
                log.info(f"Loaded pooling weights from: {pooling_path}")

            # Load trained projector weights from checkpoint
            projector_path = checkpoint_dir / "projector.pt"
            if projector_path.exists() and self.protein_llm.projector is not None:
                self.protein_llm.projector.load_state_dict(
                    torch.load(
                        projector_path, map_location=self.device, weights_only=True
                    )
                )
                log.info(f"Loaded projector weights from: {projector_path}")

            if self.is_main_process:
                self.protein_llm.print_trainable_parameters()

    def _load_base_llm_with_adapter(self, adapter_path: Path) -> None:
        """Load base LLM and apply LoRA adapter from a checkpoint path.

        Args:
            adapter_path: Directory containing adapter_config.json.
        """
        model_path = self.cfg.model.path

        # Load base LLM (no device_map for FSDP2 compat)
        if self.world_size > 1 and HAS_FSDP2:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
            self.model = self.model.to(self.device)
        else:
            device_map = (
                {"": self.local_rank} if torch.cuda.is_available() else "auto"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Resize embeddings if protein special tokens were added
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            log.info(
                f"Resized model embeddings: "
                f"{self.model.config.vocab_size} -> {len(self.tokenizer)}"
            )

        # Apply LoRA adapter from checkpoint
        if (adapter_path / "adapter_config.json").exists() and HAS_PEFT:
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), is_trainable=True,
            )
            log.info(f"Loaded SFT LoRA adapter from: {adapter_path}")
        else:
            log.warning(
                f"No adapter found at {adapter_path}. "
                f"Applying fresh LoRA instead."
            )
            from .config_utils import get_qlora_config

            lora_config = get_qlora_config(self.cfg)
            self.model = get_peft_model(self.model, lora_config)

        if self.is_main_process:
            self.model.print_trainable_parameters()

    def _load_model_fresh(self) -> None:
        """Load fresh base model with new LoRA adapters (no parent checkpoint)."""
        from .config_utils import get_qlora_config, get_quantization_config

        model_path = self.cfg.model.path
        use_quantization = self.cfg.training.get("quantization", {}).get("enabled", False)

        log.info(f"Loading fresh model from: {model_path}")
        log.info(f"Using quantization: {use_quantization}")

        # Get quantization config
        quantization_config = get_quantization_config(self.cfg) if use_quantization else None

        if quantization_config is not None:
            # Quantized: use device_map (not compatible with FSDP2)
            device_map = {"": self.local_rank}
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            if HAS_PEFT:
                self.model = prepare_model_for_kbit_training(self.model)
        else:
            # Non-quantized: avoid accelerate device_map for FSDP2 compat
            if self.world_size > 1 and HAS_FSDP2:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
                self.model = self.model.to(self.device)
            else:
                device_map = (
                    {"": self.local_rank}
                    if torch.cuda.is_available()
                    else "auto"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

        # Resize embeddings if protein special tokens were added to tokenizer
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            log.info(
                f"Resized model embeddings: "
                f"{self.model.config.vocab_size} -> {len(self.tokenizer)}"
            )

        # Apply LoRA if configured and enabled (default: true)
        lora_enabled = self.cfg.training.get("lora", {}).get("enabled", True)
        if lora_enabled and self.cfg.training.get("lora", {}) and HAS_PEFT:
            lora_config = get_qlora_config(self.cfg)
            self.model = get_peft_model(self.model, lora_config)
            log.info("LoRA configuration applied")
            if self.is_main_process:
                self.model.print_trainable_parameters()

        # Load ProteinLLM for embedding approach (esm3)
        approach = self.cfg.get("approach", "text")
        if approach == "esm3":
            self._load_protein_llm()

    def _load_protein_llm(self) -> None:
        """Load ProteinLLM for multimodal GRPO training.

        Creates ProteinLLM with encoder, pooling, and projector but reuses
        the already-loaded LoRA model instead of loading a second LLM copy.
        Follows the exact pattern from sft_trainer._load_protein_llm().
        """
        try:
            from src.models.multimodal_llm import EMBEDDING_APPROACHES, ProteinLLM

            approach = self.cfg.get("approach", "esm3")
            if approach not in EMBEDDING_APPROACHES:
                log.info(
                    f"Approach '{approach}' doesn't need multimodal "
                    f"components, skipping ProteinLLM setup"
                )
                return

            log.info("Loading ProteinLLM for GRPO multimodal training...")

            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            encoder_cfg = cfg_dict.get("encoder", {})
            model_cfg = cfg_dict.get("model", {})
            training_cfg = cfg_dict.get("training", {})
            pooling_cfg = encoder_cfg.get("pooling", {})
            projector_cfg = encoder_cfg.get("projector", {})
            lora_cfg = training_cfg.get("lora", {})
            use_qlora = training_cfg.get("quantization", {}).get("enabled", False)

            self.protein_llm = ProteinLLM(
                approach=approach,
                llm_name=model_cfg.get("path", "Qwen/Qwen3-4B"),
                encoder_name=encoder_cfg.get("model_name", "esm3-sm-open-v1"),
                encoder_embed_dim=encoder_cfg.get("embedding_dim"),
                num_prefix_tokens=pooling_cfg.get("num_output_tokens", 32),
                pooling_type=pooling_cfg.get("method", "attention"),
                projector_type=projector_cfg.get("type", "mlp"),
                projector_hidden_dim=projector_cfg.get("hidden_dim", 2048),
                projector_num_layers=projector_cfg.get("num_layers", 2),
                projector_dropout=projector_cfg.get("dropout", 0.1),
                perceiver_layers=projector_cfg.get("perceiver_layers", 2),
                perceiver_heads=projector_cfg.get("perceiver_heads", 8),
                perceiver_ffn_dim=projector_cfg.get("perceiver_ffn_dim", 2048),
                perceiver_latent_dim=projector_cfg.get("perceiver_latent_dim", None),
                use_qlora=use_qlora,
                lora_r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                lora_target_modules=lora_cfg.get(
                    "target_modules", [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ]
                ),
                load_llm=False,
                load_encoder=True,
                device=str(self.device),
                encoder_dtype=encoder_cfg.get("dtype", "bfloat16"),
                encoder_batch_size=encoder_cfg.get("encoder_batch_size", 4),
            )

            # Assign already-loaded LoRA model and tokenizer
            self.protein_llm.llm = self.model
            self.protein_llm.tokenizer = self.tokenizer
            self.protein_llm.llm_hidden_size = self.model.config.hidden_size
            self.protein_llm._build_projector()

            log.info("ProteinLLM loaded for GRPO")
            if self.is_main_process:
                self.protein_llm.print_trainable_parameters()

        except ImportError as e:
            log.warning(f"Could not load ProteinLLM: {e}. Using text-only training.")
            self.protein_llm = None

    def _freeze_multimodal(self) -> None:
        """Freeze pooling and projector so only LoRA adapters are optimized.

        Sets requires_grad=False on all pooling and projector parameters.
        This eliminates the joint gradient clipping problem where multimodal
        params with a higher LR dominate the gradient norm and starve LoRA.
        The multimodal head is already well-trained from SFT.
        """
        if self.protein_llm is None:
            log.info("No ProteinLLM — nothing to freeze")
            return

        frozen_count = 0
        for module_name in ("pooling", "projector"):
            module = getattr(self.protein_llm, module_name, None)
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen_count += p.numel()

        log.info(
            f"Frozen multimodal head: {frozen_count:,} params "
            f"(pooling + projector) set to requires_grad=False"
        )

    def _create_reference_model(self) -> None:
        """Create a frozen reference model for KL penalty computation."""
        log.info("Creating reference model for KL penalty...")

        device_map = {"": self.local_rank} if torch.cuda.is_available() else "auto"
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.path,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.ref_model.eval()
        log.info("Reference model created and frozen")

    def _load_datasets(self) -> None:
        """Load training and validation datasets.

        Supports both Mol-Instructions (from paths.raw) and downstream task
        datasets (from paths.processed).  The downstream JSON files use the
        same instruction/input/output/metadata format so MolInstructionsDataset
        can load them via its ``_try_load_local_json`` fallback.
        """
        from src.data.mol_instructions import MolInstructionsDataset
        from src.models.multimodal_llm import PROTEIN_PLACEHOLDER

        data_cfg = self.cfg.data

        # Downstream tasks store processed JSON at paths.processed;
        # Mol-Instructions keeps HF download at paths.raw.
        cache_dir = (
            data_cfg.get("paths", {}).get("processed")
            or data_cfg.get("paths", {}).get("raw")
        )

        # For ESM-3 approach, replace protein text with placeholder token
        approach = self.cfg.get("approach", "text")
        placeholder = PROTEIN_PLACEHOLDER if approach in ("esm3",) else ""

        max_protein_length = data_cfg.get("processing", {}).get(
            "max_protein_length", None
        )
        common_kwargs = dict(
            dataset_name=data_cfg.get("source", "zjunlp/Mol-Instructions"),
            subset=data_cfg.get("subset", "Protein-oriented Instructions"),
            cache_dir=cache_dir,
            max_seq_length=self.cfg.training.get("max_seq_length", 2048),
            max_protein_length=max_protein_length,
            tokenizer=self.tokenizer,
            protein_placeholder=placeholder,
            limit=data_cfg.get("limit"),
        )

        log.info("Loading training dataset...")
        self.train_dataset = MolInstructionsDataset(split="train", **common_kwargs)
        log.info(f"Training dataset loaded: {len(self.train_dataset)} samples")

        log.info("Loading validation dataset...")
        self.eval_dataset = MolInstructionsDataset(split="validation", **common_kwargs)
        log.info(f"Validation dataset loaded: {len(self.eval_dataset)} samples")

    def _select_probe_prompts(self, num_probes: int = 5) -> None:
        """Select fixed probe prompts from training data for wandb logging.

        For ESMFold or stability tasks, selects ``num_probes`` samples per
        class so the probe table covers all categories. For other tasks,
        picks evenly-spaced indices.

        Stores indices in ``self._probe_indices`` for use during eval.
        """
        if self.train_dataset is None or len(self.train_dataset) == 0:
            return

        n = len(self.train_dataset)

        # Stratified selection for classification tasks
        if self._is_esmfold_reward or self._is_stability_reward:
            per_class = num_probes
            buckets: Dict[str, List[int]] = {}
            for idx in range(n):
                sample = self.train_dataset[idx]
                metadata = sample.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                if self._is_esmfold_reward:
                    plddt = metadata.get("plddt")
                    if plddt is None:
                        continue
                    plddt = float(plddt)
                    cat = "high" if plddt > 80 else "medium" if plddt > 50 else "low"
                else:
                    cat = metadata.get("stability_class")
                    if cat is None:
                        continue
                buckets.setdefault(cat, []).append(idx)

            import random as _rng
            gen = _rng.Random(42)  # deterministic
            selected = []
            for cat in sorted(buckets.keys()):
                pool = buckets[cat]
                k = min(per_class, len(pool))
                if k > 0:
                    selected.extend(gen.sample(pool, k))

            self._probe_indices = selected if selected else [
                int(i * n / num_probes) for i in range(min(num_probes, n))
            ]
        else:
            num_probes = min(num_probes, n)
            self._probe_indices = [
                int(i * n / num_probes) for i in range(num_probes)
            ]

        if self.is_main_process:
            log.info(
                f"Selected {len(self._probe_indices)} probe prompts at indices: "
                f"{self._probe_indices}"
            )

    def _log_probe_completions(self) -> None:
        """Generate completions for probe prompts and log to wandb as a Table.

        Called at eval_steps intervals. Generates 2 completions per probe
        prompt with GRPO system prompt and logs them with rewards.
        """
        from .rewards import extract_answer_content

        if not HAS_WANDB or wandb.run is None:
            return
        if not self._probe_indices or not self.is_main_process:
            return

        task = self.cfg.data.get("task", "go_prediction").lower()
        enable_thinking = self.grpo_config.get("enable_thinking", False)
        grpo_system_prompt = _get_grpo_system_prompt(task, enable_thinking=enable_thinking)

        num_completions = 2
        rows = []

        self._ensure_grad_ckpt_off()
        self.model.eval()

        with torch.no_grad():
            for idx in self._probe_indices:
                sample = self.train_dataset[idx]
                # Use raw instruction + input_text (same as _training_step).
                # For multimodal with protein, exclude raw protein text from
                # prompt — protein info delivered via ESM-3 embeddings.
                instruction = sample.get("instruction", "")
                input_text = sample.get("input_text", "")
                protein_seq = sample.get("protein_sequence", None)
                has_protein = (
                    protein_seq
                    and self.protein_llm is not None
                )
                if instruction:
                    raw_prompt = instruction.strip()
                    if input_text and input_text.strip() and not has_protein:
                        raw_prompt += f"\n\n{input_text.strip()}"
                else:
                    raw_prompt = sample.get(
                        "inference_prompt",
                        sample.get("formatted_prompt", ""),
                    )

                # Ground truth for reward (mirrors _training_step)
                task = self.cfg.data.get("task", "go_prediction").lower()
                metadata = sample.get("metadata", {})
                if task in ("stability", "ddg", "stability_prediction") and isinstance(metadata, dict) and "ddG" in metadata:
                    ground_truth = json.dumps({
                        "ddG": metadata.get("ddG", 0),
                        "stability_class": metadata.get("stability_class"),
                        "mutation": metadata.get("mutation"),
                    })
                elif (
                    task in ("esmfold", "structure", "structure_prediction", "fold_quality")
                    and isinstance(metadata, dict) and "plddt" in metadata
                ):
                    ground_truth = json.dumps({
                        "plddt": metadata.get("plddt", 0),
                        "ptm": metadata.get("ptm", 0),
                    })
                elif (
                    task in ("solubility", "solubility_prediction")
                    and isinstance(metadata, dict) and "solubility_score" in metadata
                ):
                    ground_truth = json.dumps({
                        "solubility_score": metadata.get("solubility_score", 0),
                        "solubility_class": metadata.get("solubility_class"),
                    })
                elif (
                    task in ("fold_classification", "fold_class", "cath", "cath_classification")
                    and isinstance(metadata, dict) and "cath_code" in metadata
                ):
                    ground_truth = json.dumps({
                        "cath_code": metadata.get("cath_code"),
                        "class_name": metadata.get("class_name"),
                        "architecture_name": metadata.get("architecture_name"),
                        "topology_name": metadata.get("topology_name"),
                        "homology_name": metadata.get("homology_name"),
                    })
                else:
                    ground_truth = sample.get("response", sample.get("output", ""))

                completions = []
                rewards = []

                for _ in range(num_completions):
                    if self.protein_llm is not None and protein_seq:
                        texts = self.protein_llm.generate(
                            protein_sequences=[protein_seq],
                            prompt=[raw_prompt],
                            max_new_tokens=256,
                            temperature=self.grpo_config["temperature"],
                            top_p=self.grpo_config["top_p"],
                            do_sample=True,
                            use_cache=True,
                            system_prompt=grpo_system_prompt,
                            generation_prefix=None if enable_thinking else _THINKING_PREFIX,
                        )
                        completion = texts[0]
                    else:
                        messages = [
                            {"role": "system", "content": grpo_system_prompt},
                            {"role": "user", "content": raw_prompt},
                        ]
                        wrapped = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        )
                        if not enable_thinking:
                            wrapped += _THINKING_PREFIX
                        inputs = self.tokenizer(
                            wrapped,
                            return_tensors="pt",
                            truncation=True,
                            max_length=self.cfg.training.get("max_seq_length", 2048) - 256,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            temperature=self.grpo_config["temperature"],
                            top_p=self.grpo_config["top_p"],
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        completion = self.tokenizer.decode(
                            outputs[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        )

                    # Extract answer and compute reward
                    answer_text, has_tags = extract_answer_content(completion)
                    extra_kw = {}
                    if self._is_esmfold_reward:
                        gt_str = str(ground_truth)
                        if gt_str.strip().startswith("{"):
                            second_arg = ground_truth
                        elif protein_seq:
                            second_arg = protein_seq
                        else:
                            second_arg = ground_truth
                        if self._focal_gamma > 0:
                            extra_kw["focal_gamma"] = self._focal_gamma
                        if self._binary_alignment:
                            extra_kw["binary_alignment"] = True
                        if self._classification_only:
                            extra_kw["classification_only"] = True
                    elif self._is_stability_reward:
                        second_arg = ground_truth
                        if self._focal_gamma > 0:
                            extra_kw["focal_gamma"] = self._focal_gamma
                    elif self._is_solubility_reward:
                        second_arg = ground_truth
                        if self._focal_gamma > 0:
                            extra_kw["focal_gamma"] = self._focal_gamma
                    else:
                        second_arg = ground_truth
                    reward = self.reward_fn(answer_text, second_arg, **extra_kw)

                    completions.append(completion[:500])  # truncate for table
                    rewards.append(round(reward, 4))

                # Build ground truth display string
                gt_display = str(ground_truth)[:200]
                true_category = ""
                if self._is_esmfold_reward:
                    try:
                        gt_parsed = json.loads(str(ground_truth))
                        plddt_val = float(gt_parsed.get("plddt", 0))
                        true_category = (
                            "high" if plddt_val > 80
                            else "medium" if plddt_val > 50
                            else "low"
                        )
                        gt_display = f"pLDDT={plddt_val:.1f} ({true_category})"
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                elif self._is_stability_reward:
                    try:
                        gt_parsed = json.loads(str(ground_truth))
                        ddg_val = float(gt_parsed.get("ddG", 0))
                        true_category = gt_parsed.get("stability_class", "")
                        mutation = gt_parsed.get("mutation", "")
                        gt_display = f"ddG={ddg_val:.2f} ({true_category}) {mutation}"
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                elif self._is_solubility_reward:
                    try:
                        gt_parsed = json.loads(str(ground_truth))
                        sol_score = float(gt_parsed.get("solubility_score", 0))
                        true_category = gt_parsed.get("solubility_class", "")
                        gt_display = f"solubility={sol_score:.1f}% ({true_category})"
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                elif self._is_fold_reward:
                    try:
                        gt_parsed = json.loads(str(ground_truth))
                        cath_code = gt_parsed.get("cath_code", "")
                        class_name = gt_parsed.get("class_name", "")
                        true_category = class_name
                        gt_display = f"CATH={cath_code} ({class_name})"
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass

                row = {
                    "step": self.global_step,
                    "prompt": raw_prompt[:200],
                    "ground_truth": gt_display,
                    "true_category": true_category if true_category else "",
                    "protein_seq": (protein_seq[:50] + "...") if protein_seq and len(protein_seq) > 50 else (protein_seq or ""),
                }
                for ci in range(num_completions):
                    row[f"completion_{ci+1}"] = completions[ci] if ci < len(completions) else ""
                    row[f"reward_{ci+1}"] = rewards[ci] if ci < len(rewards) else 0.0
                rows.append(row)

        self.model.train()
        self._ensure_grad_ckpt_on()

        # Log as wandb Table
        if HAS_WANDB and wandb.run is not None:
            columns = list(rows[0].keys())
            table = wandb.Table(columns=columns)
            for row in rows:
                table.add_data(*[row[c] for c in columns])
            wandb.log({"probe_completions": table}, step=self.global_step)

        # Log to file: results/{experiment_name}/logs/probe_completions.jsonl
        log_dir = Path(self.cfg.get("paths", {}).get(
            "log_dir",
            Path(self.cfg.get("paths", {}).get("experiment_dir", "results/unknown")) / "logs",
        ))
        log_dir.mkdir(parents=True, exist_ok=True)
        probe_file = log_dir / "probe_completions.jsonl"
        with open(probe_file, "a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        log.info(
            f"Logged {len(rows)} probe completions at step {self.global_step} "
            f"→ {probe_file}"
        )

    @staticmethod
    def _list_collate(batch: List[Dict[str, Any]]) -> Dict[str, list]:
        """Simple collation that groups values into lists.

        Unlike PyTorch's ``default_collate``, this avoids tensor conversion and
        gracefully handles variable-length nested structures (e.g. metadata
        dicts with different-length lists).  GRPO processes items one-by-one in
        ``_generate_completions``, so list-of-strings is the natural format.
        """
        keys = batch[0].keys()
        return {k: [item[k] for item in batch] for k in keys}

    def _setup_optimizer(self) -> None:
        """Set up optimizer with differential LR for pooling/projector.

        Uses a higher learning rate for randomly-initialized pooling and
        projector parameters (default 10x base LR), following LLaVA-style
        training. Same pattern as sft_trainer.create_optimizer().
        """
        lr = self.cfg.training.get("lr", 5e-6)
        weight_decay = self.cfg.training.get("weight_decay", 0.01)
        projector_lr = self.cfg.training.get("projector_lr", lr * 5)

        # LLM trainable parameters (LoRA adapters)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]

        param_groups = [
            {"params": lora_params, "lr": lr, "weight_decay": weight_decay},
        ]

        # Add pooling + projector params with higher LR
        if self.protein_llm is not None:
            extra_params = []
            if self.protein_llm.pooling is not None:
                extra_params.extend(
                    p for p in self.protein_llm.pooling.parameters()
                    if p.requires_grad
                )
            if self.protein_llm.projector is not None:
                extra_params.extend(
                    p for p in self.protein_llm.projector.parameters()
                    if p.requires_grad
                )
            if extra_params:
                param_groups.append({
                    "params": extra_params,
                    "lr": projector_lr,
                    "weight_decay": weight_decay,
                })
                if self.is_main_process:
                    num_extra = sum(p.numel() for p in extra_params)
                    log.info(
                        f"Added {num_extra:,} multimodal params "
                        f"(pooling+projector) with lr={projector_lr}"
                    )

        all_trainable = sum(
            sum(p.numel() for p in g["params"]) for g in param_groups
        )
        log.info(
            f"Optimizer: lr={lr}, projector_lr={projector_lr}, "
            f"total trainable={all_trainable:,}"
        )

        self.optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

        # Learning rate scheduler: linear warmup then cosine decay
        warmup_steps = self.cfg.training.get("warmup_steps", 100)
        grad_accum_steps = self.cfg.training.get("gradient_accumulation_steps", 8)
        batch_size = self.cfg.training.get("batch_size", 4)
        effective_batch = batch_size * grad_accum_steps * self.world_size
        steps_per_epoch = max(1, len(self.train_dataset) // effective_batch)
        total_steps = steps_per_epoch * self.cfg.training.get("epochs", 1)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(progress * math.pi))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_reward_function(self) -> None:
        """Set up the reward function based on task type."""
        task_type = self.cfg.data.get("task", "go_prediction")

        # ESMFold reward uses protein_sequence instead of ground_truth
        esmfold_tasks = {
            "esmfold", "structure", "structure_prediction", "fold_quality",
        }
        stability_tasks = {
            "stability", "stability_prediction", "ddg",
        }
        task_normalized = task_type.lower().replace("-", "_").replace(" ", "_")
        self._is_esmfold_reward = task_normalized in esmfold_tasks
        self._is_stability_reward = task_normalized in stability_tasks

        solubility_tasks = {"solubility", "solubility_prediction"}
        self._is_solubility_reward = task_normalized in solubility_tasks

        fold_tasks = {"fold_classification", "fold_class", "cath", "cath_classification"}
        self._is_fold_reward = task_normalized in fold_tasks

        # Focal reward weighting for category imbalance
        grpo_cfg = self.cfg.training.get("grpo", {})
        focal_enabled = grpo_cfg.get("focal_enabled", False)
        self._focal_gamma = grpo_cfg.get("focal_gamma", 2.0) if focal_enabled else 0.0
        self._binary_alignment = grpo_cfg.get("binary_alignment", False)
        self._classification_only = grpo_cfg.get("classification_only", False)
        if self._classification_only and self._is_esmfold_reward:
            log.info("Classification-only reward: correct category=1.0, wrong=0.0")
        if self._binary_alignment and self._is_esmfold_reward:
            log.info("Binary alignment scoring enabled: no partial credit, medium claims detected")
        if self._focal_gamma > 0 and self._is_esmfold_reward:
            from .rewards import _ESMFOLD_CATEGORY_FREQ, _focal_weight
            weights = {
                cat: round(_focal_weight(cat, self._focal_gamma), 3)
                for cat in _ESMFOLD_CATEGORY_FREQ
            }
            log.info(
                f"Focal reward weighting enabled (esmfold): gamma={self._focal_gamma}, "
                f"weights={weights}"
            )
        if self._focal_gamma > 0 and self._is_stability_reward:
            from .rewards import _STABILITY_CLASS_FREQ, _focal_weight
            weights = {
                cat: round(_focal_weight(cat, self._focal_gamma, _STABILITY_CLASS_FREQ), 3)
                for cat in _STABILITY_CLASS_FREQ
            }
            log.info(
                f"Focal reward weighting enabled (stability): gamma={self._focal_gamma}, "
                f"weights={weights}"
            )
        if self._focal_gamma > 0 and self._is_solubility_reward:
            from .rewards import _SOLUBILITY_CLASS_FREQ, _focal_weight
            weights = {
                cat: round(_focal_weight(cat, self._focal_gamma, _SOLUBILITY_CLASS_FREQ), 3)
                for cat in _SOLUBILITY_CLASS_FREQ
            }
            log.info(
                f"Focal reward weighting enabled (solubility): gamma={self._focal_gamma}, "
                f"weights={weights}"
            )

        try:
            self.reward_fn = get_reward_function(task_type)
            log.info(f"Using reward function for task: {task_type}")
        except ValueError:
            log.warning(f"Unknown task type: {task_type}. Using generic reward function.")
            self.reward_fn = compute_generic_reward
            self._is_esmfold_reward = False
            self._is_stability_reward = False
            self._is_solubility_reward = False
            self._is_fold_reward = False

    def _generate_completions(
        self,
        prompts: List[str],
        protein_sequences: Optional[List[str]],
        num_completions: int,
    ) -> Tuple[List[List[str]], List[List[torch.Tensor]], List[torch.Tensor]]:
        """Generate multiple completions for each prompt.

        Uses ProteinLLM.generate() for the esm3 approach so that protein
        structure embeddings flow through the encoder/pooling/projector
        pipeline.  All group_size completions for a prompt are batched
        into a single forward pass for efficiency.

        Args:
            prompts: List of input prompts.
            protein_sequences: List of protein sequences (one per prompt),
                or None for text-only approach.
            num_completions: Number of completions to generate per prompt.

        Returns:
            Tuple of:
                - List of lists of generated completion strings
                - List of lists of generated token ID tensors (one per completion)
                - List of prompt input_ids tensors (one per prompt)
        """
        all_completions = []
        all_generated_ids = []
        all_prompt_ids = []

        gen_kwargs = dict(
            max_new_tokens=self.grpo_config["max_new_tokens"],
            temperature=self.grpo_config["temperature"],
            top_p=self.grpo_config["top_p"],
            do_sample=self.grpo_config["do_sample"],
            use_cache=True,
        )

        # Explicitly enable KV cache on model config (gradient checkpointing
        # may have set it to False).  Restore after generation.
        base_model = self._get_base_model()
        _prev_use_cache = getattr(base_model.config, "use_cache", True)
        base_model.config.use_cache = True

        import time as _time
        _t_gen_start = _time.monotonic()

        # Build GRPO system prompt (base + answer tag instruction)
        task = self.cfg.data.get("task", "go_prediction").lower()
        enable_thinking = self.grpo_config.get("enable_thinking", False)
        grpo_system_prompt = _get_grpo_system_prompt(task, enable_thinking=enable_thinking)

        from src.models.multimodal_llm import PROTEIN_PLACEHOLDER

        for i, prompt in enumerate(prompts):
            # Build GRPO-formatted prompt with answer tag instruction +
            # empty thinking prefix.  Both paths share the same wrapping.
            is_multimodal = (
                self.protein_llm is not None
                and protein_sequences is not None
                and protein_sequences[i]
            )

            # Build user content (add protein placeholder for multimodal)
            user_content = prompt
            if is_multimodal:
                user_content = f"{prompt}\n\n{PROTEIN_PLACEHOLDER}"

            messages = [
                {"role": "system", "content": grpo_system_prompt},
                {"role": "user", "content": user_content},
            ]
            wrapped_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            # When thinking disabled: append empty <think></think> prefix.
            # When thinking enabled: model generates its own <think>...</think>.
            if not enable_thinking:
                wrapped_prompt += _THINKING_PREFIX

            if is_multimodal:
                # Multimodal path: use ProteinLLM.generate() with pre-wrapped prompt
                protein_seq = protein_sequences[i]
                batch_proteins = [protein_seq] * num_completions
                batch_prompts = [wrapped_prompt] * num_completions

                with torch.no_grad():
                    texts, token_ids, input_len = self.protein_llm.generate(
                        protein_sequences=batch_proteins,
                        prompt=batch_prompts,
                        return_token_ids=True,
                        wrap_chat_template=False,  # already wrapped above
                        **gen_kwargs,
                    )

                all_completions.append(texts)
                all_generated_ids.append(
                    [token_ids[j] for j in range(token_ids.shape[0])]
                )

            else:
                # Text-only path: use model.generate() directly
                inputs = self.tokenizer(
                    wrapped_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=(
                        self.cfg.training.get("max_seq_length", 2048)
                        - self.grpo_config["max_new_tokens"]
                    ),
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                prompt_length = inputs["input_ids"].shape[1]

                # Batch all completions in one generate call
                batch_ids = inputs["input_ids"].repeat(num_completions, 1)
                batch_mask = inputs["attention_mask"].repeat(num_completions, 1)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=batch_ids,
                        attention_mask=batch_mask,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **gen_kwargs,
                    )

                completions = []
                gen_ids_list = []
                for j in range(num_completions):
                    gen_ids = outputs[j, prompt_length:]
                    completion = self.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    )
                    completions.append(completion)
                    gen_ids_list.append(gen_ids)

                all_completions.append(completions)
                all_generated_ids.append(gen_ids_list)

            # Store prompt token IDs for log prob re-computation.
            # Uses the same wrapped_prompt (with GRPO system prompt + thinking
            # prefix) for consistency between generation and log-prob phases.
            prompt_inputs = self.tokenizer(
                wrapped_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=(
                    self.cfg.training.get("max_seq_length", 2048)
                    - self.grpo_config["max_new_tokens"]
                ),
            )
            all_prompt_ids.append(prompt_inputs["input_ids"].to(self.device))

        # Restore use_cache and log generation time
        base_model.config.use_cache = _prev_use_cache
        _t_gen_elapsed = _time.monotonic() - _t_gen_start
        if self.is_main_process:
            log.info(
                f"[TIMING] _generate_completions: {_t_gen_elapsed:.2f}s "
                f"({len(prompts)} prompts × {num_completions} completions)"
            )

        return all_completions, all_generated_ids, all_prompt_ids

    def _compute_sequence_log_prob(
        self,
        prompt_ids: torch.Tensor,
        full_ids: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """Compute log probability of generated sequence given prompt.

        Args:
            prompt_ids: Tokenized prompt.
            full_ids: Full sequence (prompt + completion).
            prompt_length: Length of prompt in tokens.

        Returns:
            Log probability of the completion.
        """
        # Forward pass to get logits
        with torch.no_grad():
            outputs = self.model(full_ids, return_dict=True)

        logits = outputs.logits[:, prompt_length - 1:-1, :]  # Shifted for next-token prediction
        target_ids = full_ids[:, prompt_length:]

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Sum over sequence
        sequence_log_prob = token_log_probs.sum()

        return sequence_log_prob

    def _compute_policy_log_probs(
        self,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        protein_sequence: Optional[str] = None,
        precomputed_encoder_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute differentiable log probabilities for generated tokens.

        Performs a forward pass WITH gradients so the policy gradient loss
        backpropagates through the LLM (LoRA), projector, and pooling.

        For embedding approaches, uses ProteinLLM.prepare_inputs() to
        prepend protein prefix embeddings, ensuring gradients flow through
        the full encoder → pooling → projector → LLM pipeline.

        Args:
            prompt_ids: Tokenized prompt of shape (1, prompt_len).
            generated_ids: Generated token IDs of shape (gen_len,) or
                (1, gen_len).
            protein_sequence: Protein sequence string for multimodal
                forward pass, or None for text-only.
            precomputed_encoder_embeds: Cached ESM-3 encoder output
                [1, L, D].  Skips frozen encoder, pooling+projector
                still run with gradients.

        Returns:
            Scalar tensor: sum of log probabilities over the generated
            sequence, with gradient graph attached for backpropagation.
        """
        # Ensure generated_ids is 2D
        if generated_ids.dim() == 1:
            generated_ids = generated_ids.unsqueeze(0)

        # Build full text sequence: [prompt | completion]
        full_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        prompt_length = prompt_ids.shape[1]
        attention_mask = torch.ones_like(full_ids)

        if self.protein_llm is not None and protein_sequence:
            # Multimodal: encode protein + prepend prefix embeddings
            prepared = self.protein_llm.prepare_inputs(
                protein_sequences=[protein_sequence],
                text_input_ids=full_ids,
                text_attention_mask=attention_mask,
                precomputed_encoder_embeds=precomputed_encoder_embeds,
            )
            outputs = self.protein_llm.llm(
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared["attention_mask"],
                position_ids=prepared["position_ids"],
                return_dict=True,
            )
            # Account for protein prefix tokens in logit positions.
            # prepare_inputs replaces 1 placeholder with N protein embeddings,
            # shifting all subsequent positions by N-1.  Logit at position j
            # predicts token at position j+1, so to predict the first generated
            # token (at transformed position prompt_length + N - 1) we need
            # logit at prompt_length + N - 2.
            num_prefix = self.protein_llm.num_prefix_tokens
            logits = outputs.logits[:, num_prefix + prompt_length - 2:-1, :]
        else:
            # Text-only: standard forward pass
            outputs = self.model(full_ids, return_dict=True)
            logits = outputs.logits[:, prompt_length - 1:-1, :]

        target_ids = full_ids[:, prompt_length:]

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding tokens
        pad_mask = (target_ids != self.tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * pad_mask

        return token_log_probs.sum()

    def _compute_batched_log_probs(
        self,
        prompt_ids: torch.Tensor,
        generated_ids_list: List[torch.Tensor],
        protein_sequence: Optional[str] = None,
        precomputed_encoder_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log probs for multiple completions in ONE forward pass.

        All completions share the same prompt and protein sequence.  They are
        padded to the same length and batched together for a single LLM
        forward, reducing sequential overhead from group_size calls to 1.

        Args:
            prompt_ids: Tokenized prompt [1, prompt_len].
            generated_ids_list: List of generated token ID tensors, one per
                completion.  Each is shape (gen_len,) or (1, gen_len).
            protein_sequence: Protein sequence for multimodal, or None.
            precomputed_encoder_embeds: Cached ESM-3 encoder output [1, L, D].

        Returns:
            Tensor of shape (group_size,) with per-completion log prob sums.
        """
        group_size = len(generated_ids_list)
        prompt_length = prompt_ids.shape[1]
        pad_id = self.tokenizer.pad_token_id

        # Ensure all generated_ids are 1D
        gen_ids = [
            g.squeeze(0) if g.dim() == 2 else g for g in generated_ids_list
        ]

        # Pad completions to same length
        max_gen_len = max(g.shape[0] for g in gen_ids)
        padded_gens = []
        gen_lengths = []
        for g in gen_ids:
            gen_lengths.append(g.shape[0])
            if g.shape[0] < max_gen_len:
                pad = torch.full(
                    (max_gen_len - g.shape[0],), pad_id,
                    dtype=g.dtype, device=g.device,
                )
                padded_gens.append(torch.cat([g, pad]))
            else:
                padded_gens.append(g)

        # Build batched full sequences: [prompt | padded_completion] × group_size
        batch_prompt = prompt_ids.expand(group_size, -1)  # [G, prompt_len]
        batch_gen = torch.stack(padded_gens)  # [G, max_gen_len]
        full_ids = torch.cat([batch_prompt, batch_gen], dim=1)  # [G, prompt_len + max_gen_len]

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = torch.ones_like(full_ids)
        for i, gl in enumerate(gen_lengths):
            if gl < max_gen_len:
                attention_mask[i, prompt_length + gl:] = 0

        if self.protein_llm is not None and protein_sequence:
            # Expand cached encoder embeds for the batch
            batch_embeds = (
                precomputed_encoder_embeds.expand(group_size, -1, -1)
                if precomputed_encoder_embeds is not None
                else None
            )
            prepared = self.protein_llm.prepare_inputs(
                protein_sequences=[protein_sequence] * group_size,
                text_input_ids=full_ids,
                text_attention_mask=attention_mask,
                precomputed_encoder_embeds=batch_embeds,
            )
            outputs = self.protein_llm.llm(
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared["attention_mask"],
                position_ids=prepared["position_ids"],
                return_dict=True,
            )
            num_prefix = self.protein_llm.num_prefix_tokens
            logits = outputs.logits[:, num_prefix + prompt_length - 2:-1, :]
        else:
            outputs = self.model(full_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits[:, prompt_length - 1:-1, :]

        target_ids = full_ids[:, prompt_length:]  # [G, max_gen_len]

        # Log probs with padding mask
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [G, max_gen_len]

        # Mask padding
        pad_mask = (target_ids != pad_id).float()
        token_log_probs = token_log_probs * pad_mask

        # Sum per completion
        return token_log_probs.sum(dim=-1)  # [G]

    def _compute_rewards(
        self,
        completions: List[List[str]],
        ground_truths: List[str],
        protein_sequences: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute rewards for all completions with supplementary metrics.

        Extracts content from ``<answer>`` tags before passing to reward
        functions.  Adds a format reward bonus when ``<answer>`` tags are
        present, encouraging consistent output structure.

        Args:
            completions: List of lists of completions (one list per prompt).
            ground_truths: List of ground truth responses.
            protein_sequences: Optional protein sequences for ESMFold reward.

        Returns:
            Tuple of:
                - Tensor of rewards with shape (batch_size, group_size).
                - Dict of supplementary metric lists (one value per completion).
        """
        from .rewards import FORMAT_REWARD_BONUS, extract_answer_content

        rewards = []
        all_metrics: Dict[str, List[float]] = {}

        for idx, (prompt_completions, ground_truth) in enumerate(
            zip(completions, ground_truths)
        ):
            prompt_rewards = []
            for completion in prompt_completions:
                # Extract answer content from <answer> tags
                answer_text, has_answer_tags = extract_answer_content(completion)

                # Route ground truth to reward function
                if self._is_esmfold_reward:
                    gt_str = str(ground_truth)
                    if gt_str.strip().startswith("{"):
                        second_arg = ground_truth
                    elif protein_sequences is not None:
                        second_arg = protein_sequences[idx]
                    else:
                        second_arg = ground_truth
                else:
                    second_arg = ground_truth

                reward_kwargs = {"detailed": True}
                # ESMFold-specific kwargs
                if self._is_esmfold_reward and self._focal_gamma > 0:
                    reward_kwargs["focal_gamma"] = self._focal_gamma
                if self._is_esmfold_reward and self._binary_alignment:
                    reward_kwargs["binary_alignment"] = True
                if self._is_esmfold_reward and self._classification_only:
                    reward_kwargs["classification_only"] = True
                # Stability-specific kwargs
                if self._is_stability_reward and self._focal_gamma > 0:
                    reward_kwargs["focal_gamma"] = self._focal_gamma
                # Solubility-specific kwargs
                if self._is_solubility_reward and self._focal_gamma > 0:
                    reward_kwargs["focal_gamma"] = self._focal_gamma

                reward, metrics = self.reward_fn(
                    answer_text, second_arg, **reward_kwargs
                )

                # Format reward: bonus for using <answer> tags
                format_bonus = FORMAT_REWARD_BONUS if has_answer_tags else 0.0
                reward += format_bonus

                prompt_rewards.append(reward)
                all_metrics.setdefault("format_bonus", []).append(format_bonus)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        all_metrics.setdefault(k, []).append(float(v))
            rewards.append(torch.tensor(prompt_rewards, device=self.device))

        return torch.stack(rewards), all_metrics

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages.

        For GRPO, advantages are computed relative to the group mean.
        This eliminates the need for a value function/critic.

        Args:
            rewards: Tensor of rewards with shape (batch_size, group_size).

        Returns:
            Tensor of advantages with same shape as rewards.
        """
        # Compute group mean and std
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True)

        # Compute advantages (relative to group mean)
        advantages = rewards - group_mean

        # Normalize advantages if configured (not for Dr. GRPO)
        if self.grpo_config["normalize_advantages"]:
            advantages = advantages / (group_std + 1e-8)

        return advantages

    def _compute_kl_penalty(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty between policy and reference model.

        Args:
            prompt_ids: Tokenized prompts.
            completion_ids: Tokenized completions.

        Returns:
            KL divergence penalty.
        """
        if self.ref_model is None:
            return torch.tensor(0.0, device=self.device)

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Get logits from both models
        with torch.no_grad():
            ref_outputs = self.ref_model(full_ids, return_dict=True)
            ref_logits = ref_outputs.logits

        policy_outputs = self.model(full_ids, return_dict=True)
        policy_logits = policy_outputs.logits

        # Compute KL divergence
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)

        kl_div = F.kl_div(
            policy_log_probs,
            ref_log_probs.exp(),
            reduction="batchmean",
            log_target=False,
        )

        return kl_div

    def _get_base_model(self):
        """Return unwrapped base model (past PeftModel wrapper)."""
        m = self.model
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        return m

    def _ensure_grad_ckpt_off(self) -> None:
        """Force-disable gradient checkpointing so generation uses KV cache."""
        base = self._get_base_model()
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
        # Also clear the flag on model config and all submodules
        if hasattr(base, "config"):
            base.config.gradient_checkpointing = False
        # Force-clear per-module gradient_checkpointing flags (belt + suspenders)
        for mod in base.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = False

    def _ensure_grad_ckpt_on(self) -> None:
        """Enable gradient checkpointing for training forward pass."""
        base = self._get_base_model()
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _sync_multimodal_gradients(self) -> None:
        """All-reduce pooling+projector gradients across DDP ranks.

        Pooling and projector are NOT FSDP-wrapped (replicated), so they
        need manual gradient synchronization. Same pattern as
        sft_trainer.ProteinLLMTrainer._sync_multimodal_gradients().
        """
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        if self.protein_llm is None:
            return
        for module in [self.protein_llm.pooling, self.protein_llm.projector]:
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad and p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _sync_all_gradients(self) -> None:
        """All-reduce ALL trainable gradients across ranks (non-FSDP mode).

        When FSDP is disabled, no automatic gradient sync happens.
        This manually all-reduces gradients for all trainable params
        (LoRA + pooling + projector).
        """
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if self.protein_llm is not None:
            for module in [self.protein_llm.pooling, self.protein_llm.projector]:
                if module is not None:
                    for p in module.parameters():
                        if p.requires_grad and p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _training_step(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute a single training step; return (loss, metrics).

        The caller (train loop) is responsible for backward, gradient sync,
        clipping, and optimizer step — enabling gradient accumulation.

        Steps:
        1. Generate completions inside torch.no_grad() (efficient sampling)
        2. Compute verifiable rewards (no grad needed)
        3. Compute group-relative advantages (detached)
        4. Re-compute log probs WITH gradients through ProteinLLM pipeline
        5. Compute and return policy gradient loss

        Args:
            batch: Batch of data containing prompts, ground truth, and
                protein sequences.

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        # Extract batch data — use raw instruction + input for GRPO prompt
        # rebuilding (so we can inject GRPO system prompt + answer tags).
        # Falls back to inference_prompt for backward compatibility.
        raw_instructions = batch.get("instruction", [])
        raw_inputs = batch.get("input_text", [])
        protein_sequences = batch.get("protein_sequence", None)

        if raw_instructions:
            # Build user content from raw fields.
            # For multimodal with protein sequences: use instruction only
            # (protein goes through ESM-3 encoder, placeholder added later).
            # For text-only or no protein: include input_text in prompt.
            prompts = []
            for i, inst in enumerate(raw_instructions):
                inp = (raw_inputs[i] if raw_inputs and i < len(raw_inputs) else "")
                has_protein = (
                    protein_sequences is not None
                    and i < len(protein_sequences)
                    and protein_sequences[i]
                    and self.protein_llm is not None
                )
                p = inst.strip()
                if inp and inp.strip() and not has_protein:
                    # Text-only: include raw input (protein sequence as text)
                    p += f"\n\n{inp.strip()}"
                prompts.append(p)
        else:
            prompts = batch.get(
                "inference_prompt",
                batch.get("formatted_prompt", batch.get("instruction", [])),
            )

        # Task-aware ground truth extraction:
        # - stability/ddg: pass JSON with ddG, stability_class, mutation from metadata
        # - esmfold/structure: pre-computed pLDDT from metadata, or protein_sequences
        # - go_prediction: use text response (comma-separated GO terms)
        # - default: use text response
        task = self.cfg.data.get("task", "go_prediction").lower()
        if task in ("stability", "ddg", "stability_prediction"):
            metadata_list = batch.get("metadata", [])
            if metadata_list and isinstance(metadata_list, list):
                ground_truths = []
                for m in metadata_list:
                    if isinstance(m, dict) and "ddG" in m:
                        ground_truths.append(json.dumps({
                            "ddG": m.get("ddG", 0),
                            "stability_class": m.get("stability_class"),
                            "mutation": m.get("mutation"),
                        }))
                    else:
                        ground_truths.append("")
            else:
                ground_truths = batch.get("response", batch.get("output", []))
        elif task in ("esmfold", "structure", "structure_prediction", "fold_quality"):
            # Pre-computed pLDDT in metadata — pass as JSON ground truth
            metadata_list = batch.get("metadata", [])
            if metadata_list and isinstance(metadata_list[0], dict) and "plddt" in metadata_list[0]:
                ground_truths = [
                    json.dumps({"plddt": m.get("plddt", 0), "ptm": m.get("ptm", 0)})
                    for m in metadata_list
                ]
            else:
                # No pre-computed metrics — fall through to protein_sequences path
                ground_truths = batch.get("response", batch.get("output", []))
        elif task in (
            "ss_composition", "structure_properties_a",
            "ss_sequence", "ss_per_residue", "structure_properties_b",
            "structure_composite", "structure_properties", "structure_properties_c",
        ):
            # Structural property tasks: reward functions expect metadata dict
            # with keys like helix_fraction, ss3_string, mean_rsa, etc.
            metadata_list = batch.get("metadata", [])
            if metadata_list and isinstance(metadata_list[0], dict):
                ground_truths = [json.dumps(m) for m in metadata_list]
            else:
                ground_truths = batch.get("response", batch.get("output", []))
        elif task in ("solubility", "solubility_prediction"):
            metadata_list = batch.get("metadata", [])
            if metadata_list and isinstance(metadata_list, list):
                ground_truths = []
                for m in metadata_list:
                    if isinstance(m, dict) and "solubility_score" in m:
                        ground_truths.append(json.dumps({
                            "solubility_score": m.get("solubility_score", 0),
                            "solubility_class": m.get("solubility_class"),
                        }))
                    else:
                        ground_truths.append("")
            else:
                ground_truths = batch.get("response", batch.get("output", []))
        elif task in ("fold_classification", "fold_class", "cath", "cath_classification"):
            metadata_list = batch.get("metadata", [])
            if metadata_list and isinstance(metadata_list, list):
                ground_truths = []
                for m in metadata_list:
                    if isinstance(m, dict) and "cath_code" in m:
                        ground_truths.append(json.dumps({
                            "cath_code": m.get("cath_code"),
                            "class_name": m.get("class_name"),
                            "architecture_name": m.get("architecture_name"),
                            "topology_name": m.get("topology_name"),
                            "homology_name": m.get("homology_name"),
                        }))
                    else:
                        ground_truths.append("")
            else:
                ground_truths = batch.get("response", batch.get("output", []))
        else:
            ground_truths = batch.get("response", batch.get("output", []))

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        if isinstance(protein_sequences, str):
            protein_sequences = [protein_sequences]

        group_size = self.grpo_config["group_size"]

        import time as _time

        # Step 1: Generate completions (no grad)
        # Disable gradient checkpointing and switch to eval mode so KV cache
        # works during generation.  HF's per-layer check is:
        #   self.gradient_checkpointing AND self.training AND use_cache
        # eval() sets training=False, which is sufficient.  We also disable
        # grad ckpt explicitly as belt-and-suspenders.
        self._ensure_grad_ckpt_off()
        self.model.eval()
        _t0 = _time.monotonic()

        completions, all_generated_ids, all_prompt_ids = (
            self._generate_completions(prompts, protein_sequences, group_size)
        )

        _t1 = _time.monotonic()
        # Switch back to train mode and re-enable gradient checkpointing
        self.model.train()
        self._ensure_grad_ckpt_on()

        # Step 2: Compute rewards
        rewards, reward_metrics = self._compute_rewards(
            completions, ground_truths, protein_sequences=protein_sequences
        )
        _t2 = _time.monotonic()

        # Step 3: Compute advantages (detached from reward computation)
        advantages = self._compute_advantages(rewards).detach()

        # Pre-compute ESM-3 encoder embeddings for all unique proteins.
        # ESM-3 is frozen → output is deterministic.  Cache avoids redundant
        # encoding during log-prob recomputation (group_size calls per protein).
        # Encode each sequence individually to avoid padding mismatch when
        # the cached tensor is used later with a single-sequence call.
        esm_cache = {}
        if (
            self.protein_llm is not None
            and protein_sequences is not None
            and self.protein_llm.encoder is not None
        ):
            unique_seqs = list(set(s for s in protein_sequences if s))
            if unique_seqs:
                # Try persistent LMDB cache first
                lmdb_cache = getattr(self, "_embedding_cache", None)
                with torch.no_grad():
                    for seq in unique_seqs:
                        cached = lmdb_cache.get(seq) if lmdb_cache else None
                        if cached is not None:
                            esm_cache[seq] = cached.unsqueeze(0).to(self.device)
                        else:
                            enc_out = self.protein_llm.encoder.encode([seq])
                            esm_cache[seq] = enc_out["embeddings"]  # [1, L, D]

        # Step 4: Re-compute log probs WITH gradients (differentiable forward)
        # Batch all completions for each prompt into a single forward pass.
        diff_log_probs = []
        for prompt_idx in range(len(prompts)):
            protein_seq = (
                protein_sequences[prompt_idx]
                if protein_sequences is not None
                else None
            )
            cached_embeds = esm_cache.get(protein_seq) if protein_seq else None
            prompt_log_probs = self._compute_batched_log_probs(
                all_prompt_ids[prompt_idx],
                all_generated_ids[prompt_idx],
                protein_sequence=protein_seq,
                precomputed_encoder_embeds=cached_embeds,
            )
            diff_log_probs.append(prompt_log_probs)
        log_probs = torch.stack(diff_log_probs)  # (batch_size, group_size)
        _t3 = _time.monotonic()

        # Step 5: Policy gradient loss = -E[advantage * log_prob]
        pg_loss = -(advantages * log_probs).mean()

        # KL penalty (stubbed for DAPO; use_kl_penalty defaults to False)
        if self.grpo_config["use_kl_penalty"]:
            kl_penalty = torch.tensor(0.0, device=self.device)
            loss = pg_loss + self.grpo_config["kl_coef"] * kl_penalty
        else:
            loss = pg_loss
            kl_penalty = torch.tensor(0.0)

        _t_total = _t3 - _t0  # total excluding backward (added in train loop)

        # --- Completion length diagnostics ---
        comp_lengths = []
        for prompt_gen_ids in all_generated_ids:
            for gen_ids in prompt_gen_ids:
                comp_lengths.append(gen_ids.shape[-1])

        # --- Reward diagnostics ---
        # frac_reward_zero_std: fraction of groups where all completions
        # get the same reward (no learning signal for that group)
        group_stds = rewards.std(dim=1)  # (batch_size,)
        frac_zero_std = (group_stds < 1e-8).float().mean().item()

        step_metrics = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "reward_std": rewards.std().item(),
            "timing/generate_s": _t1 - _t0,
            "timing/reward_s": _t2 - _t1,
            "timing/log_prob_s": _t3 - _t2,
            "timing/total_step_s": _t_total,
            "diagnostics/completion_mean_length": sum(comp_lengths) / max(len(comp_lengths), 1),
            "diagnostics/completion_max_length": max(comp_lengths) if comp_lengths else 0,
            "diagnostics/frac_reward_zero_std": frac_zero_std,
            "diagnostics/advantage_mean": advantages.mean().item(),
            "diagnostics/advantage_std": advantages.std().item(),
        }

        # Add averaged supplementary metrics from reward functions
        for k, values in reward_metrics.items():
            valid = [v for v in values if not math.isnan(v)]
            if valid:
                step_metrics[f"reward/{k}"] = sum(valid) / len(valid)

        return loss, step_metrics

    def train(self) -> Dict[str, Any]:
        """Run GRPO training with gradient accumulation and multi-GPU support.

        Returns:
            Dictionary of final training metrics.
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        batch_size = self.cfg.training.get("batch_size", 4)
        grad_accum_steps = self.cfg.training.get("gradient_accumulation_steps", 8)
        num_epochs = self.cfg.training.get("epochs", 1)
        logging_steps = self.cfg.training.get("logging_steps", 10)
        save_steps = self.cfg.training.get("save_steps", 100)
        eval_steps = self.cfg.training.get("eval_steps", 50)
        max_grad_norm = self.cfg.training.get("max_grad_norm", 1.0)

        if self.is_main_process:
            log.info("Starting GRPO training...")
            log.info(f"  Epochs: {num_epochs}")
            log.info(f"  Batch size: {batch_size}")
            log.info(f"  Gradient accumulation: {grad_accum_steps}")
            log.info(f"  Effective batch: {batch_size * grad_accum_steps * self.world_size}")
            log.info(f"  Group size: {self.grpo_config['group_size']}")
            log.info(f"  World size: {self.world_size}")
            log.info(f"  FSDP2: {self.use_fsdp}")
            log.info(f"  Learning rate: {self.cfg.training.get('lr', 5e-6)}")

        # Create dataloader with DistributedSampler for multi-GPU
        # Use _list_collate to avoid default_collate issues with
        # variable-length metadata (e.g. GO aspect lists).
        if self.world_size > 1:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                collate_fn=self._list_collate,
            )
        else:
            sampler = None
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=self._list_collate,
            )

        all_metrics = []

        for epoch in range(num_epochs):
            self.epoch = epoch
            if sampler is not None:
                sampler.set_epoch(epoch)

            if self.is_main_process:
                log.info(f"Epoch {epoch + 1}/{num_epochs}")

            epoch_metrics = []
            self.model.train()
            self.optimizer.zero_grad()
            accum_count = 0

            for step, batch in enumerate(dataloader):
                import time as _time

                loss, metrics = self._training_step(batch)

                # Scale loss for gradient accumulation
                scaled_loss = loss / grad_accum_steps
                _t_bwd_start = _time.monotonic()
                scaled_loss.backward()
                _t_bwd_end = _time.monotonic()
                metrics["timing/backward_s"] = _t_bwd_end - _t_bwd_start
                # Update total_step_s to include backward
                metrics["timing/total_step_s"] = (
                    metrics.get("timing/total_step_s", 0) + metrics["timing/backward_s"]
                )
                accum_count += 1
                epoch_metrics.append(metrics)

                # Optimizer step after accumulation
                if accum_count % grad_accum_steps == 0:
                    # Sync gradients across ranks
                    if self.use_fsdp:
                        # FSDP handles LLM grads; only sync multimodal
                        self._sync_multimodal_gradients()
                    else:
                        # No FSDP: manually sync ALL trainable grads
                        self._sync_all_gradients()

                    # Gradient clipping
                    if self.use_fsdp:
                        # FSDP-aware clipping for LLM params
                        llm = (
                            self.protein_llm.llm
                            if self.protein_llm is not None
                            else self.model
                        )
                        # FSDP2-wrapped models don't have clip_grad_norm_.
                        # Use torch.nn.utils instead, which handles DTensors.
                        llm_params = [
                            p for p in llm.parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        if llm_params:
                            torch.nn.utils.clip_grad_norm_(llm_params, max_grad_norm)
                        # Clip multimodal params separately
                        if self.protein_llm is not None:
                            mm_params = []
                            for mod in [self.protein_llm.pooling, self.protein_llm.projector]:
                                if mod is not None:
                                    mm_params.extend(
                                        p for p in mod.parameters()
                                        if p.requires_grad and p.grad is not None
                                    )
                            if mm_params:
                                torch.nn.utils.clip_grad_norm_(mm_params, max_grad_norm)
                    else:
                        all_params = [
                            p for p in self.model.parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        if self.protein_llm is not None:
                            for mod in [self.protein_llm.pooling, self.protein_llm.projector]:
                                if mod is not None:
                                    all_params.extend(
                                        p for p in mod.parameters()
                                        if p.requires_grad and p.grad is not None
                                    )
                        if all_params:
                            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

                    # Compute gradient norms (after clipping, for diagnostics)
                    _lora_grad_norm = 0.0
                    _mm_grad_norm = 0.0
                    lora_model = self.model
                    lora_grads = [
                        p.grad.detach().float().norm()
                        for p in lora_model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    if lora_grads:
                        _lora_grad_norm = torch.stack(lora_grads).norm().item()
                    if self.protein_llm is not None:
                        mm_grads = []
                        for mod in [self.protein_llm.pooling, self.protein_llm.projector]:
                            if mod is not None:
                                mm_grads.extend(
                                    p.grad.detach().float().norm()
                                    for p in mod.parameters()
                                    if p.requires_grad and p.grad is not None
                                )
                        if mm_grads:
                            _mm_grad_norm = torch.stack(mm_grads).norm().item()
                    # Store in the most recent metrics entry
                    if epoch_metrics:
                        epoch_metrics[-1]["diagnostics/grad_norm_lora"] = _lora_grad_norm
                        epoch_metrics[-1]["diagnostics/grad_norm_multimodal"] = _mm_grad_norm

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.is_main_process and self.global_step % logging_steps == 0:
                        window = min(
                            logging_steps * grad_accum_steps,
                            len(epoch_metrics),
                        )
                        recent = epoch_metrics[-window:]
                        # Collect all keys across recent entries
                        all_recent_keys = set()
                        for m in recent:
                            all_recent_keys.update(m.keys())
                        avg_metrics = {
                            k: sum(m.get(k, 0) for m in recent) / max(sum(1 for m in recent if k in m), 1)
                            for k in all_recent_keys
                        }
                        avg_metrics["lr"] = self.scheduler.get_last_lr()[0]
                        t_gen = avg_metrics.get("timing/generate_s", 0)
                        t_logp = avg_metrics.get("timing/log_prob_s", 0)
                        t_bwd = avg_metrics.get("timing/backward_s", 0)
                        t_total = avg_metrics.get("timing/total_step_s", 0)
                        grad_lora = avg_metrics.get("diagnostics/grad_norm_lora", 0)
                        grad_mm = avg_metrics.get("diagnostics/grad_norm_multimodal", 0)
                        log.info(
                            f"Step {self.global_step}: "
                            f"loss={avg_metrics['loss']:.4f}, "
                            f"reward={avg_metrics['mean_reward']:.4f}, "
                            f"lr={avg_metrics['lr']:.2e}, "
                            f"t_gen={t_gen:.1f}s, t_logp={t_logp:.1f}s, "
                            f"t_bwd={t_bwd:.1f}s, t_total={t_total:.1f}s, "
                            f"grad_lora={grad_lora:.4f}, grad_mm={grad_mm:.4f}"
                        )
                        if HAS_WANDB and wandb.run is not None:
                            wandb.log(avg_metrics, step=self.global_step)

                    # Evaluation
                    if self.global_step % eval_steps == 0:
                        eval_metrics = self.evaluate()
                        if self.is_main_process:
                            log.info(f"Eval metrics: {eval_metrics}")
                            if HAS_WANDB and wandb.run is not None:
                                wandb.log(
                                    {f"eval_{k}": v for k, v in eval_metrics.items()},
                                    step=self.global_step,
                                )
                        # Log probe completions to wandb
                        self._log_probe_completions()

                    # Save checkpoint
                    if self.is_main_process and self.global_step % save_steps == 0:
                        checkpoint_dir = Path(
                            self.cfg.get("paths", {}).get(
                                "checkpoint_dir", "./checkpoints"
                            )
                        )
                        window = min(
                            logging_steps * grad_accum_steps,
                            len(epoch_metrics),
                        )
                        recent = epoch_metrics[-window:]
                        ckpt_metrics = {
                            k: sum(m.get(k, 0) for m in recent) / max(sum(1 for m in recent if k in m), 1)
                            for k in metrics.keys()
                        }
                        self.save_checkpoint(
                            path=checkpoint_dir / f"checkpoint-{self.global_step}",
                            metrics=ckpt_metrics,
                        )

            all_metrics.extend(epoch_metrics)

        # Compute final metrics (handle varying keys across steps)
        final_metrics = {}
        if all_metrics:
            all_keys = set()
            for m in all_metrics:
                all_keys.update(m.keys())
            final_metrics = {
                k: sum(m.get(k, 0) for m in all_metrics) / len(all_metrics)
                for k in all_keys
            }

        # Save final checkpoint (main process only)
        if self.is_main_process:
            self.save_checkpoint(metrics=final_metrics)
            log.info(f"Training completed. Final metrics: {final_metrics}")

        return final_metrics

    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """Run evaluation on validation set using ProteinLLM when available.

        Uses GRPO system prompt with answer tag instruction for consistency
        with training.  Extracts answer content from ``<answer>`` tags before
        computing rewards.

        Args:
            num_samples: Number of samples to evaluate on.

        Returns:
            Dictionary of evaluation metrics.
        """
        from .rewards import extract_answer_content

        if self.eval_dataset is None:
            return {}

        # Disable gradient checkpointing + eval mode for KV cache during generation
        self._ensure_grad_ckpt_off()
        self.model.eval()

        task = self.cfg.data.get("task", "go_prediction").lower()
        enable_thinking = self.grpo_config.get("enable_thinking", False)
        grpo_system_prompt = _get_grpo_system_prompt(task, enable_thinking=enable_thinking)

        num_samples = min(num_samples, len(self.eval_dataset))
        eval_rewards = []
        format_hits = 0
        # Per-class reward tracking for classification tasks
        class_rewards: Dict[str, List[float]] = {}

        with torch.no_grad():
            for i in range(num_samples):
                sample = self.eval_dataset[i]

                # Use raw instruction + input_text for GRPO prompt rebuilding
                instruction = sample.get("instruction", "")
                input_text = sample.get("input_text", "")
                if instruction and input_text:
                    raw_prompt = f"{instruction.strip()}\n\n{input_text.strip()}"
                else:
                    raw_prompt = sample.get(
                        "inference_prompt",
                        sample.get("formatted_prompt", sample.get("instruction", "")),
                    )
                protein_seq = sample.get("protein_sequence", None)

                # Task-aware ground truth extraction (mirrors _training_step)
                task = self.cfg.data.get("task", "go_prediction").lower()
                metadata = sample.get("metadata", {})
                true_category = None
                if task in ("stability", "ddg", "stability_prediction") and isinstance(metadata, dict) and "ddG" in metadata:
                    ground_truth = json.dumps({
                        "ddG": metadata.get("ddG", 0),
                        "stability_class": metadata.get("stability_class"),
                        "mutation": metadata.get("mutation"),
                    })
                    true_category = metadata.get("stability_class")
                elif (
                    task in ("esmfold", "structure", "structure_prediction", "fold_quality")
                    and isinstance(metadata, dict) and "plddt" in metadata
                ):
                    plddt_val = float(metadata.get("plddt", 0))
                    true_category = (
                        "high" if plddt_val > 80
                        else "medium" if plddt_val > 50
                        else "low"
                    )
                    ground_truth = json.dumps({
                        "plddt": metadata.get("plddt", 0),
                        "ptm": metadata.get("ptm", 0),
                    })
                elif (
                    task in ("solubility", "solubility_prediction")
                    and isinstance(metadata, dict) and "solubility_score" in metadata
                ):
                    ground_truth = json.dumps({
                        "solubility_score": metadata.get("solubility_score", 0),
                        "solubility_class": metadata.get("solubility_class"),
                    })
                    true_category = metadata.get("solubility_class")
                elif (
                    task in ("fold_classification", "fold_class", "cath", "cath_classification")
                    and isinstance(metadata, dict) and "cath_code" in metadata
                ):
                    ground_truth = json.dumps({
                        "cath_code": metadata.get("cath_code"),
                        "class_name": metadata.get("class_name"),
                        "architecture_name": metadata.get("architecture_name"),
                        "topology_name": metadata.get("topology_name"),
                        "homology_name": metadata.get("homology_name"),
                    })
                    true_category = metadata.get("class_name")
                else:
                    ground_truth = sample.get("response", sample.get("output", ""))

                # Multimodal: use ProteinLLM.generate() with GRPO system prompt
                if self.protein_llm is not None and protein_seq:
                    texts = self.protein_llm.generate(
                        protein_sequences=[protein_seq],
                        prompt=[raw_prompt],
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        use_cache=True,
                        system_prompt=grpo_system_prompt,
                        generation_prefix=None if enable_thinking else _THINKING_PREFIX,
                    )
                    completion = texts[0]
                else:
                    # Text-only: wrap with GRPO system prompt + thinking prefix
                    messages = [
                        {"role": "system", "content": grpo_system_prompt},
                        {"role": "user", "content": raw_prompt},
                    ]
                    wrapped = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                    if not enable_thinking:
                        wrapped += _THINKING_PREFIX

                    inputs = self.tokenizer(
                        wrapped,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.cfg.training.get("max_seq_length", 2048) - 256,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    completion = self.tokenizer.decode(
                        outputs[0, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )

                # Extract answer content from <answer> tags
                answer_text, has_answer_tags = extract_answer_content(completion)
                if has_answer_tags:
                    format_hits += 1

                # Route reward computation with task-specific kwargs
                extra_kw = {}
                if self._is_esmfold_reward:
                    gt_str = str(ground_truth)
                    if gt_str.strip().startswith("{"):
                        second_arg = ground_truth
                    elif protein_seq:
                        second_arg = protein_seq
                    else:
                        second_arg = ground_truth
                elif self._is_stability_reward:
                    second_arg = ground_truth
                    if self._focal_gamma > 0:
                        extra_kw["focal_gamma"] = self._focal_gamma
                elif self._is_solubility_reward:
                    second_arg = ground_truth
                    if self._focal_gamma > 0:
                        extra_kw["focal_gamma"] = self._focal_gamma
                else:
                    second_arg = ground_truth
                reward = self.reward_fn(answer_text, second_arg, **extra_kw)
                eval_rewards.append(reward)

                # Track per-class reward for classification tasks
                if true_category is not None:
                    class_rewards.setdefault(true_category, []).append(reward)

        # Re-enable gradient checkpointing and train mode
        self.model.train()
        self._ensure_grad_ckpt_on()

        results = {
            "mean_reward": sum(eval_rewards) / len(eval_rewards),
            "max_reward": max(eval_rewards),
            "min_reward": min(eval_rewards),
            "format_rate": format_hits / max(num_samples, 1),
        }

        # Per-class average rewards for classification tasks (ESMFold or stability)
        for cat, cat_rewards in class_rewards.items():
            if cat_rewards:
                results[f"reward/mean_{cat}"] = (
                    sum(cat_rewards) / len(cat_rewards)
                )
                results[f"reward/count_{cat}"] = len(cat_rewards)

        return results

    def _save_model_inner(self, path: Path) -> None:
        """Save model weights, handling FSDP2 sharded params if needed.

        For FSDP2 (``fully_shard``), parameters are DTensor-sharded.  PEFT's
        ``save_pretrained`` can't access ``.data_ptr()`` on sharded tensors.
        We gather full state dict via ``get_model_state_dict()`` and manually
        save the LoRA adapter weights.
        """
        if self.use_fsdp:
            self._save_model_fsdp2(path)
        elif self.protein_llm is not None:
            self.protein_llm.save_pretrained(path / "protein_llm")
        elif HAS_PEFT and isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

    def _save_model_fsdp2(self, path: Path) -> None:
        """Save model under FSDP2.

        FSDP2 (``fully_shard``) uses DTensor for sharded params. PEFT's
        ``save_pretrained`` calls ``storage_ptr()`` which fails on DTensors.

        Current approach: save multimodal components (pooling/projector)
        directly on rank 0 (not FSDP-sharded), and log a warning that
        LoRA adapter save under FSDP2 is not yet supported.

        TODO(engineer): Implement proper FSDP2 adapter save. Options:
        1. Disable FSDP before final save (reshard → unshard)
        2. Use torch.distributed.checkpoint with proper process group
        3. Save during training via intermediate HF Trainer checkpoints
        """
        # Save multimodal components (pooling + projector + config) on rank 0.
        # These are NOT FSDP-sharded, so direct save works.
        if self.protein_llm is not None and self.is_main_process:
            plm_path = path / "protein_llm"
            plm_path.mkdir(parents=True, exist_ok=True)

            import json as _json
            config = {
                "approach": self.protein_llm.approach,
                "llm_name": self.protein_llm.llm_name,
                "encoder_name": self.protein_llm.encoder_name,
                "num_prefix_tokens": self.protein_llm.num_prefix_tokens,
                "pooling_type": self.protein_llm.pooling_type,
                "projector_type": self.protein_llm.projector_type,
                "encoder_embed_dim": self.protein_llm.encoder_embed_dim,
                "llm_hidden_size": self.protein_llm.llm_hidden_size,
            }
            with open(plm_path / "config.json", "w") as f:
                _json.dump(config, f, indent=2)

            if self.protein_llm.pooling is not None:
                torch.save(
                    self.protein_llm.pooling.state_dict(), plm_path / "pooling.pt"
                )
            if self.protein_llm.projector is not None:
                torch.save(
                    self.protein_llm.projector.state_dict(), plm_path / "projector.pt"
                )
            if self.protein_llm.tokenizer is not None:
                self.protein_llm.tokenizer.save_pretrained(plm_path / "tokenizer")

        if self.is_main_process:
            log.warning(
                "FSDP2 checkpoint: saved pooling/projector but LoRA adapter "
                "save is not yet supported with FSDP2. The LoRA weights from "
                "the training run are NOT persisted. Disable FSDP for the "
                "final checkpoint save, or use intermediate checkpoints."
            )

        if dist.is_initialized():
            dist.barrier()

    def save_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model checkpoint to experiment_dir/checkpoints/.

        Saves all artifacts under the unified experiment directory:
        - checkpoints/protein_llm/: ProteinLLM checkpoint (or adapter at root)
        - checkpoints/tokenizer/: Tokenizer files
        - checkpoints/training_state.pt: Optimizer/scheduler state for resuming
        - training_args.json: All hyperparameters (at experiment root)
        - metrics.json: Final train/val loss and eval metrics (at experiment root)

        Args:
            path: Checkpoint directory path. If None, uses experiment_dir/checkpoints/.
            metrics: Training/eval metrics to save. If None, empty dict is saved.

        Returns:
            Path to the saved checkpoint directory.
        """
        if path is None:
            path = Path(
                self.cfg.get("paths", {}).get("checkpoint_dir", "./checkpoints")
            )
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving checkpoint to: {path}")

        # Save model: ProteinLLM (pooling + projector + adapter + config)
        # or bare LoRA adapter for text-only approach.
        # _save_model_inner handles FSDP2 internally via get_model_state_dict.
        # NOTE: all ranks must call this (FSDP2 gather is collective).
        self._save_model_inner(path)

        # Only rank 0 writes remaining artifacts
        if not self.is_main_process:
            return path

        # Save tokenizer
        self.tokenizer.save_pretrained(path / "tokenizer")

        # Save training_args.json at experiment root level
        experiment_dir = Path(
            self.cfg.get("paths", {}).get("experiment_dir", path.parent)
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        training_args = {
            "method": self.cfg.training.get("method", "grpo"),
            "approach": self.cfg.get("approach", "text"),
            "model": self.cfg.model.get("name", "unknown"),
            "model_path": self.cfg.model.get("path", "unknown"),
            "dataset": self.cfg.data.get("name", "unknown"),
            "task": self.cfg.data.get("task", "go_prediction"),
            "lr": self.cfg.training.get("lr", None),
            "projector_lr": self.cfg.training.get("projector_lr", None),
            "epochs": self.cfg.training.get("epochs", None),
            "batch_size": self.cfg.training.get("batch_size", None),
            "gradient_accumulation_steps": self.cfg.training.get(
                "gradient_accumulation_steps", None
            ),
            "max_seq_length": self.cfg.training.get("max_seq_length", None),
            "max_grad_norm": self.cfg.training.get("max_grad_norm", None),
            "grpo": {
                "group_size": self.grpo_config.get("group_size", 4),
                "temperature": self.grpo_config.get("temperature", 1.0),
                "use_kl_penalty": self.grpo_config.get("use_kl_penalty", False),
                "normalize_advantages": self.grpo_config.get(
                    "normalize_advantages", False
                ),
                "kl_coef": self.grpo_config.get("kl_coef", 0.1),
                "clip_range": self.grpo_config.get("clip_range", 0.2),
                "max_new_tokens": self.grpo_config.get("max_new_tokens", 512),
            },
            "lora": {
                "r": self.cfg.training.get("lora", {}).get("r", None),
                "alpha": self.cfg.training.get("lora", {}).get("alpha", None),
                "dropout": self.cfg.training.get("lora", {}).get("dropout", None),
                "target_modules": list(
                    self.cfg.training.get("lora", {}).get(
                        "target_modules", [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ]
                    )
                ),
            },
            "global_step": self.global_step,
            "epoch": self.epoch,
            "timestamp": datetime.now().isoformat(),
        }
        with open(experiment_dir / "training_args.json", "w") as f:
            json.dump(training_args, f, indent=2, default=str)

        # Save metrics.json at experiment root level
        metrics_to_save = metrics if metrics is not None else {}
        with open(experiment_dir / "metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2, default=str)

        # Save training state for resuming
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "grpo_config": self.grpo_config,
            },
            path / "training_state.pt",
        )

        log.info(f"Checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint.

        Args:
            path: Directory path to load the checkpoint from.
        """
        path = Path(path)

        log.info(f"Loading checkpoint from: {path}")

        # Load training state
        state = torch.load(path / "training_state.pt", map_location=self.device, weights_only=True)
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])

        # Load model
        if (path / "adapter").exists() and HAS_PEFT:
            self.model = PeftModel.from_pretrained(
                self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model,
                path / "adapter",
            )
        elif (path / "model").exists():
            self.model = AutoModelForCausalLM.from_pretrained(path / "model")

        log.info(f"Checkpoint loaded from {path}")


# =============================================================================
# Main Training Functions
# =============================================================================


def run_grpo(cfg: DictConfig) -> Dict[str, Any]:
    """Run full GRPO training pipeline.

    This is the main entry point for GRPO training. It creates a GRPOTrainer,
    sets up all components, runs training, and returns final metrics.

    Args:
        cfg: Hydra configuration containing all training settings.

    Returns:
        Dictionary of training metrics.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("configs/training/grpo.yaml")
        >>> metrics = run_grpo(cfg)
    """
    log.info("=" * 60)
    log.info("Starting GRPO Training")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model.get('path', cfg.model.get('name', 'unknown'))}")
    log.info(f"Learning rate: {cfg.training.get('lr', 5e-6)}")
    log.info(f"Batch size: {cfg.training.get('batch_size', 4)}")
    log.info(f"Group size: {cfg.training.get('grpo', {}).get('group_size', 4)}")

    # Create trainer
    trainer = GRPOTrainer(cfg)

    # Setup
    trainer.setup()

    # Train
    metrics = trainer.train()

    # Final evaluation
    eval_metrics = trainer.evaluate()
    metrics.update({f"final_eval_{k}": v for k, v in eval_metrics.items()})

    log.info("=" * 60)
    log.info("GRPO Training Completed")
    log.info("=" * 60)
    log.info(f"Final metrics: {metrics}")

    return metrics


def run_grpo_with_trl(cfg: DictConfig) -> Dict[str, Any]:
    """Run GRPO training using TRL's GRPOTrainer.

    This is an alternative implementation that uses TRL's native GRPOTrainer
    if available. Falls back to custom implementation if TRL GRPO is not installed.

    Args:
        cfg: Hydra configuration.

    Returns:
        Dictionary of training metrics.
    """
    if not HAS_TRL_GRPO:
        log.warning("TRL GRPOTrainer not available. Using custom implementation.")
        return run_grpo(cfg)

    log.info("Running GRPO with TRL's native GRPOTrainer...")

    # This would use TRL's GRPOTrainer with a custom reward function
    # Implementation depends on TRL version and API
    raise NotImplementedError(
        "TRL GRPO integration not yet implemented. Use run_grpo() instead."
    )


# =============================================================================
# Utility Functions
# =============================================================================


def create_reward_dataset(
    dataset: Dataset,
    reward_fn: Callable,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    group_size: int = 4,
) -> List[Dict[str, Any]]:
    """Create a dataset with precomputed rewards for offline GRPO.

    This can be used to precompute rewards for faster training or analysis.

    Args:
        dataset: Source dataset with prompts and ground truths.
        reward_fn: Reward function to use.
        model: Model for generating completions.
        tokenizer: Tokenizer for the model.
        group_size: Number of completions per prompt.

    Returns:
        List of samples with completions and rewards.
    """
    reward_data = []

    for sample in dataset:
        prompt = sample.get("formatted_prompt", sample.get("instruction", ""))
        ground_truth = sample.get("response", sample.get("output", ""))

        # Generate completions
        completions = []
        rewards = []

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        for _ in range(group_size):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                )

            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = completion[len(prompt):]  # Remove prompt
            completions.append(completion)

            reward = reward_fn(completion, ground_truth)
            rewards.append(reward)

        reward_data.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "completions": completions,
            "rewards": rewards,
        })

    return reward_data
