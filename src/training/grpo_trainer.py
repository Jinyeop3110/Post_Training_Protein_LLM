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

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        PreTrainedTokenizer,
        PreTrainedModel,
        GenerationConfig,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig
    HAS_TRL_GRPO = True
except ImportError:
    HAS_TRL_GRPO = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


log = logging.getLogger(__name__)


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
    }


def get_reward_function(task: str) -> Callable[[str, Any], float]:
    """Get verifiable reward function for a specific protein task.

    Returns the appropriate reward function based on task type. These reward
    functions compute scores by comparing model predictions to ground truth,
    enabling verifiable rewards without a separate reward model.

    Args:
        task: Task type identifier. Supported values:
            - "go_prediction": Gene Ontology term prediction
            - "go_terms": Alias for go_prediction
            - "ppi": Protein-protein interaction prediction
            - "stability": Protein stability (ddG) prediction
            - "function": Generic function prediction (uses GO reward)

    Returns:
        Reward function that takes (generated_text, ground_truth) and returns float.

    Raises:
        ValueError: If task type is not supported.
    """
    task_lower = task.lower().replace("-", "_").replace(" ", "_")

    reward_functions = {
        "go_prediction": compute_go_reward,
        "go_terms": compute_go_reward,
        "go": compute_go_reward,
        "ppi": compute_ppi_reward,
        "ppi_prediction": compute_ppi_reward,
        "protein_protein_interaction": compute_ppi_reward,
        "stability": compute_stability_reward,
        "stability_prediction": compute_stability_reward,
        "ddg": compute_stability_reward,
        "function": compute_go_reward,
        "function_prediction": compute_go_reward,
    }

    if task_lower not in reward_functions:
        supported = list(set(reward_functions.values()))
        raise ValueError(
            f"Unsupported task type: {task}. "
            f"Supported tasks: go_prediction, ppi, stability"
        )

    return reward_functions[task_lower]


# =============================================================================
# Reward Functions for Verifiable Tasks
# =============================================================================


def compute_go_reward(
    generated_text: str,
    ground_truth_go_terms: Union[str, List[str]],
) -> float:
    """Compute reward based on F1 score of predicted GO terms.

    Extracts GO terms from generated text and computes F1 score against
    ground truth terms. GO terms are expected in format GO:XXXXXXX.

    Args:
        generated_text: Model-generated text containing GO term predictions.
        ground_truth_go_terms: Ground truth GO terms as string (comma/space
            separated) or list of GO term strings.

    Returns:
        F1 score between 0 and 1. Returns 0 if no terms could be extracted.

    Example:
        >>> compute_go_reward("The protein has GO:0003674 and GO:0005575",
        ...                   ["GO:0003674", "GO:0008150"])
        0.5  # Precision: 0.5, Recall: 0.5, F1: 0.5
    """
    # Extract GO terms from generated text (format: GO:XXXXXXX)
    go_pattern = r"GO:\d{7}"
    predicted_terms = set(re.findall(go_pattern, generated_text.upper()))

    # Normalize ground truth
    if isinstance(ground_truth_go_terms, str):
        ground_truth_terms = set(re.findall(go_pattern, ground_truth_go_terms.upper()))
    else:
        ground_truth_terms = set()
        for term in ground_truth_go_terms:
            matches = re.findall(go_pattern, str(term).upper())
            ground_truth_terms.update(matches)

    # Handle edge cases
    if not ground_truth_terms:
        # No valid GO terms in ground truth - use binary match
        return 1.0 if not predicted_terms else 0.0

    if not predicted_terms:
        return 0.0

    # Compute F1 score
    true_positives = len(predicted_terms & ground_truth_terms)
    precision = true_positives / len(predicted_terms) if predicted_terms else 0.0
    recall = true_positives / len(ground_truth_terms) if ground_truth_terms else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_ppi_reward(
    generated_text: str,
    ground_truth_label: Union[str, int, bool],
) -> float:
    """Compute reward based on correct PPI (protein-protein interaction) prediction.

    Determines if the model correctly predicted whether two proteins interact.
    Searches for positive/negative indicators in the generated text.

    Args:
        generated_text: Model-generated text containing interaction prediction.
        ground_truth_label: True interaction label. Can be:
            - Boolean: True (interacts) or False (does not interact)
            - Integer: 1 (interacts) or 0 (does not interact)
            - String: "yes"/"interact"/"positive" vs "no"/"not"/"negative"

    Returns:
        1.0 if prediction matches ground truth, 0.0 otherwise.

    Example:
        >>> compute_ppi_reward("Yes, these proteins interact strongly.", True)
        1.0
        >>> compute_ppi_reward("The proteins do not interact.", True)
        0.0
    """
    # Normalize ground truth to boolean
    if isinstance(ground_truth_label, bool):
        gt_interacts = ground_truth_label
    elif isinstance(ground_truth_label, int):
        gt_interacts = ground_truth_label == 1
    elif isinstance(ground_truth_label, str):
        gt_label_lower = ground_truth_label.lower().strip()
        positive_indicators = {"yes", "true", "1", "interact", "interacts", "positive", "binding"}
        gt_interacts = any(ind in gt_label_lower for ind in positive_indicators)
    else:
        # Try to convert to int
        try:
            gt_interacts = int(ground_truth_label) == 1
        except (ValueError, TypeError):
            log.warning(f"Could not parse ground truth label: {ground_truth_label}")
            return 0.0

    # Parse generated text for prediction
    text_lower = generated_text.lower()

    # Check for explicit positive indicators
    positive_patterns = [
        r"\byes\b",
        r"\binteract[s]?\b",
        r"\bbind[s]?\b",
        r"\bpositive\b",
        r"\btrue\b",
        r"\bwill interact\b",
        r"\bdo interact\b",
        r"\blikely to interact\b",
    ]

    # Check for explicit negative indicators
    negative_patterns = [
        r"\bno\b",
        r"\bnot interact\b",
        r"\bdon't interact\b",
        r"\bdo not interact\b",
        r"\bnegative\b",
        r"\bfalse\b",
        r"\bunlikely\b",
        r"\bwill not\b",
        r"\bwon't\b",
    ]

    # Count matches
    positive_score = sum(1 for p in positive_patterns if re.search(p, text_lower))
    negative_score = sum(1 for p in negative_patterns if re.search(p, text_lower))

    # Determine prediction
    if positive_score > negative_score:
        pred_interacts = True
    elif negative_score > positive_score:
        pred_interacts = False
    else:
        # Ambiguous - check for any yes/no at the start
        if text_lower.strip().startswith("yes"):
            pred_interacts = True
        elif text_lower.strip().startswith("no"):
            pred_interacts = False
        else:
            # Cannot determine, give partial credit based on any match
            pred_interacts = None

    if pred_interacts is None:
        return 0.5  # Uncertain prediction gets partial credit

    return 1.0 if pred_interacts == gt_interacts else 0.0


def compute_stability_reward(
    generated_text: str,
    ground_truth_ddg: Union[str, float],
    tolerance: float = 1.0,
) -> float:
    """Compute reward based on ddG prediction accuracy.

    Extracts predicted ddG (change in Gibbs free energy) from generated text
    and computes reward based on how close it is to ground truth. Uses a
    smooth reward function based on the error magnitude.

    Args:
        generated_text: Model-generated text containing stability prediction.
        ground_truth_ddg: True ddG value in kcal/mol. Can be string or float.
        tolerance: Error tolerance in kcal/mol for reward scaling. Default 1.0.
            Errors within tolerance get high rewards; errors beyond 2*tolerance
            get near-zero rewards.

    Returns:
        Reward between 0 and 1 based on prediction accuracy:
            - 1.0: Perfect prediction
            - High (>0.5): Error within tolerance
            - Low (<0.5): Error beyond tolerance

    Example:
        >>> compute_stability_reward("The predicted ddG is 2.5 kcal/mol", 2.3)
        0.96  # Small error, high reward
        >>> compute_stability_reward("ddG = -1.0", 3.0)
        0.02  # Large error, low reward
    """
    # Parse ground truth
    if isinstance(ground_truth_ddg, str):
        try:
            # Extract number from string
            numbers = re.findall(r"-?\d+\.?\d*", ground_truth_ddg)
            if numbers:
                gt_value = float(numbers[0])
            else:
                log.warning(f"Could not parse ground truth ddG: {ground_truth_ddg}")
                return 0.0
        except ValueError:
            log.warning(f"Could not parse ground truth ddG: {ground_truth_ddg}")
            return 0.0
    else:
        gt_value = float(ground_truth_ddg)

    # Extract predicted ddG from generated text
    # Look for patterns like: "ddG = X", "ddG: X", "X kcal/mol", etc.
    patterns = [
        r"ddG\s*[=:]\s*(-?\d+\.?\d*)",
        r"ΔΔG\s*[=:]\s*(-?\d+\.?\d*)",
        r"delta\s*G\s*[=:]\s*(-?\d+\.?\d*)",
        r"(-?\d+\.?\d*)\s*kcal\s*/?\s*mol",
        r"stability[:\s]+(-?\d+\.?\d*)",
        r"change[:\s]+(-?\d+\.?\d*)",
        r"predicted[:\s]+(-?\d+\.?\d*)",
    ]

    pred_value = None
    for pattern in patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            try:
                pred_value = float(match.group(1))
                break
            except ValueError:
                continue

    # If no pattern matched, try to find any number
    if pred_value is None:
        numbers = re.findall(r"-?\d+\.?\d+", generated_text)
        if numbers:
            # Take the first reasonable-looking ddG value
            for num_str in numbers:
                num = float(num_str)
                if -20 <= num <= 20:  # Typical ddG range in kcal/mol
                    pred_value = num
                    break

    if pred_value is None:
        return 0.0

    # Compute reward based on error
    error = abs(pred_value - gt_value)

    # Smooth reward function: exp(-error^2 / (2 * tolerance^2))
    # This gives ~1.0 for small errors and decays smoothly
    reward = float(torch.exp(torch.tensor(-error**2 / (2 * tolerance**2))))

    return reward


def compute_generic_reward(
    generated_text: str,
    ground_truth: str,
    task_type: Optional[str] = None,
) -> float:
    """Compute a generic reward by attempting to match task type or using text similarity.

    This is a fallback reward function when the task type is not explicitly known.
    It attempts to detect the task type from the content and use the appropriate
    reward function.

    Args:
        generated_text: Model-generated text.
        ground_truth: Expected output/ground truth.
        task_type: Optional task type hint.

    Returns:
        Reward value between 0 and 1.
    """
    # Try to detect task type from ground truth content
    if re.search(r"GO:\d{7}", ground_truth):
        return compute_go_reward(generated_text, ground_truth)

    if any(kw in ground_truth.lower() for kw in ["interact", "binding", "yes", "no"]):
        return compute_ppi_reward(generated_text, ground_truth)

    if any(kw in ground_truth.lower() for kw in ["kcal", "ddg", "stability"]):
        return compute_stability_reward(generated_text, ground_truth)

    # Fallback: simple text matching
    gen_lower = generated_text.lower().strip()
    gt_lower = ground_truth.lower().strip()

    if gen_lower == gt_lower:
        return 1.0

    # Partial match based on common words
    gen_words = set(gen_lower.split())
    gt_words = set(gt_lower.split())

    if not gt_words:
        return 0.0

    overlap = len(gen_words & gt_words)
    jaccard = overlap / len(gen_words | gt_words) if gen_words else 0.0

    return jaccard


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

        # Validate dependencies
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers is required. Install with: pip install transformers"
            )

    def setup(self) -> None:
        """Set up model, tokenizer, dataset, and GRPO configuration.

        This method must be called before train(). It:
        1. Initializes logging (wandb if enabled)
        2. Loads tokenizer and model
        3. Sets up LoRA adapters if configured
        4. Creates reference model for KL penalty
        5. Loads datasets
        6. Configures optimizer and scheduler
        7. Sets up reward function
        """
        log.info("Setting up GRPO trainer...")

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

        # Initialize logging
        self._setup_logging()

        # Load GRPO config
        self.grpo_config = get_grpo_config(self.cfg)
        log.info(f"GRPO config: {self.grpo_config}")

        # Load tokenizer
        self._load_tokenizer()

        # Load model
        self._load_model()

        # Create reference model for KL penalty (if enabled)
        if self.grpo_config["use_kl_penalty"]:
            self._create_reference_model()

        # Load datasets
        self._load_datasets()

        # Set up optimizer and scheduler
        self._setup_optimizer()

        # Set up reward function
        self._setup_reward_function()

        log.info("GRPO trainer setup complete")

    def _setup_logging(self) -> None:
        """Set up wandb and other logging."""
        logging_cfg = self.cfg.get("logging", {})

        if logging_cfg.get("wandb", {}).get("enabled", False) and HAS_WANDB:
            wandb.init(
                project=logging_cfg.wandb.get("project", "protein_llm"),
                name=logging_cfg.wandb.get("name", f"grpo_{self.cfg.get('experiment_name', 'run')}"),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=["grpo", "protein"],
            )
            log.info("Wandb logging initialized for GRPO training")

    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        model_path = self.cfg.model.path
        log.info(f"Loading tokenizer from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

    def _load_model(self) -> None:
        """Load the model with optional quantization and LoRA."""
        from .sft_trainer import get_qlora_config, get_quantization_config

        model_path = self.cfg.model.path
        use_quantization = self.cfg.training.get("quantization", {}).get("enabled", False)

        log.info(f"Loading model from: {model_path}")
        log.info(f"Using quantization: {use_quantization}")

        # Get quantization config
        quantization_config = get_quantization_config(self.cfg) if use_quantization else None

        # Load base model
        if quantization_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            if HAS_PEFT:
                self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Apply LoRA if configured
        if self.cfg.training.get("lora", {}) and HAS_PEFT:
            lora_config = get_qlora_config(self.cfg)
            self.model = get_peft_model(self.model, lora_config)
            log.info("LoRA configuration applied")
            self.model.print_trainable_parameters()

        self.model.train()

    def _create_reference_model(self) -> None:
        """Create a frozen reference model for KL penalty computation."""
        log.info("Creating reference model for KL penalty...")

        # Clone the model weights
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.ref_model.eval()
        log.info("Reference model created and frozen")

    def _load_datasets(self) -> None:
        """Load training and validation datasets."""
        from src.data.mol_instructions import MolInstructionsDataset

        data_cfg = self.cfg.data

        log.info("Loading training dataset...")
        self.train_dataset = MolInstructionsDataset(
            split="train",
            dataset_name=data_cfg.get("source", "zjunlp/Mol-Instructions"),
            subset=data_cfg.get("subset", "Protein-oriented Instructions"),
            cache_dir=data_cfg.get("paths", {}).get("raw"),
            max_seq_length=self.cfg.training.get("max_seq_length", 2048),
        )
        log.info(f"Training dataset loaded: {len(self.train_dataset)} samples")

        log.info("Loading validation dataset...")
        self.eval_dataset = MolInstructionsDataset(
            split="validation",
            dataset_name=data_cfg.get("source", "zjunlp/Mol-Instructions"),
            subset=data_cfg.get("subset", "Protein-oriented Instructions"),
            cache_dir=data_cfg.get("paths", {}).get("raw"),
            max_seq_length=self.cfg.training.get("max_seq_length", 2048),
        )
        log.info(f"Validation dataset loaded: {len(self.eval_dataset)} samples")

    def _setup_optimizer(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        lr = self.cfg.training.get("lr", 5e-6)
        weight_decay = self.cfg.training.get("weight_decay", 0.01)

        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        log.info(f"Setting up optimizer with lr={lr}, weight_decay={weight_decay}")
        log.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        warmup_steps = self.cfg.training.get("warmup_steps", 100)
        total_steps = (
            len(self.train_dataset)
            // self.cfg.training.get("batch_size", 4)
            // self.cfg.training.get("gradient_accumulation_steps", 8)
            * self.cfg.training.get("epochs", 1)
        )

        # Linear warmup then cosine decay
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps + 1)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_reward_function(self) -> None:
        """Set up the reward function based on task type."""
        task_type = self.cfg.data.get("task", "go_prediction")

        try:
            self.reward_fn = get_reward_function(task_type)
            log.info(f"Using reward function for task: {task_type}")
        except ValueError:
            log.warning(f"Unknown task type: {task_type}. Using generic reward function.")
            self.reward_fn = compute_generic_reward

    def _generate_completions(
        self,
        prompts: List[str],
        num_completions: int,
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """Generate multiple completions for each prompt.

        Args:
            prompts: List of input prompts.
            num_completions: Number of completions to generate per prompt.

        Returns:
            Tuple of:
                - List of lists of generated completions
                - Tensor of log probabilities for each completion
        """
        all_completions = []
        all_log_probs = []

        generation_config = GenerationConfig(
            max_new_tokens=self.grpo_config["max_new_tokens"],
            temperature=self.grpo_config["temperature"],
            top_p=self.grpo_config["top_p"],
            do_sample=self.grpo_config["do_sample"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        for prompt in prompts:
            completions = []
            log_probs_list = []

            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.training.get("max_seq_length", 2048) - self.grpo_config["max_new_tokens"],
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]

            for _ in range(num_completions):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                    )

                # Extract generated tokens (excluding prompt)
                generated_ids = outputs.sequences[0, prompt_length:]
                completion = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                completions.append(completion)

                # Compute log probability of the completion
                # Use the scores from generation
                if hasattr(outputs, "scores") and outputs.scores:
                    log_prob = self._compute_sequence_log_prob(
                        inputs["input_ids"],
                        outputs.sequences,
                        prompt_length,
                    )
                else:
                    log_prob = torch.tensor(0.0, device=self.device)

                log_probs_list.append(log_prob)

            all_completions.append(completions)
            all_log_probs.append(torch.stack(log_probs_list))

        return all_completions, torch.stack(all_log_probs)

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

    def _compute_rewards(
        self,
        completions: List[List[str]],
        ground_truths: List[str],
    ) -> torch.Tensor:
        """Compute rewards for all completions.

        Args:
            completions: List of lists of completions (one list per prompt).
            ground_truths: List of ground truth responses.

        Returns:
            Tensor of rewards with shape (batch_size, group_size).
        """
        rewards = []

        for prompt_completions, ground_truth in zip(completions, ground_truths):
            prompt_rewards = []
            for completion in prompt_completions:
                reward = self.reward_fn(completion, ground_truth)
                prompt_rewards.append(reward)
            rewards.append(torch.tensor(prompt_rewards, device=self.device))

        return torch.stack(rewards)

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

    def _training_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Batch of data containing prompts and ground truth.

        Returns:
            Dictionary of metrics from this step.
        """
        # Extract batch data
        prompts = batch["formatted_prompt"] if "formatted_prompt" in batch else batch.get("instruction", [])
        ground_truths = batch.get("response", batch.get("output", []))

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

        group_size = self.grpo_config["group_size"]

        # Generate completions
        completions, log_probs = self._generate_completions(prompts, group_size)

        # Compute rewards
        rewards = self._compute_rewards(completions, ground_truths)

        # Compute advantages
        advantages = self._compute_advantages(rewards)

        # Compute policy gradient loss
        # Loss = -E[advantage * log_prob]
        pg_loss = -(advantages * log_probs).mean()

        # Add KL penalty if enabled (not for DAPO)
        if self.grpo_config["use_kl_penalty"]:
            # Note: Full KL computation would require tokenizing completions
            # For simplicity, we use a simplified version here
            kl_penalty = torch.tensor(0.0, device=self.device)  # Placeholder
            loss = pg_loss + self.grpo_config["kl_coef"] * kl_penalty
        else:
            loss = pg_loss
            kl_penalty = torch.tensor(0.0)

        # Backward pass
        loss.backward()

        # Gradient clipping
        max_grad_norm = self.cfg.training.get("max_grad_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "reward_std": rewards.std().item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self) -> Dict[str, Any]:
        """Run GRPO training loop.

        Returns:
            Dictionary of final training metrics.
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        log.info("Starting GRPO training...")
        log.info(f"  Epochs: {self.cfg.training.get('epochs', 1)}")
        log.info(f"  Batch size: {self.cfg.training.get('batch_size', 4)}")
        log.info(f"  Group size: {self.grpo_config['group_size']}")
        log.info(f"  Learning rate: {self.cfg.training.get('lr', 5e-6)}")
        log.info(f"  KL penalty: {self.grpo_config['use_kl_penalty']}")
        log.info(f"  Normalize advantages: {self.grpo_config['normalize_advantages']}")

        # Create dataloader
        batch_size = self.cfg.training.get("batch_size", 4)
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        num_epochs = self.cfg.training.get("epochs", 1)
        gradient_accumulation_steps = self.cfg.training.get("gradient_accumulation_steps", 8)
        logging_steps = self.cfg.training.get("logging_steps", 10)
        save_steps = self.cfg.training.get("save_steps", 100)
        eval_steps = self.cfg.training.get("eval_steps", 50)

        all_metrics = []

        for epoch in range(num_epochs):
            self.epoch = epoch
            log.info(f"Epoch {epoch + 1}/{num_epochs}")

            epoch_metrics = []
            self.model.train()

            for step, batch in enumerate(dataloader):
                metrics = self._training_step(batch)
                epoch_metrics.append(metrics)

                # Logging
                if self.global_step % logging_steps == 0:
                    avg_metrics = {
                        k: sum(m[k] for m in epoch_metrics[-logging_steps:]) / len(epoch_metrics[-logging_steps:])
                        for k in metrics.keys()
                    }
                    log.info(
                        f"Step {self.global_step}: "
                        f"loss={avg_metrics['loss']:.4f}, "
                        f"reward={avg_metrics['mean_reward']:.4f}"
                    )

                    if HAS_WANDB and wandb.run is not None:
                        wandb.log(avg_metrics, step=self.global_step)

                # Evaluation
                if self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate()
                    log.info(f"Eval metrics: {eval_metrics}")

                    if HAS_WANDB and wandb.run is not None:
                        wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()}, step=self.global_step)

                # Save checkpoint
                if self.global_step % save_steps == 0:
                    checkpoint_dir = Path(self.cfg.get("paths", {}).get("checkpoint_dir", "./checkpoints"))
                    self.save_checkpoint(checkpoint_dir / f"checkpoint-{self.global_step}")

            all_metrics.extend(epoch_metrics)

        # Save final checkpoint
        checkpoint_dir = Path(self.cfg.get("paths", {}).get("checkpoint_dir", "./checkpoints"))
        self.save_checkpoint(checkpoint_dir / "final")

        # Compute final metrics
        final_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0].keys()
        }

        log.info(f"Training completed. Final metrics: {final_metrics}")

        return final_metrics

    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """Run evaluation on validation set.

        Args:
            num_samples: Number of samples to evaluate on.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataset is None:
            return {}

        self.model.eval()

        # Sample from eval dataset
        num_samples = min(num_samples, len(self.eval_dataset))
        eval_rewards = []

        with torch.no_grad():
            for i in range(num_samples):
                sample = self.eval_dataset[i]

                # Generate single completion for evaluation
                prompt = sample.get("formatted_prompt", sample.get("instruction", ""))
                ground_truth = sample.get("response", sample.get("output", ""))

                # Generate completion
                inputs = self.tokenizer(
                    prompt,
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

                # Compute reward
                reward = self.reward_fn(completion, ground_truth)
                eval_rewards.append(reward)

        self.model.train()

        return {
            "mean_reward": sum(eval_rewards) / len(eval_rewards),
            "max_reward": max(eval_rewards),
            "min_reward": min(eval_rewards),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint.

        Args:
            path: Directory path to save the checkpoint.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        log.info(f"Saving checkpoint to: {path}")

        # Save model (LoRA adapters if using PEFT)
        if HAS_PEFT and isinstance(self.model, PeftModel):
            self.model.save_pretrained(path / "adapter")
        else:
            self.model.save_pretrained(path / "model")

        # Save tokenizer
        self.tokenizer.save_pretrained(path / "tokenizer")

        # Save training state
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

        # Save config
        OmegaConf.save(self.cfg, path / "config.yaml")

        log.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint.

        Args:
            path: Directory path to load the checkpoint from.
        """
        path = Path(path)

        log.info(f"Loading checkpoint from: {path}")

        # Load training state
        state = torch.load(path / "training_state.pt", map_location=self.device)
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
