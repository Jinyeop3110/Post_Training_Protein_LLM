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
    compute_proteinlm_bench_reward,
    compute_stability_reward,
    get_reward_function,
)

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

        # Load datasets
        self._load_datasets()

        # Set up optimizer and scheduler (with differential LR)
        self._setup_optimizer()

        # Set up reward function
        self._setup_reward_function()

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
        task_normalized = task_type.lower().replace("-", "_").replace(" ", "_")
        self._is_esmfold_reward = task_normalized in esmfold_tasks

        try:
            self.reward_fn = get_reward_function(task_type)
            log.info(f"Using reward function for task: {task_type}")
        except ValueError:
            log.warning(f"Unknown task type: {task_type}. Using generic reward function.")
            self.reward_fn = compute_generic_reward
            self._is_esmfold_reward = False

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
        )

        for i, prompt in enumerate(prompts):
            # Multimodal path: use ProteinLLM.generate()
            if (
                self.protein_llm is not None
                and protein_sequences is not None
                and protein_sequences[i]
            ):
                protein_seq = protein_sequences[i]
                batch_proteins = [protein_seq] * num_completions
                batch_prompts = [prompt] * num_completions

                with torch.no_grad():
                    texts, token_ids, input_len = self.protein_llm.generate(
                        protein_sequences=batch_proteins,
                        prompt=batch_prompts,
                        return_token_ids=True,
                        **gen_kwargs,
                    )

                all_completions.append(texts)
                all_generated_ids.append(
                    [token_ids[j] for j in range(token_ids.shape[0])]
                )

            else:
                # Text-only path: use model.generate() directly
                inputs = self.tokenizer(
                    prompt,
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
            # For multimodal, must include chat template + protein placeholder
            # so prepare_inputs() finds the placeholder during log prob forward.
            if (
                self.protein_llm is not None
                and protein_sequences is not None
                and protein_sequences[i]
            ):
                from src.data.mol_instructions import DEFAULT_SYSTEM_PROMPT
                from src.models.multimodal_llm import PROTEIN_PLACEHOLDER

                user_content = f"{prompt}\n\n{PROTEIN_PLACEHOLDER}"
                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                wrapped_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                prompt_inputs = self.tokenizer(
                    wrapped_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=(
                        self.cfg.training.get("max_seq_length", 2048)
                        - self.grpo_config["max_new_tokens"]
                    ),
                )
            else:
                prompt_inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=(
                        self.cfg.training.get("max_seq_length", 2048)
                        - self.grpo_config["max_new_tokens"]
                    ),
                )
            all_prompt_ids.append(prompt_inputs["input_ids"].to(self.device))

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

        if self.protein_llm is not None and protein_sequence is not None:
            # Multimodal: encode protein + prepend prefix embeddings
            prepared = self.protein_llm.prepare_inputs(
                protein_sequences=[protein_sequence],
                text_input_ids=full_ids,
                text_attention_mask=attention_mask,
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

    def _compute_rewards(
        self,
        completions: List[List[str]],
        ground_truths: List[str],
        protein_sequences: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute rewards for all completions with supplementary metrics.

        Args:
            completions: List of lists of completions (one list per prompt).
            ground_truths: List of ground truth responses.
            protein_sequences: Optional protein sequences for ESMFold reward.

        Returns:
            Tuple of:
                - Tensor of rewards with shape (batch_size, group_size).
                - Dict of supplementary metric lists (one value per completion).
        """
        rewards = []
        all_metrics: Dict[str, List[float]] = {}

        for idx, (prompt_completions, ground_truth) in enumerate(
            zip(completions, ground_truths)
        ):
            prompt_rewards = []
            for completion in prompt_completions:
                # ESMFold reward: prefer pre-computed metrics (JSON) over live fold
                if self._is_esmfold_reward:
                    gt_str = str(ground_truth)
                    if gt_str.strip().startswith("{"):
                        # Pre-computed pLDDT/pTM in ground truth JSON
                        second_arg = ground_truth
                    elif protein_sequences is not None:
                        # Live ESMFold: pass protein sequence
                        second_arg = protein_sequences[idx]
                    else:
                        second_arg = ground_truth
                else:
                    second_arg = ground_truth
                reward, metrics = self.reward_fn(
                    completion, second_arg, detailed=True
                )
                prompt_rewards.append(reward)
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
        # Also clear the flag on model config
        if hasattr(base, "config"):
            base.config.gradient_checkpointing = False

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
        # Extract batch data
        prompts = batch.get(
            "inference_prompt",
            batch.get("formatted_prompt", batch.get("instruction", [])),
        )
        protein_sequences = batch.get("protein_sequence", None)

        # Task-aware ground truth extraction:
        # - stability/ddg: use metadata.ddG (float) instead of text output
        # - esmfold/structure: pre-computed pLDDT from metadata, or protein_sequences
        # - default (go_prediction, etc.): use text response
        task = self.cfg.data.get("task", "go_prediction").lower()
        if task in ("stability", "ddg"):
            metadata_list = batch.get("metadata", [])
            if metadata_list and isinstance(metadata_list, list):
                ground_truths = [
                    m.get("ddG", m.get("ddg", "")) if isinstance(m, dict) else ""
                    for m in metadata_list
                ]
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
        elif task in ("proteinlm_bench", "protein_lm_bench", "multiple_choice"):
            # ProteinLMBench: ground truth is "option N" from metadata or response
            metadata_list = batch.get("metadata", [])
            if metadata_list and isinstance(metadata_list, list) and isinstance(metadata_list[0], dict):
                ground_truths = [m.get("answer", "") for m in metadata_list]
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

        # Step 1: Generate completions (no grad)
        # Switch to eval mode so gradient checkpointing deactivates
        # (HF check: self.gradient_checkpointing AND self.training).
        # This allows generation to use KV cache.
        self.model.eval()

        completions, all_generated_ids, all_prompt_ids = (
            self._generate_completions(prompts, protein_sequences, group_size)
        )

        # Switch back to train mode for differentiable forward pass
        self.model.train()

        # Step 2: Compute rewards
        rewards, reward_metrics = self._compute_rewards(
            completions, ground_truths, protein_sequences=protein_sequences
        )

        # Step 3: Compute advantages (detached from reward computation)
        advantages = self._compute_advantages(rewards).detach()

        # Step 4: Re-compute log probs WITH gradients (differentiable forward)
        diff_log_probs = []
        for prompt_idx in range(len(prompts)):
            prompt_log_probs = []
            protein_seq = (
                protein_sequences[prompt_idx]
                if protein_sequences is not None
                else None
            )
            for comp_idx in range(group_size):
                log_prob = self._compute_policy_log_probs(
                    all_prompt_ids[prompt_idx],
                    all_generated_ids[prompt_idx][comp_idx],
                    protein_sequence=protein_seq,
                )
                prompt_log_probs.append(log_prob)
            diff_log_probs.append(torch.stack(prompt_log_probs))
        log_probs = torch.stack(diff_log_probs)  # (batch_size, group_size)

        # Step 5: Policy gradient loss = -E[advantage * log_prob]
        pg_loss = -(advantages * log_probs).mean()

        # KL penalty (stubbed for DAPO; use_kl_penalty defaults to False)
        if self.grpo_config["use_kl_penalty"]:
            kl_penalty = torch.tensor(0.0, device=self.device)
            loss = pg_loss + self.grpo_config["kl_coef"] * kl_penalty
        else:
            loss = pg_loss
            kl_penalty = torch.tensor(0.0)

        step_metrics = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "reward_std": rewards.std().item(),
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
                loss, metrics = self._training_step(batch)

                # Scale loss for gradient accumulation
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()
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
                        avg_metrics = {
                            k: sum(m[k] for m in recent) / len(recent)
                            for k in metrics.keys()
                        }
                        avg_metrics["lr"] = self.scheduler.get_last_lr()[0]
                        log.info(
                            f"Step {self.global_step}: "
                            f"loss={avg_metrics['loss']:.4f}, "
                            f"reward={avg_metrics['mean_reward']:.4f}, "
                            f"lr={avg_metrics['lr']:.2e}"
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
                            k: sum(m[k] for m in recent) / max(len(recent), 1)
                            for k in metrics.keys()
                        }
                        self.save_checkpoint(
                            path=checkpoint_dir / f"checkpoint-{self.global_step}",
                            metrics=ckpt_metrics,
                        )

            all_metrics.extend(epoch_metrics)

        # Compute final metrics
        final_metrics = {}
        if all_metrics:
            final_metrics = {
                k: sum(m[k] for m in all_metrics) / len(all_metrics)
                for k in all_metrics[0].keys()
            }

        # Save final checkpoint (main process only)
        if self.is_main_process:
            self.save_checkpoint(metrics=final_metrics)
            log.info(f"Training completed. Final metrics: {final_metrics}")

        return final_metrics

    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """Run evaluation on validation set using ProteinLLM when available.

        Args:
            num_samples: Number of samples to evaluate on.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataset is None:
            return {}

        # eval mode deactivates gradient checkpointing → KV cache works
        self.model.eval()

        num_samples = min(num_samples, len(self.eval_dataset))
        eval_rewards = []

        with torch.no_grad():
            for i in range(num_samples):
                sample = self.eval_dataset[i]

                prompt = sample.get(
                    "inference_prompt",
                    sample.get("formatted_prompt", sample.get("instruction", "")),
                )
                protein_seq = sample.get("protein_sequence", None)

                # Task-aware ground truth extraction (mirrors _training_step)
                task = self.cfg.data.get("task", "go_prediction").lower()
                metadata = sample.get("metadata", {})
                if task in ("stability", "ddg") and isinstance(metadata, dict) and "ddG" in metadata:
                    ground_truth = metadata["ddG"]
                elif (
                    task in ("esmfold", "structure", "structure_prediction", "fold_quality")
                    and isinstance(metadata, dict) and "plddt" in metadata
                ):
                    ground_truth = json.dumps({
                        "plddt": metadata.get("plddt", 0),
                        "ptm": metadata.get("ptm", 0),
                    })
                else:
                    ground_truth = sample.get("response", sample.get("output", ""))

                # Multimodal: use ProteinLLM.generate()
                if self.protein_llm is not None and protein_seq:
                    texts = self.protein_llm.generate(
                        protein_sequences=[protein_seq],
                        prompt=[prompt],
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                    )
                    completion = texts[0]
                else:
                    # Text-only fallback
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

                # ESMFold reward: pass pre-computed metrics or protein sequence
                if self._is_esmfold_reward:
                    gt_str = str(ground_truth)
                    if gt_str.strip().startswith("{"):
                        second_arg = ground_truth
                    elif protein_seq:
                        second_arg = protein_seq
                    else:
                        second_arg = ground_truth
                    reward = self.reward_fn(completion, second_arg)
                else:
                    reward = self.reward_fn(completion, ground_truth)
                eval_rewards.append(reward)

        # train() re-activates gradient checkpointing (checked via self.training)
        self.model.train()

        return {
            "mean_reward": sum(eval_rewards) / len(eval_rewards),
            "max_reward": max(eval_rewards),
            "min_reward": min(eval_rewards),
        }

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
