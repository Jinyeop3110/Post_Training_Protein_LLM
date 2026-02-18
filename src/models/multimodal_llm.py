"""
Multimodal Protein-LLM Model

This module implements the ProteinLLM class that combines:
1. ESM-2 protein encoder (frozen)
2. Attention/Mean pooling to create fixed-size prefix tokens
3. MLP projector to map protein embeddings to LLM hidden space
4. Large Language Model (Qwen-2.5-7B with QLoRA)

Architecture (Prefix Token approach):
    Protein Sequence -> ESM-2 (frozen) -> [B, L, 1280]
                            |
                  AttentionPooling -> [B, N, 1280] (N = num_prefix_tokens)
                            |
                  MLPProjector -> [B, N, 4096]
                            |
                  Prepend as prefix tokens to LLM input
                            |
                  LLM (Qwen-2.5-7B with QLoRA) -> Generate response
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from src.models.protein_encoder import ESMProteinEncoder
from src.models.pooling import get_pooling, AttentionPooling, BasePooling
from src.models.projector import MLPProjector, get_projector

logger = logging.getLogger(__name__)


class ProteinLLM(nn.Module):
    """
    Multimodal Protein-LLM model combining ESM-2 encoder with an LLM.

    This model encodes protein sequences using ESM-2, pools the embeddings
    into a fixed number of prefix tokens, projects them to the LLM's hidden
    dimension, and prepends them to the text input for the LLM.

    Args:
        llm_name: HuggingFace model path for the LLM.
        encoder_name: ESM-2 model name (e.g., "esm2_t33_650M_UR50D").
        num_prefix_tokens: Number of protein prefix tokens.
        pooling_type: Type of pooling ("attention" or "mean").
        projector_hidden_dim: Hidden dimension for the MLP projector.
        projector_num_layers: Number of layers in the MLP projector.
        projector_dropout: Dropout rate for the projector.
        freeze_encoder: Whether to freeze the ESM-2 encoder (always True).
        use_qlora: Whether to use QLoRA for the LLM.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        lora_target_modules: Target modules for LoRA.
        device: Device to load the model on.

    Example:
        >>> model = ProteinLLM(
        ...     llm_name="Qwen/Qwen2.5-7B-Instruct",
        ...     encoder_name="esm2_t33_650M_UR50D",
        ...     num_prefix_tokens=32,
        ...     use_qlora=True,
        ... )
        >>> protein_sequences = ["MKTAYIAKQRQISFVK", "MNIFEMLRIDEGLR"]
        >>> input_ids = tokenizer(["What is this protein?"], return_tensors="pt")
        >>> outputs = model(protein_sequences, input_ids["input_ids"], input_ids["attention_mask"])
    """

    # Default configuration values
    DEFAULT_LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
    DEFAULT_ENCODER_NAME = "esm2_t33_650M_UR50D"
    DEFAULT_NUM_PREFIX_TOKENS = 32
    DEFAULT_POOLING_TYPE = "attention"
    DEFAULT_PROJECTOR_HIDDEN_DIM = 2048
    DEFAULT_PROJECTOR_NUM_LAYERS = 2
    DEFAULT_PROJECTOR_DROPOUT = 0.1
    DEFAULT_LORA_R = 8
    DEFAULT_LORA_ALPHA = 16
    DEFAULT_LORA_DROPOUT = 0.05
    DEFAULT_LORA_TARGET_MODULES = ["k_proj", "v_proj"]

    # ESM-2 model dimensions
    ESM_EMBED_DIMS = {
        "esm2_t6_8M_UR50D": 320,
        "esm2_t12_35M_UR50D": 480,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t36_3B_UR50D": 2560,
        "esm2_t48_15B_UR50D": 5120,
    }

    def __init__(
        self,
        llm_name: str = DEFAULT_LLM_NAME,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        num_prefix_tokens: int = DEFAULT_NUM_PREFIX_TOKENS,
        pooling_type: str = DEFAULT_POOLING_TYPE,
        projector_hidden_dim: int = DEFAULT_PROJECTOR_HIDDEN_DIM,
        projector_num_layers: int = DEFAULT_PROJECTOR_NUM_LAYERS,
        projector_dropout: float = DEFAULT_PROJECTOR_DROPOUT,
        freeze_encoder: bool = True,
        use_qlora: bool = True,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_target_modules: Optional[List[str]] = None,
        device: Optional[str] = None,
        load_llm: bool = True,
        load_encoder: bool = True,
    ):
        super().__init__()

        self.llm_name = llm_name
        self.encoder_name = encoder_name
        self.num_prefix_tokens = num_prefix_tokens
        self.pooling_type = pooling_type
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_num_layers = projector_num_layers
        self.projector_dropout = projector_dropout
        self.freeze_encoder = freeze_encoder  # Always True for ESM-2
        self.use_qlora = use_qlora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or self.DEFAULT_LORA_TARGET_MODULES

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Get encoder embedding dimension
        self.encoder_embed_dim = self.ESM_EMBED_DIMS.get(encoder_name, 1280)

        # Initialize components (may be lazy loaded)
        self.encoder: Optional[ESMProteinEncoder] = None
        self.pooling: Optional[BasePooling] = None
        self.projector: Optional[MLPProjector] = None
        self.llm: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.llm_hidden_size: int = 4096  # Default, will be updated when LLM is loaded

        # Load components
        if load_encoder:
            self._load_encoder()
            self._build_pooling()

        if load_llm:
            self._load_llm()
            self._build_projector()

    def _load_encoder(self) -> None:
        """Load the ESM-2 protein encoder."""
        logger.info(f"Loading ESM-2 encoder: {self.encoder_name}")

        self.encoder = ESMProteinEncoder(
            model_name=self.encoder_name,
            pooling="per_residue",  # Get full embeddings, pooling handled separately
            device=self.device,
        )

        # Force model loading to get embedding dim
        self.encoder_embed_dim = self.encoder.get_embedding_dim()

        # Freeze encoder parameters
        if self.freeze_encoder and self.encoder.model is not None:
            for param in self.encoder.model.parameters():
                param.requires_grad = False
            logger.info("ESM-2 encoder frozen")

    def _build_pooling(self) -> None:
        """Build the pooling module."""
        logger.info(f"Building {self.pooling_type} pooling with {self.num_prefix_tokens} output tokens")

        if self.pooling_type == "attention":
            self.pooling = get_pooling(
                "attention",
                embed_dim=self.encoder_embed_dim,
                num_output_tokens=self.num_prefix_tokens,
                num_heads=8,
                dropout=0.1,
                layer_norm=True,
            )
        elif self.pooling_type == "mean":
            self.pooling = get_pooling("mean", keepdim=True)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        self.pooling = self.pooling.to(self.device)

    def _load_llm(self) -> None:
        """Load the LLM with optional QLoRA configuration."""
        logger.info(f"Loading LLM: {self.llm_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_name,
            trust_remote_code=True,
            padding_side="left",
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization for QLoRA
        if self.use_qlora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT is required for QLoRA. Install with: pip install peft"
                )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            # Prepare model for k-bit training
            self.llm = prepare_model_for_kbit_training(self.llm)

            # Apply LoRA
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            self.llm = get_peft_model(self.llm, lora_config)
            logger.info("Applied QLoRA configuration")
            self.llm.print_trainable_parameters()
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Get LLM hidden size
        self.llm_hidden_size = self.llm.config.hidden_size
        logger.info(f"LLM hidden size: {self.llm_hidden_size}")

    def _build_projector(self) -> None:
        """Build the MLP projector to map encoder embeddings to LLM space."""
        logger.info(
            f"Building projector: {self.encoder_embed_dim} -> "
            f"{self.projector_hidden_dim} -> {self.llm_hidden_size}"
        )

        self.projector = MLPProjector(
            input_dim=self.encoder_embed_dim,
            hidden_dim=self.projector_hidden_dim,
            output_dim=self.llm_hidden_size,
            num_layers=self.projector_num_layers,
            activation="gelu",
            dropout=self.projector_dropout,
        ).to(self.device)

    def encode_protein(
        self,
        sequences: List[str],
        return_attention_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode protein sequences into prefix embeddings.

        This method:
        1. Encodes sequences with ESM-2 to get per-residue embeddings
        2. Pools embeddings into fixed-size prefix tokens
        3. Projects to LLM hidden dimension

        Args:
            sequences: List of protein sequences (amino acid strings).
            return_attention_mask: Whether to return attention mask for prefix tokens.

        Returns:
            If return_attention_mask is False:
                Tensor of shape [B, num_prefix_tokens, llm_hidden_size]
            If return_attention_mask is True:
                Tuple of (embeddings, attention_mask)
        """
        if self.encoder is None or self.pooling is None or self.projector is None:
            raise RuntimeError(
                "Model components not initialized. "
                "Ensure encoder, pooling, and projector are loaded."
            )

        # Get ESM-2 embeddings [B, L, encoder_embed_dim]
        with torch.no_grad():
            encoder_output = self.encoder.encode(sequences)
            embeddings = encoder_output["embeddings"]  # [B, L, D]

        # Pool to fixed size [B, num_prefix_tokens, encoder_embed_dim]
        pooled = self.pooling(embeddings)

        # Project to LLM space [B, num_prefix_tokens, llm_hidden_size]
        projected = self.projector(pooled)

        if return_attention_mask:
            # All prefix tokens are valid
            batch_size = projected.shape[0]
            num_tokens = projected.shape[1]
            attention_mask = torch.ones(
                batch_size, num_tokens,
                dtype=torch.long,
                device=projected.device,
            )
            return projected, attention_mask

        return projected

    def prepare_inputs(
        self,
        protein_sequences: List[str],
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare combined inputs by prepending protein prefix tokens to text.

        This method:
        1. Encodes protein sequences into prefix embeddings
        2. Gets text embeddings from the LLM
        3. Concatenates prefix embeddings before text embeddings
        4. Adjusts attention mask and labels accordingly

        Args:
            protein_sequences: List of protein sequences.
            text_input_ids: Text input token IDs [B, T].
            text_attention_mask: Text attention mask [B, T].
            labels: Optional labels for training [B, T].

        Returns:
            Dictionary containing:
                - inputs_embeds: Combined embeddings [B, N + T, D]
                - attention_mask: Combined attention mask [B, N + T]
                - labels: Combined labels [B, N + T] (if provided)
                - position_ids: Position IDs [B, N + T]
        """
        batch_size = text_input_ids.shape[0]
        text_seq_len = text_input_ids.shape[1]
        device = text_input_ids.device

        # Get protein prefix embeddings [B, N, D]
        protein_embeds, protein_mask = self.encode_protein(
            protein_sequences, return_attention_mask=True
        )
        num_prefix_tokens = protein_embeds.shape[1]

        # Get text embeddings from LLM [B, T, D]
        if self.use_qlora and hasattr(self.llm, "get_base_model"):
            # For PEFT models, get the base model's embeddings
            base_model = self.llm.get_base_model()
            text_embeds = base_model.model.embed_tokens(text_input_ids)
        else:
            text_embeds = self.llm.model.embed_tokens(text_input_ids)

        # Concatenate: [protein_prefix | text_tokens]
        # Shape: [B, N + T, D]
        combined_embeds = torch.cat([protein_embeds, text_embeds], dim=1)

        # Combine attention masks [B, N + T]
        combined_attention_mask = torch.cat(
            [protein_mask.to(device), text_attention_mask],
            dim=1,
        )

        # Create position IDs [B, N + T]
        total_seq_len = num_prefix_tokens + text_seq_len
        position_ids = torch.arange(
            total_seq_len, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        result = {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_attention_mask,
            "position_ids": position_ids,
        }

        # Handle labels if provided
        if labels is not None:
            # Prepend -100 for prefix tokens (ignore in loss)
            prefix_labels = torch.full(
                (batch_size, num_prefix_tokens),
                fill_value=-100,
                dtype=labels.dtype,
                device=device,
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)
            result["labels"] = combined_labels

        return result

    def forward(
        self,
        protein_sequences: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            protein_sequences: List of protein sequences.
            input_ids: Text input token IDs [B, T].
            attention_mask: Text attention mask [B, T].
            labels: Labels for language modeling loss [B, T].
            **kwargs: Additional arguments passed to the LLM.

        Returns:
            Dictionary containing:
                - loss: Language modeling loss (if labels provided)
                - logits: Output logits [B, N + T, vocab_size]
        """
        # Prepare combined inputs
        prepared = self.prepare_inputs(
            protein_sequences=protein_sequences,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            labels=labels,
        )

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            position_ids=prepared["position_ids"],
            labels=prepared.get("labels"),
            **kwargs,
        )

        return {
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        protein_sequences: List[str],
        prompt: Union[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generate_kwargs,
    ) -> List[str]:
        """
        Generate text responses given protein sequences and prompts.

        Args:
            protein_sequences: List of protein sequences.
            prompt: Text prompt(s) for generation.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling parameter.
            do_sample: Whether to use sampling (vs greedy decoding).
            **generate_kwargs: Additional arguments for generate().

        Returns:
            List of generated text responses.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        # Handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt] * len(protein_sequences)

        # Tokenize prompts
        text_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Prepare combined inputs
        prepared = self.prepare_inputs(
            protein_sequences=protein_sequences,
            text_input_ids=text_inputs["input_ids"],
            text_attention_mask=text_inputs["attention_mask"],
        )

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )

        # Decode outputs (skip the input tokens)
        # The prefix tokens are handled via inputs_embeds, so output starts fresh
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        return generated_texts

    @classmethod
    def from_config(cls, cfg: Union[DictConfig, Dict[str, Any]]) -> "ProteinLLM":
        """
        Create ProteinLLM from a Hydra/OmegaConf configuration.

        Expected config structure:
            model:
                path: "Qwen/Qwen2.5-7B-Instruct"
                architecture:
                    hidden_size: 4096
            encoder:
                model_name: "esm2_t33_650M_UR50D"
                embedding_dim: 1280
                pooling:
                    method: "attention"
                projector:
                    hidden_dim: 2048
                    num_layers: 2
                    dropout: 0.1
            training:
                quantization:
                    enabled: true
                lora:
                    r: 8
                    alpha: 16
                    dropout: 0.05
                    target_modules: [k_proj, v_proj]

        Args:
            cfg: Hydra/OmegaConf configuration object.

        Returns:
            Configured ProteinLLM instance.
        """
        # Convert to dict if necessary
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        # Extract model config
        model_cfg = cfg.get("model", {})
        encoder_cfg = cfg.get("encoder", {})
        training_cfg = cfg.get("training", {})

        # Extract pooling config
        pooling_cfg = encoder_cfg.get("pooling", {})
        num_prefix_tokens = pooling_cfg.get("num_output_tokens", cls.DEFAULT_NUM_PREFIX_TOKENS)

        # Extract projector config
        projector_cfg = encoder_cfg.get("projector", {})

        # Extract LoRA config
        lora_cfg = training_cfg.get("lora", {})
        quantization_cfg = training_cfg.get("quantization", {})

        return cls(
            llm_name=model_cfg.get("path", cls.DEFAULT_LLM_NAME),
            encoder_name=encoder_cfg.get("model_name", cls.DEFAULT_ENCODER_NAME),
            num_prefix_tokens=num_prefix_tokens,
            pooling_type=pooling_cfg.get("method", cls.DEFAULT_POOLING_TYPE),
            projector_hidden_dim=projector_cfg.get("hidden_dim", cls.DEFAULT_PROJECTOR_HIDDEN_DIM),
            projector_num_layers=projector_cfg.get("num_layers", cls.DEFAULT_PROJECTOR_NUM_LAYERS),
            projector_dropout=projector_cfg.get("dropout", cls.DEFAULT_PROJECTOR_DROPOUT),
            freeze_encoder=encoder_cfg.get("freeze", True),
            use_qlora=quantization_cfg.get("enabled", True),
            lora_r=lora_cfg.get("r", cls.DEFAULT_LORA_R),
            lora_alpha=lora_cfg.get("alpha", cls.DEFAULT_LORA_ALPHA),
            lora_dropout=lora_cfg.get("dropout", cls.DEFAULT_LORA_DROPOUT),
            lora_target_modules=lora_cfg.get("target_modules", cls.DEFAULT_LORA_TARGET_MODULES),
        )

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Save the model to a directory.

        This saves:
        - Model configuration (config.json)
        - Pooling module weights (pooling.pt)
        - Projector weights (projector.pt)
        - LoRA adapter weights (if applicable)

        Note: ESM-2 encoder weights are not saved as they are frozen
        and loaded from the pretrained checkpoint.

        Args:
            path: Directory path to save the model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "llm_name": self.llm_name,
            "encoder_name": self.encoder_name,
            "num_prefix_tokens": self.num_prefix_tokens,
            "pooling_type": self.pooling_type,
            "projector_hidden_dim": self.projector_hidden_dim,
            "projector_num_layers": self.projector_num_layers,
            "projector_dropout": self.projector_dropout,
            "freeze_encoder": self.freeze_encoder,
            "use_qlora": self.use_qlora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "encoder_embed_dim": self.encoder_embed_dim,
            "llm_hidden_size": self.llm_hidden_size,
        }

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save pooling weights
        if self.pooling is not None:
            torch.save(self.pooling.state_dict(), path / "pooling.pt")

        # Save projector weights
        if self.projector is not None:
            torch.save(self.projector.state_dict(), path / "projector.pt")

        # Save LoRA adapter weights
        if self.use_qlora and self.llm is not None:
            adapter_path = path / "adapter"
            self.llm.save_pretrained(adapter_path)

        logger.info(f"Model saved to {path}")

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
        load_llm: bool = True,
        load_encoder: bool = True,
    ) -> "ProteinLLM":
        """
        Load a model from a saved directory.

        Args:
            path: Directory path containing the saved model.
            device: Device to load the model on.
            load_llm: Whether to load the LLM (can skip for testing).
            load_encoder: Whether to load the encoder (can skip for testing).

        Returns:
            Loaded ProteinLLM instance.
        """
        path = Path(path)

        # Load configuration
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Create model instance
        model = cls(
            llm_name=config["llm_name"],
            encoder_name=config["encoder_name"],
            num_prefix_tokens=config["num_prefix_tokens"],
            pooling_type=config["pooling_type"],
            projector_hidden_dim=config["projector_hidden_dim"],
            projector_num_layers=config["projector_num_layers"],
            projector_dropout=config["projector_dropout"],
            freeze_encoder=config["freeze_encoder"],
            use_qlora=config["use_qlora"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            lora_target_modules=config["lora_target_modules"],
            device=device,
            load_llm=load_llm,
            load_encoder=load_encoder,
        )

        # Load pooling weights
        pooling_path = path / "pooling.pt"
        if pooling_path.exists() and model.pooling is not None:
            model.pooling.load_state_dict(torch.load(pooling_path, map_location=model.device))

        # Load projector weights
        projector_path = path / "projector.pt"
        if projector_path.exists() and model.projector is not None:
            model.projector.load_state_dict(torch.load(projector_path, map_location=model.device))

        # Load LoRA adapter weights
        adapter_path = path / "adapter"
        if adapter_path.exists() and model.use_qlora and model.llm is not None:
            if PEFT_AVAILABLE:
                model.llm = PeftModel.from_pretrained(
                    model.llm.get_base_model() if hasattr(model.llm, "get_base_model") else model.llm,
                    adapter_path,
                )

        logger.info(f"Model loaded from {path}")
        return model

    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get the number of trainable parameters for each component.

        Returns:
            Dictionary with component names and trainable parameter counts.
        """
        def count_params(module: nn.Module) -> Tuple[int, int]:
            """Count total and trainable parameters."""
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        result = {}

        if self.encoder is not None and self.encoder.model is not None:
            total, trainable = count_params(self.encoder.model)
            result["encoder"] = {"total": total, "trainable": trainable}

        if self.pooling is not None:
            total, trainable = count_params(self.pooling)
            result["pooling"] = {"total": total, "trainable": trainable}

        if self.projector is not None:
            total, trainable = count_params(self.projector)
            result["projector"] = {"total": total, "trainable": trainable}

        if self.llm is not None:
            total, trainable = count_params(self.llm)
            result["llm"] = {"total": total, "trainable": trainable}

        # Sum totals
        total_params = sum(c["total"] for c in result.values())
        trainable_params = sum(c["trainable"] for c in result.values())
        result["total"] = {"total": total_params, "trainable": trainable_params}

        return result

    def print_trainable_parameters(self) -> None:
        """Print a summary of trainable parameters."""
        params = self.get_trainable_parameters()

        print("\n" + "=" * 60)
        print("Trainable Parameters Summary")
        print("=" * 60)

        for name, counts in params.items():
            if name != "total":
                trainable_pct = 100 * counts["trainable"] / counts["total"] if counts["total"] > 0 else 0
                print(f"{name:15s}: {counts['trainable']:,} / {counts['total']:,} ({trainable_pct:.2f}%)")

        print("-" * 60)
        total = params["total"]
        trainable_pct = 100 * total["trainable"] / total["total"] if total["total"] > 0 else 0
        print(f"{'TOTAL':15s}: {total['trainable']:,} / {total['total']:,} ({trainable_pct:.2f}%)")
        print("=" * 60 + "\n")

    def train(self, mode: bool = True) -> "ProteinLLM":
        """
        Set the module to training mode.

        Note: ESM-2 encoder remains in eval mode even during training.
        """
        super().train(mode)

        # Keep encoder in eval mode (frozen)
        if self.encoder is not None and self.encoder.model is not None:
            self.encoder.model.eval()

        return self

    def eval(self) -> "ProteinLLM":
        """Set the module to evaluation mode."""
        return self.train(False)
