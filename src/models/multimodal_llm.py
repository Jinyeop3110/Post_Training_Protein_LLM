"""
Multimodal Protein-LLM Model

This module implements the ProteinLLM class that combines a protein encoder
with a Large Language Model through an approach-based architecture.

Supported approaches:
    - text: Raw protein sequence as text tokens (no encoder/pooling/projector)
    - esm3: Frozen ESM-3 encoder -> attention pooling -> MLP projector -> LLM

Architecture (esm3 approach):
    Protein Sequence -> Encoder (frozen) -> [B, L, D_enc]
                            |
                  AttentionPooling -> [B, N, D_enc] (N = num_prefix_tokens)
                            |
                  MLPProjector -> [B, N, D_llm]
                            |
                  Replace <|protein_embed|> placeholder inline in prompt
                            |
                  LLM (with QLoRA on k/v projections) -> Generate response

    Prompt layout: [system] [instruction] [protein_embeds] [assistant]
    The protein text is NOT included — only learned embeddings.

Architecture (text approach):
    Protein Sequence -> Format as "<protein>MKTL...</protein>"
                            |
                  Tokenize and feed directly to LLM
                            |
                  LLM (with QLoRA on k/v projections) -> Generate response
"""

import json
import logging
import os
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
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from src.models.pooling import BasePooling, get_pooling
from src.models.projector import MLPProjector
from src.models.protein_encoder import (
    ESM3ProteinEncoder,
    ProteinEncoder,
    TextProteinEncoder,
)

logger = logging.getLogger(__name__)

# Valid approach identifiers
VALID_APPROACHES = ("text", "esm3")

# Approaches that use an embedding encoder (require pooling + projector)
EMBEDDING_APPROACHES = ("esm3",)

# Special tokens for inline protein embeddings (ESM-3 approach).
# In the prompt, the protein sequence text is replaced with
# <|protein_start|><|protein_embed|><|protein_end|>.
# At forward time, prepare_inputs() replaces the <|protein_embed|> token
# with the ESM-3 → pooling → projector output, while the boundary tokens
# remain as regular LLM embeddings to help the model identify the protein
# modality region.
PROTEIN_START_TOKEN = "<|protein_start|>"
PROTEIN_EMBED_TOKEN = "<|protein_embed|>"
PROTEIN_END_TOKEN = "<|protein_end|>"
PROTEIN_PLACEHOLDER = f"{PROTEIN_START_TOKEN}{PROTEIN_EMBED_TOKEN}{PROTEIN_END_TOKEN}"
# All three must be registered as special tokens in the tokenizer.
PROTEIN_SPECIAL_TOKENS = [PROTEIN_START_TOKEN, PROTEIN_EMBED_TOKEN, PROTEIN_END_TOKEN]


class ProteinLLM(nn.Module):
    """
    Multimodal Protein-LLM model with approach-based encoder selection.

    Supports two approaches for incorporating protein information:
    - text: Raw protein sequence as text tokens (no encoder needed)
    - esm3: Frozen ESM-3 encoder with pooling and projection

    For the esm3 approach, the pipeline is:
        Encoder (frozen) -> AttentionPooling -> MLPProjector -> LLM prefix tokens

    For the text approach, proteins are formatted as text and tokenized directly.

    Critical rules:
    - NEVER modify ESM-3 encoder weights (always frozen)
    - LoRA on all linear layers (q/k/v/o + gate/up/down projections), r=8
    - Attention pooling is the default (not mean pooling)

    Args:
        approach: Protein encoding approach ("text" or "esm3").
        llm_name: HuggingFace model path for the LLM.
        encoder_name: Encoder model name (e.g., "esm3-sm-open-v1").
        encoder_embed_dim: Embedding dimension of the encoder output. If None,
            determined automatically from the encoder.
        num_prefix_tokens: Number of protein prefix tokens for pooling.
        pooling_type: Type of pooling ("attention" or "mean").
        projector_hidden_dim: Hidden dimension for the MLP projector.
        projector_num_layers: Number of layers in the MLP projector.
        projector_dropout: Dropout rate for the projector.
        freeze_encoder: Whether to freeze the encoder (always True).
        use_qlora: Whether to use QLoRA for the LLM.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        lora_target_modules: Target modules for LoRA (default: all linear layers).
        device: Device to load the model on.
        load_llm: Whether to eagerly load the LLM.
        load_encoder: Whether to eagerly load the encoder.

    Example:
        >>> # ESM-3 approach
        >>> model = ProteinLLM(
        ...     approach="esm3",
        ...     llm_name="Qwen/Qwen3-4B",
        ...     encoder_name="esm3-sm-open-v1",
        ...     num_prefix_tokens=32,
        ... )
        >>> # Text approach
        >>> model = ProteinLLM(approach="text", llm_name="Qwen/Qwen3-4B")
    """

    # Default configuration values
    DEFAULT_APPROACH = "esm3"
    DEFAULT_LLM_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    DEFAULT_ENCODER_NAME = "esm3-sm-open-v1"
    DEFAULT_NUM_PREFIX_TOKENS = 32
    DEFAULT_POOLING_TYPE = "attention"
    DEFAULT_PROJECTOR_HIDDEN_DIM = 5120
    DEFAULT_PROJECTOR_NUM_LAYERS = 2
    DEFAULT_PROJECTOR_DROPOUT = 0.1
    DEFAULT_LORA_R = 8
    DEFAULT_LORA_ALPHA = 16
    DEFAULT_LORA_DROPOUT = 0.05
    DEFAULT_LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Known encoder embedding dimensions (for lookup without loading)
    ENCODER_EMBED_DIMS: Dict[str, int] = {
        # ESM-3 models
        "esm3-sm-open-v1": 1536,
        "esm3_sm_open_v1": 1536,
    }

    def __init__(
        self,
        approach: str = DEFAULT_APPROACH,
        llm_name: str = DEFAULT_LLM_NAME,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        encoder_embed_dim: Optional[int] = None,
        num_prefix_tokens: int = DEFAULT_NUM_PREFIX_TOKENS,
        pooling_type: str = DEFAULT_POOLING_TYPE,
        projector_type: str = "mlp",
        projector_hidden_dim: int = DEFAULT_PROJECTOR_HIDDEN_DIM,
        projector_num_layers: int = DEFAULT_PROJECTOR_NUM_LAYERS,
        projector_dropout: float = DEFAULT_PROJECTOR_DROPOUT,
        perceiver_layers: int = 2,
        perceiver_heads: int = 8,
        perceiver_ffn_dim: int = 2048,
        perceiver_latent_dim: Optional[int] = None,
        freeze_encoder: bool = True,
        use_qlora: bool = True,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_target_modules: Optional[List[str]] = None,
        device: Optional[str] = None,
        load_llm: bool = True,
        load_encoder: bool = True,
        encoder_dtype: str = "bfloat16",
        encoder_batch_size: int = 4,
    ) -> None:
        super().__init__()

        # Validate approach
        if approach not in VALID_APPROACHES:
            raise ValueError(
                f"Unknown approach: '{approach}'. "
                f"Valid approaches: {VALID_APPROACHES}"
            )

        self.approach = approach
        self.llm_name = llm_name
        self.encoder_name = encoder_name
        self.num_prefix_tokens = num_prefix_tokens
        self.pooling_type = pooling_type
        self.projector_type = projector_type
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_num_layers = projector_num_layers
        self.projector_dropout = projector_dropout
        self.perceiver_layers = perceiver_layers
        self.perceiver_heads = perceiver_heads
        self.perceiver_ffn_dim = perceiver_ffn_dim
        self.perceiver_latent_dim = perceiver_latent_dim
        self.freeze_encoder = freeze_encoder  # Always True for protein encoders
        self.use_qlora = use_qlora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or self.DEFAULT_LORA_TARGET_MODULES
        self.encoder_dtype = encoder_dtype
        self.encoder_batch_size = encoder_batch_size

        # Determine device (DDP-aware: use LOCAL_RANK for correct GPU)
        if device is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.device = f"cuda:{local_rank}"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Resolve encoder embedding dimension
        if encoder_embed_dim is not None:
            self.encoder_embed_dim = encoder_embed_dim
        else:
            self.encoder_embed_dim = self.ENCODER_EMBED_DIMS.get(encoder_name, 1280)

        # Initialize components (may be lazy loaded)
        self.encoder: Optional[ProteinEncoder] = None
        self.pooling: Optional[BasePooling] = None
        self.projector: Optional[nn.Module] = None  # MLPProjector or PerceiverResampler
        self.llm: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.llm_hidden_size: int = 2560  # Default for Qwen3-4B, updated on LLM load

        # For text approach, create a text encoder (lightweight, no GPU needed)
        if self.approach == "text":
            self.encoder = TextProteinEncoder()
            logger.info("Text approach: no embedding encoder, pooling, or projector needed")
        elif load_encoder and self.approach in EMBEDDING_APPROACHES:
            self._load_encoder()
            # Only build separate pooling for MLP path
            # (Perceiver handles pooling internally)
            if self.projector_type == "mlp":
                self._build_pooling()

        if load_llm:
            self._load_llm()
            # Only build projector for embedding approaches
            if self.approach in EMBEDDING_APPROACHES:
                self._build_projector()

    def _load_encoder(self) -> None:
        """Load the protein encoder based on the selected approach.

        Instantiates the ESM-3 encoder based on self.approach.
        The encoder is always frozen after loading.

        Raises:
            ValueError: If the approach does not use an embedding encoder.
        """
        if self.approach == "text":
            logger.info("Text approach: skipping encoder loading")
            return

        if self.approach == "esm3":
            logger.info(f"Loading ESM-3 encoder: {self.encoder_name}")
            self.encoder = ESM3ProteinEncoder(
                model_name=self.encoder_name,
                device=self.device,
                dtype=self.encoder_dtype,
                encoder_batch_size=self.encoder_batch_size,
            )
            # ESM3ProteinEncoder freezes internally, but get embedding dim
            self.encoder_embed_dim = self.encoder.get_embedding_dim()
            logger.info(
                f"ESM-3 encoder loaded: embedding_dim={self.encoder_embed_dim}"
            )

        else:
            raise ValueError(
                f"Cannot load encoder for approach '{self.approach}'. "
                f"Embedding approaches: {EMBEDDING_APPROACHES}"
            )

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

    def _load_llm(self, tokenizer_path: Optional[str] = None) -> None:
        """Load the LLM with optional QLoRA configuration.

        Args:
            tokenizer_path: If provided, load tokenizer from this path instead
                of self.llm_name. Used by from_pretrained() to load the saved
                tokenizer (which already has protein special tokens).
        """
        logger.info(f"Loading LLM: {self.llm_name}")

        # Load tokenizer — from checkpoint path if available, else from HF hub
        tokenizer_source = tokenizer_path or self.llm_name
        logger.info(f"Loading tokenizer from: {tokenizer_source}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            padding_side="left",
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add protein special tokens for embedding approaches (idempotent —
        # add_special_tokens returns 0 if tokens already exist, e.g. when
        # loading from a saved tokenizer that already has them).
        if self.approach in EMBEDDING_APPROACHES:
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": PROTEIN_SPECIAL_TOKENS}
            )
            if num_added > 0:
                logger.info(
                    f"Added {num_added} protein special tokens: "
                    f"{PROTEIN_SPECIAL_TOKENS} (vocab size: {len(self.tokenizer)})"
                )

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

            device_map = {"": self.device} if "cuda" in self.device else "auto"
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            # Prepare model for k-bit training
            self.llm = prepare_model_for_kbit_training(self.llm)

            # Resize embeddings if protein special tokens were added
            if len(self.tokenizer) != self.llm.config.vocab_size:
                old_vocab = self.llm.config.vocab_size
                self.llm.resize_token_embeddings(len(self.tokenizer))
                logger.info(
                    f"Resized embeddings: {old_vocab} -> {len(self.tokenizer)}"
                )

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
            device_map = {"": self.device} if "cuda" in self.device else "auto"
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            # Resize embeddings if protein special tokens were added
            if len(self.tokenizer) != self.llm.config.vocab_size:
                old_vocab = self.llm.config.vocab_size
                self.llm.resize_token_embeddings(len(self.tokenizer))
                logger.info(
                    f"Resized embeddings: {old_vocab} -> {len(self.tokenizer)}"
                )

        # Get LLM hidden size
        self.llm_hidden_size = self.llm.config.hidden_size
        logger.info(f"LLM hidden size: {self.llm_hidden_size}")

    def _build_projector(self) -> None:
        """Build the projector module (MLP or Perceiver Resampler)."""
        if self.projector_type == "perceiver":
            from src.models.perceiver import PerceiverResampler

            latent_str = f", latent_dim={self.perceiver_latent_dim}" if self.perceiver_latent_dim else ""
            logger.info(
                f"Building Perceiver Resampler: {self.encoder_embed_dim} -> "
                f"{self.llm_hidden_size} ({self.perceiver_layers} layers, "
                f"{self.num_prefix_tokens} queries{latent_str})"
            )
            self.projector = PerceiverResampler(
                encoder_dim=self.encoder_embed_dim,
                output_dim=self.llm_hidden_size,
                latent_dim=self.perceiver_latent_dim,
                num_queries=self.num_prefix_tokens,
                num_layers=self.perceiver_layers,
                num_heads=self.perceiver_heads,
                ffn_dim=self.perceiver_ffn_dim,
                dropout=self.projector_dropout,
            ).to(self.device)
        else:
            logger.info(
                f"Building MLP projector: {self.encoder_embed_dim} -> "
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
        1. Encodes sequences with the protein encoder (ESM-3) to get per-residue embeddings
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
        if self.encoder is None or self.projector is None:
            raise RuntimeError(
                "Model components not initialized. "
                "Ensure encoder and projector are loaded."
            )
        if self.projector_type == "mlp" and self.pooling is None:
            raise RuntimeError(
                "Pooling module not initialized for MLP projector path."
            )

        # Get encoder embeddings [B, L, encoder_embed_dim]
        # ESM-3 encoder manages its own autocast (bf16 by default) inside
        # _encode_batched, so we just need torch.no_grad() here.
        encoder_output = self.encoder.encode(sequences)
        embeddings = encoder_output["embeddings"]  # [B, L, D]

        if self.pooling is not None:
            # MLP path: pool then project
            pooled = self.pooling(embeddings)
            projected = self.projector(pooled)
        else:
            # Perceiver path: projector handles pooling internally
            projected = self.projector(embeddings)

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
        Prepare combined inputs by replacing the protein placeholder token
        with ESM-3 embeddings inline.

        The prompt contains a single ``<|protein_embed|>`` token where the
        protein sequence would normally appear as text.  This method:

        1. Encodes protein sequences into N embedding tokens via
           ESM-3 → pooling → projector.
        2. Gets text embeddings from the LLM.
        3. Finds the placeholder token position in each sequence.
        4. Splices the protein embeddings in place of the placeholder,
           yielding ``[..., instruction, protein_embeds, assistant, ...]``.

        The final sequence length is ``T - 1 + N`` (1 placeholder removed,
        N protein tokens inserted).  Because every sequence has exactly one
        placeholder, all sequences grow by the same amount and padding
        alignment is preserved.

        Args:
            protein_sequences: List of protein sequences.
            text_input_ids: Text input token IDs [B, T].
            text_attention_mask: Text attention mask [B, T].
            labels: Optional labels for training [B, T].

        Returns:
            Dictionary containing:
                - inputs_embeds: Combined embeddings [B, T - 1 + N, D]
                - attention_mask: Combined attention mask [B, T - 1 + N]
                - labels: Combined labels [B, T - 1 + N] (if provided)
                - position_ids: Position IDs [B, T - 1 + N]
        """
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device

        # Get protein embeddings [B, N, D]
        protein_embeds, protein_mask = self.encode_protein(
            protein_sequences, return_attention_mask=True
        )
        num_protein_tokens = protein_embeds.shape[1]

        # Get text embeddings from LLM [B, T, D]
        if hasattr(self.llm, "get_base_model"):
            base_model = self.llm.get_base_model()
            text_embeds = base_model.model.embed_tokens(text_input_ids)
        else:
            text_embeds = self.llm.model.embed_tokens(text_input_ids)

        # Find the embed placeholder token ID (boundary tokens stay as LLM embeddings)
        placeholder_id = self.tokenizer.convert_tokens_to_ids(PROTEIN_EMBED_TOKEN)

        # Build combined sequences by replacing placeholder with protein embeds
        all_embeds = []
        all_masks = []
        all_labels = [] if labels is not None else None

        for i in range(batch_size):
            positions = (text_input_ids[i] == placeholder_id).nonzero(as_tuple=True)[0]

            if len(positions) == 0:
                # No placeholder found — fallback: prepend protein embeddings
                logger.warning(
                    "No protein placeholder found in sequence %d; "
                    "falling back to prepend mode", i
                )
                emb = torch.cat(
                    [protein_embeds[i].to(text_embeds.dtype), text_embeds[i]], dim=0
                )
                mask = torch.cat(
                    [protein_mask[i].to(device), text_attention_mask[i]], dim=0
                )
                all_embeds.append(emb)
                all_masks.append(mask)
                if labels is not None:
                    prefix_lbl = torch.full(
                        (num_protein_tokens,), -100,
                        dtype=labels.dtype, device=device,
                    )
                    all_labels.append(torch.cat([prefix_lbl, labels[i]], dim=0))
                continue

            pos = positions[0].item()

            # Split text embeddings at placeholder position
            before_emb = text_embeds[i, :pos]           # [pos, D]
            after_emb = text_embeds[i, pos + 1:]        # [T-pos-1, D]
            prot_emb = protein_embeds[i].to(text_embeds.dtype)  # [N, D]
            all_embeds.append(torch.cat([before_emb, prot_emb, after_emb], dim=0))

            # Attention mask
            before_mask = text_attention_mask[i, :pos]
            after_mask = text_attention_mask[i, pos + 1:]
            all_masks.append(
                torch.cat([before_mask, protein_mask[i].to(device), after_mask], dim=0)
            )

            # Labels
            if labels is not None:
                before_lbl = labels[i, :pos]
                after_lbl = labels[i, pos + 1:]
                prot_lbl = torch.full(
                    (num_protein_tokens,), -100,
                    dtype=labels.dtype, device=device,
                )
                all_labels.append(
                    torch.cat([before_lbl, prot_lbl, after_lbl], dim=0)
                )

        # Stack — all sequences have the same length (T - 1 + N)
        combined_embeds = torch.stack(all_embeds, dim=0)
        combined_mask = torch.stack(all_masks, dim=0)

        total_seq_len = combined_embeds.shape[1]
        position_ids = torch.arange(
            total_seq_len, dtype=torch.long, device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        result = {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_mask,
            "position_ids": position_ids,
        }

        if all_labels is not None:
            result["labels"] = torch.stack(all_labels, dim=0)

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
        # During eval (no grad), compute loss ourselves with chunked CE
        # to avoid the 40 GiB logits.float() in HF's ForCausalLMLoss.
        fwd_labels = prepared.get("labels")
        if not torch.is_grad_enabled() and fwd_labels is not None:
            # Eval mode: don't pass labels to LLM (avoids float32 logits copy)
            outputs = self.llm(
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared["attention_mask"],
                position_ids=prepared["position_ids"],
                **kwargs,
            )
            # Chunked CE loss in bf16 — process vocab in 32K-token chunks
            logits = outputs.logits  # [B, T, V] in bf16
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = fwd_labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            # Training mode: let HF compute loss normally (grad checkpointing
            # keeps memory manageable)
            outputs = self.llm(
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared["attention_mask"],
                position_ids=prepared["position_ids"],
                labels=fwd_labels,
                **kwargs,
            )
            loss = outputs.loss if hasattr(outputs, "loss") else None

        return {
            "loss": loss,
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
        return_token_ids: bool = False,
        **generate_kwargs,
    ) -> Union[List[str], Tuple[List[str], torch.Tensor, int]]:
        """
        Generate text responses given protein sequences and prompts.

        Args:
            protein_sequences: List of protein sequences.
            prompt: Text prompt(s) for generation.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling parameter.
            do_sample: Whether to use sampling (vs greedy decoding).
            return_token_ids: If True, return (texts, token_ids, input_length)
                tuple instead of just texts. Used by GRPO to re-compute
                differentiable log probabilities from sampled tokens.
            **generate_kwargs: Additional arguments for generate().

        Returns:
            If return_token_ids is False: List of generated text responses.
            If return_token_ids is True: Tuple of (generated_texts,
                generated_token_ids, input_length) where token_ids has shape
                [B, gen_len] and input_length is the number of prefix+prompt
                positions in the model output.
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
        input_length = prepared["inputs_embeds"].shape[1]
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

        # Slice off the input positions from the output.  Behavior depends
        # on the transformers version:
        #   - Older: output includes placeholder IDs for each input position
        #     followed by generated token IDs → slice at input_length
        #   - Newer (with inputs_embeds): output contains ONLY the generated
        #     token IDs → no slicing needed
        output_len = outputs.shape[1]
        if output_len > input_length:
            # Output includes input positions — slice them off
            generated_tokens = outputs[:, input_length:]
        else:
            # Output contains only generated tokens
            generated_tokens = outputs

        generated_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        if return_token_ids:
            return generated_texts, generated_tokens, input_length

        return generated_texts

    @classmethod
    def from_config(cls, cfg: Union[DictConfig, Dict[str, Any]]) -> "ProteinLLM":
        """
        Create ProteinLLM from a Hydra/OmegaConf configuration.

        Expected config structure:
            approach: "esm3"  # "text" or "esm3"
            model:
                path: "Qwen/Qwen3-4B"
                architecture:
                    hidden_size: 2560
            encoder:
                model_name: "esm3-sm-open-v1"
                embedding_dim: 1536
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
                    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

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

        # Extract approach from top-level config (default to DEFAULT_APPROACH)
        approach = cfg.get("approach", cls.DEFAULT_APPROACH)

        # Extract encoder embedding dim from config if available
        encoder_embed_dim = encoder_cfg.get("embedding_dim", None)

        return cls(
            approach=approach,
            llm_name=model_cfg.get("path", cls.DEFAULT_LLM_NAME),
            encoder_name=encoder_cfg.get("model_name", cls.DEFAULT_ENCODER_NAME),
            encoder_embed_dim=encoder_embed_dim,
            num_prefix_tokens=num_prefix_tokens,
            pooling_type=pooling_cfg.get("method", cls.DEFAULT_POOLING_TYPE),
            projector_type=projector_cfg.get("type", "mlp"),
            projector_hidden_dim=projector_cfg.get("hidden_dim", cls.DEFAULT_PROJECTOR_HIDDEN_DIM),
            projector_num_layers=projector_cfg.get("num_layers", cls.DEFAULT_PROJECTOR_NUM_LAYERS),
            projector_dropout=projector_cfg.get("dropout", cls.DEFAULT_PROJECTOR_DROPOUT),
            perceiver_layers=projector_cfg.get("perceiver_layers", 2),
            perceiver_heads=projector_cfg.get("perceiver_heads", 8),
            perceiver_ffn_dim=projector_cfg.get("perceiver_ffn_dim", 2048),
            perceiver_latent_dim=projector_cfg.get("perceiver_latent_dim", None),
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

        Note: Encoder weights (ESM-3) are not saved as they are frozen
        and loaded from the pretrained checkpoint.

        Args:
            path: Directory path to save the model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "approach": self.approach,
            "llm_name": self.llm_name,
            "encoder_name": self.encoder_name,
            "num_prefix_tokens": self.num_prefix_tokens,
            "pooling_type": self.pooling_type,
            "projector_type": self.projector_type,
            "projector_hidden_dim": self.projector_hidden_dim,
            "projector_num_layers": self.projector_num_layers,
            "projector_dropout": self.projector_dropout,
            "perceiver_layers": self.perceiver_layers,
            "perceiver_heads": self.perceiver_heads,
            "perceiver_ffn_dim": self.perceiver_ffn_dim,
            "perceiver_latent_dim": self.perceiver_latent_dim,
            "freeze_encoder": self.freeze_encoder,
            "use_qlora": self.use_qlora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "encoder_embed_dim": self.encoder_embed_dim,
            "llm_hidden_size": self.llm_hidden_size,
            # Vocab info for self-contained checkpoint loading
            "vocab_size": len(self.tokenizer) if self.tokenizer else None,
            "protein_special_tokens": (
                PROTEIN_SPECIAL_TOKENS
                if self.approach in EMBEDDING_APPROACHES
                else None
            ),
        }

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save tokenizer inside checkpoint for self-contained loading
        if self.tokenizer is not None:
            tokenizer_path = path / "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")

        # Save pooling weights
        if self.pooling is not None:
            torch.save(self.pooling.state_dict(), path / "pooling.pt")

        # Save projector weights
        if self.projector is not None:
            torch.save(self.projector.state_dict(), path / "projector.pt")

        # Save LoRA adapter weights (works for both LoRA and QLoRA)
        if self.llm is not None and hasattr(self.llm, "save_pretrained"):
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

        The checkpoint is self-contained: tokenizer, config, pooling/projector
        weights, and LoRA adapter are all inside the checkpoint directory.

        Tokenizer search order:
            1. ``path/tokenizer/`` (new self-contained format)
            2. ``path/../tokenizer/`` (backward compat — SFT trainer saves
               tokenizer as a sibling of protein_llm/)

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

        # Locate tokenizer: inside checkpoint (new) or sibling dir (backward compat)
        tokenizer_path = None
        if (path / "tokenizer").exists():
            tokenizer_path = str(path / "tokenizer")
            logger.info(f"Found tokenizer inside checkpoint: {tokenizer_path}")
        elif (path.parent / "tokenizer").exists():
            tokenizer_path = str(path.parent / "tokenizer")
            logger.info(
                f"Found tokenizer at sibling path (backward compat): "
                f"{tokenizer_path}"
            )

        # Create model with load_llm=False — we call _load_llm() manually
        # with tokenizer_path so the LLM loads with the correct vocab size.
        model = cls(
            approach=config.get("approach", cls.DEFAULT_APPROACH),
            llm_name=config["llm_name"],
            encoder_name=config["encoder_name"],
            encoder_embed_dim=config.get("encoder_embed_dim"),
            num_prefix_tokens=config["num_prefix_tokens"],
            pooling_type=config["pooling_type"],
            projector_type=config.get("projector_type", "mlp"),
            projector_hidden_dim=config["projector_hidden_dim"],
            projector_num_layers=config["projector_num_layers"],
            projector_dropout=config["projector_dropout"],
            perceiver_layers=config.get("perceiver_layers", 2),
            perceiver_heads=config.get("perceiver_heads", 8),
            perceiver_ffn_dim=config.get("perceiver_ffn_dim", 2048),
            perceiver_latent_dim=config.get("perceiver_latent_dim", None),
            freeze_encoder=config["freeze_encoder"],
            use_qlora=config["use_qlora"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            lora_target_modules=config["lora_target_modules"],
            device=device,
            load_llm=False,  # Deferred — load with tokenizer_path below
            load_encoder=load_encoder,
        )

        # Load LLM with the saved tokenizer (ensures correct vocab size
        # before loading the LoRA adapter)
        if load_llm:
            model._load_llm(tokenizer_path=tokenizer_path)
            # Build projector after LLM (needs llm_hidden_size)
            if model.approach in EMBEDDING_APPROACHES:
                model._build_projector()

        # Load pooling weights
        pooling_path = path / "pooling.pt"
        if pooling_path.exists() and model.pooling is not None:
            model.pooling.load_state_dict(
                torch.load(pooling_path, map_location=model.device, weights_only=True)
            )

        # Load projector weights
        projector_path = path / "projector.pt"
        if projector_path.exists() and model.projector is not None:
            model.projector.load_state_dict(
                torch.load(projector_path, map_location=model.device, weights_only=True)
            )

        # Load LoRA adapter weights (works for both LoRA and QLoRA)
        adapter_path = path / "adapter"
        if adapter_path.exists() and model.llm is not None:
            if PEFT_AVAILABLE:
                # Get the base model if already wrapped by PeftModel
                base = (
                    model.llm.get_base_model()
                    if hasattr(model.llm, "get_base_model")
                    else model.llm
                )
                model.llm = PeftModel.from_pretrained(base, adapter_path)

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

        Note: Protein encoder (ESM-3) remains in eval mode even during training.
        """
        super().train(mode)

        # Keep encoder in eval mode (frozen)
        if self.encoder is not None and self.encoder.model is not None:
            self.encoder.model.eval()

        return self

    def eval(self) -> "ProteinLLM":
        """Set the module to evaluation mode."""
        return self.train(False)
