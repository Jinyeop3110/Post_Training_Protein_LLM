#!/usr/bin/env python3
"""Test ProteinLLM.generate() under FSDP with summon_full_params.

Validates that the FSDP generation fix works correctly on multiple GPUs.
Loads from a training checkpoint and wraps in FSDP matching training config.

Usage:
    torchrun --nproc_per_node=8 --master_port=29600 scripts/test_fsdp_generate.py

Set CHECKPOINT_DIR below or pass as env var:
    CHECKPOINT_DIR=results/.../checkpoints/checkpoint-2250 torchrun ...
"""

import logging
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

logging.basicConfig(
    level=logging.INFO,
    format="[rank %(process)d] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# Default checkpoint path (override with CHECKPOINT_DIR env var)
DEFAULT_CHECKPOINT = (
    "results/sft_esm3_mlp_combined_qwen3_8b_it_0304_150545/"
    "checkpoints/checkpoint-2250"
)

# Test protein sequences across different task categories
TEST_CASES = [
    {
        "category": "function",
        "sequence": "MKTLLILAVVGLIASAQARSTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLEH",
        "instruction": "What is the function of this protein?",
    },
    {
        "category": "domain",
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        "instruction": "Identify the structural domains in this protein.",
    },
    {
        "category": "catalytic",
        "sequence": "MRGSHHHHHHTDPALRAQLAALGAEDLFVTHNLVARAQPAGATYRAHLAAQG",
        "instruction": "Predict the catalytic activity of this enzyme.",
    },
    {
        "category": "general",
        "sequence": "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFVMVPDKRGPASFEIKPGQNNPSVDIAH",
        "instruction": "Describe this protein.",
    },
]


def build_prompt(instruction: str) -> str:
    """Build chat-template prompt matching training format."""
    return (
        "<|im_start|>system\n"
        "You are an expert protein scientist. Analyze protein sequences and "
        "provide detailed, accurate descriptions of their properties, functions, "
        "and characteristics.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    log.info(f"Rank {rank}/{world_size}, local_rank={local_rank}")

    # Import ProteinLLM
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models.multimodal_llm import ProteinLLM

    # Resolve checkpoint path
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", DEFAULT_CHECKPOINT)
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        dist.destroy_process_group()
        sys.exit(1)
    log.info(f"Loading from checkpoint: {checkpoint_path}")

    # --- Step 1: Load ProteinLLM from checkpoint ---
    # FSDP checkpoints don't have config.json (that's ProteinLLM.save_pretrained format).
    # Instead, construct ProteinLLM with known params matching the training config.
    log.info("Creating ProteinLLM matching training config...")

    protein_llm = ProteinLLM(
        approach="esm3",
        llm_name="Qwen/Qwen3-8B",
        encoder_name="esm3-sm-open-v1",
        encoder_embed_dim=1536,
        num_prefix_tokens=32,
        pooling_type="attention",
        projector_type="mlp",
        projector_hidden_dim=5120,
        projector_num_layers=2,
        projector_dropout=0.1,
        freeze_encoder=True,
        use_qlora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        device=f"cuda:{local_rank}",
        load_llm=True,
        load_encoder=True,
    )
    log.info("Base ProteinLLM created")

    # Load pooling weights
    pooling_path = checkpoint_path / "pooling.pt"
    if pooling_path.exists() and protein_llm.pooling is not None:
        protein_llm.pooling.load_state_dict(
            torch.load(pooling_path, map_location=f"cuda:{local_rank}", weights_only=True)
        )
        log.info("Loaded pooling weights")

    # Load projector weights
    projector_path = checkpoint_path / "projector.pt"
    if projector_path.exists() and protein_llm.projector is not None:
        protein_llm.projector.load_state_dict(
            torch.load(projector_path, map_location=f"cuda:{local_rank}", weights_only=True)
        )
        log.info("Loaded projector weights")

    # LoRA adapter: ProteinLLM constructor already applied LoRA.
    # Loading FSDP checkpoint adapter weights requires matching the FSDP
    # save format. For this test, we use fresh LoRA weights — the goal is
    # to validate summon_full_params under FSDP, not checkpoint quality.
    log.info("Using fresh LoRA (FSDP generation test — checkpoint adapter skipped)")

    # Move components to GPU
    if protein_llm.pooling is not None:
        protein_llm.pooling = protein_llm.pooling.to(f"cuda:{local_rank}")
    if protein_llm.projector is not None:
        protein_llm.projector = protein_llm.projector.to(f"cuda:{local_rank}")

    # --- Step 2: Cache embed_tokens BEFORE FSDP wrapping ---
    embed_layer = protein_llm.llm.model.embed_tokens
    if hasattr(embed_layer, "weight"):
        embed_weight = embed_layer.weight
    else:
        # PeftModel wrapping
        base = protein_llm.llm.get_base_model() if hasattr(protein_llm.llm, "get_base_model") else protein_llm.llm
        embed_weight = base.model.embed_tokens.weight
    protein_llm._fsdp_embed_cache = embed_weight.detach().clone().to(f"cuda:{local_rank}")
    log.info(f"Cached embed_tokens {tuple(protein_llm._fsdp_embed_cache.shape)}")

    # --- Step 3: Wrap LLM in FSDP (matching training config) ---
    log.info("Wrapping LLM in FSDP (shard_grad_op, min_num_params=100M)...")
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000,
    )
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    protein_llm.llm = FSDP(
        protein_llm.llm,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        use_orig_params=True,
        device_id=local_rank,
    )
    log.info("FSDP wrapping complete")

    # --- Step 4: Verify _find_fsdp_module ---
    fsdp_module = protein_llm._find_fsdp_module()
    if fsdp_module is not None:
        log.info(f"_find_fsdp_module() found: {type(fsdp_module).__name__}")
    else:
        log.error("_find_fsdp_module() returned None — fix is broken!")
        dist.destroy_process_group()
        sys.exit(1)

    # --- Step 5: Warmup forward pass (FSDP lazy init) ---
    log.info("Running warmup forward pass...")
    dummy_ids = torch.tensor([[1, 2, 3]], device=f"cuda:{local_rank}")
    with torch.no_grad():
        _ = protein_llm.llm(input_ids=dummy_ids)
    log.info("FSDP warmup complete")

    # --- Step 6: Run generation on test cases ---
    sequences = [tc["sequence"] for tc in TEST_CASES]
    prompts = [build_prompt(tc["instruction"]) for tc in TEST_CASES]

    log.info(f"Running generate() under FSDP on {len(TEST_CASES)} sequences...")
    protein_llm.eval()
    try:
        with torch.no_grad():
            generated = protein_llm.generate(
                protein_sequences=sequences,
                prompt=prompts,
                max_new_tokens=50,
                do_sample=False,
                repetition_penalty=1.2,
            )

        if rank == 0:
            log.info("=" * 60)
            log.info("FSDP GENERATION TEST — SUCCESS")
            log.info("=" * 60)
            for tc, text in zip(TEST_CASES, generated):
                log.info(f"[{tc['category']}] {tc['instruction']}")
                log.info(f"  Sequence: {tc['sequence'][:40]}...")
                log.info(f"  Output:   {text[:300]}")
                log.info("")
            log.info("summon_full_params fix works under FSDP!")

    except Exception as e:
        log.error(f"FSDP generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    # Cleanup
    dist.destroy_process_group()
    if rank == 0:
        log.info("Test complete — all ranks successful.")


if __name__ == "__main__":
    main()
