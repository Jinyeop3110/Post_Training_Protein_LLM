#!/usr/bin/env python3
"""Integration test: Train → Save → Load → Generate for MLP and Perceiver.

Runs sequentially:
  1. MLP SFT (10 steps) → eval with generation → save → load → generate
  2. Perceiver SFT (10 steps) → eval with generation → save → load → generate
  3. Backward compat: load existing checkpoint (sibling tokenizer layout)

Usage:
    source /home/yeopjin/orcd/pool/init_protein_llm.sh
    python scripts/test_checkpoint_roundtrip.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# Set triton cache to local (CRITICAL RULE)
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{os.environ.get('USER', 'test')}"
# Disable wandb for this test
os.environ["WANDB_MODE"] = "disabled"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("checkpoint_roundtrip")


# ============================================================================
# Helpers
# ============================================================================

def run_sft_training(projector_type: str, experiment_name: str, max_steps: int = 10):
    """Run SFT training for a given projector type using Hydra CLI."""
    log.info(f"\n{'='*70}")
    log.info(f"TRAINING: {projector_type.upper()} projector — {max_steps} steps")
    log.info(f"{'='*70}")

    cmd = [
        sys.executable, "scripts/train.py",
        f"experiment_name={experiment_name}",
        "approach=esm3",
        "model=qwen3_4b",
        "training=sft_lora",
        f"encoder.projector.type={projector_type}",
        # Minimal training for speed
        "training.epochs=1",
        "training.batch_size=2",
        "training.gradient_accumulation_steps=1",
        "training.max_seq_length=512",
        f"training.save_steps={max_steps}",
        f"training.eval_steps={max_steps}",
        "training.logging_steps=5",
        "training.lr=1e-4",
        "training.projector_lr=5e-4",
        "training.warmup_ratio=0.0",
        # Use small number of eval samples
        "evaluation.sft_gen_samples=3",
        # Limit data to ~50 samples (enough for ~10+ train steps with bs=2
        # and a few validation samples)
        "data.limit=50",
    ]

    log.info(f"Command: {' '.join(cmd)}")
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        log.error(f"TRAINING FAILED (exit {result.returncode})")
        log.error(f"STDOUT:\n{result.stdout[-3000:]}")
        log.error(f"STDERR:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"Training failed for {projector_type}")

    log.info("Training completed successfully")
    # Print last few lines of output
    for line in result.stdout.strip().split('\n')[-10:]:
        log.info(f"  {line}")

    return True


def check_checkpoint_structure(checkpoint_dir: Path, projector_type: str):
    """Verify checkpoint has expected files."""
    log.info(f"\nChecking checkpoint structure at: {checkpoint_dir}")

    protein_llm_dir = checkpoint_dir / "protein_llm"
    assert protein_llm_dir.exists(), f"Missing protein_llm dir: {protein_llm_dir}"

    # Required files
    required = ["config.json", "projector.pt"]
    if projector_type == "mlp":
        required.append("pooling.pt")

    for fname in required:
        fpath = protein_llm_dir / fname
        assert fpath.exists(), f"Missing: {fpath}"
        log.info(f"  ✓ {fname} ({fpath.stat().st_size / 1024:.1f} KB)")

    # Adapter
    adapter_dir = protein_llm_dir / "adapter"
    assert adapter_dir.exists(), f"Missing adapter dir: {adapter_dir}"
    assert (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / "adapter_model.safetensors").exists()
    log.info("  ✓ adapter/ (adapter_config.json + adapter_model.safetensors)")

    # Tokenizer (NEW: inside protein_llm/)
    tokenizer_dir = protein_llm_dir / "tokenizer"
    assert tokenizer_dir.exists(), f"Missing tokenizer inside checkpoint: {tokenizer_dir}"
    assert (tokenizer_dir / "tokenizer_config.json").exists()
    log.info("  ✓ tokenizer/ (self-contained)")

    # Config should have new fields
    with open(protein_llm_dir / "config.json") as f:
        config = json.load(f)
    assert "vocab_size" in config, "config.json missing vocab_size"
    assert config["vocab_size"] is not None, "vocab_size is None"
    assert "protein_special_tokens" in config, "config.json missing protein_special_tokens"
    log.info(f"  ✓ config.json has vocab_size={config['vocab_size']}, "
             f"protein_special_tokens={config['protein_special_tokens']}")

    # Sibling tokenizer (saved by SFT trainer — backward compat)
    sibling_tokenizer = checkpoint_dir / "tokenizer"
    if sibling_tokenizer.exists():
        log.info("  ✓ sibling tokenizer/ also present (backward compat)")

    return protein_llm_dir


def load_and_generate(checkpoint_path: Path, test_name: str):
    """Load a ProteinLLM checkpoint and run generation."""
    log.info(f"\n{'='*70}")
    log.info(f"LOAD & GENERATE: {test_name}")
    log.info(f"Checkpoint: {checkpoint_path}")
    log.info(f"{'='*70}")

    from src.models.multimodal_llm import ProteinLLM

    t0 = time.time()
    model = ProteinLLM.from_pretrained(str(checkpoint_path))
    load_time = time.time() - t0
    log.info(f"Model loaded in {load_time:.1f}s")

    # Verify model components
    assert model.tokenizer is not None, "Tokenizer not loaded"
    assert model.llm is not None, "LLM not loaded"
    assert model.encoder is not None, "Encoder not loaded"
    assert model.projector is not None, "Projector not loaded"
    log.info(f"  approach={model.approach}, projector_type={model.projector_type}")
    log.info(f"  vocab_size={len(model.tokenizer)}, llm_hidden_size={model.llm_hidden_size}")

    if model.pooling is not None:
        log.info("  pooling present (MLP path)")
    else:
        log.info("  no pooling (Perceiver path)")

    # Test generation
    model.eval()

    test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ"
    test_prompt = model.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a protein science expert."},
            {"role": "user", "content": (
                "Analyze the following protein sequence and describe its likely "
                "function:\n<|protein_start|><|protein_embed|><|protein_end|>"
            )},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    log.info("\nGenerating response...")
    log.info(f"  Protein: {test_sequence[:30]}...")
    t0 = time.time()
    outputs = model.generate(
        protein_sequences=[test_sequence],
        prompt=test_prompt,
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
    )
    gen_time = time.time() - t0
    log.info(f"  Generation took {gen_time:.1f}s")
    log.info(f"  Output: {outputs[0][:200]}")

    if len(outputs[0].strip()) == 0:
        log.warning("  ⚠ Empty generation output!")
    else:
        log.info(f"  ✓ Non-empty generation ({len(outputs[0])} chars)")

    return model, outputs


def save_load_roundtrip(model, original_checkpoint: Path, test_name: str):
    """Save model to new location, load it back, verify."""
    log.info(f"\n{'='*70}")
    log.info(f"SAVE-LOAD ROUNDTRIP: {test_name}")
    log.info(f"{'='*70}")

    with tempfile.TemporaryDirectory(prefix="roundtrip_") as tmpdir:
        save_path = Path(tmpdir) / "protein_llm"
        log.info(f"Saving to: {save_path}")
        model.save_pretrained(save_path)

        # Verify new checkpoint structure
        assert (save_path / "config.json").exists()
        assert (save_path / "tokenizer").exists()
        assert (save_path / "projector.pt").exists()
        if model.pooling is not None:
            assert (save_path / "pooling.pt").exists()

        with open(save_path / "config.json") as f:
            config = json.load(f)
        assert config["vocab_size"] == len(model.tokenizer)
        log.info(f"  ✓ Checkpoint saved with vocab_size={config['vocab_size']}")

        # Load it back
        from src.models.multimodal_llm import ProteinLLM
        model2 = ProteinLLM.from_pretrained(str(save_path))
        log.info("  ✓ Loaded from roundtrip checkpoint")

        # Verify consistency
        assert len(model2.tokenizer) == len(model.tokenizer), \
            f"Vocab mismatch: {len(model2.tokenizer)} vs {len(model.tokenizer)}"
        assert model2.approach == model.approach
        assert model2.projector_type == model.projector_type
        log.info("  ✓ Config matches original")

        # Test generation on reloaded model
        model2.eval()
        test_seq = "MKTAYIAK"
        prompt = model2.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a protein science expert."},
                {"role": "user", "content": (
                    "What is this protein?\n"
                    "<|protein_start|><|protein_embed|><|protein_end|>"
                )},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        outputs = model2.generate(
            protein_sequences=[test_seq],
            prompt=prompt,
            max_new_tokens=32,
            temperature=0.7,
            do_sample=True,
        )
        log.info(f"  Roundtrip generation: {outputs[0][:100]}")
        log.info("  ✓ Roundtrip generation successful")

    return True


def test_backward_compat():
    """Test loading existing checkpoint with sibling tokenizer layout."""
    log.info(f"\n{'='*70}")
    log.info("BACKWARD COMPAT: Load existing checkpoint (sibling tokenizer)")
    log.info(f"{'='*70}")

    existing_checkpoint = Path(
        "results/sft_lora_esm3_qwen3_8b_it_0224_192743/checkpoints/protein_llm"
    )
    if not existing_checkpoint.exists():
        log.warning(f"  Skipping: {existing_checkpoint} not found")
        return None

    # This checkpoint has sibling tokenizer at ../tokenizer/
    assert not (existing_checkpoint / "tokenizer").exists(), \
        "Old checkpoint shouldn't have embedded tokenizer yet"
    assert (existing_checkpoint.parent / "tokenizer").exists(), \
        "Old checkpoint should have sibling tokenizer"

    model, outputs = load_and_generate(existing_checkpoint, "backward_compat_qwen3_8b")
    log.info("  ✓ Backward compatibility verified")
    return model


# ============================================================================
# Main test sequence
# ============================================================================

def main():
    log.info("=" * 70)
    log.info("CHECKPOINT ROUNDTRIP INTEGRATION TEST")
    log.info("=" * 70)
    log.info(f"Working dir: {os.getcwd()}")
    log.info(f"GPUs: {torch.cuda.device_count()}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")

    results = {}
    t_start = time.time()

    # ========================================================================
    # Test 1: MLP path
    # ========================================================================
    mlp_experiment = "test_mlp_roundtrip"
    mlp_results_dir = Path(f"results/{mlp_experiment}")

    try:
        # Clean up any previous test run
        if mlp_results_dir.exists():
            shutil.rmtree(mlp_results_dir)

        # Train
        run_sft_training("mlp", mlp_experiment, max_steps=10)

        # Check checkpoint
        checkpoint_dir = mlp_results_dir / "checkpoints"
        protein_llm_path = check_checkpoint_structure(checkpoint_dir, "mlp")

        # Load and generate
        model_mlp, outputs_mlp = load_and_generate(protein_llm_path, "MLP")

        # Save-load roundtrip
        save_load_roundtrip(model_mlp, protein_llm_path, "MLP")

        results["mlp"] = "PASSED"
        log.info("\n✓ MLP test PASSED")

        # Free GPU memory
        del model_mlp
        torch.cuda.empty_cache()

    except Exception as e:
        results["mlp"] = f"FAILED: {e}"
        log.error(f"\n✗ MLP test FAILED: {e}", exc_info=True)

    # ========================================================================
    # Test 2: Perceiver path
    # ========================================================================
    perceiver_experiment = "test_perceiver_roundtrip"
    perceiver_results_dir = Path(f"results/{perceiver_experiment}")

    try:
        # Clean up any previous test run
        if perceiver_results_dir.exists():
            shutil.rmtree(perceiver_results_dir)

        # Train
        run_sft_training("perceiver", perceiver_experiment, max_steps=10)

        # Check checkpoint
        checkpoint_dir = perceiver_results_dir / "checkpoints"
        protein_llm_path = check_checkpoint_structure(checkpoint_dir, "perceiver")

        # Load and generate
        model_perceiver, outputs_perceiver = load_and_generate(
            protein_llm_path, "Perceiver"
        )

        # Save-load roundtrip
        save_load_roundtrip(model_perceiver, protein_llm_path, "Perceiver")

        results["perceiver"] = "PASSED"
        log.info("\n✓ Perceiver test PASSED")

        # Free GPU memory
        del model_perceiver
        torch.cuda.empty_cache()

    except Exception as e:
        results["perceiver"] = f"FAILED: {e}"
        log.error(f"\n✗ Perceiver test FAILED: {e}", exc_info=True)

    # ========================================================================
    # Test 3: Backward compatibility (existing Qwen3-8B checkpoint)
    # ========================================================================
    try:
        model_compat = test_backward_compat()
        if model_compat is not None:
            results["backward_compat"] = "PASSED"
            del model_compat
            torch.cuda.empty_cache()
        else:
            results["backward_compat"] = "SKIPPED (no checkpoint)"
    except Exception as e:
        results["backward_compat"] = f"FAILED: {e}"
        log.error(f"\n✗ Backward compat test FAILED: {e}", exc_info=True)

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - t_start
    log.info(f"\n{'='*70}")
    log.info(f"TEST RESULTS (total time: {total_time:.0f}s)")
    log.info(f"{'='*70}")
    all_passed = True
    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else ("⊘" if "SKIPPED" in result else "✗")
        log.info(f"  {status} {test_name}: {result}")
        if "FAILED" in result:
            all_passed = False

    if all_passed:
        log.info("\nALL TESTS PASSED ✓")
    else:
        log.info("\nSOME TESTS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
