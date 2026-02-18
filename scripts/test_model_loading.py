#!/usr/bin/env python3
"""Test script to verify model loading capabilities.

NOTE: This script requires GPU access. Run on a compute node, not login node.

Usage (on compute node with GPU):
    source /home/yeopjin/orcd/pool/init_protein_llm.sh
    python scripts/test_model_loading.py
    python scripts/test_model_loading.py --model Qwen/Qwen3-1.5B
    python scripts/test_model_loading.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/test_model_loading.py --cpu  # Test on CPU only (slow)

SLURM Example:
    srun --gres=gpu:1 --mem=32G --time=00:30:00 \\
        bash -c "source /home/yeopjin/orcd/pool/init_protein_llm.sh && \\
        python scripts/test_model_loading.py --model Qwen/Qwen3-1.5B"
"""

import argparse
import sys
import time
from pathlib import Path


def check_environment():
    """Check if required packages are available."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    # Check Python version
    print(f"Python: {sys.version}")

    # Check CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    except ImportError as e:
        print(f"PyTorch not available: {e}")
        return False

    # Check transformers
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"Transformers not available: {e}")
        return False

    # Check HuggingFace cache
    hf_home = Path.home() / ".cache" / "huggingface"
    if "HF_HOME" in __import__("os").environ:
        hf_home = Path(__import__("os").environ["HF_HOME"])
    print(f"HuggingFace cache: {hf_home}")

    print()
    return True


def test_model_loading(model_name: str, dtype: str = "auto", trust_remote_code: bool = True):
    """Test loading a model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path.
        dtype: Data type (auto, float16, bfloat16, float32).
        trust_remote_code: Whether to trust remote code.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print(f"Testing Model: {model_name}")
    print("=" * 60)

    # Determine dtype
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    print(f"Using dtype: {torch_dtype}")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        print(f"   ✓ Tokenizer loaded in {time.time() - start:.1f}s")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   Model max length: {tokenizer.model_max_length}")
    except Exception as e:
        print(f"   ✗ Failed to load tokenizer: {e}")
        return False

    # Load model
    print("\n2. Loading model...")
    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        print(f"   ✓ Model loaded in {time.time() - start:.1f}s")
        print(f"   Parameters: {model.num_parameters() / 1e9:.2f}B")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   Dtype: {next(model.parameters()).dtype}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    # Test inference
    print("\n3. Testing inference...")
    try:
        test_input = "What is a protein?"
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✓ Inference completed in {time.time() - start:.1f}s")
        print(f"\n   Input: {test_input}")
        print(f"   Output: {response[:200]}...")
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        return False

    # Memory usage
    if torch.cuda.is_available():
        print("\n4. GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.5B",
        help="Model name or path (default: Qwen/Qwen3-1.5B)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode (slow, for testing without GPU)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List recommended models to test",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check environment, don't load model",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Recommended models to test:")
        print()
        print("Small (1-3B, good for testing):")
        print("  - Qwen/Qwen3-1.5B              # Latest Qwen 3")
        print("  - Qwen/Qwen2.5-1.5B-Instruct   # Instruction-tuned")
        print("  - meta-llama/Llama-3.2-1B-Instruct")
        print()
        print("Medium (7-8B, recommended for training):")
        print("  - Qwen/Qwen2.5-7B-Instruct")
        print("  - meta-llama/Llama-3.1-8B-Instruct")
        print()
        print("Large (70B+, requires multi-GPU):")
        print("  - Qwen/Qwen2.5-72B-Instruct")
        print("  - meta-llama/Llama-3.1-70B-Instruct")
        return

    # Check environment first
    if not check_environment():
        print("Environment check failed. Please ensure all dependencies are installed.")
        sys.exit(1)

    # Check if GPU is available
    import torch
    if not torch.cuda.is_available() and not args.cpu:
        print("\n" + "=" * 60)
        print("WARNING: No GPU detected!")
        print("=" * 60)
        print("You are likely on a login node without GPU access.")
        print()
        print("Options:")
        print("  1. Run on a compute node with GPU:")
        print("     srun --gres=gpu:1 --mem=32G --time=00:30:00 \\")
        print("         bash -c \"source /home/yeopjin/orcd/pool/init_protein_llm.sh && \\")
        print(f"         python scripts/test_model_loading.py --model {args.model}\"")
        print()
        print("  2. Force CPU mode (slow, just for testing):")
        print(f"     python scripts/test_model_loading.py --model {args.model} --cpu")
        print()
        sys.exit(1)

    if args.check_only:
        print("Environment check passed!")
        return

    # Force CPU if requested
    if args.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("\nForcing CPU mode (this will be slow)...")

    # Test model loading
    success = test_model_loading(args.model, args.dtype)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
