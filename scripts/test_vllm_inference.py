#!/usr/bin/env python3
"""Test vLLM inference with Qwen models.

Usage (on compute node with GPU):
    source /home/yeopjin/orcd/pool/init_protein_llm.sh
    python scripts/test_vllm_inference.py
    python scripts/test_vllm_inference.py --model Qwen/Qwen3-8B
    python scripts/test_vllm_inference.py --model Qwen/Qwen2.5-7B-Instruct

SLURM Example:
    srun --gres=gpu:1 --mem=48G --time=00:30:00 \
        bash -c "source /home/yeopjin/orcd/pool/init_protein_llm.sh && \
        python scripts/test_vllm_inference.py"
"""

import argparse
import sys
import time


def check_gpu():
    """Check GPU availability."""
    import torch

    print("=" * 60)
    print("GPU Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        print("Run this script on a compute node with GPU access.")
        return False

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print()
    return True


def test_vllm_inference(model_name: str, max_model_len: int = 4096):
    """Test vLLM inference with a model.

    Args:
        model_name: HuggingFace model name.
        max_model_len: Maximum model context length.
    """
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print(f"Testing vLLM with: {model_name}")
    print("=" * 60)

    # Load model
    print("\n1. Loading model with vLLM...")
    start = time.time()
    try:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            dtype="bfloat16",
        )
        load_time = time.time() - start
        print(f"   ✓ Model loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    # Test inference
    print("\n2. Testing inference...")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )

    # Test prompts
    test_prompts = [
        "What is a protein?",
        "Explain the function of hemoglobin in one sentence.",
    ]

    try:
        start = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        inference_time = time.time() - start

        print(f"   ✓ Inference completed in {inference_time:.2f}s")
        print(f"   Throughput: {len(test_prompts) / inference_time:.2f} prompts/s")

        print("\n3. Sample outputs:")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated = output.outputs[0].text
            print(f"\n   Prompt {i+1}: {prompt}")
            print(f"   Response: {generated[:200]}...")

    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        return False

    # Memory usage
    import torch
    print("\n4. GPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    print("\n" + "=" * 60)
    print("✓ vLLM test passed!")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test vLLM inference")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.5B",
        help="Model name (default: Qwen/Qwen3-1.5B)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List recommended Qwen models",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Recommended Qwen models for vLLM:")
        print()
        print("Qwen 3 (latest):")
        print("  - Qwen/Qwen3-0.6B        # ~1.5 GB VRAM")
        print("  - Qwen/Qwen3-1.5B        # ~3 GB VRAM")
        print("  - Qwen/Qwen3-4B          # ~8 GB VRAM")
        print("  - Qwen/Qwen3-8B          # ~16 GB VRAM")
        print()
        print("Qwen 2.5 (instruction-tuned):")
        print("  - Qwen/Qwen2.5-1.5B-Instruct")
        print("  - Qwen/Qwen2.5-7B-Instruct   # Recommended for training")
        print("  - Qwen/Qwen2.5-14B-Instruct")
        print()
        return

    # Check GPU
    if not check_gpu():
        sys.exit(1)

    # Test vLLM
    success = test_vllm_inference(args.model, args.max_model_len)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
