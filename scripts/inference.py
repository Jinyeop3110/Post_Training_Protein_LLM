#!/usr/bin/env python3
"""Inference/demo script for protein-LLM.

Usage::

    # ProteinLLM checkpoint (ESM-3 approach)
    python scripts/inference.py --checkpoint results/.../checkpoints/protein_llm \
        --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ \
        --prompt "Describe the function of this protein."

    # Custom prompt with max tokens
    python scripts/inference.py --checkpoint results/.../checkpoints/protein_llm \
        --sequence MKTAYIAK --prompt "What domains does this protein have?" \
        --max-tokens 256 --temperature 0.3
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Set Triton cache to local filesystem (CRITICAL: must be before any torch import)
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_{os.environ.get('USER', 'unknown')}")

# Ensure project root on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from src.models.multimodal_llm import ProteinLLM

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Protein-LLM Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to protein_llm checkpoint dir")
    parser.add_argument("--sequence", required=True, help="Protein amino acid sequence")
    parser.add_argument("--prompt", default="Describe the function of this protein.",
                        help="Question about the protein")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load model
    log.info(f"Loading model from {args.checkpoint}")
    model = ProteinLLM.from_pretrained(args.checkpoint, device=args.device)
    model.eval()
    log.info(f"Model loaded: approach={model.approach}, projector_type={model.projector_type}")

    # Build prompt with protein placeholder
    if model.approach in ("esm3",):
        user_content = f"{args.prompt}\n<|protein_start|><|protein_embed|><|protein_end|>"
    else:
        user_content = f"{args.prompt}\n<protein>{args.sequence}</protein>"

    prompt = model.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a protein science expert."},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Generate
    log.info(f"Protein: {args.sequence[:50]}{'...' if len(args.sequence) > 50 else ''}")
    log.info(f"Prompt: {args.prompt}")

    outputs = model.generate(
        protein_sequences=[args.sequence],
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True,
    )

    print(f"\n{'='*60}")
    print("Response:")
    print(f"{'='*60}")
    print(outputs[0])


if __name__ == "__main__":
    main()
