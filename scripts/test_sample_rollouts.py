#!/usr/bin/env python3
"""Sample actual rollouts from both ESM3+MLP and Text SFT checkpoints.

Loads both parent SFT models, runs generation with the GRPO system prompt
and enable_thinking=True, and prints the raw completions side-by-side.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.multimodal_llm import PROTEIN_PLACEHOLDER, ProteinLLM
from src.training.grpo_trainer import (
    _THINKING_PREFIX,
    _get_grpo_system_prompt,
)

# ── Configs ──────────────────────────────────────────────────
MLP_EXPERIMENT_DIR = "results/sft_esm3_mlp_combined_qwen3_8b_it_0306_010408"
MLP_SFT_CKPT = f"{MLP_EXPERIMENT_DIR}/checkpoints/checkpoint-9750"
TEXT_SFT_CKPT = "results/sft_text_combined_qwen3_8b_it_0307_190324/checkpoints/checkpoint-9647"
BASE_MODEL = "Qwen/Qwen3-8B"
TASK = "esmfold"
ENABLE_THINKING = False
NUM_SAMPLES = 3
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.95

# Sample proteins (short, medium, long-ish)
SAMPLE_PROTEINS = [
    "MKTIIALSYIFCLVFA",  # short signal peptide
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",  # hemoglobin fragment
    "MHHHHHHENLYFQGSMTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",  # His-tagged GFP fragment
]

SAMPLE_INSTRUCTION = (
    "Assess the structural quality of this protein. Report the predicted "
    "fold quality (high/medium/low), estimated pLDDT confidence score (0-100), "
    "and whether the protein is well-folded or likely disordered."
)


def load_text_model(device="cuda:0"):
    """Load text-only SFT checkpoint."""
    print(f"\n{'='*60}")
    print(f"Loading TEXT SFT model from {TEXT_SFT_CKPT}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Text SFT was trained with original vocab (no protein tokens added).
    # Model config has vocab_size=151936 but actual weights are len(tokenizer)=151669.
    # Resize to match tokenizer so LoRA adapter loads cleanly.
    model.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, TEXT_SFT_CKPT)
    model = model.to(device)
    model.eval()
    print(f"  Text model loaded on {device}")
    return model, tokenizer


def load_mlp_model(device="cuda:0"):
    """Load ESM3+MLP SFT checkpoint via ProteinLLM.from_fsdp_checkpoint.

    Intermediate FSDP checkpoints (checkpoint-9750) don't have config.json,
    so we use from_fsdp_checkpoint which reads the experiment's config.yaml.
    """
    print(f"\n{'='*60}")
    print(f"Loading ESM3+MLP SFT model from {MLP_SFT_CKPT}")
    print(f"{'='*60}")

    protein_llm = ProteinLLM.from_fsdp_checkpoint(
        experiment_dir=MLP_EXPERIMENT_DIR,
        checkpoint_dir=MLP_SFT_CKPT,
        device=device,
    )
    print(f"  MLP model loaded on {device}")
    return protein_llm


def generate_text_rollout(model, tokenizer, prompt, enable_thinking, device="cuda:0"):
    """Generate using text-only path (mirrors grpo_trainer lines 1341-1376)."""
    system_prompt = _get_grpo_system_prompt(TASK, enable_thinking=enable_thinking)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if not enable_thinking:
        wrapped += _THINKING_PREFIX

    inputs = tokenizer(wrapped, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=False)
    return wrapped, completion


def generate_mlp_rollout(protein_llm, protein_seq, prompt, enable_thinking):
    """Generate using ESM3+MLP path (mirrors grpo_trainer lines 1321-1334)."""
    system_prompt = _get_grpo_system_prompt(TASK, enable_thinking=enable_thinking)

    # Path 1: Using the GRPO _generate_completions logic (wrap_chat_template=False)
    # Build wrapped prompt manually like GRPO does
    tokenizer = protein_llm.tokenizer
    user_content = f"{prompt}\n\n{PROTEIN_PLACEHOLDER}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if not enable_thinking:
        wrapped += _THINKING_PREFIX

    # Generate using ProteinLLM.generate with pre-wrapped prompt
    with torch.no_grad():
        texts = protein_llm.generate(
            protein_sequences=[protein_seq],
            prompt=[wrapped],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            use_cache=True,
            wrap_chat_template=False,  # already wrapped
        )

    return wrapped, texts[0]


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("SAMPLE ROLLOUT COMPARISON: TEXT vs ESM3+MLP")
    print(f"enable_thinking={ENABLE_THINKING}, task={TASK}")
    print(f"temperature={TEMPERATURE}, top_p={TOP_P}, max_new_tokens={MAX_NEW_TOKENS}")
    print("=" * 60)

    system_prompt = _get_grpo_system_prompt(TASK, enable_thinking=ENABLE_THINKING)
    print(f"\n{'='*60}")
    print("SYSTEM PROMPT (shared):")
    print(f"{'='*60}")
    print(system_prompt)

    # Collect results per sample
    text_results = []
    mlp_results = []

    # ── Phase 1: Text model rollouts ──
    print(f"\n{'='*60}")
    print("PHASE 1: TEXT MODEL ROLLOUTS")
    print(f"{'='*60}")
    text_model, tokenizer = load_text_model(device)

    for i, protein_seq in enumerate(SAMPLE_PROTEINS[:NUM_SAMPLES]):
        text_prompt = f"{SAMPLE_INSTRUCTION}\n\n{protein_seq}"
        wrapped, completion = generate_text_rollout(
            text_model, tokenizer, text_prompt, ENABLE_THINKING, device
        )
        text_results.append((protein_seq, wrapped, completion))
        print(f"\n  Sample {i+1} ({len(protein_seq)} AA): done")

    # Free text model before loading MLP
    del text_model, tokenizer
    torch.cuda.empty_cache()
    print("  Text model freed from GPU")

    # ── Phase 2: MLP model rollouts ──
    print(f"\n{'='*60}")
    print("PHASE 2: ESM3+MLP MODEL ROLLOUTS")
    print(f"{'='*60}")
    mlp_model = load_mlp_model(device)

    for i, protein_seq in enumerate(SAMPLE_PROTEINS[:NUM_SAMPLES]):
        wrapped, completion = generate_mlp_rollout(
            mlp_model, protein_seq, SAMPLE_INSTRUCTION, ENABLE_THINKING
        )
        mlp_results.append((protein_seq, wrapped, completion))
        print(f"\n  Sample {i+1} ({len(protein_seq)} AA): done")

    del mlp_model
    torch.cuda.empty_cache()
    print("  MLP model freed from GPU")

    # ── Phase 3: Print comparison ──
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")

    for i in range(len(text_results)):
        protein_seq = text_results[i][0]
        text_wrapped, text_completion = text_results[i][1], text_results[i][2]
        mlp_wrapped, mlp_completion = mlp_results[i][1], mlp_results[i][2]

        print(f"\n{'#'*60}")
        print(f"SAMPLE {i+1}: protein={protein_seq[:30]}... ({len(protein_seq)} AA)")
        print(f"{'#'*60}")

        print("\n--- TEXT PATH ---")
        print(f"  Wrapped prompt ({len(text_wrapped)} chars):")
        print(f"    ...{text_wrapped[-120:]}")
        print("  Completion:")
        print(f"    {repr(text_completion)}")

        print("\n--- ESM3+MLP PATH ---")
        print(f"  Wrapped prompt ({len(mlp_wrapped)} chars):")
        print(f"    ...{mlp_wrapped[-120:]}")
        print("  Completion:")
        print(f"    {repr(mlp_completion)}")

        # ── Compare ──
        print("\n--- COMPARISON ---")
        for label, comp in [("Text", text_completion), ("MLP", mlp_completion)]:
            has_think = "<think>" in comp
            has_answer = "<answer>" in comp
            think_empty = "<think>\n</think>" in comp or "<think>\n\n</think>" in comp
            # Extract think content if present
            think_content = ""
            if "<think>" in comp and "</think>" in comp:
                think_content = comp.split("<think>")[1].split("</think>")[0].strip()
            think_len = len(think_content)
            print(f"  {label:5s}: has_think={has_think}, think_empty={think_empty}, "
                  f"think_len={think_len}, has_answer={has_answer}")
            if think_content:
                print(f"         think: {repr(think_content[:150])}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
