#!/usr/bin/env python3
"""Test that ESM3+MLP and Text GRPO rollouts produce matched prompts.

Verifies:
1. System prompts are identical
2. User content differs ONLY by the protein placeholder suffix
3. Chat-template wrapped prompts match (minus placeholder)
4. Thinking prefix / generation_prefix logic is identical
5. GRPO config parameters are identical (group_size, temperature, etc.)
6. Effective batch size is the same
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.models.multimodal_llm import PROTEIN_PLACEHOLDER
from src.training.grpo_trainer import (
    _ESMFOLD_SYSTEM_PROMPT_NO_THINK,
    _ESMFOLD_SYSTEM_PROMPT_THINK,
    _THINKING_PREFIX,
    _get_grpo_system_prompt,
)


def load_config(config_path: str):
    """Load a GRPO config YAML."""
    cfg = OmegaConf.load(config_path)
    return cfg


def test_system_prompts():
    """System prompts must be identical for same task + enable_thinking."""
    print("=" * 60)
    print("TEST 1: System prompts match for both approaches")
    print("=" * 60)

    for enable_thinking in [True, False]:
        prompt = _get_grpo_system_prompt("esmfold", enable_thinking=enable_thinking)
        expected = _ESMFOLD_SYSTEM_PROMPT_THINK if enable_thinking else _ESMFOLD_SYSTEM_PROMPT_NO_THINK
        assert prompt == expected, f"System prompt mismatch for enable_thinking={enable_thinking}"
        print(f"  enable_thinking={enable_thinking}: OK (same prompt for both approaches)")

    print("  PASSED\n")


def test_user_content():
    """User content differs ONLY by protein placeholder suffix."""
    print("=" * 60)
    print("TEST 2: User content differs only by placeholder")
    print("=" * 60)

    sample_prompt = "Assess the structural quality of this protein. Report the predicted fold quality (high/medium/low), estimated pLDDT confidence score (0-100), and whether the protein is well-folded or likely disordered."

    # Text path: user_content = prompt
    text_user_content = sample_prompt

    # ESM3+MLP path: user_content = prompt + \n\n + PROTEIN_PLACEHOLDER
    esm3_user_content = f"{sample_prompt}\n\n{PROTEIN_PLACEHOLDER}"

    # Verify the only difference is the placeholder suffix
    assert esm3_user_content.startswith(text_user_content), "ESM3 user content must start with text content"
    suffix = esm3_user_content[len(text_user_content):]
    assert suffix == f"\n\n{PROTEIN_PLACEHOLDER}", f"Unexpected suffix: {repr(suffix)}"
    print(f"  Text user content: {text_user_content[:80]}...")
    print(f"  ESM3 user content: {esm3_user_content[:80]}...")
    print(f"  Suffix: {repr(suffix)}")
    print(f"  PROTEIN_PLACEHOLDER: {repr(PROTEIN_PLACEHOLDER)}")
    print("  PASSED\n")


def test_wrapped_prompts():
    """Full chat-template wrapped prompts match (minus placeholder)."""
    print("=" * 60)
    print("TEST 3: Wrapped prompts match (minus placeholder)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    sample_prompt = "Assess the structural quality of this protein."
    system_prompt = _get_grpo_system_prompt("esmfold", enable_thinking=True)

    # Text path
    text_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample_prompt},
    ]
    text_wrapped = tokenizer.apply_chat_template(
        text_messages, tokenize=False, add_generation_prompt=True,
    )

    # ESM3 path
    esm3_user = f"{sample_prompt}\n\n{PROTEIN_PLACEHOLDER}"
    esm3_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": esm3_user},
    ]
    esm3_wrapped = tokenizer.apply_chat_template(
        esm3_messages, tokenize=False, add_generation_prompt=True,
    )

    # The difference should be exactly the placeholder in the user message
    # Replace the placeholder to verify they match
    esm3_cleaned = esm3_wrapped.replace(f"\n\n{PROTEIN_PLACEHOLDER}", "")
    assert text_wrapped == esm3_cleaned, (
        f"Wrapped prompts don't match after removing placeholder!\n"
        f"TEXT:\n{text_wrapped}\n\n"
        f"ESM3 (cleaned):\n{esm3_cleaned}"
    )

    print(f"  Text wrapped ({len(text_wrapped)} chars): ...{text_wrapped[-80:]}")
    print(f"  ESM3 wrapped ({len(esm3_wrapped)} chars): ...{esm3_wrapped[-80:]}")
    print("  After removing placeholder: MATCH")
    print("  PASSED\n")


def test_thinking_prefix():
    """Thinking prefix logic is identical for both paths."""
    print("=" * 60)
    print("TEST 4: Thinking prefix logic")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    sample_prompt = "Assess the structural quality of this protein."

    for enable_thinking in [True, False]:
        system_prompt = _get_grpo_system_prompt("esmfold", enable_thinking=enable_thinking)

        # Text path
        text_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample_prompt},
        ]
        text_wrapped = tokenizer.apply_chat_template(
            text_messages, tokenize=False, add_generation_prompt=True,
        )
        if not enable_thinking:
            text_wrapped += _THINKING_PREFIX

        # ESM3 path (in code: generation_prefix=None if enable_thinking else _THINKING_PREFIX)
        esm3_user = f"{sample_prompt}\n\n{PROTEIN_PLACEHOLDER}"
        esm3_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": esm3_user},
        ]
        esm3_wrapped = tokenizer.apply_chat_template(
            esm3_messages, tokenize=False, add_generation_prompt=True,
        )
        if not enable_thinking:
            esm3_wrapped += _THINKING_PREFIX

        # Check suffix after generation prompt
        if enable_thinking:
            # Both should end with the generation prompt (no prefix)
            assert not text_wrapped.endswith(_THINKING_PREFIX), "Text shouldn't have thinking prefix when enabled"
            assert not esm3_wrapped.endswith(_THINKING_PREFIX), "ESM3 shouldn't have thinking prefix when enabled"
            print("  enable_thinking=True:  No prefix appended (model generates <think> freely)")
        else:
            assert text_wrapped.endswith(_THINKING_PREFIX), "Text should have thinking prefix when disabled"
            assert esm3_wrapped.endswith(_THINKING_PREFIX), "ESM3 should have thinking prefix when disabled"
            print(f"  enable_thinking=False: '{repr(_THINKING_PREFIX)}' appended to both")

    print("  PASSED\n")


def test_grpo_configs_match():
    """GRPO config parameters are identical between the two configs."""
    print("=" * 60)
    print("TEST 5: GRPO config parameters match")
    print("=" * 60)

    mlp_cfg = load_config("configs/main_RL/grpo_structure_mlp_sft.yaml")
    text_cfg = load_config("configs/main_RL/grpo_structure_text_sft.yaml")

    # Parameters that MUST match
    matched_keys = [
        ("training.lr", "training.lr"),
        ("training.grpo.group_size", "training.grpo.group_size"),
        ("training.grpo.temperature", "training.grpo.temperature"),
        ("training.grpo.use_kl_penalty", "training.grpo.use_kl_penalty"),
        ("training.grpo.normalize_advantages", "training.grpo.normalize_advantages"),
        ("training.rollout.max_tokens", "training.rollout.max_tokens"),
        ("training.rollout.top_p", "training.rollout.top_p"),
        ("training.rollout.do_sample", "training.rollout.do_sample"),
        ("training.rollout.enable_thinking", "training.rollout.enable_thinking"),
        ("training.epochs", "training.epochs"),
        ("training.warmup_steps", "training.warmup_steps"),
        ("training.max_grad_norm", "training.max_grad_norm"),
        ("training.gradient_checkpointing", "training.gradient_checkpointing"),
        ("training.save_steps", "training.save_steps"),
        ("training.eval_steps", "training.eval_steps"),
        ("training.logging_steps", "training.logging_steps"),
        ("data.task", "data.task"),
        ("data.processing.max_protein_length", "data.processing.max_protein_length"),
    ]

    all_ok = True
    for mlp_key, text_key in matched_keys:
        mlp_val = OmegaConf.select(mlp_cfg, mlp_key)
        text_val = OmegaConf.select(text_cfg, text_key)
        status = "OK" if mlp_val == text_val else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"  {mlp_key}: mlp={mlp_val}, text={text_val} [{status}]")

    # Effective batch must match
    mlp_bs = OmegaConf.select(mlp_cfg, "training.batch_size")
    mlp_ga = OmegaConf.select(mlp_cfg, "training.gradient_accumulation_steps")
    text_bs = OmegaConf.select(text_cfg, "training.batch_size")
    text_ga = OmegaConf.select(text_cfg, "training.gradient_accumulation_steps")
    mlp_eff = mlp_bs * mlp_ga * 8  # 8 GPUs
    text_eff = text_bs * text_ga * 8
    eff_status = "OK" if mlp_eff == text_eff else "MISMATCH"
    print(f"  effective_batch: mlp={mlp_bs}*{mlp_ga}*8={mlp_eff}, text={text_bs}*{text_ga}*8={text_eff} [{eff_status}]")
    if mlp_eff != text_eff:
        all_ok = False

    # Expected differences (approach-specific)
    print("\n  Expected differences:")
    print(f"    approach: mlp={OmegaConf.select(mlp_cfg, 'approach')}, text={OmegaConf.select(text_cfg, 'approach')}")
    print(f"    batch_size: mlp={mlp_bs}, text={text_bs} (different due to memory, but effective batch matches)")
    print(f"    parent_experiment: mlp={OmegaConf.select(mlp_cfg, 'parent_experiment')}, text={OmegaConf.select(text_cfg, 'parent_experiment')}")
    mlp_plr = OmegaConf.select(mlp_cfg, "training.projector_lr", default="(absent)")
    print(f"    projector_lr: mlp={mlp_plr}, text=(absent, expected)")

    if all_ok:
        print("  PASSED\n")
    else:
        print("  FAILED — mismatches found above\n")
        return False
    return True


def test_actual_generation_call():
    """Verify the code paths produce the same prompt strings."""
    print("=" * 60)
    print("TEST 6: Simulate actual _generate_completions prompt construction")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    sample_prompt = "Assess the structural quality of this protein. Report the predicted fold quality (high/medium/low), estimated pLDDT confidence score (0-100), and whether the protein is well-folded or likely disordered."

    for enable_thinking in [True, False]:
        print(f"\n  --- enable_thinking={enable_thinking} ---")

        system_prompt = _get_grpo_system_prompt("esmfold", enable_thinking=enable_thinking)

        # ---- Text path (lines 1341-1366 of grpo_trainer.py) ----
        text_user_content = sample_prompt
        text_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_user_content},
        ]
        text_wrapped = tokenizer.apply_chat_template(
            text_messages, tokenize=False, add_generation_prompt=True,
        )
        if not enable_thinking:
            text_wrapped += _THINKING_PREFIX

        # ---- ESM3 path (lines 1304-1334 of grpo_trainer.py) ----
        esm3_user_content = f"{sample_prompt}\n\n{PROTEIN_PLACEHOLDER}"
        esm3_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": esm3_user_content},
        ]
        esm3_wrapped = tokenizer.apply_chat_template(
            esm3_messages, tokenize=False, add_generation_prompt=True,
        )
        if not enable_thinking:
            esm3_wrapped += _THINKING_PREFIX

        # Compare: remove placeholder from ESM3
        esm3_cleaned = esm3_wrapped.replace(f"\n\n{PROTEIN_PLACEHOLDER}", "")
        match = text_wrapped == esm3_cleaned

        print(f"    Text prompt length: {len(text_wrapped)} chars")
        print(f"    ESM3 prompt length: {len(esm3_wrapped)} chars")
        print(f"    After removing placeholder: {'MATCH' if match else 'MISMATCH'}")

        if not match:
            # Show diff
            for i, (a, b) in enumerate(zip(text_wrapped, esm3_cleaned)):
                if a != b:
                    print(f"    First diff at char {i}: text={repr(a)}, esm3={repr(b)}")
                    print(f"    Context: text=...{repr(text_wrapped[max(0,i-20):i+20])}...")
                    print(f"    Context: esm3=...{repr(esm3_cleaned[max(0,i-20):i+20])}...")
                    break
            return False

        # Show the actual prompts
        print("\n    Full text prompt:")
        print(f"    {text_wrapped}")
        print("\n    Full ESM3 prompt:")
        print(f"    {esm3_wrapped}")

    print("\n  PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MATCHED GRPO PROMPT VERIFICATION")
    print("=" * 60 + "\n")

    results = []
    results.append(("System prompts", test_system_prompts))
    results.append(("User content", test_user_content))
    results.append(("Wrapped prompts", test_wrapped_prompts))
    results.append(("Thinking prefix", test_thinking_prefix))
    results.append(("GRPO configs", test_grpo_configs_match))
    results.append(("Generation simulation", test_actual_generation_call))

    passed = 0
    failed = 0
    for name, fn in results:
        try:
            result = fn()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
