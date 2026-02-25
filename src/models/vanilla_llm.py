"""
Vanilla LLM Wrapper for baseline evaluation.

Wraps a HuggingFace causal LM to match ProteinLLM's generate() and
compute_loss() interface so that evaluation code can treat vanilla and
fine-tuned models interchangeably.
"""

import logging
from typing import List, Union

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.mol_instructions import DEFAULT_SYSTEM_PROMPT

log = logging.getLogger(__name__)


class VanillaLLMWrapper:
    """Wrapper around a HuggingFace causal LM matching ProteinLLM's interface.

    The downstream evaluation prompts (create_go_prompt, create_ppi_prompt,
    create_stability_prompt) already embed protein sequences in the prompt
    text, so the ``protein_sequences`` parameter in ``generate()`` is ignored.
    """

    def __init__(self, model_name: str, dtype=torch.bfloat16):
        log.info(f"Loading vanilla LLM: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm.eval()
        self.device = next(self.llm.parameters()).device
        log.info(f"Vanilla LLM loaded on {self.device}")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "VanillaLLMWrapper":
        """Create from Hydra config (uses ``cfg.model.path``)."""
        model_path = cfg.model.path
        return cls(model_name=model_path)

    def eval(self):
        self.llm.eval()
        return self

    def generate(
        self,
        protein_sequences: List[str],
        prompt: Union[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> List[str]:
        """Generate text responses.

        ``protein_sequences`` is accepted for API compatibility but ignored;
        the sequence is already embedded in the prompt text by the evaluation
        helpers.
        """
        if isinstance(prompt, str):
            prompt = [prompt] * len(protein_sequences)

        results = []
        for p in prompt:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]

            # Build input via chat template when available.
            # enable_thinking=False is Qwen3-specific (disables reasoning);
            # non-Qwen tokenizers raise TypeError, caught below.
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Non-Qwen models (Llama, Mistral, etc.)
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )

            generated_ids = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(response.strip())

        return results

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Forward pass with labels, returning scalar loss for perplexity."""
        with torch.no_grad():
            outputs = self.llm(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                labels=labels.to(self.device),
            )
        return outputs.loss.item()
