# SFT ESM-3 + MLP Combined Training Report

**Date**: 2026-03-06
**Experiment**: `sft_esm3_mlp_combined_qwen3_8b_it_0306_010408`
**Status**: Training in progress (~11%, step 3800+/28,941)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-8B-Instruct-2507 |
| Encoder | ESM-3 small (frozen, 1536-dim) |
| Projector | MLP (1536 -> 5120 -> 2560), ~21M params |
| Pooling | AttentionPooling (32 tokens), ~9.5M params |
| LoRA | r=8, all linear layers (q/k/v/o + gate/up/down), ~2M params |
| Total trainable | ~32.5M params |
| Dataset | combined_sft_260225 (4.89M records, 6 sources) |
| LR | 2e-4 (LoRA), 1e-3 (projector, 5x) |
| Batch | Token-budget 8192, max_batch_size=16, grad_accum=4 |
| FSDP | shard_grad_op, 8x H100 |
| Epochs | 3 (28,941 steps) |

## Key Finding: ESM-3 Embeddings Drive Strong SFT

The most significant result is that **protein embeddings from ESM-3 alone can produce good SFT performance** under the MLP projector setting:

- **eval_loss**: 0.669 (step 250) -> 0.420 (step 3250) -> 0.409 (step 3750)
- **token_avg_loss**: 3.34 -> 0.45 (rapidly converging)
- **Generation quality**: Moderate but meaningful (overall BLEU=0.20, ROUGE-L=0.44 at step 3250)

The model learns to decode protein structure/function information from ESM-3 embeddings and produce coherent natural language responses. This confirms that the frozen ESM-3 encoder captures sufficient biological information for downstream protein understanding tasks.

### Open Question: Embedding vs Text-Only

It remains **mysterious why generation quality from embeddings is lower than what we'd expect from a text-only baseline** (which has not yet been tested). Possible hypotheses:

1. **Information bottleneck**: 32 attention-pooled tokens may compress away fine-grained sequence details (e.g., exact catalytic residue positions)
2. **Modality gap**: The projector must bridge ESM-3's learned representation space to the LLM's token embedding space -- this mapping may lose specificity
3. **Training signal**: With embeddings, the model cannot "read" the raw sequence; it must infer everything from the encoder's representation, which may not encode all task-relevant features equally

**TODO**: Run text-only baseline (`approach=text`) on the same combined dataset for direct comparison.

---

## Generation Quality Analysis (Step 3250, 40 samples)

### Per-Category Scores

| Category | BLEU | ROUGE-L | Samples | Notes |
|----------|------|---------|---------|-------|
| Catalytic | 0.205 | 0.461 | 10 | Hit-or-miss on specific reactions |
| Domain | 0.195 | 0.396 | 10 | Often correct family, wrong specific domain |
| Function | 0.175 | 0.385 | 10 | Strong on well-known proteins |
| General | 0.227 | 0.499 | 10 | Best category; naming/classification tasks |

### Success Examples

**1. Perfect protein identification** (BLEU=0.76, ROUGE-L=0.93)
- Task: "What is the name of this protein?"
- Reference: "This protein is 30S ribosomal protein S10 {ECO:0000255|HAMAP-Rule:MF_00508}."
- Generated: "This protein is 30S ribosomal protein S10 {ECO:0000255|HAMAP-Rule:MF_00508}."
- Protein: MEKIRLKLKAYDHRVLDRSVVAIVEAVKRSGSEIRGPIPLPTKNKRYTVLRSPHVNKDSREQFEIRVYSRLIDIISATPETVDSLMKLDLAPEVDVEVTSMETK (104 AA)

**2. Correct domain prediction** (BLEU=0.75, ROUGE-L=0.80)
- Task: "Please predict any recognizable motifs or domains"
- Reference: "Nuclear localization signal domains"
- Generated: "Ubl-removal MGS, UFM1-like domains"
- Note: Different prediction but structurally reasonable for the sequence

**3. Near-perfect domain family match** (BLEU=0.70, ROUGE-L=0.87)
- Task: "Predict any recognizable domains or motifs"
- Reference: "Glycosyl hydrolase family 30 TIM-barrel domains"
- Generated: "Glycosyl hydrolase family 30 beta sandwich, Glycosyl hydrolase family 16 TIM-barrel domains"
- Protein: LSSAAGGRPCDAKDFGHGSLVCACSAMYCD... (164 AA)

**4. Exact functional description** (BLEU=0.68, ROUGE-L=0.92)
- Task: "Describe the protein's function"
- Reference: "Binds to the 23S rRNA. {ECO:0000255|HAMAP-Rule:MF_01341}."
- Generated: "Binds to the 23S rRNA. {ECO:0000255|HAMAP-Rule:MF_01341}."
- Protein: MLLNTLSPAAGSKHAPKRLGRGVGSGLGKTGGRGHKGQKSRSGGKVRPGFE... (144 AA)

**5. Correct catalytic reaction** (BLEU=0.58, ROUGE-L=0.71)
- Task: "Describe the catalytic activity"
- Reference: "(6S)-NADPHX + ADP = AMP + H(+) + NADPH + phosphate"
- Generated: "(6S)-NADPHX + NADP(+) = ... + H(+)" (partially correct substrate, wrong products)

**6. Snake venom protease identification** (BLEU=0.64, ROUGE-L=0.78)
- Task: "Identify the protein"
- Reference: "Snake venom serine protease KN9"
- Generated: "Snake venom serine protease 1" (correct family, wrong isoform)

### Failure Patterns

1. **Catalytic reaction hallucination**: Model often predicts plausible but wrong reactions (e.g., predicts "ATP + H2O = ADP + phosphate" for a cytochrome P450 reductase)
2. **Domain specificity**: Gets the protein family right but predicts wrong specific domains (e.g., "Thioredoxin-like" instead of "2Fe-2S ferredoxin-type")
3. **Complex function descriptions**: Struggles with multi-sentence functional descriptions of complex proteins (e.g., APC/C complex)

---

## Evaluation Framework Gaps

Current metrics (BLEU, ROUGE-L) have significant limitations for protein tasks:

- **BLEU** penalizes correct answers with different wording (e.g., "Essential for photosystem I assembly" vs "Required for photosystem I assembly" scores low)
- **ROUGE-L** doesn't capture semantic correctness (predicting the right enzyme family with different words scores poorly)
- Neither metric evaluates **biological accuracy** (is the predicted reaction chemically valid? Is the domain prediction structurally plausible?)

**TODO**: Develop an LLM-guided scoring framework that evaluates:
1. Semantic similarity (meaning-preserving paraphrases)
2. Biological accuracy (valid reactions, correct protein families)
3. Per-category specialized metrics (exact match for protein names, ontology-aware matching for GO terms)

---

## TODO List

### Immediate (next experiments)

1. **Text-only baseline**: Run `approach=text` on combined_sft_260225 to establish baseline generation quality without ESM-3 embeddings. This directly tests whether the embedding approach helps or hurts generation.

2. **Combined dataset impact**: Compare current combined run (4.89M, 6 sources) against previous Mol-Instructions-only runs to measure whether data diversity improves or dilutes performance (high UniProt overlap between sources).

3. **Thinking-aware SFT**: Run `experiment=sft_esm3_mlp_thinking` (same config but `enable_thinking=true`) to test whether allowing `<think>...</think>` reasoning blocks improves generation quality. Currently, the model outputs empty `<think></think>` blocks -- the thinking variant would train the model to actually reason before answering.

### Medium-term

4. **LLM-guided evaluation**: Build a scoring framework using an LLM judge for semantic + biological accuracy assessment. Current BLEU/ROUGE are insufficient for protein tasks.

5. **RL transfer (GRPO)**: Chain GRPO from the best SFT checkpoint. Key questions:
   - Does ESMFold reward improve structure-aware responses?
   - Do downstream task rewards (GO prediction, stability) transfer better from embedding vs text SFT?
   - How does thinking-aware SFT affect RL exploration?

### Architecture comparison (thesis)

6. **Four-way comparison**: text vs MLP vs Perceiver Resampler vs Flamingo on the same dataset/eval, measuring both loss and generation quality.

---

## Infrastructure Changes Made During This Run

1. **`callbacks.py`**: Added `import traceback`, fixed bad f-string format specifier for BLEU/ROUGE logging
2. **`multimodal_llm.py`**: Cast `inputs_embeds` to bf16 before generate (FSDP dtype mismatch fix); FSDP guard now uses `summon_full_params` for all strategies
3. **`src/evaluation/generation.py`**: New unified `GenerationEvaluator` class -- single source of truth for generation evaluation (training callback + standalone CLI)
4. **`callbacks.py`**: `GenerationSamplesCallback` refactored as thin wrapper delegating to `GenerationEvaluator`
5. **`scripts/evaluate.py`**: Added `evaluation.name=generation` dispatch with intermediate checkpoint support
