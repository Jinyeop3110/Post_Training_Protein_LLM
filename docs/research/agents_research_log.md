# Multi-Agent Research & Development Log

## Project: Post-Training Protein LLM

**Goal**: Train/post-train existing LLMs to understand proteins as multimodal embeddings using SFT (mid-training) and RL (problem-specific training).

**Repository**: `/home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM`

---

## Quick Links

| Section | Description |
|---------|-------------|
| [Development Log](#development-log) | Implementation progress & milestones |
| [Results & Metrics](#results--metrics) | Training results & benchmarks |
| [Decisions](#decision-log) | Key decisions & rationale |
| [Research Findings](#agent-interactions-log) | Literature review |
| [Action Items](#action-items) | TODO list |

---

---

## Literature Review: Multimodal Protein-LLM Systems (2024-2026)

### 2026-03-02: Comprehensive Survey of Frozen-Encoder Protein-LLM Architectures

**Objective**: Catalog published implementations of multimodal protein-LLM systems that use frozen protein encoders with LLM backbones, directly comparable to our ESM-3 + Qwen3 approach.

---

#### 1. EvoLlama (Dec 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Enhancing LLMs' Understanding of Proteins via Multimodal Structure and Sequence Representations" |
| **Venue** | arXiv 2412.11618 / OpenReview |
| **Protein Encoder** | ESM-2 650M (sequence) + ProteinMPNN (structure) |
| **LLM Backbone** | LLaMA-3-8B |
| **Projection** | MLP (separate MLPs for sequence and structure features; combined via element-wise addition) |
| **Encoder Frozen?** | Stage 1: both encoders + LLM frozen (only MLP trained). Stage 2: LLM frozen, encoders + MLP updated |
| **LoRA** | None -- full parameter updates for projection and encoders in stage 2 |
| **Training Data** | 369K proteins (Swiss-Prot) for projection; ~460K for SFT (Mol-Instructions + PEER) |
| **Trainable Params** | 690-720M (7.9-8.2% of 8.8B total) |
| **Key Results** | Zero-shot: +1-8% over fine-tuned baselines; SFT: +6% avg over SOTA on Mol-Instructions; 62-64% avg on PEER tasks |
| **GitHub** | Not publicly released (as of search date) |
| **Notes** | Structure+sequence fusion via element-wise addition reduces tokens by ~50%, improves latency ~20% |

---

#### 2. ProteinGPT (Aug 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Multimodal LLM for Protein Property Prediction and Structure Understanding" |
| **Venue** | ICLR 2025 / arXiv 2408.11363 |
| **Protein Encoder** | ESM-2 3B (sequence) + ESM-IF1 GVP 142M (structure/inverse folding) |
| **LLM Backbone** | Vicuna, LLaMA-2, LLaMA-3, Mistral (Mistral best performer) |
| **Projection** | Linear projection layers (soft prompts for LLM) |
| **Encoder Frozen?** | Both encoders frozen throughout both training stages |
| **LoRA** | Not mentioned |
| **Training Data** | ProteinQA: 132K proteins from RCSB-PDB, ~40 QA pairs each |
| **Key Results** | Mistral: BERTScore 0.821, PubMedBERT 0.758; 70-80% closed-ended accuracy |
| **GitHub** | https://github.com/OviaLabs/ProteinGPT |
| **Notes** | Two-stage: modality alignment (MA) then instruction tuning (IT). GPT-4o used for dataset construction |

---

#### 3. ProtChatGPT (Feb 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Towards Understanding Proteins with Large Language Models" |
| **Venue** | SIGIR 2025 / arXiv 2402.09649 |
| **Protein Encoder** | ESM-1b (sequence, 768-dim) + ESM-IF1 (structure, 512-dim) |
| **LLM Backbone** | Vicuna-13B |
| **Projection** | PLP-Former (32 learnable query tokens, 768-dim, initialized from PubMedBERT; self-attn + cross-attn) + FC projection layers |
| **Encoder Frozen?** | Both encoders + LLM frozen throughout training |
| **LoRA** | Not mentioned |
| **Training Data** | Stage 1: ProtDescribe 553K protein-text pairs; Stage 2: RCSB-PDB 143K structure-description pairs |
| **Key Results** | SPICE 0.316, PubMed BERTScore 0.457 |
| **GitHub** | Not publicly available |
| **Notes** | Three pre-training objectives in PLP-Former: contrastive learning, text generation, matching. BLIP-2-inspired design |

---

#### 4. Prot2Chat (Feb 2025)

| Component | Details |
|-----------|---------|
| **Paper** | "Protein LLM with Early-Fusion of Text, Sequence and Structure" |
| **Venue** | Bioinformatics (Oxford) 2025 / arXiv 2502.06846 |
| **Protein Encoder** | Modified ProteinMPNN (9 released models concatenated, output dim 1152 = 128x9) |
| **LLM Backbone** | LLaMA-3-8B-Instruct |
| **Projection** | Text-aware protein-text adapter: 256 learnable queries + cross-attention with question vector from LLM |
| **Encoder Frozen?** | Encoder frozen; LLM fine-tuned with LoRA |
| **LoRA** | r=8, alpha=16, target: q_proj + v_proj, dropout=0.1 |
| **Training Data** | Mol-Instructions 404K train + UniProtQA 25K train |
| **Trainable Params** | 109M total (adapter 106.5M + LoRA 3.4M) |
| **Key Results** | BLEU-2: 35.85, ROUGE-1: 57.21, ROUGE-L: 50.51 (vs Evolla-10B: 8.69/29.09/20.04) |
| **GitHub** | Not found |
| **Notes** | Key innovation: early fusion -- question text informs adapter before LLM generation. Only 109M trainable params vs 1.7B (Evolla) or 3B (BioMedGPT) |

---

#### 5. Evolla / ProteinChat (Jan 2025)

| Component | Details |
|-----------|---------|
| **Paper** | "Decoding the Molecular Language of Proteins with Evolla" |
| **Venue** | bioRxiv 2025.01.05.630192 |
| **Protein Encoder** | SaProt-650M (Evolla-10B) or SaProt-1.3B (Evolla-80B) |
| **LLM Backbone** | LLaMA-3-8B (10B version), LLaMA-3-70B (80B version) |
| **Projection** | Cross-attention Transformer: Sequence Compressor (variable->fixed length) + Sequence Aligner (injected between LLM layers with learnable gates) |
| **Encoder Frozen?** | Both SaProt + LLaMA frozen; only compressor + aligner trained |
| **LoRA** | None -- selective freezing instead |
| **Training Data** | 546M protein-QA triples, 150B word tokens (Swiss-Prot + ProTrek-annotated) |
| **Trainable Params** | 1.7B (Evolla-10B), 8.2B (Evolla-80B) |
| **Key Results** | GPT score 74.10 (vs GPT-4o 37.07, DeepSeek-v3 40.49); EC prediction 41.2% 4-digit match |
| **GitHub** | https://github.com/westlake-repl/Evolla |
| **Notes** | Largest protein-LLM training set. RAG at inference (DQS/QGS strategies). Chose SaProt over ESM-2 based on ablation (72.41 vs lower) |

---

#### 6. ProtT3 (May 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Protein-to-Text Generation for Text-based Protein Understanding" |
| **Venue** | ACL 2024 |
| **Protein Encoder** | ESM-2 150M (frozen) |
| **LLM Backbone** | Galactica 1.3B |
| **Projection** | Q-Former: 8 learnable query tokens, initialized from PubMedBERT, cross-attn every 2 layers (326M params in stage 1) |
| **Encoder Frozen?** | Stage 1: ESM-2 frozen, Q-Former trained. Stage 2: ESM-2 + Q-Former frozen, LoRA on Galactica |
| **LoRA** | r=8, applied to attention + FFN projections (7M params, 0.54% of LM) |
| **Training Data** | Swiss-Prot ~430K + ProteinKG25 ~422K + PDB-QA ~3.36M QA pairs |
| **Key Results** | Captioning BLEU-2: 55.03% (+10.55); Retrieval accuracy: 68.3% (+24.2); QA exact match: 65.0% |
| **GitHub** | https://github.com/acharkq/ProtT3 |
| **Notes** | Three pre-training objectives: Protein-Text Contrasting, Protein-Text Matching, Protein Captioning. Smallest encoder (150M) yet strong results |

---

#### 7. ProtLLM (Mar 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "An Interleaved Protein-Language LLM with Protein-as-Word Pre-Training" |
| **Venue** | ACL 2024 |
| **Protein Encoder** | ProtST (ESM-2-based) + 2-layer MLP projection head |
| **LLM Backbone** | LLaMA-7B |
| **Projection** | Trainable projection matrix (input connector) + output connector for protein prediction |
| **Encoder Frozen?** | Frozen during pre-training; unfrozen for task-specific fine-tuning (except PPI) |
| **LoRA** | Applied to all LLaMA linear modules during pre-training |
| **Training Data** | InterPT: 429K samples (165K PubMed articles + 90K UniProt/STRING pairs + 174K instruction data) |
| **Key Results** | SOTA on protein-centric benchmarks; novel in-context learning + zero-shot enzyme retrieval |
| **GitHub** | https://github.com/ProtLLM/ProtLLM |
| **Notes** | Dynamic protein mounting: handles arbitrary number of interleaved proteins in text. Protein-as-word vocabulary for candidate protein prediction |

---

#### 8. InstructProtein (Oct 2023 / ACL 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Aligning Human and Protein Language via Knowledge Instruction" |
| **Venue** | ACL 2024 |
| **Protein Encoder** | None (single unified LLM processes both protein sequences and text) |
| **LLM Backbone** | LLaMA-based (exact size unspecified in public materials) |
| **Projection** | None -- protein sequences tokenized directly into LLM vocabulary |
| **Encoder Frozen?** | N/A -- pre-trained on both protein + text corpora, then instruction-tuned |
| **Training Data** | UniRef100 (proteins) + PubMed (text) for pre-training; 2.8M knowledge-graph instruction pairs for SFT |
| **Key Results** | Outperforms SOTA LLMs on bidirectional protein-text generation |
| **GitHub** | https://github.com/HICAI-ZJU/InstructProtein |
| **Notes** | No separate encoder -- treats protein sequences as a language. KG-based instruction generation addresses annotation imbalance. Bidirectional: protein->text and text->protein |

---

#### 9. Mol-Instructions / LLaMA-Molinst (ICLR 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models" |
| **Venue** | ICLR 2024 |
| **Protein Encoder** | None (text-only approach) |
| **LLM Backbone** | LLaMA-7B-chat (original), updated to LLaMA-3 (May 2024) |
| **Projection** | None -- raw protein sequences as text tokens |
| **Training Data** | 148K molecule instructions + 505K protein instructions + 53K biotext instructions = ~706K total |
| **Key Results** | Establishes protein instruction-tuning baseline; dataset used by many subsequent works |
| **GitHub** | https://github.com/zjunlp/Mol-Instructions |
| **Notes** | **This is the dataset our project uses.** Text-only baseline. Protein-oriented tasks: function prediction, localization, fold classification, GO annotation, catalytic activity |

---

#### 10. SEPIT (Oct 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Structure-Enhanced Protein Instruction Tuning: Towards General-Purpose Protein Understanding" |
| **Venue** | KDD 2025 / arXiv 2410.03553 |
| **Protein Encoder** | ESM-2 650M + structure-aware module (Gaussian Basis Kernels for 3D info) |
| **LLM Backbone** | TinyLLaMA 1.1B, OpenLLaMA-v2 3B, LLaMA-2 7B |
| **Projection** | Linear projector (deliberately chosen over Q-Former to retain all residue information) |
| **Encoder Frozen?** | Stage 0: ESM-2 frozen, structure module trained. Stages 1-2: structure module + projector + LLM trained |
| **LoRA** | Not specified |
| **Training Data** | 10.5M instructions (5.47M Swiss-Prot/RCSB + 5.25M TrEMBL supplementary) -- largest instruction set |
| **Key Results** | BLEU-4: 52.37, closed-set accuracy: 79.97%; outperforms GPT-4o and Claude-3 |
| **GitHub** | Not yet released |
| **Notes** | Three-stage pipeline: warm-up (contrastive + denoising) -> caption pre-training -> MoE instruction tuning. Argues linear projector better than Q-Former for proteins |

---

#### 11. STELLA (Jun 2025)

| Component | Details |
|-----------|---------|
| **Paper** | "Towards Protein Function Prediction with Multimodal LLMs Integrating Sequence-Structure Representations" |
| **Venue** | arXiv 2506.03800 |
| **Protein Encoder** | ESM-3 small (esm3_sm_open_v1, 1.4B) -- **same encoder as our project** |
| **LLM Backbone** | LLaMA-3.1-8B-Instruct |
| **Projection** | Linear layer (simple adapter) |
| **Encoder Frozen?** | Stage 1: encoder + LLM frozen, only adapter trained. Stage 2: encoder frozen, adapter + LLM trained |
| **LoRA** | Optional during stage 2 |
| **Training Data** | OPI-Struc: ~301K samples (248K function + 24K multiple-choice + 29K enzyme) from AlphaFold DB + Swiss-Prot |
| **Key Results** | ROUGE-L: 0.5257, BERT Score: 0.8564; FP accuracy: 80.56%; Enzyme accuracy: 88.85% |
| **GitHub** | Not found |
| **Notes** | **Most directly comparable to our system**: same ESM-3 encoder, similar LLM backbone. Uses ESM-3's unified sequence+structure encoding. Simple linear adapter outperforms more complex designs |

---

#### 12. BioMedGPT-10B (Aug 2023 / updated 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Open Multimodal Generative Pre-trained Transformer for BioMedicine" |
| **Venue** | IEEE / arXiv 2308.09442 |
| **Protein Encoder** | ESM-2 3B (36-layer, 2560 hidden dim) |
| **LLM Backbone** | BioMedGPT-LM-7B (LLaMA-2-7B-Chat continually trained on biomedical literature) |
| **Projection** | Single fully-connected layer per modality |
| **Encoder Frozen?** | Encoder frozen; FC adaptor + LLM fine-tuned |
| **Training Data** | Multimodal fine-tuning on molecule + protein QA |
| **GitHub** | https://huggingface.co/PharMolix/BioMedGPT-LM-7B |
| **Notes** | Also handles molecules (via molecule encoder). Simple FC projection layer |

---

#### 13. Prot2Text (Jul 2023 / AAAI 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Multimodal Protein's Function Generation with GNNs and Transformers" |
| **Venue** | AAAI 2024 |
| **Protein Encoder** | ESM-2 35M (sequence) + 6-layer RGCN (structure graph) |
| **LLM Backbone** | GPT-2 (768 hidden dim) |
| **Projection** | Linear projection layer (ESM dim -> graph embedding dim) |
| **Training Data** | Swiss-Prot protein-function pairs |
| **Key Results** | Generates free-text function descriptions from protein structure+sequence |
| **GitHub** | Available (see paper) |
| **Notes** | Earlier work; smaller scale. GNN + PLM fusion into GPT-2 decoder |

---

#### 14. ProLLaMA (Feb 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "A Protein Large Language Model for Multi-Task Protein Language Processing" |
| **Venue** | IEEE TPAMI 2024 / arXiv 2402.16445 |
| **Protein Encoder** | None (protein sequences tokenized directly into LLM) |
| **LLM Backbone** | LLaMA-2 |
| **Projection** | None |
| **LoRA** | Used in both pre-training and instruction tuning stages |
| **Training Data** | ~13M samples with 11K+ superfamily annotations |
| **Key Results** | 67.1% exact match on superfamily prediction; strong protein generation |
| **GitHub** | https://github.com/PKU-YuanGroup/ProLLaMA |
| **Notes** | Text-only approach. Evolutionary Protein Generation Framework (EPGF). Continual learning on UniRef50 |

---

#### 15. ProteinCLIP (May 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Enhancing Protein Language Models with Natural Language" |
| **Venue** | bioRxiv 2024.05.14 |
| **Protein Encoder** | ESM-2 family + ProtT5 (provides adapter models for both) |
| **Projection** | Contrastive learning adapter (refines PLM embeddings to function-centric space) |
| **Training Data** | UniProt protein-text pairs |
| **Key Results** | SOTA on PPI prediction and homolog detection despite low sequence similarity |
| **GitHub** | https://github.com/wukevin/proteinclip |
| **Notes** | Not a generative system -- contrastive alignment of protein and text embeddings. Useful for representation learning |

---

#### 16. ProLLM (May 2024)

| Component | Details |
|-----------|---------|
| **Paper** | "Protein Chain-of-Thoughts Enhanced LLM for Protein-Protein Interaction Prediction" |
| **Venue** | COLM 2024 |
| **Protein Encoder** | None (protein data converted to natural language via ProCoT format) |
| **LLM Backbone** | LLM (unspecified base) |
| **Projection** | Embedding replacement of protein sites in natural language prompts |
| **Training Data** | Signaling pathway data in ProCoT format |
| **Key Results** | Improved PPI prediction accuracy via chain-of-thought reasoning |
| **GitHub** | https://github.com/MingyuJ666/ProLLM |
| **Notes** | Novel approach: treats signaling pathways as reasoning chains. Not a traditional multimodal encoder system |

---

### Key Patterns and Implications for Our Project

#### Projection Method Comparison

| Method | Used By | Params | Pros | Cons |
|--------|---------|--------|------|------|
| **Linear/FC** | ProteinGPT, BioMedGPT, STELLA, SEPIT, Prot2Text | Minimal | Simple, retains all residue info | May not align well without more capacity |
| **MLP (2+ layers)** | EvoLlama, **Our project** | ~20M | Good capacity, standard | More params than linear |
| **Q-Former** | ProtChatGPT, ProtT3 | 300M+ | BLIP-2-inspired, proven in vision | Heavy, may lose residue info |
| **Cross-attention adapter** | Prot2Chat, Evolla | 100M-1.7B | Question-aware compression | Complex architecture |
| **Attention Pooling + MLP** | **Our project (MLP path)** | ~30M | Reduces token count + projects | Our specific innovation |
| **Perceiver Resampler** | **Our project (Perceiver path)** | ~29M | Learned compression + projection | Less tested in protein domain |

**Key insight**: SEPIT explicitly argues linear projectors are better than Q-Former for proteins because "any change in the amino acid sequence can lead to significant structural differences" -- retaining all residue tokens matters. However, this increases compute. Our AttentionPooling (32 tokens) + MLP provides a middle ground.

#### Encoder Choices

| Encoder | Size | Used By | Notes |
|---------|------|---------|-------|
| ESM-2 150M | 150M | ProtT3 | Smallest, still effective |
| ESM-2 650M | 650M | EvoLlama, SEPIT, ProteinCLIP | Most popular choice |
| ESM-2 3B | 3B | ProteinGPT, BioMedGPT | Highest capacity |
| ESM-1b | 650M | ProtChatGPT | Older model |
| ESM-3 small | 1.4B | **STELLA, Our project** | Unified seq+struct encoding |
| SaProt-650M | 650M | Evolla | Structure-aware (SA tokens) |
| ProteinMPNN | varies | EvoLlama, Prot2Chat | Structure encoder (not sequence) |

**Key insight**: We are one of only two systems (alongside STELLA) using ESM-3 as the protein encoder. ESM-3's unified sequence+structure encoding is a differentiator. Most systems still use ESM-2.

#### LLM Backbone Choices

| LLM | Used By |
|-----|---------|
| LLaMA-3-8B | EvoLlama, Prot2Chat, Evolla, STELLA |
| LLaMA-2-7B | ProtLLM, BioMedGPT, SEPIT |
| Vicuna-13B | ProtChatGPT |
| Galactica 1.3B | ProtT3 |
| Multiple (LLaMA/Mistral) | ProteinGPT |
| **Qwen3-4B/8B** | **Our project** |

**Key insight**: We are the only system using Qwen3 as the backbone. Most systems use LLaMA variants. Using a different backbone family provides novelty.

#### Training Data Scale

| System | Data Size | Source |
|--------|-----------|--------|
| Evolla | 546M QA triples, 150B tokens | Swiss-Prot + ProTrek |
| SEPIT | 10.5M instructions | Swiss-Prot + TrEMBL |
| ProLLaMA | 13M samples | UniRef50 |
| Mol-Instructions | 706K total (505K protein) | Various databases |
| ProteinGPT | 132K proteins x 40 QA | RCSB-PDB |
| Prot2Chat | 430K | Mol-Instructions + UniProtQA |
| ProtT3 | ~4.2M pairs | Swiss-Prot + ProteinKG25 + PDB-QA |
| **Our project** | 50K-505K | Mol-Instructions protein subset |

#### Frozen Encoder Strategy (All Systems)

Every surveyed system keeps the protein encoder frozen during at least the initial alignment stage. This validates our approach of keeping ESM-3 frozen throughout training. The pattern is:
1. Stage 1: Freeze everything except projector/adapter (alignment)
2. Stage 2: Optionally unfreeze LLM (with LoRA) for instruction tuning
3. Some systems unfreeze the encoder in later stages (EvoLlama, ProtLLM) but this is less common

#### Closest Comparable: STELLA

STELLA is the most directly comparable system to ours:
- Same ESM-3 encoder (esm3_sm_open_v1, 1.4B)
- Similar LLM class (LLaMA-3.1-8B-Instruct vs our Qwen3-4B/8B)
- Simple linear adapter (vs our AttentionPooling+MLP or Perceiver)
- Two-stage training with frozen encoder
- However: STELLA uses only a linear layer, while we offer MLP and Perceiver Resampler options
- STELLA uses ~301K training samples vs our 50K-505K from Mol-Instructions

---

## Development Log

### 2026-02-20: Unified Experiment Pipeline

**Milestone**: Implemented unified experiment directory structure for the full base→SFT→GRPO pipeline.

**Changes**:
- All artifacts now stored under `results/{experiment_name}/` (config, lineage, checkpoints, logs, eval)
- `lineage.json` tracks stage, parent_experiment, parent_checkpoint, encoder, approach, timestamps
- GRPO chains from SFT via `parent_experiment=<name>` (auto-resolves checkpoint path)
- `experiment_name` auto-generated or user-set; Hydra output now goes to `results/{name}/logs/`
- `src/utils/experiment.py`: write_lineage, read_lineage, complete_lineage, resolve_parent_checkpoint, list_experiments
- SFT/GRPO trainers save `training_args.json` and `metrics.json` at experiment root (not inside checkpoints)
- Evaluate can auto-detect checkpoint from `experiment_name`
- All 414 existing tests still pass

**Files modified**: configs/config.yaml, scripts/train.py, scripts/evaluate.py, src/training/sft_trainer.py, src/training/grpo_trainer.py
**Files created**: src/utils/experiment.py

---

### 2026-02-20: Perceiver Resampler GPU Testing & Efficiency Analysis

**Milestone**: Full GPU integration testing and efficiency comparison of Perceiver Resampler vs MLP.

**Key results** (100 samples, single H100):
| Metric | MLP | Perceiver 2L | Ratio |
|--------|-----|-------------|-------|
| Trainable params | ~20M | ~132M | 6.6x |
| GPU peak | 18.35 GB | 19.4 GB | 1.06x |
| Training speed | 1.28 steps/s | 1.14 steps/s | 0.89x |
| Final eval_loss | 4.384 | 4.371 | ~same |
| Checkpoint size | ~34 MB | 520 MB | 15x |

**Decision**: Default Perceiver layers changed from 6→2 (6L = 382M params, too expensive for marginal gain).

---

### 2026-02-20: Research -- Projection-Only Multimodal LLM Architectures (2024-2026)

**Milestone**: Comprehensive survey of state-of-the-art "projection-only" multimodal LLM architectures that map non-text features into LLM embedding space WITHOUT modifying the LLM architecture (no cross-attention layers added). Full findings below in [Projection-Only Multimodal LLM Survey](#projection-only-multimodal-llm-survey-2026-02-20).

**Key Takeaways for Our Project**:

1. **Pixel shuffle / space-to-depth is the dominant token reduction technique**. InternVL, DeepSeek-VL2, and Qwen2-VL all use 2x2 spatial merging to achieve 4x token reduction. Our attention pooling (L residues -> 32 tokens) serves an analogous role but is more aggressive. The field consensus is: reduce tokens before the LLM, not inside it.

2. **2-layer MLP with GELU/SiLU is the universal projector**. Every top-performing system (LLaVA, InternVL, DeepSeek-VL2, Qwen2-VL) uses a 2-layer MLP. The projector is NOT the bottleneck -- data quality and training recipe matter far more than connector complexity (confirmed by MM1/MM1.5 ablations).

3. **Multi-stage training is essential**. All architectures use at least 2 stages: (1) alignment (projector-only, encoder+LLM frozen) on large paired data, then (2) instruction tuning (projector+LLM trainable, encoder frozen or with very low LR). Our current approach matches this pattern exactly.

4. **Dynamic resolution and tiling are critical for vision but less relevant for proteins**. Vision models spend enormous effort on multi-scale processing. For proteins, sequence length variation is the analogous challenge, and our attention pooling already handles it by compressing variable-length sequences to fixed 32 tokens.

5. **Multi-encoder fusion (Cambrian-1) is interesting for proteins**. Cambrian-1 combines 4 vision encoders via cross-attention. An analogous protein approach could combine ESM-3 (sequence+structure) with a GNN encoder (graph topology) or ProtTrans (different pretraining objective). The Spatial Vision Aggregator pattern could be adapted.

6. **Molmo's approach is closest to ours architecturally**: attention pooling on 2x2 windows + MLP projection + SwiGLU activation. Their key insight is that careful data curation matters more than architectural novelty.

---

### 2026-02-20: Research -- Protein Generation / Embedding-to-Sequence Decoding

**Milestone**: Literature review on the "reverse direction" problem: generating protein sequences and structures from LLM hidden states or text descriptions.

**Research Summary**: Surveyed 20+ models/frameworks spanning autoregressive protein generation, text-conditioned design, diffusion-based generation, bidirectional understanding+generation, and embedding-to-sequence decoding. Full findings below in [Protein Generation Research](#protein-generation-research-2026-02-20).

**Key Takeaways for Our Project**:

1. **Adding a protein decoder head to Qwen3-4B is feasible**. The simplest approach: a linear head mapping LLM hidden states (2560-dim) to a 20-amino-acid vocabulary, trained with next-token prediction loss on protein sequences. This is exactly what ProGen, ProtGPT2, and ProLLaMA do. Our LLM already knows natural language; we would fine-tune it to also output protein tokens.

2. **The ProteinDT / unCLIP paradigm is the closest analogy to our architecture**. ProteinDT uses: (a) contrastive pretraining to align text and protein embeddings (like our SFT alignment stage), (b) a "facilitator" that maps text embeddings to protein embeddings, and (c) a decoder (autoregressive T5 or diffusion) that generates sequences from those embeddings. We already have steps (a) and partially (b) via our projector. We would need to add step (c).

3. **Two practical paths for our system**:
   - **Path A (Simple)**: Train Qwen3-4B to output amino acid tokens directly in its text vocabulary. Use `<protein>MKTL...</protein>` tags. The LLM generates protein sequences as text. This is what ProLLaMA and InstructProtein do. Requires protein sequence SFT data.
   - **Path B (Advanced)**: Add a separate protein decoder that takes LLM hidden states and generates amino acid sequences. This decouples protein generation from text generation. ProteinDT and Pinal use this approach.

4. **For structure generation**, the best path is to use ESM-3 or ESMFold as a downstream tool. Generate a sequence first (via Path A or B), then fold it with ESMFold/ESM-3/AlphaFold. Pinal takes the inverse approach: generate structure tokens first, then design sequences to fold into that structure.

5. **RL/DPO for protein generation is an active area**. CtrlProt uses multi-listwise preference optimization for controllable protein generation. g-DPO adapts DPO for experimentally labeled protein data. This aligns directly with our planned GRPO stage.

---

### 2026-02-19: SFT Training & Evaluation Pipeline

**Milestone**: Multimodal SFT training at scale, evaluation pipeline preparation

**Training Experiments** (ESM-3 + Qwen3-4B, LoRA k/v, differential LR):

| Run | Samples | Epochs | Config | Loss Start → End | Notes |
|-----|---------|--------|--------|-------------------|-------|
| 500-sample test | 500 | 3 | lr=2e-4, projector_lr=2e-3 | 17.35 → 4.08 | First multimodal success |
| 10K baseline | 10,000 | 3 | lr=2e-4, warmup=100 | 35.80 → 27.84 | ~96 min |
| 500 freeze_lora | 500 | 5 | freeze LoRA, train projector only | 37.5 → 31.7 | Flat loss, abandoned |
| **50K full** | **50,000** | **5** | **lr=2e-4, projector_lr=2e-3, warmup=50** | **34.25 → 31.26 (step 100)** | **Running ~8.5hr** |

**Key Fixes This Session**:
1. **Differential learning rate**: Added `projector_lr: 2e-3` (10x base) for randomly-initialized pooling + projector (LLaVA-style)
2. **use_qlora flag**: Fixed ProteinLLM to save `use_qlora: false` for non-quantized LoRA training
3. **from_pretrained**: Fixed to pass `encoder_embed_dim` from saved config
4. **generate method**: Fixed output slicing to exclude prompt tokens when using `inputs_embeds`
5. **Evaluation config**: Added `evaluation`, `checkpoint_path`, `output_dir` to config.yaml
6. **Zero-init / gating**: Both cause NaN explosion with bf16 + gradient checkpointing. Reverted.
7. **freeze_lora**: LoRA freezing doesn't help - joint training needed for LLM to attend to prefix tokens

**Training Parameters (50K run)**:
- Model: ESM-3 (frozen, 1536-dim) → AttentionPooling (32 tokens) → MLP (1536→2048→2560) → Qwen3-4B (LoRA k/v)
- Trainable: pooling 9.5M + projector 8.4M + LoRA 2M = ~20M (0.49% of 4B)
- LR: base=2e-4 (LoRA), projector=2e-3, cosine schedule, warmup=50 steps
- Batch: 4 × 8 grad_accum = 32 effective, 7815 total steps

**Evaluation Pipeline** (prepared, ready to test):
- GO prediction (demo dataset, 10 proteins)
- PPI prediction
- Stability prediction
- All benchmarks aggregated
- Config: `python scripts/evaluate.py checkpoint_path=<path>/protein_llm evaluation.name=go_prediction`

**Checkpoint**: `data/checkpoints/2026-02-19_esm3_qwen3_4b_sft_lora/` (saving every 500 steps)
**wandb**: https://wandb.ai/sjinyeop/protein-llm-sft/runs/qnrez9g6

---

### 2026-02-18: ESM-3 + Qwen3-4B Integration Sprint

**Milestone**: Multi-agent sprint to integrate ESM-3 encoder and Qwen3-4B LLM into the pipeline

**Sprint Focus**: ESM-3 + Qwen3-4B integration, approach-based architecture, RL investigation

**Changes Made** (14 files, 782+ lines modified):
- **ESM-3 encoder implementation** (`src/models/protein_encoder.py`): Added ESM-3 small (esm3-sm-open-v1, 1.4B params, 1536-dim embeddings). Status: partial, being verified on GPU.
- **Approach-based architecture** (`configs/config.yaml`): Added `approach: text|esm3` config switching. Each approach selects its own encoder, data processing, and model architecture.
- **ESM-3 encoder config** (`configs/encoder/esm3_small.yaml`): New config with 1536-dim embeddings, attention pooling (32 tokens), MLP projector (1536->2048->2560).
- **Qwen3-4B model config** (`configs/model/qwen3_4b.yaml`): Added as default LLM (hidden_size=2560).
- **Multimodal LLM wrapper** (`src/models/multimodal_llm.py`): Updated to support approach-based model selection and ESM-3 encoder path.
- **SFT trainer** (`src/training/sft_trainer.py`): Updated for approach-aware training, wandb project separation.
- **GRPO trainer** (`src/training/grpo_trainer.py`): Expanded with reward functions (GO, PPI, stability). Gradient flow bug found and being fixed.
- **wandb separation**: DONE. SFT logs to `protein-llm-sft`, RL logs to `protein-llm-rl`.
- **Training configs** (`configs/training/*.yaml`): Updated all training configs (sft_qlora, sft_lora, grpo, dpo) with wandb project tags.
- **Train script** (`scripts/train.py`): Updated for approach-based dispatch.
- **Evaluation** (`src/evaluation/go_prediction.py`): Confirmed encoder-agnostic. Bugs found in evaluate.py argument parsing.
- **ESM-3 test** (`tests/models/test_esm3_encoder.py`): New unit test for ESM-3 encoder loading and forward pass.
- **Documentation** (`PROJECT_GOALS.md`, `REVIEW_POINTS.md`, `SWE_AGENT_TEAM.md`): New project management docs for agent team coordination.

**RL Investigation**:
- GRPO reward functions work (GO F1, PPI binary, stability Gaussian)
- Gradient flow bug found: gradients not propagating through generation step
- Fix in progress -- need to verify with actual training run

**Evaluation Suite**:
- Encoder-agnostic confirmed: evaluation works regardless of approach (text/esm3)
- Bugs found in `scripts/evaluate.py` argument parsing (being fixed)

**Issues Encountered**:
- GRPO gradient flow issue: reward computation works but gradients do not flow back through the model correctly
- ESM-3 encoder needs GPU verification (cannot test on login node)

**Next Steps**:
- [ ] Complete ESM-3 encoder GPU verification
- [ ] Fix GRPO gradient flow bug
- [ ] Run first ESM-3 + Qwen3-4B SFT experiment
- [ ] Fix evaluate.py argument parsing bugs
- [ ] Validate end-to-end: encode -> pool -> project -> LLM forward pass

---

### 2026-02-16: Model Loading Test Script

**Milestone**: Created model loading test script for Qwen 3 1.5B verification

**Created**:
- `scripts/test_model_loading.py` - Tests model loading, tokenizer, and inference

**Script Features**:
- Environment check (PyTorch, CUDA, Transformers versions)
- GPU detection with helpful error messages
- Model loading with configurable dtype
- Tokenizer verification
- Inference sanity check
- Memory usage reporting

**Usage** (requires GPU compute node):
```bash
# On compute node with GPU
srun --gres=gpu:1 --mem=32G --time=00:30:00 \
    bash -c "source /home/yeopjin/orcd/pool/init_protein_llm.sh && \
    python scripts/test_model_loading.py --model Qwen/Qwen3-1.5B"

# List available models
python scripts/test_model_loading.py --list-models

# Check environment only (no GPU needed)
python scripts/test_model_loading.py --check-only
```

**Models to Test**:
| Model | Size | Use Case |
|-------|------|----------|
| Qwen/Qwen3-1.5B | 1.5B | Quick testing |
| Qwen/Qwen2.5-7B-Instruct | 7B | Production training |
| meta-llama/Llama-3.1-8B-Instruct | 8B | Alternative |

**Note**: Script requires GPU access. Login nodes don't have GPUs - use `srun` or submit a job.

**Next Steps**:
- [ ] Run test on compute node with GPU
- [ ] Verify Qwen 3 1.5B loads correctly
- [x] ESM-3 encoder integration verified

---

### 2026-02-16: Dataset Download & Exploration

**Milestone**: Downloaded all training datasets and created exploration notebook

**Datasets Downloaded**:

| Dataset | Location | Size | Status |
|---------|----------|------|--------|
| IPD PDB Sample | `data/raw/pdb_2021aug02_sample/` | 870 `.pt` files | ✅ Ready |
| Swiss-Prot | `data/raw/swissprot/uniprot_sprot.fasta.gz` | ~90MB (~570K sequences) | ✅ Ready |
| Mol-Instructions | `data/raw/mol_instructions/data/Protein-oriented_Instructions/` | 5 JSON files (~678MB total) | ✅ Ready |

**Mol-Instructions Files**:
- `protein_design.json` (281 MB)
- `protein_function.json` (155 MB)
- `general_function.json` (127 MB)
- `catalytic_activity.json` (65 MB)
- `domain_motif.json` (50 MB)

**IPD PDB Sample Structure** (per `.pt` file):
- `seq`: Amino acid sequence (string)
- `xyz`: Atomic coordinates `[L, 14, 3]` tensor
- `mask`: Boolean mask `[L, 14]`
- `bfac`: Temperature factors `[L, 14]`
- `occ`: Occupancy `[L, 14]`

**Created**:
- `scripts/explore_datasets.ipynb` - Jupyter notebook for dataset exploration
- Download utilities in `src/data/download.py`

**Usage**:
```bash
# Download datasets
python src/data/download.py --dataset ipd_pdb_sample --output_dir ./data/raw
python src/data/download.py --dataset swissprot --output_dir ./data/raw/swissprot
python src/data/download.py --dataset mol_instructions --output_dir ./data/raw

# Explore datasets
jupyter notebook scripts/explore_datasets.ipynb
```

**Next Steps**:
- [ ] Preprocess datasets into training format
- [ ] Create train/val/test splits
- [ ] Build instruction dataset loader

---

### 2026-02-16: Project Restructuring

**Milestone**: Complete project restructuring for Claude Code integration and Hydra configs

**Changes Made**:
- Created `.claude/` directory with skills, commands, and agents
- Implemented Hydra configuration system in `configs/`
- Created entry point scripts in `scripts/`
- Moved documentation to `docs/`
- Slimmed CLAUDE.md from 80 to ~50 lines
- Set up pytest test structure in `tests/`
- Created `pyproject.toml` for package management

**New Structure**:
```
Post_Training_Protein_LLM/
├── .claude/           # Claude Code integration
│   ├── agents/        # engineer, qa, researcher
│   ├── commands/      # /train, /eval, /data-prep, /debug
│   └── skills/        # protein-encoding, rl-training, hydra-configs
├── configs/           # Hydra hierarchical configs
│   ├── model/         # qwen3_4b (default), llama3_8b
│   ├── encoder/       # esm3_small (default)
│   ├── training/      # sft_qlora, grpo, dpo
│   └── experiment/    # baseline_sft, full_pipeline
├── scripts/           # Entry points
├── src/               # Core implementation
├── tests/             # pytest suite
└── docs/              # Documentation
```

**Next Steps**:
- [ ] Implement SFT trainer in `src/training/sft_trainer.py`
- [ ] Implement GRPO trainer in `src/training/grpo_trainer.py`
- [ ] Add attention pooling in `src/models/pooling.py`
- [ ] Create MLP projector in `src/models/projector.py`

---

### 2026-03-02: Arrow Preprocessing for Combined Dataset (Fast Load)

**Milestone**: Converted combined SFT dataset (4.89M samples, 21 JSON files) from JSON to Arrow format. Dataset loading dropped from ~10 minutes to <0.1s via memory-mapped Arrow files.

**Changes Made**:
- **NEW** `scripts/prepare_arrow.py`: One-time preprocessing script that reads JSON files, injects metadata, applies temperature sampling (α=0.7), computes `__length__` column, filters by protein length, shuffles, splits, and saves as Arrow via `save_to_disk()`
- **Modified** `src/data/mol_instructions.py`:
  - Added `_try_load_arrow()` method: detects `{cache_dir}_arrow/{split}/` and loads via `load_from_disk()`
  - Updated `_load_dataset()` to try Arrow first, then JSON, then HuggingFace (backward compatible)
  - Updated `lengths` property to use pre-computed `__length__` column when available (instant vs iterating 4.89M rows)

**Results**:
| Metric | Before (JSON) | After (Arrow) |
|--------|--------------|---------------|
| Train load time | ~10 min | 0.08s |
| Validation load time | ~30s | 0.01s |
| `lengths` computation | ~60s | 0.4ms |
| Arrow output | — | `data/processed/combined_sft_260225_arrow/{train,validation,test}/` |
| Train split | 4,891,437 | 4,891,437 (identical) |
| Val split | 271,746 | 271,746 (identical) |

**Files modified**: `src/data/mol_instructions.py`
**Files created**: `scripts/prepare_arrow.py`

---

### Template: New Entry
```markdown
### YYYY-MM-DD: Title

**Milestone**: What was achieved

**Changes Made**:
- Item 1
- Item 2

**Results** (if applicable):
| Metric | Value |
|--------|-------|
| ... | ... |

**Issues Encountered**:
- Issue and resolution

**Next Steps**:
- [ ] Task 1
- [ ] Task 2
```

---

## Results & Metrics

### Experiments Summary

| Date | Experiment | Model | Dataset | Key Metric | Value | Notes |
|------|------------|-------|---------|------------|-------|-------|
| - | - | - | - | - | - | No experiments run yet |

### Best Configurations

*To be updated after experiments*

---

## Decision Log

### 2026-02-18: Sprint Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default protein encoder | ESM-3 small (esm3-sm-open-v1) | Project pivot to ESM-3 per PROJECT_GOALS.md |
| Default LLM | Qwen3-4B-Instruct-2507 | Smaller model for fast iteration |
| Architecture pattern | Approach-based (`text\|esm3`) | Modular, config-driven, easy to extend |
| wandb separation | SFT -> `protein-llm-sft`, RL -> `protein-llm-rl` | Cleaner experiment tracking |
| ESM-3 projector dims | 1536 -> 2048 -> 2560 | Match ESM-3 embedding dim to Qwen3-4B hidden_size |

### 2026-02-16: Project Structure Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Config system | Hydra | CLI overrides, sweeps, hierarchical composition |
| Claude integration | .claude/ directory | Skills, commands, agents for automation |
| Package manager | pyproject.toml | Modern Python standard |
| Test framework | pytest | Industry standard, good fixtures |

### 2026-02-14: Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Protein encoder | ESM-3 small | Multimodal (sequence + structure), best for our use case |
| Keep encoder frozen | Yes | Preserve protein knowledge (applies to ALL encoders) |
| LoRA targets | k/v matrices only | Protein-specific finding |
| LoRA rank | r=8 (minimum r=4) | Sufficient for protein tasks |
| Pooling method | Attention pooling | 4% better than mean pooling |
| RL method | GRPO | 50% less memory than PPO |

---

## Agent Rules & Protocols

### 1. Research Agent Protocols
- Each agent focuses on a specific topic area
- Findings must include: paper title, authors, year, key contributions, relevance to our project
- Agents should identify: datasets, methods, evaluation metrics, open questions
- All interactions and findings are logged in this document

### 2. Topic Assignments
| Agent ID | Topic Focus | Status |
|----------|-------------|--------|
| Agent-1 | Protein Language Models & Embeddings | Completed |
| Agent-2 | LLM Post-Training Methods (SFT/RL) | Completed |
| Agent-3 | Multimodal LLM Integration | Completed |
| Agent-4 | Protein-Related Datasets & Benchmarks | Completed |

### 3. Documentation Standards
- Each finding should be timestamped
- Cross-references between agents encouraged
- Questions for human input marked with `[QUESTION]`
- Action items marked with `[ACTION]`

### 4. Decision Log
All major decisions and their rationale are recorded here.

---

## Agent Interactions Log

### Session: Mid-Training Taxonomy and Protein-LLM Training Stages
**Date**: 2026-02-21
**Objective**: Research the taxonomy of LLM training stages (pre-training, mid-training, post-training), how protein-LLMs implement each stage, and best practices from both the multimodal LLM and protein-LLM literature.

---

#### 1. Taxonomy of LLM Training Stages

The modern LLM training pipeline has evolved from a simple pretrain-then-finetune paradigm into a multi-stage process. The terminology is not yet fully standardized, but the following taxonomy captures the current consensus from multiple surveys (arXiv 2510.06826, arXiv 2510.23081, Raschka 2024).

| Stage | Also Called | Purpose | Data | Trainable |
|-------|-----------|---------|------|-----------|
| **Pre-training** | Foundation training | Learn general language from scratch | Web-scale (1-15T tokens), noisy, diverse | All weights |
| **Mid-training** | Continued pre-training (CPT), domain adaptation, annealing | Specialize the model for a domain or capability (code, math, bio, long-context) | Curated, high-quality, domain-specific (10B-1T tokens) | All weights (lower LR) |
| **SFT** | Supervised fine-tuning, instruction tuning | Learn to follow instructions and produce structured outputs | Instruction-response pairs (100K-1M) | All weights or LoRA |
| **RLHF/RL** | Alignment, preference optimization, post-training | Align outputs with human preferences or verifiable rewards | Preference pairs, reward signals | Policy weights (often LoRA) |

**Key distinctions**:
- **Pre-training vs Mid-training**: Pre-training starts from random init on diverse web data. Mid-training continues from a pre-trained checkpoint on higher-quality, domain-concentrated data with a lower learning rate. Mid-training uses the *same objective* (next-token prediction) but on better data.
- **Mid-training vs SFT**: Mid-training uses a language modeling objective on raw text/documents. SFT uses a language modeling objective only on the *response* portion of instruction-response pairs (the prompt tokens are masked from the loss).
- **SFT vs RL**: SFT teaches the model *what* to say via supervised examples. RL teaches the model *how well* it said it via reward signals, improving quality beyond what supervised data alone provides.

**The mid-training survey** (arXiv 2510.06826) identifies three primary dimensions of mid-training:
1. **Data distribution**: Shifting from noisy web data to curated, high-quality sources (code, math, scientific text, synthetic textbooks). Quality outweighs quantity.
2. **Learning rate scheduling**: Typically cosine or WSD (warmup-stable-decay). The key insight is that mid-training uses a *lower peak LR* than pre-training, often with re-warming.
3. **Long-context extension**: RoPE frequency remapping (NTK-aware, YaRN, LongRoPE) to extend context from 4K to 32K-128K tokens.

**Examples from major labs**:
- **Apple**: 3-stage pre-training: (1) broad pre-training, (2) 1T tokens mid-training with math/code emphasis, (3) 100B tokens for context extension to 32K.
- **Qwen-2**: Pre-training then 2-phase post-training: 500K SFT examples then DPO alignment.
- **OLMo 2**: Uses curriculum learning during mid-training with controlled data distribution shifts.
- **DeepSeek-R1**: Pre-training then mid-training then SFT then GRPO (the full 4-stage pipeline).

Sources: [Mid-Training Survey](https://arxiv.org/abs/2510.06826), [LLM Mid-Training Survey](https://arxiv.org/abs/2510.23081), [Raschka: New LLM Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training)

---

#### 2. Multimodal LLM Training Stages (Vision-Language Analogy)

Multimodal LLMs face the same challenge as protein-LLMs: bridging a frozen encoder's embedding space to an LLM's token space via a projector. The vision-language literature has converged on a 2-3 stage recipe that is directly applicable.

**LLaVA / LLaVA-1.5 (the canonical recipe)**:

| Stage | What | Frozen | Trainable | Data | Duration |
|-------|------|--------|-----------|------|----------|
| Stage 1: Feature Alignment | Teach projector to translate visual features into LLM-compatible tokens | Vision encoder + LLM | MLP projector only | 558K image-caption pairs (LAION-CC-SBU) | ~6h on 8xA100 |
| Stage 2: Visual Instruction Tuning | Teach the full system to follow multimodal instructions | Vision encoder | MLP projector + LLM (full finetune or LoRA) | 665K instruction-following + VQA data | ~20h on 8xA100 |

LLaVA-1.5 key insight: replacing the linear projector with a 2-layer MLP (with GELU) significantly improved alignment quality. The projector architecture matters.

**Qwen-VL (3-stage recipe)**:

| Stage | What | Frozen | Trainable |
|-------|------|--------|-----------|
| Stage 1: Pre-training | Broad multimodal alignment | LLM | Vision encoder + adapter |
| Stage 2: Multi-task Fine-tuning | Curated fine-grained tasks (captioning, VQA, OCR) | Nothing frozen | Vision encoder + adapter + LLM |
| Stage 3: Instruction Tuning | Conversational alignment | Vision encoder | Adapter + LLM |

**Why the 2-stage recipe works**: Stage 1 (projector warm-up) prevents catastrophic interference. If you train the projector and LLM jointly from scratch, the randomly-initialized projector produces garbage embeddings that corrupt the LLM's pre-trained representations. By first training only the projector while the LLM is frozen, you establish a meaningful mapping without destabilizing the LLM. This is exactly the "random projector init causes loss ~35" problem observed in our project.

Sources: [LLaVA](https://llava-vl.github.io/), [LLaVA-1.5 Paper (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf), [Qwen-VL](https://arxiv.org/abs/2409.12191), [LLaVA Architecture Guide](https://learnopencv.com/llava-training-a-visual-assistant/)

---

#### 3. How Protein-LLMs Implement Each Training Stage

The protein-LLM literature has adopted the multimodal LLM recipe with domain-specific adaptations. Here is a comprehensive comparison of training pipelines across major protein-LLM papers:

**A. Two-Stage: Alignment then Instruction Tuning**

| Model | Year | Encoder | Projector | Stage 1 (Alignment) | Stage 2 (Instruction) | LLM |
|-------|------|---------|-----------|---------------------|----------------------|-----|
| **ProteinGPT** | 2024 | ESM-2 3B + ESM-IF1 (both frozen) | Linear projection | Train projector only (encoders+LLM frozen), 132K proteins | Train projector + LLM, 3.7M QA pairs | LLaMA-3 |
| **EvoLlama** | 2024 | ESM-2 + ProteinMPNN (both frozen in S1, trainable in S2) | Two MLPs (seq+struct), element-wise addition | Train MLPs only (all else frozen), 369K Swiss-Prot pairs | Train MLPs + encoders (LLM frozen!), 10 PEER/Mol-Instructions tasks | LLaMA-3 |
| **ProtChatGPT** | 2024 | ESM-1b + ESM-IF1 (frozen) | PLP-former + FC layers | Train PLP-former (contrastive+generative losses), 553K pairs | Train FC adapter only (all else frozen), 143K PDB pairs | Vicuna-13B |

**B. Single-Stage: Joint Training**

| Model | Year | Encoder | Projector | Training | LLM |
|-------|------|---------|-----------|----------|-----|
| **Prot2Chat** | 2025 | ProteinMPNN (frozen) | Cross-attention adapter (89.7M params) | Joint: adapter (full) + LLM (LoRA), Mol-Instructions 404K | LLaMA-3 |
| **ProtLLM** | 2024 | Protein-as-Word tokenizer | Interleaved embedding | Multi-task pre-training on InterPT dataset | InternLM2-7B |

**C. Notable Architectural Choices**:

- **ProteinGPT**: Simplest projector (linear), largest instruction dataset (3.7M QA). Follows LLaVA most closely. In Stage 2, both projector AND LLM are unfrozen.
- **EvoLlama**: Unique in that Stage 2 unfreezes the protein encoders but keeps the LLM frozen. Uses element-wise addition of sequence and structure features to reduce token count by ~50%.
- **ProtChatGPT**: Most complex Stage 1, using three losses simultaneously (contrastive, generative, matching). Stage 2 trains only the adapter.
- **Prot2Chat**: Skips the alignment stage entirely; trains adapter + LoRA jointly from scratch. Claims "early fusion" (combining seq+struct at the encoder level) makes separate alignment unnecessary.
- **ProtLLM**: Completely different approach: tokenizes proteins as discrete words and interleaves them with text tokens, avoiding the projector problem entirely.

**Key finding**: There is no consensus on whether to freeze the LLM in Stage 2. ProteinGPT unfreezes it, EvoLlama keeps it frozen, ProtChatGPT keeps it frozen. Prot2Chat uses LoRA on the LLM. Our project's approach (LoRA on k/v only) is a reasonable middle ground.

Sources: [ProteinGPT](https://arxiv.org/abs/2408.11363), [EvoLlama](https://arxiv.org/abs/2412.11618), [ProtChatGPT](https://arxiv.org/abs/2402.09649), [Prot2Chat](https://arxiv.org/abs/2502.06846), [ProtLLM](https://arxiv.org/abs/2403.07920)

---

#### 4. The Projector Warm-Up / Feature Alignment Stage in Detail

This is the most critical stage for our project. The key question: should we train the projector alone first, or jointly with LoRA?

**Evidence for projector-only warm-up (Stage 1)**:
- LLaVA, ProteinGPT, EvoLlama, ProtChatGPT all use it
- Prevents randomly-initialized projector from corrupting LLM representations
- LLaVA shows it takes only ~6 hours (much cheaper than Stage 2)
- Our own observation: random projector init causes loss ~35 (vs expected ~11.9)

**Evidence against (or for skipping)**:
- Prot2Chat achieves competitive results with single-stage joint training (adapter + LoRA)
- The LLaVA team noted in LLaVA-1.5 that with a stronger projector (2-layer MLP vs linear), Stage 1 becomes less critical
- LLaVA-OneVision-1.5 experiments suggest the projector warm-up may be less important when using LoRA on the LLM instead of full fine-tuning

**Best practice recommendation for our project**:
1. **Stage 1 (Projector Warm-Up)**: Train only pooling + projector (freeze LLM, freeze encoder). Use simple protein-description pairs (Swiss-Prot). ~50K-370K samples, ~1-3 epochs. LR: 1e-3 to 2e-3 (higher than SFT). This teaches the projector to produce embeddings the LLM can interpret.
2. **Stage 2 (SFT with LoRA)**: Unfreeze LoRA adapters on LLM. Continue training projector + LoRA jointly on instruction-following data (Mol-Instructions). LR: 1e-4 to 2e-4 for LLM, keep projector LR higher (2e-3). This teaches the system to follow instructions.
3. **Stage 3 (RL / GRPO)**: Optional. Train LoRA (possibly freeze projector) with reward signals. This optimizes for specific properties (structure quality, functional accuracy).

---

#### 5. Reinforcement Learning for Protein LLMs

RL for protein models is an active area with several recent papers (2024-2025):

**ProtRL** (AI4PDLab, 2024-2025):
- Framework implementing wDPO and GRPO for protein language models
- Applied to autoregressive pLMs like ZymCTRL
- Key result: designed low-nanomolar EGFR inhibitors using GRPO
- GRPO preferred over DPO due to greater flexibility with non-preference data distributions
- Works with synthetic data and few RL iterations

**ProteinZero** (2025):
- Self-improving protein generation via online RL
- Uses ESMFold as a proxy reward model (similar to our GRPO reward setup)
- Balances multi-reward maximization + KL divergence + diversity regularization
- Reduces design failure rates by 36-48% vs baselines (ProteinMPNN, ESM-IF)
- Runs on a single 8xGPU node in 3 days

**"From Supervision to Exploration"** (arXiv 2510.01571, 2025):
- Key insight: RL does NOT teach the model new capabilities beyond pre-training
- Instead, RL improves sampling efficiency toward high-reward regions already implicit in the model
- Analogy: "hill-climbing where task difficulty sets the height, reward accuracy sets direction, policy capacity sets starting altitude"
- GRPO showed superior exploration in antimicrobial peptide tasks through group-based loss
- Trade-off: RL reduces diversity and novelty (concentrates on high-fitness regions)

**Functional Alignment via RL** (bioRxiv, 2025):
- Aligns protein language models to functional objectives using RL
- Demonstrates that RL can steer generation toward experimentally validated properties

**Implications for our project**:
- Our ESMFold-based GRPO reward is well-aligned with current best practices (ProteinZero uses the same approach)
- GRPO is the preferred algorithm over DPO for protein tasks due to flexibility with non-preference rewards
- RL should be applied AFTER a well-trained SFT model (it refines, not creates, capabilities)
- KL divergence regularization is important to prevent mode collapse

Sources: [ProtRL](https://github.com/AI4PDLab/ProtRL), [ProteinZero](https://arxiv.org/abs/2506.07459), [Supervision to Exploration](https://arxiv.org/abs/2510.01571), [Functional Alignment](https://www.biorxiv.org/content/10.1101/2025.05.02.651993v1)

---

#### 6. Continued Pre-training (Mid-training) for Biomedical/Protein Domains

Several papers have applied continued pre-training to adapt general LLMs to the biomedical domain:

| Model | Base | CPT Data | CPT Duration | Result |
|-------|------|----------|-------------|--------|
| **PMC-LLaMA** | LLaMA 7B/13B | 4.8M biomedical papers + 30K textbooks | Not specified | First biomedical domain-specific LLM |
| **BioMistral** | Mistral 7B | PubMed articles | 20h on 32xA100 | 0.9-point decrease initially; +2.9 with model merging |
| **BioMedGPT** | LLaMA2-Chat-7B | S2ORC biomedical literature | Not specified | Uses ESM-2 3B as protein encoder after CPT |
| **ESM-DBP** | ESM (general) | 170K DNA-binding protein sequences | Not specified | Domain-adaptive pre-training for protein subcategory |

**Key insight from BioMistral**: Continued pre-training can initially *decrease* performance on general benchmarks (catastrophic forgetting). Model merging with the original model (averaging weights) can recover general capabilities while retaining domain knowledge. This "forgetting then recovering" pattern is a known risk.

**Relevance to our project**: We do NOT do continued pre-training of the LLM (Qwen3). Instead, we use the frozen protein encoder (ESM-3) to inject domain knowledge via the projector. This is architecturally similar to how LLaVA avoids re-training the LLM on image data -- the encoder already has domain knowledge, the projector translates it. Our "mid-training" is actually the SFT stage (Stage 2), not continued pre-training in the traditional sense.

Sources: [BioMistral](https://arxiv.org/abs/2402.10373), [PMC-LLaMA](https://arxiv.org/abs/2304.14454), [ESM-DBP](https://www.nature.com/articles/s41467-024-52293-7)

---

#### 7. Summary: Where Our Project Sits in the Taxonomy

Our project's training pipeline maps to the multimodal LLM paradigm as follows:

```
Standard LLM:    Pre-train  -->  Mid-train (CPT)  -->  SFT  -->  RLHF/GRPO
Multimodal LLM:  [Frozen encoder]  -->  Projector Warm-up  -->  Instruction Tuning  -->  RL
Our Project:     [Frozen ESM-3]    -->  ??? (not yet)       -->  SFT (current)       -->  GRPO (planned)
```

**What we are currently doing** (single-stage SFT):
- Jointly training projector + LoRA from scratch on Mol-Instructions
- This combines Stages 1 and 2 into a single stage
- Differential LR (projector_lr=2e-3 vs base lr=2e-4) partially compensates

**What the literature recommends** (two-stage):
- Stage 1: Projector warm-up with simple protein-description pairs, LLM frozen
- Stage 2: Joint SFT with instruction data, projector + LoRA both trainable

**Recommendation**: Implement a projector warm-up stage before SFT. The evidence strongly suggests this will:
1. Reduce the initial loss spike (from ~35 to closer to ~12)
2. Speed up overall convergence
3. Produce better final performance
4. Only cost ~1-3 hours of additional training (based on LLaVA's 6h for 558K samples)

The Swiss-Prot protein-description pairs used by EvoLlama (369K samples) and ProtChatGPT (553K samples) would be suitable Stage 1 data. We already have Swiss-Prot processing capability in `src/data/swissprot_converter.py`.

---

### Session: Protein-Language Multimodal Model Survey (2023-2026)
**Date**: 2026-02-20
**Objective**: Survey how existing protein-language multimodal models project protein embeddings into LLM space, analogous to vision-language model approaches

---

### Comprehensive Survey: Protein-Language Multimodal Models

#### 1. ProteinGPT (2024, ICLR 2025)

**Paper**: "ProteinGPT: Multimodal LLM for Protein Property Prediction and Structure Understanding" ([arXiv 2408.11363](https://arxiv.org/abs/2408.11363))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | ESM-2 3B (esm2_t36_3B_UR50D, frozen) for sequence; ESM-IF1 (esm_if1_gvp4_t16_142M_UR50, frozen) for 3D structure |
| **Projection** | Linear projection layers (one per encoder), aligning to LLM embedding space |
| **Protein Tokens** | Residue-level tokens from both encoders, concatenated with special tokens `<Protein><Struct><Seq>` |
| **LLM Backbone** | Tested on Vicuna, LLaMA-2, LLaMA-3, Mistral |
| **Training Stage 1** | Modality Alignment: freeze encoders, train linear projectors only (10 epochs, lr=1e-4) |
| **Training Stage 2** | Instruction Tuning: fine-tune projectors + LLM on 3.7M QA pairs from 132K proteins (10 epochs, lr=1e-5) |
| **Frozen** | Both protein encoders (always frozen) |
| **Generation** | Understanding only (protein to text); no protein sequence generation |
| **Performance** | ~80% accuracy on closed-ended QA; BERTScore F1 0.70-0.82; outperforms GPT-3.5/GPT-4 on protein tasks |

**Relevance to our project**: Very similar to our architecture (frozen ESM encoder + linear/MLP projection + LLM). Key difference: they use ESM-2 3B + ESM-IF1 dual encoders, while we use ESM-3 (which already captures structure). Their 2-stage training (alignment then instruction tuning) is the standard recipe we follow.

---

#### 2. ProtChatGPT (2024)

**Paper**: "ProtChatGPT: Towards Understanding Proteins with Large Language Models" ([arXiv 2402.09649](https://arxiv.org/abs/2402.09649))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | ESM-1b (768-dim, frozen) for sequence; ESM-IF1 (512-dim, frozen) for structure |
| **Projection** | PLP-Former (Protein-Language Pretraining Transformer): 32 learnable query tokens (dim 768), cross-attention to ESM-1b features, contrastive + text-gen + matching objectives. Then a Multi-Level Projection Adapter (2 FC layers) maps to LLM space |
| **Protein Tokens** | 32 tokens (from learned queries) + structure projection, concatenated as soft prompts |
| **LLM Backbone** | Vicuna-13B (frozen) |
| **Training Stage 1** | PLP-Former training (20K epochs) on ProtDescribe dataset (553K sequence-description pairs) with 3 joint objectives |
| **Training Stage 2** | Adapter training (1K epochs) on RCSB-PDB (143.5K structure-description pairs); PLP-Former frozen in this stage |
| **Frozen** | Both protein encoders always frozen; LLM always frozen; PLP-Former frozen in stage 2 |
| **Generation** | Understanding only (protein to text) |
| **Performance** | BLEU-4: 0.394, ROUGE-L: 0.489, SPICE: 0.316, PubMed BERTScore: 0.457 |

**Relevance to our project**: The PLP-Former is essentially a Q-Former (from BLIP-2) adapted for proteins. Its advantage over our MLP projector is the fixed 32-token output regardless of protein length, plus the contrastive pre-training step. However, the added complexity (3 training objectives, separate PLP-Former stage) may not justify the gains. Our attention pooling to 32 tokens achieves a similar token compression with less complexity.

---

#### 3. Prot2Chat (2025)

**Paper**: "Prot2Chat: Protein LLM with Early Fusion of Sequence and Structure" ([arXiv 2502.06846](https://arxiv.org/abs/2502.06846), Bioinformatics 2025)

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | Modified ProteinMPNN (frozen): 9 pre-trained models concatenated (128x9=1152 dim). Structure via 3D coords (N, Ca, C, O), sequence via embedding initialization ("early fusion") |
| **Projection** | Cross-attention adapter with 256 learnable queries (BLIP-2 inspired). Queries attend to projected protein features via multi-head cross-attention |
| **Protein Tokens** | 256 tokens (from query count) fed as soft prompts to LLM |
| **LLM Backbone** | LLaMA-3 8B-Instruct |
| **Training** | Full adapter training (89.7M params) + LoRA on LLM (3.4M params), 2 epochs. Total: 93M trainable |
| **Frozen** | ProteinMPNN encoder |
| **Generation** | Understanding only (protein to text) |
| **Performance** | Mol-Instructions: BLEU-2=33.25, ROUGE-L=47.90; significantly outperforms sequence-only baselines |

**Relevance to our project**: The "early fusion" of sequence and structure within ProteinMPNN is interesting -- ESM-3 already achieves something similar natively with its multimodal pre-training. Their 256-query cross-attention adapter is heavier than our 32-token attention pooling. They demonstrate that joint sequence+structure encoding significantly outperforms sequence-only, which supports our choice of ESM-3 (which encodes both).

---

#### 4. InstructProtein (2024, ACL 2024)

**Paper**: "InstructProtein: Aligning Human and Protein Language via Knowledge Instruction" ([ACL Anthology](https://aclanthology.org/2024.acl-long.62/))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | No separate encoder -- protein sequences are directly tokenized as amino acid characters and fed into the LLM |
| **Projection** | None -- unified vocabulary for protein and text tokens |
| **Protein Tokens** | Each amino acid = one token (character-level), interleaved with natural language |
| **LLM Backbone** | Pre-trained on combined protein + natural language corpora (approach is model-agnostic) |
| **Training Stage 1** | Pre-training on both protein sequences and natural language text |
| **Training Stage 2** | Supervised instruction tuning with knowledge-graph-based instruction dataset |
| **Frozen** | Not applicable (single unified model) |
| **Generation** | BIDIRECTIONAL: protein to text (function description) AND text to protein (sequence generation) |
| **Performance** | Outperforms OPT, LLaMA, Alpaca on bidirectional protein-text tasks by large margins |

**Relevance to our project**: This is our "text" approach baseline. By tokenizing proteins as characters, it avoids the engineering of projectors entirely. However, it requires expensive pre-training on protein corpora (not just fine-tuning). The bidirectional generation capability is notable -- our embedding-based approach cannot generate protein sequences. If protein generation is needed, the text approach or a hybrid is required.

---

#### 5. ProLLaMA (2024)

**Paper**: "ProLLaMA: A Protein Large Language Model for Multi-Task Protein Language Processing" ([arXiv 2402.16445](https://arxiv.org/abs/2402.16445))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | No separate encoder -- amino acids tokenized directly using Pruned Vocabulary Partition (PVP): retain only uppercase AA letters + special tokens |
| **Projection** | None -- proteins share LLM vocabulary (pruned per stage) |
| **Protein Tokens** | Each amino acid = one token (character-level) |
| **LLM Backbone** | LLaMA-2 |
| **Training Stage 1** | Continual Learning: LoRA (rank=128, high rank) on all attention + FFN + embedding + head layers, on UniRef50 protein sequences |
| **Training Stage 2** | Instruction Tuning: LoRA (rank=64) on same targets, on UniRef50 + InterPro property texts (~13M instruction samples) |
| **Frozen** | Original LLaMA-2 weights frozen; only LoRA adapters trainable (~10% params) |
| **Generation** | BIDIRECTIONAL: protein understanding (67.1% superfamily exact match) AND unconditional/controllable protein generation |
| **Performance** | pLDDT 66.49 (comparable to natural proteins 68.25); TM-scores 0.71-0.93; +4.3% biophysical, +14.5% structural vs. baselines |

**Relevance to our project**: Demonstrates that text-based approaches (no encoder) can achieve both understanding and generation when equipped with enough protein pre-training. The high LoRA rank (128) for protein language learning is notable -- much higher than our r=8 for k/v only. This reflects the fundamental difference: they are teaching the LLM protein language from scratch, while we inject pre-computed embeddings from a dedicated protein model.

---

#### 6. ProteinCLIP (2024)

**Paper**: "ProteinCLIP: Enhancing Protein Language Models with Natural Language" ([bioRxiv 2024.05.14.594226](https://www.biorxiv.org/content/10.1101/2024.05.14.594226v1))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | ESM-2 (supports 6/12/30/33/36-layer variants) + ProtT5; encoder frozen, adapter added on top |
| **Text Encoder** | OpenAI text-embedding-3-large (frozen) |
| **Projection** | MLP adapter with single hidden layer: input_dim to hidden (GELU + LayerNorm) to 128-dim shared space (L2-normalized) |
| **Training** | CLIP-style symmetric cross-entropy contrastive loss with learnable temperature, on 465K UniProt sequence-function pairs |
| **Frozen** | Both encoders frozen; only adapter MLP trainable (~293K params for ESM-2 12-layer) |
| **Generation** | Neither -- alignment method only. Produces improved embeddings for downstream classifiers |
| **Performance** | Improves mutation sensitivity in 37/41 cases; PPI AUPRC: 0.697; CATH S20 homology top-1: ~0.661 |

**Relevance to our project**: ProteinCLIP is not a generative model but an alignment pre-training step. It demonstrates that CLIP-style contrastive alignment between protein embeddings and text descriptions can dramatically improve protein representations. This could be used as a pre-alignment step before our SFT training. The adapter is tiny (~293K params), making it very efficient.

---

#### 7. EvoLlama (2024)

**Paper**: "EvoLlama: Enhancing LLMs' Understanding of Proteins via Multimodal Structure and Sequence Representations" ([arXiv 2412.11618](https://arxiv.org/abs/2412.11618))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | ESM-2 650M (sequence) + ProteinMPNN encoder (structure); dual encoder |
| **Projection** | Separate MLPs for each encoder, mapping to LLM dimension. Then ELEMENT-WISE ADDITION of sequence + structure features (not concatenation). Reduces tokens by ~50% vs concat |
| **Protein Tokens** | Residue-level (L tokens for L residues after fusion) |
| **LLM Backbone** | LLaMA-3 8B |
| **Training Stage 1** | Projection Tuning: only MLP projectors train; both encoders + LLM frozen |
| **Training Stage 2** | Supervised Fine-tuning: projectors + protein encoders trainable; LLM frozen. Total trainable: 7.9% of model |
| **Frozen** | LLM always frozen (no LoRA); encoders frozen in stage 1, trainable in stage 2 |
| **Generation** | Understanding only (protein to text) |
| **Performance** | +1-8% over baselines in zero-shot; +6% average over SOTA with supervised fine-tuning |

**Relevance to our project**: EvoLlama uses element-wise addition of sequence+structure features (instead of concatenation) to keep token count manageable. Since ESM-3 already encodes both modalities, we get this "free." Notable: they unfreeze the protein encoders in stage 2 (we keep ESM-3 frozen always). Their LLM is fully frozen (no LoRA), relying entirely on the projectors -- opposite to our approach where LoRA adapts the LLM.

---

#### 8. ProtLLM (2024, ACL 2024)

**Paper**: "ProtLLM: An Interleaved Protein-Language LLM with Protein-as-Word Pre-Training" ([ACL Anthology](https://aclanthology.org/2024.acl-long.484/))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | ProtST (ESM-2 backbone + 2-layer MLP projection head, pre-trained with contrastive learning on protein-text pairs) |
| **Projection** | Input-layer: trainable projection matrix (protein to LLM space). Output-layer: reverse projection (LLM to protein space) for retrieval |
| **Protein Tokens** | Entire protein = 1 token ("protein-as-word"). Each protein encoded into a single vector, treated like a vocabulary word |
| **LLM Backbone** | LLaMA-7B with LoRA on all linear modules |
| **Training** | Protein-as-word language modeling on InterPT dataset (interleaved protein-text from annotations + papers). Protein cache for pre-computed vectors |
| **Frozen** | Protein encoder frozen during some tasks, trainable for others |
| **Generation** | Protein retrieval (not de novo generation); handles interleaved protein+text input with arbitrary protein count |
| **Performance** | EC Fmax: 0.860; GO-CC Fmax: 0.596; PPI accuracy: 89.87%; zero-shot + in-context learning demonstrated |

**Relevance to our project**: The "protein-as-word" paradigm (1 token per protein) is the extreme compression end of the spectrum, opposite to residue-level approaches. It works well for protein-level tasks (EC, GO, PPI) but loses residue-level information. Their PPI accuracy (89.87%) with in-context learning is impressive.

---

#### 9. BioMedGPT (2023-2024)

**Paper**: "BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine" ([arXiv 2308.09442](https://ar5iv.labs.arxiv.org/html/2308.09442))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | ESM-2 3B (36-layer transformer, frozen initially, then fine-tuned) |
| **Projection** | Fully-connected layer (modality adaptor) mapping residue features to LLM space |
| **Protein Tokens** | Each amino acid residue = one token in unified representation |
| **LLM Backbone** | BioMedGPT-LM-7B (Llama2-Chat-7B fine-tuned on 4.2M biomedical articles, 26B+ tokens) |
| **Training Stage 1** | Fine-tune Llama2-Chat-7B on biomedical literature to create BioMedGPT-LM-7B |
| **Training Stage 2** | Multimodal alignment: freeze LLM, train protein encoder + adaptors on UniProtQA |
| **Frozen** | LLM frozen during multimodal alignment |
| **Generation** | Understanding only (protein to text QA) |
| **Performance** | ROUGE-1: 0.743, BLEU-4: 0.535, METEOR: 0.754 on UniProtQA |

**Relevance to our project**: BioMedGPT uses the simplest possible projector (single FC layer), similar to early LLaVA. The domain-specific LLM pre-training (biomedical literature) before multimodal alignment is an extra step we do not currently do. Their strong results suggest domain-specific LLM adaptation helps significantly.

---

#### 10. Galactica (2022, Meta AI)

**Paper**: "Galactica: A Large Language Model for Science" ([arXiv 2211.09085](https://arxiv.org/abs/2211.09085))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | No separate encoder -- amino acid sequences directly tokenized |
| **Projection** | None -- proteins are inline text with special tokens |
| **Protein Tokens** | Each amino acid = one character token, wrapped in `[START_AMINO]`...`[END_AMINO]` special tokens |
| **LLM Backbone** | Galactica (decoder-only transformer, GeLU activation, learned positional embeddings, BPE vocabulary, no bias) |
| **Training** | Trained from scratch on 106B tokens of scientific literature, including protein sequences and SMILES |
| **Frozen** | N/A (trained from scratch) |
| **Generation** | Bidirectional: can generate text about proteins and generate protein sequences |

**Relevance to our project**: Galactica is a predecessor to the text-approach paradigm. The `[START_AMINO]`/`[END_AMINO]` special token wrapping is exactly what our text approach uses with `<protein>`/`</protein>` tags.

---

#### 11. Prot2Text (2024, AAAI 2024)

**Paper**: "Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers" ([AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28948))

| Aspect | Detail |
|--------|--------|
| **Protein Encoder** | ESM-2 (sequence) + RGCN (Relational Graph Convolution Network, for structure graphs) |
| **Projection** | Cross-attention between GNN graph embeddings and ESM sequence embeddings, fed into GPT-2 decoder |
| **Protein Tokens** | Fused graph+sequence representations |
| **LLM Backbone** | GPT-2 (smaller decoder, 398M total) |
| **Training** | End-to-end on 256K SwissProt proteins with textual descriptions |
| **Generation** | Understanding only (protein to text function description) |

---

#### 12. Newer Methods (2025-2026)

**InstructPLM-mu (2025)** ([arXiv 2510.03370](https://arxiv.org/abs/2510.03370)): Compares three multimodal fusion strategies for ESM-2 + structure encoders: Cross Attention, Channel-wise Concat, and Token-wise Concat. Uses MLP projector with "disentangled attention." Fine-tuned models match or surpass ESM-3 on mutation prediction, suggesting simpler models + good fusion can compete with massive multimodal pre-training.

**MULAN (2025)** ([Bioinformatics Advances](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf117/8139638)): Multimodal PLM for sequence + angle-based structure encoding. Uses a parameter-efficient "Structure Adapter" fused with pre-trained sequence encoder.

**Design Space Study (2025, ByteDance)** ([arXiv 2504.11454](https://arxiv.org/abs/2504.11454)): "Elucidating the Design Space of Multimodal Protein Language Models" systematically studies design choices. Key finding: tokenization loss and inaccurate structure token predictions are the major bottlenecks. Their 650M model achieves RMSD 2.36 on PDB testset (down from 5.52), outperforming 3B baselines.

---

### Comparative Summary Table

| Model | Year | Protein Encoder | Projector Type | Protein Tokens | LLM | Bidirectional? |
|-------|------|----------------|----------------|----------------|-----|----------------|
| **ProteinGPT** | 2024 | ESM-2 3B + ESM-IF1 (frozen) | Linear projection | Per-residue | Vicuna/LLaMA/Mistral | No |
| **ProtChatGPT** | 2024 | ESM-1b + ESM-IF1 (frozen) | PLP-Former (Q-Former) + FC | 32 (learned queries) | Vicuna-13B (frozen) | No |
| **Prot2Chat** | 2025 | ProteinMPNN x9 (frozen) | Cross-attn (256 queries) | 256 | LLaMA-3 8B | No |
| **InstructProtein** | 2024 | None (text tokens) | None | Per-AA character | Custom pre-trained | Yes |
| **ProLLaMA** | 2024 | None (text tokens) | None | Per-AA character | LLaMA-2 | Yes |
| **ProteinCLIP** | 2024 | ESM-2/ProtT5 (frozen) | MLP adapter to 128d | N/A (alignment) | N/A | N/A |
| **EvoLlama** | 2024 | ESM-2 650M + ProteinMPNN | MLPs + elem-wise add | Per-residue | LLaMA-3 8B (frozen) | No |
| **ProtLLM** | 2024 | ProtST/ESM-2 (frozen) | Projection matrix | 1 (protein-as-word) | LLaMA-7B + LoRA | Retrieval |
| **BioMedGPT** | 2023 | ESM-2 3B | Single FC layer | Per-residue | Llama2-Chat-7B | No |
| **Galactica** | 2022 | None (text tokens) | None | Per-AA character | Galactica (scratch) | Yes |
| **Our Project** | 2026 | ESM-3 small (frozen) | AttentionPooling + 2-layer MLP | 32 (pooled) | Qwen3-4B + LoRA k/v | No |

---

### Key Takeaways for Our Project

1. **Our architecture is well-positioned.** The ESM-3 (frozen) + Attention Pooling (32 tokens) + MLP Projector + LLM (LoRA) pattern closely mirrors the most successful approaches (ProteinGPT, EvoLlama, BioMedGPT). Using ESM-3 instead of ESM-2 + separate structure encoder simplifies the pipeline while capturing both sequence and structure.

2. **32 tokens is a reasonable compression.** ProtChatGPT uses 32 (Q-Former), Prot2Chat uses 256 (cross-attention), ProtLLM uses 1 (protein-as-word). Our 32-token attention pooling is a good middle ground -- enough to preserve important residue-level patterns without overwhelming the LLM context.

3. **Two-stage training is universal.** Every successful model uses Stage 1 (alignment/projection training with frozen LLM) followed by Stage 2 (instruction tuning with LLM adaptation). Our single-stage joint training is simpler but may benefit from separating stages.

4. **MLP projection is sufficient.** Despite the existence of Q-Former, cross-attention adapters, and other complex projection methods, simple MLP projectors (ProteinGPT, EvoLlama, BioMedGPT) perform comparably. The LLaVA lesson applies to proteins: MLP works.

5. **Consider contrastive pre-alignment.** ProteinCLIP shows that a cheap contrastive alignment step (~293K params) between protein embeddings and text significantly improves downstream performance. Adding a CLIP-style pre-alignment of ESM-3 embeddings before SFT could be beneficial.

6. **Protein generation requires text-based approach.** All models that support bidirectional protein generation (InstructProtein, ProLLaMA, Galactica) use text-based tokenization, not embeddings. If protein generation is a goal, our `text` approach baseline is the path forward.

7. **Domain LLM pre-training helps.** BioMedGPT's pre-training on biomedical literature before multimodal alignment improved performance. Our use of Qwen3-4B (general-purpose) could potentially benefit from intermediate biomedical text fine-tuning.

8. **High LoRA rank for text-based approaches.** ProLLaMA uses rank=128 when teaching the LLM protein language from scratch. Our rank=8 is appropriate for the embedding-based approach (where the protein encoder does the heavy lifting), but if we explore the text approach, higher ranks are needed.

---

### Session: Initial Research Phase
**Date**: 2026-02-14
**Objective**: Conduct literature review for protein-LLM post-training

---

## Agent-1: Protein Language Models & Embeddings

**Focus Areas**:
- Existing protein language models (ESM, ProtTrans, etc.)
- Protein embedding representations
- How to integrate protein embeddings into LLMs

### Findings Summary:

#### Key Protein Language Models

| Model | Parameters | Embedding Dim | Training Data |
|-------|------------|---------------|---------------|
| ESM-2 650M | 650M | 1,280 | UniProt 2021_04 |
| ESM-2 3B | 3B | 2,560 | UniProt 2021_04 |
| ESM-3 | 98B | Multimodal | 2.78B proteins |
| ProtT5-XL | 3B | 1,024 | UniRef50 |
| Ankh | 1.15B | ~1,536 | Efficient training |

#### Key Insights:
1. **ESM-3 small** selected as default encoder (multimodal: sequence + structure)
2. Performance levels off around 650M parameters - larger models don't significantly outperform
3. **ESM-3** is the first multimodal generative model with sequence, structure, AND function tokens
4. **ProtT5** encoder-only version fits on 8GB GPU in half-precision

#### Integration Patterns:
- **Pattern A**: Linear Projection (ProteinGPT) - simple alignment
- **Pattern B**: Cross-Attention Adapter (Prot2Chat) - early fusion
- **Pattern C**: Knowledge Graph Instruction (InstructProtein) - bidirectional generation

**Full report**: See detailed research in agent output logs

---

## Agent-2: LLM Post-Training Methods (SFT/RL)

**Focus Areas**:
- Supervised Fine-Tuning (SFT) techniques
- Reinforcement Learning methods (RLHF, DPO, PPO)
- Training frameworks (TRL, veRL, OpenRLHF)

### Findings Summary:

**Full report saved to**: `LLM_Post_Training_Methods_Summary.md`

#### Framework Comparison

| Framework | Key Strength | Best For |
|-----------|--------------|----------|
| TRL | HuggingFace integration | Beginners, prototyping |
| veRL | Scalability (671B models) | Large-scale research |
| OpenRLHF | Easy + high-performance | Production, multimodal |

#### Recommended Training Strategy:

**Phase 1 - SFT with QLoRA:**
- 4-bit quantization: 70B model → ~46GB VRAM
- LoRA rank: r=8 (minimum r=4 for proteins)
- Learning rate: 2e-4
- Epochs: 1-3

**Phase 2 - Preference Alignment:**
- **DPO**: Simple, no reward model needed
- **GRPO**: 50% less memory than PPO, better for reasoning
- **Avoid PPO** unless significant compute available

#### Protein-Specific Insights:
- Apply LoRA to key/value matrices only (differs from NLP!)
- Fine-tuned ESM2-150M can outperform ESM2-650M/3B without fine-tuning
- LoRA reduces ProtT5 (1.2B) trainable params to ~3.5M

---

## Agent-3: Multimodal LLM Integration

**Focus Areas**:
- How to incorporate non-text modalities into LLMs
- Projection layers, adapters, cross-attention mechanisms
- Examples from vision-language models

### Findings Summary:

#### Projection Methods Comparison

| Method | Pros | Cons |
|--------|------|------|
| MLP Projector | Simple, preserves spatial info | High token count |
| Q-Former | Fixed output tokens | Information loss risk |
| Perceiver Resampler | Variable input, fixed output | Struggles with spatial |
| Linear | Simplest | Limited expressiveness |

#### Key VLM Architectures for Reference:
- **LLaVA**: CLIP + 2-layer MLP + Vicuna (two-stage training)
- **InternVL 2.5**: Large ViT-6B + pixel unshuffle → 256 tokens
- **Qwen2-VL**: Dynamic resolution with 2D-RoPE

#### Recommended Architecture for Proteins:

```
ESM-3 small (frozen) → Per-residue [L, 1536]
       ↓
Attention Pooling (32 tokens)
       ↓
MLP Projector (1536 → LLM_dim)
       ↓
[protein_tokens] + [text_tokens] → LLM (with LoRA)
```

#### Training Recipe:
| Stage | Frozen | Trainable | Data |
|-------|--------|-----------|------|
| 1. Alignment | ESM-3 + LLM | Projector only | Protein-text pairs |
| 2. Instruction | ESM-3 | Projector + LLM (LoRA) | QA + text-only |

#### Variable Length Handling:
- **Direct mapping**: Each residue → one token (preserves info)
- **Attention pooling**: BoM-Pooling with window 80 outperforms mean pooling by 4%
- **Avoid mean pooling**: Treats all residues equally (biologically incorrect)

---

## Agent-4: Protein Datasets & Benchmarks

**Focus Areas**:
- Existing protein-text datasets
- Evaluation benchmarks for protein understanding
- Task definitions (function prediction, interaction, etc.)

### Findings Summary:

**Full report saved to**: `research/protein_datasets_and_benchmarks.md`

#### Training Datasets

| Dataset | Size | Format |
|---------|------|--------|
| Mol-Instructions (protein) | 505K | Instruction format |
| ProteinChat | 1.5M+ triplets | (protein, prompt, answer) |
| Swiss-Prot | 570K+ proteins | Curated annotations |
| UniRef50 | 30M sequences | Pre-training |

#### Evaluation Benchmarks

| Task | Benchmark | Metrics |
|------|-----------|---------|
| Function Prediction | CAFA5, PROBE | Fmax, AUPR |
| PPI Prediction | BioSNAP, STRING | AUC, MCC |
| Stability | S669, Megascale | Pearson, Spearman |
| Subcellular Localization | PEER | Accuracy, F1 |
| Drug-Target Interaction | Davis, KIBA | MSE, CI |

#### Recommended Evaluation Plan:
1. **Core tasks**: GO term prediction, PPI, stability, localization
2. **Baseline comparisons**: ESM2-650M/3B, ProtT5-XL, ProstT5
3. **Phases**: Zero-shot → task-specific → cross-domain transfer

---

## Protein Generation Research (2026-02-20)

### Overview: The Reverse Direction Problem

Our current pipeline goes: Protein -> ESM-3 embedding -> Projector -> LLM -> Text output. The "reverse direction" asks: can we go from LLM hidden states or text descriptions BACK to protein sequences or structures? This is critical for protein design applications.

### 1. Autoregressive Protein Generation Models

These models generate proteins token-by-token, exactly like GPT generates text.

| Model | Params | Architecture | Conditioning | Key Result |
|-------|--------|-------------|-------------|------------|
| **ProGen** | 1.2B | Decoder Transformer | Family/function tags | Functional lysozymes validated experimentally |
| **ProGen2** | up to 6.4B | Decoder Transformer | Unconditional + conditional | Explores boundaries of protein LMs |
| **ProtGPT2** | 738M | GPT-2 style | Unconditional | 88% globular proteins, natural-like |
| **xTrimoPGLM** | 100B | Unified (MLM + AR) | Programmable after SFT | SOTA on 18 benchmarks + generation |
| **ProLLaMA** | LLaMA-based | Decoder Transformer | Text descriptions | 67.1% superfamily prediction, controllable gen |
| **MP4** | Transformer | Decoder | Natural language prompts | 84% expression rate, Tm >62C |

**How they work**: Standard next-token prediction over a vocabulary of 20 amino acids (+ special tokens). ProGen adds control tags (taxonomy, function, localization) as prefix tokens. ProLLaMA fine-tunes LLaMA on both protein sequences and natural language instructions.

**Relevance to our project**: The simplest path to protein generation. Our Qwen3-4B already has amino acid characters in its tokenizer. We could fine-tune it to generate protein sequences autoregressively using `<protein>...</protein>` delimiters. ProLLaMA demonstrates this works with LLaMA.

### 2. Text-Conditioned Protein Design Frameworks

These take natural language descriptions and produce protein sequences.

#### ProteinDT (Nature Machine Intelligence, 2025)
- **Architecture**: Three-stage pipeline inspired by DALL-E 2 / unCLIP
  1. **ProteinCLAP**: Contrastive learning to align text and protein embeddings (like CLIP)
  2. **ProteinFacilitator**: Maps text embeddings -> protein representation space (Gaussian mapping, L2 loss)
  3. **Decoder**: Generates sequences from protein representations
- **Decoder variants tested**:
  - Autoregressive T5-Base (best performance)
  - Diffusion with RNN transition
  - Diffusion with BERT-Base transition
- **Training data**: SwissProtCLAP (441K text-protein pairs)
- **Performance**: >90% accuracy on text-guided generation
- **Key insight**: The AR decoder outperformed diffusion decoders for text-to-protein

#### Pinal (bioRxiv, 2024)
- **Architecture**: Two-stage decomposition (16B parameters total)
  1. **T2struct**: Text -> structural tokens (encoder-decoder, up to 15B params)
  2. **SaProt-T**: Structure + text -> amino acid sequence
- **Training data**: 1.7 billion text-protein pairs
- **Text encoder**: PubMedBERT (109M)
- **Key insight**: Generates structure FIRST, then designs sequence to fold into it. 4/8 designed ADH enzymes showed functional activity.

#### BioM3 (bioRxiv, 2024)
- **Architecture**: Three-stage framework
  1. Contrastive alignment of protein and text representations
  2. Text embedding refinement
  3. Conditional generation via discrete autoregressive diffusion
- **Validation**: In vivo and in vitro tests of designed SH3 domain proteins with native-like folds

#### MP4 (bioRxiv, 2025)
- **Architecture**: Transformer-based generative model
- **Input**: Natural language prompts encoding fitness criteria, physical properties, source organism
- **Training**: 3.2B data points, 138K tokens
- **Validation**: 96 prompts tested; 84% expression rate, Tm >62C, some approaching 90C
- **Key insight**: Generalist model -- single model handles diverse functions

### 3. Bidirectional Models (Understand AND Generate)

| Model | Base | Direction | Key Feature |
|-------|------|-----------|-------------|
| **InstructProtein** | Custom LLM | Text <-> Protein | Knowledge graph instruction tuning |
| **ProLLaMA** | LLaMA | Text <-> Protein | EPGF framework for biological viability |
| **xTrimoPGLM** | Custom 100B | Understanding + Generation | Unified MLM + autoregressive pretraining |
| **Prot2Token** | PLM + decoder | Understanding + Generation | All tasks as next-token prediction |

**InstructProtein** is the most relevant: it pre-trains on both protein and natural language corpora, then uses supervised instruction tuning to align the two languages. It can take a protein sequence and predict its function (understanding) OR take a text description and generate a protein sequence (generation). Training uses knowledge-graph-based instructions.

### 4. Diffusion-Based Protein Generation

| Model | Domain | Conditioning | Notes |
|-------|--------|-------------|-------|
| **RFdiffusion** | 3D backbone structures | Symmetry, motif scaffolding, shape | State-of-the-art structure generation |
| **Chroma** | 3D structures + sequences | Symmetry, shape, class, **natural language** | Text-conditioned structure generation |
| **FrameDiff** | 3D backbone frames | Unconditional | SE(3) diffusion on frames |
| **EvoDiff** | Sequences only | Motif scaffolding, unconditional | First sequence-space diffusion model |

**Chroma** is notable: it accepts natural language text prompts to condition protein structure generation. RFdiffusion uses 2D constraints. EvoDiff operates purely in sequence space using discrete diffusion, generating proteins inaccessible to structure-based models (e.g., disordered regions).

### 5. Embedding-to-Sequence Decoding

#### CHEAP Embeddings (Cell Patterns, 2025)
- Compresses ESMFold latent space into compact "CHEAP" embeddings
- Sequence decoder: 2-layer FC network (hidden=1024) maps embeddings back to amino acid sequences
- Achieves 128x channel compression and 8x length compression while retaining <2A structure accuracy
- Demonstrates that continuous protein embeddings CAN be decoded back to discrete sequences

#### ESMFold / AlphaFold as Downstream Tools
- ESMFold uses ESM-2 token embeddings to predict 3D structure end-to-end (no MSA needed)
- Structure prediction accuracy correlates with language model perplexity
- Can be used as a "folding oracle" after sequence generation: generate sequence -> fold with ESMFold

### 6. Structure Generation from LLM Outputs

**Direct structure prediction from embeddings is possible but complex.** The practical approach:
1. Generate amino acid sequence using LLM
2. Fold it with ESMFold (fast, single-sequence) or AlphaFold (accurate, needs MSA)
3. Validate designability (does the generated sequence actually fold into a stable structure?)

ESM-3 can directly generate both sequences AND structures simultaneously by reasoning over joint sequence/structure/function tokens. It generated esmGFP (58% identity to nearest natural GFP), equivalent to simulating 500M years of evolution.

### 7. RL/DPO for Protein Generation

| Method | Paper | Key Idea |
|--------|-------|----------|
| **CtrlProt** | AAAI 2025 | Multi-listwise preference optimization for controllable protein generation |
| **g-DPO** | 2024 | Adapts DPO for experimentally labeled protein data (scalar labels -> preferences) |
| **ResiDPO** | 2025 | Residue-level structural feedback for designability optimization |
| **KPO** | 2025 | Knowledge preference optimization for safe/controllable protein generation |

**CtrlProt** fine-tunes a protein LLM with multi-listwise preference optimization to support multi-attribute controllable generation. **g-DPO** addresses the challenge that protein datasets have scalar fitness labels rather than pairwise preferences (converts scalar to preference pairs). Both are directly applicable to our GRPO stage.

### 8. Practical Recommendations for Our Project

#### Immediate Path: Text-Based Protein Generation (Path A)
```
User: "Design a thermostable enzyme that catalyzes alcohol dehydrogenation"
     |
     v
Qwen3-4B (LoRA) -> "<protein>MKTLIVG...LASTAA</protein>"
     |
     v
ESMFold -> 3D structure prediction + validation
```

**Implementation**:
1. Add protein sequences to our training data with `<protein>...</protein>` tags
2. Create instruction pairs: "Design a protein with [function]" -> "<protein>[sequence]</protein>"
3. Fine-tune Qwen3-4B with SFT on these pairs (same as current pipeline)
4. Use GRPO with structural validation rewards (ESMFold pLDDT, pTM)

**Training data needed**: Swiss-Prot function descriptions paired with sequences (~570K pairs)

#### Advanced Path: Dedicated Protein Decoder (Path B)
```
User: "Design a protein that binds to insulin receptor"
     |
     v
Qwen3-4B -> hidden states [batch, seq_len, 2560]
     |
     v
Protein Decoder Head (2560 -> 20 amino acids)
     |
     v
Generated sequence: "MKTLIVG..."
```

**Implementation**:
1. Add a linear projection head: `nn.Linear(2560, 20)` (20 standard amino acids)
2. Train with cross-entropy loss on protein sequence prediction
3. During generation, use special `<generate_protein>` token to trigger protein decoding mode
4. Optionally, use a small autoregressive decoder (like T5-small) instead of single linear head

#### Structure Generation Pipeline
```
Generated sequence -> ESMFold/ESM-3 -> 3D structure
                   -> ESM-3 (with structure track) -> refined structure
                   -> AlphaFold -> high-accuracy structure
```

### 9. Key References

| Paper | Year | URL |
|-------|------|-----|
| ProteinDT | 2025 | https://www.nature.com/articles/s42256-025-01011-z |
| Pinal | 2024 | https://www.biorxiv.org/content/10.1101/2024.08.01.606258v4 |
| BioM3 | 2024 | https://github.com/PraljakReps/BioM3 |
| MP4 | 2025 | https://www.biorxiv.org/content/10.1101/2025.03.21.644400v2 |
| InstructProtein | 2023 | https://arxiv.org/abs/2310.03269 |
| ProLLaMA | 2024 | https://arxiv.org/abs/2402.16445 |
| xTrimoPGLM | 2025 | https://www.nature.com/articles/s41592-025-02636-z |
| ProGen | 2023 | https://www.nature.com/articles/s41587-022-01618-2 |
| ProtGPT2 | 2022 | https://www.nature.com/articles/s41467-022-32007-7 |
| ESM-3 | 2024 | https://www.science.org/doi/10.1126/science.ads0018 |
| RFdiffusion | 2023 | https://www.nature.com/articles/s41586-023-06728-8 |
| Chroma | 2023 | https://www.pnas.org/doi/10.1073/pnas.2311500121 |
| EvoDiff | 2024 | https://github.com/microsoft/evodiff |
| CHEAP | 2025 | https://www.cell.com/patterns/fulltext/S2666-3899(25)00137-0 |
| Prot2Token | 2024 | https://arxiv.org/abs/2505.20589 |
| Prot2Chat | 2025 | https://arxiv.org/html/2502.06846v1 |
| CtrlProt | 2025 | https://arxiv.org/abs/2501.15007 |
| g-DPO | 2024 | https://arxiv.org/pdf/2510.19474v1 |
| EvoLlama | 2024 | https://arxiv.org/html/2412.11618v1 |
| ProteinGPT | 2024 | https://arxiv.org/html/2408.11363v1 |

---

## Open Questions for Discussion

1. [QUESTION] Which base LLM should we use? (LLaMA-3, Mistral, Qwen, etc.)
   - **Agent-2 suggests**: Qwen-2.5 7B or Llama-3 8B as starting points

2. [QUESTION] What specific downstream tasks are highest priority?
   - **Agent-4 suggests**: GO term prediction, PPI, stability (fundamental biology)

3. [QUESTION] GPU resources available for training?
   - **RESOLVED**: 8x NVIDIA H100 80GB available (CUDA 13.0)

4. [QUESTION] Should we focus on single proteins or protein-protein interactions?
   - **Recommendation**: Start with single protein understanding, add PPI later

5. [QUESTION] Third protein incorporation method (TBD)?
   - **Options identified**:
     - Structure-aware (ESM-IF1, ProteinMPNN)
     - Graph Neural Networks (GearNet, GVP)
     - Hybrid text + embedding approach

---

### Session: Protein Language Model Encoders Survey (Alternative to ESM-3)
**Date**: 2026-03-02
**Objective**: Survey protein language model encoders that could serve as frozen encoders in a multimodal LLM setup (frozen encoder -> pooling/projector -> LLM), as alternatives or comparisons to ESM-3 small.

---

#### 1. Encoder Comparison Table

| Model | Params | Embed Dim | Layers | VRAM (fp16) | Per-Residue | Structure-Aware | HuggingFace |
|-------|--------|-----------|--------|-------------|-------------|-----------------|-------------|
| **ESM-3 small** (current) | 1.4B | 1536 | 48 | ~3-5 GB | Yes | Yes (multi-track) | `EvolutionaryScale/esm3-sm-open-v1` |
| **ESM-2 650M** | 650M | 1280 | 33 | ~1.3 GB | Yes | No (seq only) | `facebook/esm2_t33_650M_UR50D` |
| **ESM-2 3B** | 3B | 2560 | 36 | ~6 GB | Yes | No (seq only) | `facebook/esm2_t36_3B_UR50D` |
| **ESM C 300M** | 300M | 960 | 30 | ~0.6 GB | Yes | No (seq only) | `EvolutionaryScale/esmc-300m-2024-12` |
| **ESM C 600M** | 600M | 1152 | 36 | ~1.2 GB | Yes | No (seq only) | `EvolutionaryScale/esmc-600m-2024-12` |
| **ProtT5-XL-UniRef50** (enc) | ~1.2B (enc) | 1024 | 24 | ~2.4 GB | Yes | No (seq only) | `Rostlab/prot_t5_xl_half_uniref50-enc` |
| **ProstT5** | ~3B (full) | 1024 | 24 (enc) | ~2.4 GB (enc) | Yes | Yes (3Di) | `Rostlab/ProstT5` |
| **SaProt 650M** | 650M | 1280 | 33 | ~1.3 GB | Yes | Yes (3Di+AA) | `westlake-repl/SaProt_650M_AF2` |
| **SaProt 1.3B** | 1.3B | TBD | TBD | ~2.6 GB | Yes | Yes (3Di+AA) | `westlake-repl/SaProt_1.3B_AF2` |
| **ProSST** | ~100M | 768 | 12 | ~0.2 GB | Yes | Yes (quantized) | `AI4Protein/ProSST-2048` |
| **Ankh2-Large** | ~2B (full) | 1536 | TBD | ~2-3 GB (enc) | Yes | No (seq only) | `ElnaggarLab/ankh2-large` |

#### 2. Detailed Encoder Profiles

##### ESM-2 (Meta/FAIR)
- **Paper**: Lin et al. 2023, "Evolutionary-scale prediction of atomic-level protein structure with a language model"
- **Architecture**: BERT-style masked language model trained on UniRef50
- **Model sizes**: 8M, 35M, 150M, 650M, 3B, 15B
- **Key specs (650M)**: 33 layers, 1280 hidden dim, 20 attention heads
- **Key specs (3B)**: 36 layers, 2560 hidden dim, 40 attention heads
- **Per-residue**: Yes, last_hidden_state gives [seq_len, embed_dim]
- **Advantages over ESM-3**:
  - Much smaller (650M vs 1.4B) = less VRAM for frozen encoder
  - Native HuggingFace Transformers integration (no custom SDK)
  - Extensively benchmarked; well-understood embedding space
  - 3B variant has 2560-dim embeddings (matches Qwen3-4B hidden size directly)
  - Can use standard `EsmModel.from_pretrained()`, no license agreement needed
- **Disadvantages**: Sequence-only (no structure awareness), older model (2023)
- **HuggingFace**: `facebook/esm2_t33_650M_UR50D`, `facebook/esm2_t36_3B_UR50D`

##### ESM C (EvolutionaryScale, Dec 2024)
- **Paper**: Hayes et al. 2024, "Simulating 500 million years of evolution with a language model"
- **Architecture**: Transformer masked language model, successor to ESM-2 with improved scaling
- **Model sizes**: 300M (960-dim), 600M (1152-dim), 6B (API-only)
- **Key performance**: 300M matches ESM-2 650M; 600M rivals ESM-2 3B
- **Per-residue**: Yes
- **Advantages over ESM-3**:
  - Much more parameter-efficient (600M matches ESM-2 3B quality)
  - 300M variant extremely lightweight (~0.6 GB) for rapid iteration
  - Open weights for 300M and 600M
- **Disadvantages**: Newer model with less community adoption; requires `esm` SDK
- **HuggingFace**: `EvolutionaryScale/esmc-300m-2024-12`, `EvolutionaryScale/esmc-600m-2024-12`

##### ProtT5-XL-UniRef50 (Rostlab/TUM)
- **Paper**: Elnaggar et al. 2022, "ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning"
- **Architecture**: T5-based (encoder-decoder), encoder-only variant available
- **Key specs**: ~1.2B encoder params, 1024-dim embeddings, 24 encoder layers
- **Per-residue**: Yes, encoder last_hidden_state gives [seq_len, 1024]
- **Advantages over ESM-3**:
  - Encoder-only half-precision variant fits in 8GB VRAM
  - T5 architecture is well-understood in NLP community
  - Extensive benchmarking on ProteinGym and other tasks
  - Native HuggingFace Transformers (T5EncoderModel)
- **Disadvantages**: Sequence-only, older (2022), 1024-dim is smaller than ESM-3's 1536
- **HuggingFace**: `Rostlab/prot_t5_xl_half_uniref50-enc`

##### ProstT5 (Rostlab/TUM)
- **Paper**: Heinzinger et al. 2024, "Bilingual Language Model for Protein Sequence and Structure"
- **Architecture**: ProtT5-XL fine-tuned to translate between AA sequences and 3Di structure tokens
- **Key specs**: ~3B total (encoder ~1.2B), 1024-dim, 24 encoder layers
- **Per-residue**: Yes (same as ProtT5)
- **Advantages over ESM-3**:
  - Structure-aware through 3Di tokens (Foldseek)
  - Can generate 3Di from sequence (no structure needed at inference)
  - Builds on well-established ProtT5 architecture
- **Disadvantages**: Still sequence-only input for encoder embeddings (3Di is a separate mode); full model is 3B
- **HuggingFace**: `Rostlab/ProstT5`

##### SaProt (Westlake University)
- **Paper**: Su et al. 2024, "SaProt: Protein Language Modeling with Structure-aware Vocabulary" (ICLR 2024)
- **Architecture**: ESM-2 architecture with structure-aware vocabulary (AA+3Di tokens via Foldseek)
- **Model sizes**: 650M (33 layers, 1280-dim), 1.3B
- **Per-residue**: Yes, same as ESM-2 but with interleaved structure tokens
- **Advantages over ESM-3**:
  - Explicit structure awareness through 3Di vocabulary
  - Same architecture as ESM-2 (drop-in replacement, HuggingFace compatible)
  - #1 on ProteinGym benchmark at release
  - 650M variant is very lightweight (~1.3 GB)
  - Requires Foldseek preprocessing but AlphaFold2 structures suffice
- **Disadvantages**: Requires 3Di tokens (need Foldseek or pre-computed structures); interleaved AA+3Di doubles effective sequence length; 1280-dim (vs ESM-3's 1536)
- **HuggingFace**: `westlake-repl/SaProt_650M_AF2`, `westlake-repl/SaProt_1.3B_AF2`

##### ProSST (NeurIPS 2024)
- **Paper**: Li et al. 2024, "ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention"
- **Architecture**: BERT-style with disentangled attention for sequence and structure tokens
- **Key specs**: ~100M params, 768-dim, 12 layers, 12 attention heads
- **Per-residue**: Yes
- **Advantages over ESM-3**:
  - Extremely lightweight (100M vs 1.4B)
  - Disentangled attention explicitly models seq-structure relationships
  - State-of-the-art zero-shot mutation effect prediction
- **Disadvantages**: Much smaller model (768-dim vs 1536-dim); less proven for general protein understanding; requires structure quantization preprocessing
- **HuggingFace**: `AI4Protein/ProSST-2048`

##### Ankh2/Ankh3 (ElnaggarLab/Proteinea)
- **Paper**: Elnaggar et al. 2023, "Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling"
- **Architecture**: T5-based, encoder-decoder; Ankh3 adds multi-task pre-training (MLM + sequence completion)
- **Model sizes**: Ankh2-Large (~2B, 1536-dim), Ankh3-XL (TBD), Ankh3-Large
- **Per-residue**: Yes, encoder outputs [seq_len, 1536]
- **Advantages over ESM-3**:
  - Ankh2 has 1536-dim embeddings (same as ESM-3!)
  - Optimized training (fewer parameters for comparable performance)
  - Ankh3 (2025) adds sequence completion pre-training objective
- **Disadvantages**: Less widely adopted; T5-based (encoder-decoder overhead if not using encoder-only); Ankh3-XL license is CC-BY-NC-SA
- **HuggingFace**: `ElnaggarLab/ankh2-large`, `ElnaggarLab/ankh3-xl`

#### 3. Recommendations for Our Pipeline

**Tier 1 (Best candidates for comparison experiments)**:
1. **ESM-2 650M** -- Drop-in replacement with ~half the VRAM. Native HuggingFace. Would need projector adjustment (1280 -> LLM dim instead of 1536). Best baseline to isolate "does a bigger encoder help?"
2. **ESM C 600M** -- Newest generation, matches ESM-2 3B performance at 600M params. 1152-dim embeddings. Tests whether newer training data/methods matter more than model size.
3. **SaProt 650M** -- Same size as ESM-2 650M but structure-aware. Tests whether structure information in the encoder improves downstream tasks.

**Tier 2 (Worth exploring)**:
4. **ProtT5-XL encoder** -- Different architecture family (T5 vs BERT). 1024-dim. Tests whether architecture diversity matters.
5. **ESM-2 3B** -- Larger model with 2560-dim (matches Qwen3-4B hidden size directly, could skip projector entirely). Tests scaling.

**Tier 3 (Niche use cases)**:
6. **ProSST** -- Very lightweight (100M). Good for ablation on "how small can the encoder be?"
7. **ProstT5** -- Structure-aware ProtT5. Interesting but complex setup.
8. **Ankh2-Large** -- 1536-dim like ESM-3 but less tested.

#### 4. VRAM Budget Analysis (80GB H100)

Current setup: ESM-3 (1.4B, fp32) + Qwen3-4B (LoRA) = ~10-12 GB model weights

| Configuration | Encoder VRAM | LLM VRAM | Projector | Total (est.) |
|---------------|-------------|----------|-----------|--------------|
| ESM-3 1.4B (fp32) + Qwen3-4B | ~5.6 GB | ~8 GB | ~0.1 GB | ~14 GB |
| ESM-2 650M (fp32) + Qwen3-4B | ~2.6 GB | ~8 GB | ~0.1 GB | ~11 GB |
| ESM C 600M (fp32) + Qwen3-4B | ~2.4 GB | ~8 GB | ~0.1 GB | ~11 GB |
| SaProt 650M (fp32) + Qwen3-4B | ~2.6 GB | ~8 GB | ~0.1 GB | ~11 GB |
| ESM-2 3B (fp32) + Qwen3-4B | ~12 GB | ~8 GB | ~0.1 GB | ~20 GB |
| ProtT5-XL enc (fp16) + Qwen3-4B | ~2.4 GB | ~8 GB | ~0.1 GB | ~11 GB |

All configurations fit comfortably on a single H100 80GB. Activation memory (batch-dependent) adds 20-40 GB.

#### 5. Integration Complexity

- **ESM-2**: Easiest. HuggingFace `EsmModel`, `.last_hidden_state` gives per-residue embeddings directly.
- **ESM C**: Moderate. Requires EvolutionaryScale `esm` SDK (same as ESM-3).
- **ProtT5-XL**: Easy. HuggingFace `T5EncoderModel`, input needs space-separated amino acids.
- **SaProt**: Moderate. Needs Foldseek 3Di tokens. Can use pre-computed AF2 structures.
- **ProSST**: Moderate. Needs structure quantization preprocessing.
- **ProstT5**: Moderate. Same as ProtT5 but with 3Di token support.

#### 6. Related Multimodal Protein-LLM Architectures

Several recent papers have used the frozen-encoder + projector + LLM paradigm:
- **EvoLlama** (Dec 2024): ESM-2 + ProteinMPNN (structure) + projector + Llama-3
- **ProteinGPT** (ICLR 2025): Protein encoder + linear projection + LLM
- **Prot2Chat** (2025): Early fusion of sequence + structure encoders + LLM
- **ProtLLM**: Protein-as-Word pre-training with dynamic protein mounting

These validate our LLaVA-style approach and suggest ESM-2 is the most commonly used frozen encoder in practice.

---

## Action Items

### Completed
- [x] Complete literature review with 4 agents
- [x] Set up conda environment (`setup_env.sh`)
- [x] Create base project structure (`src/`)
- [x] Project restructuring for Claude Code & Hydra (2026-02-16)
- [x] Create Hydra configuration system
- [x] Set up `.claude/` with skills, commands, agents
- [x] Create entry point scripts (`scripts/`)
- [x] Set up test structure (`tests/`)

### In Progress
- [ ] ESM-3 + Qwen3-4B integration (GPU verification pending)
- [ ] Fix GRPO gradient flow bug
- [ ] Fix evaluate.py argument parsing bugs
- [x] Select training framework: TRL for SFT, custom for GRPO
- [x] Select base LLM: **Qwen3-4B-Instruct-2507** (default), Qwen-2.5-7B and Llama-3.1-8B as alternatives
- [x] Select protein encoder: **ESM-3 small** (default)

### TODO: Model Testing
- [ ] Run `scripts/test_model_loading.py` on compute node with GPU
- [ ] Verify Qwen/Qwen3-1.5B loads and runs inference
- [x] ESM-3 encoder loading verified
- [ ] Test combined encoder + LLM memory usage

### TODO: Implementation
- [ ] Implement attention pooling (`src/models/pooling.py`)
- [ ] Implement MLP projector (`src/models/projector.py`)
- [ ] Implement multimodal LLM wrapper (`src/models/multimodal_llm.py`)
- [ ] Implement SFT trainer (`src/training/sft_trainer.py`)
- [ ] Implement GRPO trainer (`src/training/grpo_trainer.py`)
- [ ] Create instruction dataset loader (`src/data/instruction_dataset.py`)
- [ ] Implement evaluation benchmarks (`src/evaluation/`)

### TODO: Data
- [x] Download Mol-Instructions dataset
- [x] Download IPD PDB Sample dataset
- [x] Download Swiss-Prot sequences
- [x] Create dataset exploration notebook (`scripts/explore_datasets.ipynb`)
- [ ] Preprocess protein-text pairs
- [ ] Create train/val/test splits

### TODO: Training
- [ ] Run baseline SFT experiment
- [ ] Run GRPO alignment
- [ ] Evaluate on GO prediction, PPI, stability

---

## Project Structure (Updated 2026-02-16)

```
Post_Training_Protein_LLM/
├── CLAUDE.md                     # Concise project overview (~50 lines)
├── pyproject.toml                # Package configuration
├── setup_env.sh                  # Conda environment setup
│
├── .claude/                      # Claude Code integration
│   ├── settings.json             # Permissions, hooks
│   ├── agents/                   # engineer, qa, researcher
│   ├── commands/                 # /train, /eval, /data-prep, /debug
│   └── skills/                   # protein-encoding, rl-training, hydra-configs
│
├── configs/                      # Hydra configuration
│   ├── config.yaml               # Main config
│   ├── model/                    # qwen2_7b, llama3_8b
│   ├── encoder/                  # esm3_small
│   ├── data/                     # mol_instructions, ipd_pdb
│   ├── training/                 # sft_qlora, grpo, dpo
│   ├── evaluation/               # go_prediction, ppi, stability
│   └── experiment/               # baseline_sft, full_pipeline
│
├── scripts/                      # Entry points
│   ├── train.py                  # python scripts/train.py experiment=...
│   ├── evaluate.py
│   ├── prepare_data.py
│   └── inference.py
│
├── src/                          # Core implementation
│   ├── models/                   # protein_encoder.py, pooling.py, projector.py
│   ├── data/                     # pdb_dataset.py, instruction_dataset.py
│   ├── training/                 # sft_trainer.py, grpo_trainer.py
│   ├── evaluation/               # go_prediction.py, ppi_prediction.py
│   └── utils/                    # logging.py, checkpoint.py, distributed.py
│
├── tests/                        # pytest test suite
│   ├── models/
│   ├── data/
│   └── training/
│
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── training_guide.md
│   ├── troubleshooting.md
│   └── research/                 # This file + research reports
│
├── data/                         # Data directory (gitignored)
│   ├── raw/
│   ├── processed/
│   └── checkpoints/
│
└── notebooks/                    # Jupyter notebooks
```

---

## Environment Info

- **Hardware**: 8x NVIDIA H100 80GB HBM3
- **CUDA Version**: 13.0
- **Driver**: 580.105.08
- **Conda**: miniforge/25.11.0-0 (via module)

---

## Next Steps

### Immediate (This Week)
1. ~~**[ACTION]** Decide on base LLM~~ **RESOLVED**: Qwen3-4B-Instruct-2507
2. **[ACTION]** Complete ESM-3 encoder GPU verification
3. **[ACTION]** Fix GRPO gradient flow bug
4. **[ACTION]** Fix evaluate.py argument parsing bugs
5. **[ACTION]** Run first ESM-3 + Qwen3-4B SFT experiment

### Short-term (Next 2 Weeks)
6. **[ACTION]** Preprocess downloaded datasets into training format
7. **[ACTION]** Run text-approach baseline for comparison
8. **[ACTION]** Evaluate on GO prediction benchmark
9. **[ACTION]** Run GRPO alignment after SFT

### Medium-term
10. **[ACTION]** Run full SFT -> GRPO pipeline
11. **[ACTION]** Compare approaches: text vs ESM-3 (MLP vs Perceiver)
12. **[ACTION]** Compare against baselines (ESM2-650M, ProtT5)

---

## Projection-Only Vision-Language Multimodal LLM Survey (2026-02-20)

### Overview

This survey covers state-of-the-art multimodal LLM architectures from 2024-2026 that integrate non-text modalities (primarily vision) into LLMs using projection-based connectors WITHOUT adding cross-attention layers to the LLM backbone. These are "projection-only" or "LLM-as-decoder" architectures where visual tokens are simply concatenated with text tokens in the input sequence.

The universal pattern is: **Encoder (frozen) -> Token Reduction -> Projector (MLP) -> LLM (LoRA or full fine-tune)**

This is directly analogous to our protein pipeline: **ESM-3 (frozen) -> Attention Pooling -> MLP Projector -> Qwen3-4B (LoRA k/v)**

---

### 1. InternVL2.5 & InternVL3 (Dec 2024 / Apr 2025)

**Paper**: "Expanding Performance Boundaries of Open-Source Multimodal Models" (arXiv 2412.05271)

**Architecture**: InternViT-6B + Pixel Unshuffle + 2-Layer MLP + LLM (InternLM2.5 / Qwen2.5 series)

**Projector**: Randomly initialized 2-layer MLP projector. Maps from InternViT hidden dim to LLM hidden dim.

**Token Reduction -- Pixel Unshuffle**:
- Each 448x448 image tile produces 1024 visual tokens from the ViT
- A pixel unshuffle (space-to-depth) operation with factor 2 rearranges each 2x2 spatial block into 4 channel dimensions
- This reduces 1024 tokens to 256 tokens per tile (4x reduction)
- The channel dimension quadruples correspondingly, preserving information
- Mathematically: feature map [H, W, C] -> [H/2, W/2, 4C], then the MLP reduces 4C back to LLM dim

**Dynamic Resolution Tiling**:
- Images are divided into tiles of 448x448 pixels
- A configurable `n_max` parameter controls max tiles per image (6-12 for standard images, 24-36 for high-res/multi-image, 1 for video frames)
- When more than 1 tile is used, a thumbnail of the full image is also included
- Total visual tokens = (n_tiles + 1) x 256 for multi-tile, or 256 for single-tile

**Training Stages**:

| Stage | Trainable | Frozen | Learning Rate | Data |
|-------|-----------|--------|---------------|------|
| Stage 1 (MLP Warmup) | MLP projector only | ViT + LLM | 2e-4 | Image-text pairs |
| Stage 1.5 (ViT Warmup) | ViT + MLP | LLM | 1e-5 | Image-text pairs |
| Stage 2 (Full Fine-tune) | All parameters | None | 2e-5 to 4e-5 | Instruction data (16.3M for v2.5, 21.7M for v3) |

**InternViT-6B Specifications**: 5.5B parameters, 45 layers, 3200 hidden size, 25 attention heads, with QK-Norm and RMSNorm.

**InternVL3 Additions**:
- **V2PE (Variable Visual Position Encoding)**: Uses smaller, more flexible position increments for visual tokens. Instead of each visual token consuming one position ID, V2PE assigns fractional position increments, enabling much longer multimodal contexts without exhausting the position embedding range.
- **Native Multimodal Pre-Training**: Interleaves image-text, video-text data with pure text corpora during LLM pre-training (single stage instead of sequential adaptation).
- **Mixed Preference Optimization (MPO)**: Adds preference supervision from positive/negative sample pairs during fine-tuning.

**Key Insight**: The pixel unshuffle operation is simple, parameter-free, and preserves all spatial information by trading spatial resolution for channel depth. The MLP then learns to compress the expanded channels.

**Performance**: InternVL3-8B achieves 62.7% on MMMU, 92.7% on DocVQA. InternVL3-78B reaches 72.2% MMMU.

**Sources**: [InternVL2.5 Paper](https://arxiv.org/html/2412.05271v1), [InternVL3 Blog](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/), [InternVL Docs](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html)

---

### 2. LLaVA-NeXT / LLaVA-v1.6 / LLaVA-OneVision (2024)

**Paper**: "LLaVA-OneVision: Easy Visual Task Transfer" (arXiv 2408.03326)

**Architecture**: SigLIP (384x384) + 2-Layer MLP (GELU) + LLM (Qwen-2 series)

**Projector**: 2-layer MLP with GELU activation ("mlp2x_gelu" configuration). First linear layer maps vision encoder hidden dim to LLM hidden dim; second linear layer maintains LLM hidden dim. This design originated in LLaVA-1.5 and became the de facto standard.

**AnyRes Dynamic Resolution**:
- Images are divided into a x b crops (tiles), where (a,b) is chosen from a predefined set of spatial configurations to match the image's aspect ratio
- Each crop is processed by SigLIP at 384x384 resolution, producing 729 tokens per crop (27x27 grid)
- A full-image thumbnail is always included
- Total tokens: L = (a*b + 1) * 729
- When L exceeds a threshold, bilinear interpolation pools tokens down to the budget
- "Higher-AnyRes" in OneVision dynamically adapts the crop strategy across single-image, multi-image, and video scenarios

**Visual Token Counts**:
- Base resolution (384x384): 729 tokens
- Stage 1: 729 tokens (no tiling)
- Stage 2 (single-image): up to 729 x 5 = 3,645 tokens
- Stage 2 (multi-image/video): up to 729 x 10 = 7,290 tokens

**Training Stages**:

| Stage | Trainable | Frozen | LR (ViT) | LR (Projector) |
|-------|-----------|--------|-----------|----------------|
| Stage 1 (Alignment) | Projector only | LLM + ViT | - | 1e-3 |
| Stage 1.5 (High-Quality Knowledge) | ViT + Projector | LLM | 2e-6 | 1e-5 |
| Stage 2 (Visual Instruction Tuning) | All | None | 2e-6 | 1e-5 |

**Data Scale**: 3.2M single-image + 0.56M multi-image + 0.35M video samples for instruction tuning.

**Key Insight**: The 2-layer MLP is sufficient -- more complex connectors (Q-Former, Perceiver) do not consistently outperform it. The critical factor is training data quality and the multi-stage recipe. LLaVA proved that simplicity wins.

**Performance**: LLaVA-OneVision-72B achieves 91.3% DocVQA, 85.9% MMBench, 66.2% VideoMME.

**Sources**: [LLaVA-OneVision Paper](https://arxiv.org/html/2408.03326v1), [LLaVA-OneVision Blog](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/), [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)

---

### 3. Qwen2-VL / Qwen2.5-VL (Sep 2024 / Feb 2025)

**Paper**: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution" (arXiv 2409.12191)

**Architecture**: Custom ViT (675M, trained from scratch) + 2D-RoPE + MLP Merger (2x2 compression) + LLM (Qwen2/2.5 series)

**Projector / Vision-Language Merger**:
- After the ViT encoder, adjacent 2x2 tokens are grouped
- A 2-layer MLP compresses each group of 4 tokens into 1 token (4x compression)
- Output dimension: 3584 (matching Qwen2.5 LLM hidden dim)
- Special `<vision_start>` and `<vision_end>` delimiter tokens bracket the visual sequence

**Naive Dynamic Resolution**:
- Unlike tiling approaches, Qwen2-VL processes images at their native resolution
- Images are encoded with patch_size=14, then the merger compresses 2x2 patches into single tokens
- A 224x224 image yields ~66 tokens after compression
- Resolution bounds controlled by min_pixels and max_pixels parameters
- Average ~1,924 tokens per image during inference

**2D-RoPE (2D Rotary Position Embedding)**:
- Replaces standard absolute position embeddings in the ViT
- Encodes spatial positions as 2D coordinates (row, column) rather than flattened 1D sequence positions
- Enables the ViT to handle variable resolutions without retraining or interpolation
- Each patch receives position encoding based on its (x, y) coordinate in the image grid

**Window Attention** (Qwen2.5-VL):
- The ViT uses window attention operating over local spatial windows
- Reduces computational complexity from quadratic to linear with respect to patch count
- Enables processing of very high-resolution images efficiently

**Training Stages** (Qwen2-VL):

| Stage | Trainable | Data | Tokens |
|-------|-----------|------|--------|
| Stage 1 (ViT Pre-training) | ViT only | Image-text pairs | ~600B |
| Stage 2 (Joint Pre-training) | All unfrozen | Diverse multimodal | ~800B |
| Stage 3 (Instruction Fine-tuning) | LLM (ViT frozen) | Instruction data | - |

Total pre-training: 1.4 trillion tokens. Qwen2.5-VL further extends with 1.5T tokens for ViT-only pretraining and 4.1T tokens total.

**Key Insight**: "Naive" dynamic resolution (process at native resolution, then compress) outperforms forced tiling because it preserves the natural aspect ratio and spatial relationships. The 2D-RoPE elegantly handles variable resolutions in the ViT itself.

**Performance**: Qwen2-VL-72B achieves 96.5% DocVQA (previous SOTA 94.1%), 77.8% RealWorldQA, 89.6% UI operation accuracy.

**Sources**: [Qwen2-VL Paper](https://arxiv.org/html/2409.12191v1), [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923), [Qwen2.5-VL HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

---

### 4. Cambrian-1 (Jun 2024, NeurIPS 2024 Oral)

**Paper**: "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs" (arXiv 2406.16860)

**Architecture**: Multi-Encoder (SigLIP + CLIP + DINOv2 + ConvNeXt) + Spatial Vision Aggregator (SVA) + LLM

**Spatial Vision Aggregator (SVA)**:
- A learned cross-attention connector that replaces the simple MLP projector
- Uses a set of learnable latent queries (default: 576 queries) that interact with all vision encoder feature maps via cross-attention
- **Spatial inductive bias**: Each query is explicitly associated with a specific sub-region of the feature maps. Queries are spatially indexed so that nearby queries attend to nearby spatial regions across all encoders.
- **Multi-layer vision aggregation**: Cross-attention occurs multiple times throughout the LLM layers (not just at input), providing consistent access to uncompressed visual information
- Hyperparameters: D (number of cross-attention layers), G (number of distinct groups of learnable queries)
- All vision encoder outputs are resized to a common spatial resolution before aggregation

**Vision Encoders Used**:
- SigLIP (CLIP-ViT-SO400M-14-384): Semantic understanding
- OpenAI CLIP (ViT-L/14-336): General visual-language alignment
- DINOv2-Giant (378px): Self-supervised spatial features
- CLIP-ConvNeXt-XXL (multi-stage): Hierarchical features

**Token Efficiency**: 576 image tokens vs 2,880 for LLaVA-NeXT (5x reduction) at comparable performance.

**Training Stages**:

| Stage | Trainable | Frozen | Data |
|-------|-----------|--------|------|
| 1. Visual Connector Training | SVA connector | All vision encoders + LLM | 2.5M Cambrian Alignment Data |
| 2. Instruction Tuning | SVA + LLM | All vision encoders | 7M Cambrian Instruction Data |

**Key Insight**: Different vision encoders capture complementary information (CLIP for semantics, DINOv2 for spatial/geometric features, ConvNeXt for hierarchical features). Combining them via cross-attention learnable queries outperforms any single encoder. The paper systematically evaluated 20+ vision encoders.

**Performance**: State-of-the-art with only 576 tokens. Achieves competitive results with LLaVA-NeXT and Mini-Gemini while using 5x fewer visual tokens.

**Sources**: [Cambrian-1 Paper](https://arxiv.org/abs/2406.16860), [Cambrian GitHub](https://github.com/cambrian-mllm/cambrian), [NeurIPS 2024](https://neurips.cc/virtual/2024/oral/97972)

---

### 5. Molmo (Sep 2024)

**Paper**: "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models" (arXiv 2409.17146)

**Architecture**: CLIP ViT-L/14 (336px) + Multi-Crop + Attention Pooling + MLP (SwiGLU) + LLM (OLMo/Qwen2)

**Projector/Connector**:
- Multi-headed attention pooling on 2x2 patch windows (mean of patches as query, 16 attention heads)
- Followed by an MLP with SwiGLU activation
- MLP intermediate dimension matches the LLM's intermediate dimension
- Connector parameters: 12M (1B model) to 310M (72B model)

**Multi-Crop Strategy**:
- Pre-processor converts input image into multiple overlapping crops at different scales
- Training: 12 high-resolution crops + 1 low-resolution full image
- Testing: 36 high-resolution crops + 1 low-resolution full image
- Crops overlap by 4 patches (56 pixels) for context continuity
- Learned crop position embeddings (3 types: no padding, some padding, all padding)

**Visual Token Counts**:
- Each crop: 144 patches (12x12 grid from ViT-L/14)
- After 2x2 attention pooling: 36 tokens per crop
- Training total: ~1,849 tokens (12 crops x ~144 + 1 low-res)
- Testing total: ~4,453 tokens (36 crops)

**Training** (simplified 2-stage):

| Stage | Trainable | Data | Key LR |
|-------|-----------|------|--------|
| Pre-training | All parameters | PixMo-Cap (proprietary captions) | ViT: 6e-6, Connector: 2e-4 |
| Fine-tuning | All parameters | Mixed instruction data | Similar differential LR |

**Key Insight**: Data quality matters more than architectural novelty. Molmo's architecture is relatively simple (attention pooling + MLP), but its PixMo dataset of highly detailed, speech-transcript-based image captions drives its performance. They show that careful data curation can substitute for architectural complexity.

**Performance**: Molmo-72B ranks second in human evaluation (Elo 1077 vs GPT-4o's 1079). Molmo-7B-D scores 93.2% AI2D, 85.6% VQAv2.

**Sources**: [Molmo Paper](https://arxiv.org/html/2409.17146v2), [Molmo Blog](https://allenai.org/blog/molmo), [Molmo GitHub](https://github.com/allenai/molmo)

---

### 6. DeepSeek-VL2 (Dec 2024)

**Paper**: "DeepSeek-VL2: Mixture-of-Experts Vision-Language Models" (arXiv 2412.10302)

**Architecture**: SigLIP-SO400M (384px) + Dynamic Tiling + Pixel Shuffle + 2-Layer MLP + DeepSeekMoE LLM (with MLA)

**Projector**: 2-layer MLP that projects compressed visual tokens into LLM embedding space.

**Token Reduction -- Pixel Shuffle**:
- SigLIP produces 27x27 = 729 embeddings per 384x384 tile (at 1152 dimensions)
- A 2x2 pixel shuffle compresses 729 tokens to 14x14 = 196 tokens per tile
- Channel dimension expands correspondingly (1152 -> 4x1152 then MLP reduces)
- Separator tokens delineate spatial structure and global-local context

**Dynamic Tiling**:
- Candidate resolutions: {(m*384, n*384) | m,n in N, 1<=m,n, m*n<=9}
- Select resolution minimizing padding area
- Divide into m x n tiles of 384x384 plus one global thumbnail
- For extreme aspect ratios (e.g., infographics): constraint expands to m*n<=18

**Total Visual Tokens**: Per image: 210 + 1 + m_i * 14 * (n_i * 14 + 1) tokens (including thumbnail, separators, newlines).

**Training Stages**:

| Stage | Trainable | Frozen | Data |
|-------|-----------|--------|------|
| 1. Alignment | Vision encoder + MLP | LLM | Image-text pairs |
| 2. Pre-training | All | None | ~800B image-text tokens |
| 3. SFT | All | None | Instruction data (loss on answers only) |

**MoE Integration**: The LLM uses Mixture-of-Experts with Multi-head Latent Attention (MLA), which compresses Key-Value cache into latent vectors for efficient inference. Top-6 experts selected per token. Model variants: Tiny (1.0B activated), Small (2.8B activated), Full (4.5B activated).

**Key Insight**: Combining MoE with multimodal processing allows large effective capacity while keeping inference cost low. The pixel shuffle + MLP pipeline is nearly identical to InternVL but uses a different ViT (SigLIP vs InternViT).

**Sources**: [DeepSeek-VL2 Paper](https://arxiv.org/html/2412.10302v1), [DeepSeek-VL2 GitHub](https://github.com/deepseek-ai/DeepSeek-VL2)

---

### 7. Phi-3-Vision / Phi-3.5-Vision (Jun/Aug 2024)

**Architecture**: CLIP ViT-L/14 + num_crops tiling + MLP Projector + Phi-3 Mini (3.8B)

**Projector**: Multi-Layer Perceptron that transforms visual patch embeddings into the text feature space. Follows the LLaVA-style 2-layer MLP pattern.

**Image Processing**: Uses a `num_crops` parameter (4 for multi-frame, 16 for single-frame) to control how images are cropped and processed. Each crop is processed independently by the ViT, then projected.

**Training**: End-to-end on 500B tokens (visual + textual) using 256 A100-80G GPUs over 6 days.

**Key Insight**: A small (4.2B) model with efficient architecture can achieve strong multimodal performance. Emphasizes on-device deployment capability with 128K context length.

**Sources**: [Phi-3.5-Vision HuggingFace](https://huggingface.co/microsoft/Phi-3.5-vision-instruct), [Microsoft Blog](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/phi-3-vision-%E2%80%93-catalyzing-multimodal-innovation/4170251)

---

### 8. MM1 / MM1.5 (Mar/Sep 2024)

**Paper**: "MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning" (arXiv 2409.20566)

**Architecture**: CLIP image encoder + dynamic image splitting + connector + LLM (1B-30B, dense and MoE)

**Key Finding on Connectors**: The image encoder, image resolution, and token count have **substantial impact** on performance, while the vision-language connector design is of **comparatively negligible importance**. This is one of the most important ablation results in the multimodal LLM literature -- it shows the projector is NOT the bottleneck.

**Image Processing**: High-resolution images up to 4 Megapixels, with dynamic splitting for various aspect ratios.

**Training**: Data-centric approach with emphasis on data mixture optimization across continual pre-training (high-quality OCR data, synthetic captions) and supervised fine-tuning.

**Key Insight**: Data quality and training recipe matter far more than connector architecture. This validates using a simple 2-layer MLP projector and investing effort in data quality.

**Sources**: [MM1 Paper](https://arxiv.org/abs/2403.09611), [MM1.5 Paper](https://arxiv.org/abs/2409.20566)

---

### 9. Vision-Language Architecture Comparison Table

| Model | Encoder | Projector | Token Reduction | Tokens/Image | Training Stages | Key Innovation |
|-------|---------|-----------|-----------------|--------------|-----------------|----------------|
| **InternVL2.5/3** | InternViT-6B (5.5B) | 2-layer MLP | Pixel unshuffle 2x2 (4x) | 256/tile | 3 (MLP -> ViT+MLP -> All) | Large ViT + pixel unshuffle + V2PE |
| **LLaVA-OneVision** | SigLIP (384px) | 2-layer MLP (GELU) | AnyRes tiling + interpolation | 729-7290 | 3 (Proj -> ViT+Proj -> All) | AnyRes dynamic cropping, simplicity |
| **Qwen2-VL/2.5-VL** | Custom ViT (675M) | 2-layer MLP merger | 2x2 patch grouping (4x) | ~66-1924 | 3 (ViT -> All -> SFT) | 2D-RoPE, naive dynamic resolution |
| **Cambrian-1** | 4 encoders (SigLIP+CLIP+DINOv2+ConvNeXt) | SVA (cross-attention) | Learnable queries | 576 | 2 (SVA -> SVA+LLM) | Multi-encoder + spatial aggregation |
| **Molmo** | CLIP ViT-L/14 | Attn pooling + MLP (SwiGLU) | 2x2 attn pooling (4x) | ~1849-4453 | 2 (All -> All) | Overlapping multi-crop, data quality |
| **DeepSeek-VL2** | SigLIP-SO400M | 2-layer MLP | Pixel shuffle 2x2 (4x) | 196/tile | 3 (Enc+MLP -> All -> SFT) | MoE LLM + pixel shuffle |
| **Phi-3.5-Vision** | CLIP ViT-L/14 | MLP projector | num_crops tiling | Variable | End-to-end (500B tokens) | Small model, on-device |
| **MM1.5** | CLIP | Connector (various) | Dynamic splitting | Variable | Pre-train + SFT | Data-centric, connector doesn't matter |
| **Our Pipeline** | ESM-3 (1.4B, frozen) | 2-layer MLP (1536->2048->2560) | Attention pooling (L->32) | 32 | 2 (Proj -> Proj+LoRA) | Protein-specific, aggressive pooling |

---

### 10. Cross-Domain Implications for Our Protein LLM Pipeline

#### What We Are Already Doing Right
1. **2-layer MLP projector**: Matches the field consensus. MM1.5 confirmed connector design is not the bottleneck.
2. **Frozen encoder**: Every top system freezes the vision encoder (at least in later stages). We freeze ESM-3 always.
3. **Multi-stage training**: Our alignment (projector-only) then instruction tuning (projector + LoRA) matches the dominant pattern.
4. **Differential learning rate**: Our projector_lr=2e-3 vs base lr=2e-4 aligns with LLaVA and Molmo approaches.

#### Potential Improvements to Consider
1. **Token count**: We use 32 tokens (very aggressive). Most vision models use 256-576 tokens. For proteins, this may lose per-residue detail needed for tasks like contact prediction or active site identification. Consider experimenting with 64 or 128 pooled tokens.
2. **Pooling mechanism**: Molmo's 2x2 attention pooling with SwiGLU is worth comparing against our global attention pooling. Their approach preserves local spatial context better.
3. **Multi-encoder fusion**: Following Cambrian-1, we could combine ESM-3 with a complementary encoder (e.g., ProtTrans for different pretraining bias, or a GNN for structural topology) using cross-attention aggregation.
4. **Position encoding for protein tokens**: InternVL3's V2PE (fractional position increments for visual tokens) could be adapted. Currently our 32 protein tokens consume 32 position IDs. Using fractional increments could improve how the LLM distinguishes protein context from text.
5. **Residue shuffle analog for proteins**: Instead of attention pooling, we could try a "residue shuffle" -- grouping adjacent residues (e.g., windows of 4) and merging their embeddings via concatenation + MLP. This would be more parameter-efficient and preserve local sequence context, analogous to pixel shuffle.

---

## How to Update This Log

1. **Add new entries** under [Development Log](#development-log) with date
2. **Record results** in [Results & Metrics](#results--metrics) table
3. **Document decisions** in [Decision Log](#decision-log)
4. **Update action items** in [Action Items](#action-items)

**Quick commands**:
```bash
# Open this file
code docs/research/agents_research_log.md

# Run training
python scripts/train.py experiment=baseline_sft

# Run evaluation
python scripts/evaluate.py evaluation=go_prediction
```
