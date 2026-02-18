# LLM Post-Training Methods: Comprehensive Summary for Multimodal Protein-LLM Projects

**Research compiled by Agent-2 | February 2025**

---

## Table of Contents
1. [Training Frameworks Comparison](#1-training-frameworks-comparison)
2. [SFT Methods](#2-sft-methods)
3. [RL Methods](#3-rl-methods)
4. [Recent Advances (2024-2025)](#4-recent-advances-2024-2025)
5. [Recommendations for Protein-LLM Project](#5-recommendations-for-protein-llm-project)

---

## 1. Training Frameworks Comparison

### Overview Table

| Framework | Lines of Code | Key Strength | Backend | Multimodal Support | Ease of Use |
|-----------|--------------|--------------|---------|-------------------|-------------|
| **TRL** | ~19K | HuggingFace integration | HF Trainer | Via HF ecosystem | ⭐⭐⭐⭐⭐ |
| **veRL** | ~32K | Performance & scalability | FSDP/DeepSpeed/Megatron | Limited | ⭐⭐⭐ |
| **OpenRLHF** | ~8.5K | Easy-to-use, high-performance | Ray + vLLM + DeepSpeed | OpenRLHF-M extension | ⭐⭐⭐⭐ |
| **DeepSpeed-Chat** | - | Accessibility | DeepSpeed | Limited | ⭐⭐⭐ |

---

### TRL (Transformer Reinforcement Learning - Hugging Face)

**Overview**: The most popular library from Hugging Face, tightly integrated with the HF ecosystem.

**Pros**:
- Seamless integration with Hugging Face transformers, PEFT, and datasets
- Excellent documentation and community support
- SFTTrainer, DPOTrainer, PPOTrainer all available
- Works with Unsloth for 2x faster training
- Supports LoRA/QLoRA out of the box

**Cons**:
- Performance is 3.1x slower than OpenRLHF for PPO training
- Lacks sophisticated orchestration for large-scale distributed training
- Memory overhead can be higher than specialized frameworks

**GPU Requirements**:
- 7B model SFT: Single 24GB GPU (with QLoRA)
- 70B model: Multiple A100 80GB GPUs

**Scalability**: Moderate - best for single-node or small cluster setups

**Best For**: Beginners, rapid prototyping, HuggingFace ecosystem users

---

### veRL (Volcano Engine RL)

**Overview**: High-performance RL stack from ByteDance, optimized for scalability.

**Pros**:
- Supports FSDP, DeepSpeed, and Megatron backends
- Actor and Rollout modules share weights in memory (no CUDA IPC overhead)
- Scales up to 671B models and hundreds of GPUs
- Supports PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO
- DAPO (achieving 50 points on AIME 2024) was trained using veRL

**Cons**:
- Largest codebase (~32K lines) - steeper learning curve
- More complex setup than TRL
- Less documentation compared to TRL

**GPU Requirements**:
- Optimized for multi-GPU setups
- 80-90% of training time is on sample generation (inference bottleneck)

**Scalability**: Excellent - designed for large-scale distributed training

**Best For**: Large-scale training, research teams with significant compute

---

### OpenRLHF

**Overview**: User-friendly, scalable RLHF framework built on Ray, vLLM, DeepSpeed, and HuggingFace.

**Pros**:
- **Most concise codebase** (~8.5K lines)
- 3-4x faster than DeepSpeed-Chat with Hybrid Engine
- 1.22x to 1.68x faster than other state-of-the-art frameworks
- Separates Actor, Reward, Reference, Critic models across GPUs
- Full RLHF fine-tuning for 70B+ models
- **OpenRLHF-M**: Dedicated multimodal extension for VLMs

**Cons**:
- Requires understanding of Ray for distributed training
- Smaller ecosystem than TRL

**GPU Requirements**:
- 7B models: Multiple RTX 4090 24GB GPUs
- 70B+ models: Multiple A100 80GB GPUs

**Scalability**: Excellent - adopted by Google, ByteDance, Baidu, NVIDIA, Tencent

**Best For**: Production RLHF, multimodal models, teams needing both ease-of-use and performance

---

### DeepSpeed-Chat

**Overview**: Microsoft's accessible RLHF implementation.

**Pros**:
- Good accessibility and documentation
- Integrated with DeepSpeed ZeRO optimization
- Supports model parallelism

**Cons**:
- 3-4x slower than OpenRLHF
- Lacks sophisticated orchestration capabilities
- Struggles with inference optimization

**GPU Requirements**: Similar to other frameworks

**Scalability**: Moderate

**Best For**: Teams already using DeepSpeed ecosystem

---

## 2. SFT Methods

### Method Selection Guide

| Method | VRAM Usage | Training Speed | Quality vs Full FT | Recommended GPU |
|--------|-----------|----------------|-------------------|-----------------|
| **Full Fine-Tuning** | 100-120GB (7B) | Baseline | 100% | 4-8x A100 80GB |
| **LoRA (16-bit)** | ~28GB (7B) | Faster | 90-95% | A100 40GB |
| **QLoRA (4-bit)** | ~8-10GB (7B) | Slightly slower | 80-90% | RTX 4090 24GB |

---

### QLoRA (Recommended Starting Point)

**Why Start Here**: "A common mistake is jumping straight into full fine-tuning (FFT). Start by testing with LoRA or QLoRA first - if it won't work there, it almost certainly won't work with FFT."

**Memory Requirements by Model Size**:

| Model Size | QLoRA VRAM | LoRA VRAM | Full FT VRAM |
|------------|-----------|-----------|--------------|
| 7B | 8-10GB | 20-28GB | 100-120GB |
| 13B | ~15GB | 35-40GB | 200+GB |
| 70B | ~46GB | 140+GB | 500+GB |

**Key Hyperparameters**:
- **Learning Rate**: 2e-4 for SFT, 5e-6 for RL (DPO, GRPO)
- **Epochs**: 1-3 (more risks overfitting)
- **LoRA Rank**: Minimum r=4 recommended for PLMs; r=8 or r=16 typical
- **Target Modules**: Apply to both attention AND MLP layers (QLoRA-All performs best)

---

### Protein Language Model Specifics

**Key Finding**: Fine-tuning protein language models (ESM2, ProtT5, Ankh) with LoRA achieves competitive performance with significantly fewer resources.

**LoRA for Protein Models**:
- ProtT5 (1.2B params): LoRA reduces trainable parameters to ~3.5M
- Fine-tunable on GPU with ~10GB memory
- **Contrary to NLP**: For PLMs, applying LoRA to only key and value matrices achieves optimal performance
- **Minimum rank**: r=4 recommended (performance drops below this)

**Performance**:
- SETH-LoRA improved performance by 2.2 percentage points (Spearman 0.72 → 0.736)
- Fine-tuned ESM2-150M outperforms larger ESM2-650M/3B without fine-tuning
- LoRA fine-tuning is ~12.5% faster than full fine-tuning

---

### Best Practices

1. **Data Quality**: Ensure dataset is clean, representative; 50-100K examples for multi-task learning
2. **Train on Completions Only**: Masking inputs and training only on outputs increases accuracy by ~1%
3. **Use SFTTrainer**: HuggingFace's SFTTrainer handles logging, evaluation, checkpointing automatically
4. **Start Small**: Begin with smaller models/datasets to validate approach

---

## 3. RL Methods

### Comparison Table

| Method | Complexity | Compute | Memory | Best Use Case |
|--------|-----------|---------|--------|---------------|
| **PPO (RLHF)** | High | High | High (needs critic) | Maximum alignment quality |
| **DPO** | Low | Low | Low | Simple preference learning |
| **GRPO** | Medium | Medium | ~50% less than PPO | Reasoning tasks |
| **REINFORCE++** | Low | Low | Low | Stable, efficient training |

---

### PPO (Proximal Policy Optimization)

**Overview**: The classic RL algorithm for RLHF, used in ChatGPT and Claude.

**Pros**:
- State-of-the-art results on challenging tasks (code competitions)
- Well-understood, extensively studied
- Stable training with clipping mechanism

**Cons**:
- Requires separate reward model
- Requires critic model (doubles memory)
- Notoriously tricky to tune
- Computationally expensive

**When to Use**: When you have resources for reward model training and need maximum performance

---

### DPO (Direct Preference Optimization)

**Overview**: Simplifies RLHF by directly optimizing from preference data without explicit reward modeling.

**Pros**:
- No reward model needed
- Simple classification-style objective
- Lower compute requirements
- Easier to implement and tune

**Cons**:
- Can find biased solutions exploiting out-of-distribution responses
- Performance affected by distribution shift
- May underperform PPO on complex tasks

**When to Use**: Teams wanting simpler architectures, limited compute, straightforward preference data

---

### GRPO (Group Relative Policy Optimization)

**Overview**: Introduced by DeepSeek, eliminates the critic model.

**Pros**:
- **~50% less memory** than PPO (no separate critic)
- Learns from multiple ranked completions per prompt
- More scalable than pairwise methods
- KL regularization for stable updates
- Better generalization across prompts

**Cons**:
- Newer, less battle-tested than PPO
- May require more careful hyperparameter tuning

**When to Use**: Reasoning tasks (math, coding), resource-constrained RL training

---

### REINFORCE++ (New in 2024-2025)

**Overview**: A simplified and efficient approach proposed in December 2024.

**Pros**:
- Matches or exceeds PPO-based RLHF results
- Simpler than PPO
- Saves compute and memory
- More stable than GRPO in training
- Faster than PPO

**Cons**:
- Very new, limited production validation
- Less documentation

**When to Use**: When you want PPO-level results with GRPO-level efficiency

---

### DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

**Overview**: State-of-the-art RL algorithm (March 2025), achieved 50 points on AIME 2024.

**Key Innovations**:
1. **Clip-Higher**: Promotes diversity, avoids entropy collapse
2. **Dynamic Sampling**: Improves efficiency and stability
3. **Token-Level Policy Gradient**: Critical for long chain-of-thought
4. **Overlong Reward Shaping**: Reduces noise, stabilizes training

**Best For**: Cutting-edge reasoning capabilities, research applications

---

## 4. Recent Advances (2024-2025)

### Unsloth: 2x Faster Training

**Key Features**:
- 2x faster, 70% less VRAM compared to standard training
- Works by rewriting Pytorch modules into Triton kernels
- **Zero accuracy degradation** (no approximations)
- Supports GTX 1070 through H100
- Compatible with TRL (SFTTrainer, DPOTrainer, PPOTrainer)

**Performance Benchmarks**:
- Llama 3.1 8B: 2.1x faster, 60% less memory vs Flash Attention 2
- Llama 3.1 70B: 1.9x faster, 65% less VRAM
- MoE models: 12x faster, 35% less VRAM

**New in 2025**:
- FP8 GRPO reinforcement learning on consumer GPUs
- 3x faster training with Padding Free + Packing
- 500K+ context training on 80GB GPU

---

### Quantization Advances

| Method | Description | Year |
|--------|-------------|------|
| **SpinQuant** | LLM quantization with learned rotations | ICLR 2025 |
| **AutoMixQ** | Self-adjusting mixed precision | 2024 |
| **RILQ** | Rank-insensitive LoRA quantization | 2024 |
| **EfficientQAT** | Efficient quantization-aware training | ACL 2025 |

---

### Efficient PEFT Methods

| Method | Innovation | Conference |
|--------|-----------|------------|
| **LoRMA** | Low-Rank Multiplicative Adaptation | ACL Findings 2025 |
| **HydraLoRA** | Asymmetric LoRA Architecture | NeurIPS 2024 |
| **MEFT** | Memory-Efficient via Sparse Adapters | ACL 2024 |

---

### Industry Trends

1. **Post-training > Pre-training**: Post-training can extract 2-5x more value from existing architectures
2. **Hybrid approaches**: DPO with synthetic preferences, GRPO after SFT
3. **Reasoning focus**: RLVR (RL with Verifiable Rewards) for math/coding
4. **Efficiency**: DeepSeek R1 showed reasoning can emerge from RL alone

---

## 5. Recommendations for Protein-LLM Project

### Given: Limited Compute, Multimodal Requirements

---

### Framework Recommendation: **OpenRLHF** or **TRL + Unsloth**

**Option A: TRL + Unsloth (Best for Starting)**
- Easiest to set up and iterate
- 2x faster, 70% less VRAM
- Perfect HuggingFace integration (ESM, ProtTrans work seamlessly)
- Use SFTTrainer → DPOTrainer pipeline

**Option B: OpenRLHF-M (Best for Scaling)**
- If you need multimodal RLHF later
- More complex but more powerful
- 3x faster than alternatives for PPO

---

### Training Strategy

#### Phase 1: SFT with QLoRA
```
Method: QLoRA (4-bit)
Framework: TRL + Unsloth
Target: All attention + MLP layers
Rank: r=8 (minimum r=4 for protein models)
Learning Rate: 2e-4
Epochs: 1-3
```

**VRAM Requirements**:
| Model Size | GPU Needed |
|------------|-----------|
| 150M-650M (ESM2) | Single RTX 3090/4090 |
| 1.2B (ProtT5) | Single RTX 4090 or A100 |
| 3B (ESM2-3B) | A100 40GB or 2x RTX 4090 |

#### Phase 2: Preference Alignment

**Recommended: DPO or GRPO (NOT PPO)**

| If you have... | Use... | Why |
|----------------|--------|-----|
| Preference pairs | DPO | Simplest, lowest compute |
| Ranked completions | GRPO | Better for reasoning tasks |
| Verifiable rewards | REINFORCE++ | Stable, efficient |

**Learning Rate**: 5e-6 (much lower than SFT!)

---

### Multimodal Considerations

For protein structure + sequence:
1. **Use OpenRLHF-M** if doing RLHF on multimodal inputs
2. **VL-RLHF** framework supports multiple VLM architectures
3. **Protein-specific**: LoRA on key/value matrices only (not all layers)

---

### Quick Start Checklist

- [ ] Install: `pip install trl peft bitsandbytes unsloth`
- [ ] Start with QLoRA on smallest viable model
- [ ] Use rank r=4 minimum (r=8 recommended)
- [ ] Train on completions only (mask inputs)
- [ ] 1-3 epochs maximum
- [ ] Validate on held-out set before scaling

---

### GPU Cost Estimates (Cloud)

| Setup | Model Size | Cost/Hour | Time for 1 Epoch (50K samples) |
|-------|-----------|-----------|-------------------------------|
| RTX 4090 (24GB) | 7B QLoRA | ~$0.75 | ~3-4 hours |
| A100 40GB | 13B QLoRA | ~$3.00 | ~2-3 hours |
| A100 80GB | 70B QLoRA | ~$5.00 | ~6-8 hours |

---

## Sources

### Training Frameworks
- [Open Source RL Libraries for LLMs - Anyscale](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)
- [OpenRLHF: An Easy-to-use, Scalable RLHF Framework](https://arxiv.org/html/2405.11143v6)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [OpenRLHF-M (Multimodal) GitHub](https://github.com/OpenRLHF/OpenRLHF-M)
- [veRL GitHub](https://github.com/volcengine/verl)
- [Accelerating RLHF with vLLM - OpenRLHF](https://blog.vllm.ai/2025/04/23/openrlhf-vllm.html)

### SFT Methods
- [How to fine-tune open LLMs in 2025 - Phil Schmid](https://www.philschmid.de/fine-tune-llms-in-2025)
- [Efficient Fine-Tuning with LoRA - Databricks](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [LoRA Hyperparameters Guide - Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Fine-Tuning Infrastructure: LoRA, QLoRA at Scale](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025)
- [PEFT Methods - HuggingFace](https://huggingface.co/blog/samuellimabraz/peft-methods)

### RL Methods
- [Preference Tuning LLMs: PPO, DPO, GRPO Guide](https://anukriti-ranjan.medium.com/preference-tuning-llms-ppo-dpo-grpo-a-simple-guide-135765c87090)
- [Group Relative Policy Optimization (GRPO)](https://cameronrwolfe.substack.com/p/grpo)
- [DPO, GRPO, RLHF and All That](https://mlops.substack.com/p/dpo-grpo-rlhf-and-all-that)
- [DAPO: Open-Source LLM RL at Scale](https://arxiv.org/abs/2503.14476)
- [REINFORCE: Easy Online RL for LLMs](https://cameronrwolfe.substack.com/p/reinforce)

### Recent Advances
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Make LLM Fine-tuning 2x faster with Unsloth - HuggingFace](https://huggingface.co/blog/unsloth-trl)
- [State of LLMs 2025 - Sebastian Raschka](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
- [Post-training methods for language models - Red Hat](https://developers.redhat.com/articles/2025/11/04/post-training-methods-language-models)

### Protein Language Models
- [Fine-tuning protein language models boosts predictions - Nature Communications](https://www.nature.com/articles/s41467-024-51844-2)
- [Democratizing protein language models with PEFT - PNAS](https://www.pnas.org/doi/10.1073/pnas.2405840121)
- [SeqProFT: LoRA for Protein Property Predictions](https://arxiv.org/html/2411.11530v1)
- [Fine tune ProtTrans using HuggingFace - Galaxy Training](https://training.galaxyproject.org/training-material/topics/statistics/tutorials/fine_tuning_protTrans/tutorial.html)
- [Efficient inference and fine-tuning of protein language models - iScience](https://www.cell.com/iscience/fulltext/S2589-0042(25)01756-0)

---

*Last updated: February 2025*
