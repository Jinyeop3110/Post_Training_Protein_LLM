---
title: "Project Kickoff: Post-Training Protein LLM"
date: 2026-02-19
author: yeopjin
tags: [kickoff, architecture]
---

# Project Kickoff: Post-Training Protein LLM

Today marks the official start of the **Post-Training Protein LLM** project — a research effort to build a multimodal language model that truly understands proteins by combining frozen protein language model embeddings with large language models.

## Motivation

Protein language models like ESM-3 have learned rich structural and functional representations from millions of protein sequences. Meanwhile, LLMs excel at reasoning and instruction-following. The question driving this project: **what happens when we bridge these two worlds?**

Rather than training a protein model from scratch, we take a post-training approach — freeze the protein encoder, attach a learnable projector, and fine-tune an LLM to interpret protein embeddings alongside natural language. This lets us leverage the best of both worlds without the prohibitive cost of pre-training.

## Core Architecture

The system follows a modular pipeline:

1. **Protein Encoder** (frozen): ESM-3 small (1.4B params, 1536-dim embeddings)
2. **Attention Pooling**: Learned pooling that compresses variable-length residue embeddings into 32 fixed tokens
3. **MLP Projector** (trainable): Maps 1536-dim protein space into 2560-dim LLM input space
4. **LLM**: Qwen3-4B with LoRA on key/value matrices

The design supports two encoding approaches -- raw text sequences and ESM-3 embeddings -- switchable via a single config flag.

## Training Plan

**Phase 1: Supervised Fine-Tuning (SFT)**
- Train on 505K instruction pairs from Mol-Instructions dataset
- QLoRA/LoRA with differential learning rates (projector at 10x base LR)
- Target tasks: GO term prediction, PPI prediction, stability prediction

**Phase 2: Reinforcement Learning (GRPO/DPO)**
- Task-specific reward functions (F1 for GO terms, accuracy for PPI, Gaussian for stability)
- veRL framework with FSDP backend

## Infrastructure

We're running on **8x NVIDIA H100 80GB** GPUs with CUDA 12.4 — more than enough compute to iterate quickly on 4B-scale models. The full stack includes PyTorch 2.5.1, vLLM, Hydra configs, and wandb for experiment tracking.

## Datasets Secured

| Dataset | Size | Purpose |
|---------|------|---------|
| Mol-Instructions | 505K pairs | SFT training (instruction-following) |
| IPD PDB | 556K chains | Protein structure data |
| Swiss-Prot | 570K sequences | Curated protein sequences |

## What's Next

The immediate priorities are:

1. Complete the first full SFT run on Mol-Instructions
2. Evaluate the trained checkpoint on GO prediction and PPI benchmarks
3. Fix the GRPO gradient flow issue for Phase 2 RL training
4. Build out test coverage for critical model components

This project sits at the intersection of protein science and language modeling — a space with enormous potential. Let's see where it goes.
