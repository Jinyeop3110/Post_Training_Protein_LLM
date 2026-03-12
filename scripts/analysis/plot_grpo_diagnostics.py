#!/usr/bin/env python3
"""
GRPO diagnostic figures for protein-LLM experiments.

Generates:
  fig5_sft_loss_curves    - SFT eval loss (MLP vs Text)
  fig7_grpo_reward_curves - GRPO reward vs step (all runs)
  fig8_grpo_reward_breakdown - Reward component breakdown (bar chart)
  fig9_grpo_grad_norms    - GRPO gradient norms vs step
"""

import matplotlib

matplotlib.use("Agg")
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import THE style system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "analysis"))
from figure_style import (
    annotate_bars,
    annotate_best,
    get_label,
    save_main_figure,
    style,
)

# Apply blog style
colors = style.apply("blog")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

BASE = "/orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM"

EXPERIMENTS = {
    "sft_mlp": {
        "path": f"{BASE}/results/sft_esm3_mlp_combined_qwen3_8b_it_0306_010408",
        "approach": "mlp",
        "label": "SFT: ESM3+MLP",
        "type": "sft",
    },
    "sft_text": {
        "path": f"{BASE}/results/sft_text_combined_qwen3_8b_it_0307_190324",
        "approach": "text",
        "label": "SFT: Text-only",
        "type": "sft",
    },
    "grpo_plm_v1": {
        "path": f"{BASE}/results/grpo_proteinlm_mlp_sft_qwen3_8b_it_0310_214259",
        "approach": "mlp",
        "label": "ProteinLM v1 (3 ep)",
        "type": "grpo",
        "task": "proteinlm_bench",
    },
    "grpo_plm_v2": {
        "path": f"{BASE}/results/grpo_proteinlm_mlp_sft_qwen3_8b_it_0311_091513",
        "approach": "mlp",
        "label": "ProteinLM v2 (10 ep)",
        "type": "grpo",
        "task": "proteinlm_bench",
    },
    "grpo_struct": {
        "path": f"{BASE}/results/grpo_structure_mlp_sft_qwen3_8b_it_0311_114205",
        "approach": "mlp",
        "label": "Structure (10 ep)",
        "type": "grpo",
        "task": "structure",
    },
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_sft_trainer_state(exp_path):
    """Load log_history from HF Trainer trainer_state.json (latest checkpoint)."""
    ckpt_dir = os.path.join(exp_path, "checkpoints")
    # Find latest checkpoint
    ckpts = sorted(
        [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    if not ckpts:
        return []
    state_path = os.path.join(ckpt_dir, ckpts[-1], "trainer_state.json")
    if not os.path.exists(state_path):
        return []
    with open(state_path) as f:
        state = json.load(f)
    return state.get("log_history", [])


def parse_grpo_step_log(log_path):
    """Parse step-level metrics from GRPO train.log."""
    pattern = re.compile(
        r"Step (\d+): loss=([-\d.e+]+), reward=([-\d.e+]+), "
        r"lr=([-\d.e+]+), .* grad_lora=([-\d.e+]+), grad_mm=([-\d.e+]+)"
    )
    rows = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "reward": float(m.group(3)),
                    "lr": float(m.group(4)),
                    "grad_lora": float(m.group(5)),
                    "grad_mm": float(m.group(6)),
                })
    return pd.DataFrame(rows)


def parse_grpo_eval_log(log_path):
    """Parse eval metrics from GRPO train.log."""
    pattern = re.compile(r"Eval metrics: (\{.*\})")
    rows = []
    step_pattern = re.compile(r"Step (\d+):")
    current_step = 0
    with open(log_path) as f:
        for line in f:
            sm = step_pattern.search(line)
            if sm:
                current_step = int(sm.group(1))
            em = pattern.search(line)
            if em:
                data = eval(em.group(1))  # Safe: we control the log format
                data["step"] = current_step
                rows.append(data)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# FIG 5: SFT EVAL LOSS CURVES
# ─────────────────────────────────────────────────────────────

def plot_sft_loss_curves():
    """Plot eval loss vs step for SFT MLP and SFT Text."""
    print("\n=== Fig 5: SFT Loss Curves ===")

    fig, ax = style.subplots()

    for key in ["sft_mlp", "sft_text"]:
        exp = EXPERIMENTS[key]
        log_history = load_sft_trainer_state(exp["path"])
        if not log_history:
            print(f"  WARNING: No trainer_state.json found for {key}")
            continue

        # Extract eval entries (they have eval_loss)
        eval_entries = [e for e in log_history if "eval_loss" in e]
        if not eval_entries:
            print(f"  WARNING: No eval entries for {key}")
            continue

        steps = [e["step"] for e in eval_entries]
        eval_loss = [e["eval_loss"] for e in eval_entries]

        color = style.color(exp["approach"])
        ax.plot(steps, eval_loss, color=color, label=exp["label"],
                marker="o", markersize=4, alpha=0.85)

        # Annotate best
        best_idx = np.argmin(eval_loss)
        annotate_best(ax, steps[best_idx], eval_loss[best_idx],
                      eval_loss[best_idx], color, offset=(15, 10))

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("eval_loss"))
    ax.set_title("SFT Eval Loss: ESM3+MLP vs Text-only (Qwen3-8B, 4.89M)")
    ax.legend(loc="upper right")
    style.clean_axes(ax)
    fig.tight_layout()

    saved = save_main_figure(fig, "fig5_sft_loss_curves")
    print(f"  Saved: {saved}")
    return saved


# ─────────────────────────────────────────────────────────────
# FIG 7: GRPO REWARD CURVES
# ─────────────────────────────────────────────────────────────

def plot_grpo_reward_curves():
    """Plot mean reward vs step for all GRPO runs."""
    print("\n=== Fig 7: GRPO Reward Curves ===")

    fig, ax = style.subplots()

    # Use distinct colors for tasks + linestyles for versions
    task_colors = {
        "proteinlm_bench": style.color("mlp"),      # Blue
        "structure":       style.color("perceiver"),  # Orange
    }
    run_styles = {
        "grpo_plm_v1": {"ls": "--", "marker": "s", "label_suffix": ""},
        "grpo_plm_v2": {"ls": "-",  "marker": "o", "label_suffix": ""},
        "grpo_struct": {"ls": "-",  "marker": "D", "label_suffix": ""},
    }

    for key in ["grpo_plm_v1", "grpo_plm_v2", "grpo_struct"]:
        exp = EXPERIMENTS[key]
        log_path = os.path.join(exp["path"], "train.log")
        df = parse_grpo_step_log(log_path)
        if df.empty:
            print(f"  WARNING: No step data for {key}")
            continue

        task = exp["task"]
        rs = run_styles[key]
        color = task_colors[task]

        ax.plot(df["step"], df["reward"],
                color=color, linestyle=rs["ls"], marker=rs["marker"],
                markersize=4, label=exp["label"], alpha=0.85)

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Mean Reward")
    ax.set_title("GRPO Training: Reward vs Step")
    ax.legend(loc="lower right", framealpha=0.9)
    style.clean_axes(ax)
    fig.tight_layout()

    saved = save_main_figure(fig, "fig7_grpo_reward_curves")
    print(f"  Saved: {saved}")
    return saved


# ─────────────────────────────────────────────────────────────
# FIG 8: GRPO REWARD BREAKDOWN
# ─────────────────────────────────────────────────────────────

def plot_grpo_reward_breakdown():
    """Grouped bar chart showing reward components for each GRPO task."""
    print("\n=== Fig 8: GRPO Reward Breakdown ===")

    fig, axes = plt.subplots(1, 2, figsize=style.figsize("wide"))

    # ── Panel A: ProteinLM Bench (v2) ──
    ax = axes[0]
    plm_metrics_path = os.path.join(
        EXPERIMENTS["grpo_plm_v2"]["path"], "metrics.json")
    with open(plm_metrics_path) as f:
        plm = json.load(f)

    plm_components = {
        "Correct\nIndex": plm.get("reward/correct_index", 0),
        "Predicted\nIndex": plm.get("reward/predicted_index", 0),
        "Format\nBonus": plm.get("reward/format_bonus", 0),
    }
    # These are raw averages across the dataset; correct_index and predicted_index
    # are average index values, not reward components in [0,1].
    # The actual reward is mean_reward. Let's show mean/max/min as summary,
    # and the component averages as a separate informational chart.

    # Show mean_reward composition: the final reward is a combination
    # For ProteinLM bench: reward = correct_match * 1.0 + format_bonus
    plm_bars = {
        "Mean\nReward": plm.get("mean_reward", 0),
        "Max\nReward": plm.get("max_reward", 0),
        "Min\nReward": plm.get("min_reward", 0),
        "Format\nBonus": plm.get("reward/format_bonus", 0),
    }

    x = np.arange(len(plm_bars))
    color = style.color("mlp")
    bars = ax.bar(x, list(plm_bars.values()), color=color, alpha=0.8, width=0.6)
    annotate_bars(ax, bars, list(plm_bars.values()), fmt="{:.3f}", offset=0.01)
    ax.set_xticks(x)
    ax.set_xticklabels(list(plm_bars.keys()), fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("ProteinLM Bench (v2, 10 ep)")
    style.clean_axes(ax)

    # ── Panel B: Structure ──
    ax = axes[1]
    struct_metrics_path = os.path.join(
        EXPERIMENTS["grpo_struct"]["path"], "metrics.json")
    with open(struct_metrics_path) as f:
        struct = json.load(f)

    struct_components = {
        "Quality\nAlign.": struct.get("reward/quality_alignment", 0),
        "Numerical\nAccuracy": struct.get("reward/numerical_accuracy", 0),
        "Category\nMatch": struct.get("reward/category_match", 0),
        "Format\nBonus": struct.get("reward/format_bonus", 0),
    }

    x = np.arange(len(struct_components))
    color = style.color("perceiver")  # Orange for structure task
    bars = ax.bar(x, list(struct_components.values()), color=color, alpha=0.8, width=0.6)
    annotate_bars(ax, bars, list(struct_components.values()), fmt="{:.3f}", offset=0.005)
    ax.set_xticks(x)
    ax.set_xticklabels(list(struct_components.keys()), fontsize=9)
    ax.set_ylabel("Reward Component Value")
    ax.set_title("Structure Quality (10 ep)")
    style.clean_axes(ax)

    fig.suptitle("GRPO Reward Component Breakdown", fontsize=13, y=1.02)
    fig.tight_layout()

    saved = save_main_figure(fig, "fig8_grpo_reward_breakdown")
    print(f"  Saved: {saved}")
    return saved


# ─────────────────────────────────────────────────────────────
# FIG 9: GRPO GRADIENT NORMS
# ─────────────────────────────────────────────────────────────

def plot_grpo_grad_norms():
    """Plot grad_norm_lora and grad_norm_multimodal vs step for GRPO runs."""
    print("\n=== Fig 9: GRPO Gradient Norms ===")

    fig, axes = plt.subplots(1, 2, figsize=style.figsize("wide"), sharey=False)

    task_colors = {
        "proteinlm_bench": style.color("mlp"),
        "structure":       style.color("perceiver"),
    }
    run_styles = {
        "grpo_plm_v1": {"ls": "--", "marker": "s"},
        "grpo_plm_v2": {"ls": "-",  "marker": "o"},
        "grpo_struct": {"ls": "-",  "marker": "D"},
    }

    # ── Panel A: LoRA Grad Norms ──
    ax = axes[0]
    for key in ["grpo_plm_v1", "grpo_plm_v2", "grpo_struct"]:
        exp = EXPERIMENTS[key]
        log_path = os.path.join(exp["path"], "train.log")
        df = parse_grpo_step_log(log_path)
        if df.empty:
            continue

        rs = run_styles[key]
        color = task_colors[exp["task"]]

        ax.plot(df["step"], df["grad_lora"],
                color=color, linestyle=rs["ls"], marker=rs["marker"],
                markersize=3, label=exp["label"], alpha=0.8)

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("LoRA Gradient Norm")
    ax.set_title("LoRA Gradient Norms")
    ax.legend(loc="upper right", fontsize=8)
    style.clean_axes(ax)

    # ── Panel B: Multimodal Grad Norms ──
    ax = axes[1]
    for key in ["grpo_plm_v1", "grpo_plm_v2", "grpo_struct"]:
        exp = EXPERIMENTS[key]
        log_path = os.path.join(exp["path"], "train.log")
        df = parse_grpo_step_log(log_path)
        if df.empty:
            continue

        rs = run_styles[key]
        color = task_colors[exp["task"]]

        ax.plot(df["step"], df["grad_mm"],
                color=color, linestyle=rs["ls"], marker=rs["marker"],
                markersize=3, label=exp["label"], alpha=0.8)

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Multimodal Gradient Norm")
    ax.set_title("Multimodal (Pooling+Projector) Gradient Norms")
    ax.legend(loc="upper right", fontsize=8)
    style.clean_axes(ax)

    fig.suptitle("GRPO Gradient Norms", fontsize=13, y=1.02)
    fig.tight_layout()

    saved = save_main_figure(fig, "fig9_grpo_grad_norms")
    print(f"  Saved: {saved}")
    return saved


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GRPO Diagnostic Figures")
    print("=" * 60)

    all_saved = []

    all_saved.extend(plot_sft_loss_curves())
    all_saved.extend(plot_grpo_reward_curves())
    all_saved.extend(plot_grpo_reward_breakdown())
    all_saved.extend(plot_grpo_grad_norms())

    print("\n" + "=" * 60)
    print(f"All figures saved ({len(all_saved)} files):")
    for p in all_saved:
        print(f"  {p}")
    print("=" * 60)
