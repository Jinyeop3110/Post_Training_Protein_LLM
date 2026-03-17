#!/usr/bin/env python3
"""
Comprehensive diagnostic plots for all experiments — 2026-03-16 refresh.
Reads directly from results/ trainer_state.json and train.log files.
Saves to blog/figures/supple_figures/ and paper/figures/supplementary/.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from figure_style import (
    RESULTS_DIR,
    annotate_bars,
    annotate_best,
    get_approach_label,
    get_label,
    save_figure,
    shorten,
    smooth,
    style,
    to_rgba_alpha,
)

# Apply blog style
colors = style.apply("blog")

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

SFT_EXPERIMENTS = {
    "sft_esm3_mlp_combined_qwen3_8b_it_0306_010408": {
        "label": "ESM3+MLP (4.89M)",
        "approach": "mlp",
        "checkpoint": "checkpoint-9750",
    },
    "sft_text_combined_qwen3_8b_it_0307_190324": {
        "label": "Text-only (4.89M)",
        "approach": "text",
        "checkpoint": "checkpoint-9647",
    },
    "sft_esm3_mlp_thinking_qwen3_8b_it_0309_115011": {
        "label": "ESM3+MLP+Thinking",
        "approach": "mlp",
        "checkpoint": "checkpoint-500",
    },
}

def load_sft_data():
    """Load trainer_state.json for all SFT experiments."""
    data = {}
    for exp_name, meta in SFT_EXPERIMENTS.items():
        ckpt = meta["checkpoint"]
        path = RESULTS_DIR / exp_name / "checkpoints" / ckpt / "trainer_state.json"
        if not path.exists():
            print(f"  SKIP {exp_name}: no trainer_state.json")
            continue
        with open(path) as f:
            state = json.load(f)
        df = pd.DataFrame(state["log_history"])
        data[exp_name] = {
            "df": df,
            "label": meta["label"],
            "approach": meta["approach"],
        }
        print(f"  Loaded {exp_name}: {len(df)} entries, max_step={df['step'].max()}")
    return data


def parse_grpo_log(log_path):
    """Parse GRPO train.log for step-level metrics."""
    pattern = re.compile(
        r"Step (\d+): loss=([-\d.e+]+), reward=([-\d.e+]+), "
        r"lr=([-\d.e+]+).*grad_lora=([-\d.e+]+), grad_mm=([-\d.e+]+)"
    )
    eval_pattern = re.compile(
        r"Eval metrics: \{.*?'mean_reward': ([-\d.e+]+).*?'format_rate': ([-\d.e+]+)"
    )

    steps, losses, rewards, lrs, grad_loras, grad_mms = [], [], [], [], [], []
    eval_steps, eval_rewards, eval_format_rates = [], [], []

    with open(log_path) as f:
        last_step = 0
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                steps.append(step)
                losses.append(float(m.group(2)))
                rewards.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
                grad_loras.append(float(m.group(5)))
                grad_mms.append(float(m.group(6)))
                last_step = step

            em = eval_pattern.search(line)
            if em:
                eval_steps.append(last_step)
                eval_rewards.append(float(em.group(1)))
                eval_format_rates.append(float(em.group(2)))

    train_df = pd.DataFrame({
        "step": steps, "loss": losses, "reward": rewards,
        "lr": lrs, "grad_lora": grad_loras, "grad_mm": grad_mms,
    })
    eval_df = pd.DataFrame({
        "step": eval_steps, "eval_reward": eval_rewards,
        "format_rate": eval_format_rates,
    })
    return train_df, eval_df


def load_grpo_data():
    """Load train.log for all GRPO experiments."""
    data = {}
    grpo_dirs = sorted(RESULTS_DIR.glob("grpo_*"))
    for d in grpo_dirs:
        name = d.name
        log_path = d / "train.log"
        lineage_path = d / "lineage.json"
        if not log_path.exists():
            continue

        # Get approach from lineage
        approach = "mlp"
        if lineage_path.exists():
            with open(lineage_path) as f:
                lin = json.load(f)
            approach = lin.get("approach", "esm3")
            if approach == "esm3":
                approach = "mlp"

        # Determine task from name
        if "proteinlm" in name:
            task = "proteinlm"
        elif "structure" in name:
            task = "structure"
        else:
            task = "unknown"

        # Short label
        short = name.replace("grpo_", "").replace("_qwen3_8b_it_", " ")
        # Further shorten
        short = short.split(" ")[0]  # Keep only the config part

        train_df, eval_df = parse_grpo_log(log_path)
        if len(train_df) == 0:
            print(f"  SKIP {name}: no step data in train.log")
            continue

        data[name] = {
            "train": train_df,
            "eval": eval_df,
            "approach": approach,
            "task": task,
            "label": short,
        }
        print(f"  Loaded {name}: {len(train_df)} steps, approach={approach}, task={task}")

    return data


# ═══════════════════════════════════════════════════════════════════
# SFT PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_sft_loss_curves(sft_data):
    """Plot 1: SFT training loss (token_avg_loss) for all runs."""
    fig, ax = style.subplots()

    for exp_name, d in sft_data.items():
        df = d["df"]
        # Filter to training entries (those without eval_loss)
        if "eval_loss" in df.columns:
            train = df[df["eval_loss"].isna() & df["token_avg_loss"].notna()]
        else:
            train = df[df["token_avg_loss"].notna()]

        color = style.color(d["approach"])
        # Differentiate MLP runs with linestyle
        ls = "--" if "Thinking" in d["label"] else "-"

        # Raw data (light)
        ax.plot(train["step"], train["token_avg_loss"],
                color=to_rgba_alpha(color, 0.15), linewidth=0.5)
        # Smoothed
        smoothed = smooth(train["token_avg_loss"].values, window=20)
        ax.plot(train["step"], smoothed,
                color=color, label=d["label"], linewidth=1.5, linestyle=ls)

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("token_avg_loss"))
    ax.set_title("SFT Training Loss — All Experiments")
    ax.legend(loc="upper right")
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_sft_loss_curves_0316")
    style.apply("paper")
    fig2, ax2 = style.subplots()
    for exp_name, d in sft_data.items():
        df = d["df"]
        train = df[df["eval_loss"].isna()] if "eval_loss" in df.columns else df
        train = train[train["token_avg_loss"].notna()]
        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"
        smoothed = smooth(train["token_avg_loss"].values, window=20)
        ax2.plot(train["step"], smoothed,
                 color=color, label=d["label"], linewidth=1.0, linestyle=ls)
    ax2.set_xlabel(get_label("step"))
    ax2.set_ylabel(get_label("token_avg_loss"))
    ax2.legend(loc="upper right")
    style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_sft_loss_curves_0316", target="paper")
    style.apply("blog")


def plot_sft_eval_loss(sft_data):
    """Plot 2: SFT eval loss curves with best-point annotations."""
    fig, ax = style.subplots()

    for exp_name, d in sft_data.items():
        df = d["df"]
        evals = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
        if evals.empty:
            continue

        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"

        ax.plot(evals["step"], evals["eval_loss"],
                color=color, label=d["label"], linewidth=1.5,
                marker="o", markersize=4, linestyle=ls)

        # Annotate best
        best_idx = evals["eval_loss"].idxmin()
        best_step = evals.loc[best_idx, "step"]
        best_val = evals.loc[best_idx, "eval_loss"]
        annotate_best(ax, best_step, best_val, best_val, color)

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("eval_loss"))
    ax.set_title("SFT Eval Loss — All Experiments")
    ax.legend(loc="upper right")
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_sft_eval_loss_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots()
    for exp_name, d in sft_data.items():
        df = d["df"]
        evals = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
        if evals.empty:
            continue
        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"
        ax2.plot(evals["step"], evals["eval_loss"],
                 color=color, label=d["label"], linewidth=1.0,
                 marker="o", markersize=2, linestyle=ls)
        best_idx = evals["eval_loss"].idxmin()
        best_step = evals.loc[best_idx, "step"]
        best_val = evals.loc[best_idx, "eval_loss"]
        annotate_best(ax2, best_step, best_val, best_val, color, offset=(8, -10))
    ax2.set_xlabel(get_label("step"))
    ax2.set_ylabel(get_label("eval_loss"))
    ax2.legend(loc="upper right")
    style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_sft_eval_loss_0316", target="paper")
    style.apply("blog")


def plot_sft_gradient_norms(sft_data):
    """Plot 3: SFT gradient norms (log scale)."""
    fig, ax = style.subplots()

    for exp_name, d in sft_data.items():
        df = d["df"]
        train = df[df["grad_norm"].notna()]
        if "eval_loss" in df.columns:
            train = train[train["eval_loss"].isna()]

        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"

        # Raw (light)
        ax.plot(train["step"], train["grad_norm"],
                color=to_rgba_alpha(color, 0.12), linewidth=0.5)
        # Smoothed
        smoothed = smooth(train["grad_norm"].values, window=20)
        ax.plot(train["step"], smoothed,
                color=color, label=d["label"], linewidth=1.5, linestyle=ls)

    ax.set_yscale("log")
    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("grad_norm"))
    ax.set_title("SFT Gradient Norms — All Experiments")
    ax.legend(loc="upper right")
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_sft_gradient_norms_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots()
    for exp_name, d in sft_data.items():
        df = d["df"]
        train = df[df["grad_norm"].notna()]
        if "eval_loss" in df.columns:
            train = train[train["eval_loss"].isna()]
        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"
        smoothed = smooth(train["grad_norm"].values, window=20)
        ax2.plot(train["step"], smoothed,
                 color=color, label=d["label"], linewidth=1.0, linestyle=ls)
    ax2.set_yscale("log")
    ax2.set_xlabel(get_label("step"))
    ax2.set_ylabel(get_label("grad_norm"))
    ax2.legend(loc="upper right")
    style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_sft_gradient_norms_0316", target="paper")
    style.apply("blog")


def plot_sft_lr_schedule(sft_data):
    """Plot 4: SFT learning rate schedules."""
    fig, ax = style.subplots()

    for exp_name, d in sft_data.items():
        df = d["df"]
        train = df[df["learning_rate"].notna()]
        if "eval_loss" in df.columns:
            train = train[train["eval_loss"].isna()]

        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"

        ax.plot(train["step"], train["learning_rate"],
                color=color, label=d["label"], linewidth=1.5, linestyle=ls)

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("learning_rate"))
    ax.set_title("SFT Learning Rate Schedule — All Experiments")
    ax.legend(loc="upper right")
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_sft_lr_schedule_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots()
    for exp_name, d in sft_data.items():
        df = d["df"]
        train = df[df["learning_rate"].notna()]
        if "eval_loss" in df.columns:
            train = train[train["eval_loss"].isna()]
        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"
        ax2.plot(train["step"], train["learning_rate"],
                 color=color, label=d["label"], linewidth=1.0, linestyle=ls)
    ax2.set_xlabel(get_label("step"))
    ax2.set_ylabel(get_label("learning_rate"))
    ax2.legend(loc="upper right")
    style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_sft_lr_schedule_0316", target="paper")
    style.apply("blog")


def plot_sft_loss_comparison_bar(sft_data):
    """Plot 5: SFT loss comparison bar chart."""
    fig, ax = style.subplots()

    experiments = []
    final_train_losses = []
    best_eval_losses = []
    bar_colors = []

    for exp_name, d in sft_data.items():
        df = d["df"]
        # Final train loss: last non-eval entry
        train = df[df["eval_loss"].isna()] if "eval_loss" in df.columns else df
        train = train[train["token_avg_loss"].notna()]
        final_train = train["token_avg_loss"].iloc[-1] if len(train) > 0 else np.nan

        # Best eval loss
        evals = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
        best_eval = evals["eval_loss"].min() if not evals.empty else np.nan

        experiments.append(d["label"])
        final_train_losses.append(final_train)
        best_eval_losses.append(best_eval)
        bar_colors.append(style.color(d["approach"]))

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width/2, final_train_losses, width, label="Final Train Loss",
                   color=[to_rgba_alpha(c, 0.6) for c in bar_colors], edgecolor=bar_colors)
    bars2 = ax.bar(x + width/2, best_eval_losses, width, label="Best Eval Loss",
                   color=bar_colors, edgecolor=bar_colors)

    annotate_bars(ax, bars1, final_train_losses)
    annotate_bars(ax, bars2, best_eval_losses)

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha="right")
    ax.set_ylabel("Loss")
    ax.set_title("SFT Loss Comparison — Final Train vs Best Eval")
    ax.legend()
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_sft_loss_comparison_bar_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots("wide")
    bars1 = ax2.bar(x - width/2, final_train_losses, width, label="Final Train Loss",
                    color=[to_rgba_alpha(c, 0.6) for c in bar_colors], edgecolor=bar_colors)
    bars2 = ax2.bar(x + width/2, best_eval_losses, width, label="Best Eval Loss",
                    color=bar_colors, edgecolor=bar_colors)
    annotate_bars(ax2, bars1, final_train_losses)
    annotate_bars(ax2, bars2, best_eval_losses)
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments, rotation=15, ha="right")
    ax2.set_ylabel("Loss")
    ax2.legend()
    style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_sft_loss_comparison_bar_0316", target="paper")
    style.apply("blog")


# ═══════════════════════════════════════════════════════════════════
# GRPO PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_grpo_reward_curves(grpo_data):
    """Plot 6: GRPO reward vs step for all runs."""
    fig, ax = style.subplots("wide")

    linestyles = ["-", "--", "-.", ":"]
    # Group by approach
    mlp_runs = {k: v for k, v in grpo_data.items() if v["approach"] == "mlp"}
    text_runs = {k: v for k, v in grpo_data.items() if v["approach"] == "text"}

    for i, (name, d) in enumerate(sorted(mlp_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("mlp")
        label_str = f"MLP: {d['label']}" if i < 6 else None  # Limit legend items
        ax.plot(df["step"], df["reward"],
                color=to_rgba_alpha(color, 0.7), linestyle=ls,
                linewidth=1.2, label=label_str)

    for i, (name, d) in enumerate(sorted(text_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("text")
        ax.plot(df["step"], df["reward"],
                color=to_rgba_alpha(color, 0.9), linestyle=ls,
                linewidth=1.5, label=f"Text: {d['label']}")

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Mean Reward")
    ax.set_title("GRPO Reward Trajectories — All Experiments")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_grpo_reward_curves_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots("wide")
    for i, (name, d) in enumerate(sorted(mlp_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("mlp")
        label_str = f"MLP: {d['label']}" if i < 6 else None
        ax2.plot(df["step"], df["reward"],
                 color=to_rgba_alpha(color, 0.7), linestyle=ls,
                 linewidth=0.8, label=label_str)
    for i, (name, d) in enumerate(sorted(text_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("text")
        ax2.plot(df["step"], df["reward"],
                 color=to_rgba_alpha(color, 0.9), linestyle=ls,
                 linewidth=1.0, label=f"Text: {d['label']}")
    ax2.set_xlabel(get_label("step"))
    ax2.set_ylabel("Mean Reward")
    ax2.legend(loc="lower right", fontsize=5, ncol=2)
    style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_grpo_reward_curves_0316", target="paper")
    style.apply("blog")


def plot_grpo_gradient_norms(grpo_data):
    """Plot 7: GRPO gradient norms — 2-panel: LoRA + Multimodal."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    linestyles = ["-", "--", "-.", ":"]
    mlp_runs = {k: v for k, v in grpo_data.items() if v["approach"] == "mlp"}
    text_runs = {k: v for k, v in grpo_data.items() if v["approach"] == "text"}

    # Panel A: LoRA grad norms
    for i, (name, d) in enumerate(sorted(mlp_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("mlp")
        label_str = f"MLP: {d['label']}" if i < 6 else None
        ax1.plot(df["step"], df["grad_lora"],
                 color=to_rgba_alpha(color, 0.7), linestyle=ls,
                 linewidth=1.2, label=label_str)
    for i, (name, d) in enumerate(sorted(text_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("text")
        ax1.plot(df["step"], df["grad_lora"],
                 color=to_rgba_alpha(color, 0.9), linestyle=ls,
                 linewidth=1.5, label=f"Text: {d['label']}")

    ax1.set_yscale("log")
    ax1.set_ylim(bottom=1e-5)
    ax1.set_xlabel(get_label("step"))
    ax1.set_ylabel("LoRA Gradient Norm")
    ax1.set_title("(A) LoRA Gradient Norms")
    ax1.legend(loc="upper right", fontsize=6, ncol=2)
    style.clean_axes(ax1)

    # Panel B: Multimodal grad norms
    for i, (name, d) in enumerate(sorted(mlp_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("mlp")
        label_str = f"MLP: {d['label']}" if i < 6 else None
        ax2.plot(df["step"], df["grad_mm"],
                 color=to_rgba_alpha(color, 0.7), linestyle=ls,
                 linewidth=1.2, label=label_str)
    for i, (name, d) in enumerate(sorted(text_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("text")
        ax2.plot(df["step"], df["grad_mm"],
                 color=to_rgba_alpha(color, 0.9), linestyle=ls,
                 linewidth=1.5, label=f"Text: {d['label']}")

    ax2.set_xlabel(get_label("step"))
    ax2.set_ylabel("Multimodal Gradient Norm")
    ax2.set_title("(B) Multimodal Gradient Norms")
    ax2.legend(loc="upper right", fontsize=6, ncol=2)
    style.clean_axes(ax2)

    fig.tight_layout()
    save_figure(fig, "all_grpo_gradient_norms_0316")
    # Paper version
    style.apply("paper")
    fig2, (ax1p, ax2p) = plt.subplots(1, 2, figsize=style.figsize("wide"))
    for i, (name, d) in enumerate(sorted(mlp_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("mlp")
        label_str = f"MLP: {d['label']}" if i < 4 else None
        ax1p.plot(df["step"], df["grad_lora"],
                  color=to_rgba_alpha(color, 0.7), linestyle=ls, linewidth=0.8, label=label_str)
    for i, (name, d) in enumerate(sorted(text_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("text")
        ax1p.plot(df["step"], df["grad_lora"],
                  color=to_rgba_alpha(color, 0.9), linestyle=ls, linewidth=1.0, label=f"Text: {d['label']}")
    ax1p.set_yscale("log"); ax1p.set_ylim(bottom=1e-5)
    ax1p.set_xlabel(get_label("step")); ax1p.set_ylabel("LoRA Grad Norm")
    ax1p.legend(loc="upper right", fontsize=4, ncol=2); style.clean_axes(ax1p)
    for i, (name, d) in enumerate(sorted(mlp_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("mlp")
        label_str = f"MLP: {d['label']}" if i < 4 else None
        ax2p.plot(df["step"], df["grad_mm"],
                  color=to_rgba_alpha(color, 0.7), linestyle=ls, linewidth=0.8, label=label_str)
    for i, (name, d) in enumerate(sorted(text_runs.items())):
        df = d["train"]
        ls = linestyles[i % len(linestyles)]
        color = style.color("text")
        ax2p.plot(df["step"], df["grad_mm"],
                  color=to_rgba_alpha(color, 0.9), linestyle=ls, linewidth=1.0, label=f"Text: {d['label']}")
    ax2p.set_xlabel(get_label("step")); ax2p.set_ylabel("MM Grad Norm")
    ax2p.legend(loc="upper right", fontsize=4, ncol=2); style.clean_axes(ax2p)
    fig2.tight_layout()
    save_figure(fig2, "all_grpo_gradient_norms_0316", target="paper")
    style.apply("blog")


def plot_grpo_eval_reward(grpo_data):
    """Plot 8: GRPO eval reward trajectories."""
    fig, ax = style.subplots()

    has_data = False
    for name, d in sorted(grpo_data.items()):
        eval_df = d["eval"]
        if eval_df.empty:
            continue
        has_data = True
        color = style.color(d["approach"])
        ls = "--" if d["approach"] == "text" else "-"
        ax.plot(eval_df["step"], eval_df["eval_reward"],
                color=color, linestyle=ls, marker="o", markersize=4,
                linewidth=1.5, label=shorten(d["label"], 25))

    if not has_data:
        plt.close(fig)
        print("  SKIP: no eval reward data for any GRPO run")
        return

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Eval Mean Reward")
    ax.set_title("GRPO Eval Reward Trajectories")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_grpo_eval_reward_trajectories_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots()
    for name, d in sorted(grpo_data.items()):
        eval_df = d["eval"]
        if eval_df.empty:
            continue
        color = style.color(d["approach"])
        ls = "--" if d["approach"] == "text" else "-"
        ax2.plot(eval_df["step"], eval_df["eval_reward"],
                 color=color, linestyle=ls, marker="o", markersize=2,
                 linewidth=1.0, label=shorten(d["label"], 20))
    ax2.set_xlabel(get_label("step")); ax2.set_ylabel("Eval Mean Reward")
    ax2.legend(loc="lower right", fontsize=5, ncol=2); style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_grpo_eval_reward_trajectories_0316", target="paper")
    style.apply("blog")


def plot_grpo_format_compliance(grpo_data):
    """Plot 9: GRPO format compliance rate at eval steps."""
    fig, ax = style.subplots()

    has_data = False
    for name, d in sorted(grpo_data.items()):
        eval_df = d["eval"]
        if eval_df.empty or "format_rate" not in eval_df.columns:
            continue
        has_data = True
        color = style.color(d["approach"])
        ls = "--" if d["approach"] == "text" else "-"
        ax.plot(eval_df["step"], eval_df["format_rate"] * 100,
                color=color, linestyle=ls, marker="s", markersize=4,
                linewidth=1.5, label=shorten(d["label"], 25))

    if not has_data:
        plt.close(fig)
        print("  SKIP: no format compliance data")
        return

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Format Compliance (%)")
    ax.set_title("GRPO Format Compliance Rate")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_grpo_format_compliance_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots()
    for name, d in sorted(grpo_data.items()):
        eval_df = d["eval"]
        if eval_df.empty or "format_rate" not in eval_df.columns:
            continue
        color = style.color(d["approach"])
        ls = "--" if d["approach"] == "text" else "-"
        ax2.plot(eval_df["step"], eval_df["format_rate"] * 100,
                 color=color, linestyle=ls, marker="s", markersize=2,
                 linewidth=1.0, label=shorten(d["label"], 20))
    ax2.set_xlabel(get_label("step")); ax2.set_ylabel("Format Compliance (%)")
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="lower right", fontsize=5, ncol=2); style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_grpo_format_compliance_0316", target="paper")
    style.apply("blog")


def plot_grpo_final_reward_bar(grpo_data):
    """Plot 10: GRPO final reward bar chart."""
    fig, ax = style.subplots("wide")

    names = []
    final_rewards = []
    bar_colors = []

    for name, d in sorted(grpo_data.items()):
        df = d["train"]
        if df.empty:
            continue
        names.append(d["label"])
        final_rewards.append(df["reward"].iloc[-1])
        bar_colors.append(style.color(d["approach"]))

    x = np.arange(len(names))
    bars = ax.bar(x, final_rewards, color=bar_colors, edgecolor=bar_colors, alpha=0.8)
    annotate_bars(ax, bars, final_rewards)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final Mean Reward")
    ax.set_title("GRPO Final Reward — All Experiments")
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_grpo_final_reward_bar_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots("wide")
    bars2 = ax2.bar(x, final_rewards, color=bar_colors, edgecolor=bar_colors, alpha=0.8)
    annotate_bars(ax2, bars2, final_rewards)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
    ax2.set_ylabel("Final Mean Reward")
    style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_grpo_final_reward_bar_0316", target="paper")
    style.apply("blog")


def plot_grpo_step_level_reward(grpo_data):
    """Plot 11: GRPO step-level reward for top runs (smoothed)."""
    fig, ax = style.subplots()

    # Select runs with most steps for clarity
    top_runs = sorted(grpo_data.items(), key=lambda x: len(x[1]["train"]), reverse=True)[:6]

    for name, d in top_runs:
        df = d["train"]
        color = style.color(d["approach"])
        ls = "--" if d["approach"] == "text" else "-"

        # Raw (light)
        ax.plot(df["step"], df["reward"],
                color=to_rgba_alpha(color, 0.15), linewidth=0.5)
        # Smoothed
        if len(df) > 5:
            smoothed = smooth(df["reward"].values, window=5)
        else:
            smoothed = df["reward"].values
        ax.plot(df["step"], smoothed,
                color=color, linestyle=ls, linewidth=1.5,
                label=f"{get_approach_label(d['approach'])}: {shorten(d['label'], 20)}")

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Mean Reward")
    ax.set_title("GRPO Reward (Top 6 Longest Runs, Smoothed)")
    ax.legend(loc="lower right", fontsize=8)
    style.clean_axes(ax)
    fig.tight_layout()

    save_figure(fig, "all_grpo_reward_step_level_0316")
    # Paper version
    style.apply("paper")
    fig2, ax2 = style.subplots()
    for name, d in top_runs:
        df = d["train"]
        color = style.color(d["approach"])
        ls = "--" if d["approach"] == "text" else "-"
        smoothed = smooth(df["reward"].values, window=5) if len(df) > 5 else df["reward"].values
        ax2.plot(df["step"], smoothed,
                 color=color, linestyle=ls, linewidth=1.0,
                 label=f"{get_approach_label(d['approach'])}: {shorten(d['label'], 18)}")
    ax2.set_xlabel(get_label("step")); ax2.set_ylabel("Mean Reward")
    ax2.legend(loc="lower right", fontsize=5); style.clean_axes(ax2)
    fig2.tight_layout()
    save_figure(fig2, "all_grpo_reward_step_level_0316", target="paper")
    style.apply("blog")


def plot_training_overview(sft_data, grpo_data):
    """Plot 12: 4-panel overview figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) SFT eval loss
    ax = axes[0, 0]
    for exp_name, d in sft_data.items():
        df = d["df"]
        evals = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
        if evals.empty:
            continue
        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"
        ax.plot(evals["step"], evals["eval_loss"],
                color=color, label=d["label"], linewidth=1.5,
                marker="o", markersize=3, linestyle=ls)
    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("eval_loss"))
    ax.set_title("(A) SFT Eval Loss")
    ax.legend(fontsize=8)
    style.clean_axes(ax)

    # (B) GRPO final rewards bar
    ax = axes[0, 1]
    names, rewards, bcolors = [], [], []
    for name, d in sorted(grpo_data.items()):
        df = d["train"]
        if df.empty:
            continue
        names.append(shorten(d["label"], 15))
        rewards.append(df["reward"].iloc[-1])
        bcolors.append(style.color(d["approach"]))
    x = np.arange(len(names))
    bars = ax.bar(x, rewards, color=bcolors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Final Reward")
    ax.set_title("(B) GRPO Final Rewards")
    style.clean_axes(ax)

    # (C) SFT grad norms
    ax = axes[1, 0]
    for exp_name, d in sft_data.items():
        df = d["df"]
        train = df[df["grad_norm"].notna()]
        if "eval_loss" in df.columns:
            train = train[train["eval_loss"].isna()]
        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"
        smoothed = smooth(train["grad_norm"].values, window=20)
        ax.plot(train["step"], smoothed,
                color=color, label=d["label"], linewidth=1.5, linestyle=ls)
    ax.set_yscale("log")
    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("grad_norm"))
    ax.set_title("(C) SFT Gradient Norms")
    ax.legend(fontsize=8)
    style.clean_axes(ax)

    # (D) GRPO reward trajectories (best per config)
    ax = axes[1, 1]
    # Pick best run per approach+task combo
    seen = {}
    for name, d in sorted(grpo_data.items(), key=lambda x: -len(x[1]["train"])):
        key = f"{d['approach']}_{d['task']}"
        if key in seen:
            continue
        seen[key] = True
        df = d["train"]
        color = style.color(d["approach"])
        ls = "--" if d["approach"] == "text" else "-"
        smoothed = smooth(df["reward"].values, window=5) if len(df) > 5 else df["reward"].values
        ax.plot(df["step"], smoothed,
                color=color, linestyle=ls, linewidth=1.5,
                label=f"{get_approach_label(d['approach'])}: {d['task']}")
    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Mean Reward")
    ax.set_title("(D) GRPO Best Reward Trajectories")
    ax.legend(fontsize=8)
    style.clean_axes(ax)

    fig.tight_layout()
    save_figure(fig, "all_training_overview_0316")
    # Paper version
    style.apply("paper")
    fig2, axes2 = plt.subplots(2, 2, figsize=style.figsize("wide"))
    # Simplified for paper — just eval loss + reward trajectories
    ax = axes2[0, 0]
    for exp_name, d in sft_data.items():
        df = d["df"]
        evals = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
        if evals.empty:
            continue
        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"
        ax.plot(evals["step"], evals["eval_loss"],
                color=color, linewidth=0.8, marker="o", markersize=1.5, linestyle=ls,
                label=d["label"])
    ax.set_xlabel(get_label("step")); ax.set_ylabel(get_label("eval_loss"))
    ax.legend(fontsize=4); style.clean_axes(ax)

    ax = axes2[0, 1]
    x = np.arange(len(names))
    ax.bar(x, rewards, color=bcolors, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=4)
    ax.set_ylabel("Final Reward"); style.clean_axes(ax)

    ax = axes2[1, 0]
    for exp_name, d in sft_data.items():
        df = d["df"]
        train = df[df["grad_norm"].notna()]
        if "eval_loss" in df.columns:
            train = train[train["eval_loss"].isna()]
        color = style.color(d["approach"])
        ls = "--" if "Thinking" in d["label"] else "-"
        smoothed = smooth(train["grad_norm"].values, window=20)
        ax.plot(train["step"], smoothed, color=color, linewidth=0.8, linestyle=ls, label=d["label"])
    ax.set_yscale("log"); ax.set_xlabel(get_label("step")); ax.set_ylabel(get_label("grad_norm"))
    ax.legend(fontsize=4); style.clean_axes(ax)

    ax = axes2[1, 1]
    seen2 = {}
    for name2, d in sorted(grpo_data.items(), key=lambda x: -len(x[1]["train"])):
        key = f"{d['approach']}_{d['task']}"
        if key in seen2:
            continue
        seen2[key] = True
        df = d["train"]
        color = style.color(d["approach"])
        ls = "--" if d["approach"] == "text" else "-"
        smoothed = smooth(df["reward"].values, window=5) if len(df) > 5 else df["reward"].values
        ax.plot(df["step"], smoothed, color=color, linestyle=ls, linewidth=0.8,
                label=f"{get_approach_label(d['approach'])}: {d['task']}")
    ax.set_xlabel(get_label("step")); ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=4); style.clean_axes(ax)
    fig2.tight_layout()
    save_figure(fig2, "all_training_overview_0316", target="paper")
    style.apply("blog")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Loading SFT experiment data...")
    print("=" * 60)
    sft_data = load_sft_data()

    print("\n" + "=" * 60)
    print("Loading GRPO experiment data...")
    print("=" * 60)
    grpo_data = load_grpo_data()

    print(f"\nLoaded {len(sft_data)} SFT + {len(grpo_data)} GRPO experiments")

    print("\n" + "=" * 60)
    print("Generating SFT plots...")
    print("=" * 60)

    print("\n[1/12] SFT Loss Curves")
    plot_sft_loss_curves(sft_data)

    print("\n[2/12] SFT Eval Loss")
    plot_sft_eval_loss(sft_data)

    print("\n[3/12] SFT Gradient Norms")
    plot_sft_gradient_norms(sft_data)

    print("\n[4/12] SFT LR Schedule")
    plot_sft_lr_schedule(sft_data)

    print("\n[5/12] SFT Loss Comparison Bar")
    plot_sft_loss_comparison_bar(sft_data)

    print("\n" + "=" * 60)
    print("Generating GRPO plots...")
    print("=" * 60)

    print("\n[6/12] GRPO Reward Curves")
    plot_grpo_reward_curves(grpo_data)

    print("\n[7/12] GRPO Gradient Norms")
    plot_grpo_gradient_norms(grpo_data)

    print("\n[8/12] GRPO Eval Reward")
    plot_grpo_eval_reward(grpo_data)

    print("\n[9/12] GRPO Format Compliance")
    plot_grpo_format_compliance(grpo_data)

    print("\n[10/12] GRPO Final Reward Bar")
    plot_grpo_final_reward_bar(grpo_data)

    print("\n[11/12] GRPO Step-Level Reward")
    plot_grpo_step_level_reward(grpo_data)

    print("\n[12/12] Training Overview (4-panel)")
    plot_training_overview(sft_data, grpo_data)

    print("\n" + "=" * 60)
    print("ALL DONE — 12 figures generated (blog + paper)")
    print("=" * 60)


if __name__ == "__main__":
    main()
