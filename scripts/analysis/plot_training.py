#!/usr/bin/env python3
"""
Reusable training analysis and plotting tool.

Reads experiment data from multiple sources (trainer_state.json, metrics.json,
run_histories CSV, generation eval JSONs) and produces standardized diagnostic
plots for one or more experiments.

Usage:
    # Single experiment
    python scripts/analysis/plot_training.py \
        --experiments sft_esm3_mlp_combined_qwen3_8b_it_0306_010408 \
        --output blog/figures/supple_figures

    # Multiple experiments for comparison
    python scripts/analysis/plot_training.py \
        --experiments exp1 exp2 exp3 \
        --output blog/figures/supple_figures

    # With specific plots only
    python scripts/analysis/plot_training.py \
        --experiments exp1 --output blog/figures/supple_figures \
        --plots train_loss eval_loss generation_quality

    # Paper output (PDF + PNG)
    python scripts/analysis/plot_training.py \
        --experiments exp1 \
        --output blog/figures/supple_figures \
        --paper-output paper/figures/supplementary \
        --style blog paper
"""

import os
import sys

# Ensure this directory is on sys.path for figure_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import (
    DATA_DIR,
    RESULTS_DIR,
    SUPPLE_FIGURES_DIR,
    annotate_bars,
    annotate_best,
    get_approach_label,
    get_label,
    smooth,
    style,
    to_rgba_alpha,
)

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

class ExperimentData:
    """Loads and holds data for a single experiment from multiple sources."""

    def __init__(self, name, results_dir=None, data_dir=None):
        self.name = name
        self.results_path = os.path.join(
            str(results_dir or RESULTS_DIR), name)
        self.data_dir = str(data_dir or DATA_DIR)
        self.approach = "unknown"
        self.projector_type = None
        self.base_model = None
        self.max_steps = None
        self.learning_rate = None
        self.projector_lr = None

        # DataFrames
        self.train_df = pd.DataFrame()  # step, token_avg_loss, loss, grad_norm, lr, epoch
        self.eval_df = pd.DataFrame()   # step, eval_loss
        self.gen_df = pd.DataFrame()    # step, bleu, rouge_l

        self._load()

    def _load(self):
        """Try loading from trainer_state.json first, fall back to CSV."""
        loaded = self._try_trainer_state()
        if not loaded:
            loaded = self._try_csv()
        if not loaded:
            print(f"  WARNING: No training data found for {self.name}")

        self._load_metadata()
        self._load_generation_metrics()

    def _try_trainer_state(self):
        """Load from trainer_state.json in the latest checkpoint."""
        checkpoints = sorted(glob.glob(
            os.path.join(self.results_path, "checkpoints", "checkpoint-*", "trainer_state.json")
        ))
        if not checkpoints:
            return False

        state_path = checkpoints[-1]
        print(f"  Loading trainer_state from: {os.path.basename(os.path.dirname(state_path))}")

        with open(state_path) as f:
            state = json.load(f)

        self.max_steps = state.get("max_steps")

        log_history = state.get("log_history", [])
        if not log_history:
            return False

        train_rows = []
        eval_rows = []

        for entry in log_history:
            step = entry.get("step", 0)
            if "eval_loss" in entry:
                eval_rows.append({
                    "step": step,
                    "eval_loss": entry["eval_loss"],
                    "epoch": entry.get("epoch"),
                })
            if "token_avg_loss" in entry or "loss" in entry:
                train_rows.append({
                    "step": step,
                    "token_avg_loss": entry.get("token_avg_loss"),
                    "loss": entry.get("loss"),
                    "grad_norm": entry.get("grad_norm"),
                    "learning_rate": entry.get("learning_rate"),
                    "epoch": entry.get("epoch"),
                    "total_tokens": entry.get("total_tokens"),
                })

        if train_rows:
            self.train_df = pd.DataFrame(train_rows).sort_values("step").reset_index(drop=True)
        if eval_rows:
            self.eval_df = pd.DataFrame(eval_rows).sort_values("step").reset_index(drop=True)

        return len(train_rows) > 0

    def _try_csv(self):
        """Load from run_histories.csv in blog/data/."""
        date_dirs = sorted(glob.glob(os.path.join(self.data_dir, "*")))
        for date_dir in reversed(date_dirs):
            csv_path = os.path.join(date_dir, "run_histories.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                exp_df = df[df["experiment"] == self.name]
                if len(exp_df) == 0:
                    continue

                print(f"  Loading from CSV: {csv_path}")

                train_mask = exp_df["token_avg_loss"].notna()
                eval_mask = exp_df["eval_loss"].notna()

                if train_mask.any():
                    self.train_df = exp_df[train_mask][[
                        "step", "token_avg_loss", "loss", "grad_norm",
                        "learning_rate", "epoch"
                    ]].copy().sort_values("step").reset_index(drop=True)

                if eval_mask.any():
                    self.eval_df = exp_df[eval_mask][[
                        "step", "eval_loss", "epoch"
                    ]].copy().sort_values("step").reset_index(drop=True)

                return True
        return False

    def _load_metadata(self):
        """Load experiment metadata from lineage.json or config.yaml."""
        lineage_path = os.path.join(self.results_path, "lineage.json")
        if os.path.exists(lineage_path):
            with open(lineage_path) as f:
                lineage = json.load(f)
            self.approach = lineage.get("approach", self._infer_approach())
            self.projector_type = lineage.get("projector_type")
            self.base_model = lineage.get("base_model")
        else:
            self.approach = self._infer_approach()

        args_path = os.path.join(self.results_path, "training_args.json")
        if os.path.exists(args_path):
            with open(args_path) as f:
                args = json.load(f)
            self.learning_rate = args.get("learning_rate")
            self.projector_lr = args.get("projector_lr")

        for date_dir in sorted(glob.glob(os.path.join(self.data_dir, "*")), reverse=True):
            meta_path = os.path.join(date_dir, "experiment_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                for exp in meta.get("experiments", []):
                    if exp["name"] == self.name:
                        if self.approach == "unknown":
                            self.approach = exp.get("approach", "unknown")
                        if not self.projector_type:
                            self.projector_type = exp.get("projector_type")
                        if not self.base_model:
                            self.base_model = exp.get("base_model")
                        if not self.max_steps:
                            self.max_steps = exp.get("max_steps") or exp.get("total_steps")
                        if not self.learning_rate:
                            self.learning_rate = exp.get("learning_rate")
                        if not self.projector_lr:
                            self.projector_lr = exp.get("projector_lr")
                        break

    def _load_generation_metrics(self):
        """Load generation quality (BLEU/ROUGE-L) from eval/ directory."""
        eval_dir = os.path.join(self.results_path, "eval")
        gen_files = sorted(glob.glob(os.path.join(eval_dir, "generation_step_*.json")))

        if not gen_files:
            return

        rows = []
        for f in gen_files:
            with open(f) as fh:
                d = json.load(fh)
            rows.append({
                "step": d["step"],
                "bleu": d.get("bleu", 0),
                "rouge_l": d.get("rouge_l", 0),
                "num_samples": d.get("num_samples", 0),
            })

        if rows:
            self.gen_df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
            print(f"  Loaded {len(rows)} generation eval checkpoints")

    def _infer_approach(self):
        """Infer approach from experiment name."""
        name = self.name.lower()
        if "text" in name:
            return "text"
        if "perceiver" in name:
            return "perceiver"
        if "flamingo" in name:
            return "flamingo"
        if "esm3" in name or "mlp" in name:
            return "mlp"
        return "unknown"

    @property
    def label(self):
        """Short label for legends."""
        parts = [get_approach_label(self.approach)]
        if self.base_model:
            model_short = self.base_model.split("/")[-1].replace("Instruct-2507", "").replace("-", "")
            parts.append(model_short)
        return " ".join(parts)

    @property
    def latest_step(self):
        if len(self.train_df) > 0:
            return int(self.train_df["step"].max())
        return 0

    @property
    def final_token_avg_loss(self):
        if len(self.train_df) > 0 and "token_avg_loss" in self.train_df:
            return float(self.train_df["token_avg_loss"].iloc[-1])
        return None

    @property
    def best_eval_loss(self):
        if len(self.eval_df) > 0:
            return float(self.eval_df["eval_loss"].min())
        return None

    @property
    def best_eval_step(self):
        if len(self.eval_df) > 0:
            idx = self.eval_df["eval_loss"].idxmin()
            return int(self.eval_df.loc[idx, "step"])
        return None

    def summary(self):
        s = f"  {self.name}\n"
        s += f"    Approach: {self.approach}, Model: {self.base_model or 'unknown'}\n"
        s += f"    Train points: {len(self.train_df)}, Eval points: {len(self.eval_df)}, Gen evals: {len(self.gen_df)}\n"
        s += f"    Steps: {self.latest_step}"
        if self.max_steps:
            pct = self.latest_step / self.max_steps * 100
            s += f" / {self.max_steps} ({pct:.1f}%)"
        s += "\n"
        if self.final_token_avg_loss is not None:
            s += f"    Final token_avg_loss: {self.final_token_avg_loss:.4f}\n"
        if self.best_eval_loss is not None:
            s += f"    Best eval_loss: {self.best_eval_loss:.4f} (step {self.best_eval_step})\n"
        if len(self.gen_df) > 0:
            latest_gen = self.gen_df.iloc[-1]
            s += f"    Latest gen: BLEU={latest_gen['bleu']:.4f}, ROUGE-L={latest_gen['rouge_l']:.4f} (step {int(latest_gen['step'])})\n"
        return s


# ═══════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def plot_train_loss(experiments, figsize=None, save_path=None, fmt="png",
                    smoothing_window=50):
    """Training loss curves (token_avg_loss) for all experiments."""
    fig, ax = plt.subplots(figsize=figsize or style.figsize())

    for exp in experiments:
        if len(exp.train_df) == 0 or "token_avg_loss" not in exp.train_df:
            continue
        df = exp.train_df.dropna(subset=["token_avg_loss"])
        color = style.color(exp.approach)

        if len(df) > smoothing_window:
            ax.plot(df["step"], df["token_avg_loss"],
                    color=color, linewidth=0.3, alpha=0.15)
            smoothed = smooth(df["token_avg_loss"], smoothing_window)
            ax.plot(df["step"], smoothed,
                    color=color, linewidth=1.8, alpha=0.9,
                    label=f"{exp.label} (MA-{smoothing_window})")
        else:
            ax.plot(df["step"], df["token_avg_loss"],
                    color=color, linewidth=1.5, alpha=0.85,
                    marker="o", markersize=3, label=exp.label)

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("token_avg_loss"))
    ax.set_title("Training Loss")
    ax.legend()
    style.clean_axes(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format=fmt, dpi=style.dpi(), bbox_inches="tight")
        plt.close()
        print(f"  OK {os.path.basename(save_path)}")
    return fig, ax


def plot_eval_loss(experiments, figsize=None, save_path=None, fmt="png"):
    """Eval loss curves for all experiments."""
    fig, ax = plt.subplots(figsize=figsize or style.figsize())

    has_data = False
    for exp in experiments:
        if len(exp.eval_df) == 0:
            continue
        has_data = True
        df = exp.eval_df.sort_values("step")
        color = style.color(exp.approach)
        ax.plot(df["step"], df["eval_loss"],
                color=color, marker="o", markersize=4,
                linewidth=1.8, alpha=0.85, label=exp.label)

        best_idx = df["eval_loss"].idxmin()
        best_step = df.loc[best_idx, "step"]
        best_val = df.loc[best_idx, "eval_loss"]
        annotate_best(ax, best_step, best_val, best_val, color)

    if not has_data:
        plt.close()
        if save_path:
            print(f"  SKIP {os.path.basename(save_path)} (no eval data)")
        return None, None

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("eval_loss"))
    ax.set_title("Validation Loss")
    ax.legend()
    style.clean_axes(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()
        print(f"  OK {os.path.basename(save_path)}")
    return fig, ax


def plot_generation_quality(experiments, figsize=None, save_path=None, fmt="png"):
    """BLEU and ROUGE-L over training steps."""
    fig, ax = plt.subplots(figsize=figsize or style.figsize())

    has_data = False
    for exp in experiments:
        if len(exp.gen_df) == 0:
            continue
        has_data = True
        df = exp.gen_df.sort_values("step")
        color = style.color(exp.approach)

        ax.plot(df["step"], df["bleu"],
                color=color, linewidth=1.5, alpha=0.85,
                marker="o", markersize=3, label=f"{exp.label} {get_label('bleu')}")
        ax.plot(df["step"], df["rouge_l"],
                color=color, linewidth=1.5, alpha=0.85,
                linestyle="--", marker="s", markersize=3,
                label=f"{exp.label} {get_label('rouge_l')}")

    if not has_data:
        plt.close()
        if save_path:
            print(f"  SKIP {os.path.basename(save_path)} (no generation data)")
        return None, None

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Score")
    ax.set_title("Generation Quality Over Training")
    ax.legend(loc="lower right")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    style.clean_axes(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()
        print(f"  OK {os.path.basename(save_path)}")
    return fig, ax


def plot_lr_schedule(experiments, figsize=None, save_path=None, fmt="png"):
    """Learning rate schedule over training."""
    fig, ax = plt.subplots(figsize=figsize or style.figsize())

    has_data = False
    for exp in experiments:
        if len(exp.train_df) == 0 or "learning_rate" not in exp.train_df:
            continue
        df = exp.train_df.dropna(subset=["learning_rate"])
        if len(df) == 0:
            continue
        has_data = True
        color = style.color(exp.approach)
        ax.plot(df["step"], df["learning_rate"],
                color=color, linewidth=1.5, alpha=0.85, label=exp.label)

    if not has_data:
        plt.close()
        if save_path:
            print(f"  SKIP {os.path.basename(save_path)} (no LR data)")
        return None, None

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(get_label("learning_rate"))
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    style.clean_axes(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()
        print(f"  OK {os.path.basename(save_path)}")
    return fig, ax


def plot_grad_norms(experiments, figsize=None, save_path=None, fmt="png",
                    smoothing_window=30):
    """Gradient norms (log scale) with anomaly detection."""
    fig, ax = plt.subplots(figsize=figsize or style.figsize())

    has_data = False
    for exp in experiments:
        if len(exp.train_df) == 0 or "grad_norm" not in exp.train_df:
            continue
        df = exp.train_df.dropna(subset=["grad_norm"])
        if len(df) == 0:
            continue
        has_data = True
        color = style.color(exp.approach)

        ax.plot(df["step"], df["grad_norm"],
                color=color, linewidth=0.4, alpha=0.3)

        if len(df) > smoothing_window:
            smoothed = smooth(df["grad_norm"], smoothing_window)
            ax.plot(df["step"], smoothed,
                    color=color, linewidth=1.5, alpha=0.9,
                    label=f"{exp.label} (MA-{smoothing_window})")

        mean_gn = df["grad_norm"].mean()
        ax.axhline(mean_gn, color=color, linestyle="--", alpha=0.4,
                   label=f"Mean: {mean_gn:.3f}")

        std_gn = df["grad_norm"].std()
        spikes = df[df["grad_norm"] > mean_gn + 3 * std_gn]
        if len(spikes) > 0:
            ax.scatter(spikes["step"], spikes["grad_norm"],
                       color="#E74C3C", marker="x", s=30, zorder=5, alpha=0.7,
                       label=f"Spikes ({len(spikes)})")

    if not has_data:
        plt.close()
        if save_path:
            print(f"  SKIP {os.path.basename(save_path)} (no grad norm data)")
        return None, None

    ax.set_yscale("log")
    ax.set_xlabel(get_label("step"))
    ax.set_ylabel(f"{get_label('grad_norm')} (log)")
    ax.set_title("Gradient Norms")
    ax.legend()
    style.clean_axes(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()
        print(f"  OK {os.path.basename(save_path)}")
    return fig, ax


def plot_train_eval_combined(experiments, figsize=None, save_path=None, fmt="png",
                             smoothing_window=50):
    """Combined train + eval loss on same axes for each experiment."""
    fig, ax = plt.subplots(figsize=figsize or style.figsize())

    for exp in experiments:
        color = style.color(exp.approach)

        if len(exp.train_df) > 0 and "token_avg_loss" in exp.train_df:
            df = exp.train_df.dropna(subset=["token_avg_loss"])
            if len(df) > smoothing_window:
                ax.plot(df["step"], df["token_avg_loss"],
                        color=color, linewidth=0.3, alpha=0.12)
                smoothed = smooth(df["token_avg_loss"], smoothing_window)
                ax.plot(df["step"], smoothed,
                        color=color, linewidth=1.5, alpha=0.8,
                        label=f"{exp.label} Train")
            else:
                ax.plot(df["step"], df["token_avg_loss"],
                        color=color, linewidth=1.2, alpha=0.7,
                        label=f"{exp.label} Train")

        if len(exp.eval_df) > 0:
            df = exp.eval_df.sort_values("step")
            ax.plot(df["step"], df["eval_loss"],
                    color=color, marker="D", markersize=5, linewidth=0,
                    zorder=5, alpha=0.9, label=f"{exp.label} Eval")

    ax.set_xlabel(get_label("step"))
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    style.clean_axes(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()
        print(f"  OK {os.path.basename(save_path)}")
    return fig, ax


def plot_comparison_bar(experiments, figsize=None, save_path=None, fmt="png"):
    """Bar chart comparing final train loss and best eval loss across experiments."""
    names = []
    train_vals = []
    eval_vals = []
    bar_colors = []

    for exp in experiments:
        if exp.final_token_avg_loss is None and exp.best_eval_loss is None:
            continue
        names.append(exp.label)
        train_vals.append(exp.final_token_avg_loss or 0)
        eval_vals.append(exp.best_eval_loss or 0)
        bar_colors.append(style.color(exp.approach))

    if not names:
        if save_path:
            print(f"  SKIP {os.path.basename(save_path)} (no metrics)")
        return None, None

    fig, ax = plt.subplots(figsize=figsize or style.figsize())
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_vals, width,
                   label=get_label("final_train_loss"),
                   color=[to_rgba_alpha(c, 0.5) for c in bar_colors],
                   edgecolor=bar_colors, linewidth=1.2)
    bars2 = ax.bar(x + width/2, eval_vals, width,
                   label=get_label("best_eval_loss"),
                   color=bar_colors, edgecolor="white", linewidth=0.5)

    annotate_bars(ax, bars2, eval_vals)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=plt.rcParams['font.size'] - 1)
    ax.set_ylabel("Loss")
    ax.set_title("Final Metrics Comparison")
    ax.legend()
    ax.set_ylim(0, max(max(train_vals), max(eval_vals)) * 1.3)
    style.clean_axes(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()
        print(f"  OK {os.path.basename(save_path)}")
    return fig, ax


def plot_convergence_table(experiments, figsize=None, save_path=None, fmt="png"):
    """Summary table rendered as image."""
    headers = ["Experiment", "Approach", "Steps", get_label("train_loss"),
               get_label("eval_loss"), get_label("bleu"), get_label("rouge_l")]
    rows = []
    approach_order = []

    for exp in experiments:
        step_str = str(exp.latest_step)
        if exp.max_steps:
            pct = exp.latest_step / exp.max_steps * 100
            step_str += f" ({pct:.0f}%)"

        train_loss = f"{exp.final_token_avg_loss:.4f}" if exp.final_token_avg_loss else "\u2014"
        eval_loss = f"{exp.best_eval_loss:.4f}" if exp.best_eval_loss else "\u2014"

        bleu = "\u2014"
        rouge = "\u2014"
        if len(exp.gen_df) > 0:
            latest = exp.gen_df.iloc[-1]
            bleu = f"{latest['bleu']:.3f}"
            rouge = f"{latest['rouge_l']:.3f}"

        rows.append([exp.label, exp.approach.upper(), step_str,
                     train_loss, eval_loss, bleu, rouge])
        approach_order.append(exp.approach)

    fig, ax = plt.subplots(figsize=figsize or style.figsize("wide"))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    style.style_table(table, approach_order=approach_order,
                      n_rows=len(rows), n_cols=len(headers))

    ax.set_title("Experiment Summary",
                 fontsize=plt.rcParams.get('axes.titlesize', 13), pad=15)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()
        print(f"  OK {os.path.basename(save_path)}")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def compute_convergence_step(df, window=10, threshold=0.01):
    """Find step where rolling mean change drops below threshold."""
    if "token_avg_loss" not in df.columns or len(df) < window:
        return -1
    rolling = df["token_avg_loss"].rolling(window).mean()
    pct_change = rolling.pct_change().abs()
    converged = pct_change[pct_change < threshold]
    if len(converged) > 0:
        return int(df.iloc[converged.index[0]]["step"])
    return -1


def detect_anomalies(df):
    """Detect training anomalies: NaN, spikes, divergence."""
    anomalies = []
    if "token_avg_loss" in df.columns:
        if df["token_avg_loss"].isna().any():
            nan_steps = df[df["token_avg_loss"].isna()]["step"].tolist()
            anomalies.append(f"nan_loss_steps_{nan_steps}")
        mean = df["token_avg_loss"].mean()
        std = df["token_avg_loss"].std()
        if std > 0:
            spikes = df[df["token_avg_loss"] > mean + 3 * std]
            for _, row in spikes.iterrows():
                anomalies.append(f"loss_spike_step_{int(row['step'])}")
    if "grad_norm" in df.columns:
        gn = df["grad_norm"].dropna()
        if len(gn) > 0:
            mean = gn.mean()
            std = gn.std()
            if std > 0:
                spikes = df[df["grad_norm"] > mean + 3 * std]
                for _, row in spikes.iterrows():
                    anomalies.append(f"grad_spike_step_{int(row['step'])}")
    return anomalies


def generate_analysis_summary(experiments, output_path):
    """Generate analysis_summary.json."""
    summary = {"experiments": {}, "comparison": {}}

    for exp in experiments:
        exp_summary = {
            "approach": exp.approach,
            "projector_type": exp.projector_type,
            "base_model": exp.base_model,
            "total_steps": exp.latest_step,
            "max_steps": exp.max_steps,
            "final_token_avg_loss": exp.final_token_avg_loss,
            "min_token_avg_loss": float(exp.train_df["token_avg_loss"].min()) if len(exp.train_df) > 0 and "token_avg_loss" in exp.train_df else None,
            "best_eval_loss": exp.best_eval_loss,
            "best_eval_step": exp.best_eval_step,
            "convergence_step": compute_convergence_step(exp.train_df),
            "anomalies": detect_anomalies(exp.train_df),
        }

        if len(exp.gen_df) > 0:
            latest = exp.gen_df.iloc[-1]
            exp_summary["latest_bleu"] = float(latest["bleu"])
            exp_summary["latest_rouge_l"] = float(latest["rouge_l"])
            exp_summary["best_bleu"] = float(exp.gen_df["bleu"].max())
            exp_summary["best_rouge_l"] = float(exp.gen_df["rouge_l"].max())

        if len(exp.train_df) > 0 and "grad_norm" in exp.train_df:
            gn = exp.train_df["grad_norm"].dropna()
            if len(gn) > 0:
                exp_summary["max_grad_norm"] = float(gn.max())
                exp_summary["mean_grad_norm"] = float(gn.mean())

        summary["experiments"][exp.name] = exp_summary

    eval_losses = {name: s.get("best_eval_loss") for name, s in summary["experiments"].items()
                   if s.get("best_eval_loss") is not None}
    if eval_losses:
        best_name = min(eval_losses, key=eval_losses.get)
        summary["comparison"] = {
            "best_experiment": best_name,
            "metric": "best_eval_loss",
            "value": eval_losses[best_name],
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  OK {os.path.basename(output_path)}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

ALL_PLOTS = [
    "train_loss", "eval_loss", "train_eval_combined",
    "generation_quality", "lr_schedule", "grad_norms",
    "comparison_bar", "convergence_table",
]

def main():
    parser = argparse.ArgumentParser(
        description="Generate training analysis plots for protein-LLM experiments."
    )
    parser.add_argument("--experiments", "-e", nargs="+", required=True,
                        help="Experiment names (directories under results/)")
    parser.add_argument("--output", "-o", default=str(SUPPLE_FIGURES_DIR),
                        help="Output directory for blog-style figures")
    parser.add_argument("--paper-output", default=None,
                        help="Output directory for paper-style figures (PDF+PNG)")
    parser.add_argument("--plots", nargs="+", default=ALL_PLOTS,
                        choices=ALL_PLOTS,
                        help="Which plots to generate (default: all)")
    parser.add_argument("--prefix", default="",
                        help="Filename prefix for output figures")
    parser.add_argument("--smoothing", type=int, default=50,
                        help="Moving average window for loss smoothing")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="Path to results/ directory")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="Path to blog/data/ directory")
    parser.add_argument("--summary-output", default=None,
                        help="Path for analysis_summary.json")

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    paper_dir = None
    if args.paper_output:
        paper_dir = os.path.abspath(args.paper_output)
        os.makedirs(paper_dir, exist_ok=True)

    # Load experiments
    print("=" * 60)
    print("LOADING EXPERIMENTS")
    print("=" * 60)
    experiments = []
    for name in args.experiments:
        print(f"\n[{name}]")
        exp = ExperimentData(name, results_dir=args.results_dir, data_dir=args.data_dir)
        experiments.append(exp)
        print(exp.summary())

    if not experiments:
        print("ERROR: No experiments loaded.")
        sys.exit(1)

    pfx = f"{args.prefix}_" if args.prefix else ""

    # Blog figures
    print("\n" + "=" * 60)
    print("GENERATING BLOG FIGURES")
    print("=" * 60)
    style.apply("blog")

    blog_figsize = style.figsize()
    blog_wide = style.figsize("wide")

    plot_dispatch = {
        "train_loss": lambda: plot_train_loss(
            experiments, blog_figsize,
            f"{output_dir}/{pfx}train_loss.png", smoothing_window=args.smoothing),
        "eval_loss": lambda: plot_eval_loss(
            experiments, blog_figsize,
            f"{output_dir}/{pfx}eval_loss.png"),
        "train_eval_combined": lambda: plot_train_eval_combined(
            experiments, blog_figsize,
            f"{output_dir}/{pfx}train_eval_combined.png", smoothing_window=args.smoothing),
        "generation_quality": lambda: plot_generation_quality(
            experiments, blog_figsize,
            f"{output_dir}/{pfx}generation_quality.png"),
        "lr_schedule": lambda: plot_lr_schedule(
            experiments, blog_figsize,
            f"{output_dir}/{pfx}lr_schedule.png"),
        "grad_norms": lambda: plot_grad_norms(
            experiments, blog_figsize,
            f"{output_dir}/{pfx}grad_norms.png", smoothing_window=args.smoothing),
        "comparison_bar": lambda: plot_comparison_bar(
            experiments, blog_figsize,
            f"{output_dir}/{pfx}comparison_bar.png"),
        "convergence_table": lambda: plot_convergence_table(
            experiments, blog_wide,
            f"{output_dir}/{pfx}convergence_table.png"),
    }

    for plot_name in args.plots:
        if plot_name in plot_dispatch:
            plot_dispatch[plot_name]()

    # Paper figures
    if paper_dir:
        print("\n" + "=" * 60)
        print("GENERATING PAPER FIGURES")
        print("=" * 60)
        style.apply("paper")

        paper_figsize = style.figsize()
        paper_wide = style.figsize("wide")

        paper_dispatch = {
            "train_loss": lambda: plot_train_loss(
                experiments, paper_figsize,
                f"{paper_dir}/{pfx}train_loss.pdf", fmt="pdf", smoothing_window=args.smoothing),
            "eval_loss": lambda: plot_eval_loss(
                experiments, paper_figsize,
                f"{paper_dir}/{pfx}eval_loss.pdf", fmt="pdf"),
            "train_eval_combined": lambda: plot_train_eval_combined(
                experiments, paper_figsize,
                f"{paper_dir}/{pfx}train_eval_combined.pdf", fmt="pdf", smoothing_window=args.smoothing),
            "generation_quality": lambda: plot_generation_quality(
                experiments, paper_wide,
                f"{paper_dir}/{pfx}generation_quality.pdf", fmt="pdf"),
            "lr_schedule": lambda: plot_lr_schedule(
                experiments, paper_figsize,
                f"{paper_dir}/{pfx}lr_schedule.pdf", fmt="pdf"),
            "grad_norms": lambda: plot_grad_norms(
                experiments, paper_figsize,
                f"{paper_dir}/{pfx}grad_norms.pdf", fmt="pdf", smoothing_window=args.smoothing),
            "comparison_bar": lambda: plot_comparison_bar(
                experiments, paper_wide,
                f"{paper_dir}/{pfx}comparison_bar.pdf", fmt="pdf"),
            "convergence_table": lambda: plot_convergence_table(
                experiments, paper_wide,
                f"{paper_dir}/{pfx}convergence_table.pdf", fmt="pdf"),
        }

        for plot_name in args.plots:
            if plot_name in paper_dispatch:
                paper_dispatch[plot_name]()

    # Analysis summary
    summary_path = args.summary_output or os.path.join(output_dir, f"{pfx}analysis_summary.json")
    print("\nGenerating analysis summary...")
    generate_analysis_summary(experiments, summary_path)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Blog figures:  {output_dir}/{pfx}*.png")
    if paper_dir:
        print(f"Paper figures: {paper_dir}/{pfx}*.pdf")
    print(f"Summary:       {summary_path}")


if __name__ == "__main__":
    main()
