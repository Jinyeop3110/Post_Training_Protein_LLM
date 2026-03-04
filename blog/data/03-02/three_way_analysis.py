#!/usr/bin/env python3
"""Three-way SFT comparison analysis: MLP vs Perceiver vs Text-only.

Reads run_histories.csv and experiment_metadata.json from blog/data/03-02/,
produces diagnostic plots in blog/figures/ and analysis_summary.json in blog/data/03-02/.

Usage:
    source /home/yeopjin/orcd/pool/init_protein_llm.sh
    python blog/data/03-02/three_way_analysis.py
"""

import matplotlib

matplotlib.use('Agg')  # Headless rendering -- MUST be before pyplot import
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Standard style
sns.set_theme(style="whitegrid", palette="colorblind")

# Approach color scheme (consistent across all plots)
APPROACH_COLORS = {
    "mlp": "#1f77b4",        # Blue
    "perceiver": "#ff7f0e",  # Orange
    "text": "#2ca02c",       # Green
}

# Readable legend labels
APPROACH_LABELS = {
    "mlp": "MLP (esm3)",
    "perceiver": "Perceiver (esm3)",
    "text": "Text-only",
}

FIG_DPI = 150
FIG_SIZE = (10, 6)
FIG_SIZE_WIDE = (14, 6)
FIG_SIZE_TALL = (10, 8)

# Paths
BLOG_DIR = "/orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM/blog"
DATA_DIR = os.path.join(BLOG_DIR, "03-02")

# ---- Helper Functions ----

def shorten(exp_name: str, max_len: int = 30) -> str:
    """Shorten experiment name for legends."""
    if len(exp_name) <= max_len:
        return exp_name
    return exp_name[:max_len - 3] + "..."


def get_approach(exp_name: str, metadata: dict) -> str:
    """Get approach label (mlp/perceiver/text) from metadata."""
    for exp in metadata.get("experiments", []):
        if exp["name"] == exp_name:
            if exp.get("approach") == "text":
                return "text"
            proj = exp.get("projector_type", "unknown")
            return proj if proj else "unknown"
    # Fallback: parse from name
    if "text" in exp_name:
        return "text"
    return "unknown"


def compute_convergence_step(df: pd.DataFrame, window: int = 10, threshold: float = 0.01) -> int:
    """Find step where rolling mean change drops below threshold."""
    if "token_avg_loss" not in df.columns or len(df) < window:
        return -1
    # Reset index to avoid iloc/loc mismatch on filtered DataFrames
    df_reset = df.reset_index(drop=True)
    rolling = df_reset["token_avg_loss"].rolling(window).mean()
    pct_change = rolling.pct_change().abs()
    converged = pct_change[pct_change < threshold]
    if len(converged) > 0:
        return int(df_reset.loc[converged.index[0], "step"])
    return -1


def detect_anomalies(df: pd.DataFrame) -> list:
    """Detect training anomalies: NaN, spikes, divergence."""
    anomalies = []
    if "token_avg_loss" in df.columns:
        if df["token_avg_loss"].isna().any():
            nan_steps = df[df["token_avg_loss"].isna()]["step"].tolist()
            anomalies.append(f"nan_loss_steps_{nan_steps}")
        # Loss spikes > 3 std
        clean = df["token_avg_loss"].dropna()
        if len(clean) > 0:
            mean = clean.mean()
            std = clean.std()
            if std > 0:
                spikes = df[df["token_avg_loss"] > mean + 3 * std]
                for _, row in spikes.iterrows():
                    anomalies.append(f"loss_spike_step_{int(row['step'])}")
    if "grad_norm" in df.columns:
        clean_gn = df["grad_norm"].dropna()
        if clean_gn.isna().all() or len(clean_gn) == 0:
            pass
        else:
            if df["grad_norm"].isna().any():
                anomalies.append("nan_grad_norm_present")
            mean = clean_gn.mean()
            std = clean_gn.std()
            if std > 0:
                spikes = df[df["grad_norm"] > mean + 3 * std]
                for _, row in spikes.iterrows():
                    anomalies.append(f"grad_spike_step_{int(row['step'])}")
    return anomalies


# ---- Load Data ----

print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "run_histories.csv"))
with open(os.path.join(DATA_DIR, "experiment_metadata.json"), "r") as f:
    metadata = json.load(f)

print(f"  Total rows: {len(df)}")
print(f"  Experiments: {df['experiment'].unique().tolist()}")
print(f"  Columns: {df.columns.tolist()}")

# Separate training rows (have grad_norm) from eval-only rows (have eval_loss but no grad_norm)
# and summary rows (mostly NaN)
# Training rows: have token_avg_loss and (grad_norm or learning_rate)
# Eval-only rows: have eval_loss but no grad_norm

# Build experiment -> approach mapping
exp_approach = {}
for exp_name in df["experiment"].unique():
    exp_approach[exp_name] = get_approach(exp_name, metadata)

print(f"  Approach mapping: {exp_approach}")

# Split into training data and eval data
# Training rows: have grad_norm (non-NaN)
train_df = df[df["grad_norm"].notna()].copy()
# Eval rows: have eval_loss (non-NaN)
eval_df = df[df["eval_loss"].notna()].copy()

print(f"  Training rows: {len(train_df)}")
print(f"  Eval rows: {len(eval_df)}")

# ---- Plot 1: Training Loss Curves ----
print("\nPlot 1: Training Loss Curves...")
fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name, group in train_df.groupby("experiment"):
    approach = exp_approach[exp_name]
    color = APPROACH_COLORS.get(approach, "#333333")
    label = APPROACH_LABELS.get(approach, approach)
    sorted_group = group.sort_values("step")
    ax.plot(sorted_group["step"], sorted_group["token_avg_loss"],
            color=color, label=label, alpha=0.8, linewidth=1.5)

ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Token Average Loss", fontsize=12)
ax.set_title("Three-Way SFT Comparison: Training Loss", fontsize=14)
ax.legend(loc="upper right", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(BLOG_DIR, "three_way_loss_curves.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: three_way_loss_curves.png")

# ---- Plot 2: Eval Loss Curves ----
print("\nPlot 2: Eval Loss Curves...")
fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name, group in eval_df.groupby("experiment"):
    approach = exp_approach[exp_name]
    color = APPROACH_COLORS.get(approach, "#333333")
    label = APPROACH_LABELS.get(approach, approach)
    sorted_group = group.sort_values("step")
    ax.plot(sorted_group["step"], sorted_group["eval_loss"],
            color=color, label=label, alpha=0.8, linewidth=1.5,
            marker="o", markersize=6)

ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Eval Loss", fontsize=12)
ax.set_title("Three-Way SFT Comparison: Evaluation Loss", fontsize=14)
ax.legend(loc="upper right", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(BLOG_DIR, "three_way_eval_loss.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: three_way_eval_loss.png")

# ---- Plot 3: Gradient Norms (Log Scale) ----
print("\nPlot 3: Gradient Norms...")
fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name, group in train_df.groupby("experiment"):
    approach = exp_approach[exp_name]
    color = APPROACH_COLORS.get(approach, "#333333")
    label = APPROACH_LABELS.get(approach, approach)
    sorted_group = group.sort_values("step")
    gn = sorted_group["grad_norm"].dropna()
    steps = sorted_group.loc[gn.index, "step"]
    ax.plot(steps, gn, color=color, label=label, alpha=0.6, linewidth=0.8)

    # Flag anomalies: spikes > 3 std above mean
    mean_gn = gn.mean()
    std_gn = gn.std()
    if std_gn > 0:
        spike_mask = gn > mean_gn + 3 * std_gn
        if spike_mask.any():
            spike_steps = sorted_group.loc[gn[spike_mask].index, "step"]
            spike_vals = gn[spike_mask]
            ax.scatter(spike_steps, spike_vals, color="red", s=30, zorder=5,
                       marker="x", linewidths=1.5)

ax.set_yscale("log")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Gradient Norm (log scale)", fontsize=12)
ax.set_title("Three-Way SFT Comparison: Gradient Norms", fontsize=14)
ax.legend(loc="upper right", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(BLOG_DIR, "three_way_grad_norms.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: three_way_grad_norms.png")

# ---- Plot 4: Learning Rate Schedule ----
print("\nPlot 4: Learning Rate Schedule...")
fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name, group in train_df.groupby("experiment"):
    approach = exp_approach[exp_name]
    color = APPROACH_COLORS.get(approach, "#333333")
    label = APPROACH_LABELS.get(approach, approach)
    sorted_group = group.sort_values("step")
    lr = sorted_group["learning_rate"].dropna()
    steps = sorted_group.loc[lr.index, "step"]
    ax.plot(steps, lr, color=color, label=label, alpha=0.8, linewidth=1.5)

ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.set_title("Three-Way SFT Comparison: Learning Rate Schedule", fontsize=14)
ax.legend(loc="upper right", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(BLOG_DIR, "three_way_lr_schedule.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: three_way_lr_schedule.png")

# ---- Plot 5: Final Metrics Bar Chart ----
print("\nPlot 5: Final Metrics Bar Chart...")

# Get final metrics from metadata
approaches = []
final_train_losses = []
best_eval_losses = []
labels = []

for exp in metadata["experiments"]:
    approach = get_approach(exp["name"], metadata)
    approaches.append(approach)
    final_train_losses.append(exp["final_token_avg_loss"])
    best_eval_losses.append(exp["best_eval_loss"])
    labels.append(APPROACH_LABELS.get(approach, approach))

x = np.arange(len(approaches))
width = 0.35

fig, ax = plt.subplots(figsize=FIG_SIZE)
colors = [APPROACH_COLORS.get(a, "#333333") for a in approaches]

bars1 = ax.bar(x - width / 2, final_train_losses, width, label="Final Train Loss",
               color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width / 2, best_eval_losses, width, label="Best Eval Loss",
               color=colors, alpha=1.0, edgecolor="black", linewidth=0.5)

# Annotate bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Three-Way SFT Comparison: Final Metrics", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, max(max(final_train_losses), max(best_eval_losses)) * 1.2)
fig.tight_layout()
fig.savefig(os.path.join(BLOG_DIR, "three_way_final_metrics.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: three_way_final_metrics.png")

# ---- Plot 6: Convergence Table ----
print("\nPlot 6: Convergence Table...")

headers = [
    "Approach", "Total\nSteps", "Final Train\nLoss", "Best Eval\nLoss",
    "Best Eval\nStep", "Convergence\nStep", "Max Grad\nNorm", "Mean Grad\nNorm",
    "Runtime\n(hours)", "GPU Max\nAlloc (GB)"
]

table_data = []
for exp in metadata["experiments"]:
    exp_name = exp["name"]
    approach = get_approach(exp_name, metadata)
    label = APPROACH_LABELS.get(approach, approach)

    # Get training data for this experiment
    exp_train = train_df[train_df["experiment"] == exp_name].sort_values("step")
    conv_step = compute_convergence_step(exp_train)

    max_gn = float(exp_train["grad_norm"].max()) if "grad_norm" in exp_train and not exp_train["grad_norm"].isna().all() else None
    mean_gn = float(exp_train["grad_norm"].mean()) if "grad_norm" in exp_train and not exp_train["grad_norm"].isna().all() else None

    row = [
        label,
        str(exp["total_steps"]),
        f"{exp['final_token_avg_loss']:.4f}",
        f"{exp['best_eval_loss']:.4f}",
        str(exp.get("best_eval_step", "N/A")),
        str(conv_step) if conv_step > 0 else "N/A",
        f"{max_gn:.2f}" if max_gn is not None else "N/A",
        f"{mean_gn:.2f}" if mean_gn is not None else "N/A",
        f"{exp.get('train_runtime_hours', 0):.1f}",
        f"{exp.get('gpu_memory_max_allocated_gb', 0):.1f}",
    ]
    table_data.append(row)

fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
ax.axis("off")
ax.set_title("Three-Way SFT Comparison: Convergence Summary", fontsize=14, pad=20)

table = ax.table(
    cellText=table_data,
    colLabels=headers,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

# Color the approach column cells
for i, row in enumerate(table_data):
    approach = list(APPROACH_LABELS.keys())[list(APPROACH_LABELS.values()).index(row[0])]
    color = APPROACH_COLORS.get(approach, "#cccccc")
    table[i + 1, 0].set_facecolor(color)
    table[i + 1, 0].set_text_props(color="white", fontweight="bold")

# Style header
for j in range(len(headers)):
    table[0, j].set_facecolor("#404040")
    table[0, j].set_text_props(color="white", fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(BLOG_DIR, "three_way_convergence_table.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: three_way_convergence_table.png")

# ---- Analysis Summary JSON ----
print("\nGenerating analysis_summary.json...")

summary = {
    "experiments": {},
    "comparison": {}
}

for exp in metadata["experiments"]:
    exp_name = exp["name"]
    approach = get_approach(exp_name, metadata)

    # Training data
    exp_train = train_df[train_df["experiment"] == exp_name].sort_values("step")
    # Eval data
    exp_eval = eval_df[eval_df["experiment"] == exp_name].sort_values("step")

    max_gn = float(exp_train["grad_norm"].max()) if not exp_train["grad_norm"].isna().all() else None
    mean_gn = float(exp_train["grad_norm"].mean()) if not exp_train["grad_norm"].isna().all() else None

    best_eval_loss = None
    best_eval_step = None
    if len(exp_eval) > 0 and not exp_eval["eval_loss"].isna().all():
        best_eval_loss = float(exp_eval["eval_loss"].min())
        best_eval_step = int(exp_eval.loc[exp_eval["eval_loss"].idxmin(), "step"])

    summary["experiments"][exp_name] = {
        "approach": approach,
        "projector_type": exp.get("projector_type"),
        "total_steps": exp["total_steps"],
        "final_token_avg_loss": exp["final_token_avg_loss"],
        "min_token_avg_loss": float(exp_train["token_avg_loss"].min()) if not exp_train["token_avg_loss"].isna().all() else None,
        "best_eval_loss": best_eval_loss,
        "best_eval_step": best_eval_step,
        "convergence_step": compute_convergence_step(exp_train),
        "max_grad_norm": max_gn,
        "mean_grad_norm": mean_gn,
        "train_runtime_hours": exp.get("train_runtime_hours"),
        "gpu_memory_max_allocated_gb": exp.get("gpu_memory_max_allocated_gb"),
        "anomalies": detect_anomalies(exp_train),
    }

# Find best experiment by best_eval_loss
best_exp_name = min(
    summary["experiments"].items(),
    key=lambda x: x[1].get("best_eval_loss", float("inf"))
)

# Compute relative improvements
mlp_eval = summary["experiments"].get("sft_lora_esm3_qwen3_8b_it_0226_151416", {}).get("best_eval_loss")
perc_eval = summary["experiments"].get("sft_lora_esm3_qwen3_8b_it_0225_203237", {}).get("best_eval_loss")
text_eval = summary["experiments"].get("sft_text_qwen3_8b_it_0227_145821", {}).get("best_eval_loss")

summary["comparison"] = {
    "best_experiment": best_exp_name[0],
    "best_approach": best_exp_name[1]["approach"],
    "metric": "best_eval_loss",
    "value": best_exp_name[1].get("best_eval_loss"),
    "mlp_vs_text_improvement_pct": round((text_eval - mlp_eval) / text_eval * 100, 2) if mlp_eval and text_eval else None,
    "perceiver_vs_text_improvement_pct": round((text_eval - perc_eval) / text_eval * 100, 2) if perc_eval and text_eval else None,
    "mlp_vs_perceiver_improvement_pct": round((perc_eval - mlp_eval) / perc_eval * 100, 2) if mlp_eval and perc_eval else None,
}

with open(os.path.join(DATA_DIR, "analysis_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("  Saved: 03-02/analysis_summary.json")

# ---- Print Summary ----
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nBest experiment: {best_exp_name[0]}")
print(f"  Approach: {best_exp_name[1]['approach']}")
print(f"  Best eval loss: {best_exp_name[1].get('best_eval_loss')}")
print("\nRelative improvements over Text-only:")
if summary["comparison"]["mlp_vs_text_improvement_pct"]:
    print(f"  MLP:       {summary['comparison']['mlp_vs_text_improvement_pct']:.1f}%")
if summary["comparison"]["perceiver_vs_text_improvement_pct"]:
    print(f"  Perceiver: {summary['comparison']['perceiver_vs_text_improvement_pct']:.1f}%")
if summary["comparison"]["mlp_vs_perceiver_improvement_pct"]:
    print(f"  MLP vs Perceiver: {summary['comparison']['mlp_vs_perceiver_improvement_pct']:.1f}%")

print("\nGenerated files:")
print(f"  {BLOG_DIR}/three_way_loss_curves.png")
print(f"  {BLOG_DIR}/three_way_eval_loss.png")
print(f"  {BLOG_DIR}/three_way_grad_norms.png")
print(f"  {BLOG_DIR}/three_way_lr_schedule.png")
print(f"  {BLOG_DIR}/three_way_final_metrics.png")
print(f"  {BLOG_DIR}/three_way_convergence_table.png")
print(f"  {DATA_DIR}/analysis_summary.json")
print(f"  {DATA_DIR}/three_way_analysis.py")
