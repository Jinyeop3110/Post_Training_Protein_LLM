#!/usr/bin/env python3
"""Analyst: Create diagnostic plots and analysis_summary.json for 03-04 report."""

import matplotlib

matplotlib.use('Agg')  # Headless — MUST be before pyplot import
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Standard style
sns.set_theme(style="whitegrid", palette="colorblind")

# Approach color scheme
APPROACH_COLORS = {
    "mlp": "#1f77b4",        # Blue
    "perceiver": "#ff7f0e",  # Orange
    "text": "#2ca02c",       # Green
}

FIG_DPI = 150
FIG_SIZE = (10, 6)
FIG_SIZE_WIDE = (14, 6)
FIG_SIZE_TALL = (10, 8)

# Paths
BASE = "/home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM"
FIG_DIR = f"{BASE}/blog/figures"
DATA_DIR_NEW = f"{BASE}/blog/data/03-04"
DATA_DIR_OLD = f"{BASE}/blog/data/03-02"

os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────
df_new = pd.read_csv(f"{DATA_DIR_NEW}/run_histories.csv")
df_old = pd.read_csv(f"{DATA_DIR_OLD}/run_histories.csv")

with open(f"{DATA_DIR_NEW}/experiment_metadata.json") as f:
    meta_new = json.load(f)
with open(f"{DATA_DIR_OLD}/experiment_metadata.json") as f:
    meta_old = json.load(f)

# Build metadata lookup
metadata = {}
for exp in meta_old["experiments"] + meta_new["experiments"]:
    metadata[exp["name"]] = exp

# Focus experiment (new long MLP run)
LONG_MLP = "sft_esm3_mlp_long_qwen3_8b_it_0302_175459"
df_long = df_new[df_new["experiment"] == LONG_MLP].copy()

# Key comparison experiments
COMPARISON_EXPS = {
    "sft_lora_esm3_qwen3_8b_it_0226_151416": ("mlp", "MLP (0226, combined)"),
    "sft_lora_esm3_qwen3_8b_it_0225_203237": ("perceiver", "Perceiver (0225, combined)"),
    "sft_text_qwen3_8b_it_0227_145821": ("text", "Text-only (0227, combined)"),
    LONG_MLP: ("mlp", "MLP-long (0302, mol_inst)"),
}


def shorten(name, max_len=30):
    return name if len(name) <= max_len else name[:max_len-3] + "..."


def get_approach(exp_name):
    m = metadata.get(exp_name, {})
    pt = m.get("projector_type")
    if pt == "perceiver":
        return "perceiver"
    approach = m.get("approach", "unknown")
    if approach == "text":
        return "text"
    return "mlp"


def compute_convergence_step(series, steps, window=10, threshold=0.01):
    if len(series) < window:
        return -1
    rolling = series.rolling(window).mean()
    pct_change = rolling.pct_change().abs()
    converged = pct_change[pct_change < threshold]
    if len(converged) > 0:
        idx = converged.index[0]
        return int(steps.iloc[steps.index.get_loc(idx)])
    return -1


def detect_anomalies(df_exp, warmup_steps=50):
    """Detect training anomalies, ignoring warmup period."""
    anomalies = []
    # Filter to post-warmup for spike detection
    df_post = df_exp[df_exp["step"] > warmup_steps]
    if "token_avg_loss" in df_post.columns:
        tal = df_post["token_avg_loss"].dropna()
        if len(tal) > 5:
            mean, std = tal.mean(), tal.std()
            spikes = df_post[df_post["token_avg_loss"] > mean + 3 * std]
            for _, row in spikes.iterrows():
                anomalies.append(f"loss_spike_step_{int(row['step'])}")
        # Check for NaN in ALL data (not just post-warmup)
        if df_exp["token_avg_loss"].isna().any():
            nan_steps = df_exp[df_exp["token_avg_loss"].isna()]["step"].tolist()
            # Only flag if NaN is in training rows (not eval-only rows)
            train_nan = df_exp[df_exp["token_avg_loss"].isna() & df_exp["eval_loss"].isna()]
            if len(train_nan) > 0:
                anomalies.append(f"nan_loss_steps_{[int(s) for s in train_nan['step'].tolist()]}")
    if "grad_norm" in df_post.columns:
        gn = df_post["grad_norm"].dropna()
        if len(gn) > 5:
            mean, std = gn.mean(), gn.std()
            spikes = df_post[df_post["grad_norm"] > mean + 3 * std]
            for _, row in spikes.iterrows():
                anomalies.append(f"grad_spike_step_{int(row['step'])}")
    return anomalies


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Standard diagnostic plots for the long MLP run
# ═══════════════════════════════════════════════════════════════════════

# Filter out eval-only rows for training metrics
df_long_train = df_long.dropna(subset=["token_avg_loss"])
df_long_eval = df_long.dropna(subset=["eval_loss"])

# ── 1. Loss Curve ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(df_long_train["step"], df_long_train["token_avg_loss"],
        color=APPROACH_COLORS["mlp"], linewidth=1.5, alpha=0.8,
        label="token_avg_loss (training)")
if len(df_long_eval) > 0:
    ax.plot(df_long_eval["step"], df_long_eval["eval_loss"],
            color="#d62728", marker="o", markersize=8, linewidth=2,
            label="eval_loss (validation)")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("MLP-Long Run (0302_175459): Training & Eval Loss", fontsize=13)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/long_mlp_loss_curve.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("✓ long_mlp_loss_curve.png")

# ── 2. Gradient Norms ──────────────────────────────────────────────────
df_long_gn = df_long_train.dropna(subset=["grad_norm"])
if len(df_long_gn) > 0:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(df_long_gn["step"], df_long_gn["grad_norm"],
            color=APPROACH_COLORS["mlp"], linewidth=1, alpha=0.7)
    ax.set_yscale("log")
    # Flag anomalies: spikes > 3 std above mean
    mean_gn = df_long_gn["grad_norm"].mean()
    std_gn = df_long_gn["grad_norm"].std()
    threshold = mean_gn + 3 * std_gn
    spikes = df_long_gn[df_long_gn["grad_norm"] > threshold]
    if len(spikes) > 0:
        ax.scatter(spikes["step"], spikes["grad_norm"],
                   color="red", s=50, zorder=5, label=f"Spike (>{threshold:.2f})")
    ax.axhline(mean_gn, color="gray", linestyle="--", alpha=0.5, label=f"Mean: {mean_gn:.3f}")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Gradient Norm (log scale)", fontsize=12)
    ax.set_title("MLP-Long Run (0302_175459): Gradient Norms", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/long_mlp_gradient_norms.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("✓ long_mlp_gradient_norms.png")

# ── 3. LR Schedule ─────────────────────────────────────────────────────
df_long_lr = df_long_train.dropna(subset=["learning_rate"])
if len(df_long_lr) > 0:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(df_long_lr["step"], df_long_lr["learning_rate"],
            color=APPROACH_COLORS["mlp"], linewidth=1.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("MLP-Long Run (0302_175459): Learning Rate Schedule", fontsize=13)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-4, -4))
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/long_mlp_lr_schedule.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("✓ long_mlp_lr_schedule.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Cross-approach comparison plots (merging 03-02 + 03-04 data)
# ═══════════════════════════════════════════════════════════════════════

# Merge old and new dataframes, deduplicate
# Eval rows share step+epoch with training rows but have eval_loss filled and loss NaN
# Training rows have loss filled and eval_loss NaN
df_all = pd.concat([df_old, df_new], ignore_index=True)
df_all["_is_eval"] = df_all["eval_loss"].notna()
df_all = df_all.drop_duplicates(subset=["experiment", "step", "_is_eval"], keep="first")
df_all = df_all.drop(columns=["_is_eval"])

# Filter to comparison experiments only
comp_names = list(COMPARISON_EXPS.keys())
df_comp = df_all[df_all["experiment"].isin(comp_names)].copy()

# ── 4. text_vs_esm_eval_loss.png ────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name in comp_names:
    approach, label = COMPARISON_EXPS[exp_name]
    color = APPROACH_COLORS[approach]
    exp_data = df_comp[df_comp["experiment"] == exp_name]
    eval_data = exp_data.dropna(subset=["eval_loss"])
    if len(eval_data) > 0:
        ax.plot(eval_data["step"], eval_data["eval_loss"],
                color=color, marker="o", markersize=6, linewidth=2,
                alpha=0.85, label=label)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Eval Loss", fontsize=12)
ax.set_title("Eval Loss: Text vs MLP vs Perceiver", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/text_vs_esm_eval_loss.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("✓ text_vs_esm_eval_loss.png")

# ── 5. text_vs_esm_final_comparison.png ─────────────────────────────────
# Bar chart: best eval_loss per experiment
bar_data = []
for exp_name in comp_names:
    approach, label = COMPARISON_EXPS[exp_name]
    m = metadata.get(exp_name, {})
    # Get best eval loss from data
    exp_data = df_comp[df_comp["experiment"] == exp_name]
    eval_data = exp_data.dropna(subset=["eval_loss"])
    best_eval = eval_data["eval_loss"].min() if len(eval_data) > 0 else None
    # Fallback to metadata
    if best_eval is None or np.isnan(best_eval):
        best_eval = m.get("best_eval_loss") or m.get("final_eval_loss")
    if best_eval is not None:
        bar_data.append({
            "label": label,
            "approach": approach,
            "best_eval_loss": best_eval,
            "dataset": m.get("dataset", "unknown"),
        })

fig, ax = plt.subplots(figsize=FIG_SIZE)
x = range(len(bar_data))
colors = [APPROACH_COLORS[d["approach"]] for d in bar_data]
bars = ax.bar(x, [d["best_eval_loss"] for d in bar_data], color=colors, width=0.6, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([d["label"] for d in bar_data], rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Best Eval Loss", fontsize=12)
ax.set_title("Best Eval Loss: Cross-Approach Comparison", fontsize=13)
# Annotate bars
for bar, d in zip(bars, bar_data):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{d['best_eval_loss']:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
# Add dataset note
ax.text(0.02, 0.98,
        "Note: MLP-long uses mol_instructions only;\nothers use combined_sft dataset",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
# Set y-axis to start from reasonable value
min_val = min(d["best_eval_loss"] for d in bar_data)
ax.set_ylim(0, max(d["best_eval_loss"] for d in bar_data) * 1.15)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/text_vs_esm_final_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("✓ text_vs_esm_final_comparison.png")

# ── 6. approach_convergence_comparison.png ──────────────────────────────
# Show steps to reach specific eval_loss thresholds
thresholds = [3.0, 2.5, 2.2, 2.0]

fig, ax = plt.subplots(figsize=FIG_SIZE)
bar_width = 0.18
x_base = np.arange(len(thresholds))

for i, exp_name in enumerate(comp_names):
    approach, label = COMPARISON_EXPS[exp_name]
    color = APPROACH_COLORS[approach]
    exp_data = df_comp[df_comp["experiment"] == exp_name]
    eval_data = exp_data.dropna(subset=["eval_loss"]).sort_values("step")

    steps_to_threshold = []
    for t in thresholds:
        reached = eval_data[eval_data["eval_loss"] <= t]
        if len(reached) > 0:
            steps_to_threshold.append(int(reached.iloc[0]["step"]))
        else:
            steps_to_threshold.append(0)  # Never reached

    offset = (i - len(comp_names) / 2 + 0.5) * bar_width
    bars = ax.bar(x_base + offset, steps_to_threshold, bar_width,
                  color=color, alpha=0.85, label=label, edgecolor="white")

    # Annotate non-zero bars
    for bar, val in zip(bars, steps_to_threshold):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(val), ha="center", va="bottom", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 10,
                    "N/R", ha="center", va="bottom", fontsize=7, color="gray")

ax.set_xticks(x_base)
ax.set_xticklabels([f"≤ {t}" for t in thresholds], fontsize=11)
ax.set_xlabel("Eval Loss Threshold", fontsize=12)
ax.set_ylabel("Steps to Reach Threshold", fontsize=12)
ax.set_title("Convergence Speed: Steps to Reach Eval Loss Thresholds", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/approach_convergence_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("✓ approach_convergence_comparison.png")

# ── 7. Training loss curves overlaid ────────────────────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name in comp_names:
    approach, label = COMPARISON_EXPS[exp_name]
    color = APPROACH_COLORS[approach]
    exp_data = df_comp[df_comp["experiment"] == exp_name]
    train_data = exp_data.dropna(subset=["token_avg_loss"]).sort_values("step")
    if len(train_data) > 0:
        ax.plot(train_data["step"], train_data["token_avg_loss"],
                color=color, linewidth=1.2, alpha=0.8, label=label)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Token Average Loss", fontsize=12)
ax.set_title("Training Loss: All Approaches", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/loss_curves.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("✓ loss_curves.png")

# ── 8. Convergence Table ────────────────────────────────────────────────
headers = ["Experiment", "Approach", "Dataset", "LR", "Epochs",
           "Steps", "Final Train Loss", "Best Eval Loss", "Best Eval Step"]
table_data = []
for exp_name in comp_names:
    approach, label = COMPARISON_EXPS[exp_name]
    m = metadata.get(exp_name, {})
    exp_data = df_comp[df_comp["experiment"] == exp_name]
    train_data = exp_data.dropna(subset=["token_avg_loss"])
    eval_data = exp_data.dropna(subset=["eval_loss"])

    final_tal = f"{train_data['token_avg_loss'].iloc[-1]:.3f}" if len(train_data) > 0 else "—"
    best_eval = f"{eval_data['eval_loss'].min():.3f}" if len(eval_data) > 0 else "—"
    best_step = str(int(eval_data.loc[eval_data['eval_loss'].idxmin(), 'step'])) if len(eval_data) > 0 else "—"
    total_steps = m.get("total_steps") or m.get("last_step") or (int(train_data["step"].max()) if len(train_data) > 0 else "—")

    dataset = m.get("dataset") or ("combined_sft" if "combined" in label.lower() else "unknown")
    table_data.append([
        label,
        approach.upper(),
        dataset[:15],
        f"{m.get('learning_rate', '?')}",
        str(m.get("num_epochs", "?")),
        str(total_steps),
        final_tal,
        best_eval,
        best_step,
    ])

fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
ax.axis("off")
ax.set_title("Experiment Summary Table", fontsize=14, fontweight="bold", pad=20)
table = ax.table(
    cellText=table_data,
    colLabels=headers,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.6)
# Style header row
for j in range(len(headers)):
    table[0, j].set_facecolor("#4472C4")
    table[0, j].set_text_props(color="white", fontweight="bold")
# Alternating row colors
for i in range(len(table_data)):
    for j in range(len(headers)):
        if i % 2 == 0:
            table[i + 1, j].set_facecolor("#D9E2F3")
        else:
            table[i + 1, j].set_facecolor("#FFFFFF")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/convergence_table.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("✓ convergence_table.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: analysis_summary.json
# ═══════════════════════════════════════════════════════════════════════

summary = {"experiments": {}, "comparison": {}, "notes": {}}

for exp_name in comp_names:
    approach, label = COMPARISON_EXPS[exp_name]
    m = metadata.get(exp_name, {})
    exp_data = df_comp[df_comp["experiment"] == exp_name]
    train_data = exp_data.dropna(subset=["token_avg_loss"]).sort_values("step")
    eval_data = exp_data.dropna(subset=["eval_loss"]).sort_values("step")

    convergence_step = compute_convergence_step(
        train_data["token_avg_loss"].reset_index(drop=True),
        train_data["step"].reset_index(drop=True),
    )
    anomalies = detect_anomalies(exp_data)

    entry = {
        "approach": approach,
        "projector_type": m.get("projector_type"),
        "dataset": m.get("dataset"),
        "learning_rate": m.get("learning_rate"),
        "num_epochs": m.get("num_epochs"),
        "total_steps": int(train_data["step"].max()) if len(train_data) > 0 else None,
        "final_token_avg_loss": float(train_data["token_avg_loss"].iloc[-1]) if len(train_data) > 0 else None,
        "min_token_avg_loss": float(train_data["token_avg_loss"].min()) if len(train_data) > 0 else None,
        "best_eval_loss": float(eval_data["eval_loss"].min()) if len(eval_data) > 0 else None,
        "best_eval_step": int(eval_data.loc[eval_data["eval_loss"].idxmin(), "step"]) if len(eval_data) > 0 else None,
        "convergence_step": convergence_step,
        "max_grad_norm": float(exp_data["grad_norm"].max()) if "grad_norm" in exp_data and exp_data["grad_norm"].notna().any() else None,
        "mean_grad_norm": float(exp_data["grad_norm"].mean()) if "grad_norm" in exp_data and exp_data["grad_norm"].notna().any() else None,
        "anomalies": anomalies,
    }
    summary["experiments"][exp_name] = entry

# Find best experiment
best_exp_name = min(
    summary["experiments"].items(),
    key=lambda x: x[1].get("best_eval_loss") or float("inf")
)
summary["comparison"] = {
    "best_experiment": best_exp_name[0],
    "best_experiment_label": COMPARISON_EXPS[best_exp_name[0]][1],
    "metric": "best_eval_loss",
    "value": best_exp_name[1].get("best_eval_loss"),
    "ranking": sorted(
        [(name, d.get("best_eval_loss")) for name, d in summary["experiments"].items() if d.get("best_eval_loss")],
        key=lambda x: x[1]
    ),
}
summary["notes"] = {
    "dataset_caveat": "MLP-long (0302) uses mol_instructions only (not combined_sft). Other runs use combined_sft_260225.",
    "hyperparameter_diff": "MLP-long uses lr=1.2e-4, 7 epochs. Old runs use lr=2e-4, 3 epochs.",
    "mlp_long_status": "Stopped at step 500/4809 (~10%). Eval loss 2.04 may improve further.",
    "metric_note": "All training loss values use token_avg_loss (NOT HF running average 'loss').",
}

with open(f"{DATA_DIR_NEW}/analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✓ analysis_summary.json")

# Print summary
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)
for name, data in summary["experiments"].items():
    label = COMPARISON_EXPS.get(name, ("?", name))[1]
    print(f"\n{label}:")
    print(f"  Final token_avg_loss: {data['final_token_avg_loss']:.4f}")
    print(f"  Best eval_loss:       {data['best_eval_loss']:.4f}" if data['best_eval_loss'] else "  Best eval_loss: N/A")
    print(f"  Total steps:          {data['total_steps']}")
    print(f"  Anomalies:            {data['anomalies'] if data['anomalies'] else 'None'}")
print(f"\nBest: {summary['comparison']['best_experiment_label']} "
      f"(eval_loss={summary['comparison']['value']:.4f})")
print("=" * 60)
