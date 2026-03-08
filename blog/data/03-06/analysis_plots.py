#!/usr/bin/env python3
"""Analyst: Create diagnostic plots and analysis_summary.json for 03-06 report."""

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

# Per-experiment colors (all MLP, so use distinct blues/teals)
EXP_COLORS = {
    "sft_esm3_mlp_combined_qwen3_8b_it_0306_010408": "#1f77b4",  # Blue (main run)
    "sft_esm3_mlp_combined_qwen3_8b_it_0305_100040": "#17becf",  # Cyan (short predecessor)
    "sft_esm3_mlp_long_qwen3_8b_it_0305_105828": "#ff7f0e",      # Orange (long/mol_inst)
    "sft_esm3_mlp_combined_qwen3_8b_it_0302_201516": "#9467bd",  # Purple (1-step)
}

EXP_LABELS = {
    "sft_esm3_mlp_combined_qwen3_8b_it_0306_010408": "MLP-Combined (0306, main)",
    "sft_esm3_mlp_combined_qwen3_8b_it_0305_100040": "MLP-Combined (0305, short)",
    "sft_esm3_mlp_long_qwen3_8b_it_0305_105828": "MLP-Long (0305, mol_inst)",
    "sft_esm3_mlp_combined_qwen3_8b_it_0302_201516": "MLP-Combined (0302, 1-step)",
}

FIG_DPI = 150
FIG_SIZE = (10, 6)
FIG_SIZE_WIDE = (14, 6)
FIG_SIZE_TALL = (10, 8)

# Paths
BASE = "/home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM"
FIG_DIR = f"{BASE}/blog/figures"
DATA_DIR = f"{BASE}/blog/data/03-06"

os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────
df = pd.read_csv(f"{DATA_DIR}/run_histories.csv")

with open(f"{DATA_DIR}/experiment_metadata.json") as f:
    meta = json.load(f)

# Build metadata lookup
metadata = {}
for exp in meta["experiments"]:
    metadata[exp["name"]] = exp

# Focus on experiments with actual data (>1 log entry)
FOCUS_EXPS = [
    "sft_esm3_mlp_combined_qwen3_8b_it_0306_010408",  # Main run (728 entries)
    "sft_esm3_mlp_combined_qwen3_8b_it_0305_100040",  # Short predecessor (26 entries)
    "sft_esm3_mlp_long_qwen3_8b_it_0305_105828",      # Long/mol_inst (156 entries)
]

MAIN_RUN = "sft_esm3_mlp_combined_qwen3_8b_it_0306_010408"


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
    df_post = df_exp[df_exp["step"] > warmup_steps]
    if "token_avg_loss" in df_post.columns:
        tal = df_post["token_avg_loss"].dropna()
        if len(tal) > 5:
            mean, std = tal.mean(), tal.std()
            spikes = df_post[df_post["token_avg_loss"] > mean + 3 * std]
            for _, row in spikes.iterrows():
                anomalies.append(f"loss_spike_step_{int(row['step'])}")
        if df_exp["token_avg_loss"].isna().any():
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
# PART 1: Training Loss Curves (all runs overlaid)
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name in FOCUS_EXPS:
    exp_data = df[df["experiment"] == exp_name]
    train_data = exp_data.dropna(subset=["token_avg_loss"]).sort_values("step")
    # For eval-only rows (where token_avg_loss appears but is actually eval), filter
    # by checking that eval_loss is NaN (pure training rows) OR it's a shared row
    if len(train_data) > 0:
        color = EXP_COLORS.get(exp_name, "#333333")
        label = EXP_LABELS.get(exp_name, shorten(exp_name))
        ax.plot(train_data["step"], train_data["token_avg_loss"],
                color=color, linewidth=1.2, alpha=0.85, label=label)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Token Average Loss", fontsize=12)
ax.set_title("Training Loss Curves: ESM3+MLP SFT Runs (March 2026)", fontsize=13)
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/mlp_combined_train_loss.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("OK mlp_combined_train_loss.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Eval Loss Curves (all runs overlaid)
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name in FOCUS_EXPS:
    exp_data = df[df["experiment"] == exp_name]
    eval_data = exp_data.dropna(subset=["eval_loss"]).sort_values("step")
    if len(eval_data) > 0:
        color = EXP_COLORS.get(exp_name, "#333333")
        label = EXP_LABELS.get(exp_name, shorten(exp_name))
        ax.plot(eval_data["step"], eval_data["eval_loss"],
                color=color, marker="o", markersize=5, linewidth=2,
                alpha=0.85, label=label)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Eval Loss", fontsize=12)
ax.set_title("Eval Loss Curves: ESM3+MLP SFT Runs (March 2026)", fontsize=13)
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/mlp_combined_eval_loss.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("OK mlp_combined_eval_loss.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Main Run Deep Dive — Loss + Eval on same axes
# ═══════════════════════════════════════════════════════════════════════

df_main = df[df["experiment"] == MAIN_RUN].copy()
df_main_train = df_main.dropna(subset=["token_avg_loss"]).sort_values("step")
# Separate eval rows: they have eval_loss filled
df_main_eval = df_main.dropna(subset=["eval_loss"]).sort_values("step")

fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(df_main_train["step"], df_main_train["token_avg_loss"],
        color=APPROACH_COLORS["mlp"], linewidth=1, alpha=0.7,
        label="token_avg_loss (train)")
if len(df_main_eval) > 0:
    ax.plot(df_main_eval["step"], df_main_eval["eval_loss"],
            color="#d62728", marker="o", markersize=6, linewidth=2,
            label="eval_loss (validation)")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title(f"Main Run (0306_010408): Train & Eval Loss\n"
             f"Step {int(df_main_train['step'].max())}/28941 — Still Running", fontsize=13)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/main_run_loss_detail.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("OK main_run_loss_detail.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Gradient Norms
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name in FOCUS_EXPS:
    exp_data = df[df["experiment"] == exp_name]
    gn_data = exp_data.dropna(subset=["grad_norm"]).sort_values("step")
    if len(gn_data) > 0:
        color = EXP_COLORS.get(exp_name, "#333333")
        label = EXP_LABELS.get(exp_name, shorten(exp_name))
        ax.plot(gn_data["step"], gn_data["grad_norm"],
                color=color, linewidth=0.8, alpha=0.7, label=label)
ax.set_yscale("log")
# Flag anomalies on main run
main_gn = df[df["experiment"] == MAIN_RUN].dropna(subset=["grad_norm"])
if len(main_gn) > 5:
    mean_gn = main_gn["grad_norm"].mean()
    std_gn = main_gn["grad_norm"].std()
    threshold = mean_gn + 3 * std_gn
    spikes = main_gn[main_gn["grad_norm"] > threshold]
    if len(spikes) > 0:
        ax.scatter(spikes["step"], spikes["grad_norm"],
                   color="red", s=40, zorder=5, label=f"Spikes (main, >{threshold:.2f})")
    ax.axhline(mean_gn, color="gray", linestyle="--", alpha=0.4, label=f"Main mean: {mean_gn:.3f}")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Gradient Norm (log scale)", fontsize=12)
ax.set_title("Gradient Norms: ESM3+MLP SFT Runs", fontsize=13)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/mlp_combined_grad_norms.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("OK mlp_combined_grad_norms.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 5: Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=FIG_SIZE)
for exp_name in FOCUS_EXPS:
    exp_data = df[df["experiment"] == exp_name]
    lr_data = exp_data.dropna(subset=["learning_rate"]).sort_values("step")
    if len(lr_data) > 0:
        color = EXP_COLORS.get(exp_name, "#333333")
        label = EXP_LABELS.get(exp_name, shorten(exp_name))
        ax.plot(lr_data["step"], lr_data["learning_rate"],
                color=color, linewidth=1.5, alpha=0.85, label=label)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.set_title("Learning Rate Schedule: ESM3+MLP SFT Runs", fontsize=13)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(-4, -4))
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/mlp_combined_lr_schedule.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("OK mlp_combined_lr_schedule.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 6: Best Run Comparison (bar chart)
# ═══════════════════════════════════════════════════════════════════════

bar_data = []
for exp_name in FOCUS_EXPS:
    m = metadata.get(exp_name, {})
    label = EXP_LABELS.get(exp_name, shorten(exp_name))
    exp_data = df[df["experiment"] == exp_name]
    eval_data = exp_data.dropna(subset=["eval_loss"])
    train_data = exp_data.dropna(subset=["token_avg_loss"])

    best_eval = float(eval_data["eval_loss"].min()) if len(eval_data) > 0 else None
    final_train = float(train_data["token_avg_loss"].iloc[-1]) if len(train_data) > 0 else None
    total_steps = int(train_data["step"].max()) if len(train_data) > 0 else 0

    bar_data.append({
        "name": exp_name,
        "label": label,
        "best_eval_loss": best_eval,
        "final_train_loss": final_train,
        "total_steps": total_steps,
        "dataset": m.get("data_config", "unknown"),
        "lr": m.get("learning_rate", "?"),
    })

# Bar chart: final train loss and best eval loss side by side
fig, ax = plt.subplots(figsize=FIG_SIZE)
x = np.arange(len(bar_data))
width = 0.35

train_vals = [d["final_train_loss"] or 0 for d in bar_data]
eval_vals = [d["best_eval_loss"] or 0 for d in bar_data]

bars1 = ax.bar(x - width/2, train_vals, width, label="Final Train Loss",
               color="#1f77b4", alpha=0.8, edgecolor="white")
bars2 = ax.bar(x + width/2, eval_vals, width, label="Best Eval Loss",
               color="#d62728", alpha=0.8, edgecolor="white")

# Annotate
for bar, val in zip(bars1, train_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar, val in zip(bars2, eval_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([d["label"] for d in bar_data], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Final Metrics Comparison: ESM3+MLP SFT Runs", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, max(max(train_vals), max(eval_vals)) * 1.2)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/mlp_combined_final_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("OK mlp_combined_final_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 7: Convergence Table
# ═══════════════════════════════════════════════════════════════════════

headers = ["Experiment", "Dataset", "LR", "Proj LR",
           "Steps Logged", "Total Steps", "Final Train", "Best Eval", "Best Eval Step"]
table_data = []
for exp_name in FOCUS_EXPS:
    m = metadata.get(exp_name, {})
    label = EXP_LABELS.get(exp_name, shorten(exp_name))
    exp_data = df[df["experiment"] == exp_name]
    train_data = exp_data.dropna(subset=["token_avg_loss"]).sort_values("step")
    eval_data = exp_data.dropna(subset=["eval_loss"]).sort_values("step")

    final_tal = f"{train_data['token_avg_loss'].iloc[-1]:.3f}" if len(train_data) > 0 else "-"
    best_eval = f"{eval_data['eval_loss'].min():.3f}" if len(eval_data) > 0 else "-"
    best_step = str(int(eval_data.loc[eval_data['eval_loss'].idxmin(), 'step'])) if len(eval_data) > 0 else "-"
    max_step = int(train_data["step"].max()) if len(train_data) > 0 else 0
    total_steps = m.get("total_steps") or "-"

    table_data.append([
        label,
        str(m.get("data_config", "?"))[:18],
        f"{m.get('learning_rate', '?')}",
        f"{m.get('projector_lr', '?')}",
        str(max_step),
        str(total_steps),
        final_tal,
        best_eval,
        best_step,
    ])

fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
ax.axis("off")
ax.set_title("Experiment Summary Table (March 2026)", fontsize=14, fontweight="bold", pad=20)
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
fig.savefig(f"{FIG_DIR}/mlp_combined_convergence_table.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("OK mlp_combined_convergence_table.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 8: Main Run — Smoothed Loss with Moving Average
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=FIG_SIZE)
# Raw loss (faint)
ax.plot(df_main_train["step"], df_main_train["token_avg_loss"],
        color=APPROACH_COLORS["mlp"], linewidth=0.5, alpha=0.3, label="Raw")
# Moving average
window = 50
if len(df_main_train) > window:
    smoothed = df_main_train["token_avg_loss"].rolling(window, min_periods=1).mean()
    ax.plot(df_main_train["step"], smoothed,
            color=APPROACH_COLORS["mlp"], linewidth=2, alpha=0.9,
            label=f"MA-{window}")
# Eval points
if len(df_main_eval) > 0:
    ax.plot(df_main_eval["step"], df_main_eval["eval_loss"],
            color="#d62728", marker="D", markersize=7, linewidth=0, zorder=5,
            label="Eval Loss")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title(f"Main Run (0306): Smoothed Training Loss (MA-{window}) + Eval", fontsize=13)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/main_run_smoothed_loss.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("OK main_run_smoothed_loss.png")


# ═══════════════════════════════════════════════════════════════════════
# PART 9: analysis_summary.json
# ═══════════════════════════════════════════════════════════════════════

summary = {"experiments": {}, "comparison": {}, "notes": {}}

for exp_name in FOCUS_EXPS:
    m = metadata.get(exp_name, {})
    exp_data = df[df["experiment"] == exp_name]
    train_data = exp_data.dropna(subset=["token_avg_loss"]).sort_values("step")
    eval_data = exp_data.dropna(subset=["eval_loss"]).sort_values("step")

    convergence_step = compute_convergence_step(
        train_data["token_avg_loss"].reset_index(drop=True),
        train_data["step"].reset_index(drop=True),
    )
    anomalies = detect_anomalies(exp_data)

    entry = {
        "approach": "mlp",
        "projector_type": m.get("projector_type"),
        "dataset": m.get("data_config"),
        "learning_rate": m.get("learning_rate"),
        "projector_lr": m.get("projector_lr"),
        "total_steps_planned": m.get("total_steps"),
        "total_steps_logged": int(train_data["step"].max()) if len(train_data) > 0 else None,
        "progress_pct": round(int(train_data["step"].max()) / m.get("total_steps", 1) * 100, 1) if len(train_data) > 0 and m.get("total_steps") else None,
        "final_token_avg_loss": float(train_data["token_avg_loss"].iloc[-1]) if len(train_data) > 0 else None,
        "min_token_avg_loss": float(train_data["token_avg_loss"].min()) if len(train_data) > 0 else None,
        "best_eval_loss": float(eval_data["eval_loss"].min()) if len(eval_data) > 0 else None,
        "best_eval_step": int(eval_data.loc[eval_data["eval_loss"].idxmin(), "step"]) if len(eval_data) > 0 else None,
        "convergence_step": convergence_step,
        "max_grad_norm": float(exp_data["grad_norm"].max()) if "grad_norm" in exp_data.columns and exp_data["grad_norm"].notna().any() else None,
        "mean_grad_norm": float(exp_data["grad_norm"].mean()) if "grad_norm" in exp_data.columns and exp_data["grad_norm"].notna().any() else None,
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
    "best_experiment_label": EXP_LABELS.get(best_exp_name[0], best_exp_name[0]),
    "metric": "best_eval_loss",
    "value": best_exp_name[1].get("best_eval_loss"),
    "ranking": sorted(
        [(name, d.get("best_eval_loss")) for name, d in summary["experiments"].items() if d.get("best_eval_loss")],
        key=lambda x: x[1]
    ),
}
summary["notes"] = {
    "main_run_status": f"0306_010408 at step {summary['experiments'][MAIN_RUN]['total_steps_logged']}/28941 ({summary['experiments'][MAIN_RUN]['progress_pct']}%). Still running.",
    "dataset_info": "Main runs use combined_sft_260225 (4.89M samples). Long run uses mol_instructions only.",
    "hyperparameter_info": "Main: lr=2e-4, proj_lr=1e-3. Long: lr=1.2e-4, proj_lr=6e-4.",
    "metric_note": "All training loss values use token_avg_loss (NOT HF running average 'loss').",
    "all_experiments_count": f"{len(meta['experiments'])} total experiments found, {len(FOCUS_EXPS)} with sufficient data for analysis.",
}

with open(f"{DATA_DIR}/analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("OK analysis_summary.json")

# Print summary
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)
for name, data in summary["experiments"].items():
    label = EXP_LABELS.get(name, name)
    print(f"\n{label}:")
    print(f"  Progress:             {data['total_steps_logged']}/{data['total_steps_planned']} steps ({data['progress_pct']}%)")
    print(f"  Final token_avg_loss: {data['final_token_avg_loss']:.4f}")
    if data['best_eval_loss']:
        print(f"  Best eval_loss:       {data['best_eval_loss']:.4f} (step {data['best_eval_step']})")
    else:
        print("  Best eval_loss:       N/A")
    print(f"  Anomalies:            {data['anomalies'] if data['anomalies'] else 'None'}")
print(f"\nBest: {summary['comparison']['best_experiment_label']} "
      f"(eval_loss={summary['comparison']['value']:.4f})")
print("=" * 60)
