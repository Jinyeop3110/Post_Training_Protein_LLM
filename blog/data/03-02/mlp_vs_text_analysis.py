"""
MLP vs Text-Only SFT Comparison — Data Collection, Plotting & Analysis

Collects training metrics from local experiment files, generates diagnostic plots,
and produces an analysis summary.

Usage:
    python blog/data/03-02/mlp_vs_text_analysis.py

Outputs:
    blog/figures/mlp_vs_text_*.png       — 5 diagnostic plots
    blog/data/03-02/run_histories.csv    — Step-level metrics for both experiments
    blog/data/03-02/experiment_metadata.json — Per-experiment metadata
    blog/data/03-02/analysis_summary.json   — Statistical summary + anomaly detection
"""
import matplotlib

matplotlib.use('Agg')
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Config ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="colorblind")
APPROACH_COLORS = {"mlp": "#1f77b4", "perceiver": "#ff7f0e", "text": "#2ca02c"}
FIG_DPI = 150
FIG_SIZE = (10, 6)

BLOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BLOG_DIR), "results")

EXPERIMENTS = [
    "sft_lora_esm3_qwen3_8b_it_0226_151416",  # MLP (ESM-3 + attention pooling + MLP projector)
    "sft_text_qwen3_8b_it_0227_145821",         # Text-only (raw AA tokens)
]


# ── Data Collection ─────────────────────────────────────────────────────────
def collect_data():
    all_rows = []
    metadata_list = []

    for exp_name in EXPERIMENTS:
        exp_dir = os.path.join(RESULTS_DIR, exp_name)

        lineage = _read_json(os.path.join(exp_dir, "lineage.json"))
        metrics = _read_json(os.path.join(exp_dir, "metrics.json"))
        train_args = _read_json(os.path.join(exp_dir, "training_args.json"))
        trainer_state = _best_trainer_state(exp_dir)

        if trainer_state and "log_history" in trainer_state:
            for entry in trainer_state["log_history"]:
                all_rows.append({
                    "experiment": exp_name,
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "token_avg_loss": entry.get("token_avg_loss"),
                    "loss": entry.get("loss"),
                    "eval_loss": entry.get("eval_loss"),
                    "grad_norm": entry.get("grad_norm"),
                    "learning_rate": entry.get("learning_rate"),
                })

        approach = lineage.get("approach", "text" if "text" in exp_name else "esm3")
        projector_type = lineage.get("projector_type", "mlp" if approach == "esm3" else None)

        final_tal, final_eval, total_steps = None, None, 0
        if trainer_state and "log_history" in trainer_state:
            train_e = [e for e in trainer_state["log_history"] if "token_avg_loss" in e]
            eval_e = [e for e in trainer_state["log_history"] if "eval_loss" in e]
            if train_e:
                final_tal = train_e[-1]["token_avg_loss"]
                total_steps = train_e[-1].get("step", 0)
            if eval_e:
                final_eval = min(e["eval_loss"] for e in eval_e)

        metadata_list.append({
            "name": exp_name,
            "approach": approach,
            "projector_type": projector_type,
            "base_model": lineage.get("base_model", train_args.get("model_name_or_path", "unknown")),
            "stage": lineage.get("stage", "sft_lora"),
            "learning_rate": train_args.get("learning_rate"),
            "projector_lr": train_args.get("projector_lr"),
            "num_epochs": train_args.get("num_train_epochs"),
            "total_steps": total_steps,
            "final_token_avg_loss": metrics.get("token_avg_loss", final_tal),
            "final_eval_loss": final_eval,
            "gpu_memory_max_gb": metrics.get("gpu_memory_max_allocated_gb"),
            "train_runtime_seconds": metrics.get("train_runtime"),
            "created_at": lineage.get("created_at"),
            "completed_at": lineage.get("completed_at"),
        })

    df = pd.DataFrame(all_rows).sort_values(["experiment", "step"]).reset_index(drop=True)
    df.to_csv(os.path.join(DATA_DIR, "run_histories.csv"), index=False)
    with open(os.path.join(DATA_DIR, "experiment_metadata.json"), "w") as f:
        json.dump({"experiments": metadata_list}, f, indent=2)

    return df, {"experiments": metadata_list}


def _read_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _best_trainer_state(exp_dir):
    candidates = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "**", "trainer_state.json"), recursive=True))
    best, best_count = None, 0
    for p in candidates:
        with open(p) as f:
            ts = json.load(f)
        count = len(ts.get("log_history", []))
        if count > best_count:
            best_count = count
            best = ts
    return best


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_approach(name, metadata):
    for e in metadata["experiments"]:
        if e["name"] == name:
            a = e["approach"]
            return "mlp" if a == "esm3" else a
    return "unknown"


def label_for(name, metadata):
    a = get_approach(name, metadata)
    return f"{a.upper()} (ESM-3)" if a == "mlp" else f"{a.upper()} (raw tokens)"


def get_color(name, metadata):
    return APPROACH_COLORS.get(get_approach(name, metadata), "#333")


# ── Plotting ────────────────────────────────────────────────────────────────
def make_plots(df, metadata):
    train_df = df[df['token_avg_loss'].notna()].copy()
    eval_df = df[df['eval_loss'].notna()].copy()

    # 1. Training loss
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for name, grp in train_df.groupby("experiment"):
        ax.plot(grp["step"], grp["token_avg_loss"], color=get_color(name, metadata),
                label=label_for(name, metadata), alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Token Average Loss", fontsize=12)
    ax.set_title("Training Loss: MLP (ESM-3) vs Text-Only SFT", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(BLOG_DIR, "mlp_vs_text_training_loss.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    # 2. Eval loss
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for name, grp in eval_df.groupby("experiment"):
        ax.plot(grp["step"], grp["eval_loss"], color=get_color(name, metadata),
                label=label_for(name, metadata), alpha=0.85, linewidth=1.5, marker='o', markersize=5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Eval Loss", fontsize=12)
    ax.set_title("Validation Loss: MLP (ESM-3) vs Text-Only SFT", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(BLOG_DIR, "mlp_vs_text_eval_loss.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    # 3. Gradient norms
    grad_df = train_df[train_df['grad_norm'].notna()].copy()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for name, grp in grad_df.groupby("experiment"):
        ax.plot(grp["step"], grp["grad_norm"], color=get_color(name, metadata),
                label=label_for(name, metadata), alpha=0.7, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Gradient Norm (log scale)", fontsize=12)
    ax.set_title("Gradient Norms During Training", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(BLOG_DIR, "mlp_vs_text_grad_norms.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    # 4. LR schedule
    lr_df = train_df[train_df['learning_rate'].notna()].copy()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for name, grp in lr_df.groupby("experiment"):
        ax.plot(grp["step"], grp["learning_rate"], color=get_color(name, metadata),
                label=label_for(name, metadata), alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14)
    ax.legend(fontsize=10)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(os.path.join(BLOG_DIR, "mlp_vs_text_lr_schedule.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    # 5. Final metrics bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [label_for(e["name"], metadata) for e in metadata["experiments"]]
    colors = [get_color(e["name"], metadata) for e in metadata["experiments"]]

    tl = [e.get("final_token_avg_loss") for e in metadata["experiments"]]
    axes[0].bar(labels, tl, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(tl):
        if v: axes[0].text(i, v + 0.03, f"{v:.3f}", ha='center', fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Token Average Loss", fontsize=11)
    axes[0].set_title("Final Training Loss", fontsize=13)

    el = [e.get("final_eval_loss") for e in metadata["experiments"]]
    axes[1].bar(labels, el, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(el):
        if v: axes[1].text(i, v + 0.03, f"{v:.3f}", ha='center', fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Eval Loss", fontsize=11)
    axes[1].set_title("Best Validation Loss", fontsize=13)

    fig.suptitle("Final Metrics: MLP vs Text-Only", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(BLOG_DIR, "mlp_vs_text_final_metrics.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    return train_df, eval_df


# ── Analysis Summary ────────────────────────────────────────────────────────
def compute_summary(train_df, eval_df, metadata):
    summary = {"experiments": {}, "comparison": {}}

    for exp in metadata["experiments"]:
        name = exp["name"]
        approach = get_approach(name, metadata)
        et = train_df[train_df["experiment"] == name]
        ee = eval_df[eval_df["experiment"] == name]

        convergence_step = -1
        if len(et) >= 10:
            rolling = et["token_avg_loss"].rolling(10).mean()
            pct = rolling.pct_change().abs()
            converged = pct[pct < 0.01]
            if len(converged) > 0:
                convergence_step = int(et.iloc[converged.index[0] - et.index[0]]["step"])

        anomalies = []
        if len(et) > 0:
            m, s = et["token_avg_loss"].mean(), et["token_avg_loss"].std()
            for _, r in et[et["token_avg_loss"] > m + 3*s].iterrows():
                anomalies.append(f"loss_spike_step_{int(r['step'])}")
            if et["grad_norm"].notna().any():
                gn = et["grad_norm"].dropna()
                mg, sg = gn.mean(), gn.std()
                for idx in gn[gn > mg + 3*sg].index:
                    anomalies.append(f"grad_spike_step_{int(et.loc[idx,'step'])}")

        summary["experiments"][name] = {
            "approach": approach,
            "projector_type": exp.get("projector_type"),
            "total_steps": exp.get("total_steps", 0),
            "final_token_avg_loss": exp.get("final_token_avg_loss"),
            "best_eval_loss": exp.get("final_eval_loss"),
            "best_eval_step": int(ee.loc[ee["eval_loss"].idxmin(), "step"]) if len(ee) else None,
            "convergence_step": convergence_step,
            "max_grad_norm": float(et["grad_norm"].max()) if et["grad_norm"].notna().any() else None,
            "mean_grad_norm": float(et["grad_norm"].mean()) if et["grad_norm"].notna().any() else None,
            "anomalies": anomalies,
        }

    best = min(summary["experiments"].items(), key=lambda x: x[1].get("best_eval_loss") or float("inf"))
    summary["comparison"] = {"best_experiment": best[0], "metric": "best_eval_loss", "value": best[1].get("best_eval_loss")}

    with open(os.path.join(DATA_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Collecting data...")
    df, metadata = collect_data()
    print(f"  {len(df)} rows, {len(metadata['experiments'])} experiments")

    print("Generating plots...")
    train_df, eval_df = make_plots(df, metadata)
    print("  5 figures saved to blog/")

    print("Computing analysis summary...")
    summary = compute_summary(train_df, eval_df, metadata)
    print(f"  Best: {summary['comparison']['best_experiment']} (eval_loss={summary['comparison']['value']:.4f})")

    print("\nDone.")
