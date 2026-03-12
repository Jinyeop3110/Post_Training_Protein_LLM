#!/usr/bin/env python3
"""
Plot GRPO Structure Quality comparison: Text-only vs ESM3+MLP.

Generates 4 main figures (fig13-fig16) for the blog post.
"""

import matplotlib

matplotlib.use("Agg")

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "analysis"))
from figure_style import (
    annotate_bars,
    save_main_figure,
    style,
)

colors = style.apply("blog")

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

steps = np.arange(5, 90, 5)

mlp_reward = np.array([0.5184, 0.5822, 0.5850, 0.6205, 0.5815, 0.6161, 0.5853,
                        0.5744, 0.5967, 0.6004, 0.5985, 0.5616, 0.5934, 0.5871,
                        0.6000, 0.6064, 0.5367])
mlp_loss = np.array([-0.5743, -0.7730, -0.8413, -0.5876, -1.0047, -1.1075, -1.0207,
                      -1.4122, -0.9213, -0.9926, -1.2237, -1.0625, -1.3141, -1.2080,
                      -1.1078, -1.3049, -0.7647])
mlp_grad_lora = np.array([0.0]*17)
mlp_grad_mm = np.array([1.0]*11 + [0.8] + [1.0]*5)

text_reward = np.array([0.6280, 0.6789, 0.7078, 0.7795, 0.8431, 0.8140, 0.8735,
                         0.8439, 0.8259, 0.8997, 0.8550, 0.9310, 0.9619, 0.8763,
                         0.9047, 0.8223, 0.8658])
text_loss = np.array([-0.6533, -0.5099, -0.4099, -0.0740, -0.0243, -0.0248, -0.0072,
                       -0.0021, 0.0078, -0.0093, 0.0087, -0.0107, -0.0062, -0.0162,
                       0.0090, -0.0006, -0.0113])
text_grad_lora = np.array([1.0, 1.0, 1.0, 0.8874, 0.7028, 0.7164, 1.0, 1.0, 1.0,
                            1.0, 1.0, 0.9979, 0.8755, 0.9990, 1.0, 0.9654, 1.0])
text_grad_mm = np.array([0.0]*17)

# Final metrics
mlp_final = {"mean_reward": 0.582, "quality_alignment": 0.326,
             "numerical_accuracy": 0.086, "category_match": 0.104, "format_bonus": 0.065}
text_final = {"mean_reward": 0.832, "quality_alignment": 0.351,
              "numerical_accuracy": 0.185, "category_match": 0.197, "format_bonus": 0.099}

# Eval metrics
eval_steps = [25, 50, 75]
mlp_eval_reward = [0.544, 0.462, 0.526]
mlp_eval_format = [0.74, 0.76, 0.74]
text_eval_reward = [0.754, 0.797, 0.844]
text_eval_format = [1.0, 1.0, 1.0]

# Colors and labels
c_mlp = style.color("mlp")
c_text = style.color("text")
l_mlp = "ESM3+MLP"
l_text = "Text-only"


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 13: Reward Curves (2-panel)
# ═══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: training reward
marker_idx = np.arange(1, len(steps), 2)  # every 10 steps
ax1.plot(steps, text_reward, color=c_text, label=l_text, alpha=0.8, linewidth=1.5)
ax1.plot(steps[marker_idx], text_reward[marker_idx], "o", color=c_text, markersize=5)
ax1.plot(steps, mlp_reward, color=c_mlp, label=l_mlp, alpha=0.8, linewidth=1.5)
ax1.plot(steps[marker_idx], mlp_reward[marker_idx], "s", color=c_mlp, markersize=5)

# Final mean_reward dashed lines
ax1.axhline(y=text_final["mean_reward"], color=c_text, linestyle="--", alpha=0.5, linewidth=1.0)
ax1.text(87, text_final["mean_reward"] + 0.01, f'{text_final["mean_reward"]:.3f}',
         color=c_text, fontsize=9, ha="right", fontweight="bold")
ax1.axhline(y=mlp_final["mean_reward"], color=c_mlp, linestyle="--", alpha=0.5, linewidth=1.0)
ax1.text(87, mlp_final["mean_reward"] - 0.025, f'{mlp_final["mean_reward"]:.3f}',
         color=c_mlp, fontsize=9, ha="right", fontweight="bold")

ax1.set_xlabel("Training Step")
ax1.set_ylabel("Mean Reward")
ax1.set_title("Training Reward")
ax1.legend(loc="lower right")
style.clean_axes(ax1)

# Right: eval reward
ax2.plot(eval_steps, text_eval_reward, "o-", color=c_text, label=l_text,
         markersize=8, linewidth=1.5)
ax2.plot(eval_steps, mlp_eval_reward, "s-", color=c_mlp, label=l_mlp,
         markersize=8, linewidth=1.5)

# Annotate eval points
for i, s in enumerate(eval_steps):
    ax2.annotate(f"{text_eval_reward[i]:.3f}", (s, text_eval_reward[i]),
                 textcoords="offset points", xytext=(0, 10), ha="center",
                 fontsize=9, color=c_text, fontweight="bold")
    ax2.annotate(f"{mlp_eval_reward[i]:.3f}", (s, mlp_eval_reward[i]),
                 textcoords="offset points", xytext=(0, -15), ha="center",
                 fontsize=9, color=c_mlp, fontweight="bold")

ax2.set_xlabel("Training Step")
ax2.set_ylabel("Mean Reward")
ax2.set_title("Eval Reward")
ax2.set_xticks(eval_steps)
ax2.legend(loc="lower right")
style.clean_axes(ax2)

fig.suptitle("GRPO Structure Quality: Text vs MLP Reward", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
saved = save_main_figure(fig, "fig13_grpo_structure_reward_curves")
print(f"Figure 13 saved: {saved}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 14: Reward Component Breakdown
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))

components = ["quality_alignment", "numerical_accuracy", "category_match", "format_bonus"]
comp_labels = ["Quality\nAlignment", "Numerical\nAccuracy", "Category\nMatch", "Format\nBonus"]
text_vals = [text_final[c] for c in components]
mlp_vals = [mlp_final[c] for c in components]

x = np.arange(len(components))
width = 0.35

bars_text = ax.bar(x - width/2, text_vals, width, color=c_text, label=l_text, edgecolor="white")
bars_mlp = ax.bar(x + width/2, mlp_vals, width, color=c_mlp, label=l_mlp, edgecolor="white")

annotate_bars(ax, bars_text, text_vals, fmt="{:.3f}", offset=0.005)
annotate_bars(ax, bars_mlp, mlp_vals, fmt="{:.3f}", offset=0.005)

# Total mean_reward annotation
ax.text(0.98, 0.95, f"Total Mean Reward\n{l_text}: {text_final['mean_reward']:.3f}\n{l_mlp}: {mlp_final['mean_reward']:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

ax.set_xlabel("")
ax.set_ylabel("Reward Component Score")
ax.set_title("Structure Quality: Reward Component Breakdown", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(comp_labels)
ax.legend(loc="upper left")
ax.set_ylim(0, max(max(text_vals), max(mlp_vals)) * 1.25)
style.clean_axes(ax)
fig.tight_layout()
saved = save_main_figure(fig, "fig14_grpo_structure_reward_breakdown")
print(f"Figure 14 saved: {saved}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 15: Format Rate + Training Loss (2-panel)
# ═══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: format rate bar groups
x_eval = np.arange(len(eval_steps))
width = 0.35
bars_t = ax1.bar(x_eval - width/2, text_eval_format, width, color=c_text, label=l_text, edgecolor="white")
bars_m = ax1.bar(x_eval + width/2, mlp_eval_format, width, color=c_mlp, label=l_mlp, edgecolor="white")

annotate_bars(ax1, bars_t, text_eval_format, fmt="{:.2f}", offset=0.01)
annotate_bars(ax1, bars_m, mlp_eval_format, fmt="{:.2f}", offset=0.01)

ax1.set_xlabel("Eval Step")
ax1.set_ylabel("Format Rate")
ax1.set_title("Format Compliance Rate")
ax1.set_xticks(x_eval)
ax1.set_xticklabels([f"Step {s}" for s in eval_steps])
ax1.set_ylim(0, 1.15)
ax1.legend(loc="lower left")
style.clean_axes(ax1)

# Right: training loss (PG loss)
ax2.plot(steps, text_loss, color=c_text, label=l_text, alpha=0.8, linewidth=1.5)
ax2.plot(steps, mlp_loss, color=c_mlp, label=l_mlp, alpha=0.8, linewidth=1.5)
ax2.axhline(y=0, color="black", linestyle=":", alpha=0.3, linewidth=0.8)
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Policy Gradient Loss")
ax2.set_title("Training Loss (PG Loss)")
ax2.legend(loc="lower left")
style.clean_axes(ax2)

fig.suptitle("GRPO Structure Quality: Format Rate & Loss", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
saved = save_main_figure(fig, "fig15_grpo_structure_format_rate")
print(f"Figure 15 saved: {saved}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 16: Gradient Norm Diagnostic (2-panel)
# ═══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: LoRA grad norms
ax1.plot(steps, text_grad_lora, color=c_text, label=f"{l_text} LoRA", alpha=0.8, linewidth=1.5)
# MLP LoRA grads are 0 — plot as near-zero on log scale
mlp_lora_display = np.full_like(mlp_grad_lora, 1e-10)
ax1.plot(steps, mlp_lora_display, color=c_mlp, label=f"{l_mlp} LoRA", alpha=0.8,
         linewidth=1.5, linestyle="--")

ax1.set_yscale("log")
ax1.set_ylim(1e-12, 5)
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Normalized Gradient Norm (log)")
ax1.set_title("LoRA Gradient Norms")
ax1.legend(loc="center right")
style.clean_axes(ax1)

# Annotation
ax1.annotate("MLP LoRA grad = 0\n(cannot learn format)",
             xy=(45, 1e-10), xytext=(55, 1e-5),
             fontsize=9, color=c_mlp, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=c_mlp, lw=1.2),
             ha="center")

# Right: Multimodal grad norms
ax2.plot(steps, mlp_grad_mm, color=c_mlp, label=f"{l_mlp} MM", alpha=0.8, linewidth=1.5)
text_mm_display = np.full_like(text_grad_mm, 1e-10)
ax2.plot(steps, text_mm_display, color=c_text, label=f"{l_text} MM", alpha=0.8,
         linewidth=1.5, linestyle="--")

ax2.set_yscale("log")
ax2.set_ylim(1e-12, 5)
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Normalized Gradient Norm (log)")
ax2.set_title("Multimodal Gradient Norms")
ax2.legend(loc="center right")
style.clean_axes(ax2)

ax2.annotate("Text has no MM params",
             xy=(45, 1e-10), xytext=(55, 1e-5),
             fontsize=9, color=c_text, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=c_text, lw=1.2),
             ha="center")

fig.suptitle("Gradient Flow Diagnostic: LoRA vs Multimodal", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
saved = save_main_figure(fig, "fig16_grpo_structure_grad_norms")
print(f"Figure 16 saved: {saved}")

print("\nAll 4 figures created successfully.")
