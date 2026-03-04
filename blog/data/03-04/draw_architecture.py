"""
Architecture comparison diagram: Text vs ESM-3 (MLP/Perceiver/Flamingo)
Shows how protein information enters the token stream differently.

Key insight: Text approach tokenizes protein as literal text tokens.
ESM approaches replace a <|protein_embed|> placeholder with projected embeddings.

Usage:
    python blog/data/03-04/draw_architecture.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────
C = {
    # Fills
    "frozen":      "#B8D4E8",
    "trainable":   "#FFD49B",
    "llm":         "#C8B8E8",
    "lora":        "#E0D0F0",
    "input":       "#E8E8E8",
    "output":      "#D4E8D4",
    "xattn":       "#FFB3B3",
    "text_tok":    "#E0E0F0",  # text token
    "prot_tok":    "#FFE0B0",  # protein token (from ESM)
    "prot_text":   "#C8E8C8",  # protein as text token
    "placeholder": "#FFD0D0",  # placeholder token

    # Borders
    "frozen_e":    "#4472C4",
    "train_e":     "#D46A00",
    "llm_e":       "#7B5EA7",
    "input_e":     "#999999",
    "output_e":    "#5A8A5A",
    "xattn_e":     "#CC5555",

    # Other
    "text_dark":   "#1a1a1a",
    "arrow":       "#444444",
    "arrow_lt":    "#999999",
    "muted":       "#888888",
}


# ── Drawing primitives ─────────────────────────────────────────────────

def block(ax, x, y, w, h, label, sub=None,
          fc="#B8D4E8", ec="#4472C4", fs=9, lw=1.8,
          ls="-", alpha=1.0, zorder=3):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.03",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    if sub:
        ax.text(x, y + 0.06, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["text_dark"], zorder=zorder+1)
        ax.text(x, y - 0.08, sub, ha="center", va="center",
                fontsize=fs - 1.5, color="#666", style="italic", zorder=zorder+1)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["text_dark"], zorder=zorder+1)


def arrow(ax, x0, y0, x1, y1, color="#444", lw=1.5,
          style="-|>", ls="-", cs="arc3,rad=0"):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color, linewidth=lw,
        linestyle=ls, connectionstyle=cs, mutation_scale=13, zorder=5,
    )
    ax.add_patch(a)


def label(ax, x, y, text, fs=7, color="#888", **kw):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color, zorder=4, **kw)


def dashed_group(ax, x, y, w, h, text=None, color="#aaa"):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.06",
        facecolor="none", edgecolor=color,
        linewidth=1.0, linestyle="--", zorder=1,
    )
    ax.add_patch(box)
    if text:
        ax.text(x, y + h/2 + 0.06, text, ha="center", va="bottom",
                fontsize=6.5, color=color, style="italic", zorder=2)


def draw_token_bar(ax, x, y, tokens, colors, w_each=0.22, h=0.22):
    """Draw a horizontal bar of tokens (like a sequence of colored boxes)."""
    total_w = len(tokens) * w_each
    start_x = x - total_w / 2
    for i, (tok, col) in enumerate(zip(tokens, colors)):
        tx = start_x + i * w_each + w_each / 2
        box = FancyBboxPatch(
            (tx - w_each/2 + 0.01, y - h/2), w_each - 0.02, h,
            boxstyle="round,pad=0.01",
            facecolor=col, edgecolor="#999999",
            linewidth=0.8, zorder=6,
        )
        ax.add_patch(box)
        ax.text(tx, y, tok, ha="center", va="center",
                fontsize=5, color=C["text_dark"], zorder=7,
                fontweight="bold" if tok in ["P₁", "P₂", "...P_N"] else "normal")


# ── Architecture: Text-Only ────────────────────────────────────────────

def draw_text(ax):
    bw = 1.6

    # Input protein
    block(ax, 0, 0.0, bw, 0.32, "Protein: MKTL...AVFG",
          fc=C["input"], ec=C["input_e"], fs=8)

    # Tokenizer
    arrow(ax, 0, 0.16, 0, 0.42)
    block(ax, 0, 0.60, bw, 0.28, "Tokenizer",
          sub="chars → token IDs",
          fc=C["input"], ec=C["input_e"], fs=8)

    # Show token stream
    arrow(ax, 0, 0.75, 0, 1.02)
    label(ax, 0, 0.90, "Token sequence:", fs=6.5, color="#555")

    # Token bar — all text tokens including protein as text
    tokens = ["sys", "user", "<prot>", "M", "K", "T", "L", "...", "</prot>", "asst"]
    colors = [C["text_tok"]] * 2 + [C["prot_text"]] * 7 + [C["text_tok"]]
    draw_token_bar(ax, 0, 1.12, tokens, colors, w_each=0.155, h=0.18)

    label(ax, 0, 1.30, "protein is literal text tokens (can be 100s of tokens)",
          fs=5.5, color=C["train_e"], fontweight="bold")

    # embed_tokens
    arrow(ax, 0, 1.22, 0, 1.52)
    block(ax, 0, 1.68, bw, 0.25, "embed_tokens()",
          sub="vocab lookup → d_model",
          fc=C["text_tok"], ec="#8888CC", fs=8)

    # LLM
    arrow(ax, 0, 1.82, 0, 2.20)
    block(ax, 0, 2.55, bw, 0.62, "LLM + LoRA",
          sub="Qwen3-4B",
          fc=C["llm"], ec=C["llm_e"], fs=11)

    # Output
    arrow(ax, 0, 2.87, 0, 3.10)
    block(ax, 0, 3.28, bw, 0.28, "Text Output",
          fc=C["output"], ec=C["output_e"], fs=9)

    label(ax, 0, -0.30, "~2M trainable (LoRA only)", fs=7, color=C["muted"])


# ── Architecture: MLP ──────────────────────────────────────────────────

def draw_mlp(ax):
    bw = 1.6

    # Input
    block(ax, 0, 0.0, bw, 0.32, "Protein: MKTL...AVFG",
          fc=C["input"], ec=C["input_e"], fs=8)

    # ESM-3
    arrow(ax, 0, 0.16, 0, 0.42)
    block(ax, 0, 0.65, bw, 0.38, "ESM-3 Encoder",
          sub="frozen, 1536-dim",
          fc=C["frozen"], ec=C["frozen_e"], ls="--")

    # Attention Pooling
    arrow(ax, 0, 0.85, 0, 1.05)
    block(ax, 0, 1.20, bw, 0.24, "Attention Pooling",
          sub="L residues → 32 tokens",
          fc=C["trainable"], ec=C["train_e"], fs=8)

    # MLP Projector
    arrow(ax, 0, 1.33, 0, 1.48)
    block(ax, 0, 1.62, bw, 0.22, "MLP Projector",
          sub="1536 → 5120 → 2560",
          fc=C["trainable"], ec=C["train_e"], fs=8)

    # Show the token replacement
    arrow(ax, 0, 1.74, 0, 1.95)
    label(ax, 0, 1.87, "Token sequence (after replacement):", fs=6.5, color="#555")

    # Token bar — placeholder replaced with protein embeddings
    tokens = ["sys", "user", "P₁", "P₂", "...", "P₃₂", "asst"]
    colors = ([C["text_tok"]] * 2 +
              [C["prot_tok"]] * 4 +
              [C["text_tok"]])
    draw_token_bar(ax, 0, 2.04, tokens, colors, w_each=0.20, h=0.18)

    label(ax, 0, 2.20, "<|protein_embed|> replaced by 32 projected embeddings",
          fs=5.5, color=C["train_e"], fontweight="bold")

    # LLM
    arrow(ax, 0, 2.14, 0, 2.40)
    block(ax, 0, 2.68, bw, 0.48, "LLM + LoRA",
          sub="Qwen3-4B",
          fc=C["llm"], ec=C["llm_e"], fs=11)

    # Output
    arrow(ax, 0, 2.93, 0, 3.10)
    block(ax, 0, 3.28, bw, 0.28, "Text Output",
          fc=C["output"], ec=C["output_e"], fs=9)

    # Trainable group
    dashed_group(ax, 0, 1.41, 1.75, 0.72, "Trainable (~30.5M)", color=C["train_e"])
    label(ax, 0, -0.30, "~32.5M trainable total", fs=7, color=C["muted"])


# ── Architecture: Perceiver ────────────────────────────────────────────

def draw_perceiver(ax):
    bw = 1.6

    # Input
    block(ax, 0, 0.0, bw, 0.32, "Protein: MKTL...AVFG",
          fc=C["input"], ec=C["input_e"], fs=8)

    # ESM-3
    arrow(ax, 0, 0.16, 0, 0.42)
    block(ax, 0, 0.65, bw, 0.38, "ESM-3 Encoder",
          sub="frozen, 1536-dim",
          fc=C["frozen"], ec=C["frozen_e"], ls="--")

    # Perceiver Resampler
    arrow(ax, 0, 0.85, 0, 1.10)
    block(ax, 0, 1.40, bw, 0.50, "Perceiver Resampler",
          sub="2 layers, latent=1024\n32 queries → 2560-dim",
          fc=C["trainable"], ec=C["train_e"], fs=9)

    # Token replacement
    arrow(ax, 0, 1.66, 0, 1.95)
    label(ax, 0, 1.87, "Token sequence (after replacement):", fs=6.5, color="#555")

    tokens = ["sys", "user", "P₁", "P₂", "...", "P₃₂", "asst"]
    colors = ([C["text_tok"]] * 2 +
              [C["prot_tok"]] * 4 +
              [C["text_tok"]])
    draw_token_bar(ax, 0, 2.04, tokens, colors, w_each=0.20, h=0.18)

    label(ax, 0, 2.20, "<|protein_embed|> replaced by 32 resampled embeddings",
          fs=5.5, color=C["train_e"], fontweight="bold")

    # LLM
    arrow(ax, 0, 2.14, 0, 2.40)
    block(ax, 0, 2.68, bw, 0.48, "LLM + LoRA",
          sub="Qwen3-4B",
          fc=C["llm"], ec=C["llm_e"], fs=11)

    # Output
    arrow(ax, 0, 2.93, 0, 3.10)
    block(ax, 0, 3.28, bw, 0.28, "Text Output",
          fc=C["output"], ec=C["output_e"], fs=9)

    dashed_group(ax, 0, 1.40, 1.75, 0.65, "Trainable (~29.4M)", color=C["train_e"])
    label(ax, 0, -0.30, "~31.4M trainable total", fs=7, color=C["muted"])


# ── Architecture: Flamingo ─────────────────────────────────────────────

def draw_flamingo(ax):
    bw = 1.6

    # Input
    block(ax, 0, 0.0, bw, 0.32, "Protein: MKTL...AVFG",
          fc=C["input"], ec=C["input_e"], fs=8)

    # ESM-3
    arrow(ax, 0, 0.16, 0, 0.42)
    block(ax, 0, 0.65, bw, 0.38, "ESM-3 Encoder",
          sub="frozen, 1536-dim",
          fc=C["frozen"], ec=C["frozen_e"], ls="--")

    # Flamingo Perceiver
    arrow(ax, 0, 0.85, 0, 1.10)
    block(ax, 0, 1.40, bw, 0.50, "Perceiver Resampler",
          sub="6 layers, latent=1024\n64 queries",
          fc=C["trainable"], ec=C["train_e"], fs=9)

    dashed_group(ax, 0, 1.40, 1.75, 0.65, "Trainable (~50-60M)", color=C["train_e"])

    # The key difference: NO prefix injection
    # Instead, visual tokens feed into cross-attention at every 4th LLM layer
    arrow(ax, 0, 1.66, 0, 1.88)
    label(ax, 0, 1.80, "visual tokens (no prefix injection)", fs=6, color=C["train_e"])

    # LLM stack with interleaved xattn
    lh = 0.20
    gap = 0.26
    y0 = 2.02

    # Layer pair 1: LM + XAttn
    block(ax, 0, y0, bw, lh, "LM Self-Attention (frozen)",
          fc=C["frozen"], ec=C["frozen_e"], fs=7, lw=1.2, ls="--")

    y0 += gap
    arrow(ax, 0, y0 - gap + lh/2 + 0.01, 0, y0 - lh/2 - 0.01,
          color=C["arrow_lt"], lw=1.0)
    block(ax, 0, y0, bw, lh, "Gated Cross-Attn  ← visual tokens",
          fc=C["xattn"], ec=C["xattn_e"], fs=7, lw=1.2)
    ax.text(0.86, y0, "tanh(0)", ha="left", va="center",
            fontsize=5.5, color=C["xattn_e"], style="italic", zorder=6)

    # Layer pair 2
    y0 += gap
    arrow(ax, 0, y0 - gap + lh/2 + 0.01, 0, y0 - lh/2 - 0.01,
          color=C["arrow_lt"], lw=1.0)
    block(ax, 0, y0, bw, lh, "LM Self-Attention (frozen)",
          fc=C["frozen"], ec=C["frozen_e"], fs=7, lw=1.2, ls="--")

    y0 += gap
    arrow(ax, 0, y0 - gap + lh/2 + 0.01, 0, y0 - lh/2 - 0.01,
          color=C["arrow_lt"], lw=1.0)
    block(ax, 0, y0, bw, lh, "Gated Cross-Attn  ← visual tokens",
          fc=C["xattn"], ec=C["xattn_e"], fs=7, lw=1.2)

    # Dots
    y0 += gap * 0.65
    ax.text(0, y0, "· · ·  (every 4th layer)", ha="center", va="center",
            fontsize=8, color=C["muted"], zorder=4)

    # Group
    stack_top = y0 + 0.12
    stack_bot = 2.02 - lh/2 - 0.08
    dashed_group(ax, 0, (stack_top + stack_bot)/2,
                 1.75, stack_top - stack_bot,
                 "Frozen LLM + Gated XAttn (~70-90M)", color=C["muted"])

    # Output
    y0 += gap * 0.65
    arrow(ax, 0, y0, 0, y0 + 0.16)
    block(ax, 0, y0 + 0.32, bw, 0.28, "Text Output",
          fc=C["output"], ec=C["output_e"], fs=9)

    label(ax, 0, -0.30, "~120-150M trainable (no LoRA)", fs=7, color=C["muted"])


# ── Main ───────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(1, 4, figsize=(24, 10),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})

    approaches = [
        ("(a) Text-Only", draw_text),
        ("(b) ESM-3 + MLP Projector", draw_mlp),
        ("(c) ESM-3 + Perceiver Resampler", draw_perceiver),
        ("(d) ESM-3 + Flamingo", draw_flamingo),
    ]

    for ax, (title, drawer) in zip(axes, approaches):
        drawer(ax)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.55, 4.00)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Legend ──────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=C["frozen"], edgecolor=C["frozen_e"],
                       linewidth=1.5, linestyle="--", label="Frozen (pretrained)"),
        mpatches.Patch(facecolor=C["trainable"], edgecolor=C["train_e"],
                       linewidth=1.5, label="Trainable"),
        mpatches.Patch(facecolor=C["llm"], edgecolor=C["llm_e"],
                       linewidth=1.5, label="LLM + LoRA"),
        mpatches.Patch(facecolor=C["xattn"], edgecolor=C["xattn_e"],
                       linewidth=1.5, label="Gated Cross-Attention"),
        mpatches.Patch(facecolor=C["prot_text"], edgecolor="#999",
                       linewidth=1.0, label="Protein as text tokens"),
        mpatches.Patch(facecolor=C["prot_tok"], edgecolor="#999",
                       linewidth=1.0, label="Protein embeddings (from ESM-3)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.035),
               fancybox=True, shadow=False, edgecolor="#cccccc")

    # FSDP note
    fig.text(0.5, 0.015,
             "All approaches support FSDP (Fully Sharded Data Parallel) "
             "for efficient multi-GPU training and scalability to larger LLM backbones.",
             ha="center", va="center", fontsize=9, color="#555555", style="italic")

    fig.suptitle("Protein LLM Architecture Comparison: Four Approaches",
                 fontsize=16, fontweight="bold", y=0.97)

    plt.tight_layout(rect=[0, 0.07, 1, 0.94])

    out = "/home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM/blog/figures/architecture_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved: {out}")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved: {out.replace('.png', '.pdf')}")
    plt.close()


if __name__ == "__main__":
    main()
