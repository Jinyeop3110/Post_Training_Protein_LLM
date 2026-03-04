# Scientist Agent Team: Training Diagnostics

> **Purpose**: Analyze training experiments, create diagnostic plots, and produce concise reports. Read-only with respect to source code — only writes to `blog/`.

---

## Quick Start

```bash
# 1. Enable agent teams
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

# 2. Start Claude Code
claude

# 3. Request scientist team
> Create a scientist team with 3 teammates:
> - data-collector: fetch metrics from wandb and local experiment files
> - analyst: create diagnostic plots and statistical analysis
> - reporter: write markdown report with embedded figures
>
> Question: "Compare loss curves for MLP vs text-only SFT"
```

---

## Team Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                 YOU                                     │
│                  (Human - Asks analysis questions)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │      TEAM LEAD      │
                        │                     │
                        │ • YOUR interface    │
                        │ • Scopes question   │
                        │ • Coordinates flow  │
                        │ • Delivers report   │
                        └──────────┬──────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
┌──────────────┐         ┌──────────────┐            ┌──────────────┐
│DATA-COLLECTOR│         │   ANALYST    │            │   REPORTER   │
│              │         │              │            │              │
│ • wandb API  │────────>│ • Loss plots │───────────>│ • post.html  │
│ • Local files│  CSVs   │ • Grad norms │   PNGs     │ • Figures    │
│ • Metadata   │  JSONs  │ • LR sched   │   JSON     │ • Findings   │
│ • Configs    │         │ • Statistics  │            │ • Recs       │
└──────────────┘         └──────────────┘            └──────────────┘
       │                         │                          │
       └─────────────────────────┴──────────────────────────┘
                                 │
                  blog/
                  ├── data/MM-DD/  (CSVs, JSONs)
                  ├── figures/     (PNGs)
                  └── posts/       (HTML reports)
```

---

## Agent Specifications

### 1. DATA-COLLECTOR

**Focus**: Fetch training metrics from wandb API and local experiment files

**Responsibilities**:
- Query `wandb.Api()` for run histories from protein-LLM projects
- Read local `trainer_state.json`, `metrics.json`, `training_args.json`, `lineage.json`
- Parse HF Trainer log history from `trainer_state.json`
- Output organized CSVs and JSONs to `blog/data/MM-DD/`
- Collect and normalize experiment metadata (approach, model, LR, epochs, etc.)

**Key Files Read**:
```
results/{experiment_name}/
├── config.yaml            # Full Hydra config
├── lineage.json           # Stage, approach, parent, timestamps
├── training_args.json     # Hyperparameters
├── metrics.json           # Final summary metrics
├── checkpoints/
│   ├── trainer_state.json           # Full training log history
│   └── checkpoint-*/
│       └── trainer_state.json       # Per-checkpoint state
└── train.log              # Raw training output
```

**Output Format**:
```
blog/data/MM-DD/
├── run_histories.csv           # Step-level: step, loss, eval_loss, lr, grad_norm, ...
├── experiment_metadata.json    # Per-run: name, approach, model, LR, epochs, ...
└── wandb_summaries.json        # wandb run summaries (if available)
```

**Critical Rules**:
- NEVER write outside `blog/`
- NEVER modify source code or experiment files
- Distinguish `loss` (HF running average) from `token_avg_loss` (true average)
- Always include `approach` field (text/esm3) and `projector_type` (mlp/perceiver) in metadata

**Spawn Prompt**:
```
You are the data-collector agent for the protein-LLM scientist team.

FIRST: Read SCIENTIST_TEAM.md for team context, then CLAUDE.md for project context.

Your job: Gather training metrics and experiment metadata.

Data sources (in priority order):
1. Local trainer_state.json — log_history field has per-step metrics
2. Local metrics.json — final summary metrics
3. Local lineage.json — experiment metadata (approach, model, timestamps)
4. Local training_args.json — hyperparameters
5. Local config.yaml — full Hydra resolved config
6. wandb API — if local data is incomplete

trainer_state.json log_history fields:
- loss: HF Trainer running average (inflated by early high losses — DO NOT use for plots)
- token_avg_loss: true per-token average loss (USE THIS)
- eval_loss: validation loss (computed periodically)
- learning_rate: current LR
- grad_norm: gradient norm
- epoch: fractional epoch
- step: global step

Output to: blog/data/MM-DD/ (where MM-DD is today's date)
Format: CSV for time series, JSON for metadata

CRITICAL: NEVER write outside blog/. NEVER modify experiment files.
```

---

### 2. ANALYST

**Focus**: Create diagnostic plots and statistical analysis from collected data

**Responsibilities**:
- Create matplotlib/seaborn plots (headless: `matplotlib.use('Agg')`)
- Standard plot catalog: loss curves, gradient norms, LR schedule, convergence comparison
- Compute summary statistics: min/max/final loss, convergence step, gradient stats
- Detect anomalies: NaN occurrences, loss spikes, gradient explosions
- Output PNGs to `blog/figures/`
- Output `analysis_summary.json` to `blog/data/MM-DD/`

**Standard Plot Catalog**:
| Plot | X-axis | Y-axis | Notes |
|------|--------|--------|-------|
| `loss_curves.png` | Step | token_avg_loss | All runs overlaid |
| `eval_loss_curves.png` | Step | eval_loss | Validation curves |
| `gradient_norms.png` | Step | grad_norm | Log scale Y |
| `lr_schedule.png` | Step | learning_rate | Per run |
| `loss_comparison_bar.png` | Experiment | Final eval_loss | Bar chart |
| `convergence_table.png` | — | — | Rendered table image |
| `gpu_memory.png` | Experiment | GB | Allocated vs reserved |

**Style Requirements**:
```python
import matplotlib
matplotlib.use('Agg')  # Headless — MUST be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")
APPROACH_COLORS = {"mlp": "#1f77b4", "perceiver": "#ff7f0e", "text": "#2ca02c"}
FIG_DPI = 150
FIG_SIZE = (10, 6)  # Default
```

**Output Format**:
```
blog/figures/
├── loss_curves.png
├── eval_loss_curves.png
├── gradient_norms.png
├── lr_schedule.png
├── loss_comparison_bar.png
└── convergence_table.png

blog/data/MM-DD/
└── analysis_summary.json     # Per-experiment stats + anomalies
```

**analysis_summary.json schema**:
```json
{
  "experiments": {
    "<name>": {
      "approach": "esm3",
      "projector_type": "mlp",
      "final_train_loss": 2.49,
      "best_eval_loss": 3.64,
      "best_eval_step": 200,
      "total_steps": 7815,
      "convergence_step": 150,
      "max_grad_norm": 1.2,
      "anomalies": ["loss_spike_step_530"]
    }
  },
  "comparison": {
    "best_experiment": "<name>",
    "metric": "best_eval_loss"
  }
}
```

**Critical Rules**:
- ALWAYS use `token_avg_loss`, NOT `loss` (HF running average is misleading)
- Use `matplotlib.use('Agg')` BEFORE importing pyplot
- 150 DPI PNGs, consistent color scheme
- NEVER write outside `blog/`
- Include legend with experiment names and approach type

**Spawn Prompt**:
```
You are the analyst agent for the protein-LLM scientist team.

FIRST: Read SCIENTIST_TEAM.md for team context, then CLAUDE.md for project context.

Your job: Create diagnostic plots and statistical analysis from training data.

CRITICAL setup (MUST be first lines of any plotting code):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set_theme(style="whitegrid", palette="colorblind")

Approach color scheme:
  MLP = "#1f77b4" (blue), Perceiver = "#ff7f0e" (orange), Text = "#2ca02c" (green)

Standard DPI: 150. Default figure size: (10, 6).

CRITICAL: Use 'token_avg_loss' for loss plots, NOT 'loss' (which is HF running average
and is heavily inflated by early high losses). 'eval_loss' is always reliable.

Input: CSVs and JSONs from blog/data/MM-DD/ (produced by data-collector)
Output: PNGs to blog/figures/
        analysis_summary.json to blog/data/MM-DD/

Every plot must have:
- Title, axis labels, legend
- Consistent color scheme by approach type
- Grid lines (via seaborn whitegrid)
- Saved as PNG at 150 DPI

Also produce analysis_summary.json with per-experiment stats and anomaly flags.

CRITICAL: NEVER write outside blog/. NEVER modify source code or experiment files.
```

---

### 3. REPORTER

**Focus**: Write concise markdown analysis reports with embedded figures

**Responsibilities**:
- Synthesize data-collector's metadata and analyst's findings into a coherent report
- Follow standard report template (see below)
- Embed PNGs via relative paths (`../figures/name.png` from posts)
- Use scientific tone: specific numbers, not vague qualifiers
- Save HTML to `blog/posts/YYYY-MM-DD_title-in-kebab-case.html`

**Report Template**:
```markdown
# {Report Title}

**Date**: YYYY-MM-DD
**Question**: {The original analysis question}
**Experiments analyzed**: {list}

## Executive Summary

{2-3 sentences: key finding, best performer, notable issues}

## Methodology

- **Data source**: {local trainer_state.json / wandb / both}
- **Metrics**: {which metrics were compared}
- **Experiments**: {N} runs spanning {date range}

## Experiment Configuration

| Parameter | {Exp 1} | {Exp 2} | ... |
|-----------|---------|---------|-----|
| Approach | esm3 | text | ... |
| Projector | mlp | — | ... |
| Base model | Qwen3-8B | Qwen3-8B | ... |
| LR | 2e-4 | 2e-4 | ... |
| Epochs | 3 | 3 | ... |

## Key Findings

### 1. {Finding title}

{Description with specific numbers}

![Loss Curves](../figures/loss_curves.png)

### 2. {Finding title}

{Description}

![Gradient Norms](../figures/gradient_norms.png)

## Summary Results

| Metric | {Exp 1} | {Exp 2} | ... |
|--------|---------|---------|-----|
| Final train loss | 2.49 | 2.53 | ... |
| Best eval loss | 3.64 | 3.71 | ... |
| Convergence step | 150 | 180 | ... |

## Recommendations

1. {Actionable recommendation}
2. {Actionable recommendation}
```

**Critical Rules**:
- NEVER write outside `blog/`
- NEVER modify source code
- Use RELATIVE paths for figure references (e.g., `../figures/loss_curves.png` from posts)
- Numbers over vague qualifiers ("eval loss decreased 8.2%" not "loss improved significantly")
- Always state which loss metric was used (token_avg_loss vs eval_loss)
- Include experiment names in full (for reproducibility)

**Spawn Prompt**:
```
You are the reporter agent for the protein-LLM scientist team.

FIRST: Read SCIENTIST_TEAM.md for team context, then CLAUDE.md for project context.

Your job: Write a concise HTML blog post synthesizing the analysis.

Input:
- blog/data/MM-DD/experiment_metadata.json (from data-collector)
- blog/data/MM-DD/analysis_summary.json (from analyst)
- blog/figures/*.png (from analyst)

Output: blog/posts/YYYY-MM-DD_title-in-kebab-case.html
Also regenerate blog/index.html to include the new post.

Report structure:
1. Executive Summary (2-3 sentences)
2. Methodology (data sources, metrics, scope)
3. Experiment Configuration table
4. Key Findings (with embedded figures using relative paths)
5. Summary Results table
6. Recommendations (actionable)

Style rules:
- Scientific tone, specific numbers ("eval loss 3.64" not "good performance")
- Relative paths for figures: <img src="../figures/filename.png">
- Always state which loss metric is used
- Include full experiment names for reproducibility
- Keep it concise: aim for 200-400 lines

CRITICAL: NEVER write outside blog/. NEVER modify source code.
```

---

## Workflow Patterns

### Pattern 1: Single-Run Diagnostics

```
User: "Analyze the MLP SFT run sft_lora_esm3_qwen3_8b_it_0227_022604"

Lead → data-collector: Fetch all metrics for this single run
       data-collector → blog/data/MM-DD/
Lead → analyst:        Plot loss curve, grad norms, LR schedule for single run
       analyst → blog/figures/
Lead → reporter:       Write single-run diagnostic HTML post
       reporter → blog/posts/YYYY-MM-DD_mlp-sft-diagnostics.html
Lead → User:           "Report ready at blog/posts/YYYY-MM-DD_mlp-sft-diagnostics.html"
```

### Pattern 2: Multi-Run Comparison

```
User: "Compare loss curves for MLP vs text-only SFT"

Lead → data-collector: Fetch metrics for all MLP and text-only runs
       data-collector → blog/data/MM-DD/
Lead → analyst:        Overlay loss curves, create comparison bar chart
       analyst → blog/figures/
Lead → reporter:       Write comparison HTML post with recommendations
       reporter → blog/posts/YYYY-MM-DD_mlp-vs-text-sft-comparison.html
Lead → User:           "Report ready. Key finding: MLP achieves 8% lower eval loss."
```

### Pattern 3: Anomaly Investigation

```
User: "Investigate the NaN issue in the 0225 runs"

Lead → data-collector: Fetch all 0225 runs, focus on gradient norms and loss near NaN
       data-collector → blog/data/MM-DD/
Lead → analyst:        Plot gradient norms with NaN markers, loss before/after spike
       analyst → blog/figures/
Lead → reporter:       Document root cause, affected runs, resolution
       reporter → blog/posts/YYYY-MM-DD_nan-investigation.html
Lead → User:           "Root cause: multimodal params not clipped. See report."
```

### Pattern 4: Periodic Health Check

```
User: "Give me a health check on all completed experiments"

Lead → data-collector: Fetch metadata and final metrics for ALL experiments
       data-collector → blog/data/MM-DD/
Lead → analyst:        Create summary dashboard: bar charts, convergence comparison
       analyst → blog/figures/
Lead → reporter:       Write overview with experiment status table
       reporter → blog/posts/YYYY-MM-DD_health-check.html
Lead → User:           "9 experiments analyzed. 6 converged, 3 had issues. See report."
```

---

## Data Sources Reference

### Local Experiment Files

All experiments are stored under `results/{experiment_name}/`:

| File | Contents | Key Fields |
|------|----------|------------|
| `lineage.json` | Experiment identity | `approach`, `projector_type`, `base_model`, `stage`, `created_at`, `completed_at` |
| `training_args.json` | Hyperparameters | `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`, `projector_lr` |
| `metrics.json` | Final summary | `train_loss`, `token_avg_loss`, `train_runtime`, `gpu_memory_*` |
| `config.yaml` | Full Hydra config | Everything (model, training, data, encoder) |
| `checkpoints/trainer_state.json` | Step-by-step log | `log_history` array with per-step metrics |

### trainer_state.json Field Catalog

The `log_history` array contains objects with these fields (not all present at every step):

**Training steps** (every `logging_steps`):
| Field | Description | Notes |
|-------|-------------|-------|
| `loss` | HF Trainer running average | **DO NOT USE for plots** — inflated by early high losses |
| `token_avg_loss` | True per-token average loss | **USE THIS** for training loss |
| `grad_norm` | Gradient L2 norm | Log-scale for plots |
| `learning_rate` | Current LR | Shows warmup + decay schedule |
| `epoch` | Fractional epoch | `1.5` = halfway through epoch 2 |
| `step` | Global step count | X-axis for most plots |

**Evaluation steps** (every `eval_steps`):
| Field | Description | Notes |
|-------|-------------|-------|
| `eval_loss` | Validation loss | Most reliable performance metric |
| `eval_runtime` | Eval duration (seconds) | |
| `eval_samples_per_second` | Throughput | |

### wandb Projects

| Project | Contents |
|---------|----------|
| `protein-llm-sft` | SFT training runs (loss, eval_loss, LR, grad_norm) |
| `protein-llm-rl` | GRPO training runs (reward, policy loss) |

API access:
```python
import wandb
api = wandb.Api()
runs = api.runs("protein-llm-sft")
for run in runs:
    history = run.history()  # DataFrame
    config = run.config       # Dict
    summary = run.summary     # Dict
```

### Available Experiments (as of 2026-03-01)

| Experiment | Approach | Status | Notes |
|------------|----------|--------|-------|
| `sft_lora_esm3_qwen3_8b_it_0225_203237` | esm3/mlp | Partial | |
| `sft_lora_esm3_qwen3_8b_it_0226_151416` | esm3/mlp | Complete | Multiple checkpoints (1000, 1250, 1500) |
| `sft_lora_esm3_qwen3_8b_it_0227_022604` | esm3/mlp | Complete | Latest MLP run |
| `sft_text_qwen3_8b_it_0227_115556` | text | Partial | |
| `sft_text_qwen3_8b_it_0227_115751` | text | Partial | |
| `sft_text_qwen3_8b_it_0227_145821` | text | Complete | Full text-only run with metrics |

---

## Reports Directory Convention

**Output directory**:
```
/home/yeopjin/orcd/pool/workspace/Post_Training_Protein_LLM/blog
```

All agents MUST write to this absolute path, following the blog conventions in `blog/README.md`:

```
blog/
├── index.html                           # Blog index page (auto-generated)
├── README.md                            # Conventions and reading order
├── posts/                               # HTML blog posts
│   └── YYYY-MM-DD_title-in-kebab-case.html
├── figures/                             # All plot images (PNGs)
│   └── descriptive_figure_name.png
└── data/                                # Analysis code + data by date
    └── MM-DD/
        ├── analysis_script.py
        ├── run_histories.csv
        ├── experiment_metadata.json
        └── analysis_summary.json
```

**Blog post conventions**:
- Posts are HTML files in `blog/posts/`
- Filename: `YYYY-MM-DD_title-in-kebab-case.html`
- Figures in `blog/figures/`, referenced from posts as `../figures/name.png`
- Data in `blog/data/MM-DD/`, referenced from posts as `../data/MM-DD/`
- Index at `blog/index.html` links to all posts via `posts/filename.html`
- Tags: kickoff, architecture, training, evaluation, data, rl, sft, infrastructure, milestone, debug

**CRITICAL**: Never overwrite or delete existing files in `blog/`. Always create new dated content.

---

## Agent Communication

### Communication Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                 YOU                                     │
│                       (Only talks to Team Lead)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ Questions, Reports
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             TEAM LEAD                                   │
│                                                                         │
│  • Receives YOUR analysis questions                                     │
│  • Scopes question into report name and experiment list                 │
│  • Coordinates sequential pipeline: collect → analyze → report          │
│  • Delivers final report to YOU                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ Data, Figures, Report
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             TEAMMATES                                   │
│                                                                         │
│  data-collector ──→ analyst ──→ reporter                                │
│   (sequential pipeline — each depends on the previous)                  │
│                                                                         │
│  • Report results to TEAM LEAD                                          │
│  • Do NOT contact YOU directly                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Communication Rules

| From | To | Allowed? | How |
|------|----|----------|-----|
| **You** | Team Lead | Yes | Direct conversation |
| **Team Lead** | You | Yes | Progress reports, final report delivery |
| **Team Lead** | Teammates | Yes | Task assignment with question_name and experiment list |
| **Teammates** | Team Lead | Yes | Completion notifications, blockers |
| **Teammates** | Each other | Yes | Direct messages (e.g., data-collector → analyst handoff) |
| **Teammates** | You | No | Must go through Team Lead |

### Pipeline Coordination

Tasks are **sequential with dependencies**:

```
1. data-collector: Fetch data     [no dependencies]
   └─> Creates: blog/data/MM-DD/run_histories.csv
   └─> Creates: blog/data/MM-DD/experiment_metadata.json

2. analyst: Create plots          [blocked by: data-collector]
   └─> Reads:   blog/data/MM-DD/*
   └─> Creates: blog/figures/*.png
   └─> Creates: blog/data/MM-DD/analysis_summary.json

3. reporter: Write HTML post      [blocked by: analyst]
   └─> Reads:   blog/data/MM-DD/*, blog/figures/*
   └─> Creates: blog/posts/YYYY-MM-DD_title.html
```

---

## End-to-End Worked Example

**Question**: "Compare loss curves for MLP vs text-only SFT"

### Step 1: Lead scopes the question
- Post slug: `mlp-vs-text-sft-comparison`
- Experiments: `sft_lora_esm3_qwen3_8b_it_0227_022604` (MLP), `sft_text_qwen3_8b_it_0227_145821` (text)
- Data goes to `blog/data/MM-DD/`, figures to `blog/figures/`, post to `blog/posts/`

### Step 2: data-collector gathers data

Reads from each experiment:
```
results/sft_lora_esm3_qwen3_8b_it_0227_022604/checkpoints/trainer_state.json
results/sft_lora_esm3_qwen3_8b_it_0227_022604/lineage.json
results/sft_lora_esm3_qwen3_8b_it_0227_022604/metrics.json
results/sft_text_qwen3_8b_it_0227_145821/checkpoints/trainer_state.json
results/sft_text_qwen3_8b_it_0227_145821/lineage.json
results/sft_text_qwen3_8b_it_0227_145821/metrics.json
```

Outputs:
```
blog/data/MM-DD/run_histories.csv
  Columns: experiment, step, epoch, token_avg_loss, eval_loss, grad_norm, learning_rate
blog/data/MM-DD/experiment_metadata.json
  {experiment_name, approach, projector_type, base_model, lr, epochs, ...}
```

### Step 3: analyst creates plots

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", palette="colorblind")
COLORS = {"mlp": "#1f77b4", "text": "#2ca02c"}

df = pd.read_csv("blog/data/MM-DD/run_histories.csv")

# Loss curves
fig, ax = plt.subplots(figsize=(10, 6))
for name, group in df.groupby("experiment"):
    approach = "mlp" if "esm3" in name else "text"
    ax.plot(group["step"], group["token_avg_loss"],
            color=COLORS[approach], label=f"{approach}: {name[:30]}...")
ax.set_xlabel("Step")
ax.set_ylabel("Token Average Loss")
ax.set_title("Training Loss: MLP vs Text-Only")
ax.legend()
fig.savefig("blog/figures/loss_curves.png", dpi=150)
plt.close()
```

Outputs: PNGs + `analysis_summary.json`

### Step 4: reporter writes HTML post

Creates `blog/posts/2026-03-01_mlp-vs-text-sft-comparison.html` with:
- Executive Summary (2-3 sentences with specific numbers)
- Experiment Configuration table
- Key Findings with embedded figures (`<img src="../figures/loss_curves.png">`)
- Summary Results table
- Recommendations

Also regenerates `blog/index.html` to include the new post.

### Step 5: Lead delivers to user

> Report ready at `blog/posts/2026-03-01_mlp-vs-text-sft-comparison.html`.
> Key finding: MLP achieves 1.6% lower token_avg_loss (2.49 vs 2.53).
> Both converge by step 200. See report for full analysis.

---

## References

- [CLAUDE.md](CLAUDE.md) — Project context and critical rules
- [SWE_AGENT_TEAM.md](SWE_AGENT_TEAM.md) — Development agent team (separate purpose)
- [docs/research/agents_research_log.md](docs/research/agents_research_log.md) — Research log
- [PROJECT_GOALS.md](PROJECT_GOALS.md) — Strategic goals
