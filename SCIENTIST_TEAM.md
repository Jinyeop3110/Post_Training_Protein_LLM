# Scientist Agent Team: Training Diagnostics & Publishing

> **Purpose**: Analyze training experiments, create diagnostic plots, and produce reports across multiple output destinations. Parallel team-lead structure — agents work independently and the lead coordinates.

---

## Quick Start

```bash
# Start Claude Code (team lead)
claude

# Request analysis
> Analyze loss curves for MLP vs text-only SFT and publish to blog + Jekyll site
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
                        │ • Delegates tasks   │
                        │ • Runs agents in    │
                        │   PARALLEL          │
                        │ • Synthesizes work  │
                        └──────────┬──────────┘
                                   │
       ┌──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│DATA-COLLECTOR││   ANALYST    ││    ARTIST    ││   REPORTER   │
│              ││              ││              ││              │
│ • wandb scan ││ • Statistics ││ • Loss plots ││ • HTML posts │
│ • Local files││ • Anomalies  ││ • Bar charts ││ • Jekyll MD  │
│ • Discovery  ││ • Comparison ││ • Paper PDFs ││ • Blog index │
│ • Metadata   ││ • Plot specs ││ • Style/DPI  ││ • Cross-post │
└──────────────┘└──────────────┘└──────────────┘└──────────────┘
       │                │                │              │
       └────────────────┴────────────────┴──────────────┘
                                 │
                    SHARED OUTPUT DESTINATIONS
```

### Key Difference from Sequential Pipeline

Agents are **independent** — each reads `results/` directly:

| Agent | Reads | Writes | Can run in parallel? |
|-------|-------|--------|---------------------|
| **data-collector** | wandb + `results/` | `blog/data/` CSVs, JSONs | Yes (independent) |
| **analyst** | `results/` + `blog/data/` | `blog/data/` analysis JSONs | Yes (independent) |
| **artist** | `blog/data/` + `results/` | `blog/figures/`, `paper/figures/`, Jekyll assets | Yes (reads raw data directly) |
| **reporter** | `blog/data/` + `blog/figures/` + `results/` | `blog/posts/`, Jekyll `_posts/` | Yes (reads raw data directly) |

The lead spawns all 4 in parallel when possible. For simple questions, the lead may skip agents and do the work directly.

---

## Output Destinations

All agents must be aware of **4 output destinations** with different formats:

### 1. Internal Blog (`blog/`)

```
blog/
├── index.html                    # Blog index (auto-generated)
├── posts/                        # HTML blog posts
│   └── YYYY-MM-DD_title.html
├── figures/
│   ├── figure_catalog.md         # Single source of truth
│   ├── main_figures/             # Key figures (paper + website)
│   └── supple_figures/           # Supplementary figures
└── data/
    └── MM-DD/                    # Analysis data by date
        ├── run_histories.csv
        ├── experiment_metadata.json
        └── analysis_summary.json
```

- **Format**: HTML posts, PNGs for figures
- **Convention**: `YYYY-MM-DD_title-in-kebab-case.html`
- **Figure refs**: `../figures/main_figures/name.png` from posts
- **Tags**: kickoff, architecture, training, evaluation, data, rl, sft, infrastructure, milestone, debug

### 2. Jekyll Site (`Jinyeop3110.github.io/`)

```
/home/yeopjin/orcd/pool/workspace/Jinyeop3110.github.io/
├── _posts/                       # Markdown blog posts (Jekyll)
│   └── YYYY-MM-DD-title.md
└── assets/img/blog/              # Blog images
    └── protein-llm/              # Project-specific images
```

- **Format**: Markdown with Jekyll frontmatter (YAML header)
- **Convention**: `YYYY-MM-DD-title.md` (hyphens, not underscores)
- **Frontmatter**:
  ```yaml
  ---
  layout: post
  title: "Post Title"
  description: >
    One-paragraph description for meta tags.
  date: YYYY-MM-DD
  categories: [research]
  tags: [llm, protein, multimodal, esm3]
  ---
  ```
- **Figure refs**: `/assets/img/blog/protein-llm/name.png` (absolute from site root)
- **Style**: Narrative, engaging, first-person. Suitable for public audience.

### 3. Paper Figures (`paper/figures/`)

```
paper/figures/
├── main/                         # PDFs (NeurIPS-compatible)
│   └── fig{N}_{name}.pdf
└── supplementary/                # Additional PDFs + PNGs
    └── *.pdf
```

- **Format**: PDF (vector, NeurIPS-compatible), 300 DPI
- **Naming**: `fig{N}_{descriptive_name}.pdf`
- **Style**: Publication-quality, no titles (caption goes in LaTeX), minimal text
- **Always generate both** PNG (for blog) and PDF (for paper) from same script

### 4. Blog Figure Catalog (`blog/figures/figure_catalog.md`)

Single source of truth for all figures. Must be updated whenever new figures are created.

---

## Agent Specifications

### 1. DATA-COLLECTOR

**Focus**: Fetch training metrics from wandb API and local experiment files

**Works independently** — reads `results/` directly, no dependencies.

**Responsibilities**:
- Query `wandb.Api()` for run histories from protein-LLM projects
- Read local `trainer_state.json`, `metrics.json`, `training_args.json`, `lineage.json`
- Parse HF Trainer log history from `trainer_state.json`
- Output organized CSVs and JSONs to `blog/data/MM-DD/`
- Collect and normalize experiment metadata

**Output**:
```
blog/data/MM-DD/
├── run_histories.csv           # Step-level metrics
├── experiment_metadata.json    # Per-run metadata
└── wandb_summaries.json        # wandb summaries (if fetched)
```

**Critical Rules**:
- NEVER write outside `blog/data/`
- NEVER modify source code or experiment files
- Distinguish `loss` (HF running average) from `token_avg_loss` (true average)
- Always include `approach` and `projector_type` in metadata

**Agent file**: `.claude/agents/data-collector.md`

---

### 2. ANALYST

**Focus**: Statistical analysis, anomaly detection, metric computation — numbers only, no figures

**Works independently** — reads `results/` directly OR `blog/data/MM-DD/` if available.

**Responsibilities**:
- Compute per-experiment statistics (convergence, loss trajectory, gradient stats)
- Run statistical comparisons between experiments (t-tests, effect sizes)
- Detect anomalies (NaN, spikes, divergence)
- Produce structured `analysis_summary.json` with all findings
- Generate **plot specifications** for the artist agent

**Output**: `blog/data/MM-DD/analysis_summary.json` + optional CSVs

**Critical Rules**:
- NEVER create figures — that's the artist's job
- ALWAYS use `token_avg_loss`, NOT `loss`
- Every finding must include specific numbers
- NEVER write outside `blog/data/`

**Agent file**: `.claude/agents/analyst.md`

---

### 3. ARTIST

**Focus**: Publication-quality figure drawing — owns ALL visual output

**Works independently** — reads `results/`, `blog/data/`, or analyst's plot specs.

**Responsibilities**:
- Create matplotlib/seaborn plots (headless: `matplotlib.use('Agg')`)
- Use `figure_style.py` + `STYLE_GUIDE.md` as single source of truth
- Standard plot catalog: loss curves, gradient norms, LR schedule, bar charts
- **Output PNGs** to `blog/figures/` (main or supplementary)
- **Output PDFs** to `paper/figures/` for publication
- **Copy figures** to Jekyll site `assets/img/blog/protein-llm/`
- Update `blog/figures/figure_catalog.md` after creating figures

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

**Style Targets**:
| Target | Font | DPI | Size | Use |
|--------|------|-----|------|-----|
| `blog` | sans-serif 11pt | 150 | 10x6 | Internal dev blog |
| `web` | sans-serif 12pt | 200 | 9x5.5 | Jekyll site |
| `paper` | serif 8pt | 300 | 3.25x2.4 | NeurIPS paper |

**Critical Rules**:
- ALWAYS import from `figure_style.py` — NEVER define inline colors/DPI/sizes
- ALWAYS use `save_figure()` or `save_main_figure()` — NEVER raw `fig.savefig()`
- ALWAYS update `figure_catalog.md` after creating figures
- Paper figures: no titles (caption in LaTeX), minimal text

**Agent file**: `.claude/agents/artist.md`

---

### 4. REPORTER

**Focus**: Write reports across ALL output destinations

**Works independently** — reads `results/` directly, `blog/data/`, and `blog/figures/`.

**Responsibilities**:
- Write **HTML posts** to `blog/posts/` (internal dev blog)
- Write **Jekyll markdown** to `Jinyeop3110.github.io/_posts/` (public website)
- Regenerate `blog/index.html` for internal blog
- Ensure figures are in correct locations for each destination

**Multi-Destination Output**:

| Destination | Format | Path | Figure Refs |
|-------------|--------|------|-------------|
| Internal blog | HTML | `blog/posts/YYYY-MM-DD_title.html` | `../figures/main_figures/name.png` |
| Jekyll site | Markdown + YAML | `Jinyeop3110.github.io/_posts/YYYY-MM-DD-title.md` | `/assets/img/blog/protein-llm/name.png` |

**Writing Style by Destination**:
| Aspect | Internal Blog | Jekyll Site |
|--------|--------------|-------------|
| Tone | Technical, data-focused | Narrative, engaging |
| Audience | Dev team | Public (researchers, students) |
| Numbers | Exact metrics required | Key numbers + context |
| Length | 150-300 lines | 300-600 lines |
| Format | HTML tables, figures | Markdown, storytelling |

**Critical Rules**:
- NEVER modify source code or experiment files
- NEVER delete existing blog/post files
- Use RELATIVE paths for internal blog figures, ABSOLUTE paths for Jekyll
- Numbers over vague qualifiers — every claim needs a number
- Always state which loss metric is used
- Include full experiment names for reproducibility

**Agent file**: `.claude/agents/reporter.md`

---

## Workflow Patterns

### Pattern 1: Quick Analysis (Lead does it)

```
User: "What's the eval loss for the latest MLP run?"

Lead reads results/ directly -> answers immediately
(No agents needed)
```

### Pattern 2: Single-Run Diagnostics (Parallel)

```
User: "Analyze the MLP SFT run"

Lead spawns in PARALLEL:
├─ data-collector: fetch metrics -> blog/data/MM-DD/
├─ analyst: compute stats, anomalies -> blog/data/MM-DD/analysis_summary.json
├─ artist: plot loss, grad norms -> blog/figures/
└─ reporter: draft from results/ directly, fill figures when ready

Lead delivers: "Report at blog/posts/..."
```

### Pattern 3: Multi-Run Comparison (Parallel)

```
User: "Compare MLP vs text-only and publish everywhere"

Lead spawns in PARALLEL:
├─ data-collector: fetch all run metrics -> blog/data/MM-DD/
├─ analyst: statistical comparison -> blog/data/MM-DD/analysis_summary.json
├─ artist: comparison plots -> blog/figures/ + paper/figures/ + Jekyll assets/
└─ reporter: write HTML post + Jekyll markdown

Lead delivers: "Internal blog + Jekyll post ready"
```

### Pattern 4: Discovery (Data-collector alone)

```
User: "What experiments do we have?"

Lead -> data-collector: scan wandb + local results/ -> run_inventory.json
Lead delivers: "Found N runs: 4 complete, 2 partial. Here's the inventory."
```

### Pattern 5: Figure-Only (Artist alone)

```
User: "Regenerate paper figures for latest results"

Lead -> artist: read results/, generate PDFs to paper/figures/main/
(No data-collector, analyst, or reporter needed)
```

### Pattern 6: Cross-Post (Reporter alone)

```
User: "Convert the latest internal blog post to Jekyll"

Lead -> reporter: read blog/posts/latest.html -> write Jekyll markdown
(No data-collector, analyst, or artist needed)
```

### Pattern 7: Deep Analysis (Analyst + Artist)

```
User: "Is there a convergence issue in the 0227 runs?"

Lead spawns in PARALLEL:
├─ analyst: anomaly detection, convergence analysis -> analysis_summary.json
├─ artist: gradient norm plots with anomaly markers -> blog/figures/

Lead synthesizes findings and delivers.
```

---

## Agent Communication

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
│  • Breaks into independent tasks                                        │
│  • Spawns agents in PARALLEL when possible                              │
│  • Synthesizes results from all agents                                  │
│  • Delivers final report to YOU                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ Results, Status
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             TEAMMATES                                   │
│                                                                         │
│  data-collector <---> analyst <---> artist <---> reporter               │
│                                                                         │
│  • Work INDEPENDENTLY (each reads results/ directly)                    │
│  • Can communicate directly with each other                             │
│  • Report results to TEAM LEAD                                          │
│  • Do NOT contact YOU directly                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Communication Rules

| From | To | Allowed? | How |
|------|----|----------|-----|
| **You** | Team Lead | Yes | Direct conversation |
| **Team Lead** | You | Yes | Progress reports, final delivery |
| **Team Lead** | Teammates | Yes | Task assignment with destinations |
| **Teammates** | Team Lead | Yes | Completion notifications, blockers |
| **Teammates** | Each other | Yes | Direct messages (shared data paths) |
| **Teammates** | You | No | Must go through Team Lead |

---

## When Lead Should Ask You

| Situation | Example |
|-----------|---------|
| **Destination unclear** | "Should this go to Jekyll site, internal blog, or both?" |
| **Main vs supplementary** | "This figure looks important — promote to main_figures?" |
| **Conflicting data** | "wandb and local metrics disagree — which to trust?" |
| **Missing experiments** | "Can't find results for experiment X — skip or investigate?" |
| **Style choice** | "Technical report or narrative blog post for Jekyll?" |

---

## Path Reference

| Name | Absolute Path |
|------|---------------|
| **Project root** | `/orcd/pool/006/yeopjin/workspace/Post_Training_Protein_LLM` |
| **Results** | `results/` (relative to project root) |
| **Internal blog** | `blog/` |
| **Blog figures** | `blog/figures/{main,supple}_figures/` |
| **Blog data** | `blog/data/MM-DD/` |
| **Figure catalog** | `blog/figures/figure_catalog.md` |
| **Paper figures** | `paper/figures/{main,supplementary}/` |
| **Jekyll site** | `/home/yeopjin/orcd/pool/workspace/Jinyeop3110.github.io` |
| **Jekyll posts** | `/home/yeopjin/orcd/pool/workspace/Jinyeop3110.github.io/_posts/` |
| **Jekyll images** | `/home/yeopjin/orcd/pool/workspace/Jinyeop3110.github.io/assets/img/blog/protein-llm/` |
| **Figure style** | `scripts/analysis/figure_style.py` |

---

## Data Sources Reference

### trainer_state.json Field Catalog

**Training steps** (every `logging_steps`):
| Field | Description | Notes |
|-------|-------------|-------|
| `loss` | HF Trainer running average | **DO NOT USE for plots** |
| `token_avg_loss` | True per-token average loss | **USE THIS** |
| `grad_norm` | Gradient L2 norm | Log-scale for plots |
| `learning_rate` | Current LR | Shows warmup + decay |
| `epoch` | Fractional epoch | |
| `step` | Global step count | X-axis for most plots |

**Evaluation steps** (every `eval_steps`):
| Field | Description |
|-------|-------------|
| `eval_loss` | Validation loss (most reliable) |
| `eval_runtime` | Eval duration (seconds) |

### wandb Projects

| Project | Contents |
|---------|----------|
| `protein-llm-sft` | SFT training runs |
| `protein-llm-rl` | GRPO training runs |

---

## Critical Rules (All Agents)

1. **NEVER modify source code or experiment files**
2. **NEVER delete existing blog/post files** — always create new dated content
3. **ALWAYS use `token_avg_loss`** for training loss, NOT `loss`
4. **ALWAYS update `figure_catalog.md`** after creating figures
5. **ALWAYS use `figure_style.py`** for colors, DPI, and styling
6. **Use correct format per destination**: HTML for internal blog, MD for Jekyll, PDF for paper
7. **Use correct figure paths per destination**: relative for blog, absolute for Jekyll

---

## References

- [CLAUDE.md](CLAUDE.md) — Project context and critical rules
- [SWE_AGENT_TEAM.md](SWE_AGENT_TEAM.md) — Development agent team (separate purpose)
- [blog/README.md](blog/README.md) — Blog conventions
- [blog/figures/figure_catalog.md](blog/figures/figure_catalog.md) — Figure inventory
- [docs/research/agents_research_log.md](docs/research/agents_research_log.md) — Research log
