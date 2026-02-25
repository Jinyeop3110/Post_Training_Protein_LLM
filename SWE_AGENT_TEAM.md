# SWE Agent Team: Post-Training Protein LLM

> **Purpose**: Coordinate multiple Claude Code sessions to develop, train, evaluate, and maintain the multimodal protein-LLM system.

---

## Quick Start

```bash
# 1. Enable agent teams (add to ~/.claude/settings.json or shell)
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

# 2. Start Claude Code
claude

# 3. Request team creation
> Create an agent team with 5 teammates for protein-LLM development:
> - architect: model architecture and modularity
> - trainer: SFT/GRPO training pipelines
> - evaluator: benchmarks and metrics
> - qa-engineer: testing and code review
> - doc-tracker: research log and documentation
```

---

## Team Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   YOU                                       │
│                    (Human - Project Owner / Decision Maker)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │      TEAM LEAD      │
                          │                     │
                          │ • YOUR interface    │
                          │ • Asks questions    │
                          │ • Reports progress  │
                          │ • Requests approval │
                          │ • Synthesizes work  │
                          └──────────┬──────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│                           COORDINATES                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────┬───────────┼───────────┬───────────────┐
         ▼               ▼           ▼           ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  ARCHITECT   │ │   TRAINER    │ │  EVALUATOR   │ │ QA-ENGINEER  │ │ DOC-TRACKER  │
│              │ │              │ │              │ │              │ │              │
│ • Model arch │ │ • SFT impl   │ │ • GO eval    │ │ • Unit tests │ │ • Research   │
│ • Interfaces │ │ • GRPO impl  │ │ • PPI eval   │ │ • Integration│ │   log        │
│ • Extensible │ │ • DPO impl   │ │ • Stability  │ │ • Code review│ │ • Progress   │
│   design     │ │ • Checkpts   │ │ • Metrics    │ │ • Type hints │ │ • Decisions  │
│ • Config     │ │ • Logging    │ │ • Wandb      │ │ • Linting    │ │ • Goals      │
│   schema     │ │ • Versioning │ │ • Analysis   │ │ • Security   │ │ • Changelog  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
     │                 │                 │                 │               │
     └─────────────────┴─────────────────┴─────────────────┴───────────────┘
                                     │
                              SHARED TASK LIST
                    (Pending → In Progress → Completed)
```

---

## Agent Specifications

### 1. ARCHITECT

**Focus**: Model architecture, modularity, and extensibility

**Responsibilities**:
- Design clean interfaces for new components (encoders, poolers, projectors)
- Ensure ESM-3 remains frozen (critical rule enforcement)
- Maintain Hydra configuration schema consistency
- Plan how new features integrate with existing code
- Define abstract base classes for extensibility

**Key Files Owned**:
```
src/models/
├── __init__.py
├── base.py              # Abstract base classes
├── multimodal_llm.py    # Main ProteinLLM class
├── protein_encoder.py   # Encoder interface
├── pooling.py           # Pooling strategies
└── projector.py         # Projector interface

configs/
├── model/               # Model configurations
└── encoder/             # Encoder configurations
```

**Spawn Prompt**:
```
You are the Architect agent for the protein-LLM project.

FIRST: Read CLAUDE.md and PROJECT_GOALS.md for full project context.

Environment: 8x NVIDIA H100 80GB | CUDA 12.4 | Python 3.11

Your responsibilities:
1. Design clean, extensible model architectures
2. Enforce critical rules: ESM-3 ALWAYS frozen, LoRA on all linear layers (q/k/v/o + gate/up/down)
3. Maintain consistent Hydra config patterns
4. Plan interface changes before implementation
5. Review architectural decisions with the lead

Key context:
- Approach switch: `approach: text | esm3` in configs/config.yaml
  - text: Raw sequence as tokens to LLM (no encoder)
  - esm3 + mlp: ESM-3 → AttentionPooling → MLP Projector → LLM
  - esm3 + perceiver: ESM-3 → PerceiverResampler → LLM
- Base LLM: Qwen3-4B-Instruct-2507 (default), also 8B/14B Instruct variants
- Encoder: ESM-3 small (frozen, 1536-dim)
- Training uses model's native chat template with system prompt
- See docs/architecture.md for full details
- Use configs/ for all hyperparameters (never hardcode)

When adding new components:
1. Define abstract interface in src/models/base.py
2. Implement concrete class
3. Add Hydra config in configs/
4. Update __init__.py exports
5. Notify qa-engineer for tests
```

---

### 2. TRAINER

**Focus**: SFT, GRPO, DPO training pipelines with proper versioning

**Responsibilities**:
- Implement and maintain training loops (SFT, GRPO, DPO)
- Handle checkpoint saving with versioned naming
- Configure wandb/tensorboard logging
- Manage distributed training (FSDP, DeepSpeed)
- Ensure proper LoRA configuration

**Key Files Owned**:
```
src/training/
├── __init__.py
├── sft_trainer.py       # SFT with QLoRA
├── grpo_trainer.py      # GRPO reinforcement learning
├── dpo_trainer.py       # DPO alignment
├── callbacks.py         # Training callbacks
└── utils.py             # Training utilities

configs/training/
├── sft_qlora.yaml
├── sft_lora.yaml
├── grpo.yaml
└── dpo.yaml

scripts/
├── train.py             # Main training entry point
└── prepare_data.py      # Data preprocessing
```

**Checkpoint Versioning Convention**:
```
data/checkpoints/
├── sft_v1.0_20260218_143022/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── training_args.json
│   └── metrics.json
├── sft_v1.1_20260219_091534/
│   └── ...
└── grpo_v1.0_20260220_160000/
    └── ...
```

**Spawn Prompt**:
```
You are the Trainer agent for the protein-LLM project.

FIRST: Read CLAUDE.md and PROJECT_GOALS.md for full project context.

Environment: 8x NVIDIA H100 80GB | CUDA 12.4 | Python 3.11

Your responsibilities:
1. Implement SFT, GRPO, and DPO training pipelines
2. Save checkpoints with version naming: {method}_v{major}.{minor}_{timestamp}
3. Configure proper logging (wandb only — tensorboard is disabled)
4. Handle distributed training configuration
5. LoRA targets all linear layers (q/k/v/o + gate/up/down projections)

Key hyperparameters:
- SFT: lr=2e-4, epochs=3, LoRA r=8 on all linear layers
- GRPO: lr=5e-6 (much lower!), epochs=1, group_size=4
- QLoRA (4-bit) is default for SFT; LoRA (full precision) also available
- Approach: text | esm3 (set in configs/config.yaml)
- Training uses model's native chat template with system prompt (not Alpaca format)
- Base LLM: Qwen3-4B-Instruct-2507 (Instruct variant, NOT base model)

Training pipeline:
1. Phase 1: SFT with QLoRA → checkpoint
2. Phase 2: GRPO alignment (load SFT checkpoint) → final checkpoint

When saving checkpoints:
- Include training_args.json with all hyperparameters
- Include metrics.json with final train/val loss
- Log to wandb with checkpoint artifact
```

---

### 3. EVALUATOR

**Focus**: Benchmarks, metrics, and experiment tracking

**Responsibilities**:
- Implement evaluation for GO prediction, PPI, stability
- Standardize metrics computation (F1, AUPR, MAE)
- Create wandb dashboards and comparison views
- Generate evaluation reports
- Track experiment results over time

**Key Files Owned**:
```
src/evaluation/
├── __init__.py
├── go_prediction.py     # GO term evaluation
├── ppi_prediction.py    # Protein-protein interaction
├── stability.py         # Stability prediction (ddG)
├── metrics.py           # Shared metrics utilities
└── benchmarks.py        # Combined benchmark runner

configs/evaluation/
├── go_prediction.yaml
├── ppi.yaml
├── stability.yaml
└── all.yaml             # Run all benchmarks

scripts/
├── evaluate.py          # Evaluation entry point
└── analyze_results.py   # Results analysis
```

**Metrics Specification**:
```yaml
# GO Prediction
go_prediction:
  metrics: [f1_micro, f1_macro, accuracy, aupr]
  categories: [MF, BP, CC]  # Molecular Function, Biological Process, Cellular Component

# PPI Prediction
ppi:
  metrics: [accuracy, f1, precision, recall, aupr]
  output_format: binary (Yes/No)

# Stability Prediction
stability:
  metrics: [mae, correlation, classification_accuracy]
  classification_thresholds:
    stabilizing: ddG < -0.5
    neutral: -0.5 <= ddG <= 0.5
    destabilizing: ddG > 0.5
```

**Spawn Prompt**:
```
You are the Evaluator agent for the protein-LLM project.

FIRST: Read CLAUDE.md and PROJECT_GOALS.md for full project context.

Environment: 8x NVIDIA H100 80GB | CUDA 12.4 | Python 3.11

Your responsibilities:
1. Implement evaluation benchmarks (GO, PPI, Stability)
2. Compute standardized metrics for each task
3. Log all results to wandb with proper tags
4. Generate comparison tables across experiments
5. Create visualizations of model performance

Evaluation workflow:
1. Load checkpoint from path
2. Run inference on test set
3. Parse model outputs (extract GO terms, Yes/No, ddG values)
4. Compute metrics
5. Log to wandb and save local JSON

Key metrics by task:
- GO: F1 (micro/macro), AUPR by category
- PPI: Accuracy, F1, AUPR
- Stability: MAE, Pearson correlation

Always output results to:
- Wandb table with experiment tags
- Local JSON: results/{experiment_name}/eval/{task}_metrics.json
- Console summary table
```

---

### 4. QA-ENGINEER

**Focus**: Testing, code review, and code quality

**Responsibilities**:
- Write unit tests for all components
- Write integration tests for pipelines
- Review code for critical rule violations
- Enforce type hints and documentation
- Run linting and security checks

**Key Files Owned**:
```
tests/
├── conftest.py          # Shared fixtures
├── models/
│   ├── test_protein_encoder.py
│   ├── test_pooling.py
│   ├── test_projector.py
│   └── test_multimodal_llm.py
├── data/
│   ├── test_datasets.py
│   └── test_collators.py
├── training/
│   ├── test_sft_trainer.py
│   └── test_grpo_trainer.py
└── evaluation/
    ├── test_go_prediction.py
    ├── test_ppi_prediction.py
    └── test_stability.py

pyproject.toml           # Test configuration
```

**Critical Rule Checklist**:
```markdown
## Code Review Checklist

### Critical (Must Fix)
- [ ] ESM-3 encoder is frozen (requires_grad=False)
- [ ] LoRA applied to all linear layers (q/k/v/o + gate/up/down)
- [ ] Attention pooling used (not mean pooling) for MLP path
- [ ] No hardcoded paths (use config)
- [ ] TRITON_CACHE_DIR is local (/tmp/triton_cache_$USER)
- [ ] Model configs use Instruct variants (e.g., Qwen3-4B-Instruct-2507)
- [ ] Training uses chat template format with system prompt

### Type Safety
- [ ] All public functions have type hints
- [ ] Return types are explicit
- [ ] Optional types handled correctly

### Testing
- [ ] New code has unit tests
- [ ] Tests pass: pytest tests/ -v
- [ ] Coverage maintained: pytest --cov=src

### Code Quality
- [ ] Ruff passes: ruff check src/
- [ ] MyPy passes: mypy src/
- [ ] Google-style docstrings present
```

**Spawn Prompt**:
```
You are the QA-Engineer agent for the protein-LLM project.

FIRST: Read CLAUDE.md and PROJECT_GOALS.md for full project context.

Environment: 8x NVIDIA H100 80GB | CUDA 12.4 | Python 3.11

Your responsibilities:
1. Write comprehensive tests for all components
2. Review code for critical rule violations
3. Enforce type hints and docstrings
4. Run linting (ruff) and type checking (mypy)
5. Report issues to the lead with severity ratings

Critical rules to enforce:
- ESM-3 MUST be frozen (for param in encoder.parameters(): param.requires_grad = False)
- LoRA targets all linear layers (q/k/v/o + gate/up/down projections)
- Use attention pooling (AttentionPooling class) for MLP path
- No hardcoded model paths (use config.model.path)
- Model configs must use Instruct variants (e.g., Qwen3-4B-Instruct-2507)
- Training must use chat template format (not Alpaca ### Instruction: format)

Testing workflow:
1. Identify untested code: pytest --cov=src --cov-report=html
2. Write tests with fixtures from conftest.py
3. Use pytest.mark.slow for GPU tests
4. Mock expensive operations where appropriate

Before any PR merge:
- All tests pass
- Coverage >= 80%
- Ruff check passes
- Critical rules verified
```

---

### 5. DOC-TRACKER

**Focus**: Research log, progress tracking, and documentation

**Responsibilities**:
- Maintain research log with decisions and results
- Track project goals and milestones
- Update documentation when code changes
- Record experiment configurations and outcomes
- Maintain changelog

**Key Files Owned**:
```
docs/
├── research/
│   ├── agents_research_log.md    # Main research log
│   ├── experiment_results.md     # Experiment tracking
│   └── decisions.md              # Architectural decisions
├── architecture.md
├── training_guide.md
├── datasets.md
└── troubleshooting.md

CHANGELOG.md                       # Version changelog
improvement_plan_260216.md         # Project roadmap
```

**Research Log Format**:
```markdown
## [YYYY-MM-DD] Entry Title

### Summary
Brief description of what was done.

### Configuration
- Model: Qwen-3 4B
- Training: SFT with QLoRA
- Dataset: Mol-Instructions
- Hyperparameters: lr=2e-4, epochs=3

### Results
| Metric | Value |
|--------|-------|
| GO F1 (macro) | 0.45 |
| PPI Accuracy | 0.78 |
| Loss | 1.23 |

### Decisions
- Decision: Use attention pooling over mean pooling
- Rationale: 5% improvement in GO F1

### Next Steps
1. Try GRPO alignment
2. Add stability evaluation
```

**Spawn Prompt**:
```
You are the Doc-Tracker agent for the protein-LLM project.

FIRST: Read CLAUDE.md and PROJECT_GOALS.md for full project context.

Environment: 8x NVIDIA H100 80GB | CUDA 12.4 | Python 3.11

Your responsibilities:
1. Maintain the research log at docs/research/agents_research_log.md
2. Track all experiment configurations and results
3. Document architectural decisions with rationale
4. Update docs when code changes
5. Keep CHANGELOG.md current

When experiments complete:
1. Get results from evaluator
2. Add entry to research log with:
   - Date and title
   - Configuration used
   - Results table
   - Key observations
   - Next steps

When code changes:
1. Get details from architect/trainer
2. Update relevant docs (CLAUDE.md, architecture.md, training_guide.md)
3. Add entry to CHANGELOG.md
4. Ensure CLAUDE.md stays in sync with actual configs

Project goals to track:
1. Modular, extensible architecture
2. Proper SFT → RL pipeline
3. Comprehensive evaluation suite
4. Reproducible experiments
5. Model versioning

Weekly summary format:
- Completed tasks
- Key results
- Blockers
- Next week priorities
```

---

## Workflow Patterns

### Pattern 1: New Feature Development

```
1. Lead assigns feature to Architect
   └─> Architect designs interface, creates config schema
       └─> Architect notifies Trainer (if training-related)
           └─> Trainer implements feature
               └─> Trainer notifies QA-Engineer
                   └─> QA-Engineer writes tests
                       └─> QA-Engineer reviews code
                           └─> Doc-Tracker updates docs
                               └─> Lead reviews and merges
```

**Example Task Assignment**:
```
> Tell architect to design an interface for adding a new encoder.
> Once designed, have trainer implement the integration.
> QA-engineer should write tests before we merge.
> Doc-tracker should update architecture.md when done.
```

### Pattern 2: Training Experiment

```
1. Lead specifies experiment config
   └─> Trainer runs training (SFT or GRPO)
       └─> Trainer logs to wandb, saves checkpoint
           └─> Evaluator runs benchmarks
               └─> Evaluator reports metrics
                   └─> Doc-Tracker logs results
                       └─> Lead decides next steps
```

**Example Task Assignment**:
```
> Tell trainer to run SFT with lr=2e-4 on Mol-Instructions.
> Once complete, have evaluator run GO prediction benchmark.
> Doc-tracker should log the results to the research log.
```

### Pattern 3: Bug Fix / Debugging

```
1. Lead reports bug
   └─> QA-Engineer investigates, reproduces
       └─> QA-Engineer identifies root cause
           └─> (If architectural) Architect plans fix
           └─> (If training) Trainer fixes
               └─> QA-Engineer verifies fix, adds test
                   └─> Doc-Tracker updates troubleshooting.md
```

**Example Task Assignment**:
```
> QA-engineer: investigate why GO prediction accuracy dropped after GRPO.
> Check if it's a training issue (trainer) or evaluation issue (evaluator).
> Once fixed, add a regression test.
```

### Pattern 4: Code Review

```
1. Developer (any agent) completes feature
   └─> QA-Engineer reviews against checklist
       └─> QA-Engineer reports issues
           └─> Developer fixes issues
               └─> QA-Engineer approves
                   └─> Lead merges
```

---

## Task List Structure

The shared task list coordinates all agents. Example structure:

```yaml
tasks:
  # Feature: Add ESM-3 support
  - id: arch-001
    title: Design ESM-3 encoder interface
    assignee: architect
    status: completed
    dependencies: []

  - id: train-001
    title: Implement ESM-3 encoder integration
    assignee: trainer
    status: in_progress
    dependencies: [arch-001]

  - id: qa-001
    title: Write tests for ESM-3 encoder
    assignee: qa-engineer
    status: pending
    dependencies: [train-001]

  - id: doc-001
    title: Update architecture.md with ESM-3
    assignee: doc-tracker
    status: pending
    dependencies: [qa-001]

  # Experiment: GRPO training
  - id: train-002
    title: Run GRPO training on Mol-Instructions
    assignee: trainer
    status: pending
    dependencies: []

  - id: eval-001
    title: Evaluate GRPO checkpoint on all benchmarks
    assignee: evaluator
    status: pending
    dependencies: [train-002]
```

---

## Agent Communication

### Communication Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   YOU                                       │
│                         (Only talks to Team Lead)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     ▲
                                     │ Questions, Progress, Approvals
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               TEAM LEAD                                     │
│                                                                             │
│  • Receives YOUR requirements                                               │
│  • Asks YOU questions (via AskUserQuestion)                                 │
│  • Reports progress summaries to YOU                                        │
│  • Requests YOUR approval for major decisions                               │
│  • Delegates tasks to teammates                                             │
│  • Synthesizes teammate results for YOU                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     ▲
                                     │ Task assignments, Results
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TEAMMATES                                      │
│                                                                             │
│  Architect ◄──► Trainer ◄──► Evaluator ◄──► QA-Engineer ◄──► Doc-Tracker   │
│                                                                             │
│  • Communicate directly with EACH OTHER                                     │
│  • Report results to TEAM LEAD                                              │
│  • Do NOT contact YOU directly (go through Lead)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Communication Rules

| From | To | Allowed? | How |
|------|----|----------|-----|
| **You** | Team Lead | ✅ Yes | Direct conversation |
| **You** | Teammates | ⚠️ Emergency only | Shift+Down to select |
| **Team Lead** | You | ✅ Yes | AskUserQuestion, progress reports |
| **Team Lead** | Teammates | ✅ Yes | Task assignment, coordination |
| **Teammates** | You | ❌ No | Must go through Team Lead |
| **Teammates** | Team Lead | ✅ Yes | Results, questions, blockers |
| **Teammates** | Each other | ✅ Yes | Direct messages |

### Team Lead Responsibilities

The Team Lead is your **single point of contact**:

1. **Receives your requirements** → Breaks them into tasks for teammates
2. **Asks clarifying questions** → Uses AskUserQuestion tool
3. **Reports progress** → Summarizes what teammates are doing
4. **Escalates blockers** → Brings issues to your attention
5. **Requests approvals** → For major decisions, breaking changes
6. **Synthesizes results** → Combines teammate outputs into coherent reports

### Teammate-to-Teammate Messaging

Teammates communicate directly for coordination:

```
# From trainer to evaluator
> Message evaluator: "SFT checkpoint saved at data/checkpoints/sft_v1.0_20260218_143022/.
   Please run GO and PPI evaluation."

# From architect to qa-engineer
> Message qa-engineer: "I added a new PoolingStrategy base class in src/models/base.py.
   Please add unit tests for the interface."

# From evaluator to doc-tracker
> Message doc-tracker: "Evaluation complete. Results: GO F1=0.45, PPI Acc=0.78.
   Please log to research log."
```

### Teammate-to-Lead Reporting

Teammates report to Lead, NOT to you:

```
# Trainer → Lead (NOT to you)
> Message lead: "SFT training complete. 3 epochs, final loss 1.23.
   Checkpoint: sft_mol_v1.0_20260218_143022/
   Ready for evaluation."

# Lead then summarizes to YOU
> "Training Phase 1 complete. Trainer reports:
   - Final loss: 1.23
   - Checkpoint saved
   - Evaluator is now running benchmarks

   Should I proceed with GRPO phase, or wait for evaluation results?"
```

### Broadcast (Use Sparingly)
Lead broadcasts to all teammates when announcing major changes:

```
> Broadcast: "Major config schema change. All configs now use nested structure.
   Please update any hardcoded config paths in your owned files."
```

---

## Settings Configuration

Add to `.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "teammateMode": "in-process"
}
```

---

## Best Practices

### 1. Task Sizing
- **Too small**: "Fix typo in docstring" → Just do it
- **Too large**: "Implement full GRPO pipeline" → Break into subtasks
- **Right size**: "Implement GRPO reward computation" → Clear deliverable

### 2. File Ownership
- Each agent owns specific directories
- Avoid multiple agents editing same file
- Use interfaces for cross-agent dependencies

### 3. Communication
- Use shared task list for coordination
- Direct message for specific requests
- Broadcast only for team-wide announcements

### 4. Quality Gates
- QA-Engineer reviews all code before merge
- Evaluator validates training outputs
- Doc-Tracker ensures documentation is current

### 5. Token Efficiency
- Start with 3 agents for focused work
- Scale to 5 for complex parallel tasks
- Shut down idle agents to save tokens

---

## Example Session

```bash
# Start Claude Code with teams enabled
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
claude

# Request team creation
> Create an agent team for protein-LLM development with these teammates:
> 1. architect - model architecture and extensibility
> 2. trainer - SFT/GRPO training with checkpoint versioning
> 3. evaluator - benchmarks and metrics
> 4. qa-engineer - testing and code review
> 5. doc-tracker - research log and documentation
>
> First task: Implement checkpoint versioning for SFT trainer.
> Trainer owns implementation, qa-engineer writes tests, doc-tracker logs progress.

# Claude creates team, assigns tasks, coordinates work
# Use Shift+Down to cycle through teammates
# Type to send messages directly to any teammate
```

---

## User Input & Feedback

> **All communication with you goes through the Team Lead.**
> Teammates report to Lead → Lead reports to you.

### How to Provide Requirements (to Team Lead)

#### Method 1: Tell the Lead Your Requirements
```
> Lead, I want checkpoint versioning implemented.
> REQUIREMENTS:
> - Include dataset name: {method}_{dataset}_v{version}_{timestamp}
> - Save training_args.json with all hyperparameters
> - Save metrics.json with final loss values
> - Support --resume flag to continue training
> MY PREFERENCE: Use SafeTensors format, not .pt
>
> Assign this to the appropriate teammate.
```

The Lead will then delegate to Trainer with your requirements.

#### Method 2: Redirect Through Lead
```
> Lead, tell the trainer I want the learning rate in the checkpoint name too.
> Change format to: sft_mol_lr2e4_v1.0_20260218/
```

#### Method 3: Interrupt and Clarify (to Lead)
```
> Lead, stop. Before the team proceeds, I need to clarify:
> 1. We need backward compatibility with old checkpoints
> 2. Add a migration script for existing checkpoints
> 3. Update the docs first so we agree on the format
>
> Make sure all teammates are aware of this.
```

#### Method 4: Review Points File
Update [REVIEW_POINTS.md](REVIEW_POINTS.md) with your requirements.
The Lead will check this file and communicate to teammates.

```markdown
# REVIEW_POINTS.md

## Current Priorities
1. Checkpoint versioning must include dataset name
2. All training runs must log to wandb
3. Tests required before any merge

## Decisions Made
- [x] Use SafeTensors format (2024-02-18)
- [x] LoRA r=8 is sufficient (2024-02-17)
```

#### Method 5: Emergency Direct Contact (Rare)
Only if Lead is unavailable or for urgent issues:
```
# Press Shift+Down to select a specific teammate
# This bypasses normal communication flow
```

### When Team Lead Should Ask You

The Lead uses `AskUserQuestion` when:

| Situation | Example Question from Lead |
|-----------|----------------------------|
| **Multiple valid approaches** | "Architect proposes two pooling options. Which do you prefer: 32 or 64 query tokens?" |
| **Resource trade-offs** | "Trainer reports batch size 8 uses 60GB VRAM. Should we reduce to 4?" |
| **Breaking changes** | "This refactor changes the checkpoint format. Approve?" |
| **Unclear requirements** | "Evaluator asks: should stability evaluation include all mutation types?" |
| **External dependencies** | "Trainer needs to download ESM-3 weights (15GB). Proceed?" |
| **Progress checkpoint** | "Phase 1 SFT complete. Results: GO F1=0.42. Continue to Phase 2 GRPO?" |
| **Blocker escalation** | "QA-Engineer found critical issue: training produces NaN loss. How to proceed?" |
| **Teammate disagreement** | "Architect and Trainer disagree on projector design. Here are both options..." |

### Configure What Requires Your Approval

Tell the Lead upfront what decisions need your input:

```
> Lead, before proceeding with any of these, ask me first:
> - Learning rate selection
> - Number of epochs
> - Dataset selection
> - Checkpoint save frequency
> - Any architectural changes
> - Breaking changes to existing code
>
> For routine tasks (tests, docs, logging), teammates can proceed autonomously.
```

### Team Lead Escalation Protocol

The Lead will contact you when:

1. **Phase completion** → Summary of results + request to proceed
2. **Teammates disagree** → Presents options for your decision
3. **Critical rule violation** → QA-Engineer → Lead → You
4. **Resource constraints** → OOM, disk space, GPU issues
5. **Unexpected results** → Metrics off, training diverged
6. **Blocker** → Can't proceed without your input

### Feedback Loop Example (All Through Lead)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ YOU → LEAD: "Run SFT on Mol-Instructions"                                │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ LEAD asks YOU: "Which learning rate should trainer use?"                 │
│                                                                          │
│  ○ 2e-4 (Recommended) - Standard for QLoRA                              │
│  ○ 1e-4 - More conservative                                             │
│  ○ 5e-4 - Faster convergence, risk of instability                       │
│  ○ Other: _______________                                                │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ YOU: Select "2e-4 (Recommended)"                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ LEAD → TRAINER: "Run SFT with lr=2e-4 on Mol-Instructions"               │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ TRAINER (working...):                                                    │
│   "Starting SFT with lr=2e-4..."                                         │
│   "Epoch 1/3: loss=2.34"                                                 │
│   "Epoch 2/3: loss=1.67"                                                 │
│   "Epoch 3/3: loss=1.23"                                                 │
│   "Checkpoint saved: sft_mol_v1.0_20260218_143022/"                      │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ TRAINER → LEAD: "SFT complete. Final loss: 1.23"                         │
│ TRAINER → EVALUATOR: "Checkpoint ready at sft_mol_v1.0_20260218_143022/" │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ EVALUATOR (working...):                                                  │
│   "Running GO prediction benchmark..."                                   │
│   "Running PPI prediction benchmark..."                                  │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ EVALUATOR → LEAD: "Evaluation complete"                                  │
│                                                                          │
│  | Metric      | Value |                                                 │
│  |-------------|-------|                                                 │
│  | GO F1 macro | 0.42  |                                                 │
│  | PPI Acc     | 0.76  |                                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ LEAD asks YOU: "Phase 1 SFT complete. Results:"                          │
│                                                                          │
│  | Metric      | Value |                                                 │
│  |-------------|-------|                                                 │
│  | Final Loss  | 1.23  |                                                 │
│  | GO F1 macro | 0.42  |                                                 │
│  | PPI Acc     | 0.76  |                                                 │
│                                                                          │
│  "How should we proceed?"                                                │
│                                                                          │
│  ○ Continue to Phase 2 GRPO (Recommended)                               │
│  ○ Run stability evaluation first                                       │
│  ○ Adjust hyperparameters and retrain SFT                               │
│  ○ Stop here, results are sufficient                                    │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ YOU: Select "Continue to Phase 2 GRPO"                                   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ LEAD → TRAINER: "Proceed with GRPO using SFT checkpoint"                 │
│ LEAD → DOC-TRACKER: "Log Phase 1 results to research log"               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Cleanup

When finished with the team:

```
> Shut down all teammates and clean up the team
```

---

## Troubleshooting

### Teammates Not Responding
- Check if they're idle: Shift+Down to cycle
- Send a direct message to wake them
- Check their session for errors

### Conflicting File Edits
- Ensure only one agent owns each file
- Use interfaces for shared functionality
- Coordinate via task dependencies

### Token Usage Too High
- Reduce to 3 core agents (architect, trainer, qa)
- Merge evaluator into trainer
- Merge doc-tracker into lead

---

## References

- [Claude Code Agent Teams Documentation](https://code.claude.com/docs/en/agent-teams)
- [Project Architecture](docs/architecture.md)
- [Training Guide](docs/training_guide.md)
- [Research Log](docs/research/agents_research_log.md)
