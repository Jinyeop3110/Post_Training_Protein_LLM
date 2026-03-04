# SWE Agent Team: Post-Training Protein LLM

> **Purpose**: Coordinate multiple Claude Code sessions to develop, train, evaluate, and maintain the multimodal protein-LLM system.

---

## Quick Start

```bash
# 1. Enable agent teams
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

# 2. Start Claude Code
claude

# 3. Request team creation
> Create an agent team with 3 teammates for protein-LLM development:
> - engineer: architecture, implementation, training, experiments
> - qa: code review, testing, linting, critical rule enforcement
> - researcher: evaluation, documentation, literature, research logging
```

---

## Team Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                                YOU                                   │
│                  (Human - Project Owner / Decision Maker)           │
└─────────────────────────────────────────────────────────────────────┘
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
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │    ENGINEER      │  │       QA         │  │   RESEARCHER     │
  │                  │  │                  │  │                  │
  │ • Architecture   │  │ • Code review    │  │ • Literature     │
  │ • Implementation │  │ • Testing        │  │ • Evaluation     │
  │ • Training       │  │ • Critical rules │  │ • Documentation  │
  │ • Experiments    │  │ • Linting        │  │ • Research log   │
  │ • Configs        │  │ • Type safety    │  │ • Progress track │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                                │
                         SHARED TASK LIST
                  (Pending → In Progress → Completed)
```

---

## Agent Specifications

### 1. ENGINEER

**Focus**: Architecture, implementation, training pipelines, experiment execution

**Owns**: `src/`, `configs/`, `scripts/`

**Merges former roles**: Architect + Trainer + Experiment-Runner

**Key responsibilities**:
- Design clean interfaces for encoders, poolers, projectors
- Implement and maintain SFT, GRPO, DPO training pipelines
- Launch, configure, and monitor training runs
- Maintain Hydra configuration schema consistency
- Enforce ESM-3 frozen, LoRA on all linear layers

**Agent file**: `.claude/agents/engineer.md`

---

### 2. QA

**Focus**: Code review, testing, critical rule enforcement, linting

**Owns**: `tests/`, `pyproject.toml`

**Merges former roles**: Code-Reviewer + Test-Writer + QA-Engineer

**Key responsibilities**:
- Review all changes against the 7-item critical checklist
- Write unit, integration, and regression tests
- Enforce type hints, docstrings, code quality
- Run ruff, mypy, pytest
- Catch critical rule violations before they ship

**Critical Checklist** (must fix):
1. ESM-3 frozen (`requires_grad=False`)
2. LoRA on all linear layers (q/k/v/o + gate/up/down)
3. Attention pooling (not mean) for MLP path
4. Instruct model variants only
5. Chat template format (not Alpaca)
6. No secrets in code
7. Safe CUDA operations

**Agent file**: `.claude/agents/qa.md`

---

### 3. RESEARCHER

**Focus**: Literature search, evaluation, documentation, research logging

**Owns**: `docs/`, `src/evaluation/`, `scripts/evaluate.py`, `configs/evaluation/`

**Merges former roles**: Research + Evaluator + Doc-Tracker

**Key responsibilities**:
- Search and summarize relevant papers and methods
- Run evaluation benchmarks (GO, PPI, Stability)
- Maintain research log with decisions and results
- Update documentation when code changes
- Track project goals and milestones

**Evaluation metrics**:

| Task | Metrics |
|------|---------|
| GO Prediction | F1 (micro/macro), accuracy, AUPR by category |
| PPI Prediction | Accuracy, F1, precision, recall, AUPR |
| Stability | MAE, Pearson correlation, classification accuracy |

**Agent file**: `.claude/agents/researcher.md`

---

## Workflow Patterns

### Pattern 1: New Feature Development

```
1. Lead assigns feature to Engineer
   └─> Engineer designs and implements
       └─> Engineer notifies QA
           └─> QA writes tests + reviews code
               └─> Researcher updates docs
                   └─> Lead reviews and merges
```

### Pattern 2: Training Experiment

```
1. Lead specifies experiment config
   └─> Engineer runs training (SFT or GRPO)
       └─> Engineer logs to wandb, saves checkpoint
           └─> Researcher runs benchmarks
               └─> Researcher logs results to research log
                   └─> Lead decides next steps
```

### Pattern 3: Bug Fix

```
1. Lead reports bug
   └─> QA investigates, reproduces, identifies root cause
       └─> Engineer fixes
           └─> QA verifies fix + adds regression test
               └─> Researcher updates troubleshooting.md
```

---

## Agent Communication

### Communication Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                                YOU                                   │
│                       (Only talks to Team Lead)                     │
└─────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ Questions, Progress, Approvals
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            TEAM LEAD                                 │
│                                                                     │
│  • Receives YOUR requirements → Breaks into tasks                  │
│  • Asks YOU questions (via AskUserQuestion)                        │
│  • Reports progress summaries to YOU                               │
│  • Delegates tasks to teammates                                    │
│  • Synthesizes teammate results for YOU                            │
└─────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ Task assignments, Results
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            TEAMMATES                                 │
│                                                                     │
│  Engineer ◄──────────► QA ◄──────────► Researcher                  │
│                                                                     │
│  • Communicate directly with EACH OTHER                            │
│  • Report results to TEAM LEAD                                     │
│  • Do NOT contact YOU directly (go through Lead)                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Communication Rules

| From | To | Allowed? | How |
|------|----|----------|-----|
| **You** | Team Lead | Yes | Direct conversation |
| **You** | Teammates | Emergency only | Shift+Down to select |
| **Team Lead** | You | Yes | AskUserQuestion, progress reports |
| **Team Lead** | Teammates | Yes | Task assignment, coordination |
| **Teammates** | You | No | Must go through Team Lead |
| **Teammates** | Team Lead | Yes | Results, questions, blockers |
| **Teammates** | Each other | Yes | Direct messages |

---

## When Lead Should Ask You

| Situation | Example |
|-----------|---------|
| **Multiple valid approaches** | "Engineer proposes two projector options. Which do you prefer?" |
| **Resource trade-offs** | "Batch size 8 uses 60GB VRAM. Reduce to 4?" |
| **Breaking changes** | "This refactor changes the checkpoint format. Approve?" |
| **Unclear requirements** | "Should stability evaluation include all mutation types?" |
| **Phase completion** | "SFT complete. Results: GO F1=0.42. Continue to GRPO?" |
| **Blocker escalation** | "Training produces NaN loss. How to proceed?" |
| **Teammate disagreement** | "Engineer and QA disagree on approach. Options..." |

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

### Task Sizing
- **Too small**: "Fix typo in docstring" — Just do it
- **Too large**: "Implement full GRPO pipeline" — Break into subtasks
- **Right size**: "Implement GRPO reward computation" — Clear deliverable

### File Ownership
- Each agent owns specific directories (see specs above)
- Avoid multiple agents editing the same file
- Use interfaces for cross-agent dependencies

### Communication
- Use shared task list for coordination
- Direct message for specific requests
- Broadcast only for team-wide announcements (use sparingly)

### Quality Gates
- QA reviews all code before merge
- Researcher validates training outputs via benchmarks
- Researcher ensures documentation stays current

### Token Efficiency
- 3 agents is the sweet spot for most tasks
- Shut down idle agents to save tokens
- Keep messages concise — avoid echoing full configs

---

## Scientist Team (Separate)

The **scientist team** (data-collector, analyst, reporter) handles training metric analysis and blog post generation. See `SCIENTIST_TEAM.md` for details. These agents are independent of the SWE team.

---

## Cleanup

```
> Shut down all teammates and clean up the team
```

---

## References

- [Project Architecture](docs/architecture.md)
- [Training Guide](docs/training_guide.md)
- [Research Log](docs/research/agents_research_log.md)
- [Agent Team Guide](https://code.claude.com/docs/en/agent-teams)
