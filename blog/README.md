# Project Blog

Development blog for the **Post-Training Protein LLM** project.

## Conventions

- All blog posts are written in **Markdown** (`.md`)
- Filename format: `YYYY-MM-DD_title-in-kebab-case.md`
  - Example: `2026-02-19_project-kickoff.md`
- Each post starts with a YAML front-matter block:
  ```yaml
  ---
  title: "Post Title"
  date: YYYY-MM-DD
  author: yeopjin
  tags: [tag1, tag2]
  ---
  ```
- Posts are stored flat in this `blog/` directory (no subdirectories)
- Newest posts appear first when sorted alphabetically (by date prefix)
- Tags should be drawn from: `kickoff`, `architecture`, `training`, `evaluation`, `data`, `rl`, `sft`, `infrastructure`, `milestone`, `debug`

## Reading Order

| Date | Title |
|------|-------|
| 2026-02-25 | [Building a 4.5M-Record SFT Dataset: Six Sources, One Pipeline](2026-02-25_combined-sft-dataset-assembly.md) |
| 2026-02-24 | [Protein Boundary Tokens, Instruct Model Fix, and Full Pipeline Validation](2026-02-24_boundary-tokens-and-instruct-fix.md) |
| 2026-02-20 | [Baseline Experiment Results: Where We Stand](2026-02-20_baseline-experiment-results.md) |
| 2026-02-19 | [Project Kickoff](2026-02-19_project-kickoff.md) |
