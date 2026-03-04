# Project Blog

Development blog for the **Post-Training Protein LLM** project.

## Directory Structure

```
blog/
├── index.html              # Blog index — open this to browse
├── README.md               # This file
├── posts/                  # HTML blog posts
│   ├── 2026-03-02_three-way-sft-comparison.html
│   ├── 2026-03-02_mlp-vs-text-sft-comparison.html
│   └── ...
├── figures/                # All plot images (PNGs)
│   ├── three_way_loss_curves.png
│   ├── mlp_vs_text_training_loss.png
│   └── ...
└── data/                   # Analysis code and data by date
    ├── 03-02/
    │   ├── run_histories.csv
    │   ├── experiment_metadata.json
    │   ├── analysis_summary.json
    │   ├── generation_samples.json
    │   └── three_way_analysis.py
    └── 02-21/
        └── ...
```

## How to Browse

Open [`index.html`](index.html) in a browser. It links to all posts in `posts/`.

## Conventions

- Post filenames: `YYYY-MM-DD_title-in-kebab-case.html` in `posts/`
- Figures: stored in `figures/`, referenced from posts as `../figures/name.png`
- Data/code: stored in `data/MM-DD/`, referenced from posts as `../data/MM-DD/`
- Index links use `posts/filename.html`; posts link back via `../index.html`
- Tags: `kickoff`, `architecture`, `training`, `evaluation`, `data`, `rl`, `sft`, `infrastructure`, `milestone`, `debug`

## Reading Order

| Date | Title |
|------|-------|
| 2026-03-02 | [The UniProt Convergence Problem](posts/2026-03-02_scaling-multi-source-protein-data-challenges.html) |
| 2026-03-02 | [Lower Loss, Worse Outputs: The eval_loss Paradox](posts/2026-03-02_three-way-sft-comparison.html) |
| 2026-03-02 | [ESM-3 Encoder Cuts Loss by 25%: MLP vs Text-Only SFT](posts/2026-03-02_mlp-vs-text-sft-comparison.html) |
| 2026-02-25 | [Building a 4.5M-Record SFT Dataset](posts/2026-02-25_combined-sft-dataset-assembly.html) |
| 2026-02-24 | [Protein Boundary Tokens, Instruct Model Fix](posts/2026-02-24_boundary-tokens-and-instruct-fix.html) |
| 2026-02-20 | [Baseline Experiment Results](posts/2026-02-20_baseline-experiment-results.html) |
| 2026-02-19 | [Project Kickoff](posts/2026-02-19_project-kickoff.html) |
