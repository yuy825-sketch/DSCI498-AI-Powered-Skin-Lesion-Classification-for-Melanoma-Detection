# Runs (local)

This folder follows a reproducible “run contract” and is intentionally **not pushed** to GitHub by default.

Each run should store:
- immutable `config.*`
- `meta.json` (command, seed, git state)
- `metrics.json` (scalar metrics) and optional `eval.csv`
- optional `notes.md`

Small, presentation-ready outputs should be copied into `results/` for GitHub.

