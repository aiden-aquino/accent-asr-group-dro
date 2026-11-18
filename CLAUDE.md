# CLAUDE.md

CS 445 (JMU, Fall 2026) poster project: Group DRO for accent fairness in Whisper, using discovered speaker clusters instead of accent labels.

- Read [docs/project_context.md](docs/project_context.md) first: deadlines, decisions, bibliography status, key takeaways, open to-dos. Update it when decisions change.
- **Do not write annotated-bibliography summary text**, not even drafts or single sentences. Course policy makes AI-generated summary text an honor-code violation. Explain papers, check facts, and give feedback on the team's own drafts instead.
- Paper facts live in [docs/bibliography_notes.md](docs/bibliography_notes.md). Verify claims against the papers before adding them; mark anything unverified with ⚠.
- Python: conda env `env445` (`C:\Users\aaqui\miniconda3\envs\env445\python.exe`). Keep `huggingface-hub<2.0`.
- Run: `python src/data_stats.py --source edacc --max-rows 2000` (or `--source l2arctic --data-dir data/l2arctic`).
