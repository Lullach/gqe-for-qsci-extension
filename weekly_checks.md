# Weekly checks

One entry per weekly check: the core function quizzed, and where the answers
were shaky. Written by Claude at the end of each check; the date also goes to
`.git/last_weekly_check`, which the pre-commit hook reads.

## 2026-10-05
- Function: none (empty first check; weekly checks start from here)
- Weak spots: n/a

## 2026-10-13
- Function: QSCIResult.energies (gqe_qsci/qsci/schema.py), 5 questions, all correct
- Weak spots: Q4 answered the idea, not the mechanism (the zip over states["idx"] and sampler.run output in process)
