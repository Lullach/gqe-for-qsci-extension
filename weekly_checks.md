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

## 2026-10-20
- Function: CircuitDAGGNNPolicy.log_prob (dag_gnn.py), 5 questions (Q4 skipped as plumbing), plus 5 extra on operator_pool, train_pipeline/buffer, entropy scan
- Weak spots: what log_prob returns (B, L per placed gate, not B, V) and why GRPO needs it (the ratio); when the -inf error fires (stale buffer/checkpoint); why reusing a rollout 30x is safe (ratio + clipping). Strong: mask/ordering/beta data flow, pool and feature design
