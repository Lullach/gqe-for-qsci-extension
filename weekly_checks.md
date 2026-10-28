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

## 2026-10-28
- Function: DeterminantSubspace._symmetry_completion (subspace.py), 3 questions, plus 3 teach-back on the direct_qsci diff (post_select, dedup pressure, max_tokens vs n_dets)
- Weak spots: what post_select_by_nelec actually checks (particle number, not duplicates - so a fired assertion could equally be tokens_to_packed or an nelec mismatch); max_tokens vs n_dets (capacity is fixed at construction, budget varies per molecule). Strong: why spin completion exists, that truncating mid-group breaks it, why `seen` is needed even without duplicate input. He also correctly challenged my claim that advantages vanish on collapse - they are std-normalised, so they stay O(1); only the exact-tie early return degenerates.
