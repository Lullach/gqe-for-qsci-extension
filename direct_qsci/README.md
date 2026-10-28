# direct_qsci — QSCI with the circuit removed

## What this is

The same learning problem as the main pipeline, minus the quantum part:

    circuit arm   policy -> gates -> circuit -> 10k shots -> determinants -> QSCI energy
    direct arm    policy ----------------------> determinants -> QSCI energy

Everything downstream of "determinants" is shared code, so the two arms differ
only in where the determinants come from.

## What it is for

It is the **ceiling / ablation** arm, not a proposed method. Three reasons it
earns its place:

1. It measures what the circuit costs. Same policy family, same features, same
   GRPO loop, same budget K, same diagonalisation.
2. It removes CCSD entirely. No pool screening, no gate angles, no amplitude
   feature — so the MP2-vs-CCSD angle problem cannot arise.
3. It can reach determinants no perturbative criterion ranks. At stretched N2
   the important set contains excitation levels up to 6, and quadruples have
   exactly zero coupling to HF by Slater-Condon.

Prior art exists for learned determinant selection (Coe's MLCI 2018, Goings's
RLCI 2021, and the 2026 SQD/ML line — QSCI-RBM, classifier-enhanced SQD,
learning-to-rank). What is not in the literature is this controlled comparison:
one policy, with and without a circuit, on one active space.

## Layout

    policy.py     DeterminantGPT2Policy: emits K determinants as orbital tokens
    pipeline.py   DirectQSCIPipeline: determinants -> subspace -> energy
                  (Lukas, in progress)

## The contract

`sample_sequence` writes only `state["idx"]` = BOS + K*T orbital tokens. That is
the action, so the replay buffer, GRPO and training_step are untouched. The
pipeline converts tokens to determinants with `tokens_to_packed`, keeping one
source of truth for what was sampled.
