# Small-molecule transfer (LiF, BeO, BN, C2 -> N2, CO)

Status 2026-11-03: run (jobs 250442 transfer; 250444/5/6 baselines gpt2-small/dag-gnn/diff-gnn).
104 of 105 runs complete; `smallmol-base-gpt2-small-n2-r1.60-s2` hit NaN logits at epoch 357
(partial data kept). Evaluation script: `scratch/explore_smallmol.py`.

## Question

Does a policy trained on some small molecules help on others it never saw?
H-chain transfer currently does not work (H10 at 1.6 A: zero-shot 255 +/- 22 mHa,
untrained 377-437, trained directly 65.3 +/- 2.9), but H-chains are the hardest
family in the scan (~36% of the CI space). These need 0.1-12.3%.

## Setup

All 16 qubits, STO-3G, full valence, so this is transfer across chemistry, not size.

| split | molecule | bond lengths (A) |
|---|---|---|
| train | LiF | 1.55, 1.90, 2.25, 2.95 |
| train | BeO | 1.25, 1.50, 2.00, 2.50 |
| train | BN | 1.25, 1.50, 2.00, 2.50 |
| train | C2 | 1.27, 1.55, 2.10, 2.65 |
| zero-shot test | N2 | 1.07, 1.60, 2.50 |
| zero-shot test | CO | 1.16, 1.69, 2.50 |

L=15, 10,000 shots, dedup on, CCSD-screened pool, coverage 0.25 (2x the worst
requirement, BN at 1.25 A: 604/4,900 = 12.3%; cap 784 for N2/CO, 1,225 otherwise),
640 epochs (16 molecules x 40), zero-shot evaluation every 80 epochs, 5 seeds.

| model | params | lr | transfer config | baseline configs |
|---|---|---|---|---|
| gpt2-small | 459,552 | 2e-5 | `smallmol_gpt2_small` | `smallmol_base_gpt2_small_<mol>_<r>` |
| dag-gnn | 437,120 | 2e-5 | `smallmol_dag_gnn` | `smallmol_base_dag_gnn_<mol>_<r>` |
| diff-gnn | 423,936 | 4e-5 | `smallmol_diff_gnn` | `smallmol_base_diff_gnn_<mol>_<r>` |

The lr values are each model's best on N2 in the sweep, so tuned on one molecule.
Baselines train from scratch on each test geometry with the same 640 epochs;
their epoch 0 is the untrained control. Compare GQE-optimized (before refinement):
refinement at this cap is close to exact.

## CCSD at these geometries: N2 at 2.5 A does not converge

The pool, its angles and the `amplitude` feature come from CCSD, and
`PySCFMolecule.compute_ccsd` never checks `converged` (default 50 cycles).
Checked for all 22 geometries:

- **20 of 22 converge.** SCF converges everywhere.
- **N2 at 2.5 A does not converge**, and more cycles do not fix it: at 1,000 cycles
  it goes to NaN. The unconverged energy is not reproducible either (-107.153 in
  the pipeline, -106.986 in a separate script with the same 50 cycles). Also
  unconverged in that script: 2.13, 2.30, 2.40 A at 50 cycles; 2.30 and 2.40 also at
  1,000; 2.13 converges at 1,000 but to -107.591, far below the CASCI range.
- **CO at 2.5 A** is only slow: not converged at 50, converged at 1,000 with the same
  energy (-110.856), so the amplitudes used are effectively right.

Consequences:

- **Every N2 result at 2.5 A so far used a pool from unconverged CCSD amplitudes.**
  It is still a usable heuristic menu (the oracle check found all of the exact top
  102 determinants reachable from it), but it is not "the CCSD pool" in the sense a
  chemist would assume. Likely, not proven, cause of "N2's pool differs by machine".
- **"CCSD misses N2 at 2.5 A by 283-287 mHa" is not a valid CCSD error**: that
  number is from an unconverged iteration. Do not quote it as one. What can be said is
  that CCSD fails to converge there, which is the textbook breakdown for a stretched
  triple bond.
- N2 at 2.5 A stays in the test set although the plan said to replace failing
  geometries: every stretched N2 geometry tried above 2 A fails at the 50 cycles the
  pipeline uses, and it is the benchmark geometry of all earlier runs. Flagged in the
  results instead. To drop it, remove it from `configs/molecule_set/small_molecules.yaml`
  and the job lists.

CCSD minus CASCI per geometry (mHa; negative = CCSD below exact, CCSD is not variational):

| | | | | |
|---|---|---|---|---|
| LiF | 1.55: 16.5 | 1.90: 24.9 | 2.25: 21.6 | 2.95: 1.8 |
| BeO | 1.25: 12.0 | 1.50: 26.5 | 2.00: 9.8 | 2.50: -4.4 |
| BN | 1.25: -15.5 | 1.50: 32.5 | 2.00: 118.8 | 2.50: 109.3 |
| C2 | 1.27: -25.7 | 1.55: 26.4 | 2.10: 6.4 | 2.65: -18.7 |
| N2 | 1.07: 3.5 | 1.60: 12.7 | 2.50: not converged | |
| CO | 1.16: 9.1 | 1.69: 51.2 | 2.50: 177.5 | |

## Smoke test (2026-11-02, local)

All three transfer configs at 17 epochs: 16 train + 6 eval molecules in `results.csv`,
parameter counts as above, subspaces well under the cap. Three baselines at 2 epochs ok.
Pool sizes: 59-201 on train molecules (C2 smallest), 104-160 on test molecules.

## Results

All numbers: best circuit of the 10 sampled per pass, before refinement, mHa above
CASCI, mean +/- std over 5 seeds. "Untrained" is epoch 0 of the baselines (10 circuits,
same as one zero-shot pass). The CASCI references of the six TEST geometries are
correct (checked, see below); several training references are not.

| model | geometry | untrained | zero-shot ep 79 | zero-shot ep 639 | from scratch 100 ep | from scratch 640 ep |
|---|---|---|---|---|---|---|
| gpt2-small | N2 1.07 | 53.5 +/- 6.2 | 44.0 +/- 5.3 | 53.4 +/- 9.2 | 22.9 +/- 2.6 | 15.3 +/- 2.1 |
| | N2 1.60 | 86.4 +/- 24.4 | 37.8 +/- 26.2 | 41.7 +/- 38.5 | 10.1 +/- 2.0 | 6.9 +/- 2.2 (n=4) |
| | N2 2.50 | 21.8 +/- 17.2 | 21.7 +/- 29.1 | 44.3 +/- 36.3 | 5.1 +/- 0.6 | 2.7 +/- 0.6 |
| | CO 1.16 | 61.7 +/- 15.2 | 38.8 +/- 4.0 | 40.0 +/- 5.1 | 29.5 +/- 4.0 | 25.9 +/- 2.4 |
| | CO 1.69 | 101.2 +/- 15.3 | 44.0 +/- 11.7 | 64.5 +/- 40.5 | 23.4 +/- 2.6 | 15.2 +/- 5.3 |
| | CO 2.50 | 25.5 +/- 10.7 | 13.7 +/- 5.5 | 43.9 +/- 77.0 | 2.2 +/- 0.3 | 1.8 +/- 0.0 |
| dag-gnn | N2 1.07 | 57.1 +/- 5.9 | 48.5 +/- 8.3 | 73.4 +/- 18.9 | 21.7 +/- 4.3 | 19.9 +/- 3.6 |
| | N2 1.60 | 116.8 +/- 14.9 | 71.4 +/- 41.3 | 144.7 +/- 143.9 | 10.0 +/- 1.6 | 8.2 +/- 1.5 |
| | N2 2.50 | 47.5 +/- 12.8 | 43.6 +/- 35.0 | 140.5 +/- 95.1 | 5.7 +/- 0.6 | 5.2 +/- 1.2 |
| | CO 1.16 | 70.1 +/- 5.4 | 51.9 +/- 9.8 | 61.1 +/- 7.7 | 29.0 +/- 2.8 | 25.8 +/- 2.3 |
| | CO 1.69 | 98.5 +/- 6.1 | 67.2 +/- 20.8 | 100.2 +/- 28.9 | 31.2 +/- 3.1 | 27.7 +/- 4.7 |
| | CO 2.50 | 28.0 +/- 2.6 | 19.9 +/- 11.6 | 16.5 +/- 13.7 | 2.4 +/- 0.3 | 1.9 +/- 0.2 |
| diff-gnn | N2 1.07 | 53.2 +/- 6.1 | 49.1 +/- 3.4 | 43.6 +/- 7.3 | 21.8 +/- 2.9 | 17.5 +/- 3.2 |
| | N2 1.60 | 93.8 +/- 26.3 | 33.0 +/- 6.4 | 28.4 +/- 9.8 | 10.3 +/- 2.9 | 8.9 +/- 1.5 |
| | N2 2.50 | 40.8 +/- 32.4 | 36.6 +/- 30.0 | 30.6 +/- 33.5 | 4.8 +/- 0.8 | 3.8 +/- 1.2 |
| | CO 1.16 | 65.8 +/- 6.6 | 45.7 +/- 2.8 | 51.9 +/- 10.3 | 30.9 +/- 2.6 | 29.6 +/- 2.6 |
| | CO 1.69 | 101.1 +/- 21.7 | 59.2 +/- 14.7 | 53.3 +/- 17.2 | 30.4 +/- 4.4 | 30.4 +/- 4.4 |
| | CO 2.50 | 19.1 +/- 5.5 | 13.0 +/- 5.7 | 15.2 +/- 5.3 | 2.1 +/- 0.2 | 1.7 +/- 0.0 |

Findings:

1. **Transfer helps a little, early.** After 80 epochs on the training molecules, the
   zero-shot policy beats the untrained one on most test geometries, most clearly at
   the intermediate bond lengths (N2 1.60: 86-117 -> 33-71; CO 1.69: 98-101 -> 44-67).
   Near equilibrium and at 2.50 A the gain is small and inside the seed spread.
2. **It is worth only a few epochs of direct training.** The final zero-shot result is
   matched by the from-scratch baseline after 0-9 epochs (first epoch at which the
   baseline's mean best-so-far is as good). The baseline's best-so-far by then covers
   more circuits than one zero-shot pass (10 x (epoch + 1)), so this slightly favours the
   baseline, but the gap to 100 epochs of direct training (2-31 mHa) is large either way.
3. **Zero-shot gets worse with more training for gpt2-small and dag-gnn** (dag-gnn N2
   2.50: 44 -> 141; N2 1.60: 71 -> 145), as on the H-chains. **diff-gnn stays flat or
   improves slightly** (N2 1.60: 33 -> 28; CO 1.69: 59 -> 53), and is the most stable
   transfer model here.
4. **No transfer run reaches chemical accuracy** on any test geometry; the from-scratch
   baselines reach about 2 mHa at CO 2.50 and 3-5 mHa at N2 2.50.

Candidate cause of finding 3, untested: the inverse temperature beta grows by 0.02 per
epoch from 0.5, so it reaches 13.3 at epoch 640 (2.5 at the end of a 100-epoch run). A very
sharp policy late in training fits the training molecules and samples narrowly. Signs of
it: the one NaN failure (gpt2-small baseline at epoch 357, beta 7.6) and dag-gnn
baselines producing fewer distinct circuits per epoch at the end (7.2 of 10 at N2 1.60).
A run with beta capped or the ramp scaled to the run length would test it.

## Re-sampling the final policies at different beta (2026-11-03)

`hpc/eval_checkpoint_beta.py` loads a final checkpoint (epoch 639, the only one kept)
and samples the six test geometries zero-shot at fixed beta, the same way
`_zeroshot_eval` does; 5 batches of 10 circuits per point, all 15 transfer runs.
Summary: `scratch/explore_beta_eval.py`, data in `scratch/beta_eval/`. At beta 13.3
it reproduces the values the runs logged at epoch 639 (within noise).

Best circuit of 10, mHa, mean over batches and seeds:

| model | geometry | logged ep 79 (beta ~2) | logged ep 639 (beta 13.3) | beta 0.5 | beta 2 | beta 4 | beta 13.3 |
|---|---|---|---|---|---|---|---|
| gpt2-small | N2 1.07 | 44.0 | 53.4 | 32.8 | 26.4 | 32.9 | 43.1 |
| | N2 1.60 | 37.8 | 41.7 | 84.9 | 58.4 | 44.2 | 46.5 |
| | N2 2.50 | 21.7 | 44.3 | 88.6 | 36.6 | 26.9 | 37.6 |
| | CO 1.16 | 38.8 | 40.0 | 50.3 | 43.9 | 44.0 | 47.7 |
| | CO 1.69 | 44.0 | 64.5 | 72.7 | 57.8 | 50.8 | 49.3 |
| | CO 2.50 | 13.7 | 43.9 | 14.6 | 14.4 | 12.1 | 37.9 |
| dag-gnn | N2 1.07 | 48.5 | 73.4 | 34.8 | 42.7 | 49.8 | 89.2 |
| | N2 1.60 | 71.4 | 144.7 | 111.3 | 76.2 | 84.9 | 146.1 |
| | N2 2.50 | 43.6 | 140.5 | 146.6 | 98.2 | 121.1 | 154.7 |
| | CO 1.16 | 51.9 | 61.1 | 50.1 | 52.5 | 53.9 | 80.3 |
| | CO 1.69 | 67.2 | 100.2 | 102.7 | 65.8 | 63.6 | 73.1 |
| | CO 2.50 | 19.9 | 16.5 | 19.7 | 15.0 | 17.1 | 19.8 |
| diff-gnn | N2 1.07 | 49.1 | 43.6 | 28.6 | 40.9 | 42.8 | 42.5 |
| | N2 1.60 | 33.0 | 28.4 | 30.8 | 30.7 | 29.1 | 29.9 |
| | N2 2.50 | 36.6 | 30.6 | 16.6 | 9.7 | 12.5 | 17.2 |
| | CO 1.16 | 45.7 | 51.9 | 42.6 | 44.4 | 49.0 | 50.8 |
| | CO 1.69 | 59.2 | 53.3 | 35.4 | 32.2 | 35.6 | 35.7 |
| | CO 2.50 | 13.0 | 15.2 | 13.6 | 15.2 | 15.8 | 16.4 |

(The logged values are single passes of 10 circuits; the re-sampled ones average 5, so
they are less noisy. Seed spread at beta 8-13.3 is large for gpt2-small and dag-gnn,
20-140 mHa.)

- **The late degradation is mostly the beta ramp, not the policy.** At beta 2-4 the final
  gpt2-small policy is back at its epoch-79 level on every geometry (N2 1.07 better:
  26 vs 44), and dag-gnn on 5 of 6.
- **Exception: dag-gnn on stretched N2** (2.50 A: 98 at best vs 44 at epoch 79). There
  the policy itself got worse.
- **High beta collapses the DAG GNN**: 6-8 distinct circuits of 10 at beta 8-13.3 on N2
  1.07/1.60 and CO 1.16; the transformer and diff-gnn stay at 10.
- **diff-gnn barely depends on beta**, which is why it looked stable in training.
- **No single beta is best**: equilibrium geometries prefer low beta (0.5-2), stretched ones
  higher (gpt2-small: 4-8). Lower beta flattens the policy toward its prior; higher beta
  follows what it learned.
- Practical consequence: cap the beta ramp (around 4) or scale it to the run length;
  0.5 + 0.02 per epoch was set for 100-epoch runs, where it ends at 2.5. The H-chain run
  (540 epochs, beta 11.3 at the end) probably suffered from the same thing.

## Wrong CASCI references for 8 of the 16 training geometries

`PySCFMolecule.compute_casci` (molecule.py, core) calls the CASCI FCI solver for one root.
Checked against a 6-root full CI (pyscf `direct_spin1`) with S^2 for every geometry:

| geometry | reference used | lowest Ms=0 state | difference |
|---|---|---|---|
| BeO 2.00 | singlet | triplet | 42.0 mHa |
| BeO 2.50 | singlet | triplet | 84.9 mHa |
| BN 1.25 | triplet | singlet | 34.9 mHa |
| BN 2.00 | triplet | lower triplet | 28.7 mHa |
| C2 1.27 | triplet | singlet | 42.2 mHa |
| C2 1.55 | triplet | lower triplet | 35.8 mHa |
| C2 2.10 | singlet | triplet | 6.3 mHa |
| C2 2.65 | singlet | quintet | 1.5 mHa |

LiF, BeO 1.25/1.50, BN 1.50/2.50 and **all six test geometries (N2, CO) are correct**,
all with a singlet ground state.

Consequences:

- The reported errors on these training molecules are wrong, which is why some came out
  negative (BeO 2.50: -86 mHa). Training itself is unaffected: the reward is the QSCI
  energy, not the reference.
- **The lowest state QSCI targets is the lowest Ms = 0 state, which can be a triplet or
  quintet**: nothing restricts the diagonalization to S = 0. For BeO and BN when
  stretched, and C2 at 2.10/2.65, the training target is therefore not a singlet, while
  pool and CCSD are built from a closed-shell RHF reference. For the chemists, a choice
  to make explicit (target the singlet, or the lowest state of any spin).
- The fix belongs in molecule.py (core): solve for several roots, or use the same pyci
  full CI as the subspace scan, and report S^2.
