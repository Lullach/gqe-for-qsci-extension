# Small-molecule transfer (LiF, BeO, BN, C2 -> N2, CO)

Status 2026-11-02: configs written and smoke-tested, NOT yet run. Results go here.

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

Pending.
