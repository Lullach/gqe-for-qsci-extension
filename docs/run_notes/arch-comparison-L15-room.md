# Architecture comparison with room in the subspace (N2, LiF, L=15)

Status 2026-10-14. Results of three ABCI-Q jobs plus a diagnosis of the pointer
policy. Written as the entry point for follow-up work (pointer theory in a
separate chat); every number here can be regenerated from the files listed at
the end.

## Setup

Six policies x two molecules x seeds 1-5, then the three transformer policies
again at ~0.46M parameters. Everything except the model block is identical across
all runs (checked field by field in the saved `hydra/config.yaml`; the only other
difference is the pointer's `operator_pool.ccsd_screening: false`, by design).

| | N2 | LiF |
|---|---|---|
| system | 2.5 A, (10e, 8o) active space, 3,136 determinants | 1.55 A, sto-3g, (8e, 8o), 4,900 determinants |
| character | strongly multireference | ionic, weakly correlated |
| chemical accuracy needs (top-\|c\|, subspace_requirement scan) | 102 determinants | 145 determinants |
| `qsci.max_dim` (2x that) | 204 | 290 |
| CASCI | -107.4404 | -105.435 |

Common: L=15 gates, 10,000 shots, 100 iterations x 30 steps, 10 circuits per
step, dedup on, GRPO (clip 0.2/0.28), AdamW lr 5e-6, wd 0.01, numpy statevector
sampler (`hpc/train_fast_sampler.py`) on rt_QC. W&B group `arch-comparison-L15-room`.

| job | date | runs | configs |
|---|---|---|---|
| 232062 | 2026-10-13 | N2, 6 models x 5 seeds | `configs/experiment/n2L15_*.yaml` |
| 232063 | 2026-10-13 | LiF, 6 models x 5 seeds | `configs/experiment/lifL15_*.yaml` |
| 234046 | 2026-10-14 | both, 3 transformers at ~0.46M x 5 seeds | `configs/experiment/*L15_*_small.yaml` |

(Job 234043 is void: it ran before the `*_small` configs were pulled on the
cluster, and Hydra silently fell back to the defaults. Fixed in e2d7731.)

Models:

| family | class | params | notes |
|---|---|---|---|
| dag-gnn | CircuitDAGGNNPolicy | 437K | 128 wide, 6 layers |
| diff-gnn | CircuitGNNModelAbsorbing | 424K | absorbing diffusion, GAT denoiser, 128 wide |
| pointer | PointerDAGGNNPolicy | 968K | DAG GNN + orbital encoder; doubles only (`allow_singles: false`), MP2 angles, no CCSD |
| gpt2 | GPT2Model | 43.1M | `size: small` = 768 wide, 6 layers, 6 heads (every GPT-2 run before 2026-10-14) |
| gpt2-small | GPT2Model | 460K | `size: matched` = 96 wide, 4 layers, 4 heads |
| diff-absorb | CircuitDiffusionModelAbsorbing | 6.4M | 256 wide, 8 layers, T=16 |
| diff-absorb-small | same | 461K | 96 wide, 4 layers, 4 heads |
| diff-1shot | CircuitDiffusionModelSingleShot | 6.4M | 256 wide, 8 layers |
| diff-1shot-small | same | 461K | 96 wide, 4 layers, 4 heads |

## Results

Error in mHa above CASCI, mean +/- std over 5 seeds of each run's best-so-far.
"GQE-optimized" = best single circuit's QSCI energy; "global-refined" = the
refinement over everything the run has seen.

| family | params | N2 GQE-opt | N2 refined | LiF GQE-opt | LiF refined |
|---|---|---|---|---|---|
| gpt2 | 43.1M | **3.75 +/- 0.74** | 0.31 +/- 0.05 | **6.29 +/- 0.75** | 0.43 +/- 0.01 |
| diff-absorb | 6.4M | 5.10 +/- 0.97 | 0.29 +/- 0.01 | 7.65 +/- 0.62 | 0.42 +/- 0.02 |
| diff-1shot | 6.4M | 5.58 +/- 0.58 | 0.51 +/- 0.34 | 6.69 +/- 0.79 | 0.41 +/- 0.01 |
| gpt2-small | 460K | 5.10 +/- 0.67 | 0.34 +/- 0.03 | 7.21 +/- 1.56 | 0.41 +/- 0.01 |
| diff-absorb-small | 461K | 5.38 +/- 0.43 | 0.29 +/- 0.05 | 7.80 +/- 1.14 | 0.41 +/- 0.01 |
| diff-1shot-small | 461K | 5.44 +/- 0.43 | 0.37 +/- 0.16 | 7.15 +/- 0.54 | 0.41 +/- 0.01 |
| diff-gnn | 424K | 5.19 +/- 0.59 | 0.28 +/- 0.04 | 7.31 +/- 0.38 | 0.43 +/- 0.03 |
| dag-gnn | 437K | 5.88 +/- 0.31 | 0.60 +/- 0.33 | 7.60 +/- 0.98 | 0.44 +/- 0.01 |
| pointer | 968K | 5.85 +/- 0.54 | 2.34 +/- 0.48 | 32.89 +/- 1.02 | 23.16 +/- 0.11 |

Best single circuit over all 5 seeds (mHa), and subspace sizes:

| family | N2 best | N2 dim of best / median dim, all circuits | LiF best | LiF dim of best / median |
|---|---|---|---|---|
| gpt2 | 2.70 | 143 / 86 | 5.60 | 157 / 89 |
| diff-absorb | 3.59 | 137 / 65 | 6.62 | 132 / 70 |
| diff-1shot | 4.58 | 112 / 67 | 5.42 | 137 / 83 |
| gpt2-small | 3.94 | 136 / 68 | 5.47 | 137 / 66 |
| diff-absorb-small | 4.92 | 136 / 64 | 5.82 | 136 / 64 |
| diff-1shot-small | 4.89 | 121 / 65 | 6.55 | 140 / 73 |
| diff-gnn | 4.25 | 114 / 65 | 7.01 | 114 / 73 |
| dag-gnn | 5.55 | 103 / 62 | 5.85 | 145 / 71 |
| pointer | 5.27 | 94 / 37 | 31.56 | 39 / 21 |

### Findings

1. **No model reaches chemical accuracy before refinement.** Best single
   circuit: 2.70 mHa (N2, gpt2 43M). Every model except the pointer does after
   refinement.
2. **The room in the subspace is not used.** Not one of the 30,000 circuits per
   molecule reached `max_dim`. The best circuits give 35-110 distinct bitstrings
   from 10k shots, 80-160 determinants after spin completion (N2). The binding
   constraint is what the circuits produce, not the cap.
3. **On N2 the problem is which determinants, not how many.** The best circuits
   hold 100-160 determinants, more than the 102 the top-|c| selection needs,
   and sit at 4-6 mHa.
4. **Global refinement erases the architecture differences** (N2 0.28-0.6, LiF
   0.41-0.44 for every pool model). It keeps the top `max_dim` by amplitude out
   of the union of every circuit seen in training, so with a cap of 2x the
   requirement it measures how much the run explored, not how good its circuits
   are. The GQE-optimized column is the one that ranks policies. Its classical
   cost is negligible here (~10 ms per step at these sizes; it is bounded by
   2 x `max_dim` and does not grow over training).
5. **At equal size (~0.45M) the architectures are tied.** N2 GQE-optimized spans
   5.10-5.88 with seed std 0.3-0.7; with 5 seeds, differences below ~0.7 mHa are
   not resolvable. Same on LiF.
6. **GPT-2's lead was capacity (or width), not architecture.** 43M -> 0.46M:
   N2 3.75 -> 5.10, LiF 6.29 -> 7.21. Shrinking the diffusion models from 6.4M
   to 0.46M changed almost nothing (5.10 -> 5.38, 5.58 -> 5.44).
7. **Only the large GPT-2 keeps improving after epoch ~20**; everything else is
   flat by then (`figures/L15_room_n2L15/error_*.png`). All models do learn: the
   mean error of sampled circuits drops by 90-190 mHa (N2) from epochs 0-4 to
   95-99. At the end, gpt2 43M samples 8.1 mHa on average (N2) vs 18-27 for the
   rest, and its circuits are about 50% larger (median 109 vs 63-73
   determinants). The 6.4M diff-absorb samples worse on average than its 0.46M
   version (57.6 vs 18.3 mHa), although its best circuits are similar.

### Open: why does the large GPT-2 learn further?

Two candidate explanations, not tested yet:

- **Width x learning rate.** Same AdamW lr for all; Adam moves each weight by
  ~lr per step, so the per-step change of the logits grows with layer width
  (768 for gpt2 vs 256 for the 6.4M diffusion models vs 96-128 for the rest).
  Parameter count would then matter less than width.
- **Gradient signal.** GPT-2's log-prob is exact and autoregressive; the
  single-shot diffusion model samples positions independently (cannot represent
  gate-gate correlations at any size); the absorbing model's trajectory log-prob
  follows one random unmasking order (noisier).

Tested 2026-10-14 (local runs, `hpc/run_local.sh`, 5 seeds each):
`{n2,lif}L15_gpt2_small_lr8` (gpt2-small at lr 4e-5 = 8x) and
`{n2,lif}L15_gpt2_wide1` (768 wide, 1 layer, 6 heads, 7.7M, usual lr).

| family | params | lr | N2 GQE-opt | N2 refined | N2 best circuit | LiF GQE-opt | LiF refined |
|---|---|---|---|---|---|---|---|
| gpt2 | 43.1M | 5e-6 | 3.75 +/- 0.74 | 0.31 +/- 0.05 | 2.70 | 6.29 +/- 0.75 | 0.43 +/- 0.01 |
| gpt2-small | 460K | 5e-6 | 5.10 +/- 0.67 | 0.34 +/- 0.03 | 3.94 | 7.21 +/- 1.56 | 0.41 +/- 0.01 |
| gpt2-small-lr8 | 460K | 4e-5 | 3.83 +/- 1.03 | 0.21 +/- 0.04 | 2.26 | 6.70 +/- 0.71 | 0.41 +/- 0.01 |
| gpt2-wide1 | 7.7M | 5e-6 | 3.40 +/- 0.95 | 0.20 +/- 0.02 | 1.79 | 5.61 +/- 0.96 | 0.42 +/- 0.02 |

**The width x learning-rate explanation holds.** 8x the lr closes the gap on N2
(5.10 -> 3.83, level with the 43M model) and part of it on LiF. One 768-wide
layer is at least as good as six (nominally the best GPT-2 on both molecules),
so depth and parameter count are not what helped. Differences among gpt2,
lr8 and wide1 are within the seed noise (std ~1 mHa, 5 seeds); all three beat
gpt2-small.

**Consequence:** every comparison so far ran all models at the same lr 5e-6,
which gives wider models a larger effective step. The architecture ranking at
a shared lr is confounded with width; a fair comparison needs a per-model lr
(or an lr scaled with width).

### Learning-rate sweep (N2, 2026-10-15, local runs)

Configs `configs/experiment/n2L15_lrsweep_*.yaml`, lr passed as
`EXTRA="trainer.optimizer.lr=..."`; folders `n2L15-<model>-lr<lr>-s<seed>` (1e-4
is written `lr0.0001`). The `lr0.0001` folders of gpt2-small (s1-5) and
diff-absorb-small (s1-3) contain an interrupted run followed by the complete one;
use only the last run in each file (epoch counter resets to 0).

GQE-optimized, mHa, mean +/- std over 5 seeds (5e-6 = the original runs;
gpt2-small at 4e-5 = the lr8 runs):

| model | 5e-6 | 2e-5 | 4e-5 | 1e-4 |
|---|---|---|---|---|
| gpt2-small | 5.10 +/- 0.67 | **3.45 +/- 0.23** | 3.83 +/- 1.03 | 3.70 +/- 0.45 |
| diff-absorb-small | 5.38 +/- 0.43 | 4.71 +/- 0.39 | **3.54 +/- 0.45** | 4.02 +/- 0.41 |
| diff-1shot-small | 5.44 +/- 0.43 | 3.71 +/- 0.67 | 3.91 +/- 0.98 | **3.65 +/- 1.17** |
| diff-gnn | 5.19 +/- 0.59 | 3.79 +/- 1.06 | **2.81 +/- 0.60** | 4.91 +/- 1.29 |
| dag-gnn | 5.88 +/- 0.31 | **4.64 +/- 0.50** | 5.26 +/- 0.89 | 5.12 +/- 1.06 |

Global-refined at the best lr: 0.17-0.32 for all (uninformative, as before).
Best single circuits: diff-gnn 2.14, diff-1shot-small 2.18, gpt2-small 2.26 mHa.
At 1e-4 the GNNs become unstable (mean sampled error at the end: dag-gnn 175 mHa,
diff-gnn 28.6; refined dag-gnn 1.40 +/- 1.14).

- 5e-6 was too low for every model; every earlier comparison ran there.
- At each model's best lr, all but dag-gnn beat the 43M GPT-2 at 5e-6 (3.75).
- Ranking at the best lr: diff-gnn < gpt2-small ~ diff-absorb-small ~ diff-1shot-small < dag-gnn.
  The best lr was picked from the same 5 seeds it is scored on, so the best
  values are biased low; confirm on fresh seeds before ranking.

## Pointer diagnosis

Script: `scratch/explore_pointer_angles.py` (one molecule per process: a
`Factory()` memoizes its first operator pool and returns it for later configs).

The pointer builds each gate from four orbital pointers (i, j occupied; a, b
virtual) and gives it the angle 2 t_MP2 = 2 <ij||ab> / (eps_i + eps_j - eps_a -
eps_b) (`UCCSDBasedPool.mp2_angle`, `gqe_qsci/gqe/operator_pool.py`), using the
exact active-space integrals and HF orbital energies. Singles get a constant
`single_angle` (0.1) because MP2 gives them zero; in these runs they were off.
The pool models use 2 t_CCSD from the CCSD-screened pool (N2: 117 excitations,
12 singles; LiF: 200, 16 singles).

**(A) Gate angles, MP2 (pointer) vs CCSD (pool), on the pool's doubles:**

| | LiF (184 doubles) | N2 (105 doubles) |
|---|---|---|
| \|MP2\| / \|CCSD\|, median | 0.93 | 1.39 |
| 10-90% range | 0.60 - 1.18 | 0.09 - 8.9 |
| same sign | 100% | 73% |
| largest \|angle\| | CCSD 0.171, MP2 0.161 rad | CCSD 1.106, MP2 1.246 rad |

The two strongest N2 doubles have CCSD angles -1.106 and +0.930 rad and MP2
angles -0.096 and -0.088: nearly identity gates for exactly the excitations
that matter most. MP2 is first-order perturbation theory; it is fine for LiF and
breaks down for stretched N2 (small gaps, strong coupling). Caveat: compared only
on the pool's excitations; the pointer can build more (315 for N2) and those
were not checked.

**(B) Doubles-only circuits can only reach even excitation levels.** Each
pointer double moves two electrons between HF-occupied and HF-virtual orbitals
(or back), so the level changes by +/-2. Best energy from the exact ground
state's top-|c| determinants:

| | N2 | LiF |
|---|---|---|
| top-k excitation levels (k = requirement) | 0:1, 1:2, 2:39, 3:12, 4:41, 5:4, 6:3 | 0:1, 1:12, 2:87, 3:44, 4:1 |
| best k dets, all levels | 1.50 mHa (k=102) | 1.59 mHa (k=145) |
| best k dets, even levels only | 2.33 mHa | 23.8 mHa |
| best `max_dim` dets, all levels | 0.11 mHa (204) | 0.37 mHa (290) |
| best `max_dim` dets, even levels only | 1.66 mHa | 23.0 mHa |

**So two different causes:**

- **LiF: no singles.** The odd levels (12 singles, 44 triples among the top 145)
  are unreachable; the doubles-only floor is 23.0 mHa, and the pointer's refined
  result (23.16) sits on it. The policy did as well as doubles allow. Turning
  singles on only helps if their angle is sensible (0.1 is a placeholder).
- **N2: wrong angles.** The singles floor is only 1.66 mHa, far below the
  pointer's 5.85 (GQE-opt) and 2.34 (refined). The MP2 angles of the important
  doubles are off by up to 10x or have the wrong sign, which fits the pointer's
  small subspaces (median 37 determinants per circuit vs ~65 for the pool
  models).

Earlier pointer runs (`n2-pointer`, L=10, `allow_singles: true`) also
underperformed on N2, consistent with the angle explanation being the main one
there.

## Files

- Per-circuit results: `outputs/gqe-for-qsci/{n2,lif}L15-<family>-s<seed>/results.csv`
  (local copies; train.log and hydra/ next to them; checkpoints only on ABCI-Q).
- Job logs: `logs/packed_232062/`, `logs/packed_232063/`, `logs/packed_234046/`.
- Summary tables: `python hpc/plot_results.py --glob "outputs/gqe-for-qsci/n2L15-*" --summary`
  (and `lifL15-*`).
- Figures: `figures/L15_room_n2L15/`, `figures/L15_room_lifL15/` (error, final,
  subspace, sampled_basis, cost; png + pdf), made with
  `python hpc/plot_results.py --glob "outputs/gqe-for-qsci/n2L15-*" --out figures/L15_room_n2L15`.
- Predictions and outcomes: `predictions.md` (2026-10-13 and 2026-10-14).
