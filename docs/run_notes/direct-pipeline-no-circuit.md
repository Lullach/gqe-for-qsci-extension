# The direct ("no-circuit") pipeline — status, diagnosis, and what to do next

**Status: TABLED 2026-10-30**, pending discussion with chemists. The pipeline runs end
to end and is thoroughly diagnosed; it does not work in its current form, and the
reason is understood well enough to say what a working version would look like.

Everything below is measured, on N2 at 2.5 Å, sto-3g, 10e/8o (CI space 3136
determinants, CASCI −107.440409, CCSD −107.153222, HF −106.616959). Diagnostic
scripts are in `scratch/check_*.py` (gitignored); the numbers can be regenerated
from the epoch-49 checkpoint in `outputs/gqe-for-qsci/n2-direct-small-s1/`.

---

## 1. What the pipeline is

Remove the quantum circuit from GQE-QSCI: instead of generating a circuit,
simulating it and reading determinants off 10k shots, a GPT-2 policy
(`direct_qsci/`) emits determinants **directly** as sequences of spin-orbital
tokens. Everything downstream — dedup, `post_select_by_nelec`,
`enlarge(symmetry_completion)`, `diagonalize` — is the circuit pipeline's code
unchanged, so the two pipelines differ only in where determinants come from.

It is an **ablation / ceiling measurement, not a quantum method**: it asks whether
the circuit and its shot sampling are doing any work.

Config: `configs/experiment/n2_direct_small.yaml` (0.56M params, capacity-matched
to `n2L15-gpt2-small`'s 0.46M), `n_dets` auto-resolved to 102 from the measured
subspace requirement, `max_dim=204`.

## 2. The result

Stage-matched at epochs 40–49, same molecule, same `max_dim`, same batch:

| pipeline | best (mHa) | median (mHa) | subspace dim | distinct dets |
|---|---|---|---|---|
| direct-small, no circuit (1 seed) | 10.4 | 598.5 | 204 | 88.7 |
| circuit gpt2-small (5 seeds) | 5.9 (σ=0.4) | 19.8 (σ=13.5) | 70 | 30.6 |
| circuit gpt2-43M (5 seeds) | 5.5 (σ=0.5) | 9.1 (σ=1.2) | 80 | 32.2 |
| **uniform random selection** (10 trials) | 79.2 | **358.6** | 204 | — |

Circuit pipelines at their own end of training (epochs 90–99): gpt2-small 5.6/9.7,
gpt2-43M 4.3/6.6.

The direct pipeline's *typical* sample is **worse than picking determinants at
random**, while diagonalizing in a subspace 2.5–3× larger. Its best-ever sample
(10.39 mHa) is worse than the circuit pipeline's routine median.

### ⚠ The comparison is NOT information-matched

`operator_pool.ccsd_screening` is **True** for the circuit runs and **False** for
the direct pipeline. The circuit pipeline had a CCSD-screened pool of 117 excitations; the
direct pipeline had an unrestricted softmax over 16 spin-orbitals. Some unknown part
of the gap is this asymmetry, not architecture. Any future comparison must match
this flag.

## 3. What is wrong — and three explanations that are NOT the cause

Measured and **ruled out**:

- **Missing HF.** HF is rank 0 (largest coefficient) in the exact state, and the
  policy emits it only rarely. But *inserting* HF into the policy's sets changes
  the energy by 0.0–1.3 mHa (median 522.3 → 522.3). Samples that already contain
  HF still score 605 mHa.
- **A fragmented / decoupled subspace.** Every set, including random ones, forms
  a single connected component in the ≤2-excitation graph.
- **The mask forbidding good determinants.** The exact top-102, encoded as tokens,
  has **0 of 1020** mask-forbidden positions and round-trips exactly.

Measured and **confirmed**:

- **Selection quality is the whole deficit.** The 102-determinant budget is ample:
  the exact top-102 reaches 1.5 mHa raw, 0.68 mHa after symmetry completion.

  | | median \|c\| rank | mean excitation level | median energy |
  |---|---|---|---|
  | exact top-102 | — | 3.2 | 0.68 mHa |
  | uniform random | ~1638 | 4.0 | 358.6 mHa |
  | trained policy | ~1890 | 4.1 | 522–598 mHa |

  (uniform expectation for the rank is 1568, so the policy is slightly *biased
  against* important determinants.) It systematically occupies the **highest**
  allowed orbitals — emitted words look like 248 = `0b11111000` (orbitals 3–7)
  where HF is 31 = `0b00011111` (orbitals 0–4).

- **The `max_dim=204` cap binds on every single sample**, so subspace size carries
  no gradient signal and duplicates cost nothing.

- **Most of the subspace is inert.** For one sample, a 26-determinant core
  reproduces the full 204-dim energy *bitwise* (−107.430020667734); the other 178
  determinants contribute 0.00–0.03 mHa. On their own those 178 give 396–799 mHa.

- **The 10.388 mHa attractor is an artifact of the parameterization, not a
  discovery.** It recurs at 7.7% of samples (23/300) with **23 distinct emitted
  sets and 23 distinct subspaces**, and it also appears in a *freshly initialised*
  model trained purely supervised toward the exact set. It is not something the RL
  found.

- **The representation is fine; the reward signal is not.** Supervised NLL on the
  exact top-102 falls 1432 → 235 nats over 1500 steps with no plateau, and recall
  of the target determinants rises 3–8% → 24–33%. So nothing about the token
  space, masking or `log_prob` prevents a good set from being expressed.

### Why HF alone cannot help (the conceptual point)

The CI energy is a property of the **set**, not a sum over determinants. A new
determinant lowers the lowest eigenvalue only through its off-diagonal couplings
to what is already spanned, ΔE ≈ |⟨Ψ_S|H|D⟩|² / (H_DD − E_S), where Ψ_S is the
vector you *currently* have. H connects determinants differing by ≤2 excitations,
so HF reaches a subspace only through its singles and doubles:

```
exact top-102 excitation levels : {0:1, 1:2, 2:39, 3:12, 4:41, 5:4, 6:3}
policy sample levels            : {0:1,       2:7,  3:14, 4:28, 5:35, 6:9}
```

The exact set is a connected ladder anchored at HF with **39 doubles** carrying it
into the space. The policy's set has no singles, 7 doubles, and only **2–8 of ~90**
determinants within 2 excitations of HF. Correspondingly, HF matters enormously
*in its own set* — removing it from the exact top-102 costs 0.677 → 5.055 mHa, a
7.5× degradation that adding another determinant does not recover — and nothing at
all in the wrong company.

## 4. Pool reachability: what a CCSD-screened determinant generator can reach

Using the circuit pipeline's own pool (117 entries: 12 singles, 105 doubles)
**combinatorially** — a pool entry (i,j→a,b) means "move electrons i,j to a,b",
one determinant out, no angle and no shots — BFS from HF:

| steps | new | cumulative | % CI | top-102 covered | best-102 | all cumulative |
|---|---|---|---|---|---|---|
| 1 | 99 | 100 | 3% | 42/102 | 250.633 | 250.633 |
| 2 | 752 | 852 | 27% | 95/102 | 8.022 | 4.759 |
| 3 | 482 | 1334 | 43% | **102/102** | **1.502** | 0.007 |
| 4 | 4 | 1338 | 43% | 102/102 | 1.502 | 0.007 |
| 5 | 0 | — | — | saturated | | |

Key consequences:

- A **flat** singles-and-doubles determinant pool is hopeless: *all* 316
  determinants at excitation ≤2 give only 250.6 mHa. This is a hard ceiling, not a
  selection problem.
- **3-step compositions reach every one of the exact top-102**, so a
  CCSD-generated determinant pool is sufficient for chemical accuracy. 2-step
  chains cap out at 8.0 mHa (which, note, is about where the circuit pipeline's 9.1
  median sits).
- The reachable set saturates at 1338/3136 (43%); chains longer than 4 add
  nothing, so a single sequential walk yields at most 5 determinants.

## 5. Proposed design if this is resumed

Two changes, addressing credit assignment and candidate generation respectively.

**(a) CI-coefficient-weighted NLL** instead of GRPO. The diagnosis is that the
policy takes 1020 actions for one terminal scalar that is insensitive to ~80% of
them, and that `(mean − E)/std` manufactures an O(1) advantage even when a batch's
spread is meaningless. The eigenvector is already computed and discarded
(`energy, ci = self.diagonalize(...)` in `direct_qsci/pipeline.py`), and gives a
per-determinant weight for free:

    L = −(1/B) Σ_b Σ_k w_{b,k} · ℓ_{b,k},   w_{b,k} = |c_{D_k}|² − 1/K

where ℓ_{b,k} = Σ_{t∈block k} log p_θ(token_t | prefix) — i.e. reshape what
`log_prob` already returns, (B, K·T) → (B, K, T), and sum over T. The `−1/K`
baseline is what makes below-average determinants actively *suppressed* rather
than merely unreinforced; it also gives duplicates negative weight, which is the
pressure the `max_dim` cap currently removes.

Two implementation traps: the coefficients are paired *by position* with the
**enlarged** 204-determinant subspace, not the 102 emitted, so a lookup is needed;
and because symmetry completion means emitting D buys D's whole multiplet, the
honest credit for D is Σ over its group. That second one is a physics choice.

Known limitation: this makes the policy *remember* useful determinants, not
*discover* them. The weights come from the eigenvector within the subspace already
sampled, so the fixed point is self-consistent, not the exact top-k — today's core
sits at exact ranks 657–708 and would be faithfully reinforced. It needs pairing
with (b) or with exploration.

**(b) CCSD-screened determinants as the candidate generator.** Determinants are
built as **compositions of up to 3 pool excitations applied to HF**, matching the
circuit pipeline's information exactly. Generation:

1. Build the pool with `operator_pool.ccsd_screening=true` (117 entries for N2).
2. For determinant slot k, the policy emits 1–3 pool indices (plus a stop symbol),
   masked to moves valid on the determinant being built — sources occupied,
   targets empty.
3. D_k = those excitations applied to HF, in order. Pure bit manipulation.
4. The set is {HF} ∪ {D_1 … D_K}; HF is the empty chain, so it comes for free.
5. Downstream unchanged.

The policy then chooses *which excitation to apply* from a ~118-way vocabulary,
1–3 tokens per determinant, instead of spin-orbitals, 16-way, 10 tokens per
determinant. Sequence length drops from 1020 to ~200–300 tokens; since gradient
steps are 96% of runtime and attention is O(T²), this is the difference between
4.5 h and well under an hour per 100-epoch run.

The cost of (b) is that the pipeline is no longer "no physics input" — but the 250 mHa
flat-pool ceiling says the unaided version cannot work anyway, and the claim
becomes the sharper one: **is the shot-sampling step doing any work?**

### Open question for the chemists

Whether a CCSD-screened pool is a defensible determinant generator in the
strongly-correlated regime this method targets. CCSD's *energy* is 287 mHa wrong
at 2.5 Å, yet the circuit pipeline reaches 9.1 mHa using that same screen — evidence
that the amplitudes work as a candidate filter even where the energy does not.
Screening and predicting are different jobs, but that argument should be checked
by someone who does this for a living.

## 6. Infrastructure fixed along the way (independent of the pipeline's fate)

| what | where | why |
|---|---|---|
| `results.csv` rotation on a fresh run | `gqe_qsci/results_log.py`, `train_pipeline.py` | a restart appended epoch 0.. onto an existing file, silently merging two runs |
| `run_id` rotation on a fresh run | `train.py` | a restart resumed the *same* W&B run, interleaving two histories |
| **resume works at all** | `train.py`, `gqe_qsci/train_pipeline.py` | torch ≥2.6 `weights_only=True` refused our checkpoints; the standing workaround was `load_checkpoint=false`. Allowlists the 5 non-tensor globals (enumerated with `pickletools`, not guessed) and stops writing/reading `np.asarray(None)` as a 0-d object array |
| emitted determinants logged | `direct_qsci/pipeline.py` | `seq` was `None` for this pipeline, so no past run could be analysed — recovering one interesting energy needed the checkpoint plus 300 fresh samples |
| checkpoint interval 50 → 20 | `configs/experiment/n2_direct{,_small}.yaml` | an interruption cost 11 epochs |

Also noted: a long local run dies **silently** when the laptop suspends (the WSL2
VM goes down with it). Diagnose with the Kernel-Power event log, not `train.log`.
