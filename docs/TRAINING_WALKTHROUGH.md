# One training loop, end to end

Complete traces of a single epoch for the three policy families that differ in
how they *choose* a gate — **absorbing diffusion** (§1), the **DAG GNN** (§2),
and the **pointer action space** (§5) — with every tensor small enough to read.
Written up from the walkthroughs in the chats of 2026-09, checked against the
code as of commit `3b34efa`.

> Math renders on GitHub and in VS Code's markdown preview (`Ctrl+Shift+V`).

**The numbers are from a deliberately tiny toy configuration and are
illustrative, not measurements.** They show shapes, orderings and mechanisms;
energies and log-probs are plausible values, not the output of a real run. Real
defaults are given alongside in every table. Where a table *was* produced by
running the code, it says so.

Code: `gqe_qsci/train_pipeline.py`, `gqe_qsci/gqe/models/diffusion.py`,
`gqe_qsci/gqe/models/dag_gnn.py`, `gqe_qsci/gqe/models/pointer.py`,
`gqe_qsci/gqe/models/pointer_dag.py`, `gqe_qsci/gqe/loss.py`,
`gqe_qsci/gqe/buffer.py`.

The method being modified is the generative quantum eigensolver
([Nakaji et al., 2024](https://arxiv.org/abs/2401.09253)); the loop below is its
GPT-2 rollout/reward structure with the policy swapped out. Primary references
for every mechanism are collected in [§6](#6-references).

---

## 0. The shared skeleton

Every policy sits in the same loop. One **epoch** = one rollout group +
`step_per_epoch` gradient steps on that same group.

| stage | what happens | where |
|---|---|---|
| `on_fit_start` | warm-up rollouts until the buffer holds `warmup_size` samples; **no** gradient steps | `train_pipeline.py` |
| `on_train_epoch_start` | `collect_rollout()` — sample `num_samples` circuits, run QSCI, store `(seq, energy, old_log_prob, reveal_step)` | `collect_rollout` |
| `train_dataloader` | rebuilt every epoch (`reload_dataloaders_every_n_epochs=1`); `BufferDataset` indexes `idx % len(buffer)`, so **every batch is the same group** | `buffer.py` |
| `training_step` ×`step_per_epoch` | recompute `log_prob` **with gradients**, GRPO loss, backward, step | `training_step` |
| `on_train_epoch_end` | multi-molecule only: zero-shot eval, dissociation summaries | `_zeroshot_eval` |

Three invariants worth keeping in mind:

- `buffer_size == num_samples` is asserted, so the buffer holds exactly one
  rollout group. GRPO advantages are batch-relative, so mixing groups (or
  molecules) would corrupt them.
- `old_log_probs` are frozen at rollout time. Across the `step_per_epoch`
  updates the weights drift, the importance ratio $r = \exp(\log p_\text{new} -
  \log p_\text{old})$ fans out from 1, and the PPO clip starts to bite. That is
  its entire purpose.
- Only `num_samples` circuits (10 by default) are ever evaluated per epoch.
  Everything else in the loss exists to extract signal from that tiny sample
  without letting the policy over-commit to it.

---

## 1. Absorbing diffusion (`model=diffusion_absorbing`)

The forward and reverse processes are D3PM's absorbing-state variant
([Austin et al., 2021](https://arxiv.org/abs/2107.03006)): per-token masking with
$\alpha_0 = 1$, $\alpha_T = 0$, and the closed-form posterior
$q(x_{t-1} \mid x_t, \hat{x}_0)$. The masked-diffusion simplification this
implementation follows — one categorical per position, and only masked positions
contribute — is the MDLM line
([Sahoo et al., 2024](https://arxiv.org/abs/2406.07524); concurrently
[Shi et al., 2024](https://arxiv.org/abs/2406.04329)). The *objective* is not
theirs: §1.3 replaces the masked-diffusion ELBO with an exact trajectory
log-probability.

### Setup

| | toy | real default |
|---|---|---|
| $B$ = `num_samples` | 4 | 10 |
| $L$ = `ngates` | 4 | 10 |
| $T$ = `diffusion_steps` | 4 | 8 (16 for `diffusion_gnn_absorbing`) |
| $V$ (vocab) | 6 | ~118 (N2, deduped) |
| `mask_token` | 6 ($= V$) | 118 |

Cosine schedule from `_make_alpha_schedule`:

| $t$ | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| $\alpha_t$ | 1.000 | 0.854 | 0.500 | 0.146 | 0.000 |

$\alpha_0 = 1$ (clean), $\alpha_T = 0$ (fully masked).

### 1.1 Rollout: `sample_sequence`, the reverse process

Start fully masked at $t = T$ and reveal downward. The fraction of *currently
masked* positions revealed at step $t$ is

$$p_\text{reveal}(t) = \frac{(1-\alpha_t)-(1-\alpha_{t-1})}{1-\alpha_t} = \frac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}$$

— not a tuned hyperparameter, but forced by the schedule: the expected masked
fraction must go from $1 - \alpha_t$ to $1 - \alpha_{t-1}$.

| step $t$ | $p_\text{reveal}$ | tokens before | tokens after |
|---|---|---|---|
| 4 | 0.146 | `[M,M,M,M]` | `[M,3,M,M]` |
| 3 | 0.414 | `[M,3,M,M]` | `[5,3,M,1]` |
| 2 | 0.707 | `[5,3,M,1]` | `[5,3,2,1]` |
| 1 | 1.000 | `[5,3,2,1]` | `is_masked.any()` False → `break` |

**There are two independent draws per step**, and conflating them is what makes
this opaque:

```python
x0_pred = Categorical(logits=-β*logits).sample()   # DRAW 1: what goes where
reveal  = bernoulli(p_reveal).bool() & is_masked   # DRAW 2: which slots commit
tokens  = torch.where(reveal, x0_pred, tokens)
```

- **Draw 1** is a complete guess at the whole circuit: the model predicts a
  clean token $\hat{x}_0$ for *all* $L$ positions, including already-revealed
  ones. It is sampled, not argmaxed.
- **Draw 2** has nothing to do with the model — pure scheduling, $L$ coins with
  the same probability, then `& is_masked`.
- The `where` commits only the intersection; **every other prediction is thrown
  away** and re-made later with more context. That is the point: at $t = 4$ the
  model guesses a position knowing nothing; by $t = 1$ it re-guesses knowing the
  other three. Steps can be complete no-ops (all coins zero), which is normal.
- Revealed tokens are frozen — `where` only overwrites positions still `M`.
- $p_\text{reveal}$ rises monotonically ($0.15 \to 0.41 \to 0.71 \to 1.00$) and
  is exactly 1 at $t = 1$, which is what guarantees a complete sequence. The
  `still_masked` safety net after the loop covers only a float-precision
  Bernoulli miss; it *samples* (so the token has a well-defined log-prob) and
  leaves `reveal_step` at 0.

Result:

```
state["idx"] = [[0, 5, 3, 2, 1],     # (B, 1+L); column 0 is BOS = pool[0] = identity
                [0, 2, 4, 1, 3],
                [0, 5, 1, 2, 4],
                [0, 3, 3, 0, 2]]

state["reveal_step"] = [[3, 4, 2, 3],   # per position: the step that committed it
                        ...]
```

### 1.2 QSCI → energies

`qsci_pipeline.process(state)` compiles each row into Pauli-evolution gates,
simulates with CUDA-Q at `shots`, extracts determinants and diagonalizes with
PyCI:

```
energies = [-107.20, -107.44, -107.31, -107.15]   # Ha; circuit 1 is best
```

`scheduler.update(energies=...)` then nudges $\beta$ from the spread.

### 1.3 `log_prob` → `old_log_probs` (exact trajectory, DDPO)

This is the DDPO formulation — score the denoising trajectory as the action
sequence of a policy ([Black et al., 2023](https://arxiv.org/abs/2305.13301); cf.
DPOK, [Fan et al., 2023](https://arxiv.org/abs/2305.16381)). Specifically the
**importance-sampled variant, $\text{DDPO}_\text{IS}$** (their §4.3), not
$\text{DDPO}_\text{SF}$: the score-function estimator "only allows for one step
of optimization per round of data collection," and we take `step_per_epoch` = 30
steps on one frozen rollout group. See §3 for where our version departs from
theirs.

The trajectory is $\tau = (x_T, x_{T-1}, \ldots, x_0)$, the sequence of
partially-masked states the reverse process passed through. Write $C_t$ for the
set of positions **committed at step** $t$ and $m_t$ for the number still masked
entering that step. One step factorizes as

$$p(x_{t-1} \mid x_t) = \underbrace{p_\text{reveal}(t)^{|C_t|}\big(1-p_\text{reveal}(t)\big)^{m_t - |C_t|}}_{\text{reveal coins — no } \theta} \cdot \prod_{i \in C_t} p_\theta\big(x_0[i] \mid x_t, t\big)$$

The product over positions is legitimate because the logits are computed per
position from $x_t$, so the draws are conditionally independent given $x_t$; and
the proposals at *non*-committed positions marginalize to 1, since a discarded
draw never influences any later state. Multiplying over $t$:

$$\boxed{\ \log p(\tau) = \log q(\tau) + \sum_t \ \sum_{i \in C_t} \log p_\theta\big(x_0[i] \mid x_t, t\big)\ }$$

where $q(\tau)$ collects every coin factor and contains no $\theta$. It
therefore cancels in the GRPO ratio
$\exp(\log p_\text{new} - \log p_\text{old})$, which is precisely why `log_prob`
computes only the sum and drops $\log q$. So the sum is **the
$\theta$-dependent part** of the trajectory log-probability, not
$\log p(\tau)$ itself.

$x_t$ — the state the network saw at step $t$ — is reconstructed from the
recorded trajectory by one rule: **position $i$ is visible at step $t$ iff
$\texttt{reveal\_step}[i] > t$** (it was committed at some later, i.e. larger,
step; everything else is still `[MASK]`). Each position belongs to exactly one
$C_t$, so it is scored exactly once, and the code returns the **per-position
decomposition** $(B, L)$ rather than the scalar above — `GRPOLoss` uses it
elementwise.

**What one factor in the product actually is.** The network does a single
forward pass over the whole sequence and returns a $(L, V)$ matrix — for *every*
position, a distribution over all $V$ operators:

$$P^{(t)}[i, v] = \mathrm{softmax}_v\big(-\beta \cdot f_\theta(x_t, t)[i]\big)_v$$

and $p_\theta(x_0[i] \mid x_t, t)$ is one **entry** of it: row $i$, column
$x_0[i]$ — the probability the model assigned, at step $t$, to the gate that
position $i$ ended up with. ($x_0[i]$ is a *value*, the final token at that
position; the column index is what the notation hides.) The conditioning is on
the whole partially-masked $x_t$, because the denoiser is bidirectional, so
position $i$'s row depends on every currently-visible token — and on $t$, which
enters through the time embedding.

At $t = 3$ in the run below, the network sees $x_3 =$ `[M,3,M,M]` and returns:

| position | op0 | op1 | op2 | op3 | op4 | op5 | |
|---|---|---|---|---|---|---|---|
| 0 | 0.20 | 0.18 | 0.15 | 0.20 | 0.135 | **0.135** | commits → keeps op5 |
| 1 | 0.05 | 0.10 | 0.10 | 0.55 | 0.10 | 0.10 | already visible → discarded |
| 2 | 0.15 | 0.15 | 0.25 | 0.15 | 0.15 | 0.15 | masked, coin failed → discarded |
| 3 | 0.10 | **0.407** | 0.10 | 0.143 | 0.15 | 0.10 | commits → keeps op1 |

$C_3 = \{0, 3\}$, so the product has exactly two factors,
$0.135 \times 0.407$, i.e. $\log$ contributions $-2.0$ and $-0.9$. The coin half
is the other question entirely: $m_3 = 3$ masked entering the step and 2
committed give $0.414^2 \cdot 0.586 = 0.100$. **Which** slots opened is the
coins' business; **what** went into them is the product's.

The same distribution is used for sampling (`Categorical(logits=-β*logits)`) and
for scoring (`log_softmax(-β*logits)` then `gather`), which is what makes replay
exact rather than approximate.

Collecting all three steps for circuit 1, with `gate_tokens = [5,3,2,1]` and
`reveal_step = [3,4,2,3]`:

| $t$ | $x_t$ | $C_t$ | factors used | coins |
|---|---|---|---|---|
| 4 | `[M,M,M,M]` | $\{1\}$ | $P^{(4)}[1,3] = 0.165$ | $0.146^1 \cdot 0.854^3 = 0.091$ |
| 3 | `[M,3,M,M]` | $\{0,3\}$ | $P^{(3)}[0,5] = 0.135$, $P^{(3)}[3,1] = 0.407$ | $0.414^2 \cdot 0.586^1 = 0.100$ |
| 2 | `[5,3,M,1]` | $\{2\}$ | $P^{(2)}[2,2] = 0.223$ | $0.707^1 \cdot 0.293^0 = 0.707$ |

Note position 0 gets a row at every step, but only the $t = 3$ one is ever used:
at $t = 4$ the model guessed it knowing nothing, and that guess was discarded
when the coin came up 0. Same for the same position's *re-prediction* — the
model re-guesses every position every step, and all but the committed ones are
thrown away, which is the compute cost §1.1 describes.

Laid out per position, which is the shape the code returns:

| $t$ | positions committed | $x_t$ fed to `_logits` | contribution |
|---|---|---|---|
| 4 | 1 | `[M,M,M,M]` | `[ 0,-1.8, 0, 0]` |
| 3 | 0, 3 | `[M,3,M,M]` | `[-2.0, 0, 0,-0.9]` |
| 2 | 2 | `[5,3,M,1]` | `[ 0, 0,-1.5, 0]` |
| | **sum** | | `[-2.0,-1.8,-1.5,-0.9]` |

Why the trajectory and not the ELBO: GRPO needs the probability of the action
sequence actually *taken*, which is what the derivation above gives. The ELBO
bounds $\log p(x_0)$ — reconstructability — which is not the sampler's
distribution, and it needed frozen corruption masks in the buffer to keep the
ratio deterministic. See
`NOTES.md`, "Trajectory log-prob (DDPO) replaces the ELBO"; the ELBO code is in
git history (`sample_masks()`, `_corrupt()`).

Two honest caveats:
- The loop runs over the union of commit steps **across the batch**, so with
  $B = 10$ nearly every step is usually populated. Cost is "never worse than
  $T$", not a speedup.
- The ELBO scored $\sim L T / 2$ terms against the trajectory's $L$: denser
  gradient, but biased. If the trajectory version underperforms in an A/B, that
  is the likely reason rather than a bug.

### 1.4 Buffer

Four tuples `(seq, energy, old_log_prob, reveal_step)`. `reveal_step` must ride
along, or replay would score a different trajectory than the one sampled.

---

## 2. DAG GNN (`model=dag_gnn`, `dag_gnn_features`)

### Setup

| | toy | real default |
|---|---|---|
| `num_samples` = `batch_size` = `buffer_size` = `warmup_size` | 4 | 10 |
| $L$ = `ngates` | 4 | 10 |
| `n_qubits` | 6 | 16 (N2) |
| $V$ | 6 | ~118 |
| `step_per_epoch` | 3 | 30 |

Footprints (`_fp_flat`, from `pool.get_qubit_footprints()`): op0 `[]` (identity),
op1 `[0,1,2,3]`, op2 `[2,3,4,5]`, op3 `[0,1,4,5]`, op4 `[0,2]`, op5 `[1,3]`.

With `feature_scorer: true` there is **no** `nn.Embedding` over gates —
`_node_table()` is composed fresh on every forward pass:

```
rows 0..5   qubit wires  <- qubit_encoder(orbital_features)   (n_qubits, H)
rows 6..11  gate slots   <- operator_head.keys()              (V, H)  weight-tied to the output
row  12     UNPLACED     <- a learned vector                  (1, H)
```

### 2.1 `_init_dag_state`

```
node_tokens = [0,1,2,3,4,5, 12,12,12,12]   # n_qubits wires + L empty slots (12 = UNPLACED)
frontier    = [0,1,2,3,4,5]                # each qubit points at its own input node
edge_srcs/dsts = [] []
```

### 2.2 `sample_sequence` — $L$ steps

Each step: `_step_forward` → `_scaled_logits` → sample → `_advance_dag`.

**Step 0** (`gate_node = 6`). `edge_index` is empty, so the GAT layers are
**skipped entirely** and `pooled` is just the mean of the raw wire embeddings.
The first gate is chosen with no structural information, only orbital physics.
Sample → **op 3**. `_advance_dag` sets `node_tokens[6] = 6+3 = 9` and, for each
$q \in \{0,1,4,5\}$, adds edge `frontier[q]→6` and sets `frontier[q] = 6`:

```
edges    (0->6)(1->6)(4->6)(5->6)
frontier [6,6,2,3,6,6]
```

**Step 1** (`gate_node = 7`). Two subtleties bite:

- `frontier` gathers node 6 **four times**, so it dominates the mean pooling —
  wide gates get implicitly upweighted.
- Sample → **op 1**, footprint `[0,1,2,3]`; qubits 0 and 1 share frontier 6, so
  edge `6→7` is appended **twice**. Parallel edges are kept, so the GAT
  aggregates that neighbour twice.

```
edges  += (6->7)(6->7)(2->7)(3->7)
frontier [7,7,7,7,6,6]
```

**Step 2** → op 5, fp `[1,3]`: `edges += (7->8)(7->8)`, `frontier [7,8,7,8,6,6]`.
**Step 3** → op 2, fp `[2,3,4,5]`: `edges += (7->9)(8->9)(6->9)(6->9)`,
`frontier [7,8,9,9,9,9]`.

Final: `node_tokens = [0,1,2,3,4,5, 9,7,11,8]`, sequence `[3,1,5,2]`, so
`state["idx"] = [0,3,1,5,2]`.

### 2.3 Canonical masking (`canonical_masking: true`)

Operator $k$ is forbidden at step $t$ iff

$$\exists\, i < t \ :\ \big[k,\ \texttt{prefix}[m]\big] = 0 \ \ \forall m \in [i, t-1] \quad\text{and}\quad k < \texttt{prefix}[i]$$

This keeps one representative per Mazurkiewicz **trace class** — the equivalence
classes of a free partially commutative monoid, where commuting gates are only
spellable in their sorted order ([Mazurkiewicz,
1987](https://doi.org/10.1007/3-540-17906-2_30); Diekert & Rozenberg, *The Book
of Traces*, 1995).

| step | prefix | forbidden | allowed | sampled |
|---|---|---|---|---|
| 0 | — | — | `{0,1,2,3,4,5}` | 3 |
| 1 | `[3]` | `{0,2}` | `{1,3,4,5}` | 1 |
| 2 | `[3,1]` | `{0,2}` | `{1,3,4,5}` | 5 |
| 3 | `[3,1,5]` | `{0,1,3,4}` | `{2,5}` | 2 |

Two traps, both load-bearing:
- The $-\infty$ goes on the **scaled** logits, *after* the $-\beta$ multiply.
  Masking first would flip the sign to $+\infty$ and make forbidden ops
  maximally likely.
- Entropy uses an **$-\infty$-safe path**: $0 \cdot (-\infty) = \mathrm{NaN}$,
  and `log_softmax`'s backward spreads that NaN to every logit. Replace
  $-\infty$ with 0 *before* the multiply.

Measured effect of this masking on N2: none, at current seeds. Recorded as
inconclusive, not negative — see `NOTES.md`.

### 2.4 `log_prob` → `old_log_probs`

Replays the identical DAG construction with the stored sequence, reading
$p(a_t \mid \mathrm{DAG}_{t-1})$ off each step. `state.get("reveal_step")` is
`None` (only the absorbing models set it), so `lp_kwargs = {}`. Exact and
deterministic — call it twice with unchanged weights and you get identical bits.

```
circuit 1: [-1.10, -1.35, -0.95, -1.28]
```

---

## 3. The gradient steps (all models)

`BufferDataset(buffer, repetition=step_per_epoch)`, `shuffle=False`,
`__getitem__` does `idx % len(buffer)` → **every batch is the same 4 circuits.**

Advantages are recomputed each step but identical, since energies are fixed:

$$A_b = \frac{\bar{E} - E_b}{\mathrm{std}(E) + \varepsilon}$$

```
mean = -107.2750,  std = 0.1287
advantages = [-0.5827, +1.2819, +0.2719, -0.9712]
```

Circuit 1 (lowest energy) gets the largest **positive** advantage — lower energy
⇒ push probability up.

`GRPOLoss` has **two** terms. The second is GRPO proper — group-relative
advantages with a clipped PPO objective and no value network
([Shao et al., 2024](https://arxiv.org/abs/2402.03300), DeepSeekMath). The first
is an addition with **no known reference**: it is inherited from the original
NVIDIA CUDA-QX GQE code and is not part of GRPO as published. Worth knowing
before citing this loss as "GRPO" in writing.

$$\mathcal{L} = \underbrace{-\,\overline{\log p_\theta(\tau_{b^\star})}}_{\text{NLL on the winner } b^\star = \arg\min_b E_b} \;-\; \underbrace{\min\Big(\overline{r A},\ \overline{\mathrm{clip}(r,\,1-0.2,\,1+0.28)\,A}\Big)}_{\text{clipped PPO objective}} \;-\; c_H \overline{H}$$

with $r = \exp(\log p_\text{new} - \log p_\text{old})$ elementwise and
$c_H =$ `entropy_coeff` (0.01 by default; 0 disables the bonus and the second
`log_prob` output entirely).

Three steps of the toy DAG run:

| | $r$ range | term 1 (NLL on winner) | $\min(\text{clipped}, \text{unclipped})$ | loss |
|---|---|---|---|---|
| step 1 | `[1.0000, 1.0000]` | 1.1700 | −0.000015 | 1.1700 |
| step 2 | `[0.9713, 1.0392]` | 1.1315 | 0.022603 | 1.1089 |
| step 3 | `[0.9343, 1.0939]` | 1.0803 | 0.053159 | 1.0271 |

Read the $r$ column downward — that is the whole dynamic. The weights have not
moved yet at step 1, so the ratios start at 1 and the PPO term contributes
nothing; as the weights move they fan out, and over 30 real steps they reach the
clip bounds, which is what stops one rollout group from moving the policy too
far.

> **One wrinkle the toy table hides.** It was produced with $\beta$ held fixed,
> so step 1 gives exactly $1.0000$. A real run does not: `collect_rollout` scores
> `old_log_probs` at the current $\beta$ and *then* calls
> `scheduler.update()` ([train_pipeline.py:468](../gqe_qsci/train_pipeline.py)),
> so all 30 gradient steps use $\beta + \delta$ ($\delta = 0.02$ by default).
> Since $\partial \log p_i / \partial \beta = \mathbb{E}_p[z] - z_i$ and a
> *sampled* token usually sits below the mean pseudo-energy, this offsets step 1
> slightly and systematically **above** 1 — annealing, not learning, consuming a
> sliver of the clip budget at the start of every epoch. It is still correct
> importance sampling (the behavior policy really was the old $\beta$), just not
> the clean $r = 1$ the table suggests. Unquantified in a real run; worth a look
> at the logged ratio spread if the clip ever appears to bite early.

**How this differs from $\text{DDPO}_\text{IS}$ as published**, worth stating
before citing it:

| | Black et al. §4.3 | here |
|---|---|---|
| ratio granularity | one per denoising timestep, over the whole transition $p_\theta(x_{t-1} \mid x_t)$ | one per **position** — `log_prob` returns $(B, L)$ and the ratio is elementwise |
| reward weighting | $r(x_0, c)$ directly | group-relative advantage $(\bar{E} - E)/\mathrm{std}$ (GRPO) |
| extra term | — | NLL on the winner (unattributable, above) |

The granularity difference is not cosmetic. A step committing $k$ gates yields
$k$ separately-clipped ratios instead of one ratio of their product, so each
token is held inside $[0.8, 1.28]$ while the joint step may move by up to
$1.28^{\,k}$. That is the GRPO/LLM per-token convention rather than DDPO's
per-step one; both are defensible, but the trust region is not the same object.

One caveat carried over from the ELBO switch: the trajectory log-prob sums over
$L$ instead of averaging over $T$, so its **scale differs from the old ELBO
runs**. The clip range is scale-sensitive; watch it when comparing to
pre-`5786019` results. §5.5 is the same problem in a different guise.

---

## 4. Per-epoch cost

| | absorbing diffusion | DAG GNN |
|---|---|---|
| `sample_sequence` | $\leq T$ forwards (8–16) | $L$ forwards (10) |
| `log_prob` at rollout | $\leq T$ | $L$ |
| `log_prob` × `step_per_epoch` | $30 T$ (240–480) | $30 L$ (300) |

At $L = 10$ none of this matters: a single epoch's wall-clock is dominated by
`num_samples` CUDA-Q simulations plus the QSCI diagonalizations, which are
CPU-bound and not accelerated by the GPU.

---

## 5. Pointer action space (`model=pointer_dag`)

Same generative story as §2 — build the circuit gate by gate, GNN over the
partial DAG, pool the frontier — with one substitution: the pooled frontier is a
pointer **query**, not logits over an operator menu. Read §2 first; only the
differences are spelled out here.

> **Review pending.** The `PointerActionSpace` extraction this section describes
> (2026-09-28) was written by Claude. It is verified bitwise-equivalent on a stub
> pool and gated by `hpc/smoke_pointer_dag.py` (43/43), but has not been read
> through by a human and has not run on a cluster. See `NOTES.md`,
> "PointerActionSpace refactor".

### What disappears relative to §2

| §2 needs | §5 |
|---|---|
| $(V, \texttt{feat\_dim})$ operator menu + `OperatorScorer` head | gone — the head scores **orbitals** |
| $(V, \texttt{n\_qubits})$ footprint table | gone — a footprint comes from the excitation itself |
| $(V, V)$ commutation matrix | gone — 13,924 entries for N2, 5.3 GB at 30 orbitals |
| canonical masking (§2.3) | gone — $j > i$ and $b > a$ make each excitation unspellable more than one way |
| a CCSD-screened pool as *input* | the pool becomes an append-only *output* cache (§5.3) |

### Setup

| | toy | real default |
|---|---|---|
| $B$ = `num_samples` | 4 | 10 |
| $L$ = `ngates` | 4 | 10 |
| $n$ (spin-orbitals) | 6 | 16 (N2) |
| candidates scored per gate | $4(n+1) = 28$ | $4 \cdot 17 = 68$ |
| excitations reachable | 8 | 315 (vs the 117 CCSD kept) |
| `hidden_size` / `num_layers` | 64 / 2 | 128 / 6 (matched to `dag_gnn_features`) |
| `encoder_layers` / `encoder_heads` | 2 / 4 | 2 / 4 |

Toy molecule: 6 spin-orbitals, interleaved Jordan–Wigner (even = $\alpha$),
occupied $\{0, 1\}$, virtual $\{2,3,4,5\}$. That admits 4 singles and 4 doubles
— small enough to check the masks by hand.

### 5.1 One gate = four pointers

$$i\ (\text{occ}) \longrightarrow j\ (\text{occ},\, j > i)\ \text{or STOP} \longrightarrow a\ (\text{virt}) \longrightarrow b\ (\text{virt},\, b > a)\ \text{or NONE}$$

Index $n$ (= 6 here) means STOP/NONE, so every step has the same candidate axis
of size $n+1$, and a single excitation is spelled $(i, n, a, n)$. Tracing one
gate:

| step | pick | allowed candidates | why |
|---|---|---|---|
| 0 | $i = 0$ | `{0, 1}` | occupied, and able to start a single or a double |
| 1 | $j = 1$ | `{1, STOP}` | occupied with $j > i$; STOP would make it a single |
| 2 | $a = 3$ | `{2, 3, 4}` | virtual, and a partner $b$ of the required spin still exists above it |
| 3 | $b = 4$ | `{4}` | virtual, $b > a$, spin forced by $S_z$: $n_\beta(\{0,1\}) - \mathrm{spin}(3) = 0 \Rightarrow \alpha$ |

Gate $= (0, 1, 3, 4)$, footprint `[0, 1, 3, 4]`.

*(This table is machine-checked: the allowed sets are what
`ExcitationRules.step_mask` returns for this orbital table, not a hand
derivation. Step 2 for a single, $(i{=}0, j{=}\text{STOP})$, gives `{2, 4}` —
same-spin virtuals only.)*

Three things it makes visible:

1. **Legality is enforced by masking, never by rejection.** A rejected sample
   would leave the log-probability inconsistent with the sampler, and GRPO's
   ratio would then be computed against the wrong distribution.
   `ExcitationRules.step_mask` is closed-form and $O(n)$; nothing enumerates
   excitations.
2. **$S_z$ propagates backward into step 2.** $a$ is only offered when a partner
   of the required spin still exists at a higher index, so step 3 can never be
   handed an empty mask. That is what the `has_virt_after` suffix sums are for.
3. **A step can collapse to one candidate** (step 3 above), contributing exactly
   $\log p = 0$. Normal, not a bug.

Same two sign/NaN traps as §2.3, for the same reasons: the $-\infty$ goes on the
scaled logits after the $-\beta$ multiply, and the entropy replaces $-\infty$
before any arithmetic touches it.

### 5.2 The rollout

$\texttt{orb\_keys} = \texttt{space.keys()}$ once per forward — a graph
transformer over the $n$ orbitals, with
$[\Delta\varepsilon,\ \text{exchange},\ \text{same\_spin},\ \text{occ\_to\_virt}]$
entering as an additive attention bias. The exchange term is
$\lvert (pq \mid pq) \rvert$ read straight from `cas_hamiltonian.h2`, so it costs
nothing QSCI was not already paying.

Then per gate step:

```python
node_embs = _node_embeddings(orb_keys, gate_embs, B)   # wires ARE orbitals: keys used directly
query     = _pool_frontier(...)                        # (B, H) — identical to §2.2
picks, logp = space.decode(query, orb_keys, beta)      # four masked pointer steps
gate_embs.append(space.embed(picks, orb_keys))         # mean of the keys it touches
_advance(picks, n_qubits + step, ...)                  # footprint from space.footprint()
```

Two differences from §2.2 worth noting: there is no separate `qubit_encoder`,
because a wire node *is* an orbital; and `gate_embedding` is deliberately
permutation-invariant over the four orbitals, since occupied-vs-virtual is
already carried inside each key by the occupancy feature.

Since `decode` takes any $(N, H)$ query, $N$ is whatever the host batches over:
$B$ for one gate at a time, or $B \cdot L$ if every position decodes at once —
which is what a diffusion denoiser would need. See `NOTES.md` for the GPT-2 and
diffusion port sketches.

### 5.3 The index round trip — where the pool grows

Everything downstream is index-keyed: `sampler.py` does `[pool[j] for j in row]`
and the buffer stores `state["idx"]`. So the policy still emits integers.
`space.to_indices(picks)` calls `pool.ensure_excitation(key, pairs)`, which on
first sight computes the rotation angle from MP2,

$$\theta = 2 t_{ijab}, \qquad t_{ijab} = \frac{\langle ij \lVert ab \rangle}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}$$

builds the cudaq operator, appends it, and records the index ↔ key map both
ways. Singles get $t = 0$ exactly (Brillouin) and fall back to the
`single_angle` placeholder — see `NOTES.md`, "Pointer action space: open
decisions".

The pool growing is a compatibility layer, not the goal. With the default
`operator_pool.ccsd_screening: true` it starts as the CCSD-screened list the
factory built anyway and accumulates on top (measured: N2 118 → 130 operators
from 12 sampled gates). With `ccsd_screening: false` it starts as identity-only
and holds nothing but what the policy built — which is what makes "no CCSD in
the loop" true end to end.

The consequence is the one operational rule for these runs: **an index means
something only within one run**, because a new run fills the cache in whatever
order it samples. `space.to_picks()` raises rather than mis-decode a stale
index (gated by `hpc/smoke_pointer_dag.py`), but run with
`trainer.load_checkpoint=false` and a fresh `exp_tag` regardless.

### 5.4 `log_prob`

Exact, and already a trajectory log-probability — the gate sequence *is* the
trajectory, so `reveal_step` is accepted and ignored, as in §2.4:

$$\log p_\theta(\text{gate}_1 \ldots \text{gate}_L) = \sum_{t=1}^{L} \ \sum_{s=0}^{3} \log p_\theta\big(\text{pointer step } s \mid \text{gate}_t\big)$$

Replay is the same `_rollout` with `forced=space.to_picks(gate_tokens)`, so
sampling and scoring cannot drift apart. Per gate, e.g.
$[-1.9, -0.7, -1.1, -0.0] \to -3.7$; `space.decode` returns the sum, and
entropy is likewise summed over the four steps.

### 5.5 One scale caveat before comparing against §2

A per-gate log-prob here is a sum of **four** categorical terms where §2's is
**one**. The clip range $[0.8,\ 1.28]$ is scale-sensitive, so a pointer run and
a pool run are not automatically on the same footing — the same lesson as the
ELBO → trajectory switch in §1.3. Watch `trainer/loss` and the ratio spread when
reading `n2_pointer` against `n2_pool_matched`.

### 5.6 Per-epoch cost

| | forwards |
|---|---|
| `space.keys()` | 1 per rollout (**not** per gate) |
| `sample_sequence` | $L$ GNN passes + $4L$ pointer steps |
| `log_prob` at rollout | same |
| `log_prob` × `step_per_epoch` | $30 \times$ the above |

The pointer steps are einsums over $n + 1 \approx 17$ keys, so they are noise
next to the GNN passes — and all of it is noise next to the CUDA-Q simulations
and QSCI diagonalizations, as in §4. Structural shape only; there are no
measured timings here until the first cluster run.

---

## 6. References

Every arXiv identifier below was checked against the listing page, not recalled.

**The method.**
- K. Nakaji, L. B. Kristensen, R. Kemmoku, *et al.*, "The generative quantum
  eigensolver (GQE) and its application for ground state search,"
  [arXiv:2401.09253](https://arxiv.org/abs/2401.09253) (2024). The method this
  repository modifies; `gqe_qsci/gqe/models/gpt2.py` derives from its CUDA-QX
  implementation.

**Discrete diffusion (§1).**
- J. Austin, D. D. Johnson, J. Ho, D. Tarlow, R. van den Berg, "Structured
  Denoising Diffusion Models in Discrete State-Spaces,"
  [arXiv:2107.03006](https://arxiv.org/abs/2107.03006) (2021). D3PM; the
  absorbing-state process of §1.1 and the schedule of `_make_alpha_schedule`.
- S. S. Sahoo, M. Arriola, Y. Schiff, *et al.*, "Simple and Effective Masked
  Diffusion Language Models,"
  [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) (NeurIPS 2024). MDLM —
  the masked-diffusion parameterization in which only masked positions are
  scored.
- J. Shi, K. Han, Z. Wang, A. Doucet, M. K. Titsias, "Simplified and Generalized
  Masked Diffusion for Discrete Data,"
  [arXiv:2406.04329](https://arxiv.org/abs/2406.04329) (2024). Concurrent with
  MDLM and largely equivalent for our purposes.

**The objective (§1.3, §3).**
- K. Black, M. Janner, Y. Du, I. Kostrikov, S. Levine, "Training Diffusion Models
  with Reinforcement Learning,"
  [arXiv:2305.13301](https://arxiv.org/abs/2305.13301) (2023). DDPO — scoring the
  denoising trajectory as a policy's action sequence. §1.3 implements the
  importance-sampled estimator $\text{DDPO}_\text{IS}$ of their §4.3 (the one
  that permits several optimization steps per round of data collection), with
  the two deviations tabulated in §3, not the score-function
  $\text{DDPO}_\text{SF}$.
- Y. Fan, O. Watkins, Y. Du, *et al.*, "DPOK: Reinforcement Learning for
  Fine-tuning Text-to-Image Diffusion Models,"
  [arXiv:2305.16381](https://arxiv.org/abs/2305.16381) (NeurIPS 2023). The other
  early RL-on-diffusion formulation; cf. DDPO.
- Z. Shao, P. Wang, Q. Zhu, *et al.*, "DeepSeekMath: Pushing the Limits of
  Mathematical Reasoning in Open Language Models,"
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) (2024). GRPO.

*Not attributable:* the NLL-on-the-winner term in §3 has no reference. It comes
from the CUDA-QX GQE code and is not part of GRPO as published.

**Commutation / canonical forms (§2.3).**
- A. Mazurkiewicz, "Trace Theory," in *Petri Nets: Applications and Relationships
  to Other Models of Concurrency*, LNCS 255, pp. 279–324
  ([Springer, 1987](https://doi.org/10.1007/3-540-17906-2_30)).
- V. Diekert, G. Rozenberg (eds.), *The Book of Traces*, World Scientific (1995).
  Standard reference for free partially commutative monoids and their normal
  forms.

**Value proposition / baselines** (not part of this walkthrough; see `NOTES.md`,
"Value proposition: fair classical baseline"): Lee et al., *Nat. Commun.* **14**,
1952 (2023), on the contested status of ground-state quantum advantage.
