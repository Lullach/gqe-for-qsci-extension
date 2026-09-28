# One training loop, end to end

Two complete traces of a single epoch — **absorbing diffusion** and the **DAG
GNN** — with every tensor small enough to read. Written up from the walkthroughs
in the chat of 2026-09, checked against the code as of commit `40e2df8`.

**The numbers are from a deliberately tiny toy configuration and are
illustrative, not measurements.** They show shapes, orderings and mechanisms;
energies and log-probs are plausible values, not the output of a real run. Real
defaults are given alongside in every table.

Code: `gqe_qsci/train_pipeline.py`, `gqe_qsci/gqe/models/diffusion.py`,
`gqe_qsci/gqe/models/dag_gnn.py`, `gqe_qsci/gqe/loss.py`,
`gqe_qsci/gqe/buffer.py`.

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
  updates the weights drift, the importance ratio `exp(new − old)` fans out
  from 1, and the PPO clip starts to bite. That is its entire purpose.
- Only `num_samples` circuits (10 by default) are ever evaluated per epoch.
  Everything else in the loss exists to extract signal from that tiny sample
  without letting the policy over-commit to it.

---

## 1. Absorbing diffusion (`model=diffusion_absorbing`)

### Setup

| | toy | real default |
|---|---|---|
| `B` = `num_samples` | 4 | 10 |
| `L` = `ngates` | 4 | 10 |
| `T` = `diffusion_steps` | 4 | 8 (16 for `diffusion_gnn_absorbing`) |
| `V` (vocab) | 6 | ~118 (N2, deduped) |
| `mask_token` | 6 (`= vocab_size`) | 118 |

Cosine schedule from `_make_alpha_schedule`:

| t | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| α_t | 1.000 | 0.854 | 0.500 | 0.146 | 0.000 |

α₀ = 1 (clean), α_T = 0 (fully masked).

### 1.1 Rollout: `sample_sequence`, the reverse process

Start fully masked at t = T and reveal downward. The fraction of *currently
masked* positions revealed at step t is

$$p_\text{reveal} = \frac{(1-\alpha_t)-(1-\alpha_{t-1})}{1-\alpha_t} = \frac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}$$

— not a tuned hyperparameter, but forced by the schedule: the expected masked
fraction must go from 1 − α_t to 1 − α_{t−1}.

| step t | p_reveal | tokens before | tokens after |
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
  clean token for *all* L positions, including already-revealed ones. It is
  sampled, not argmaxed.
- **Draw 2** has nothing to do with the model — pure scheduling, L coins with
  the same probability, then `& is_masked`.
- The `where` commits only the intersection; **every other prediction is thrown
  away** and re-made later with more context. That is the point: at t = 4 the
  model guesses a position knowing nothing; by t = 1 it re-guesses knowing the
  other three. Steps can be complete no-ops (all coins zero), which is normal.
- Revealed tokens are frozen — `where` only overwrites positions still `M`.
- p_reveal rises monotonically (0.15 → 0.41 → 0.71 → 1.00) and is exactly 1 at
  t = 1, which is what guarantees a complete sequence. The `still_masked` safety
  net after the loop covers only a float-precision Bernoulli miss; it *samples*
  (so the token has a well-defined log-prob) and leaves `reveal_step` at 0.

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

`scheduler.update(energies=...)` then nudges β from the spread.

### 1.3 `log_prob` → `old_log_probs` (exact trajectory, DDPO)

$$\log p_\theta(\tau) = \sum_t \sum_{i \text{ committed at } t} \log p_\theta(x_0[i] \mid x_t, t)$$

`x_t` is reconstructed from the recorded trajectory by one rule: **position i is
visible at step t iff `reveal_step[i] > t`** (it was committed at some later,
i.e. larger, step). Each position is scored exactly once, at the step that
committed it, and the result is `(B, L)` with no averaging.

For circuit 1 with `gate_tokens = [5,3,2,1]` and `reveal_step = [3,4,2,3]`:

| t | positions committed | x_t fed to `_logits` | contribution |
|---|---|---|---|
| 4 | 1 | `[M,M,M,M]` | `[ 0,-1.8, 0, 0]` |
| 3 | 0, 3 | `[M,3,M,M]` | `[-2.0, 0, 0,-0.9]` |
| 2 | 2 | `[5,3,M,1]` | `[ 0, 0,-1.5, 0]` |
| | **sum** | | `[-2.0,-1.8,-1.5,-0.9]` |

Why the trajectory and not the ELBO: GRPO needs the probability of the action
sequence actually *taken*. The reveal coins are θ-independent, so they cancel in
the ratio `exp(new − old)` and only these categorical terms remain. The ELBO
bounds `log p(x_0)` — reconstructability — which is not the sampler's
distribution, and it needed frozen corruption masks in the buffer to keep the
ratio deterministic. See `NOTES.md`, "Trajectory log-prob (DDPO) replaces the
ELBO"; the ELBO code is in git history (`sample_masks()`, `_corrupt()`).

Two honest caveats:
- The loop runs over the union of commit steps **across the batch**, so with
  B = 10 nearly every step is usually populated. Cost is "never worse than T",
  not a speedup.
- The ELBO scored ~L·T/2 terms against the trajectory's L: denser gradient, but
  biased. If the trajectory version underperforms in an A/B, that is the likely
  reason rather than a bug.

### 1.4 Buffer

Four tuples `(seq, energy, old_log_prob, reveal_step)`. `reveal_step` must ride
along, or replay would score a different trajectory than the one sampled.

---

## 2. DAG GNN (`model=dag_gnn`, `dag_gnn_features`)

### Setup

| | toy | real default |
|---|---|---|
| `num_samples` = `batch_size` = `buffer_size` = `warmup_size` | 4 | 10 |
| `L` = `ngates` | 4 | 10 |
| `n_qubits` | 6 | 16 (N2) |
| `V` | 6 | ~118 |
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

### 2.2 `sample_sequence` — L steps

Each step: `_step_forward` → `_scaled_logits` → sample → `_advance_dag`.

**Step 0** (`gate_node = 6`). `edge_index` is empty, so the GAT layers are
**skipped entirely** and `pooled` is just the mean of the raw wire embeddings.
The first gate is chosen with no structural information, only orbital physics.
Sample → **op 3**. `_advance_dag` sets `node_tokens[6] = 6+3 = 9` and, for each
q in `{0,1,4,5}`, adds edge `frontier[q]→6` and sets `frontier[q] = 6`:

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

Rule: `k` is forbidden at step t ⟺ ∃ i such that `k` commutes with all of
`prefix[i..t-1]` **and** `k < prefix[i]`. This keeps one representative per
Mazurkiewicz trace class (commuting gates only spellable in their sorted order).

| step | prefix | forbidden | allowed | sampled |
|---|---|---|---|---|
| 0 | — | — | `{0,1,2,3,4,5}` | 3 |
| 1 | `[3]` | `{0,2}` | `{1,3,4,5}` | 1 |
| 2 | `[3,1]` | `{0,2}` | `{1,3,4,5}` | 5 |
| 3 | `[3,1,5]` | `{0,1,3,4}` | `{2,5}` | 2 |

Two traps, both load-bearing:
- The `-inf` goes on the **scaled** logits, *after* the `-β` multiply. Masking
  first would flip the sign to `+inf` and make forbidden ops maximally likely.
- Entropy uses an **`-inf`-safe path**: `0 · (-inf) = NaN`, and
  `log_softmax`'s backward spreads that NaN to every logit. Replace `-inf`
  with 0 *before* the multiply.

Measured effect of this masking on N2: none, at current seeds. Recorded as
inconclusive, not negative — see `NOTES.md`.

### 2.4 `log_prob` → `old_log_probs`

Replays the identical DAG construction with the stored sequence, reading
`p(a_t | DAG_{t-1})` off each step. `state.get("reveal_step")` is `None` (only
the absorbing models set it), so `lp_kwargs = {}`. Exact and deterministic —
call it twice with unchanged weights and you get identical bits.

```
circuit 1: [-1.10, -1.35, -0.95, -1.28]
```

---

## 3. The gradient steps (both models)

`BufferDataset(buffer, repetition=step_per_epoch)`, `shuffle=False`,
`__getitem__` does `idx % len(buffer)` → **every batch is the same 4 circuits.**

Advantages are recomputed each step but identical, since energies are fixed:

```
mean = -107.2750,  std = 0.1287
advantages = (mean - E)/std = [-0.5827, +1.2819, +0.2719, -0.9712]
```

Circuit 1 (lowest energy) gets the largest **positive** advantage — lower energy
⇒ push probability up.

`GRPOLoss` has **two** terms; the first is an addition to textbook GRPO:

1. `-mean(new_log_probs[win_id])` — plain NLL on the lowest-energy circuit.
2. `- min(clipped, unclipped)` where `ratio = exp(new − old)` elementwise and
   `clipped = clamp(ratio, 1-0.2, 1+0.28)`.

Then `loss -= entropy_coeff * entropy.mean()` (`entropy_coeff: 0.01` by
default; 0 disables the bonus and the second `log_prob` output entirely).

Three steps of the toy DAG run:

| | ratio range | term 1 (NLL on winner) | `min(clipped, unclipped)` | loss |
|---|---|---|---|---|
| step 1 | `[1.0000, 1.0000]` | 1.1700 | −0.000015 | 1.1700 |
| step 2 | `[0.9713, 1.0392]` | 1.1315 | 0.022603 | 1.1089 |
| step 3 | `[0.9343, 1.0939]` | 1.0803 | 0.053159 | 1.0271 |

Read the ratio column downward — that is the whole dynamic. At step 1
`new == old` exactly, so every ratio is 1 and the PPO term contributes nothing.
As weights move the ratios fan out; over 30 real steps they reach the clip
bounds, and the clip stops one rollout group from moving the policy too far.

One caveat carried over from the ELBO switch: the trajectory log-prob sums over
L instead of averaging over T, so its **scale differs from the old ELBO runs**.
The clip range is scale-sensitive; watch it when comparing to pre-`5786019`
results.

---

## 4. Per-epoch cost

| | absorbing diffusion | DAG GNN |
|---|---|---|
| `sample_sequence` | ≤ T forwards (8–16) | L forwards (10) |
| `log_prob` at rollout | ≤ T | L |
| `log_prob` × `step_per_epoch` | 30 × T (240–480) | 30 × L (300) |

At L = 10 none of this matters: a single epoch's wall-clock is dominated by
`num_samples` CUDA-Q simulations plus the QSCI diagonalizations, which are
CPU-bound and not accelerated by the GPU.
