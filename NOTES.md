# Research Notes
# CLAUDE SAYS NO MODEL ACTUALLY LEARNS BUT I THINK THATS WRONG
# THE SCHEDULER IS NOT FIXED YOU NEED TO THINK ABOUT IT AGAIN. SEE THE TRAINER-TEMPERATURE PLOT/TRAINER-LOSS PLOT
# I CAN USE OTHER SCHEDULERS NO PROBLEM THERE ARE ALREADY SOME IMPLEMENTED BUT I SHOULD CHECK WHERE THEY ARE USED FIRST
# -> Used in gqe-for-qsci\configs\trainer\default.yaml
# ALSO TIME TO FINALLY RESOLVE THIS INVERSE TEMPERATURE STUFF
# WHAT DO WE WANT THE TEMPERATURE BLOCK TO LOOK LIKE? 

---

## Alternative denoiser architectures

The current absorbing diffusion model uses a **Transformer Encoder** as its
denoiser. Given the short sequence length (L = 10–20 gates) and discrete
token space, there are several alternatives worth trying.

### Why encoder vs. decoder matters

| | Transformer Encoder | Transformer Decoder (causal) |
|---|---|---|
| Attention | Full bidirectional | Left-to-right only |
| Suited for | Denoising (all positions visible) | Autoregressive generation |
| Parallelism | All positions in one pass | Same (teacher-forced training) |
| Circuit task | Natural — every gate can attend every other | Less natural — gate i can't see gates i+1..L |

For a diffusion model the encoder is the right default: when denoising
position i you want to use the context of all other positions, not just the
ones to the left. A decoder would only make sense if you generated the
circuit left-to-right (like GPT-2 does), which is exactly what the Simple
model effectively does.

### SSM / Mamba (State Space Model)

- Linear-time sequence model, competitive with Transformers on many tasks.
- Particularly efficient for very long sequences; less relevant at L=10 but
  interesting at L=50+.
- Would require swapping `nn.TransformerEncoder` for a Mamba block.
- Reference: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
  (Gu & Dao, 2023).

### Graph Neural Network (GNN)

Model the circuit as a graph: nodes = gate positions (L nodes), edges encode
relationships between positions. The GNN replaces the TransformerEncoder inside
`_CircuitDiffusionBase._logits()` — everything else (embeddings, `sample_sequence`,
`log_prob`, GRPO training loop) stays identical.

#### Why GNN over Transformer here

The Transformer treats all L positions as equally related (full attention) and
relies entirely on positional embeddings to learn structure. A GNN lets you
*encode domain knowledge directly into the graph topology* — which positions
interact, which gates share qubits, which operators are algebraically related.
At L=10 this doesn't help with efficiency, but it gives a richer inductive bias.

#### Graph structure options

**Chain graph** (recommended starting point)
- Edges: (i, i+1) bidirectional for i in 0..L-2
- Captures gate ordering naturally — gate i is applied before gate i+1
- O(L) edges; with k layers, information travels k hops
- At L=10 with 6 layers: full sequence coverage

**Fully connected graph**
- Edges: all (i, j) pairs
- Equivalent to Transformer attention but without the softmax weighting
- Loses the sparsity advantage; mainly useful as an ablation baseline

**Qubit-sharing graph** (physically motivated, dynamic — see Future Extensions below)
- Edge between positions i and j if their current operators share ≥1 qubit
- Must be recomputed every forward pass as tokens change during denoising

#### GNN layer choice: GAT (Graph Attention Network)

Graph Attention Networks learn a separate attention weight per edge, letting the
model focus on which neighbours matter. This is the closest GNN analogue to
Transformer self-attention, and the most natural fit here.

Alternatives: GCN (simpler, uniform neighbour averaging — no attention), GIN
(most expressive in theory, but harder to tune), GraphSAGE (designed for large
graphs, overkill at L=10).

#### Implementation plan

**New file: `gqe_qsci/gqe/models/gnn.py`**

Keep `diffusion.py` clean. The new file contains:

1. **`_CircuitGNNBase(Policy)`** — mirrors `_CircuitDiffusionBase` but replaces
   `nn.TransformerEncoder` with a stack of `GATConv` layers.

   Key differences in `__init__`:
   ```python
   # Instead of:
   self.denoiser = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
   
   # Use:
   self.gnn_layers = nn.ModuleList([
       GATConv(hidden_size, hidden_size, heads=num_heads,
               concat=False,   # average heads → hidden_size stays constant
               dropout=dropout)
       for _ in range(num_layers)
   ])
   self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size)
                                     for _ in range(num_layers)])
   # Precomputed static edge index stored as a buffer
   self.register_buffer("edge_index", self._build_edge_index(ngates, graph_type))
   ```

   Key difference in `_logits()`:
   ```python
   # Flatten (B, L, H) → (B*L, H) for PyG, run message passing, reshape back
   h = h.view(B * L, -1)
   edge_index = self._batch_edge_index(B)   # offset per sample in batch
   for gnn, norm in zip(self.gnn_layers, self.layer_norms):
       h = h + F.gelu(norm(gnn(h, edge_index)))   # residual + LayerNorm
   h = h.view(B, L, -1)
   return self.output(h)
   ```

   `_batch_edge_index(B)`: offsets the stored edge index by `b * ngates` for
   each sample b in the batch, creating one large disconnected graph (standard
   PyG batching pattern).

   `_build_edge_index(ngates, graph_type)`: constructs the static edge tensor.
   For `"chain"`: pairs (i, i+1) and (i+1, i) for all i. For `"full"`: all
   (i, j) with i≠j.

2. **`CircuitGNNModelAbsorbing(_CircuitGNNBase)`** — absorbing diffusion with
   GNN denoiser. `__init__`, `_corrupt`, `sample_masks`, `sample_sequence`,
   and `log_prob` are identical to `CircuitDiffusionModelAbsorbing`; only
   `_logits()` (inherited from `_CircuitGNNBase`) differs.

3. **`CircuitGNNModelSingleShot(_CircuitGNNBase)`** — single-shot with GNN
   denoiser. Same relationship to `CircuitDiffusionModelSingleShot`.

**New config files:**
```yaml
# configs/model/diffusion_gnn_absorbing.yaml
_target_: gqe_qsci.gqe.models.gnn.CircuitGNNModelAbsorbing
hidden_size: 128
num_layers: 6          # more layers than Transformer: GNN is shallower per layer
num_heads: 4
diffusion_steps: 16
noise_schedule: cosine
dropout: 0.1
graph_type: chain      # "chain" or "full"

# configs/model/gnn_singleshot.yaml
_target_: gqe_qsci.gqe.models.gnn.CircuitGNNModelSingleShot
hidden_size: 128
num_layers: 6
num_heads: 4
diffusion_steps: 16
dropout: 0.1
graph_type: chain
```

#### Critical design notes

**Residual connections are mandatory.** Without them, deep GNNs suffer from
*over-smoothing*: all nodes converge to the same representation after enough
layers. With residual + LayerNorm (as above), this is avoided.

**`concat=False` in GATConv.** GAT can either concatenate heads
(output size = `hidden_size × num_heads`) or average them (output = `hidden_size`).
`concat=False` keeps the hidden dimension constant, which is required for
residual connections to work.

**6 layers on a chain at L=10.** Each layer propagates information 1 hop.
After 6 layers, every node has seen all others (diameter of a 10-node chain is
9, but approximate coverage is sufficient by layer 5–6). Fewer layers can be
used if training is too slow.

**Static vs dynamic edges.** The chain/full graphs are computed once in `__init__`
and registered as a buffer (fast, no recomputation). The qubit-sharing graph
(future work) requires re-running edge construction every forward pass since it
depends on current token assignments.

#### Other GNN layer implementations

The current implementation uses **GATConv** (Graph Attention Network). Other
PyG layers can be swapped in by changing only the `self.gnn_layers` block in
`_CircuitGNNBase.__init__` and the `GATConv` import — everything else stays
identical:

| Layer | Class | Character |
|---|---|---|
| **GAT** (current) | `GATConv` | Learned per-edge attention weights |
| **GCN** | `GCNConv` | Uniform neighbour averaging, simplest |
| **GIN** | `GINConv` | Most expressive (Weisfeiler-Leman equivalent), needs MLP inside |
| **GraphSAGE** | `SAGEConv` | Samples neighbours; designed for large graphs |
| **EdgeConv** | `EdgeConv` | Uses edge features; good if edge types carry information |

For `GINConv` you additionally need to wrap a small MLP: 
`GINConv(nn.Sequential(nn.Linear(H,H), nn.GELU(), nn.Linear(H,H)))`.
For others it is a direct drop-in replacement.

#### Mixin refactor (future cleanup)

`CircuitGNNModelAbsorbing` currently duplicates the absorbing diffusion logic
from `CircuitDiffusionModelAbsorbing`. A clean solution is an
`_AbsorbingDiffusionMixin` class holding `_corrupt`, `sample_masks`,
`sample_sequence` and `log_prob`, used by both:

```python
class _AbsorbingDiffusionMixin:          # no __init__, pure method provider
    def _corrupt(self, ...): ...
    def sample_masks(self, ...): ...
    def sample_sequence(self, ...): ...
    def log_prob(self, ...): ...

class CircuitDiffusionModelAbsorbing(_AbsorbingDiffusionMixin, _CircuitDiffusionBase): ...
class CircuitGNNModelAbsorbing(_AbsorbingDiffusionMixin, _CircuitGNNBase): ...
```

Not done yet to avoid touching working code.

#### Dependency: PyTorch Geometric

PyG is not currently in the Docker image. The `Dockerfile` needs:
```dockerfile
RUN pip install torch_geometric
```
(torch_scatter and torch_sparse are bundled with recent PyG versions.)
This is the only blocker before implementation.

#### Future extensions — additional graph inputs

**NOT yet implemented. Design notes only.**

**1. (Anti)commutation graph of the operator pool**

Two Pauli operators P_i and P_j either commute ([P_i, P_j] = 0) or anticommute.
Commuting operators are physically interchangeable in the circuit — swapping two
commuting positions gives the exact same quantum state.

> **Why this matters more than it first appears.** The DAG GNN is often described
> as collapsing equivalent orderings of commuting gates into one object. That is
> only *partly* true. `_advance_dag` adds an ordering edge between any two gates
> that **share a qubit**, so the DAG only collapses orderings of gates with
> **disjoint support** (which trivially commute). Two gates that share qubits but
> still commute get chained anyway — an ordering the DAG asserts but physics does
> not.
>
> Concrete N2 example (pool indices 4 and 7, same footprint `[6,8,10,12]`):
> ```
>   op 4:  Y(6)  Y(8)  Y(10)  X(12)
>   op 7:  X(6)  X(8)  Y(10)  X(12)
>          anti  anti  comm   comm     -> 2 anticommuting sites (even) => COMMUTE
> ```
> Swapping them yields the identical unitary, state, energy and reward — yet the
> DAG treats the two orderings as distinct objects.
>
> Rule: two Pauli strings commute iff they anticommute on an **even** number of
> qubits. The refinement is to add an ordering edge only when two gates on a
> shared qubit **anticommute**.
>
> **Measured on the N2 pool (154 operators, 11 628 pairs):**
>
> | pair type | count | share |
> |---|---:|---:|
> | disjoint support (DAG adds no edge — correct) | 3 874 | 33.3 % |
> | share a qubit, **anticommute** (edge is real) | 3 317 | 28.5 % |
> | share a qubit, **commute** (edge is **spurious**) | 4 437 | 38.2 % |
>
> So of the 66.7 % of pairs that share a qubit, **57 % commute** — the majority
> of the ordering edges the DAG imposes are physically meaningless.
>
> ~~This arguably makes the commutation-aware DAG the highest-value change
> available to the DAG GNN.~~ **It does not.** Removing this redundancy from the
> action space produced no measurable change in the energy curves. See "Result:
> canonical masking (NEGATIVE)" in the redesign section. The statistic is a
> pairwise property of the *pool*; the optimizer samples ~300 circuits out of
> ~1e22 and never revisits equivalent orderings, so the redundancy costs it
> nothing.
>
> Caveat: frontier mechanics currently depend on qubit-wire edges (`frontier[q]`
> is where the next gate on wire q attaches). Dropping edges for commuting gates
> changes what "frontier" means when a gate touches a wire without consuming it —
> so this is a design change, not a one-line edit.
>
> Note also that commutation is a *pairwise* relation and therefore cannot be
> encoded in the per-operator feature menu (see "Cross-molecule generalization").
> It belongs in the graph edges or as an attention bias.

- Graph has `vocab_size` nodes (one per operator), precomputed once from Pauli strings
- In the GNN: when position i holds token k_i and position j holds token k_j,
  add a typed edge between those positions based on `commutation_matrix[k_i, k_j]`
- These edges are *dynamic*: depend on current token assignments, must be rebuilt
  each forward pass
- Masked positions (during absorbing denoising) have no token → no commutation edges
- Stored as `register_buffer("commutation_matrix", ...)` — shape (V, V) bool
- Precomputed in `factory.py` from `operator_pool` before model construction

**2. Hardware connectivity graph**

On real quantum hardware (e.g. IBM heavy-hex), only adjacent qubit pairs can have
a direct CNOT. Operators acting on non-adjacent qubits need SWAP routing (extra
gates, more noise).

- Precompute `operator_qubits`: for each operator k, which qubits does it act on?
  (Readable directly from the Pauli string.)
- Edge type between circuit positions: "hardware-native" if the qubits of both
  operators are directly connected in the hardware graph; "needs routing" otherwise
- Stored as `register_buffer("hw_connectivity", ...)` — shape (Q, Q) bool
- Makes the model prefer circuits that are cheap to implement on the target device
- Becomes essential once we move from simulation to real hardware runs

Both graphs use the same node set (L circuit positions), so they slot in as
additional edge types in the GNN alongside the base chain edges.

### Reservoir Computing (Echo State Network)

Replace the trained Transformer encoder with a **fixed random recurrent network**
(reservoir). Only the linear readout layer mapping reservoir states to gate logits
is trained — the reservoir itself is never updated.

**How it works:**
Input tokens are embedded and fed sequentially into a large random RNN. The
reservoir projects the input into a high-dimensional space through its nonlinear
recurrent dynamics. A linear layer trained on top reads out gate logits. Training
is just a linear regression — milliseconds, no GPU needed.

**Why it is interesting here:**
- L=10 is short enough that an RNN reservoir can capture the full sequence without
  vanishing gradient issues
- Near-zero training cost — useful for rapid ablations
- Primary scientific value: a **controlled baseline**. If the trained Transformer
  only marginally outperforms a random reservoir, the learned representations are
  not adding much. If it wins decisively, the inductive bias of training is
  confirmed to matter.

**Implementation:**
Swap `self.denoiser = nn.TransformerEncoder(...)` in `_CircuitDiffusionBase` for a
fixed `nn.RNN` or `nn.GRU` with `requires_grad=False` on all parameters. Keep only
`self.output = nn.Linear(hidden_size, vocab_size)` trainable. No new dependencies.

**Expected outcome:** likely underperforms the Transformer, but informative as a
lower bound. The quantum reservoir computing (SYK) connection below is a
physically motivated upgrade of this same idea.

**References:**
- Jaeger & Haas, *Harnessing Nonlinearity*, Science 2004 — original ESN paper
- Lukoševičius & Jaeger, *Reservoir computing approaches*, 2009 — survey

### Quantum Reservoir Computing (SYK model)

The Sachdev-Ye-Kitaev (SYK) model as a quantum reservoir — a physically motivated
upgrade of classical RC where the reservoir is a quantum many-body system.

**Why SYK specifically:**
SYK is maximally chaotic (saturates the chaos bound, Lyapunov exponent
λ_L = 2π/β), scrambles information in O(log N) time, and has an exponentially
large (2^(N/2)) effective Hilbert space. These properties make it nearly ideal as
a reservoir: fast mixing, high-dimensional features, structured correlations.

**Three possible connections to this project:**

1. **As a fixed quantum circuit denoiser**: replace the trained Transformer with
   SYK Hamiltonian evolution as a fixed reservoir. Only the classical readout
   mapping quantum measurement outcomes to gate logits is trained. Combines
   quantum expressiveness with the simplicity of RC.

2. **As inspiration for the operator pool**: SYK's all-to-all random Pauli
   interactions resemble a richly connected operator pool. An SYK-inspired pool
   might explore more of Hilbert space per gate than a standard UCCSD pool.

3. **As a benchmark Hamiltonian**: use the SYK model itself as the target
   molecule — find its ground state via GQE-QSCI. A natural physics experiment
   given the connection.

**Main challenge:** SYK requires all-to-all connectivity (every qubit pairs with
every other), which is expensive on real hardware. Sparse-SYK approximations
(chunked or k-local SYK) trade some scrambling efficiency for hardware feasibility.

**References:**
- Sachdev & Ye, *Gapless spin-fluid ground state*, PRL 1993
- Kitaev, KITP talks 2015 — modern SYK formulation
- Fujii & Nakajima, *Harnessing disordered quantum dynamics*, PRX 2017 — QRC
- Kobayashi et al., various QRC papers using SYK-like systems

### 1-D Convolutional network

- Replace attention with dilated non-causal convolutions (similar to WaveNet).
- Much cheaper than O(L²) attention at larger L.
- Would lose long-range dependencies; acceptable if gates are nearly
  independent early in training but may hurt as correlations emerge.

### Perceiver IO

- Fixed-size latent array that attends to the variable-length gate sequence.
- Good for multi-modal / variable-length inputs; overkill here but useful
  if we condition on Hamiltonian parameters (see below).

---

## Trajectory log-prob (DDPO) replaces the ELBO — IMPLEMENTED

**What changed.** `CircuitDiffusionModelAbsorbing` and
`CircuitGNNModelAbsorbing` previously scored circuits with a denoising ELBO
averaged over all T timesteps, using corruption masks pre-sampled by
`sample_masks()` and frozen in the replay buffer. Both now compute the **exact
log-probability of the reverse trajectory that was actually sampled**:

```
log p_θ(τ) = Σ_t Σ_{i committed at t} log p_θ(x_0[i] | x_t, t)
```

`sample_sequence` records `state["reveal_step"]` — (B, L) long, the timestep at
which each position was committed. `log_prob` reconstructs the state the network
saw at step t via **visible ⟺ reveal_step > t** and scores each position exactly
once. Returns (B, L), same as before, so `GRPOLoss` is untouched.
`sample_masks()` and `_corrupt()` are deleted (git history has the ELBO version).

**Why.** GRPO needs the probability of the *action sequence taken*, not of the
final object. For absorbing diffusion the true `p(x_0)` is intractable (it
marginalises over all L! reveal orders), which is why the ELBO was there — but
the ELBO measures *reconstructability of x_0*, which is not the sampler's
distribution, so the importance ratio was a ratio of surrogates. The reveal
coins are θ-independent and therefore cancel in `exp(log p_new − log p_old)`,
leaving exactly the categorical terms above. This is the DDPO formulation:
*Training Diffusion Models with Reinforcement Learning* (Black et al., 2023);
cf. DPOK (Fan et al., 2023). It also makes absorbing consistent with
single-shot and the DAG GNN, which were already exact-trajectory.

**Secondary benefits.**
- No `sample_masks`, no frozen-mask bookkeeping; the trajectory *is* the record.
- Storage (B, L) ints instead of (B, T, L) bools.
- Forward passes = number of *distinct commit steps* ≤ T (skips steps that
  committed nothing). NB the saving shrinks as B·L grows, since the loop runs
  over the union of steps across the batch — at B=6, L=8, T=8 all 8 steps were
  used, i.e. no saving. Treat this as "never worse", not as a speedup.
- The safety net now *samples* instead of `argmax`, so every token has a
  well-defined sampling log-probability. It effectively never fires:
  `p_reveal` at t=1 is `(α_0 − α_1)/(1 − α_1 + 1e-8)` with α_0 = 1 exactly,
  i.e. ≈ 1 − 7e-8. `reveal_step = 0` marks such positions and the
  `visible ⟺ reveal_step > t` rule handles t=0 uniformly.

**The one argument for the ELBO** (worth remembering if this underperforms):
gradient *density*. The ELBO scored every masked position at every timestep
(~L·T/2 terms); the trajectory scores each position once (L terms). Denser but
biased, vs sparser but exact. DDPO's evidence favours exact; an A/B on the
existing N2 setup would settle it here (ELBO results already exist from before
this change).

**Verified** (no cudaq/PyG needed, `diffusion.py` is torch-only): the states
`log_prob` reconstructs match, element-wise, every state the network actually
saw during `sample_sequence` (instrumented `_logits`); two calls agree
bit-for-bit; every position is scored exactly once; forward passes equal the
number of distinct commit steps; missing `reveal_step` raises; gradients flow;
buffer collate/pickle round-trips both the tensor and the `None` case.

---

## Alternative diffusion formulations

### Uniform noise (D3PM absorbing → D3PM uniform) — OPEN QUESTION

> **Parked question (Lukas):** *why do masked/absorbing diffusion at all —
> why not ordinary diffusion starting from a random gate sequence?*
> Not yet answered; the sketch below is the starting point.

The current model uses *absorbing* diffusion: tokens are masked (replaced by
a single MASK token). An alternative is **uniform noise**: at each step a
token can transition to *any* other gate with some probability.

- Forward: each token is re-sampled uniformly from the vocabulary with
  probability (1 − α_t) instead of just being masked.
- Backward: the denoiser predicts x_0 but the posterior now mixes in a
  uniform component rather than a delta on MASK.
- Advantage: no special MASK token needed; every intermediate sample is a
  valid gate sequence.
- Note: `CircuitDiffusionModelSimple` is *almost* this (it starts from uniformly
  random tokens and re-samples every position each step) but has no principled
  forward process and a proxy log_prob, so it is not a fair test of the idea.
- Reference: *Structured Denoising Diffusion Models in Discrete State-Spaces*
  (Austin et al., 2021).

### Continuous relaxation / Diffusion-LM

Convert discrete tokens to continuous embeddings, apply Gaussian diffusion in
embedding space, then project back to discrete tokens via argmax or softmax
at the final step.

- Natural extension of DDPM/DDIM to discrete sequences.
- Lets us reuse continuous-diffusion theory (ELBO, DDIM sampling, etc.).
- The embedding space can be learned end-to-end.
- Reference: *Diffusion-LM Improves Controllable Text Generation*
  (Li et al., 2022).

### Discrete flow matching

Replaces the Markovian diffusion chain with a direct **flow** from noise to
data in discrete space.

- No T-step Markov chain; a single network learns the flow directly.
- Potentially fewer forward passes at inference → faster sampling.
- Very recent; see *Discrete Flow Matching* (Gat et al., 2024).

### SEDD (Score Entropy Discrete Diffusion)

Score-based approach adapted for discrete state spaces.

- Learns a *score* (gradient of log-density) over the discrete space using
  a ratio-based objective.
- Avoids the ELBO decomposition; the objective is a form of denoising score
  matching on discrete data.
- Reference: *Discrete Diffusion Modeling by Estimating the Ratios of the
  Data Distribution* (Lou et al., 2023).

### Multi-step DDIM-style sampling

The current reverse process uses ancestral sampling (re-samples x̂_0 at
every step). A deterministic DDIM-style schedule would:

- Fix x̂_0 after the first confident prediction.
- Allow fewer denoising steps at inference without retraining.
- Reference: *Denoising Diffusion Implicit Models* (Song et al., 2020) —
  the idea extends to absorbing diffusion via the MDLM framework.

---

## Other training / objective ideas

### GSPO instead of GRPO

The codebase already has GSPO. Worth doing a systematic comparison on the
same molecule to understand which objective suits discrete diffusion better.

### Temperature annealing per diffusion step

Rather than a single global inverse temperature, use a per-step temperature
τ_t that is higher at large t (more exploration when mostly masked) and
lower at small t (exploit confident predictions near the end).

### Conditioning on Hamiltonian parameters

Pass the molecular Hamiltonian as additional context to the denoiser.
- Encode Hamiltonian coefficients as a cross-attention key/value sequence.
- Allows a single model to generalise across bond lengths / molecules.

---

## Current model state

### Diffusion model

The active diffusion model is `CircuitDiffusionModelAbsorbing` (config:
`model=diffusion_absorbing_matched`):

- **Architecture**: Transformer Encoder, hidden_size=256, 8 layers, 8 heads
- **Diffusion**: Absorbing / MDLM-style, T=16 steps, cosine noise schedule
- **log_prob**: Proper denoising ELBO averaged over all T timesteps.
  Corruption masks are pre-sampled in `sample_masks()` and stored in the
  replay buffer so that the GRPO importance-weight ratio is deterministic
  (Fix 4).
- **Entropy regularisation**: `entropy_coeff=0.01` subtracted from the GRPO
  loss to encourage sequence diversity (Fix 5).

The original simplified model (`CircuitDiffusionModelSimple`, config:
`model=diffusion`) is kept for backward compatibility but should not be used
for new experiments.

### Shot count and subspace saturation

With `shots=100_000` per circuit, even a random early-training circuit samples
enough of the valid Hilbert space to saturate a `max_dim=170` subspace
immediately (from iteration 1). This means the QSCI subspace size is always
at the cap and does not reflect training progress.

The paper (Figure 3b) uses a lower shot count (~1,000 per circuit), which gives
~80 determinants for random circuits and lets the subspace grow to 170 as the
model improves. Use `experiment=n2_l10_gpt2_paper` /
`experiment=n2_l10_diffusion_paper` for paper-comparable runs.

## Value proposition: fair classical baseline

Is QSCI's quantum-sampled determinant selection actually better than a *classical*
selection at the same subspace size? Ground-state quantum advantage is contested
(Lee et al., Nat. Commun. 14, 1952 (2023)), and a critique — "Exposing a Fatal
Flaw in Sample-based Quantum Diagonalization Methods Based on Ground-State
Sampling" (Copenhagen) — argues on **N2** (this project's molecule) that QSCI
wavefunctions are *less compact* than classical selected-CI, making QSCI more
expensive rather than advantageous. See memory `project-value-prop-fair-baseline`.

### The experiment (implemented)

`gqe_qsci/qsci/baseline.py` + `baseline_selection.py`: energy vs subspace size
for three determinant-selection methods, diagonalized by the SAME pyci path QSCI
uses (only the selection differs):

- **HCI** — heat-bath CI, the classical competitor QSCI must beat
  (`pyci.add_hci` iterations from HF; natural (ndet, energy) curve).
- **random** — number/spin-conserving determinants, HF included; the floor.
- **FCI** — exact (== CASCI here, active space is full CI); the ceiling.

Imports pyscf + pyci only (no cudaq / training stack). Run:
```powershell
docker run --rm --entrypoint /bin/bash -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 baseline_selection.py molecule=n2"
```

Verdict: read your QSCI best energy at subspace_dim~170 from W&B and compare to
the HCI curve at ~170 determinants. QSCI below HCI → the quantum sampling adds
value; QSCI >= HCI → the classical diagonalization is doing the work.

Caveat: at N2 (10e,8o)=3136 dets this is a *validation* (FCI known, so you see
convergence rates); a genuine advantage test needs an active space beyond
classical FCI, where HCI/DMRG is the only ceiling.

## Dynamics pivot: GQE for time evolution instead of ground state

Design sketch, nothing implemented. Motivated by the section above: ground-state
advantage is contested, whereas Hamiltonian simulation has a cleaner story and
an unarguable classical baseline (Trotter at matched depth).

### What survives and what dies

**Survives — the whole generator.** Operator pool, all five policy models, GRPO,
the feature scorer, the DAG machinery. None of it knows what the reward means;
it is a "pick a good sequence of Pauli evolutions" engine.

**Dies — the whole evaluator.** QSCI is a ground-state tool: it diagonalises H
in a sampled determinant subspace. Time evolution has nothing to diagonalise.
`qsci/pipeline.py`, determinant extraction, PyCI, GEVP refinement all become
irrelevant. **The project becomes "replace the reward", not "replace the model".**

### The crux: scoring "this circuit ≈ U(t)"

Ground state has an unusually friendly reward — variational, so any circuit is a
valid upper bound and bad ones degrade gracefully. Matching a *unitary* is
harder: you are approximating an operator, not preparing a state.

| approach | cost | weakness |
|---|---|---|
| fidelity on one reference state | cheap | can match on \|psi_ref> and fail elsewhere |
| average over random input states | moderate | statistical proxy for process fidelity |
| Hilbert-Schmidt test (HST / local LHST) | moderate | the established method; LHST is the trainable variant |
| full process fidelity | exponential | needs tomography |

At this project's sizes (16 qubits) `expm(-iHt)` is classically computable, so
the reward can be evaluated exactly on statevectors — arguably *easier* to
prototype than the QSCI reward, with no sampling noise.

### Version A — fixed t: learned Trotterization

Variational quantum compiling for one t. The framing that fits this codebase:
the pool is *already* Pauli evolutions e^{i theta P}, and Trotter expresses
exp(-iHt) ~ prod_k exp(-i h_k P_k t/n) as exactly such a product. So this is
**learned Trotterization** — instead of the fixed product formula, learn which
Pauli evolutions in which order best approximate U(t) at a given gate budget.

Baseline is clean and unarguable: standard Trotter at matched depth. Much
crisper than the contested ground-state comparison.

### Version B — general t: variational fast forwarding

Learn ONE structure that evaluates any t at fixed depth, via an approximate
diagonalisation

    U(t) ~ W D(t) W^dagger

with W a fixed circuit and D(t) diagonal with angles **linear in t**. This is
Variational Fast Forwarding (Cirstoiu et al., npj QI 2020) — i.e. the
"circuit with one parameter t" idea is a known, studied method.

Why it does not violate the **no-fast-forwarding theorem** (generic H needs
Omega(t) gates, so fixed depth cannot be accurate for arbitrary t): VFF is an
*approximate* diagonalisation — accurate on a subspace or up to a horizon, not
universally.

Architecturally this is the "decouple structure from parameters" question:
t must enter through the ANGLES, never the gate selection.

- policy picks **structure** (W, and which diagonal generators)
- angles are theta_k = lambda_k * t with learned lambda_k
- reward = fidelity averaged over several sampled t

### Two consequences for the pool

**1. The pool should become the Pauli terms of H itself.** Ground state uses
UCCSD excitations weighted by CCSD amplitudes. For dynamics the natural
operators are the P_k in H = sum_k h_k P_k, and the natural feature is h_k —
already available in `cas_hamiltonian`. More directly motivated than the
excitation pool, and `get_operator_features()` would need a matching rewrite
(h_k replaces the CCSD amplitude as the importance signal).

**2. The commutation work stops being a null result and becomes central.**
Trotter error is *governed by commutators* [P_i, P_j]: commuting terms reorder
with zero error, and the leading error term sums over non-commuting pairs. So
`get_commutation_matrix()`, the canonical/trace masking, and the DAG GNN's
ordering structure — all of which measured as no-effect for QSCI (see "Result:
canonical masking" and the trace redesign section) — are exactly the right
machinery here. Gate ORDER genuinely matters for Trotter error in a way it did
not for ground-state sampling.

That is the most appealing part of this pivot: a chunk of already-built,
already-verified work that produced no measurable gain would become load-bearing.

### Caveats

- The references (VFF; Khatri et al. *Quantum-assisted quantum compiling*, 2019
  for HST/LHST; the no-fast-forwarding theorem) are recalled from memory —
  verify before building on them.
- The hard engineering is the REWARD, not the generator. Budget accordingly.
- Version A is a well-scoped project with a clean baseline. Version B is a
  research question bounded by the no-fast-forwarding limit. Do A first.

## Sample-efficiency / diversity reward ideas (design notes, not implemented)

Context: the QSCI value proposition hinges on the *sampling* problem — the
coupon-collector cost of discovering rare-but-important determinants (weight
|c_D|^2 needs ~1/|c_D|^2 shots). Classical HCI finds them deterministically for
free. The GQE's potential value is learning circuits that make QSCI's sampling
cheap. These are ideas for putting that into the loss. Recorded with confidence
levels; NONE implemented yet.

### What was ruled out: pairwise overlap penalties (CONFOUNDED — do not pursue)

Idea was: penalize circuits in a rollout group whose wavefunctions overlap, to
force diversity. Two versions, both fail for the same reason:
- **CI-vector overlap** (the refinement's `S = C†C`, gevp.py:37): the `sci_states`
  are *diagonalized* CI vectors, each the variational ground state within its
  subspace, so they all converge to the SAME state -> overlap -> 1 at
  convergence. Penalizing it penalizes success.
- **Sampled-distribution overlap** (Bhattacharyya of the measured histograms):
  fails identically. Good circuits must all put probability on the same
  *important* determinants (that is what makes them good), so their distributions
  overlap *because* they are good.
- Root cause: "low energy" and "high overlap" are the same thing here (both mean
  "covers the important determinants"), so ANY pairwise-overlap penalty fights
  the energy objective. Confirmed via: two circuits can both reach the ground
  state with different determinant sets only if they differ in the *negligible
  tail* while agreeing on the important determinants — i.e. good => overlapping.

### The idea that survives: "distorted circuits" (GOOD — user endorses)

A single circuit that *is* the ground state samples a rare determinant D only
~once per 1/|c_D|^2 shots. A circuit deliberately **distorted** to boost D's
amplitude (to ~0.1) samples it in ~100 shots — at the cost of being a worse
ground-state approximation itself. So an **ensemble of distorted circuits, each
over-sampling a different rare-but-important determinant**, collectively
discovers the tail far cheaper than N copies of the exact ground state. The
refinement then unions the discovered determinants and diagonalizes, recovering
the true amplitudes. Consequence: rewarding each circuit's *individual* energy is
suboptimal for the ensemble — the group benefits from circuits that sacrifice
individual energy for unique coverage. The current reward (per-circuit energy ->
GRPO advantage) cannot see this.

### Candidate reward A: marginal contribution to refined energy (UNCERTAIN — user not sure)

Reward each circuit by its leave-one-out marginal value to the group's refined
(union) energy:  A_i ∝ E_refined(group \ i) − E_refined(group). Rewards a circuit
that uniquely lowers the ensemble energy even if its own energy is higher; a
redundant circuit contributes ~0. Not confounded with individual goodness (that
is the point). Uses the refinement machinery. Cost: K+1 GEVP solves per group
(Shapley would be K-fold more). *User is not confident this is the right lever —
park it.*

### Candidate reward B: role-specialized batch (UNCERTAIN — note for later)

A `d_max`-sized batch where the **i-th circuit is trained specifically to sample
the i-th most important determinant with high efficiency**. I.e. assign each
circuit a target determinant (ranked by importance from the current refined CI
vector) and reward it for concentrating sampling probability on its assigned
target. The ensemble then covers the top-d_max determinants each via a
specialized, sample-efficient circuit — a direct, explicit version of the
"distorted circuits" idea. Open questions: how to assign/rank targets online
(from the running refined CI vector?), how the policy conditions on its assigned
role, whether d_max circuits per batch is affordable. *User not fully confident;
recorded for later.*

### The ceiling on all of this (connects to the value proposition)

Diversity pays only to the extent there is a hard-to-sample tail to cover. If the
ground state is dominated by a few important determinants with negligible tail,
every good circuit already covers them and diversity buys nothing. Diversity (and
QSCI advantage generally) matters only under **strong multireference character**,
where significant weight lives in the tail — the same regime where the ground
state is classically hard. "Does ensemble diversity help?" and "is there quantum
value here at all?" are gated by the same physics. See "Value proposition: fair
classical baseline" above and memory `project-value-prop-fair-baseline`.

## Hardware-aware features for the DAG GNN

The DAG GNN (`model=dag_gnn`) is structurally better suited to encoding real
hardware properties than the Transformer or position-based GNN models, because
each DAG edge already corresponds to exactly one (gate, qubit-wire) pair.

### Connectivity graph (easiest)

Restrict or penalise operators whose qubit footprint spans non-adjacent qubit
pairs on the target hardware (e.g. IBM heavy-hex).

**Hard constraint** — mask logits at each generation step for operators
that would require SWAP routing. The model simply cannot place them:

```python
# in _step_forward, after computing logits:
invalid = ~hardware_reachable_mask(frontier, self.hw_adjacency, self._fp_flat)
logits = logits.masked_fill(invalid, float('-inf'))
```

**Soft constraint** — add a precomputed `hw_cost` scalar (number of SWAPs
needed) as an extra feature on each gate node. The model learns to avoid
costly placements without hard prohibition.

### T1/T2 decoherence times

Per-qubit scalars (decoherence timescales). Attach as additional features on
the input nodes (0 .. n_qubits-1). The GNN propagates them along qubit-wire
edges to gate nodes, so the model learns that gates on noisy qubits should be
avoided or placed earlier (shorter wire depth = less decoherence exposure).

Implementation: extend `node_embedding` with a small MLP that fuses the
learned token embedding with a per-qubit hardware feature vector, or simply
add a separate `nn.Embedding`-like linear layer for qubit hardware features
and sum it into the node representation alongside `node_embedding`.

### Gate fidelities

Per-(operator, qubit-wire) error rates from device calibration data.
These map naturally to **edge features** on the DAG, since each edge encodes
one (gate, qubit-wire) relationship. `GATConv` supports edge features via its
`edge_attr` argument — pass a scalar fidelity for each edge and the attention
mechanism can learn to route information preferentially through high-fidelity
connections.

### Interaction with canonical-form masking

Canonical masking (see "Redesign proposal: the circuit as a trace") constrains
gate **order**; hardware connectivity constrains gate **cost**. They are
orthogonal relations over the same nodes, and there is no fundamental conflict.
Four points to keep in mind:

**1. Two edge types, not one.** In the GNN they coexist naturally:
- *directed* anticommutation edges → ordering / causality
- *undirected* connectivity edges → routing cost / locality

A separate `GATConv` per edge type with summed outputs keeps the qubit-locality
signal without letting it masquerade as an ordering constraint.

**2. The separation of concerns favours masking.** A *trace* (the equivalence
class) is the physical object. The policy should search over traces; a
hardware-aware transpiler should then pick the cheapest linear extension inside
the chosen trace. Canonical masking makes the policy emit exactly one
representative per trace, and the compiler remains free to reorder commuting
gates however routing prefers. Nothing is lost.

**3. Caveat — never measure circuit-level hardware cost off the emitted order.**
SWAP count and depth depend on the ordering. Under masking the emitted order is
the *lexicographic* representative, not the routing-optimal one, so a SWAP count
read off the raw sequence is an artifact of the operator indexing, not a real
cost. Circuit-level cost must be computed after transpilation (or over the trace
as a whole). Per-operator costs such as the existing `gate_cost` feature are
order-independent and unaffected.

**4. Opportunity — the canonical order is arbitrary, so choose it.** The
lexicographic rule breaks ties by *operator index*, which carries no meaning.
Permuting the pool's indices therefore encodes a free preference: sort operators
by `gate_cost`, or by hardware-nativeness, and the lexicographically-minimal
representative becomes the hardware-friendliest one. Costs nothing but a
relabelling in `build_operator_pool`.

**5. Composing masks needs a non-empty guard.** The canonical mask alone can
never forbid every operator (the largest index can never satisfy `k < w[i]`).
That guarantee does **not** survive union with a hardware mask that forbids
non-native operators — the two together could in principle mask everything. Any
composed mask must check that at least one operator remains, with a defined
fallback.

### Capacity-matched config for fair hardware comparison

When comparing hardware-aware DAG GNN against GPT-2, use the matched config
(see "Capacity-matched config" section below) so that performance differences
reflect inductive bias, not parameter count.

---

## Capacity-matched DAG GNN config

The default DAG GNN (`hidden_size=128, num_layers=6`) has ~825K parameters vs
GPT-2's ~10.7M (~13× smaller). For a fair comparison that isolates the effect
of the DAG inductive bias from model capacity, use a matched config:

```yaml
# configs/model/dag_gnn_matched.yaml
hidden_size: 384
num_layers: 9
num_heads: 4
```

Parameter estimate for this config (N2, V≈80):
- 9 × GATConv(384, 384, heads=4):  9 × ~1.18M ≈ 10.6M
- Embeddings + head:                          ~68K
- Total:                                     ~10.7M  ← matches GPT-2

Whether this matters: model capacity is **unlikely to be the main bottleneck**
at L=10 with a vocabulary of ~80 operators. The task is small — 10 discrete
decisions from 80 choices. Both GPT-2 (10.7M) and the default DAG GNN (825K)
are already overparameterised for this search space. Differences in convergence
speed are more likely explained by inductive bias (DAG commutativity
equivalences, explicit qubit-wire structure) than by parameter count.

A matched-size experiment is still worth running once to confirm this, but if
the default DAG GNN already matches GPT-2 performance at 13× fewer parameters,
that is itself a positive result.

---

## BUG (fixed): temperature / inverse-temperature naming confusion

Every scheduler and every model in this codebase actually tracks and consumes
$\beta$ (inverse temperature), never $T$: sampling is always
`Categorical(logits = -inv_temperature * raw_logits)` — the Boltzmann form
$\pi(a) \propto \exp(-\beta \cdot \text{logits}(a))$, with the model's raw
output acting as a learned pseudo-energy (low = good, by construction of the
policy-gradient sign). But every parameter and internal attribute up to this
point was named `temperature`, not `inv_temperature`/`beta` — including the
abstract `Policy.act`/`log_prob` signatures, every model's `act` /
`sample_sequence` / `log_prob`, and each scheduler's internal
`current_temperature` attribute (even though the public getter was already
correctly named `get_inverse_temperature()`).

This was not just cosmetic — it directly caused the `VarBasedScheduler`
direction bug below (its docstring described *temperature* behaviour while its
code adjusted *inverse* temperature, and the two got swapped without anyone
noticing) and is exactly the kind of bug a future edit would reintroduce (e.g.
someone "fixing" a model to do `logits / T` instead of `-beta * logits` because
the parameter is named `temperature`).

**Fix:** renamed the parameter/attribute consistently to `inv_temperature`
(models: `policy.py`, `gpt2.py`, `diffusion.py`, `gnn.py`, `dag_gnn.py`) and
`current_beta` (scheduler internals: `scheduler.py`), everywhere. Public API
(`get_inverse_temperature()`) unchanged — it was already correct. Also fixed:
`train_pipeline.py`'s checkpoint restore referenced the old attribute name
(`scheduler.current_temperature = ...` → `scheduler.current_beta = ...`), and
the logged metric key had a stray space (`"trainer/inv temperature"` →
`"trainer/inv_temperature"`, matching what README already documented). No
mathematical behaviour changed by this rename alone — see the scheduler
direction/default change below, which is a separate, deliberate decision.

---

## BUG (fixed): VarBasedScheduler was silently a DefaultScheduler

**Affected every run in the repo prior to the fix**, including all GPT-2 /
diffusion / GNN comparison groups.

`VarBasedScheduler.update()` (`scheduler.py:148`) branches on

```python
if current_var > self.target_var:   # sharpen
else:                               # flatten
```

with `current_var = energies.var()`. The upstream default was
`target_var: 1e-5`. But N2 batch energy std is **0.05–0.22 Ha**, i.e. variance
**2.5e-3 – 5e-2** — 100–1000× above the target. The `else` branch therefore
**never executed**: the scheduler incremented unconditionally every epoch and
`trainer/inv_temperature` was a pure linear ramp `initial + delta*epoch`
(0.55 → 1.12 over 30 epochs; slope exactly `delta`=0.02 ✓). Mask and nomask runs
overlaid *exactly*, because the update is deterministic once the branch is fixed.

Consequence: sampling sharpened on a blind fixed schedule, and the advertised
variance-adaptive behaviour did nothing. The else-branch would only fire below
std ≈ 3 mHa (~chemical accuracy) — a regime these runs never reach.

**First fix (superseded, see below):** `target_var: 1e-2` in
`configs/trainer/default.yaml` (std ≈ 0.1 Ha, mid-range of the observed spread,
so it adapts both ways). Calibrated on N2 @ 2.5 Å.

**This fix exposed a second, deeper bug.** Recalibrating `target_var` let both
branches of `VarBasedScheduler.update()` actually fire — and doing so surfaced
that the branches adjust $\beta$ in the *opposite* direction from what the
class's own docstring claimed (high variance was documented as "more
exploration" but the code raised $\beta$, which sharpens = *less* exploration).
This was visible in practice as GPT-2's $\beta$ staying persistently high
(over-exploiting) whenever its batch-energy variance sat above target. See
"BUG (fixed): VarBasedScheduler direction bug" below for the full story and the
resolution finally adopted (switch the default to `DefaultScheduler`, not a
`target_var` retune).

---

## BUG (fixed): VarBasedScheduler direction bug (docstring vs. implementation)

`VarBasedScheduler.update()`'s docstring said:

> "If current variance exceeds target, increases temperature (more
> exploration). Otherwise, decreases temperature (more exploitation)."

but the code did:

```python
if current_var > self.target_var:
    self.current_temperature += self.delta  # <- code's own comment: "decrease T"
else:
    self.current_temperature -= self.delta  # <- code's own comment: "increase T"
```

`current_temperature` is $\beta$ (returned unmodified by
`get_inverse_temperature()`), so raising it always *sharpens* (less
exploration), never the reverse. The docstring and the code's own inline
comment directly contradicted each other, two lines apart — a direct casualty
of the temperature/inverse-temperature mix-up documented above. This was
**not** hidden by the `target_var` miscalibration bug — it was orthogonal and
present the whole time; recalibrating `target_var` (previous section) just
made the wrongly-directed branch fire on a realistic schedule instead of
firing unconditionally, which is what made the effect visible on GPT-2 (its
batch-energy variance apparently sits above target more persistently than the
DAG GNN's, so $\beta$ climbed and stayed high, over-exploiting).

**Which direction is actually "correct" is not derivable from the code** — it's
a genuine design choice between two defensible annealing strategies:
- *reheat on collapse*: low variance (policy converged/homogeneous) → raise
  $T$ (lower $\beta$) to escape a possible local optimum.
- *sharpen on noise, relax on stability* (what the code did): high variance
  (noisy/unreliable batch) → sharpen toward the current best guess; low
  variance (stable) → relax.

Asked the user to choose. **Decision: drop `VarBasedScheduler` as the default
rather than pick a direction.** A single per-iteration variance estimate from
only 10 samples is a weak, noisy signal either way, with no coupling to
training progress and no ceiling — fragile regardless of which direction is
"correct". `configs/trainer/default.yaml` now uses `DefaultScheduler`: a
deterministic linear ramp on $\beta$ (`start=0.5, delta=0.02`, reproducing the
ramp already observed empirically), with no dependence on batch statistics and
therefore no way to mis-fire in the wrong direction.

`VarBasedScheduler` is left in `scheduler.py` (docstring corrected to describe
what the code actually does — code direction was *not* changed) for anyone who
wants to revisit the variance-adaptive idea later with a settled direction and
a properly calibrated, per-molecule `target_var`.

---

## Result: canonical masking — INCONCLUSIVE, not negative

An earlier 30-iteration mask/nomask ablation showed no difference, and
`configs/model/dag_gnn.yaml` was set to `canonical_masking: false` on that basis.
**That verdict was premature — the ablation was underpowered.**

`GQE-optimized/energy/mean` and `/energy/min` are **flat over all 30 epochs in
both arms**: the policy never improves. You cannot compare convergence speed
between two arms when neither converges. The arms matched because both were
sampling from a policy still at initialisation.

Why nothing trains in 30 iterations:
- `lr=5e-6`, 30 epochs × 30 steps = **900 AdamW steps**. AdamW's step is ~`lr`
  regardless of gradient scale → cumulative weight movement ≈ 4.5e-3, against an
  `nn.Linear(128,154)` head initialised to ~U(−0.088, 0.088). The head can move
  ~5% from init.
- Logits are `-temperature * operator_head(pooled)` with `inv_temperature ≈ 0.55`
  → scaled-logit std ≈ 0.55 over **154** operators → a near-flat softmax.
  The policy is close to uniform random for the whole run.

Consistent with the paper regime: optimisation is *"still improving at iteration
100 and has not fully converged"*.

**Implications, both uncomfortable:**

1. **The 30-iter model comparisons may be meaningless.** The
   `n2-L10-policy-comparison` group (GPT-2 vs diffusion vs GNN vs DAG GNN at 30
   iters) plausibly measured *random sampling*, not architectures. "DAG GNN ≈
   GPT-2" may just mean "both draw random circuits". Re-check `energy/mean` on
   any 30-iter run before citing it.
2. **Good energies with a policy that never learns ⇒ QSCI is doing the work.**
   Symmetry completion + local/global GEVP refinement over an accumulating
   determinant pool is strong enough that random circuits give good energies.
   Cf. the shot-saturation note above. Worth establishing how much the generative
   model contributes *at all* before investing further in the policy side.

The a-priori argument against masking still stands independently: the search
space is ~154¹⁰ ≈ 8e21 and the budget is 10×100 = 1000 circuits, so equivalence
classes are never revisited and deduplicating them cannot pay. But that is an
argument, not a measurement — the 100-iter re-run is the actual test.

**Sanity check to run first:** look at `energy/mean` for an existing 100-iter
`n2_l10_gpt2_paper` run. If it is *also* flat, the problem is the training setup
(lr, sample budget), not any model or the masking, and that is what to fix first.

---

## Redesign proposal: the circuit as a trace (commutation-aware DAG)

> ### Result: canonical masking (NEGATIVE) — read this first
>
> Change **(B)** below (canonical-form action masking) was implemented, verified
> correct by brute force, and run as an A/B on N2 (L=10, 30 iters, shots=1000,
> `n2-L10-canonical-masking-ablation`). **No detectable effect** on
> `Global-refined(best_so_far)/energy` or on convergence speed.
>
> **Why, in hindsight.** The 57 % figure is a *pairwise* statistic; the optimizer
> never enumerates pairs. It samples ~300 circuits from a space of
> 154^10 ≈ 1e22. Collapsing that space even a millionfold leaves 1e16 — the policy
> still visits 300 points. Two equivalent circuits are astronomically unlikely to
> both be sampled, so no QSCI evaluation was ever wasted on a duplicate.
>
> More fundamentally: action-space redundancy hurts **search** (enumeration, tree
> search, anything that revisits states). It is largely benign for **policy
> gradient**. Equivalent orderings get identical rewards and push the policy in
> consistent directions; the probability mass spread across them is partitioned,
> not wasted. The model never had to *discover* that two orderings are equivalent
> — only that both are good.
>
> **Methodological lesson:** ask "what does the optimizer actually visit?" before
> "how much of the space is redundant?".
>
> **Scope.** This tested (B), the *action space*. Change (A), the DAG's *edge
> relation*, was never implemented or tested. (A)'s main claimed benefit was the
> same redundancy argument, so its prior drops accordingly; its independent
> benefit (message passing over the true dependency structure as an inductive
> bias) remains speculative and is expensive to try — the frontier semantics must
> be redesigned. **Not currently recommended.**
>
> **Strength of evidence.** One seed, 30 iterations, GRPO advantages from 10
> rollouts — noisy. This is "no effect detectable at the scale we can afford",
> not a proof of no effect. Enough to deprioritise, not to close the question.
> The reduction factor grows with L, so this may still matter at much larger
> circuit depths.
>
> The masking code is retained (`model.canonical_masking`, **default false**)
> since it is correct and cheap. Everything below is the original proposal, kept
> for the reasoning and the measurements.

### The problem, measured

The DAG GNN chains any two gates that **share a qubit**. But the majority of
those pairs *commute* — swapping them yields the identical unitary, state,
energy and reward, so the ordering edge is physically meaningless.

Measured with `smoke_features.py <molecule>`:

| | N2 (16q, V=154) | H2O (12q, V=43) |
|---|---:|---:|
| total operator pairs | 11 628 | 861 |
| disjoint support (no edge — correct) | 33.3 % | 22.9 % |
| share qubit, **anticommute** (edge is real) | 28.5 % | 36.0 % |
| share qubit, **commute** (edge is **spurious**) | **38.2 %** | **41.1 %** |
| → spurious as share of shared-qubit pairs | **57 %** | **53 %** |

Two molecules of different symmetry (D∞h vs C2v), size, pool size and
correlation regime (N2 is at 2.5 Å, strongly multireference, CCSD amplitudes
> 1; H2O is at equilibrium, amplitudes < 0.16) give the same answer. **The
spurious-edge problem is structural**, following from how Jordan-Wigner
excitation operators overlap on qubits — not an N2 artifact.

The DAG GNN's headline claim was that equivalent circuit orderings collapse into
one object. In fact it only collapses the **disjoint-support** case (33.3 %),
while imposing spurious order on a *larger* fraction (38.2 %) than it legitimately
orders (28.5 %). So the inductive bias it was built to supply is largely not
being delivered — that part stands.

~~This is a plausible explanation for why the DAG GNN performs merely comparable
to GPT-2 rather than better.~~ **Disproved.** Removing the redundancy from the
action space (change (B)) changed nothing measurable, so the redundancy is not
what limits the DAG GNN. Why it merely matches GPT-2 — at 13x fewer parameters,
which is itself a respectable result — remains open, but it is not this.

### The right abstraction: a trace monoid

A circuit is a word over the alphabet of pool operators. Two adjacent letters may
be swapped iff their Pauli strings **commute**. This is exactly a **Mazurkiewicz
trace monoid**: an alphabet plus an independence relation (here: commutation).

- Two Pauli strings commute iff they anticommute on an **even** number of qubits.
- Words related by swaps of adjacent independent letters denote the same unitary.
- Each equivalence class (a *trace*) has a canonical **dependence graph** — the
  DAG whose edges are the *anticommuting* pairs, in placement order.
- Any two linear extensions of that DAG are related by adjacent-commuting swaps,
  hence give the **same unitary**. So the anticommutation DAG is a *faithful*
  canonical representation of the circuit.

This gives a principled replacement for the qubit-wire DAG, whose edge relation
(shares-a-qubit) is strictly coarser than the true dependence relation
(anticommutes).

### Two separable changes

**(A) State representation — commutation-aware edges.**
Precompute a `(V, V)` boolean commutation matrix once from the Pauli words
(`get_pauli_words()`; cheap — 154² for N2), `register_buffer` it. When placing
gate `t` with operator `k`, add a directed edge `s → t` from every previously
placed gate `s` whose operator **anticommutes** with `k`. Gates that commute with
everything become isolated nodes.

This makes message passing reflect the true causal structure. It changes *how the
state is encoded*, but by itself it does **not** shrink the search space.

**(B) Action space — canonical-form masking.** *(implemented)*
This is where the ~40 % payoff actually lives. Trace theory: every equivalence
class has a unique lexicographically-minimal representative. At generation step
`t` with prefix `w[0..t-1]`, operator `k` is forbidden iff it can be migrated
left onto a position holding a strictly larger operator:

```
exists i:  (for all j in [i, t-1]:  commutes(k, w[j]))   and   k < w[i]
```

Implemented as a backward scan from `t-1`, stopping at the first letter `k` does
not commute with (it is blocked there). This admits exactly one word per
equivalence class: every distinct unitary stays reachable, each reachable exactly
once. The commuting redundancy is removed *by construction*, with no loss of
expressiveness.

> **The adjacent-only rule is WRONG.** An earlier version of this note proposed
> the purely local condition `commutes(k, prev_op) and k < prev_op`. That is
> insufficient: a letter can hop left over a *run* of commuting letters. With
> prefix `(2,0,0)` where `1` commutes with both `0` and `2`, the word `(2,0,0,1)`
> passes the adjacent-only test, yet `1` migrates all the way to the front giving
> the smaller equivalent `(1,2,0,0)` — two survivors in one class. Brute-force
> enumeration of equivalence classes (including 20 random symmetric commutation
> matrices) confirms the backward-scan rule gives exactly one survivor per class
> and the adjacent-only rule does not.

Caveats:
- The mask depends only on the prefix, so `log_prob` can reconstruct it exactly.
  It **must** be applied identically in `sample_sequence` and `log_prob`, or the
  GRPO importance ratio is computed against a different distribution than the one
  sampled from.
- **Apply the mask after the `-temperature` multiply.** The code samples from
  `Categorical(logits=-temperature * logits)`; masking with `-inf` beforehand
  becomes `+inf` there, making forbidden operators maximally *likely*.
- **The entropy term is a NaN-gradient trap.** Masked entries give `p = 0` and
  `log p = -inf`, so `p·log p` is `NaN`. Selecting it away *after* the multiply —
  `torch.where(p > 0, p * logp, 0)` — gives the right forward value but still
  NaNs the backward: the product is computed regardless, and
  `d(p·logp)/dp = logp = -inf`, which times the incoming zero gradient is `NaN`.
  Worse, `log_softmax`'s backward couples all positions through the normaliser,
  so the NaN spreads to the *unmasked* logits too, poisons `operator_head`, and
  the next rollout samples from an all-NaN distribution. Replace `-inf` **before**
  any arithmetic touches it:
  ```python
  probs = log_probs.exp()                       # exactly 0 at masked entries
  safe  = torch.where(torch.isfinite(log_probs), log_probs, zeros)
  entropy = -(probs * safe).sum(-1)
  ```
  Same trap will apply to GPT-2 / diffusion if masking is ported to them.
- The identity operator commutes with everything and has the smallest index, so
  it can only appear *before* any real gate. Since it is a no-op, the reachable
  unitaries are unchanged.
- The mask can never forbid everything: the largest operator index can never
  satisfy `k < w[i]`, so it always survives.
- Turning masking on invalidates replay buffers and checkpoints from unmasked
  runs (they contain non-canonical sequences, which now score `-inf`). `log_prob`
  raises rather than silently producing `NaN` loss. Use a fresh `exp_tag` and
  `trainer.load_checkpoint=false`.

Cost: `O(V · L)` per step, `O(V · L²)` per circuit — negligible (V=154, L=10).

### Design question this forces

The current `frontier[q]` is "the last gate on wire q", and `_advance_dag`
consumes wires. Under (A) there are no wires — so *frontier* must be redefined,
e.g. as the set of gates with no outgoing anticommutation edge (the maximal
elements of the partial order), pooled to form the query.

An attractive hybrid keeps both relations as **typed edges**:
- directed **anticommutation** edges → causal / ordering structure
- undirected **qubit-sharing** edges → locality / information flow

with a separate `GATConv` per edge type whose outputs are summed. This preserves
the qubit-locality signal the current model gets, without letting it masquerade as
an ordering constraint.

### Complementary cleanup: the pool itself

`only_use_first_pauli: true` makes each pool element an arbitrary *fragment* of an
excitation generator, and `generate_excitations()` emits duplicate index orderings
of the same excitation. Switching to the `excitation` pool (complete excitations,
`ExcitationPool`) — or de-duplicating fragments — removes this ambiguity at the
source, and would shrink V considerably. Worth evaluating alongside (A)/(B).

### Suggested order — superseded by the negative result

1. ~~Measure the commutation statistics on a second molecule.~~ Done: N2 57 %,
   H2O 53 % of shared-qubit pairs. Structural, not an artifact.
2. ~~Implement **(B) alone** on the existing qubit-wire DAG.~~ Done. **No
   measurable effect** (see the box at the top of this section).
3. ~~If (B) helps, implement **(A)**.~~ (B) did not help, so **(A) is not
   recommended** — it is expensive (frontier redesign) and its main rationale
   just failed empirically.
4. Revisit the pool (fragments vs excitations) — still open, still cheap, and
   independent of all of the above. `only_use_first_pauli: true` makes each pool
   element an arbitrary generator *fragment*, and `generate_excitations()` emits
   the same excitation under several index orderings.
5. **Resume the cross-molecule phases** (Phase 1 onward). Phase 0 is complete and
   the commutation work touched none of it.

---

## Cross-molecule generalization: feature-based operator menu

### Goal

Train one policy across many molecules so it can generate a good ground-state
circuit for a *new* molecule with no retraining (zero-shot). This requires the
policy to stop depending on molecule-specific integer operator IDs and instead
reason about operators by their **physical features**, which live in the same
coordinate space for every molecule.

### Why features instead of integer IDs

The current output head is `Linear(hidden, vocab_size)` — a fixed classifier
over one molecule's ~80 operators, addressed by index. Operator "#7" is an
opaque ID whose meaning is baked into a weight column fit to that molecule. Hand
the trained network a new molecule and (a) the vocab size is wrong and (b) index
7 means a different physical thing, so the network cannot even be applied.

The fix is a **feature-based scorer**: each operator is described by a fixed-length
feature vector, encoded by a shared MLP into keys, and scored by dot product
against a query from the policy state:

```
logits[k] = query · op_encoder(operator_features[k])
```

Because operators are described by physics (not ID), a rule learned on one
molecule ("early in a circuit, favour high-amplitude low-gap doubles") is phrased
in feature space and transfers to any molecule. The number of operators V can
vary freely because logits are computed one-per-operator from features. See the
DAG GNN implementation notes and the `get_operator_features()` plan below.

### The feature vector (~10 dims)

Each menu row describes one operator. Non-redundant recommended set:

| # | Feature | What it means | Source in code | Normalize? |
|---|---------|---------------|----------------|-----------|
| 1 | **arity** | Single vs double excitation (1 or 2). Doubles dominate the correlation energy; singles are often near-negligible (Brillouin). | `len(footprint)//2`, or excitation-tuple length in `generate_excitations()` | no |
| 2 | **ε_occ** | Mean energy of the occupied orbital(s) the excitation removes electrons from. Deep vs shallow hole. | `hf.mo_energy[active_indices]` (new accessor `active_mo_energy`) | yes — reference to HOMO |
| 3 | **ε_virt** | Mean energy of the virtual orbital(s) the excitation promotes electrons into. Low- vs high-lying particle. | same | yes — reference to HOMO |
| 4 | **gap** Δε | ε_virt − ε_occ, the excitation energy. Small-gap excitations are usually the important, strongly-correlated ones. | derived from `mo_energy` | naturally relative |
| 5 | **\|θ\|** | CCSD amplitude magnitude — the classical importance of this operator in the coupled-cluster wavefunction. Already the pool's coefficient. | value in `generate_excitations()` dict / `g.parameter` | intensive |
| 6 | **Hamiltonian coupling** ⟨ij‖ab⟩ | The literal Hamiltonian matrix element connecting the HF reference to the state this operator creates: ⟨Φ₀\|H\|Φ_ij^ab⟩ = antisymmetrized two-electron integral. This is *per-operator Hamiltonian information* — it merges the operator-feature and Hamiltonian-encoding goals. Captures Brillouin's theorem automatically (vanishes for singles). | `cas_hamiltonian.h2` (ERIs, already built at `molecule.py:34`) | intensive |
| 7 | **MP2 pair energy** | \|⟨ij‖ab⟩\|² / Δε — the second-order perturbative energy contribution of this excitation. A cheap, physically-grounded "how much does this operator matter" score, complementary to the CCSD amplitude. | `h2` + `mo_energy` | intensive |
| 8 | **ladder_span** | `max(xy) − min(xy) + 1`: how many qubits the Jordan-Wigner string stretches across. The real non-locality signal. **Not** a plain footprint size — with `remove_z_ladder: true` (the default) the footprint is exactly `2 × arity`, i.e. carries zero information. The span recovers the extent the Z-ladder *would* have covered. | derived from `get_xy_qubit_footprints()` | no |
| 9 | **n_y** | Number of Y Paulis. Kept because it drives `gate_cost` (each Y costs an s + sdg pair), i.e. it is a *cost* signal, not an identity code. | `get_pauli_words()` | no |
| 10 | **gate cost** | Compiled primitive-gate count (cx+h+s+sdg+rz) for the Pauli-evolution operator — a hardware-cost / circuit-depth signal, free to compute. Partly redundant with (arity, n_y) but kept as the explicit hook for the hardware-aware extension. | `get_pauli_evolution_gate_count()` (`utils.py:21`) | optional scale |
| 11 | **spin pattern** | αα / ββ / αβ character of the excitation, from the parity of the qubit indices (even=α, odd=β). Distinguishes same-spin from opposite-spin excitations, which have different exchange behaviour. | parity of footprint qubits; see `generate_excitations()` lines 96–106 | no (categorical) |

**Deliberately excluded: the *positions* of the Y Paulis** (a `y_slot_0..3`
one-hot). It would make every operator's row unique, but it encodes no physics —
it is a disguised operator ID, exactly the thing the feature menu exists to
eliminate. It would let the model memorise which generator fragment happens to
work for one molecule and would not transfer. See below.

### Duplicate rows are expected — do not chase zero

An early version of the smoke test asserted "no two operators share a feature
row". **That target is wrong.** The menu encodes physics; two operators that
agree on every physical column *should* collide and receive equal logits.

Two distinct sources of collisions in the N2 pool (48 duplicate rows):

**1. Commuting fragments of one excitation generator.** All Pauli strings in the
JW decomposition of a fermionic excitation mutually commute — that is why
`exp(θ·Σₖ Pₖ)` factorises into `∏ₖ exp(θPₖ)`. With `only_use_first_pauli: true`
the pool stores individual fragments, and several can survive with the same
footprint. Example (footprint `[6,8,10,12]`, all α, orbitals 3,4 → 5,6):

```
op  4:  Y Y Y X      op  7:  X X Y X
op 41:  Y X Y Y      op 44:  Y X X X
```
Every pair differs at exactly 2 positions → even → **all four mutually commute.**
They have identical energy, coupling, and (for 4/41 and 7/44) identical gate
cost. Nothing physical distinguishes them, so equal logits is the correct
behaviour. `n_y` still separates 4 from 7 — legitimately, since their gate costs
genuinely differ (21 vs 17).

**2. Symmetry-equivalent excitations under orbital degeneracy.** With orbitals
3,4 (1π_u) and 5,6 (1π_g) exactly degenerate, the excitations 3→5 and 4→6 have
identical energies, gaps and couplings. They are related by the symmetry that
causes the degeneracy, and identical feature rows are again *correct* — the model
should not prefer one over the other a priori.

**Confirmed by the H2O control.** H2O is bent (C2v) and has *no* degenerate
valence orbitals. Its duplicate count collapses from N2's 48 to 8, and the
classifier reports **0** degeneracy-symmetry groups — all 8 remaining groups share
a single footprint and are mutually commuting generator fragments. So source (2)
vanishes exactly when the degeneracy does, and source (1) is the irreducible
residue present in every molecule. Run:
`python3 smoke_features.py h2o` vs `python3 smoke_features.py n2`.

The pool also contains genuinely **redundant entries**: `generate_excitations()`
emits the same physical excitation under different index orderings (e.g.
`(12,6,10,8)` and `(10,8,12,6)` both appear with angle 0.98383). Those are better
removed at the pool level than papered over with an identity feature.

### Pool redundancy — measured, and the `dedup_excitations` cleanup

Root cause (two stages):
1. The CCSD tensor obeys `t_ijab = t_jiba`, so the same excitation appears under
   two index orderings. `make_excitation_gate` reads the tuple as *pairs*, so
   `(12,6,10,8)` and `(10,8,12,6)` are the identical excitation, same angle.
2. Both gates share a generator, so in `build_operator_pool` the second gate's
   first Pauli string is already in `seen`; the `continue` sends it to its
   *second* Pauli string, which gets appended. A bookkeeping duplicate thus
   silently becomes a **different Pauli fragment** in the pool (this is the origin
   of the N2 ops 4/7/41/44 on footprint [6,8,10,12]).

`operator_pool.dedup_excitations` (default false) drops excitations that are
pair-order permutations of one already emitted (largest |amplitude| kept). It
removes operators — distinct Pauli rotations — that exist only as artifacts of
the duplicate listing, not just relabels them.

Measured (smoke_dedup.py):

| | V off→on | dup feature rows off→on | distinct footprints |
|---|---|---|---|
| N2 | 154 → 118 (−36) | 68 → 42 | 99 → 99 (unchanged) |
| H2O | 43 → 35 (−8) | 8 → 2 | 30 → 30 (unchanged) |

Distinct footprints are unchanged in both: dedup removes only redundant
*spellings*, never a distinct qubit-support. The merged pairs carry identical
angles, so nothing is lost. The **residual** duplicate rows are legitimate and
must not be chased to zero: H2O's 2 are commuting generator fragments; N2's 42
are dominated by its π-orbital degeneracy (3→5 vs 4→6 etc. — different
excitations, equal features by symmetry). H2O has no degeneracy, which is why it
nearly reaches zero.

Two bugs fixed alongside this:
- `ExcitationPool.__init__` called `super().__init__(molecule, params)` without
  `threshold`, silently ignoring the configured `ccsd_threshold` (used 1e-8).
- `get_pauli_words()` / `get_commutation_matrix()` read only an operator's first
  term — exact for `PauliEvolutionPool`, approximate for `ExcitationPool` (sums).
  Documented, not yet generalized.

### Feature explanations in depth

- **arity** — the single most structural distinction. Under Brillouin's theorem a
  single excitation does not couple directly to the HF reference, so doubles carry
  the bulk of the correlation. Letting the model know arity up front is a strong
  prior.
- **ε_occ / ε_virt** — orbital energies locate the excitation on the energy axis.
  They must be normalized (e.g. subtract the HOMO energy) because absolute orbital
  energies shift arbitrarily between molecules and basis sets; only *relative*
  positions are comparable across molecules.
- **gap** — the excitation energy Δε is the denominator in every perturbative
  importance estimate. Small gaps → strong correlation → operators the model
  should favour. Naturally intensive, so it transfers without normalization.
- **|θ|** — the CCSD amplitude already used to build the pool. It is the classical
  estimate of the operator's weight in the true wavefunction; the single best
  scalar "importance" signal we have from a cheap classical method.
- **Hamiltonian coupling ⟨ij‖ab⟩** — the most physically meaningful addition.
  It is an actual entry of the molecular Hamiltonian in the excitation basis, so
  it tells the model directly how the Hamiltonian links the reference to that
  excited configuration. Because it is computed from `h2`, it makes the menu
  carry Hamiltonian information per operator — the conditioning ("which molecule")
  and the action description ("which operator") share representation.
- **MP2 pair energy** — a rung above |θ| in rigor: the exact second-order energy
  lowering from this excitation. Complements the CCSD amplitude and is essentially
  free once ⟨ij‖ab⟩ and Δε are in hand.
- **locality / gate cost** — the two "circuit economy" features. They let the
  model trade chemical usefulness against implementation cost, and are the natural
  hooks for the hardware-aware extension (see "Hardware-aware features" above).
- **spin pattern** — same-spin (αα/ββ) and opposite-spin (αβ) excitations have
  different physics (exchange vs pure Coulomb); encoding the pattern lets the model
  distinguish them rather than treating all doubles identically.

### Reading the menu: degeneracy is not a bug

When inspecting the N2 feature matrix, the highest-amplitude rows show
**identical** `eps_occ` (≈ 0), `eps_virt` (0.2233) and `gap` (0.4466). This is
correct physics, not a broken mapping. N2's active-space orbital energies are

```
[-0.905, -0.849, -0.2244, -0.1787, -0.1787,  0.0446,  0.0446,  0.1055]
                            ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^
                         orbitals 3,4 (1πu)  orbitals 5,6 (1πg)
```

Orbitals 3/4 (the HOMO, doubly-degenerate 1π<sub>u</sub>) and 5/6 (the LUMO,
doubly-degenerate 1π<sub>g</sub>) are exactly degenerate. Since
`generate_excitations()` sorts by |amplitude| descending and N2's dominant
correlation is π→π\*, the top rows are all excitations among these degenerate
orbitals — so their energy-derived features necessarily coincide. Over the whole
pool the features do spread out (for N2: 100 distinct footprints, 33 distinct
(occ_orbs, virt_orbs) pairs across 154 operators).

Verified at the same time: the spin-orbital → spatial-orbital convention is
`spatial = qubit // 2` (interleaved, even = α, odd = β), matching
`generate_excitations()`'s `2*key`/`2*key+1` construction.

### Data sources summary

Everything above is already computed inside `molecule.py` / `operator_pool.py` /
`utils.py`; nothing needs a new PySCF calculation:

- **`generate_excitations(threshold)`** (`operator_pool.py:81`) — excitation
  structure (arity, orbitals) + amplitude θ.
- **`get_qubit_footprints()`** (`operator_pool.py:62`) — qubit support, aligned to
  `self.pool` by construction.
- **`get_pauli_evolution_gate_count()`** (`utils.py:21`) — compiled gate cost.
- **`cas_hamiltonian.h1` / `.h2`** (`molecule.py:34`) — one- and two-electron
  integrals for the coupling and MP2 features.
- **`hf.mo_energy` / `hf.mo_occ`** — orbital energies and occupancies (need new
  one-line accessors `active_mo_energy` / `active_mo_occ`).

### Implementation notes

- **New method** `get_operator_features() -> Tensor (V, feat_dim)` in
  `UCCSDBasedPool`, modelled on `get_qubit_footprints()` so rows stay aligned to
  `self.pool`.
- **Alignment gotcha:** `build_operator_pool()` prepends an identity (index 0),
  dedups paulistrings via the `seen` set, and breaks after the first paulistring
  when `only_use_first_pauli=True`. So `screened_indices` ordering is *not* 1:1
  with pool ordering. Build features by iterating `get_qubit_footprints()` (which
  iterates `self.pool`) rather than re-iterating the excitation dict, and **thread
  the amplitude θ through `build_operator_pool`** (one parallel list) rather than
  reverse-engineering it from the SpinOperator coefficient.
- **Normalization:** subtract the HOMO energy from ε_occ/ε_virt; leave gap,
  coupling, MP2, |θ| as-is (already intensive); gate cost/locality are absolute
  integers.
- **Model change** (per architecture): replace the fixed head with the
  `op_encoder(features)` scorer. **Every model needs the output head swapped;
  most also embed operator IDs on the input side** — see the touch-point table
  in "Porting the scorer to the other models" below.
- **Identity operator** (pool index 0) gets an all-zero / sentinel feature row.
- **Molecule context** (conditioning branch): an amplitude-weighted average of the
  operator keys, passed through a small MLP and added to the query. No separate
  Hamiltonian-graph encoder needed for v1 — feature #6 already injects per-operator
  Hamiltonian information.

### Phased implementation plan

Sequential, DAG GNN first. The heavy lifting (featurization, scorer module,
multi-molecule loop) is model-agnostic and shared; GPT-2/diffusion become pure
head swaps at the end. Each phase has a gate that must hold before moving on.

**Phase 0 — shared plumbing** *(done)*
- `molecule.py`: `active_mo_energy` / `active_mo_occ` accessors.
- `operator_pool.py`: `_pool_amplitudes` threaded through both
  `build_operator_pool` implementations (parallel list, appended exactly where
  operators are appended so dedup / `only_use_first_pauli` cannot misalign it).
- `operator_pool.py`: `get_xy_qubit_footprints()` (X/Y qubits = the excitation's
  spin-orbitals under JW, robust to `remove_z_ladder`), `_hf_coupling()`, and
  `get_operator_features()` returning the (V, 10) float32 menu.
- No behavior change; nothing consumes the features yet.

**Phase 1 — DAG GNN feature scorer, single molecule (N2)** *(DONE — gate
PASSED: featscorer ≈ integer-ID baseline v2 on all metrics, 30 iters, deduped
pool, 1 seed)*
- `gqe_qsci/gqe/models/operator_scorer.py`: `OperatorScorer` —
  `logits = query @ op_encoder(features).T / sqrt(H)`. Columns are z-scored per
  menu at construction (raw scales differ by 200x: gate_cost ~20 vs amplitude
  ~0.1 — unnormalized, the largest column would dominate the encoder).
  `set_features()` swaps menus in place, re-registering the buffer when V
  changes (ready for Phase 2). The 1/sqrt(H) keeps init logit scale comparable
  to the Linear head (measured 0.34x; VarBasedScheduler absorbs the rest).
- `dag_gnn.py`: `feature_scorer: true` selects OperatorScorer as
  `operator_head`; `_scaled_logits` is the single consumer, so masking,
  sampling and log_prob are untouched. Factory passes
  `pool.get_operator_features()` unconditionally.
- Configs: `model=dag_gnn_features`,
  `experiment=n2_l10_dag_gnn_30iter_featscorer` (dedup ON, V=118, W&B group
  `n2-L10-dedup-ablation` so it overlays the integer-ID dedup_on baseline —
  same pool, same settings, only the head differs).
- Verified: z-scoring exact; identical feature rows → identical logits
  (structural); distinct rows → distinct logits; gradients flow; menu swap
  V=118→43 works.
- **Gate:** must match the integer-ID DAG GNN baseline
  (`n2_l10_dag_gnn_30iter_dedup_on`) on `Global-refined(best_so_far)/energy`.
  A feature scorer has less raw capacity than a free `Linear(H, V)` (no free
  column per operator; equal-feature operators get equal logits); if it cannot
  match on a single molecule, suspect normalization or op_encoder capacity
  before concluding the approach fails.

**Phase 2 — multi-molecule training loop (infrastructure)**

Step 1 — molecule-set config *(done)*
- `configs/molecule_set/n2_bond_scan.yaml`: `base` linear-chain molecule +
  `train_bond_lengths` / `eval_bond_lengths`. Registered as an optional config
  group (`- optional molecule_set: null` in default.yaml); null keeps the
  single-molecule path unchanged. Select with `molecule_set=n2_bond_scan`.

Step 2 — per-molecule bundles in the factory *(done)*
- `MoleculeBundle` dataclass: molecule, pool, qsci_pipeline, vocab_size,
  n_qubits, operator_features, orbital_features, qubit_footprints,
  commutation_matrix.
- `Factory.create_molecule_bundles(cfg) -> dict[name, MoleculeBundle]` builds a
  fresh molecule/pool/pipeline + feature tensors per entry, ordered train-then-
  eval. Refactored `_make_pool` / `_make_qsci_pipeline` are shared with the
  single-molecule methods; the singleton caches are NOT touched, so
  single-molecule runs are unaffected. Guards feat_dim consistency across
  bundles.
- `_expand_molecule_set` currently handles the bond-scan form; a general
  `molecules: [...]` list form (Phase 4, heterogeneous) slots in there.

Step 3 — `set_molecule()` on the policy *(DONE — swap verified for GPT-2,
SingleShot, Absorbing; DAG GNN / GNN-absorbing share the code path, need Docker
to run)*
- Base `Policy.set_molecule(bundle)` raises NotImplementedError (loud failure on
  unsupported models). Each feature_scorer model overrides it.
- All swaps reassign registered buffers (not `copy_`), so changing V between
  molecules works — `nn.Module.__setattr__` routes a tensor assignment back into
  `_buffers`. Verified V=40→25 swap: output vocab dim tracks new V, sampled ids
  stay in range, [MASK] token moves, shared scorer updates, GPT-2 wte/lm_head
  stay tied.
- DAG GNN swaps `_fp_flat`, `commutes`, `orbital_features`, `vocab_size`,
  `n_qubits`, `UNPLACED`, and the scorer features. Diffusion/GNN swap the scorer
  features + `_refresh_mask_token()`. GPT-2 swaps the one shared scorer.
- Uses `update_stats=False` (frozen normalization); step 5 pre-loads global
  stats before training.
- (Original design notes retained below.)
- The model is created ONCE (shared weights). A molecule swap must atomically
  replace every per-molecule buffer, not just the operator menu:
  - `operator_head` (OperatorScorer): `set_features(bundle.operator_features,
    update_stats=False)` — frozen stats (step 5).
  - DAG GNN only: `_fp_flat` (drives DAG edge construction), `commutes` (drives
    canonical masking), `orbital_features` (qubit-wire node inputs), and
    `vocab_size` / `UNPLACED`. These are currently `__init__`-time buffers.
  - `self.vocab_size` on the policy (used by `_canonical_mask`, `_node_table`).
- Signature: `policy.set_molecule(bundle)`; each policy class implements it
  (base no-op raises NotImplementedError). Atomicity matters — a half-swapped
  model (menu from A, footprints from B) silently corrupts generation, so swap
  all buffers together and never mid-rollout.
- Shape-changing buffers: V differs across bundles, so `register_buffer`
  replacements must re-register (assign `self.buf = new`), not `copy_` — the
  same pattern OperatorScorer.set_features already uses when V changes.
- Cost note: `keys()` / feature encoding is recomputed per forward anyway, so a
  swap is just pointer reassignment + (for the DAG GNN) rebuilding the small
  footprint/commutation buffers. Cheap; do it at rollout-group boundaries.

Step 4 — the multi-molecule loop *(DONE — Docker-only to run)*
- `TrainPipeline` branches on `config.molecule_set`. Multi mode: builds bundles,
  creates the model ONCE from the first training bundle, round-robins molecules
  one-per-epoch. `_activate_bundle` atomically swaps model buffers + QSCI
  pipeline + logger reference energies; called in `on_fit_start` (warmup) and
  `on_train_epoch_start` before `collect_rollout`, so the whole epoch (rollout +
  30 gradient steps) runs with one molecule installed.
- Correctness guardrails: `buffer_size == num_samples` asserted, so the replay
  buffer holds exactly one molecule's group and never mixes molecules across the
  epoch's gradient steps (GRPO advantages are batch-relative). Best-so-far
  trackers are keyed by molecule name (`self._best[name]`), so -7.9 vs -107 Ha
  molecules never compare. Each bundle owns its QSCI pipeline, so the global-
  refinement CI accumulation stays per-molecule. Metrics are namespaced
  `<molecule>/...`.
- Known limitations: shared temperature scheduler across molecules (fine for the
  bond scan where energy scales match; per-molecule scheduler state is a future
  option). Multi-molecule checkpoint resume is NOT supported (model buffers are
  shape-changed per molecule; per-molecule refinement state isn't persisted) —
  use `load_checkpoint=false`.

Step 5 — global frozen normalization *(DONE)*
- `OperatorScorer.set_normalization(mean, std)`; `_init_multi_molecule` computes
  mean/std once over the pooled training features and installs them. `set_molecule`
  swaps raw features with `update_stats=False`, so `keys()` normalizes every
  molecule by the SAME fixed scale. Verified: the same physical operator gets
  identical normalized coords and identical encoded keys across molecules;
  per-molecule stats would break this.

Step 6 — zero-shot eval harness *(next)*: every N iters, `set_molecule` to a
held-out (`split == "eval"`) bundle, generate with no gradient update, log under
`zeroshot/<name>/...` vs its own CASCI. Bundles already carry the eval split
(`self.eval_bundles`); this is a logging/eval hook, no new infra.

**Phase 3 — bond-length scan (same molecule, different geometries)**
- First real experiment on the Phase 2 infra, and the cleanest generalization
  test: N2 at several bond lengths (e.g. r = 0.9 … 2.4 Å via the existing
  `linear_chain` geometry / `bond_length` config).
- The operator set is nearly identical across geometries — only the features
  change (amplitudes grow toward dissociation, gaps shrink, couplings shift).
  So this isolates *conditioning* from the vocabulary-portability problem.
- Train on a subset of bond lengths, zero-shot on held-out intermediate and
  extrapolated r. Compare against per-geometry from-scratch training curves.
- Config work: molecule config must accept a list of bond lengths; cache keys
  in `molecule.py` already hash geometry, so per-geometry caching works as-is.

**Phase 4 — heterogeneous molecules (different qubit counts)**
- Requires the remaining portability fix: qubit-wire node inputs in the DAG GNN
  become feature-based too (seed each wire with its orbital's energy/occupancy
  from `active_mo_energy`/`active_mo_occ` instead of an indexed embedding).
- Normalization matters here (HOMO-referenced energies); watch for feature-scale
  drift across molecules of very different size.
- Zero-shot test on a held-out molecule (e.g. train H2/LiH/H2O, test BeH2).

**Phase 5 — replicate to diffusion and GPT-2**
See "Porting the scorer to the other models" below for the verified per-model
touch points. Summary: the output head is a one-line swap everywhere; the input
side differs, and only `CircuitDiffusionModelSingleShot` needs no input change
at all.

---

## Porting the scorer to the other models

### Output side: a one-line swap everywhere

`OperatorScorer.forward` broadcasts: `(B, H) @ (H, V) -> (B, V)` for the DAG
GNN's pooled query, and `(B, L, H) @ (H, V) -> (B, L, V)` for the per-position
hidden states in `diffusion.py` / `gnn.py`. Verified to equal a per-position
loop, with gradients intact. So `self.output = nn.Linear(hidden_size,
vocab_size)` (diffusion.py:98, gnn.py:115) becomes
`self.output = OperatorScorer(features, hidden_size)` and the call site
`self.output(hidden)` is unchanged.

### Input side: verified touch points

**Correction.** Earlier notes claimed "DAG GNN and diffusion touch the vocabulary
once (output head); GPT-2 twice". That is wrong on two counts: absorbing
diffusion re-embeds revealed tokens, and the DAG GNN embeds placed operators via
`node_embedding`. Actual state:

| model | output head | embeds operator IDs as input? |
|---|---|---|
| `CircuitDiffusionModelSingleShot` | swap `self.output` | **No.** `_logits` only ever receives all-`[MASK]` (diffusion.py:509 sampling, :538 log_prob). Its `token_embedding` has V+1 rows of which only `mask_token` is ever used — the other V rows are dead parameters. |
| `CircuitDiffusionModelAbsorbing` | swap `self.output` | **Yes** — reveals tokens and feeds them back (diffusion.py:305, :413) |
| `CircuitGNNModelAbsorbing` | swap `self.output` | **Yes** — same pattern (gnn.py:303, :365) |
| `GPT2Model` | override `lm_head` | **Yes** — replays chosen tokens via `wte` (weight-tied to `lm_head`) |
| `CircuitDAGGNNPolicy` | **done** (Phase 1) | **Yes** — `node_embedding` maps `n_qubits + op_k` (dag_gnn.py:140, :203) |
| `CircuitDiffusionModelSimple` | swap `self.output` | Yes (legacy; not worth porting) |

### Implication: SingleShot is the cheapest cross-molecule vehicle

`CircuitDiffusionModelSingleShot` is the **only** model that is genuinely
output-only, so swapping `self.output` makes it fully cross-molecule in one line.
It is also T x faster at inference and has an exact (non-ELBO) `log_prob`. Worth
considering as the Phase 2 vehicle instead of the DAG GNN, whose remaining
advantage (inductive bias) was already dented by the canonical-masking null
result.

### Phase 1 gap (does not affect the gate)

The DAG GNN's `node_embedding` still embeds placed operators by integer ID. On a
single molecule that is fine, and the Phase 1 gate only tests whether
feature-based *scoring* works — so the gate remains valid. But it **blocks
Phase 2**: cross-molecule requires the input side to be feature-based too.

### The general input-side fix — DONE for all five models

All cases reuse the scorer's key matrix for embedding (GPT-2-style weight tying),
so input and output share one feature-derived matrix. Implemented via two helpers
in `operator_scorer.py`:

- `OperatorScorer.keys()` → `(V, H)` encoded operator keys, and
  `forward(query, keys=...)` to avoid encoding twice.
- `SpecialTokenEmbedding(n_special, H)`: real tokens → `keys[token]`, special
  tokens (`[MASK]`) → own learned vectors. Drop-in for the absorbing/GNN
  `token_embedding`.

Per model (all take `feature_scorer: bool`, `operator_features`):

| model | output | input side |
|---|---|---|
| `CircuitDiffusionModelSingleShot` | `OperatorScorer` | none needed (only ever sees `[MASK]`); `SpecialTokenEmbedding` present but only its mask vector is used |
| `CircuitDiffusionModelAbsorbing` / `Simple` | `OperatorScorer` | `SpecialTokenEmbedding`, keys shared with output |
| `CircuitGNNModelAbsorbing` | `OperatorScorer` | `SpecialTokenEmbedding`; graph is over positions, so no qubit-index embedding to fix |
| `GPT2Model` | `lm_head` → `_ScorerHead` | `wte` → `_FeatureWTE`, same scorer instance (tie preserved; `tie_word_embeddings=False` so HF doesn't touch `.weight`) |
| `CircuitDAGGNNPolicy` | `OperatorScorer` | `node_embedding` composed per-call in `_node_table()`: qubit wires → `qubit_encoder(get_orbital_features())`, gate nodes → scorer `keys()`, UNPLACED → learned vector |

New pool method `get_orbital_features()` → `(n_qubits, 3)` = HOMO-referenced
energy, occupancy, spin — the physical description of each qubit wire, replacing
the DAG GNN's per-qubit-index embedding (which cannot transfer: "qubit 5" is
different physics per molecule).

Configs: `model=<name>_features` for each. `feature_scorer` defaults to false
everywhere, so existing runs are unchanged.

Verified without Docker (GPT-2, SingleShot, Absorbing): forward/log_prob run, NO
operator-sized `nn.Embedding` survives, wte/lm_head share one scorer, menu swap to
a different V works, gradients reach the shared encoder. DAG GNN and GNN-absorbing
use the identical code path but need torch_geometric to instantiate (Docker).

### Normalization across molecules (Phase 2 knob)

`OperatorScorer.set_features(feats, update_stats=)`. Per-molecule stats
(`update_stats=True`) would make the SAME physical operator take DIFFERENT
normalized values in different molecules, partly defeating transfer. For Phase 2,
compute mean/std once over the training set and pass `update_stats=False` on every
swap, so normalization is a fixed physical scale. Orbital features are already O(1)
and comparable, so they are not normalized.

---

## Thesis TODO: make ELBO vs trajectory log-prob an ablatable flag

The DDPO change **replaced** the ELBO surrogate with the exact reverse-trajectory
log-probability rather than putting it behind a switch. The change is
theoretically well-motivated — the ELBO bounds `log p(x_0)`, whereas policy
gradient needs `log pi(action)`, and the trajectory log-prob *is* that (the DDPO
argument, Black et al. 2023, arXiv:2305.13301) — and it also removes the
mask-resampling noise that previously destabilised the GRPO importance ratio.

But because the old path is gone, **there is no experiment to point to** if a
reviewer asks "did the exact trajectory log-prob actually help?". Comparing
against the older W&B runs is confounded by everything else that changed since.

If this goes in the thesis, add:

```yaml
# configs/model/*.yaml
log_prob_mode: trajectory   # or: elbo
```

branching inside `CircuitDiffusionModelAbsorbing.log_prob` (and the GNN twin),
then run the A/B on N2 at fixed seed. The ELBO implementation is recoverable
from git history (the commit before `5786019`). Cheap to add; the only reason
not to do it now is that it is not on the critical path to the cross-molecule
result.

Note the two modes need different buffer contents (`reveal_step` vs the old
per-timestep `masks`), so the flag has to switch what `collect_rollout` stores
as well — that is the fiddly part, not the log_prob branch itself.

## Possible future modification: older rollout replay

The original setup keeps `num_samples`, `batch_size`, `warmup_size`, and
`buffer_size` equal to 10, so each rollout group is trained as one batch and the
buffer contains only the latest group.

A possible future experiment is to increase `buffer_size` above `num_samples`
to reuse older circuits:

```yaml
trainer:
  num_samples: 10
  batch_size: 10
  buffer_size: 100
```

This would make the training more replay-buffer/off-policy oriented, but it
should be done carefully because GRPO/GSPO advantages are batch-relative and the
current dataloader does not shuffle rollout groups.

---

## Thesis TODO: equal-weight feature normalization (ablation)

`_init_multi_molecule` pools the operator rows of every training bundle and
z-scores the concatenation:

```python
train_feats = np.concatenate([b.operator_features for b in self.train_bundles], axis=0)
scorer.set_normalization(train_feats.mean(axis=0, keepdims=True),
                         train_feats.std(axis=0, keepdims=True))
```

That is **operator-weighted**: a molecule contributes in proportion to its pool
size, so the largest pool sets the scale. Defensible (it standardizes the
operator distribution the policy actually sees during rollouts), but not the
only option.

Rejected alternative: pre-scaling each molecule's features by `1/V_m` before
concatenation. Pool size is a property of the *sample*, not of the operator, so
folding it into the feature vector either (a) computes stats on a distribution
the model never sees — `set_features` stores raw features and `keys()`
normalizes them at scoring time — or (b) makes the same physical operator land
at different normalized coordinates depending on how many other operators share
its pool, which is exactly the failure mode frozen stats exist to prevent.

The right fix, if equal molecule weighting is wanted, is to reweight the
*estimator*, not the data — mean of per-molecule means, plus the law of total
variance so the between-molecule spread is not lost:

```python
mus   = np.stack([b.operator_features.mean(axis=0) for b in self.train_bundles])   # (M, F)
vars_ = np.stack([b.operator_features.var(axis=0)  for b in self.train_bundles])   # (M, F)
mu    = mus.mean(axis=0, keepdims=True)                        # (1, F)
var   = (vars_ + (mus - mu) ** 2).mean(axis=0, keepdims=True)  # within + between
scorer.set_normalization(mu, np.sqrt(var))
```

Every molecule counts once regardless of pool size; the feature values
themselves are untouched, so the scale stays a fixed physical one.

Not worth switching the default blind. Equal weighting matters most when the
molecules are the unit of generalization and pool sizes are badly skewed —
arguably the Phase 4 case (LiH/H2O -> N2, where N2 is both the largest pool and
the held-out target, so the training set's largest molecule sets the scale).
Put it behind a config flag and run the A/B: does transfer error to N2 move?

Third option, addressing the underlying worry more directly: several columns
(`amplitude`, `mp2`, `coupling`) are heavy-tailed, so mean/std is dominated by a
few large-amplitude operators regardless of which molecule they came from.
Median/IQR or a per-column quantile transform would be more robust than any
reweighting of the mean.

Related: numpy's `std` (ddof=0) here vs torch's `.std` (ddof=1) in
`set_features` — ~0.1% apart at these pool sizes, but the two paths are not
bit-identical if a single-molecule run is ever compared against a one-molecule
`molecule_set`.

---

## Thesis TODO: revisit round-robin molecule scheduling

Multi-molecule training currently walks the training set with a bare pointer:

```python
def _next_train_bundle(self):
    b = self.train_bundles[self._rr % len(self.train_bundles)]
    self._rr += 1
    return b
```

One molecule per epoch, strict cyclic order, called from `on_train_epoch_start`
and from the warmup loop in `on_fit_start`. Wanted: something better. The
replacement is **not decided yet** — this note is here so the options are not
re-derived from scratch.

Why it may matter:

- **Deterministic period.** With M training molecules the schedule has period M,
  locked in phase with the epoch counter. Any other per-epoch cadence (LR
  schedule, temperature schedule, eval interval) that shares a factor with M
  aliases against it, so a given molecule can systematically land at the same
  temperature every time it comes up.
- **Uniform budget.** Every molecule gets identical rollout count regardless of
  how far from converged it is. Easy molecules keep consuming QSCI evaluations
  (the expensive part) after they have stopped improving.
- **One molecule per gradient group.** `buffer_size == num_samples` is asserted
  precisely so the buffer never mixes molecules, so each update is a
  single-molecule gradient. The policy sees a molecule-correlated gradient
  sequence rather than a mixed batch, which is the classic setup for
  catastrophic drift toward whichever molecule came last.

Candidate replacements, cheapest first:

1. **Shuffled epochs** — permute `train_bundles` once per pass, consume, reshuffle.
   Kills the phase-locking, ~4 lines, no other change. Do this first.
2. **Sampling proportional to something** — pool size, current error vs CASCI, or
   a bandit-style score on recent improvement. Spends the QSCI budget where the
   model is still losing. Needs a stopping/normalization rule or it starves the
   easy molecules entirely.
3. **Mixed-molecule batches** — drop the one-molecule-per-group constraint and
   collate rollouts from several bundles into one gradient step. Directly attacks
   the drift problem and is the closest thing to standard multi-task training,
   but it is the invasive option: `set_molecule()` swaps model buffers globally
   (`_activate_bundle`), so scoring a mixed batch means either per-sample bundle
   activation or making the scorer take the menu as an argument rather than
   state. Also breaks the `buffer_size == num_samples` assert and the
   per-molecule best-so-far tracking keyed off `_tracker_key`.

If only one goes in the thesis, (1) is nearly free and (3) is the one with a
real story attached.

## Pointer action space: open decisions (REMIND ME)

The pointer policy (`models/pointer.py`, `models/pointer_dag.py`) builds a gate
from four pointers into the orbital table instead of indexing a CCSD-screened
pool. Three choices were made to get a first run going, and each is a deliberate
placeholder rather than a settled answer.

### 1. `only_use_first_pauli` — currently ON, and probably shouldn't stay

A pool entry under `only_use_first_pauli: true` is ONE Pauli string of an
excitation generator, not the whole generator. `make_excitation_operator()`
mirrors that, so a pointer gate costs exactly what a pool gate costs and the
first comparison isolates the action space — the only difference between the two
runs is that the pointer can reach all 315 N2 excitations instead of the 117
CCSD kept.

But it is a strange convention on its own terms: applying one Pauli fragment of
exp(t (T - T^dag)) is not applying the excitation. It was presumably adopted to
keep gate counts down. Once the matched comparison is done, run the ablation:

    only_use_first_pauli: true   (matched, current)
    only_use_first_pauli: false  (full generator per gate)

with L reduced for the second so the COMPILED gate counts match rather than L.
If the full generator wins, the fragment convention should be retired for both
the pool and the pointer, and the earlier pool results re-read in that light.

### 2. Single excitations get an arbitrary angle

`mp2_angle()` returns 2*t_MP2, which is exactly zero for singles (Brillouin's
theorem — singles do not couple to the HF reference). A zero angle makes the gate
an identity and wastes a circuit slot, so singles fall back to `single_angle`
(0.1 rad by default). That constant is not physics.

Options, in order of how defensible they are:
  - `allow_singles: false` — doubles only. Honest, and the N2 pool is dominated
    by doubles anyway. Implemented, off by default.
  - learn a per-gate angle scale alongside the pointer (a fifth decode step over
    a small discrete set of angles).
  - keep CCSD t1 just for singles. Cheap, but reintroduces the dependence the
    whole design is trying to remove, so only as a diagnostic.

### 3. CCSD still runs, even though the pointer never uses it

`create_molecule_bundles()` builds a pool for every molecule, and that calls
CCSD. The pointer policy ignores the resulting menu — it only reads
`get_orbital_features()` and `cas_hamiltonian.h2` — but the cost is still paid
and the claim "no CCSD in the loop" is not yet TRUE end to end. Removing it means
a bundle variant that skips `build_operator_pool`, which is easy but touches the
factory and every model that expects `operator_features`. Do it before making the
no-CCSD claim in writing.

### Also worth knowing

Pointer pool indices are only meaningful WITHIN one run: `ensure_excitation()`
fills the cache in whatever order sampling visits, so a replay buffer or
checkpoint from another run maps indices to different excitations. `_to_picks()`
raises rather than silently mis-decoding, but always run with
`trainer.load_checkpoint=false` and a fresh `exp_tag`.
