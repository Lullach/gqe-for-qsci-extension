# Research Notes

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
# configs/model/gnn_absorbing.yaml
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

## Alternative diffusion formulations

### Uniform noise (D3PM absorbing → D3PM uniform)

The current model uses *absorbing* diffusion: tokens are masked (replaced by
a single MASK token). An alternative is **uniform noise**: at each step a
token can transition to *any* other gate with some probability.

- Forward: each token is re-sampled uniformly from the vocabulary with
  probability (1 − α_t) instead of just being masked.
- Backward: the denoiser predicts x_0 but the posterior now mixes in a
  uniform component rather than a delta on MASK.
- Advantage: no special MASK token needed; every intermediate sample is a
  valid gate sequence.
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
