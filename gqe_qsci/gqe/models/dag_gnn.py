import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gqe_qsci.gqe.models.operator_scorer import OperatorScorer
from gqe_qsci.gqe.models.policy import Policy

try:
    from torch_geometric.nn import GATConv
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False
    GATConv = None


def _require_pyg():
    if not _TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError(
            "torch_geometric is required for GNN models but is not installed.\n"
            "Install it with:  pip install torch_geometric"
        )


def _pack_footprints(footprints: list[list[int]], n_qubits: int) -> torch.Tensor:
    """
    Convert a list of qubit-index lists to a (V, n_qubits) bool tensor.
    Operators with an empty footprint (e.g. the identity) remain all-False:
    they become isolated nodes in the DAG with no edges and no frontier update.
    """
    V = len(footprints)
    fp = torch.zeros(V, n_qubits, dtype=torch.bool)
    for k, qubits in enumerate(footprints):
        for q in qubits:
            fp[k, q] = True
    return fp


class CircuitDAGGNNPolicy(Policy):
    """
    Circuit generation policy using a GNN over a circuit DAG with fixed depth L.

    DAG structure
    -------------
    Nodes  (n_qubits + ngates total per sample):
      - Input nodes  0 .. n_qubits-1       : one per qubit wire (always present)
      - Gate nodes   n_qubits .. n_qubits+L-1 : one per gate position

    Edges (undirected, qubit-wire connections):
      When gate t is assigned operator k, for each qubit q in k's footprint:
        edge( frontier[q], gate_node_t )
      Then frontier[q] is updated to gate_node_t.

    Generation (sample_sequence)
    ----------------------------
    L sequential steps.  At each step the GNN reads the current partial DAG,
    pools the frontier node embeddings into a global context vector, and
    samples the next operator from a linear head over that context.

    Canonical-form masking (canonical_masking=True)
    -----------------------------------------------
    A circuit is a word over the operator alphabet in which adjacent letters may
    be swapped whenever their operators commute — a Mazurkiewicz trace. Swapping
    commuting gates leaves the unitary, the state, the energy and the reward
    unchanged, so those orderings are redundant search space. Measured on the N2
    pool, 57% of qubit-sharing operator pairs commute (53% for H2O), so the
    redundancy is large.

    At each step we forbid operator k when it can be migrated left over a run of
    commuting letters onto a position holding a strictly larger operator (see
    _canonical_mask). By the lexicographic normal form of traces this admits
    exactly one word per equivalence class: every distinct unitary remains
    reachable, each reachable exactly once. See NOTES.md.

    Side effect: the identity operator commutes with everything and has the
    smallest index, so it is only ever placeable before any real gate — leading
    identities, never interleaved ones. Since identity is a no-op this does not
    change the set of reachable unitaries.

    log_prob
    --------
    Exact conditional log-probability: re-runs the L-step DAG construction
    with the stored operator sequence, computing p(a_t | DAG_{t-1}) at each step.
    Deterministic (no stochasticity) → best possible importance-weight stability.
    The mask is reconstructed from the stored prefix so it matches sampling
    exactly; a mismatch would corrupt the GRPO importance ratio.

    Batching note
    -------------
    All samples share the same node count (n_qubits + ngates) so the flat
    (B * num_nodes, H) tensor from the GNN can be reshaped to (B, num_nodes, H)
    for vectorised frontier pooling.  Different samples may have different edge
    structures (different operators → different footprints), so per-sample edge
    lists are offset-shifted and concatenated into one big disconnected graph,
    the standard PyG batching pattern.

    Use config  model=dag_gnn  to select this variant.
    """

    def __init__(
        self,
        vocab_size: int,
        ngates: int,
        n_qubits: int,
        qubit_footprints: list[list[int]],
        hidden_size: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        commutation_matrix=None,
        canonical_masking: bool = True,
        feature_scorer: bool = False,
        operator_features=None,
        orbital_features=None,
    ):
        _require_pyg()
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.ngates     = int(ngates)
        self.n_qubits   = int(n_qubits)

        # (V, n_qubits) bool — qubit footprint for each operator
        self.register_buffer("_fp_flat", _pack_footprints(qubit_footprints, n_qubits))

        # Canonical-form action masking (trace-monoid lexicographic normal form).
        # Needs the (V, V) commutation matrix; without it, masking is disabled.
        self.canonical_masking = bool(canonical_masking) and commutation_matrix is not None
        if commutation_matrix is None:
            commutation_matrix = torch.zeros(self.vocab_size, self.vocab_size, dtype=torch.bool)
        self.register_buffer(
            "commutes", torch.as_tensor(numpy.asarray(commutation_matrix), dtype=torch.bool)
        )

        # Node token layout (both modes use the same indices):
        #   index q                  (0 .. n_qubits-1)          → input node for qubit q
        #   index n_qubits + k       (n_qubits .. n_qubits+V-1) → gate node with operator k
        #   index n_qubits + V       (UNPLACED)                 → gate slot not yet filled
        self.UNPLACED = n_qubits + vocab_size
        self.feature_scorer = bool(feature_scorer)

        if self.feature_scorer:
            # Feature mode: the node table is *composed* each forward pass from
            # physical features rather than looked up by index.
            #   qubit nodes → qubit_encoder(orbital features)  (transfers across
            #                 molecules; a per-index embedding would not — "qubit
            #                 5" is different physics in every molecule)
            #   gate nodes  → the scorer's keys (weight-tied to the output head)
            #   UNPLACED    → its own learned vector (no physical features)
            if orbital_features is None:
                raise ValueError(
                    "feature_scorer=True requires orbital_features "
                    "(factory passes pool.get_orbital_features())."
                )
            orb = torch.as_tensor(numpy.asarray(orbital_features), dtype=torch.float32)
            assert orb.shape[0] == self.n_qubits, (
                f"orbital_features has {orb.shape[0]} rows, expected n_qubits={self.n_qubits}"
            )
            self.register_buffer("orbital_features", orb)          # (n_qubits, D)
            self.qubit_encoder = nn.Sequential(
                nn.Linear(orb.shape[1], hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.unplaced_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)
            self.node_embedding = None
        else:
            self.node_embedding = nn.Embedding(n_qubits + vocab_size + 1, hidden_size)

        # Binary flag: is this node the current frontier for any qubit wire?
        self.frontier_embedding = nn.Embedding(2, hidden_size)

        # GAT denoiser — same pattern as gnn.py
        self.gnn_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_size,
                out_channels=hidden_size,
                heads=num_heads,
                concat=False,   # average heads → hidden_size stays constant for residuals
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Pool frontier embeddings → distribution over V operators.
        # feature_scorer=False : free Linear column per operator (integer-ID
        #                        baseline; molecule-specific by construction).
        # feature_scorer=True  : score operators from their physical feature
        #                        vectors (Phase 1 of the cross-molecule plan;
        #                        see OperatorScorer and NOTES.md).
        if self.feature_scorer:
            if operator_features is None:
                raise ValueError(
                    "feature_scorer=True requires operator_features "
                    "(factory passes pool.get_operator_features())."
                )
            self.operator_head = OperatorScorer(operator_features, hidden_size)
        else:
            self.operator_head = nn.Linear(hidden_size, vocab_size)

    # -------------------------------------------------------------------------
    # Cross-molecule swap
    # -------------------------------------------------------------------------

    def set_molecule(self, bundle):
        """
        Atomically re-point the policy at bundle's molecule. Swaps ALL
        per-molecule buffers together (menu, footprints, commutation, orbital
        features, vocab_size, UNPLACED). Never call mid-rollout.

        Reassigning a registered-buffer attribute with a tensor keeps it
        registered (nn.Module.__setattr__ routes it back into _buffers), so this
        also handles V changing between molecules.
        """
        if not self.feature_scorer:
            raise RuntimeError(
                "set_molecule requires feature_scorer=True (the integer-ID head "
                "has a fixed per-operator column and cannot transfer)."
            )
        device = self._fp_flat.device

        self.vocab_size = int(bundle.vocab_size)
        self.n_qubits   = int(bundle.n_qubits)
        self.UNPLACED   = self.n_qubits + self.vocab_size

        self._fp_flat = _pack_footprints(
            bundle.qubit_footprints, self.n_qubits
        ).to(device)
        self.commutes = torch.as_tensor(
            numpy.asarray(bundle.commutation_matrix), dtype=torch.bool, device=device
        )
        self.orbital_features = torch.as_tensor(
            numpy.asarray(bundle.orbital_features), dtype=torch.float32, device=device
        )
        # Shares the output head's scorer (weight-tied), so this updates both the
        # gate-node input embedding and the output logits. Frozen normalization
        # stats (Phase 2 step 5).
        self.operator_head.set_features(bundle.operator_features, update_stats=False)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _node_table(self) -> torch.Tensor:
        """
        (n_qubits + V + 1, H) embedding table for node tokens.

        Integer-ID mode: the learned nn.Embedding weight.
        Feature mode: composed from physics each call —
            rows [0, n_qubits)            qubit wires  ← qubit_encoder(orbitals)
            rows [n_qubits, n_qubits+V)   gate nodes   ← scorer keys (weight-tied)
            row  n_qubits+V               UNPLACED     ← learned vector
        Composing (rather than looking up) is what makes the input side portable:
        every row is derived from features, none from an operator/qubit index.
        """
        if not self.feature_scorer:
            return self.node_embedding.weight
        return torch.cat(
            [
                self.qubit_encoder(self.orbital_features),        # (n_qubits, H)
                self.operator_head.keys(),                        # (V, H)
                self.unplaced_embedding.unsqueeze(0),             # (1, H)
            ],
            dim=0,
        )

    def _step_forward(
        self,
        node_tokens: torch.Tensor,   # (B, num_nodes) long
        frontier: torch.Tensor,      # (B, n_qubits)  long — node index per qubit
        edge_srcs: list[list[int]],  # per-sample source node indices
        edge_dsts: list[list[int]],  # per-sample destination node indices
        device: torch.device,
    ) -> torch.Tensor:
        """
        One GNN forward pass over the current partial DAG.
        Returns mean-pooled frontier embeddings: (B, H).
        """
        B, num_nodes = node_tokens.shape

        # Clone both index tensors before any autograd-tracked operation.
        # _advance_dag mutates node_tokens and frontier in-place after this
        # call; embedding and gather save their index arguments in the backward
        # graph, so in-place mutations would cause a version-mismatch error at
        # loss.backward() without these clones.
        node_tokens_s = node_tokens.clone()
        frontier_s    = frontier.clone()

        # Node features: token embedding + frontier flag embedding
        node_embs = self._node_table()[node_tokens_s]              # (B, num_nodes, H)
        frontier_flags = torch.zeros(B, num_nodes, dtype=torch.long, device=device)
        # Set flag=1 for each frontier node (may overlap for multi-qubit gates)
        frontier_flags.scatter_(1, frontier_s, torch.ones_like(frontier_s))
        x = (node_embs + self.frontier_embedding(frontier_flags)).view(
            B * num_nodes, -1
        )                                                          # (B*num_nodes, H)

        # Build a single batched edge index: offset each sample's edges by b*num_nodes
        all_src, all_dst = [], []
        for b in range(B):
            offset = b * num_nodes
            s, d = edge_srcs[b], edge_dsts[b]
            # Undirected: add both directions
            all_src.extend(u + offset for u in s + d)
            all_dst.extend(u + offset for u in d + s)

        if all_src:
            edge_index = torch.tensor(
                [all_src, all_dst], dtype=torch.long, device=device
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        # GAT message passing with residual connections + LayerNorm
        h = x
        for gnn_layer, norm in zip(self.gnn_layers, self.layer_norms):
            if edge_index.shape[1] > 0:
                h = h + F.gelu(norm(gnn_layer(h, edge_index)))

        # Reshape to (B, num_nodes, H) and gather frontier nodes
        h_3d = h.view(B, num_nodes, -1)                           # (B, num_nodes, H)
        idx  = frontier_s.unsqueeze(-1).expand(-1, -1, h_3d.shape[-1])  # (B, n_qubits, H)
        frontier_h = h_3d.gather(1, idx)                           # (B, n_qubits, H)
        return frontier_h.mean(dim=1)                              # (B, H)

    def _advance_dag(
        self,
        ops: torch.Tensor,           # (B,) long — operator index placed at this step
        gate_node: int,              # node index of the new gate
        node_tokens: torch.Tensor,   # (B, num_nodes) long — modified in-place
        frontier: torch.Tensor,      # (B, n_qubits) long — modified in-place
        edge_srcs: list[list[int]],
        edge_dsts: list[list[int]],
        fp: torch.Tensor,            # (V, n_qubits) bool
    ) -> None:
        """
        Advance the DAG state for all samples in the batch after placing `ops[b]`
        at `gate_node`.  Modifies node_tokens, frontier, edge_srcs, edge_dsts
        in-place.
        """
        for b in range(ops.shape[0]):
            op_k = ops[b].item()
            node_tokens[b, gate_node] = self.n_qubits + op_k

            qubits = fp[op_k].nonzero(as_tuple=True)[0].tolist()
            for q in qubits:
                prev = frontier[b, q].item()
                edge_srcs[b].append(prev)
                edge_dsts[b].append(gate_node)
                frontier[b, q] = gate_node

    # --- canonical-form (trace lexicographic normal form) masking -----------

    def _canonical_mask(self, prefix: torch.Tensor) -> torch.Tensor:
        """
        (B, V) bool — True where operator k is FORBIDDEN at this step.

        A circuit is a word over the operator alphabet; two adjacent letters may
        be swapped iff their operators commute (a Mazurkiewicz trace). Every
        equivalence class has a unique lexicographically-minimal representative,
        and admitting only that representative removes the redundant orderings.

        Placing k at step t, it can be migrated left to position i iff it
        commutes with EVERY letter in prefix[i..t-1]; doing so lowers the word
        lexicographically iff k < prefix[i]. So k is forbidden iff such an i
        exists:

            exists i:  (for all j in [i, t-1]: commutes(k, prefix[j]))
                       and  k < prefix[i]

        NB it is not enough to compare k against prefix[t-1] alone — a letter can
        hop left over a whole run of commuting letters. E.g. with prefix (2,0,0)
        and 1 commuting with 0 and 2, the word (2,0,0,1) passes the adjacent-only
        test yet is equivalent to the smaller (1,2,0,0). Verified by brute force
        against explicit equivalence-class enumeration.

        The largest operator index is never forbidden (k < prefix[i] can never
        hold for it), so the mask can never forbid everything.

        prefix : (B, t) long, t >= 1.
        """
        B, t = prefix.shape
        # comm_kt[b, k, j] = commutes(k, prefix[b, j])
        comm_kt = self.commutes[:, prefix].permute(1, 0, 2)               # (B, V, t)
        # reach[b, k, j] = k commutes with all of prefix[b, j..t-1]
        # (suffix-AND along j, done as a reversed cumulative product of 0/1)
        reach = comm_kt.to(torch.float32).flip(-1).cumprod(-1).flip(-1) > 0.5
        ks = torch.arange(self.vocab_size, device=prefix.device)
        smaller = ks.view(1, -1, 1) < prefix.unsqueeze(1)                 # (B, V, t)
        return (reach & smaller).any(dim=-1)                              # (B, V)

    def _scaled_logits(self, pooled, inv_temperature, prefix):
        """
        Operator logits, inv_temperature-scaled, with the canonical mask applied.

        The mask MUST be applied after the -inv_temperature multiply: masking with
        -inf beforehand would become +inf here (inv_temperature > 0), making the
        forbidden operators maximally likely instead of impossible.

        prefix : (B, t) long of already-placed operators, or None at step 0.
        """
        scaled = -inv_temperature * self.operator_head(pooled)               # (B, V)
        if self.canonical_masking and prefix is not None and prefix.shape[1] > 0:
            scaled = scaled.masked_fill(self._canonical_mask(prefix), float("-inf"))
        return scaled

    def _init_dag_state(self, B: int, device: torch.device):
        """Allocate and initialise the DAG book-keeping tensors."""
        num_nodes = self.n_qubits + self.ngates

        node_tokens = torch.zeros(B, num_nodes, dtype=torch.long, device=device)
        for q in range(self.n_qubits):
            node_tokens[:, q] = q                  # input node tokens
        node_tokens[:, self.n_qubits:] = self.UNPLACED   # gate slots start as unplaced

        # Each qubit's frontier starts at its input node
        frontier = torch.arange(
            self.n_qubits, dtype=torch.long, device=device
        ).unsqueeze(0).expand(B, -1).clone()

        edge_srcs = [[] for _ in range(B)]
        edge_dsts = [[] for _ in range(B)]

        return node_tokens, frontier, edge_srcs, edge_dsts

    # -------------------------------------------------------------------------
    # Policy interface
    # -------------------------------------------------------------------------

    def act(self, state, inv_temperature):
        raise RuntimeError(
            "CircuitDAGGNNPolicy generates whole sequences via "
            "sample_sequence(); act() is not supported."
        )

    def sample_sequence(self, state, inv_temperature):
        """
        Generate a full gate sequence by building the circuit DAG incrementally.

        At each of L steps:
          1. Run GNN over current partial DAG.
          2. Pool frontier node embeddings → operator logits.
          3. Sample operator.
          4. Connect new gate to its qubit frontiers; update frontier.

        The final flat sequence of operator indices is appended to state["idx"].
        """
        B      = state["idx"].shape[0]
        device = state["idx"].device
        fp     = self._fp_flat.to(device)

        node_tokens, frontier, edge_srcs, edge_dsts = self._init_dag_state(B, device)
        sampled_ops = []

        for step in range(self.ngates):
            gate_node = self.n_qubits + step

            pooled = self._step_forward(
                node_tokens, frontier, edge_srcs, edge_dsts, device
            )                                                        # (B, H)
            # The mask needs the whole prefix, not just the last operator.
            prefix = torch.stack(sampled_ops, dim=1) if sampled_ops else None
            scaled = self._scaled_logits(pooled, inv_temperature, prefix)      # (B, V)
            ops    = Categorical(logits=scaled).sample()                   # (B,)
            sampled_ops.append(ops)

            self._advance_dag(
                ops, gate_node, node_tokens, frontier, edge_srcs, edge_dsts, fp
            )

        ops_tensor   = torch.stack(sampled_ops, dim=1)               # (B, L)
        state["idx"] = torch.cat((state["idx"], ops_tensor), dim=1)
        return state

    def log_prob(self, indices, inv_temperature, return_entropy=False, reveal_step=None):
        """
        Exact log p_θ(a_1, …, a_L) = Σ_t log p_θ(a_t | DAG_{t-1}).

        Re-runs the L-step DAG construction with the stored operator sequence,
        computing the conditional log-probability at each step.  Fully
        deterministic — same sequence always yields the same value.

        This is already a trajectory log-probability: the gate sequence IS the
        trajectory, so nothing extra needs recording and reveal_step is
        accepted for API compatibility but ignored.

        Returns (B, L) per-step log-probabilities.
        """
        gate_tokens = indices[:, 1:]           # strip BOS  (B, L)
        B, L        = gate_tokens.shape
        device      = gate_tokens.device
        fp          = self._fp_flat.to(device)

        node_tokens, frontier, edge_srcs, edge_dsts = self._init_dag_state(B, device)

        total_logp    = torch.zeros(B, L, device=device)
        total_entropy = torch.zeros(B, L, device=device) if return_entropy else None

        for step in range(L):
            gate_node = self.n_qubits + step
            ops       = gate_tokens[:, step]   # (B,) — operator placed at this step

            pooled    = self._step_forward(
                node_tokens, frontier, edge_srcs, edge_dsts, device
            )                                                        # (B, H)
            # Same mask as sample_sequence, reconstructed from the stored prefix.
            # It must match exactly, or the GRPO importance ratio is computed
            # against a different distribution than the one that was sampled.
            prefix    = gate_tokens[:, :step] if step > 0 else None
            scaled    = self._scaled_logits(pooled, inv_temperature, prefix)    # (B, V)
            log_probs = F.log_softmax(scaled, dim=-1)                       # (B, V)

            total_logp[:, step] = log_probs[
                torch.arange(B, device=device), ops
            ]

            if return_entropy:
                # Masked entries have p = 0 and log p = -inf. Their product is
                # NaN, and it is NOT enough to select it away afterwards with
                # torch.where: the product is still *computed*, and its backward
                # gives d(p*logp)/dp = logp = -inf, which times the incoming
                # zero gradient is NaN. That NaN reaches operator_head, turns the
                # weights NaN, and the next rollout samples from NaN logits.
                # So replace -inf BEFORE any arithmetic touches it.
                probs = log_probs.exp()                       # exactly 0 if masked
                safe_logp = torch.where(
                    torch.isfinite(log_probs),
                    log_probs,
                    torch.zeros_like(log_probs),
                )
                total_entropy[:, step] = -(probs * safe_logp).sum(dim=-1)

            self._advance_dag(
                ops, gate_node, node_tokens, frontier, edge_srcs, edge_dsts, fp
            )

        # A -inf here means a stored sequence violates the canonical form, i.e.
        # it was generated by a policy without masking (a stale replay buffer or
        # a pre-masking checkpoint). That would silently poison the loss with
        # inf/NaN, so fail loudly instead.
        if self.canonical_masking and torch.isinf(total_logp).any():
            raise RuntimeError(
                "log_prob saw a non-canonical gate sequence under canonical "
                "masking: some stored operator was forbidden by the mask. This "
                "usually means the replay buffer or checkpoint predates "
                "canonical_masking. Re-run with trainer.load_checkpoint=false "
                "and a fresh exp_tag, or set model.canonical_masking=false."
            )

        if return_entropy:
            return total_logp, total_entropy
        return total_logp
