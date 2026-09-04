"""
CircuitDAGGNNPolicy's action space replaced by orbital pointers.

Same generative story as the pool-based DAG policy — build the circuit gate by
gate, run a GNN over the partial DAG, pool the frontier — but the pooled frontier
is used as a POINTER QUERY rather than as logits over a precomputed operator
menu. A gate is assembled from four pointers into the orbital table
(models/pointer.py), so no operator pool is an input.

What that removes, relative to CircuitDAGGNNPolicy:

  - the (V, feat_dim) operator menu and its OperatorScorer head
  - the (V, n_qubits) footprint table   — a footprint comes from the excitation
  - the (V, V) commutation matrix       — 13,924 entries for N2, 5.3 GB at
    30 orbitals; commutation is only ever needed for the ~L^2 pairs actually in
    a sampled circuit
  - canonical-form masking, which existed to quotient out the redundant
    orderings a flat operator alphabet admits. The ordering constraints j > i
    and b > a now make each excitation unspellable more than one way, so the
    redundancy is gone at the source rather than masked out afterwards.

Interface contract
------------------
Everything downstream is keyed by integer index — sampler.py does
`[pool[j] for j in row]`, and the replay buffer stores `state["idx"]`. So this
policy still emits integers: it asks the pool to materialise each excitation it
builds (OperatorPool.ensure_excitation) and hands back that index. The pool GROWS
as the policy explores instead of being screened up-front by CCSD.

Consequence: an index only means something within one run, because the cache is
filled in whatever order sampling happens to visit. Run with
trainer.load_checkpoint=false, which _init_ enforces loudly if a stale buffer
turns up in log_prob().
"""

from __future__ import annotations

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from gqe_qsci.gqe.models.policy import Policy
from gqe_qsci.gqe.models.pointer import (
    ExcitationPointer, ExcitationRules, OrbitalEncoder,
    build_orbital_inputs, excitation_pairs, excitation_qubits, gate_embedding,
)

try:
    from torch_geometric.nn import GATConv
except ImportError:  # pragma: no cover - optional dependency
    GATConv = None


def _require_pyg():
    if GATConv is None:
        raise ImportError(
            "torch_geometric is required for GNN models but is not installed.\n"
            "Install it with:  pip install torch_geometric"
        )


class PointerDAGGNNPolicy(Policy):
    """
    Parameters
    ----------
    pool : an OperatorPool
        Supplies the orbital/pairwise features and materialises excitations.
        Held by reference and MUTATED as the policy explores (see module
        docstring).
    single_angle : float
        Rotation angle for single excitations, which get zero from MP2 by
        Brillouin's theorem. A placeholder, not physics — see NOTES.md.
    allow_singles : bool
        Restrict the action space to doubles. Worth trying, since with an MP2
        angle a single is a somewhat arbitrary gate.
    """

    def __init__(
        self,
        pool,
        ngates: int,
        hidden_size: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        single_angle: float = 0.1,
        allow_singles: bool = True,
    ):
        _require_pyg()
        super().__init__()

        self.ngates = int(ngates)
        self.hidden_size = int(hidden_size)
        self.single_angle = float(single_angle)
        self.allow_singles = bool(allow_singles)

        orb, pair = build_orbital_inputs(pool)
        self.encoder = OrbitalEncoder(
            orb.shape[1], pair.shape[-1], hidden_size,
            num_layers=encoder_layers, num_heads=encoder_heads, dropout=dropout,
        )
        self.pointer = ExcitationPointer(hidden_size, pair.shape[-1])

        self.register_buffer("orb_feats", torch.from_numpy(orb))
        self.register_buffer("pair_feats", torch.from_numpy(pair))
        self._attach(pool, orb)

        # Gate slots that have not been placed yet are isolated nodes with no
        # edges, so this vector never reaches the pooled frontier — it exists to
        # keep the node table rectangular, matching CircuitDAGGNNPolicy.
        self.unplaced_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.frontier_embedding = nn.Embedding(2, hidden_size)

        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_size, hidden_size, heads=num_heads, concat=False,
                    dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )

    # -- molecule binding ---------------------------------------------------

    def _attach(self, pool, orb):
        self.pool = pool
        self.n_qubits = int(orb.shape[0])
        self.rules = ExcitationRules(orb, device=self.orb_feats.device,
                                     allow_singles=self.allow_singles)

    def set_molecule(self, bundle):
        """
        Re-point at another molecule. Unlike the pool-based policies there is no
        menu, no footprint table and no commutation matrix to swap — only the
        orbital table changes, and its ROW COUNT may differ without any parameter
        being resized, because the head scores orbitals rather than operators.
        Never call mid-rollout.
        """
        orb, pair = build_orbital_inputs(bundle.pool)
        device = self.orb_feats.device
        self.orb_feats = torch.from_numpy(orb).to(device)
        self.pair_feats = torch.from_numpy(pair).to(device)
        self._attach(bundle.pool, orb)

    # -- DAG plumbing -------------------------------------------------------

    def _init_dag_state(self, batch: int, device: torch.device):
        frontier = torch.arange(self.n_qubits, device=device).unsqueeze(0)
        return frontier.expand(batch, -1).clone(), [[] for _ in range(batch)], \
            [[] for _ in range(batch)]

    def _node_embeddings(self, orb_keys, gate_embs, batch):
        """(B, n_qubits + ngates, H). Qubit wires ARE orbitals, so their node
        embeddings are the orbital keys directly — no separate qubit encoder."""
        parts = [orb_keys.unsqueeze(0).expand(batch, -1, -1)]
        if gate_embs:
            parts.append(torch.stack(gate_embs, dim=1))
        unplaced = self.ngates - len(gate_embs)
        if unplaced > 0:
            parts.append(
                self.unplaced_embedding.view(1, 1, -1).expand(batch, unplaced, -1)
            )
        return torch.cat(parts, dim=1)

    def _pool_frontier(self, node_embs, frontier, edge_srcs, edge_dsts, device):
        """One GNN pass over the partial DAG -> (B, H) pooled frontier."""
        B, num_nodes, _ = node_embs.shape
        frontier_s = frontier.clone()      # frontier is mutated in place later

        flags = torch.zeros(B, num_nodes, dtype=torch.long, device=device)
        flags.scatter_(1, frontier_s, torch.ones_like(frontier_s))
        x = (node_embs + self.frontier_embedding(flags)).view(B * num_nodes, -1)

        src, dst = [], []
        for b in range(B):
            off = b * num_nodes
            s, d = edge_srcs[b], edge_dsts[b]
            src.extend(u + off for u in s + d)      # undirected: both directions
            dst.extend(u + off for u in d + s)
        edge_index = (torch.tensor([src, dst], dtype=torch.long, device=device)
                      if src else torch.zeros((2, 0), dtype=torch.long, device=device))

        h = x
        for layer, norm in zip(self.gnn_layers, self.layer_norms):
            if edge_index.shape[1] > 0:
                h = h + F.gelu(norm(layer(h, edge_index)))

        h3 = h.view(B, num_nodes, -1)
        idx = frontier_s.unsqueeze(-1).expand(-1, -1, h3.shape[-1])
        return h3.gather(1, idx).mean(dim=1)                       # (B, H)

    def _advance(self, picks, gate_node, frontier, edge_srcs, edge_dsts):
        """Wire the new gate to the frontier of every orbital it touches."""
        for b in range(picks.shape[0]):
            for q in excitation_qubits(tuple(int(v) for v in picks[b]), self.n_qubits):
                edge_srcs[b].append(int(frontier[b, q]))
                edge_dsts[b].append(gate_node)
                frontier[b, q] = gate_node

    # -- the rollout, shared by sampling and replay --------------------------

    def _rollout(self, batch, inv_temperature, device, forced=None,
                 return_entropy=False):
        """
        forced : (B, L, 4) long to replay stored gates, else None to sample.
        Returns picks (B, L, 4), logp (B, L), entropy (B, L) or None.
        """
        # The mask tables are not nn.Module state, so policy.to(device) does not
        # reach them — Lightning builds the policy on CPU and moves it to the
        # GPU afterwards. No-op once they are already there.
        self.rules.to(self.orb_feats.device)

        orb_keys = self.encoder(self.orb_feats, self.pair_feats)
        frontier, edge_srcs, edge_dsts = self._init_dag_state(batch, device)

        gate_embs, all_picks, all_logp, all_ent = [], [], [], []
        for step in range(self.ngates):
            node_embs = self._node_embeddings(orb_keys, gate_embs, batch)
            query = self._pool_frontier(node_embs, frontier, edge_srcs,
                                        edge_dsts, device)
            out = self.pointer(
                query, orb_keys, self.pair_feats, self.rules, inv_temperature,
                forced=(None if forced is None else forced[:, step]),
                return_entropy=return_entropy,
            )
            picks, step_logp = out[0], out[1]
            if return_entropy:
                all_ent.append(out[2].sum(dim=-1))

            gate_embs.append(gate_embedding(picks, orb_keys))
            all_picks.append(picks)
            # A gate's log-probability is the sum over its four pointer steps.
            all_logp.append(step_logp.sum(dim=-1))
            self._advance(picks, self.n_qubits + step, frontier, edge_srcs, edge_dsts)

        return (
            torch.stack(all_picks, dim=1),
            torch.stack(all_logp, dim=1),
            torch.stack(all_ent, dim=1) if return_entropy else None,
        )

    # -- Policy interface ---------------------------------------------------

    def act(self, state, inv_temperature):
        raise RuntimeError(
            "PointerDAGGNNPolicy generates whole sequences via "
            "sample_sequence(); act() is not supported."
        )

    def _to_indices(self, picks):
        """
        (B, L, 4) pointer tuples -> (B, L) long pool indices, materialising any
        excitation the pool has not seen before.
        """
        n = self.n_qubits
        rows = []
        for b in range(picks.shape[0]):
            row = []
            for t in range(picks.shape[1]):
                key = tuple(int(v) for v in picks[b, t])
                row.append(self.pool.ensure_excitation(
                    key, excitation_pairs(key, n), single_angle=self.single_angle
                ))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long, device=picks.device)

    def _to_picks(self, indices):
        """(B, L) pool indices -> (B, L, 4) pointer tuples, via the pool's map."""
        keys = getattr(self.pool, "excitation_keys", None)
        rows = []
        for b in range(indices.shape[0]):
            row = []
            for t in range(indices.shape[1]):
                k = int(indices[b, t])
                if keys is None or k not in keys:
                    raise RuntimeError(
                        f"operator index {k} was not produced by this policy. "
                        "Pointer indices only mean something within one run — "
                        "re-run with trainer.load_checkpoint=false and a fresh "
                        "exp_tag."
                    )
                row.append(list(keys[k]))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long, device=indices.device)

    def sample_sequence(self, state, inv_temperature):
        B = state["idx"].shape[0]
        device = state["idx"].device
        picks, _, _ = self._rollout(B, inv_temperature, device)
        ops = self._to_indices(picks)
        state["idx"] = torch.cat((state["idx"], ops), dim=1)
        return state

    def log_prob(self, indices, inv_temperature, return_entropy=False,
                 reveal_step=None):
        """
        Exact log p(gate_1 ... gate_L) = sum_t sum_s log p(pointer step s of gate t).

        Already a trajectory log-probability — the gate sequence IS the
        trajectory — so reveal_step is accepted for API compatibility and
        ignored, matching CircuitDAGGNNPolicy.
        """
        gate_tokens = indices[:, 1:]                 # strip BOS
        forced = self._to_picks(gate_tokens)
        _, logp, entropy = self._rollout(
            gate_tokens.shape[0], inv_temperature, gate_tokens.device,
            forced=forced, return_entropy=return_entropy,
        )
        if return_entropy:
            return logp, entropy
        return logp
