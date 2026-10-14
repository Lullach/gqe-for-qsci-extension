import numpy
import torch
import torch.nn as nn


# =============================================================================
# STUDY NOTES (walkthrough with scratch/explore_operator_scorer.py, 2026-10-13)
# =============================================================================
#
# In one sentence
#   At each step the policy describes the partial circuit as a vector (query),
#   and forward() rates every operator on the menu against that description.
#
# three pieces
#   query   "what the circuit needs next". The model's summary of the partial
#           circuit as H numbers. Not human-readable; its meaning is learned.
#             DAG GNN : mean of the frontier-node embeddings (latest node on each
#                       qubit) after the GAT layers -> one vector per circuit, overall (B, H)
#             GPT-2 / diffusion : one hidden state per position          (B, L, H)
#   keys    "what each operator offers". Each operator's 11 physical features
#           (FEATURE_NAMES in operator_pool.py), z-scored per column, then
#           passed through op_encoder (Linear -> GELU -> Linear)          (V, H)
#   forward query @ keys.T * 1/sqrt(H): one score per operator    (B, V) / (B, L, V)
#
# Vocab
#   B batch (circuits in parallel)  H hidden size (model setting)
#   L circuit length (positions)    V number of operators in the pool (menu)
#   Real run (archL15_dag_gnn, H10): B=10, H=128, L=15, V=546, 20 qubits.
#   No weight has V in its shape, so the same weights can score any menu size,
#   which is why a different molecule's menu can be plugged in.
#
# Sign convention (CLAUDE)
#   Every model samples from softmax(-inv_temperature * score): a LOW score
#   means LIKELY. dag_gnn._scaled_logits relies on this (mask after the minus).
#
# How the DAG GNN uses it (dag_gnn.py, sample_sequence)
#   node_tokens (B, n_qubits+L) -> embeddings (B, nodes, H) -> 6 GAT layers
#   -> gather frontier (B, n_qubits, H) -> mean -> pooled (B, H)
#   -> _scaled_logits -> self.operator_head(pooled) -> (B, V) -> sample 1 op each
#   Repeated once per gate (L times). There is no per-qubit scoring: the frontier
#   is averaged BEFORE scoring, so "which qubits" survives only implicitly.
#
# Python / PyTorch details
#   scorer(query)   is the same as self.forward(query)
#
# Identical feature rows -> identical keys -> identical scores. Expected for
# the commuting Pauli fragments of one excitation (only_use_first_pauli=false).
# =============================================================================


class OperatorScorer(nn.Module):
    """
    Feature-based replacement for a fixed ``Linear(H, V)`` operator head.

        keys        = op_encoder(normalize(features))     # (V, H)
        logits[b,k] = query[b] @ keys[k] / sqrt(H)

    Instead of one free weight column per operator, each operator is scored from its physical
    feature vector (see OperatorPool.get_operator_features and NOTES.md,
    "Cross-molecule generalization"). The encoder weights are shared across
    operators, so the head:

      - works for any number of operators V (the menu is an input, not a size),
      - transfers across molecules whose operators live in the same feature space,
      - identical feature rows receive identical logits. Intended: rows only collide for
        symmetry-equivalent excitations (degenerate orbitals) or commuting
        fragments of one generator.

    Shapes: ``forward`` broadcasts, so it serves both the DAG GNN's pooled query
    ``(B, H) -> (B, V)`` and the per-position hidden states of the diffusion /
    GNN denoisers ``(B, L, H) -> (B, L, V)``.

    Weight tying: models that also embed operator IDs on the *input* side (GPT-2's
    ``wte``, the absorbing denoisers' ``token_embedding``, the DAG GNN's gate
    nodes) reuse ``keys()`` as their embedding table. That is exactly GPT-2-style
    weight tying, except the tied matrix is *computed from features* rather than
    being a free parameter.

    Normalization
    -------------
    Feature columns have different scales, so each column is z-scored;

    ``update_stats`` controls whether ``set_features`` recomputes the mean/std.

      - Single molecule (Phase 1): irrelevant, stats are computed once.
      - Cross-molecule (Phase 2+): per-menu stats would make the SAME physical
        operator take DIFFERENT feature values in different molecules, partly
        defeating transfer. Compute stats once over the training set and then
        pass ``update_stats=False`` on every swap so the normalization is a fixed
        physical scale. See NOTES.md, "Normalization across molecules".

    The 1/sqrt(H) scale keeps logit variance comparable to the Linear head from earlier version
    at initialization, so temperature scheduling tuned for the baseline can carry over.

    """

    def __init__(self, operator_features, hidden_size: int):
        super().__init__()
        feats = self._as_tensor(operator_features)
        assert feats.dim() == 2, "operator_features must be (V, feat_dim)"

        self.feat_dim = feats.shape[1]
        self.scale = hidden_size ** -0.5

        self.op_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Raw features + normalization statistics, kept separate so the stats can
        # be frozen across molecule swaps.
        self.register_buffer("features", feats)
        self.register_buffer("feat_mean", torch.zeros(1, self.feat_dim))
        self.register_buffer("feat_std", torch.ones(1, self.feat_dim))
        self.set_features(feats, update_stats=True)

    @staticmethod
    def _as_tensor(x) -> torch.Tensor:
        return torch.as_tensor(numpy.asarray(x), dtype=torch.float32)


    def set_features(self, operator_features, update_stats: bool = True) -> None:
        """
        Install a new operator menu.

        Phase 1: called once from __init__. Phase 2+: called per molecule swap,
        normally with update_stats=False so the feature scale stays fixed across
        molecules.
        """
        feats = self._as_tensor(operator_features).to(self.features.device)
        assert feats.shape[1] == self.feat_dim, (
            f"feature dim {feats.shape[1]} != scorer feat_dim {self.feat_dim}"
        )
        if feats.shape == self.features.shape:
            self.features.copy_(feats)
        else:  # different V (new molecule): re-register the buffer
            self.features = feats

        if update_stats:
            self.feat_mean.copy_(feats.mean(dim=0, keepdim=True))
            # clamp guards constant columns (std=0): they normalize to 0 everywhere
            self.feat_std.copy_(feats.std(dim=0, keepdim=True).clamp_min(1e-6))


    def set_normalization(self, mean, std) -> None:
        """
        Install fixed feature-normalization statistics (Phase 2 step 5).

        For cross-molecule training the mean/std must be computed once over the
        whole training set and then frozen, so the SAME physical operator maps to
        the SAME normalized coordinates in every molecule. Call this once after
        model construction; thereafter use set_features(..., update_stats=False).
        """
        mean_t = self._as_tensor(mean).reshape(1, self.feat_dim).to(self.feat_mean.device)
        std_t = self._as_tensor(std).reshape(1, self.feat_dim).clamp_min(1e-6).to(self.feat_std.device)
        self.feat_mean.copy_(mean_t)
        self.feat_std.copy_(std_t)


    @property
    def vocab_size(self) -> int:
        return self.features.shape[0]


    def keys(self) -> torch.Tensor:
        """
        (V, H) encoded operator keys.

        Doubles as the input-side embedding table for models that feed operator
        IDs back in (weight tying). Recomputed per call — V is small (~100-200)
        and the encoder weights change during training.
        """
        normalized = (self.features - self.feat_mean) / self.feat_std
        return self.op_encoder(normalized)

  
    def forward(self, query: torch.Tensor, keys: torch.Tensor | None = None) -> torch.Tensor:
        """
        Score every operator in the menu against the query.
        query : (B, H) or (B, L, H)
        keys  : optional precomputed keys() — pass it when the caller already
                needed the table for input embedding, to avoid encoding twice.
        Returns (B, V) or (B, L, V).
        """
        if keys is None:
            keys = self.keys()
        return self.scale * (query @ keys.T)


class SpecialTokenEmbedding(nn.Module):
    """
    Input-side embedding for token sequences that mix operator IDs with special
    tokens (e.g. ``[MASK]`` at index vocab_size).

    Real operator tokens (index < V) are embedded as the scorer's keys — weight
    tying, so the feature representation is shared between input and output.
    Special tokens (index >= V) get their own learned vectors, since they have no
    physical features.

    Replaces ``nn.Embedding(V + n_special, H)``, whose first V rows are exactly
    the molecule-specific integer-ID lookup we are removing.
    """

    # CLAUDE
    def __init__(self, n_special: int, hidden_size: int):
        super().__init__()
        self.n_special = int(n_special)
        if self.n_special > 0:
            self.special = nn.Parameter(torch.randn(self.n_special, hidden_size) * 0.02)
        else:
            self.register_parameter("special", None)

    # CLAUDE
    def forward(self, tokens: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        tokens : (..., ) long, values in [0, V + n_special)
        keys   : (V, H) from OperatorScorer.keys()
        Returns (..., H)
        """
        vocab_size = keys.shape[0]
        # clamp keeps the gather in range for special ids; those entries are
        # overwritten below, so the bogus lookup value never survives.
        emb = keys[tokens.clamp(max=vocab_size - 1)]
        if self.n_special > 0:
            is_special = tokens >= vocab_size
            if bool(is_special.any()):
                special = self.special[(tokens - vocab_size).clamp(min=0)]
                emb = torch.where(is_special.unsqueeze(-1), special, emb)
        return emb
