"""
Pointer action space: build excitations by POINTING AT ORBITALS instead of
indexing a precomputed operator pool.

Motivation
----------
The existing policies choose an operator index k from a pool built by screening
CCSD amplitudes (OperatorPool.build_operator_pool). That pool has two problems
for cross-molecule work:

  1. it is O(n_occ^2 * n_virt^2) rows, each needing features, a gate count and a
     row of the (V, V) commutation matrix — 73k rows / 5.3 GB of commutation
     booleans for a 30-orbital active space;
  2. its *membership* is decided by CCSD, so the policy never sees an operator
     the classical method rejected, and one of its features (`amplitude`) is a
     CCSD t-amplitude — unreliable exactly where correlation is strong.

Here the molecule is described by its n_so spin-orbitals (16 rows for N2, 60 for
a 30-orbital active space) and a gate is assembled from four pointers:

    i  (occupied)  ->  j (occupied, j > i)  or STOP  ->  a (virtual)  ->  b

Couplings are fetched only for the candidates live at a given step, so the cost
per decision is O(n) rather than O(n^4) up front. Nothing here needs CCSD: only
HF orbital energies/occupations and the active-space two-electron integrals,
both of which are already computed to define the QSCI Hamiltonian.

Canonical spelling comes for free. Requiring j > i and b > a means every
excitation has exactly one representation, which is what _excitation_key() and
`dedup_excitations` currently repair after the fact in operator_pool.py.

Layout
------
    build_orbital_inputs   molecule -> (orbital features, pairwise features)
    ExcitationRules        closed-form validity masks, O(n) per step
    OrbitalEncoder         graph transformer over orbitals -> (n, H) keys
    ExcitationPointer      the four-step decode; sampling AND replay

STOP/NONE is encoded as index n (one past the last orbital), so every step has
a uniform candidate axis of size n + 1.
"""

from __future__ import annotations

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

PAIR_FEATURE_NAMES = [
    "delta_eps",   # eps_q - eps_p: the denominator of every perturbative estimate
    "exchange",    # |(pq|pq)|: decays with spatial separation — the locality signal
    "same_spin",   # 1.0 if p and q have the same spin
    "occ_to_virt", # 1.0 if p is occupied and q is virtual (a legal excitation direction)
]


def build_orbital_inputs(pool):
    """
    (orb_feats (n, 3), pair_feats (n, n, 4)) for one molecule, as float32 numpy.

    orb_feats reuses OperatorPool.get_orbital_features() unchanged:
        [eps (HOMO-referenced), occ (0/1), is_beta (0/1)]

    pair_feats is the edge input of the orbital graph. The exchange integral
    (pq|pq) is read straight out of the active-space Hamiltonian, so this costs
    nothing beyond what QSCI already needs. Spin-orbital q maps to spatial
    orbital q // 2 (interleaved JW, even = alpha), matching the convention in
    operator_pool.py.
    """
    orb = numpy.asarray(pool.get_orbital_features(), dtype=numpy.float32)
    n = orb.shape[0]
    eps, occ, spin = orb[:, 0], orb[:, 1], orb[:, 2]

    h2 = numpy.asarray(pool.molecule.cas_hamiltonian.h2)
    sp = numpy.arange(n) // 2
    # h2[p, q, r, s] = (pq|rs) in chemists' notation over SPATIAL orbitals, the
    # same convention _hf_coupling() uses. (pq|pq) is the exchange integral.
    exchange = numpy.abs(h2[sp[:, None], sp[None, :], sp[:, None], sp[None, :]])

    pair = numpy.stack(
        [
            eps[None, :] - eps[:, None],
            exchange,
            (spin[:, None] == spin[None, :]).astype(numpy.float32),
            ((occ[:, None] > 0) & (occ[None, :] == 0)).astype(numpy.float32),
        ],
        axis=-1,
    )
    return orb, pair.astype(numpy.float32)


# ---------------------------------------------------------------------------
# Validity masks
# ---------------------------------------------------------------------------

class ExcitationRules:
    """
    Closed-form legality masks for the four decode steps.

    Every constraint is enforced by masking, so an illegal excitation cannot be
    sampled at all — as opposed to being generated and rejected afterwards,
    which would leave the log-probability inconsistent with the sampler.

    Enforced:
      - i, j occupied; a, b virtual
      - j > i and b > a  (canonical spelling: one representation per excitation)
      - Sz conservation: #beta among {i, j} equals #beta among {a, b}

    Sz propagates BACKWARD into step 3: `a` is only offered when a partner `b`
    of the required spin still exists above it, so step 4 can never be empty.

    All arrays are O(n) or O(2n); nothing here enumerates excitations.
    """

    def __init__(self, orbital_features, device=None, allow_singles: bool = True):
        self.allow_singles = bool(allow_singles)
        orb = torch.as_tensor(numpy.asarray(orbital_features), dtype=torch.float32)
        if device is not None:
            orb = orb.to(device)
        self.device = orb.device
        self.n = int(orb.shape[0])

        self.occ = orb[:, 1] > 0                       # (n,) bool
        self.spin = orb[:, 2].round().long()           # (n,) 0 = alpha, 1 = beta
        self.virt = ~self.occ
        self.idx = torch.arange(self.n, device=self.device)

        # per-spin indicator rows, (2, n)
        vs = torch.stack([self.virt & (self.spin == s) for s in (0, 1)]).float()
        os_ = torch.stack([self.occ & (self.spin == s) for s in (0, 1)]).float()

        self.n_virt_spin = vs.sum(-1)                                   # (2,)
        # "is there one of these STRICTLY after index p": suffix sum minus self
        self.has_virt_after = (vs.flip(-1).cumsum(-1).flip(-1) - vs) > 0    # (2, n)
        self.has_occ_after = (os_.flip(-1).cumsum(-1).flip(-1) - os_) > 0   # (2, n)

        # double_ok[n_beta]: an occupied pair carrying this many beta electrons
        # can be completed by SOME virtual pair.
        na, nb = float(self.n_virt_spin[0]), float(self.n_virt_spin[1])
        self.double_ok = torch.tensor(
            [na >= 2, na >= 1 and nb >= 1, nb >= 2],
            dtype=torch.bool, device=self.device,
        )

    # Tensor attributes, listed once so to() cannot silently miss one.
    _TENSORS = ("occ", "spin", "virt", "idx", "n_virt_spin",
                "has_virt_after", "has_occ_after", "double_ok")

    def to(self, device):
        """
        Move the precomputed mask tables to `device`, in place.

        This class is deliberately NOT an nn.Module (it holds no learnable
        state), which means `policy.to(device)` does not reach it — the tensors
        would stay wherever they were built. Lightning constructs the policy on
        CPU and moves it to the GPU afterwards, so without this the masks stay
        on CPU and masked_fill raises

            RuntimeError: expected self and mask to be on the same device

        Callers should invoke it at the start of a rollout; it is a no-op once
        the tables are already in the right place.
        """
        device = torch.device(device)
        if self.device == device:
            return self
        for name in self._TENSORS:
            setattr(self, name, getattr(self, name).to(device))
        self.device = device
        return self

    # -- helpers ------------------------------------------------------------

    def _n_beta(self, i, j):
        """#beta among the chosen occupied orbitals (j == n means single)."""
        is_single = j.eq(self.n)
        spin_j = self.spin[j.clamp(max=self.n - 1)]
        return self.spin[i] + torch.where(is_single, torch.zeros_like(spin_j), spin_j)

    # -- the masks ----------------------------------------------------------

    def step_mask(self, s: int, picks: list, batch: int) -> torch.Tensor:
        """(batch, n + 1) bool — True where the candidate is ALLOWED at step s."""
        n = self.n
        false_col = torch.zeros(batch, 1, dtype=torch.bool, device=self.device)

        if s == 0:                                   # choose i (occupied)
            # i is usable if it can start a single, or a double with some j > i
            single = (self.n_virt_spin[self.spin] > 0) if self.allow_singles \
                else torch.zeros_like(self.occ)
            double = (
                (self.double_ok[self.spin + 0] & self.has_occ_after[0])
                | (self.double_ok[self.spin + 1] & self.has_occ_after[1])
            )
            ok = (self.occ & (single | double)).unsqueeze(0).expand(batch, -1)
            return torch.cat([ok, false_col], dim=-1)

        if s == 1:                                   # choose j (occupied, j > i) or STOP
            i = picks[0]
            n_beta = self.spin[i].unsqueeze(1) + self.spin.unsqueeze(0)      # (B, n)
            j_ok = (
                self.occ.unsqueeze(0)
                & (self.idx.unsqueeze(0) > i.unsqueeze(1))
                & self.double_ok[n_beta]
            )
            stop_ok = (self.n_virt_spin[self.spin[i]] > 0).unsqueeze(1)      # (B, 1)
            if not self.allow_singles:
                stop_ok = torch.zeros_like(stop_ok)
            return torch.cat([j_ok, stop_ok], dim=-1)

        if s == 2:                                   # choose a (virtual)
            i, j = picks[0], picks[1]
            is_single = j.eq(n).unsqueeze(1)
            n_beta = self._n_beta(i, j)

            a_single = self.virt.unsqueeze(0) & (
                self.spin.unsqueeze(0) == self.spin[i].unsqueeze(1)
            )
            # required spin of the partner b, given this a
            req = n_beta.unsqueeze(1) - self.spin.unsqueeze(0)                # (B, n)
            partner_exists = self.has_virt_after[
                req.clamp(0, 1), self.idx.unsqueeze(0).expand(batch, -1)
            ]
            a_double = (
                self.virt.unsqueeze(0) & (req >= 0) & (req <= 1) & partner_exists
            )
            a_ok = torch.where(is_single, a_single, a_double)
            return torch.cat([a_ok, false_col], dim=-1)

        if s == 3:                                   # choose b (virtual, b > a), or NONE
            i, j, a = picks[0], picks[1], picks[2]
            is_single = j.eq(n).unsqueeze(1)
            req_b = (self._n_beta(i, j) - self.spin[a]).unsqueeze(1)          # (B, 1)
            b_ok = (
                self.virt.unsqueeze(0)
                & (self.idx.unsqueeze(0) > a.unsqueeze(1))
                & (self.spin.unsqueeze(0) == req_b)
                & ~is_single
            )
            return torch.cat([b_ok, is_single], dim=-1)

        raise ValueError(f"step {s} out of range (expected 0..3)")


# ---------------------------------------------------------------------------
# Orbital encoder  (tier 2)
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    """Pre-norm transformer block with an additive pairwise attention bias."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x, bias):
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, attn_mask=bias, need_weights=False)
        x = x + attended
        return x + self.ff(self.norm2(x))


class OrbitalEncoder(nn.Module):
    """
    Graph transformer over the n spin-orbitals -> (n, H) keys.

    Fully connected: n is at most a few tens, so a cutoff graph would add
    machinery for no gain, and full attention needs no torch_geometric.

    The (n, n, pair_dim) edge features enter as an additive per-head attention
    bias, which is the standard graph-transformer treatment and keeps the module
    a plain nn.Module.

    Normalization follows OperatorScorer's design: statistics live in buffers so
    they can be computed once over a training set and then FROZEN, otherwise the
    same physical orbital would take different coordinates in different
    molecules and transfer would be undermined.
    """

    def __init__(self, orb_dim, pair_dim, hidden_size, num_layers=3,
                 num_heads=4, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(orb_dim, hidden_size)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, num_heads),
        )
        self.blocks = nn.ModuleList(
            [_Block(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_size)

        self.register_buffer("orb_mean", torch.zeros(1, orb_dim))
        self.register_buffer("orb_std", torch.ones(1, orb_dim))
        self.register_buffer("pair_mean", torch.zeros(1, 1, pair_dim))
        self.register_buffer("pair_std", torch.ones(1, 1, pair_dim))

    def set_normalization(self, orb_feats, pair_feats):
        """Fit the frozen statistics; call once, over the training molecules."""
        orb = torch.as_tensor(numpy.asarray(orb_feats), dtype=torch.float32)
        pair = torch.as_tensor(numpy.asarray(pair_feats), dtype=torch.float32)
        self.orb_mean.copy_(orb.mean(0, keepdim=True))
        self.orb_std.copy_(orb.std(0, keepdim=True).clamp_min(1e-6))
        flat = pair.reshape(-1, pair.shape[-1])
        self.pair_mean.copy_(flat.mean(0).view(1, 1, -1))
        self.pair_std.copy_(flat.std(0).clamp_min(1e-6).view(1, 1, -1))

    def forward(self, orb_feats, pair_feats):
        """orb_feats (n, orb_dim), pair_feats (n, n, pair_dim) -> (n, H)."""
        x = self.proj((orb_feats - self.orb_mean) / self.orb_std).unsqueeze(0)
        bias = self.pair_mlp((pair_feats - self.pair_mean) / self.pair_std)
        bias = bias.permute(2, 0, 1).contiguous()          # (heads, n, n)
        for block in self.blocks:
            x = block(x, bias)
        return self.out_norm(x.squeeze(0))                 # (n, H)


# ---------------------------------------------------------------------------
# Pointer head  (tier 3)
# ---------------------------------------------------------------------------

class ExcitationPointer(nn.Module):
    """
    Decode one gate as four pointers into the orbital table.

    A single code path serves both sampling (forced=None) and replay
    (forced=picks). They must agree exactly or the GRPO importance ratio is
    computed against a different distribution than the one that was sampled, so
    sharing the path is deliberate rather than an economy.

    Each step gets its OWN query projection: "which orbital do I excite out of"
    and "which do I excite into" are different questions and should not share a
    projection. From step 1 on, candidate keys are shifted by the pairwise
    features against the orbitals already chosen — this is where the coupling
    and the gap enter, and it is the only place integrals are read.
    """

    N_STEPS = 4

    def __init__(self, hidden_size: int, pair_dim: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.stop = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.step_q = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(self.N_STEPS)]
        )
        self.pair_proj = nn.ModuleList(
            [nn.Linear(pair_dim, hidden_size) for _ in range(self.N_STEPS)]
        )
        self.scale = hidden_size ** -0.5

    @staticmethod
    def _pair_context(pair_feats, picks, n):
        """
        (B, n + 1, pair_dim) — mean pairwise features against the orbitals picked
        so far. STOP picks contribute nothing; the STOP row is zero.
        """
        B = picks[0].shape[0]
        dp = pair_feats.shape[-1]
        acc = torch.zeros(B, n, dp, device=pair_feats.device, dtype=pair_feats.dtype)
        cnt = torch.zeros(B, 1, 1, device=pair_feats.device, dtype=pair_feats.dtype)
        for p in picks:
            real = (p < n).to(pair_feats.dtype).view(B, 1, 1)
            acc = acc + pair_feats[:, p.clamp(max=n - 1)].permute(1, 0, 2) * real
            cnt = cnt + real
        acc = acc / cnt.clamp_min(1.0)
        pad = torch.zeros(B, 1, dp, device=pair_feats.device, dtype=pair_feats.dtype)
        return torch.cat([acc, pad], dim=1)

    def forward(self, query, orb_keys, pair_feats, rules, inv_temperature,
                forced=None, return_entropy=False):
        """
        query      : (B, H)  circuit context — the DAG GNN's pooled frontier
        orb_keys   : (n, H)  from OrbitalEncoder
        pair_feats : (n, n, pair_dim)
        forced     : (B, 4) long to replay a stored gate, or None to sample

        Returns (picks (B, 4) long, step_logp (B, 4)), plus step entropies
        (B, 4) when return_entropy. Index n means STOP/NONE.
        """
        B = query.shape[0]
        n = rules.n
        keys_base = torch.cat([orb_keys, self.stop.unsqueeze(0)], dim=0)   # (n+1, H)

        h = query
        picks, logps, ents = [], [], []
        for s in range(self.N_STEPS):
            mask = rules.step_mask(s, picks, B)                            # (B, n+1)
            keys = keys_base.unsqueeze(0).expand(B, -1, -1)
            if picks:
                keys = keys + self.pair_proj[s](
                    self._pair_context(pair_feats, picks, n)
                )
            logits = torch.einsum("bh,bkh->bk", self.step_q[s](h), keys) * self.scale

            # Mask AFTER the -inv_temperature multiply. Masking first would turn
            # -inf into +inf and make forbidden orbitals maximally likely — the
            # same sign trap as CircuitDAGGNNPolicy._scaled_logits.
            scaled = (-inv_temperature) * logits
            scaled = scaled.masked_fill(~mask, float("-inf"))

            log_probs = F.log_softmax(scaled, dim=-1)
            pick = (forced[:, s] if forced is not None
                    else Categorical(logits=scaled).sample())
            logps.append(log_probs.gather(1, pick.unsqueeze(1)).squeeze(1))

            if return_entropy:
                # Masked entries have p = 0 and log p = -inf, whose product is
                # NaN. Selecting it away afterwards with torch.where is NOT
                # enough: the product is still computed, and its backward gives
                # d(p*logp)/dp = -inf, which times a zero incoming gradient is
                # NaN that then poisons every weight. Replace -inf BEFORE any
                # arithmetic touches it — same trap as dag_gnn.log_prob.
                probs = log_probs.exp()                    # exactly 0 if masked
                safe = torch.where(torch.isfinite(log_probs), log_probs,
                                   torch.zeros_like(log_probs))
                ents.append(-(probs * safe).sum(dim=-1))

            picks.append(pick)
            h = self.gru(keys_base[pick], h)

        stacked = (torch.stack(picks, dim=1), torch.stack(logps, dim=1))
        if return_entropy:
            return stacked + (torch.stack(ents, dim=1),)
        return stacked


# ---------------------------------------------------------------------------
# Small conversions
# ---------------------------------------------------------------------------

def gate_embedding(picks: torch.Tensor, orb_keys: torch.Tensor) -> torch.Tensor:
    """
    (B, H) DAG node embedding for a decoded gate: the mean of the orbital keys it
    touches. STOP entries are excluded, so a single and a double are handled by
    the same expression.

    Deliberately permutation-invariant over the four orbitals. The roles
    (occupied vs virtual) are already carried inside the keys themselves, via the
    occupancy column of the orbital features — so ordering the pooled orbitals
    would encode position, not physics, and would not transfer.

    This is what replaces the gate-node row of CircuitDAGGNNPolicy._node_table():
    in the pointer action space a gate has no operator ID to look up.
    """
    n = orb_keys.shape[0]
    real = (picks < n).to(orb_keys.dtype).unsqueeze(-1)        # (B, 4, 1)
    emb = orb_keys[picks.clamp(max=n - 1)]                     # (B, 4, H)
    return (emb * real).sum(dim=1) / real.sum(dim=1).clamp_min(1.0)


def excitation_qubits(row, n: int) -> list[int]:
    """
    Spin-orbitals touched by one decoded gate — its qubit footprint, which is
    what the DAG needs to wire the gate to its frontier.
    """
    i, j, a, b = (int(v) for v in row)
    qubits = [i, a]
    if j != n:
        qubits.append(j)
    if b != n:
        qubits.append(b)
    return sorted(qubits)


def excitation_pairs(row, n: int) -> list[tuple[int, int]]:
    """
    (occupied, virtual) index pairs in the form make_excitation_gate() expects,
    so a decoded gate can be handed to the existing tequila path unchanged.
    """
    i, j, a, b = (int(v) for v in row)
    pairs = [(a, i)]
    if j != n and b != n:
        pairs.append((b, j))
    return pairs
