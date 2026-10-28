# CLAUDE 2026-10-28. Reference implementation of the direct / no-circuit arm,
# for Lukas to review. NOT run end to end (transformers lives only in the
# container); the mask is verified by scratch/check_determinant_mask.py.
# Lives outside gqe_qsci on purpose -- see direct_qsci/README.md.
"""
GPT-2 that emits a SET OF DETERMINANTS instead of a sequence of gates.

ONE TOKEN IS ONE SPIN-ORBITAL THAT HOLDS AN ELECTRON.
Not a determinant, and not one symbol per orbital.

    vocabulary   V = 2 * norb spin-orbitals, EVEN = alpha, ODD = beta
                 (the convention everywhere else in the repo: spatial = q // 2)
    determinant  T = n_alpha + n_beta tokens: the alpha ones first, then beta,
                 each group strictly increasing so a determinant has exactly
                 one spelling
    one sample   K determinants = K * T tokens

Toy example, 4 spatial orbitals (so V = 8), 2 alpha + 2 beta electrons:

    determinant         spatial picture      tokens (spin-orbitals)
    HF                  alpha 0,1 beta 0,1   0 2 | 1 3
    alpha single 1->2   alpha 0,2 beta 0,1   0 4 | 1 3

THE MASK, one expression (see `allowed_mask`), three clauses:
    right spin     alpha tokens even, beta odd
    increasing     i > prev, so one spelling per determinant
    room left      i // 2 <= norb - remaining, so the group can be finished
It admits exactly the C(norb, n_alpha) * C(norb, n_beta) valid determinants, so
`post_select_by_nelec` downstream is a no-op. The mask is applied INSIDE the
sampling loop, never as a post-filter: filtering afterwards would make log_prob
the probability of something other than what was used, and GRPO's importance
ratio would then compare two different distributions.

TRANSFER. A spin-orbital is embedded by its physical features (OperatorScorer
over pool.get_orbital_features()), never by index, so the same weights work on a
molecule with a different norb.

STATE CONTRACT (what sample_sequence writes):
    state["idx"]   (B, 1 + K*T) long   BOS + flat tokens. THE ACTION, and the
                                       ONLY thing written: putting it in "idx"
                                       means the replay buffer, the GRPO loss
                                       and training_step are unchanged.

The QSCI side needs determinants, not tokens, but that is a pure function of
the tokens, so the pipeline converts rather than the policy publishing a second
copy that could fall out of sync:

    tokens = state["idx"][:, 1:]                   # (B, K*T)
    T      = n_alpha + n_beta                      # from molecule.nelec
    packed = tokens_to_packed(tokens.view(B, -1, T), n_alpha)   # (B, K, 2)
"""
from __future__ import annotations

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gqe_qsci.gqe.models.operator_scorer import OperatorScorer
from gqe_qsci.gqe.models.policy import Policy


# CLAUDE
def allowed_mask(prev, spin, remaining, n_so: int) -> torch.Tensor:
    """
    Which spin-orbitals may be placed next.

    prev       (...)  last token of this group, or -1 when the group opens
    spin       (...)  0 for an alpha slot, 1 for a beta slot
    remaining  (...)  electrons still to place in this group, THIS ONE included
    returns    (..., n_so) bool

    Module-level and dependency-free on purpose: the mask is the one piece whose
    correctness decides whether log_prob scores what sampling did, so it is
    testable without a transformer. See scratch/check_determinant_mask.py.
    """
    norb = n_so // 2
    i = torch.arange(n_so, device=prev.device)
    i = i.view(*([1] * prev.dim()), n_so)              # broadcast against prev
    right_spin = (i % 2) == spin.unsqueeze(-1)
    increasing = i > prev.unsqueeze(-1)
    room_left = (i // 2) <= (norb - remaining).unsqueeze(-1)
    return right_spin & increasing & room_left


# CLAUDE
def tokens_to_packed(tokens: torch.Tensor, n_alpha: int) -> torch.Tensor:
    """
    (..., T) spin-orbital tokens -> (..., 2) int64 words [alpha, beta].

    Bit p of the alpha word is set when spatial orbital p holds an alpha
    electron, which is what Determinant([alpha, beta]) expects.
    """
    spatial = tokens // 2                                       # (..., T)
    alpha = torch.zeros(tokens.shape[:-1], dtype=torch.int64, device=tokens.device)
    beta = torch.zeros_like(alpha)
    for t in range(tokens.shape[-1]):
        bit = torch.ones_like(alpha) << spatial[..., t].to(torch.int64)
        alpha = alpha | bit if t < n_alpha else alpha
        beta = beta if t < n_alpha else beta | bit
    return torch.stack([alpha, beta], dim=-1)


# CLAUDE
def packed_to_tokens(packed, n_alpha: int, n_beta: int) -> torch.Tensor:
    """
    Inverse of tokens_to_packed: (..., 2) int64 [alpha, beta] words -> (..., T)
    spin-orbital tokens, sorted, alpha group then beta group.

    Plumbing, not physics: it exists so a test can hand the pipeline a KNOWN
    determinant set (e.g. the exact top-k) in the format the policy would have
    produced.
    """
    out = []
    for word_pair in packed.reshape(-1, 2).tolist():
        toks = []
        for spin, word in enumerate(word_pair):                # 0 = alpha, 1 = beta
            toks += [2 * p + spin for p in range(64) if (int(word) >> p) & 1]
        a = sorted(x for x in toks if x % 2 == 0)
        b = sorted(x for x in toks if x % 2 == 1)
        if len(a) != n_alpha or len(b) != n_beta:
            raise ValueError(f"expected {n_alpha}a/{n_beta}b, got {len(a)}a/{len(b)}b")
        out.append(a + b)
    return torch.tensor(out, dtype=torch.long).view(*packed.shape[:-1], n_alpha + n_beta)


class DeterminantGPT2Policy(Policy):
    """
    n_dets (K) is the determinant budget. FIX IT: the QSCI energy only falls as
    the subspace grows, so runs at different K are not comparable.
    """

    # CLAUDE
    def __init__(self, orbital_features, n_alpha: int, n_beta: int, n_dets: int,
                 hidden_size: int = 256, num_layers: int = 6, num_heads: int = 8,
                 dropout: float = 0.1, max_tokens: int | None = None):
        super().__init__()
        from transformers import GPT2Config, GPT2Model   # container-only import

        orb = numpy.asarray(orbital_features)            # (n_so, D), per spin-orbital
        self.n_so = int(orb.shape[0])
        self.n_alpha, self.n_beta = int(n_alpha), int(n_beta)
        self.tokens_per_det = self.n_alpha + self.n_beta
        self.n_dets = int(n_dets)
        self.n_tokens = self.n_dets * self.tokens_per_det
        # Capacity, not budget. n_dets and tokens_per_det both change with the
        # molecule, so the transformer is built for the LARGEST sequence any
        # molecule in the run will need; set_molecule then checks it fits.
        self.max_tokens = int(max_tokens or self.n_tokens)

        self.scorer = OperatorScorer(orb, hidden_size)   # keys: (n_so, H)
        self.bos = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.transformer = GPT2Model(GPT2Config(
            vocab_size=1,                                # unused: we pass embeddings
            n_positions=self.max_tokens + 1,             # + BOS.  NOT ngates + 1
            n_embd=hidden_size, n_layer=num_layers, n_head=num_heads,
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
        ))

    # CLAUDE
    def set_molecule(self, bundle):
        """Re-point at another molecule; n_so may change and nothing is resized,
        because the head scores feature-derived keys. Never call mid-rollout."""
        orb = numpy.asarray(bundle.orbital_features)
        self.scorer.set_features(orb, update_stats=False)
        self.n_so = int(orb.shape[0])
        self.n_alpha, self.n_beta = (int(x) for x in bundle.molecule.nelec)
        self.tokens_per_det = self.n_alpha + self.n_beta
        # Each molecule has its OWN measured determinant requirement, so the
        # budget travels on the bundle; falls back to the current one when the
        # bundle does not carry it.
        self.n_dets = int(getattr(bundle, "n_dets", None) or self.n_dets)
        self.n_tokens = self.n_dets * self.tokens_per_det
        if self.n_tokens > self.max_tokens:
            raise ValueError(
                f"{bundle.name if hasattr(bundle, 'name') else 'molecule'} needs "
                f"{self.n_tokens} tokens but the model was built for "
                f"{self.max_tokens}; set model.max_tokens to the largest "
                f"n_dets * (n_alpha + n_beta) across the molecule set."
            )

    # -- position bookkeeping (depends only on WHERE a token sits) ----------

    # CLAUDE
    def _slots(self, length: int, device):
        """
        For every token position: its spin, how many electrons remain in its
        group (itself included), and whether it opens a group.

        returns three (length,) long/bool tensors
        """
        t = torch.arange(length, device=device)
        in_det = t % self.tokens_per_det                 # 0..T-1 within a determinant
        is_beta = in_det >= self.n_alpha
        in_group = torch.where(is_beta, in_det - self.n_alpha, in_det)
        group_size = torch.where(is_beta,
                                 torch.full_like(in_det, self.n_beta),
                                 torch.full_like(in_det, self.n_alpha))
        return is_beta.long(), group_size - in_group, in_group == 0

    # -- the two things the transformer does --------------------------------

    # CLAUDE
    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) tokens -> (B, L, H). One gather; no block bookkeeping, because
        the key table already covers every spin-orbital."""
        return self.scorer.keys()[tokens]

    # CLAUDE
    def _hidden(self, tokens: torch.Tensor | None, batch: int, device) -> torch.Tensor:
        """
        BOS + tokens through GPT-2 -> (B, 1+L, H) hidden states.

        The teacher-forced path, used by log_prob: ONE pass over the whole
        sequence. Sampling uses the incremental path in sample_sequence.
        """
        bos = self.bos.view(1, 1, -1).expand(batch, 1, -1)          # (B, 1, H)
        inputs = bos if tokens is None else torch.cat([bos, self._embed(tokens)], 1)
        return self.transformer(inputs_embeds=inputs).last_hidden_state

    # CLAUDE
    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """(..., H) -> (..., n_so). Same dot-product-against-keys head as
        OperatorScorer, with spin-orbitals in place of operators."""
        keys = self.scorer.keys()                                   # (n_so, H)
        return hidden @ keys.T * keys.shape[-1] ** -0.5

    # -- Policy interface ---------------------------------------------------

    def act(self, state, inv_temperature):
        raise RuntimeError("DeterminantGPT2Policy generates whole sets via "
                           "sample_sequence(); act() is not supported.")

    # CLAUDE
    def sample_sequence(self, state, inv_temperature):
        B = state["idx"].shape[0]
        device = state["idx"].device
        spin, remaining, opens = self._slots(self.n_tokens, device)
        placed: list[torch.Tensor] = []                             # L x (B,)

        # Incremental decoding: the cache holds the prefix, so each step feeds
        # ONE embedding instead of re-running everything placed so far. GPT-2
        # derives the position from the cache length, so no position bookkeeping
        # is needed here. Same distribution as the uncached loop, ~L times less
        # compute -- which matters at 1020 tokens per sample.
        past = None
        step_input = self.bos.view(1, 1, -1).expand(B, 1, -1)       # (B, 1, H)

        for t in range(self.n_tokens):
            out = self.transformer(inputs_embeds=step_input,
                                   past_key_values=past, use_cache=True)
            past = out.past_key_values
            h = out.last_hidden_state[:, -1]                        # (B, H)
            scaled = (-inv_temperature) * self._logits(h)           # (B, n_so)

            prev = (torch.full((B,), -1, dtype=torch.long, device=device)
                    if bool(opens[t]) else placed[-1])              # (B,)
            mask = allowed_mask(prev, spin[t].expand(B), remaining[t].expand(B),
                                self.n_so)                          # (B, n_so)
            # Mask AFTER the -inv_temperature multiply: before it, -inf would
            # become +inf and the forbidden orbitals would be the likeliest.
            scaled = scaled.masked_fill(~mask, float("-inf"))
            tok = Categorical(logits=scaled).sample()               # (B,)
            placed.append(tok)
            step_input = self._embed(tok.unsqueeze(1))              # (B, 1, H)

        tokens = torch.stack(placed, 1)                             # (B, n_tokens)
        bos_col = torch.full((B, 1), self.n_so, dtype=torch.long, device=device)
        state["idx"] = torch.cat([bos_col, tokens], 1)              # (B, 1+n_tokens)
        # Deliberately no state["dets"]: determinants are a pure function of the
        # tokens, so the pipeline calls tokens_to_packed() and there is exactly
        # one source of truth for what was sampled.
        return state

    # CLAUDE
    def log_prob(self, indices, inv_temperature, return_entropy=False, reveal_step=None):
        """
        log p(token_1 ... token_{K*T}) = sum_t log p(token_t | prefix), exact.

        ONE teacher-forced pass. The masks are rebuilt from the stored tokens
        with the same `allowed_mask` the sampler used, so they cannot drift
        apart. Returns (B, K*T), the per-token decomposition GRPOLoss wants.
        """
        tokens = indices[:, 1:]                                     # (B, L), drop BOS
        B, L = tokens.shape
        spin, remaining, opens = self._slots(L, tokens.device)

        hidden = self._hidden(tokens, B, tokens.device)[:, :-1]     # (B, L, H)
        scaled = (-inv_temperature) * self._logits(hidden)          # (B, L, n_so)

        prev = torch.cat([torch.full((B, 1), -1, dtype=torch.long, device=tokens.device),
                          tokens[:, :-1]], 1)                       # (B, L)
        prev = prev.masked_fill(opens.view(1, L), -1)               # groups restart
        mask = allowed_mask(prev,                                   # (B, L, n_so)
                            spin.view(1, L).expand(B, L),
                            remaining.view(1, L).expand(B, L),
                            self.n_so)

        logp =F.log_softmax(scaled.masked_fill(~mask, float("-inf")), dim=-1)
        token_logp = logp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)   # (B, L)

        if return_entropy:
            # 0 * -inf is NaN, and its backward poisons every logit, so replace
            # the -inf BEFORE any arithmetic (same trap as dag_gnn.log_prob).
            probs = logp.exp()
            safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
            return token_logp, -(probs * safe).sum(-1)
        return token_logp
