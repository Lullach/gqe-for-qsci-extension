import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

from gqe_qsci.gqe.models.operator_scorer import OperatorScorer, SpecialTokenEmbedding
from gqe_qsci.gqe.models.policy import Policy


# ---------------------------------------------------------------------------
# Noise schedule helper
# ---------------------------------------------------------------------------

def _make_alpha_schedule(diffusion_steps: int, schedule: str) -> torch.Tensor:
    """
    Return α_t for t = 0 … T  (length T+1).

        α_0 = 1.0  — fully clean (no masking)
        α_T = 0.0  — fully masked

    Supported schedules
    -------------------
    "cosine"  : α_t = cos²(π/2 · t/T)
                Smooth; distributes most unmasking in the middle steps.
                Standard choice in MDLM / D3PM literature.
    "linear"  : α_t = 1 − t/T
                Uniform unmasking per step; simpler baseline.
    """
    t = torch.arange(diffusion_steps + 1, dtype=torch.float32)
    if schedule == "cosine":
        return torch.cos(math.pi / 2.0 * t / diffusion_steps) ** 2
    elif schedule == "linear":
        return 1.0 - t / diffusion_steps
    else:
        raise ValueError(
            f"Unknown noise schedule '{schedule}'. Supported: 'cosine', 'linear'."
        )


# ---------------------------------------------------------------------------
# Shared transformer backbone
# ---------------------------------------------------------------------------

class _CircuitDiffusionBase(Policy):
    """
    Shared backbone for all circuit diffusion variants.

    Provides:
      - token / position / time embeddings
      - TransformerEncoder denoiser
      - _logits()  — single forward pass
      - act()      — raises RuntimeError (diffusion is whole-sequence only)

    Subclasses must implement sample_sequence() and log_prob().

    Parameters
    ----------
    token_vocab_size : int, optional
        Size of the token embedding table. Defaults to vocab_size.
        Pass vocab_size + 1 when a subclass needs an extra special token
        (e.g. a [MASK] token) without changing the output layer size.
    feature_scorer : bool
        Replace the integer-ID token embedding and the Linear output head with
        feature-based equivalents (OperatorScorer), making the model portable
        across molecules. Requires operator_features. See NOTES.md,
        "Cross-molecule generalization".
    operator_features : (V, feat_dim) array, optional
        The operator menu; required when feature_scorer=True.
    """

    def __init__(
        self,
        vocab_size: int,
        ngates: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        diffusion_steps: int,
        dropout: float,
        token_vocab_size: int | None = None,
        feature_scorer: bool = False,
        operator_features=None,
    ):
        super().__init__()
        self.vocab_size      = int(vocab_size)
        self.ngates          = int(ngates)
        self.diffusion_steps = int(diffusion_steps)
        self.feature_scorer  = bool(feature_scorer)

        tok_vocab = token_vocab_size if token_vocab_size is not None else self.vocab_size
        # Special tokens live just past the real vocabulary (e.g. [MASK] at
        # index vocab_size); everything below vocab_size is a real operator.
        n_special = tok_vocab - self.vocab_size

        if self.feature_scorer:
            if operator_features is None:
                raise ValueError(
                    "feature_scorer=True requires operator_features "
                    "(factory passes pool.get_operator_features())."
                )
            self.token_embedding = SpecialTokenEmbedding(n_special, hidden_size)
        else:
            self.token_embedding = nn.Embedding(tok_vocab, hidden_size)
        self.position_embedding = nn.Embedding(self.ngates, hidden_size)
        # Covers t = 0 … diffusion_steps  (T+1 entries)
        self.time_embedding     = nn.Embedding(self.diffusion_steps + 1, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.denoiser = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output always spans the *real* gate vocab — never special tokens.
        if self.feature_scorer:
            self.output = OperatorScorer(operator_features, hidden_size)
        else:
            self.output = nn.Linear(hidden_size, self.vocab_size)

    def _logits(self, tokens, timestep):
        """
        tokens   : (B, L)  gate indices; may include special tokens in subclasses
        timestep : int  or  (B,) long tensor

        Returns  : (B, L, vocab_size)  logits over real gate tokens
        """
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        if not torch.is_tensor(timestep):
            timestep = torch.full(
                (batch_size,), int(timestep),
                dtype=torch.long, device=tokens.device,
            )
        timestep = timestep.clamp(0, self.diffusion_steps)

        if self.feature_scorer:
            # Encode the menu once and use it for BOTH the input embedding and
            # the output head — feature-space weight tying, GPT-2 style.
            keys = self.output.keys()                          # (V, H)
            tok_emb = self.token_embedding(tokens, keys)
        else:
            keys = None
            tok_emb = self.token_embedding(tokens)

        hidden = (
            tok_emb
            + self.position_embedding(positions)
            + self.time_embedding(timestep).unsqueeze(1)
        )
        h = self.denoiser(hidden)
        if self.feature_scorer:
            return self.output(h, keys=keys)                   # (B, L, V)
        return self.output(h)

    def act(self, state, inv_temperature):
        raise RuntimeError(
            f"{self.__class__.__name__} generates whole sequences via "
            "sample_sequence(); act() is not supported."
        )

    def set_molecule(self, bundle):
        """
        Re-point at bundle's molecule. Swaps the operator menu (which is also the
        tied input embedding table). Subclasses with a [MASK] token also update
        it here via _refresh_mask_token(), since its index is vocab_size.
        Never call mid-rollout.
        """
        if not self.feature_scorer:
            raise RuntimeError(
                "set_molecule requires feature_scorer=True."
            )
        self.vocab_size = int(bundle.vocab_size)
        self.output.set_features(bundle.operator_features, update_stats=False)
        self._refresh_mask_token()

    def _refresh_mask_token(self):
        """Hook: subclasses with a [MASK] token (index == vocab_size) reset it."""
        pass


# ---------------------------------------------------------------------------
# V1 — Original simplified model  (kept for comparison / backward compat)
# ---------------------------------------------------------------------------

class CircuitDiffusionModelSimple(_CircuitDiffusionBase):
    """
    Original discrete-diffusion-*inspired* policy (first pass, kept for
    comparison and backward compatibility with existing experiment configs).

    Sampling  : starts from uniformly random tokens; replaces *all* positions
                at every denoising step — no principled forward process.
    log_prob  : simplified proxy evaluated at all-zero context with t=0.

    Use config  model=diffusion  to select this variant.
    """

    def __init__(
        self,
        vocab_size,
        ngates,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        diffusion_steps=8,
        dropout=0.1,
        feature_scorer=False,
        operator_features=None,
    ):
        super().__init__(
            vocab_size, ngates, hidden_size, num_layers,
            num_heads, diffusion_steps, dropout,
            feature_scorer=feature_scorer,
            operator_features=operator_features,
        )

    def sample_sequence(self, state, inv_temperature):
        batch_size = state["idx"].shape[0]
        device     = state["idx"].device

        tokens = torch.randint(
            0, self.vocab_size,
            (batch_size, self.ngates),
            device=device,
        )
        for step in range(self.diffusion_steps, 0, -1):
            logits = self._logits(tokens, step)
            tokens = Categorical(logits=-inv_temperature * logits).sample()

        state["idx"] = torch.cat((state["idx"], tokens), dim=1)
        return state

    def log_prob(self, indices, inv_temperature, return_entropy=False):
        gate_tokens  = indices[:, 1:]
        batch_size   = gate_tokens.shape[0]
        device       = gate_tokens.device
        noisy_tokens = torch.zeros_like(gate_tokens)
        timestep     = torch.zeros(batch_size, dtype=torch.long, device=device)
        logits       = self._logits(noisy_tokens, timestep)
        log_probs    = F.log_softmax(-inv_temperature * logits, dim=-1)
        token_logp   = torch.gather(
            log_probs, 2, gate_tokens.unsqueeze(-1)
        ).squeeze(-1)
        if return_entropy:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            return token_logp, entropy
        return token_logp


# ---------------------------------------------------------------------------
# V2 — Absorbing diffusion  (principled forward / reverse process)
# ---------------------------------------------------------------------------

class CircuitDiffusionModelAbsorbing(_CircuitDiffusionBase):
    """
    Absorbing-diffusion policy for gate-sequence generation.

    Forward process  q(x_t | x_0)
        Each gate is independently replaced by a [MASK] token with
        probability (1 − α_t), where α_t is given by the noise schedule.
        α_0 = 1 (fully clean), α_T = 0 (fully masked).

    Reverse process  p_θ(x_{t-1} | x_t)
        Uses the closed-form posterior q(x_{t-1} | x_t, x̂_0), where x̂_0
        is sampled from the model's prediction at each step.
        Positions that are already revealed are never changed again.

    log_prob — exact TRAJECTORY log-probability (DDPO-style)
        sample_sequence records which timestep committed each position
        (state["reveal_step"]); log_prob replays exactly those decisions:

            log p_θ(τ) = Σ_t Σ_{i committed at t} log p_θ(x_0[i] | x_t, t)

        This is EXACT — it is the probability of the action sequence the
        policy actually took, which is what a policy gradient needs. The
        θ-independent reveal coins cancel in the GRPO importance ratio, so
        only these categorical terms remain.

        Replaces an earlier denoising-ELBO implementation (with pre-sampled
        corruption masks via sample_masks()). The ELBO bounds log p(x_0) and
        measures reconstructability, which does NOT correspond to the
        sampler's distribution; it also required freezing corruption masks in
        the replay buffer to keep the ratio deterministic. The trajectory form
        needs none of that, is cheaper, and matches what the single-shot and
        DAG-GNN policies already do. See NOTES.md, "Trajectory log-prob (DDPO)
        replaces the ELBO"; the ELBO version is in git history.

    Noise schedule
        Controlled by the 'noise_schedule' constructor argument
        ('cosine' or 'linear'). Add new schedules to _make_alpha_schedule().
        diffusion_steps (T) is also fully configurable; larger T gives more
        denoising steps and a finer-grained reverse process.

    Use config  model=diffusion_absorbing  to select this variant.
    """

    def __init__(
        self,
        vocab_size,
        ngates,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        diffusion_steps=8,
        noise_schedule="cosine",
        dropout=0.1,
        feature_scorer=False,
        operator_features=None,
    ):
        # [MASK] lives at index vocab_size — just past the real vocabulary.
        super().__init__(
            vocab_size, ngates, hidden_size, num_layers,
            num_heads, diffusion_steps, dropout,
            token_vocab_size=vocab_size + 1,
            feature_scorer=feature_scorer,
            operator_features=operator_features,
        )
        self.mask_token    = self.vocab_size
        self.noise_schedule = noise_schedule

        alpha = _make_alpha_schedule(diffusion_steps, noise_schedule)
        # register_buffer: persistent (saved in checkpoints), moves with .to(device)
        self.register_buffer("alpha", alpha)   # shape [T+1]

    def _refresh_mask_token(self):
        # [MASK] lives just past the (possibly changed) real vocabulary.
        self.mask_token = self.vocab_size

    # --- reverse process / sampling ----------------------------------------

    def sample_sequence(self, state, inv_temperature):
        """
        Generate a full gate sequence via the absorbing reverse process.

        Starts from a fully-masked sequence at t = T and iteratively
        reveals gate tokens from t = T down to t = 1.

        At each step t the fraction of currently-masked positions that get
        revealed is  (α_{t-1} − α_t) / (1 − α_t), which is the probability
        derived from the exact posterior  q(x_{t-1} | x_t, x_0).

        A token that has been revealed is never changed again.

        Records the TRAJECTORY in state["reveal_step"] — a (B, L) long tensor
        holding, for each position, the timestep at which it was committed.
        log_prob() needs it to score the exact path that was sampled; see the
        class docstring.
        """
        batch_size = state["idx"].shape[0]
        device     = state["idx"].device

        # t = T: all positions start as [MASK]
        tokens = torch.full(
            (batch_size, self.ngates), self.mask_token,
            dtype=torch.long, device=device,
        )
        # reveal_step[b, i] = t at which position i was committed.
        # 0 is reserved for the safety net below (see there).
        reveal_step = torch.zeros(
            batch_size, self.ngates, dtype=torch.long, device=device
        )

        for step in range(self.diffusion_steps, 0, -1):
            is_masked = tokens.eq(self.mask_token)    # (B, L)
            if not is_masked.any():
                break

            # Model predicts the clean token x̂_0 at every position
            logits  = self._logits(tokens, step)                           # (B, L, V)
            x0_pred = Categorical(logits=-inv_temperature * logits).sample()   # (B, L)

            # Probability of revealing a masked token at this step:
            #   p_reveal = (α_{t-1} − α_t) / (1 − α_t)
            alpha_t    = self.alpha[step]
            alpha_prev = self.alpha[step - 1]
            p_reveal   = ((alpha_prev - alpha_t) / (1.0 - alpha_t + 1e-8)).clamp(0.0, 1.0)

            # Sample which masked positions to reveal
            reveal = (
                torch.bernoulli(
                    torch.full_like(tokens, p_reveal.item(), dtype=torch.float)
                )
                .bool()
                .logical_and(is_masked)
            )
            tokens = torch.where(reveal, x0_pred, tokens)
            reveal_step = torch.where(
                reveal, torch.full_like(reveal_step, step), reveal_step
            )

        # Safety net: fill any position still masked after the loop.
        # p_reveal at t=1 is (α_0 − α_1)/(1 − α_1 + 1e-8) with α_0 = 1 exactly,
        # i.e. ~1 − 7e-8, so this fires only on an astronomically unlikely
        # Bernoulli miss. Sample (rather than argmax) so the token still has a
        # well-defined sampling log-probability, and leave reveal_step at 0 —
        # log_prob's "visible iff reveal_step > t" rule then handles t=0
        # uniformly with every other step.
        still_masked = tokens.eq(self.mask_token)
        if still_masked.any():
            t0      = torch.zeros(batch_size, dtype=torch.long, device=device)
            logits0 = self._logits(tokens, t0)
            sampled = Categorical(logits=-inv_temperature * logits0).sample()
            tokens  = torch.where(still_masked, sampled, tokens)

        state["idx"]         = torch.cat((state["idx"], tokens), dim=1)
        state["reveal_step"] = reveal_step
        return state

    # --- log-probability ---------------------------------------------------

    def log_prob(self, indices, inv_temperature, return_entropy=False, reveal_step=None):
        """
        Exact log-probability of the sampled reverse TRAJECTORY (DDPO-style).

            log p_θ(τ) = Σ_t Σ_{i committed at t} log p_θ(x_0[i] | x_t, t)

        Each position is scored exactly once — at the step it was committed —
        so this returns (B, L) with no averaging.

        Why the trajectory and not the ELBO: GRPO needs the probability of the
        action sequence actually taken, and the reveal coins are θ-independent
        so they cancel in the importance ratio exp(log p_new − log p_old),
        leaving only the categorical terms above. This is exact, whereas the
        ELBO is a bound on log p(x_0) that does not correspond to the sampler.
        See NOTES.md, "Trajectory log-prob (DDPO) replaces the ELBO".

        x_t is reconstructed from reveal_step: position i is visible at step t
        iff reveal_step[i] > t (it was committed at some later, i.e. larger,
        step). Only the steps that actually committed something are visited.

        Parameters
        ----------
        reveal_step : (B, L) long, REQUIRED
            Per-position commit timestep, as recorded by sample_sequence into
            state["reveal_step"] and carried through the replay buffer.

        Returns : (B, L) per-position log-probabilities
        """
        gate_tokens = indices[:, 1:]          # strip BOS token  (B, L)
        B, L        = gate_tokens.shape
        device      = gate_tokens.device

        if reveal_step is None:
            raise ValueError(
                f"{type(self).__name__}.log_prob requires reveal_step (the "
                "trajectory recorded by sample_sequence into "
                "state['reveal_step']). A trajectory log-probability is "
                "undefined without the trajectory that produced it."
            )
        reveal_step = reveal_step.to(device)

        total_logp    = torch.zeros(B, L, device=device)
        total_entropy = torch.zeros(B, L, device=device) if return_entropy else None

        # Only steps that committed something anywhere in the batch matter.
        # Descending mirrors the reverse process, though order is irrelevant
        # here since each step is scored independently from its own x_t.
        steps = sorted(reveal_step.unique().tolist(), reverse=True)

        for t_int in steps:
            committed = reveal_step.eq(t_int)                              # (B, L)
            # State the model saw at step t: everything committed LATER
            # (larger t) is already visible; the rest is still [MASK].
            visible = reveal_step.gt(t_int)                                # (B, L)
            x_t = torch.where(
                visible,
                gate_tokens,
                gate_tokens.new_full(gate_tokens.shape, self.mask_token),
            )

            t          = torch.full((B,), t_int, dtype=torch.long, device=device)
            logits     = self._logits(x_t, t)                              # (B, L, V)
            log_probs  = F.log_softmax(-inv_temperature * logits, dim=-1)  # (B, L, V)
            token_logp = torch.gather(
                log_probs, 2, gate_tokens.unsqueeze(-1)
            ).squeeze(-1)                                                  # (B, L)

            # Score ONLY the positions committed at this step.
            total_logp = total_logp + token_logp * committed.float()

            if return_entropy:
                entropy       = -(log_probs.exp() * log_probs).sum(dim=-1)
                total_entropy = total_entropy + entropy * committed.float()

        if return_entropy:
            return total_logp, total_entropy
        return total_logp


# ---------------------------------------------------------------------------
# V3 — Single-shot absorbing diffusion  (USS / consistency-model analogue)
# ---------------------------------------------------------------------------

class CircuitDiffusionModelSingleShot(_CircuitDiffusionBase):
    """
    Single-shot absorbing diffusion policy for gate-sequence generation.

    Inspired by the Unitary Single-Sampling (USS) architecture in
    "Quantum Denoising Diffusion Models" (Kölle et al., 2024, arXiv:2401.07049).
    In the quantum setting T unitary denoising steps compose into one matrix,
    collapsing the entire reverse process into a single circuit execution.
    Here the classical analogue is: train the model to predict x_0 directly
    from x_T (the fully-masked sequence) in one forward pass — no iterative
    refinement, no ELBO approximation.

    Forward process  q(x_T | x_0)
        Deterministic: every position is [MASK].  No noise schedule needed.

    Reverse process  p_θ(x_0 | x_T)
        A single forward pass of the transformer conditioned on t = T.
        All positions are predicted simultaneously.

    log_prob  log p_θ(x_0 | x_T)
        Exact (no ELBO averaging), fully deterministic — the best possible
        importance-weight stability for GRPO.

    Compared with absorbing (T=16):
        Inference  : 1 forward pass instead of T  (T× faster)
        log_prob   : exact, no mask sampling needed
        Trade-off  : the model must learn a harder single-step mapping;
                     may benefit from warm-starting from a trained absorbing
                     checkpoint (weights are architecturally compatible).

    Use config  model=diffusion_singleshot  to select this variant.
    """

    def __init__(
        self,
        vocab_size,
        ngates,
        hidden_size=256,
        num_layers=8,
        num_heads=8,
        diffusion_steps=16,
        dropout=0.1,
        feature_scorer=False,
        operator_features=None,
    ):
        # [MASK] lives at index vocab_size — just past the real vocabulary.
        # diffusion_steps sets the time-embedding table size; the model always
        # conditions on t = diffusion_steps (the fully-masked timestep).
        #
        # NOTE: this model never embeds a real operator token — _logits() only
        # ever receives an all-[MASK] sequence, at both sample and log_prob time.
        # So the token embedding is only ever the [MASK] vector, and the output
        # head is the model's ONLY vocabulary touch point. That makes it the
        # cheapest model to port across molecules. See NOTES.md.
        super().__init__(
            vocab_size, ngates, hidden_size, num_layers,
            num_heads, diffusion_steps, dropout,
            token_vocab_size=vocab_size + 1,
            feature_scorer=feature_scorer,
            operator_features=operator_features,
        )
        self.mask_token = self.vocab_size

    def _refresh_mask_token(self):
        self.mask_token = self.vocab_size

    # --- reverse process / sampling ----------------------------------------

    def sample_sequence(self, state, inv_temperature):
        """
        Generate a gate sequence in a single forward pass.

        The input is the fully-masked sequence (x_T), conditioned on
        timestep t = T.  All positions are sampled simultaneously from
        the model's predicted distribution — no iterative unmasking.
        """
        batch_size = state["idx"].shape[0]
        device     = state["idx"].device

        # x_T: all positions masked
        tokens = torch.full(
            (batch_size, self.ngates), self.mask_token,
            dtype=torch.long, device=device,
        )

        # Single forward pass conditioned on t = T
        logits = self._logits(tokens, self.diffusion_steps)               # (B, L, V)
        tokens = Categorical(logits=-inv_temperature * logits).sample()        # (B, L)

        state["idx"] = torch.cat((state["idx"], tokens), dim=1)
        return state

    # --- log-probability ---------------------------------------------------

    def log_prob(self, indices, inv_temperature, return_entropy=False, reveal_step=None):
        """
        Exact log p_θ(x_0 | x_T) — no ELBO approximation.

        The model always conditions on the fully-masked sequence at t = T,
        so log_prob is deterministic: the same sequence always yields the
        same value regardless of when it is called.  This gives the best
        possible importance-weight stability in GRPO.

        This IS already a trajectory log-probability — the trajectory has a
        single step (x_T -> x_0), so there is nothing to record and
        reveal_step is accepted for API compatibility but ignored.
        """
        gate_tokens = indices[:, 1:]          # strip BOS token  (B, L)
        B, L        = gate_tokens.shape
        device      = gate_tokens.device

        # Always condition on the fully-masked input at t = T
        masked    = gate_tokens.new_full((B, L), self.mask_token)
        t         = torch.full((B,), self.diffusion_steps, dtype=torch.long, device=device)

        logits    = self._logits(masked, t)                               # (B, L, V)
        log_probs = F.log_softmax(-inv_temperature * logits, dim=-1)          # (B, L, V)
        token_logp = torch.gather(
            log_probs, 2, gate_tokens.unsqueeze(-1)
        ).squeeze(-1)                                                      # (B, L)

        if return_entropy:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            return token_logp, entropy
        return token_logp
