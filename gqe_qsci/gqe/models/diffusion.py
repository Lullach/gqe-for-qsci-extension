import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

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
    ):
        super().__init__()
        self.vocab_size      = int(vocab_size)
        self.ngates          = int(ngates)
        self.diffusion_steps = int(diffusion_steps)

        tok_vocab = token_vocab_size if token_vocab_size is not None else self.vocab_size

        self.token_embedding    = nn.Embedding(tok_vocab, hidden_size)
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
        hidden = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
            + self.time_embedding(timestep).unsqueeze(1)
        )
        return self.output(self.denoiser(hidden))

    def act(self, state, temperature):
        raise RuntimeError(
            f"{self.__class__.__name__} generates whole sequences via "
            "sample_sequence(); act() is not supported."
        )


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
    ):
        super().__init__(
            vocab_size, ngates, hidden_size, num_layers,
            num_heads, diffusion_steps, dropout,
        )

    def sample_sequence(self, state, temperature):
        batch_size = state["idx"].shape[0]
        device     = state["idx"].device

        tokens = torch.randint(
            0, self.vocab_size,
            (batch_size, self.ngates),
            device=device,
        )
        for step in range(self.diffusion_steps, 0, -1):
            logits = self._logits(tokens, step)
            tokens = Categorical(logits=-temperature * logits).sample()

        state["idx"] = torch.cat((state["idx"], tokens), dim=1)
        return state

    def log_prob(self, indices, temperature, return_entropy=False):
        gate_tokens  = indices[:, 1:]
        batch_size   = gate_tokens.shape[0]
        device       = gate_tokens.device
        noisy_tokens = torch.zeros_like(gate_tokens)
        timestep     = torch.zeros(batch_size, dtype=torch.long, device=device)
        logits       = self._logits(noisy_tokens, timestep)
        log_probs    = F.log_softmax(-temperature * logits, dim=-1)
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

    log_prob
        Denoising ELBO averaged over all T timesteps — deterministic per
        sequence, which keeps the GRPO importance-weight ratio stable.

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
    ):
        # [MASK] lives at index vocab_size — just past the real vocabulary.
        super().__init__(
            vocab_size, ngates, hidden_size, num_layers,
            num_heads, diffusion_steps, dropout,
            token_vocab_size=vocab_size + 1,
        )
        self.mask_token    = self.vocab_size
        self.noise_schedule = noise_schedule

        alpha = _make_alpha_schedule(diffusion_steps, noise_schedule)
        # register_buffer: persistent (saved in checkpoints), moves with .to(device)
        self.register_buffer("alpha", alpha)   # shape [T+1]

    # --- forward process ---------------------------------------------------

    def _corrupt(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Sample  x_t ~ q(x_t | x_0).

        Each token is kept with probability α_t and replaced by [MASK]
        with probability (1 − α_t), independently across positions.

        x_0 : (B, L)  clean gate indices
        t   : (B,)    integer timesteps in [1, T]

        Returns
        -------
        x_t      : (B, L)  corrupted sequence  (some entries == mask_token)
        is_clean : (B, L)  bool — True where the original token was kept
        """
        alpha_t   = self.alpha[t].to(x_0.device)             # (B,)
        keep_prob = alpha_t.unsqueeze(1).expand_as(x_0)      # (B, L)
        is_clean  = torch.bernoulli(keep_prob).bool()         # (B, L)
        x_t = torch.where(
            is_clean,
            x_0,
            x_0.new_full(x_0.shape, self.mask_token),
        )
        return x_t, is_clean

    # --- reverse process / sampling ----------------------------------------

    def sample_sequence(self, state, temperature):
        """
        Generate a full gate sequence via the absorbing reverse process.

        Starts from a fully-masked sequence at t = T and iteratively
        reveals gate tokens from t = T down to t = 1.

        At each step t the fraction of currently-masked positions that get
        revealed is  (α_{t-1} − α_t) / (1 − α_t), which is the probability
        derived from the exact posterior  q(x_{t-1} | x_t, x_0).

        A token that has been revealed is never changed again.
        """
        batch_size = state["idx"].shape[0]
        device     = state["idx"].device

        # t = T: all positions start as [MASK]
        tokens = torch.full(
            (batch_size, self.ngates), self.mask_token,
            dtype=torch.long, device=device,
        )

        for step in range(self.diffusion_steps, 0, -1):
            is_masked = tokens.eq(self.mask_token)    # (B, L)
            if not is_masked.any():
                break

            # Model predicts the clean token x̂_0 at every position
            logits  = self._logits(tokens, step)                           # (B, L, V)
            x0_pred = Categorical(logits=-temperature * logits).sample()   # (B, L)

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

        # Safety net: fill any positions still masked after all steps
        # (can happen at very low T due to discretisation of p_reveal)
        still_masked = tokens.eq(self.mask_token)
        if still_masked.any():
            t0     = torch.zeros(batch_size, dtype=torch.long, device=device)
            tokens = torch.where(
                still_masked,
                self._logits(tokens, t0).argmax(dim=-1),
                tokens,
            )

        state["idx"] = torch.cat((state["idx"], tokens), dim=1)
        return state

    # --- mask pre-sampling ---------------------------------------------------

    def sample_masks(self, gate_tokens: torch.Tensor) -> torch.Tensor:
        """
        Pre-sample the corruption masks for all T timesteps.

        Calling this once during rollout collection and storing the result in
        the replay buffer removes the stochasticity from log_prob().  The GRPO
        importance-weight ratio  exp(log_p_new − log_p_old)  is then computed
        with exactly the same masked positions in both the old and new policy
        passes, making the ratio much more stable.

        gate_tokens : (B, L)  clean gate indices (no BOS)
        Returns     : (B, T, L) bool tensor — True where a token is masked.
                      Index convention: masks[:, t-1, :] corresponds to
                      timestep t  (1-indexed, matching the log_prob loop).
        """
        B, L   = gate_tokens.shape
        device = gate_tokens.device
        masks  = torch.zeros(
            B, self.diffusion_steps, L, dtype=torch.bool, device=device
        )
        for t_int in range(1, self.diffusion_steps + 1):
            t           = torch.full((B,), t_int, dtype=torch.long, device=device)
            _, is_clean = self._corrupt(gate_tokens, t)
            masks[:, t_int - 1, :] = ~is_clean
        return masks

    # --- log-probability ---------------------------------------------------

    def log_prob(self, indices, temperature, return_entropy=False, masks=None):
        """
        Estimate  log p_θ(x_0)  via the denoising ELBO averaged over all T
        timesteps.

        For each t in {1, …, T}:
          1. Corrupt x_0 → x_t (either from pre-sampled masks or fresh).
          2. Ask the model for  log p_θ(x_0[i] | x_t, t)  at masked positions.
        Average the per-token log-probs across timesteps.

        Parameters
        ----------
        masks : (B, T, L) bool tensor, optional
            Pre-sampled corruption masks as returned by sample_masks().
            When provided, the same masked positions used during rollout
            collection are reused here, making the GRPO importance-weight
            ratio  exp(log_p_new − log_p_old)  deterministic and stable.
            When None, fresh masks are sampled on every call (original
            behaviour — fine for logging / evaluation, noisy for training).

        Returns : (B, L) per-token log-probability estimates
        """
        gate_tokens = indices[:, 1:]          # strip BOS token  (B, L)
        B, L        = gate_tokens.shape
        device      = gate_tokens.device

        total_logp    = torch.zeros(B, L, device=device)
        total_entropy = torch.zeros(B, L, device=device) if return_entropy else None

        for t_int in range(1, self.diffusion_steps + 1):
            t = torch.full((B,), t_int, dtype=torch.long, device=device)

            if masks is not None:
                # Deterministic path: reconstruct x_t from stored masks
                is_masked = masks[:, t_int - 1, :]                        # (B, L)
                x_t = torch.where(
                    is_masked,
                    gate_tokens.new_full(gate_tokens.shape, self.mask_token),
                    gate_tokens,
                )
            else:
                # Stochastic path: sample fresh masks (eval / Simple compat)
                x_t, is_clean = self._corrupt(gate_tokens, t)
                is_masked     = ~is_clean                                  # (B, L)

            logits     = self._logits(x_t, t)                             # (B, L, V)
            log_probs  = F.log_softmax(-temperature * logits, dim=-1)     # (B, L, V)
            token_logp = torch.gather(
                log_probs, 2, gate_tokens.unsqueeze(-1)
            ).squeeze(-1)                                                  # (B, L)

            # Only masked positions require actual prediction
            total_logp += token_logp * is_masked.float()

            if return_entropy:
                entropy        = -(log_probs.exp() * log_probs).sum(dim=-1)
                total_entropy += entropy * is_masked.float()

        avg_logp = total_logp / self.diffusion_steps
        if return_entropy:
            return avg_logp, total_entropy / self.diffusion_steps
        return avg_logp


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
    ):
        # [MASK] lives at index vocab_size — just past the real vocabulary.
        # diffusion_steps sets the time-embedding table size; the model always
        # conditions on t = diffusion_steps (the fully-masked timestep).
        super().__init__(
            vocab_size, ngates, hidden_size, num_layers,
            num_heads, diffusion_steps, dropout,
            token_vocab_size=vocab_size + 1,
        )
        self.mask_token = self.vocab_size

    # --- reverse process / sampling ----------------------------------------

    def sample_sequence(self, state, temperature):
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
        tokens = Categorical(logits=-temperature * logits).sample()        # (B, L)

        state["idx"] = torch.cat((state["idx"], tokens), dim=1)
        return state

    # --- log-probability ---------------------------------------------------

    def log_prob(self, indices, temperature, return_entropy=False, masks=None):
        """
        Exact log p_θ(x_0 | x_T) — no ELBO approximation.

        The model always conditions on the fully-masked sequence at t = T,
        so log_prob is deterministic: the same sequence always yields the
        same value regardless of when it is called.  This gives the best
        possible importance-weight stability in GRPO without needing to
        store or replay masks.

        masks is accepted for API compatibility but silently ignored —
        there is no stochasticity to fix here.
        """
        gate_tokens = indices[:, 1:]          # strip BOS token  (B, L)
        B, L        = gate_tokens.shape
        device      = gate_tokens.device

        # Always condition on the fully-masked input at t = T
        masked    = gate_tokens.new_full((B, L), self.mask_token)
        t         = torch.full((B,), self.diffusion_steps, dtype=torch.long, device=device)

        logits    = self._logits(masked, t)                               # (B, L, V)
        log_probs = F.log_softmax(-temperature * logits, dim=-1)          # (B, L, V)
        token_logp = torch.gather(
            log_probs, 2, gate_tokens.unsqueeze(-1)
        ).squeeze(-1)                                                      # (B, L)

        if return_entropy:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            return token_logp, entropy
        return token_logp
