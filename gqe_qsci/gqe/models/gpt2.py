# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Modifications Copyright (c) 2026 Ryota Kemmoku
# Modified from the original file in NVIDIA CUDA-QX.
# Changes made: added fast inference with KV cache and optimized repetition_penalty processing.


from transformers import GPT2LMHeadModel, GPT2Config
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import torch

from gqe_qsci.gqe.models.operator_scorer import OperatorScorer
from gqe_qsci.gqe.models.policy import Policy


class SmallConfig(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(n_layer=6, n_head=6, **kwargs)


class _FeatureWTE(torch.nn.Module):
    """
    Feature-based drop-in for GPT-2's ``transformer.wte``.

    HuggingFace calls ``self.wte(input_ids)`` expecting (B, L) -> (B, L, H).
    Here the embedding table is the OperatorScorer's keys, so token embeddings
    are derived from operator features rather than looked up by index.
    """

    def __init__(self, scorer: OperatorScorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.scorer.keys()[input_ids]


class _ScorerHead(torch.nn.Module):
    """
    Feature-based drop-in for GPT-2's ``lm_head``.

    Holds the SAME OperatorScorer instance as _FeatureWTE, so input embedding and
    output projection share one matrix — GPT-2's weight tying preserved, except
    the tied matrix is computed from features instead of being free parameters.
    (Registering the scorer under two parents is fine: nn.Module.parameters()
    de-duplicates by identity, so it is optimized once.)
    """

    def __init__(self, scorer: OperatorScorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.scorer(hidden_states)


class GPT2Model(GPT2LMHeadModel, Policy):
    def __init__(
        self,
        small,
        repetition_penalty,
        vocab_size,
        ngates,
        feature_scorer: bool = False,
        operator_features=None,
    ):
        max_positions = int(ngates) + 1
        gpt2cfg = GPT2Config(vocab_size=vocab_size, n_positions=max_positions)
        if small:
            gpt2cfg = SmallConfig(vocab_size=vocab_size, n_positions=max_positions)
        if feature_scorer:
            # HF's tie_weights() would try `lm_head.weight = wte.weight`; our
            # replacements have no .weight. We tie by sharing the scorer instead.
            gpt2cfg.tie_word_embeddings = False
        self.repetition_penalty = repetition_penalty
        super().__init__(gpt2cfg)
        self._tril_cache = {}

        self.feature_scorer = bool(feature_scorer)
        if self.feature_scorer:
            if operator_features is None:
                raise ValueError(
                    "feature_scorer=True requires operator_features "
                    "(factory passes pool.get_operator_features())."
                )
            # Swap AFTER super().__init__() so HF's post_init()/weight init has
            # already run over the original modules and cannot touch ours.
            # GPT-2 is the only model with TWO vocabulary touch points; both are
            # replaced here so nothing indexes operators by ID.
            scorer = OperatorScorer(operator_features, gpt2cfg.n_embd)
            self.set_input_embeddings(_FeatureWTE(scorer))   # transformer.wte
            self.lm_head = _ScorerHead(scorer)

    def set_molecule(self, bundle):
        """
        Re-point at bundle's molecule. wte and lm_head share ONE scorer, so
        swapping its features updates both the input embedding and the output
        head at once. n_positions (= ngates + 1) is molecule-independent and
        the repetition penalty operates on token indices, so nothing else
        changes. Never call mid-rollout.
        """
        if not self.feature_scorer:
            raise RuntimeError("set_molecule requires feature_scorer=True.")
        self.lm_head.scorer.set_features(bundle.operator_features, update_stats=False)

    def log_prob(self, indices, inv_temperature, return_entropy=False):
        """
        Compute next-token log-probabilities (and optionally entropies) under the same
        distribution as `act()` (i.e., includes repetition penalty + inv_temperature).

        Args:
          indices: (B, L) token ids
          return_entropy: bool, optional, default=False

        Returns:
          - log_prob: (B, L-1)  per-step next-token log p(a_t | prefix)
          - entropy:  (B, L-1)  per-step entropy H[p(.|prefix)] (only if return_entropy=True)
        """
        out = self(indices)
        logits = out.logits  # (B, L, V)

        # GPT-2 convention: logits at position t predict token t+1, so shift.
        logits = logits[:, :-1, :]   # (B, L-1, V)
        labels = indices[:, 1:]      # (B, L-1)
        prefix = indices[:, :-1]     # (B, L-1) tokens already in the prefix at each step

        # Apply repetition penalty across all time steps at once (faster than Python loops).
        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            logits = self._apply_repetition_penalty_sequence(
                logits=logits,
                prefix_ids=prefix,
                repetition_penalty=float(self.repetition_penalty),
            )

        log_probs = F.log_softmax(-inv_temperature * logits, dim=-1)                  # (B, L-1, V)
        token_logp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1) # (B, L-1)
        if return_entropy:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)                  # (B, L-1)
            return token_logp, entropy
        else:
            return token_logp

    def act(self, state, inv_temperature):
        """
        Incremental decoding:
        - If state["past_key_values"] exists, use KV cache and run a forward pass on the last token only.
        - Otherwise, run a full-prefix forward pass once to initialize the cache.
        """
        idx = state["idx"]
        past = state.get("past_key_values", None)
        if past is None:
            out = self(idx, use_cache=True)
        else:
            out = self(idx[:, -1:], past_key_values=past, use_cache=True)

        # Save KV cache into state (TrainPipeline.update_state keeps non-"idx" keys intact).
        state["past_key_values"] = out.past_key_values
        logits = out.logits[:, -1, :]  # (B, V)
        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            logits = self._apply_repetition_penalty_last(
                logits=logits,
                input_ids=idx,
                repetition_penalty=float(self.repetition_penalty),
            )
        probs = Categorical(logits=-inv_temperature * logits)
        next_token = probs.sample()
        return next_token

    @staticmethod
    def _apply_repetition_penalty_last(logits, input_ids, repetition_penalty: float):
        """
        Last-step variant (updates logits via scatter).
        logits: (B, V)
        input_ids: (B, L_prefix)
        """
        gathered = torch.gather(logits, 1, input_ids)  # (B, L_prefix)
        updated = torch.where(
            gathered < 0,
            gathered / repetition_penalty,
            gathered * repetition_penalty,
        )
        # scatter returns a new tensor (functional-style), so this is safe.
        return logits.scatter(1, input_ids, updated)

    def _get_tril_indices(self, S: int, device: torch.device):
        """
        tril indices (ti, sj) with sj <= ti, cached.
        S: sequence length (time steps in logits)
        """
        key = (S, device.type, device.index if device.type == "cuda" else -1)
        cached = self._tril_cache.get(key)
        if cached is not None:
            return cached
        # offset=0 includes the diagonal (prefix at step t includes positions 0..t).
        ti, sj = torch.tril_indices(S, S, offset=0, device=device)
        self._tril_cache[key] = (ti, sj)
        return ti, sj

    def _apply_repetition_penalty_sequence(self, logits, prefix_ids, repetition_penalty: float):
        """
        Fast repetition-penalty application across all time steps (no Python loop).

        logits:     (B, S, V) where S = L-1 (number of next-token prediction steps)
        prefix_ids: (B, S) tokens in the prefix per step (penalize tokens in positions 0..t)
        """
        B, S, V = logits.shape
        device = logits.device

        ti, sj = self._get_tril_indices(S, device)     # (M,), (M,),  M = S(S+1)/2

        # Select which "row" (b*S + t) in the flattened logits to update.
        batch_offsets = (torch.arange(B, device=device) * S).unsqueeze(1)  # (B, 1)
        rows = (batch_offsets + ti.unsqueeze(0)).reshape(-1)              # (B*M,)

        # Select which token id to penalize (prefix position s).
        cols = prefix_ids[:, sj].reshape(-1)                               # (B*M,)

        flat = logits.reshape(B * S, V)                                    # (B*S, V)

        # Gather target elements and compute their updated values.
        vals = flat[rows, cols]                                            # (B*M,)
        updated = torch.where(
            vals < 0,
            vals / repetition_penalty,
            vals * repetition_penalty,
        )
        out = flat.clone()
        out.index_put_((rows, cols), updated, accumulate=False)

        return out.view(B, S, V)