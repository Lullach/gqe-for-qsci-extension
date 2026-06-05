import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gqe_qsci.gqe.models.policy import Policy
from gqe_qsci.gqe.models.diffusion import _make_alpha_schedule

# ---------------------------------------------------------------------------
# NOTE: alternative GNN implementations
# ---------------------------------------------------------------------------
# The current implementation uses Graph Attention Networks (GATConv) with a
# chain graph. Other options documented in NOTES.md include:
#
#   Layer choices  : GCNConv, GINConv, GraphSAGE  (swap GATConv below)
#   Graph topology : "full" (all pairs), qubit-sharing (dynamic, future work)
#   Future graphs  : (anti)commutation graph, hardware connectivity graph
#                    (see NOTES.md — "Future extensions — additional graph inputs")
#
# The _CircuitGNNBase._logits() method is the only thing that needs changing
# to try a different layer type; everything else (sample_sequence, log_prob,
# training loop) stays identical.
# ---------------------------------------------------------------------------

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
            "Install it with:  pip install torch_geometric\n"
            "Or add it to the Docker run command:  "
            "pip install torch_geometric && python3 train.py ..."
        )


# ---------------------------------------------------------------------------
# Shared GNN backbone
# ---------------------------------------------------------------------------

class _CircuitGNNBase(Policy):
    """
    GNN backbone for circuit diffusion models.

    Replaces the TransformerEncoder in _CircuitDiffusionBase with a stack of
    GAT (Graph Attention Network) layers.  Nodes are gate positions (L nodes);
    edges encode positional relationships via the chosen graph topology.

    The three embeddings (token, position, time) and the output projection are
    identical to _CircuitDiffusionBase — only the denoiser changes.

    Parameters
    ----------
    graph_type : str
        "chain"  Bidirectional chain i ↔ i+1.  Encodes gate ordering directly.
                 After k layers, information travels k hops; 6 layers covers
                 the full L=10 sequence.  Recommended default.
        "full"   All pairs connected.  Equivalent to Transformer without the
                 softmax weighting.  Useful as an ablation baseline; loses the
                 GNN's sparsity advantage.
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
        graph_type: str = "chain",
        token_vocab_size: int | None = None,
    ):
        _require_pyg()
        super().__init__()
        self.vocab_size      = int(vocab_size)
        self.ngates          = int(ngates)
        self.diffusion_steps = int(diffusion_steps)
        self.hidden_size     = int(hidden_size)

        tok_vocab = token_vocab_size if token_vocab_size is not None else self.vocab_size

        # Same embeddings as _CircuitDiffusionBase
        self.token_embedding    = nn.Embedding(tok_vocab, hidden_size)
        self.position_embedding = nn.Embedding(self.ngates, hidden_size)
        self.time_embedding     = nn.Embedding(self.diffusion_steps + 1, hidden_size)

        # GAT layers replacing TransformerEncoder.
        # concat=False: multi-head outputs are averaged, not concatenated, so
        # the hidden dimension stays constant across layers — required for
        # residual connections to work.
        self.gnn_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_size,
                out_channels=hidden_size,
                heads=num_heads,
                concat=False,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        # Per-layer LayerNorm prevents over-smoothing in deep GNNs
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Output projection over the real gate vocabulary
        self.output = nn.Linear(hidden_size, self.vocab_size)

        # Static edge index — computed once, stored as a buffer so it moves
        # to the correct device automatically with .to(device)
        edge_index = _build_edge_index(ngates, graph_type)
        self.register_buffer("edge_index", edge_index)   # (2, E)

    def _batch_edge_index(self, batch_size: int) -> torch.Tensor:
        """
        Replicate the single-graph edge index for a batch of B graphs by
        treating them as one large disconnected graph.

        For sample b, node i maps to global node b*ngates + i.
        Returns (2, B*E) long tensor.
        """
        offsets = (
            torch.arange(batch_size, device=self.edge_index.device) * self.ngates
        )                                                          # (B,)
        ei = self.edge_index.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, E)
        ei = ei + offsets.view(-1, 1, 1)
        return ei.permute(1, 0, 2).reshape(2, -1)                 # (2, B*E)

    def _logits(self, tokens: torch.Tensor, timestep) -> torch.Tensor:
        """
        tokens   : (B, L)  gate indices (may include [MASK] special token)
        timestep : int or (B,) long tensor

        Returns  : (B, L, vocab_size) logits over real gate tokens
        """
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        if not torch.is_tensor(timestep):
            timestep = torch.full(
                (batch_size,), int(timestep),
                dtype=torch.long, device=tokens.device,
            )
        timestep = timestep.clamp(0, self.diffusion_steps)

        # Node features: sum of three embeddings  (B, L, H)
        h = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
            + self.time_embedding(timestep).unsqueeze(1)
        )

        # Flatten to (B*L, H) for PyG message passing
        h = h.view(batch_size * seq_len, -1)

        # Build batched edge index for B disconnected graphs
        edge_index = self._batch_edge_index(batch_size)

        # Message passing: GAT + residual connection + LayerNorm per layer
        for gnn_layer, norm in zip(self.gnn_layers, self.layer_norms):
            h = h + F.gelu(norm(gnn_layer(h, edge_index)))

        # Reshape back to (B, L, H) and project to vocab
        h = h.view(batch_size, seq_len, -1)
        return self.output(h)                                      # (B, L, vocab_size)

    def act(self, state, temperature):
        raise RuntimeError(
            f"{self.__class__.__name__} generates whole sequences via "
            "sample_sequence(); act() is not supported."
        )


# ---------------------------------------------------------------------------
# Edge index construction helper
# ---------------------------------------------------------------------------

def _build_edge_index(ngates: int, graph_type: str) -> torch.Tensor:
    """
    Build the static (2, E) edge index for a single graph of ngates nodes.

    "chain" : bidirectional chain  i ↔ i+1
    "full"  : all directed pairs   i → j for i ≠ j
    """
    if graph_type == "chain":
        fwd_src = list(range(ngates - 1))
        fwd_dst = list(range(1, ngates))
        src = fwd_src + fwd_dst
        dst = fwd_dst + fwd_src
    elif graph_type == "full":
        pairs = [(i, j) for i in range(ngates) for j in range(ngates) if i != j]
        if pairs:
            src, dst = zip(*pairs)
        else:
            src, dst = [], []
    else:
        raise ValueError(
            f"Unknown graph_type '{graph_type}'. Supported: 'chain', 'full'."
        )
    return torch.tensor([list(src), list(dst)], dtype=torch.long)


# ---------------------------------------------------------------------------
# Absorbing diffusion with GNN denoiser
# ---------------------------------------------------------------------------

class CircuitGNNModelAbsorbing(_CircuitGNNBase):
    """
    Absorbing-diffusion policy using a GNN denoiser instead of TransformerEncoder.

    Forward process, reverse process, mask pre-sampling and ELBO log_prob are
    identical to CircuitDiffusionModelAbsorbing — only the backbone changes.

    NOTE: The diffusion logic is intentionally duplicated from
    CircuitDiffusionModelAbsorbing rather than using multiple inheritance, to
    keep the class self-contained and avoid cooperative-super pitfalls.
    If you change the absorbing logic in diffusion.py, update it here too.
    A mixin-based refactor that removes the duplication is described in NOTES.md.

    Use config  model=gnn_absorbing  to select this variant.
    """

    def __init__(
        self,
        vocab_size,
        ngates,
        hidden_size: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        diffusion_steps: int = 16,
        noise_schedule: str = "cosine",
        dropout: float = 0.1,
        graph_type: str = "chain",
    ):
        # [MASK] token lives at index vocab_size
        super().__init__(
            vocab_size, ngates, hidden_size, num_layers,
            num_heads, diffusion_steps, dropout, graph_type,
            token_vocab_size=vocab_size + 1,
        )
        self.mask_token     = self.vocab_size
        self.noise_schedule = noise_schedule

        alpha = _make_alpha_schedule(diffusion_steps, noise_schedule)
        self.register_buffer("alpha", alpha)                       # (T+1,)

    # --- forward process ---------------------------------------------------

    def _corrupt(self, x_0: torch.Tensor, t: torch.Tensor):
        """Sample x_t ~ q(x_t | x_0) via absorbing forward process."""
        alpha_t   = self.alpha[t].to(x_0.device)
        keep_prob = alpha_t.unsqueeze(1).expand_as(x_0)
        is_clean  = torch.bernoulli(keep_prob).bool()
        x_t = torch.where(
            is_clean,
            x_0,
            x_0.new_full(x_0.shape, self.mask_token),
        )
        return x_t, is_clean

    # --- mask pre-sampling -------------------------------------------------

    def sample_masks(self, gate_tokens: torch.Tensor) -> torch.Tensor:
        """
        Pre-sample corruption masks for all T timesteps.
        Returns (B, T, L) bool tensor — True where a token is masked.
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

    # --- reverse process / sampling ----------------------------------------

    def sample_sequence(self, state, temperature):
        """Generate gate sequence via the absorbing reverse process."""
        batch_size = state["idx"].shape[0]
        device     = state["idx"].device

        tokens = torch.full(
            (batch_size, self.ngates), self.mask_token,
            dtype=torch.long, device=device,
        )

        for step in range(self.diffusion_steps, 0, -1):
            is_masked = tokens.eq(self.mask_token)
            if not is_masked.any():
                break

            logits  = self._logits(tokens, step)
            x0_pred = Categorical(logits=-temperature * logits).sample()

            alpha_t    = self.alpha[step]
            alpha_prev = self.alpha[step - 1]
            p_reveal   = (
                (alpha_prev - alpha_t) / (1.0 - alpha_t + 1e-8)
            ).clamp(0.0, 1.0)

            reveal = (
                torch.bernoulli(
                    torch.full_like(tokens, p_reveal.item(), dtype=torch.float)
                )
                .bool()
                .logical_and(is_masked)
            )
            tokens = torch.where(reveal, x0_pred, tokens)

        # Safety net for any still-masked positions
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

    # --- log-probability ---------------------------------------------------

    def log_prob(self, indices, temperature, return_entropy=False, masks=None):
        """
        Denoising ELBO averaged over all T timesteps.

        masks : (B, T, L) bool, optional — pre-sampled corruption masks from
                sample_masks(). When provided the ELBO is deterministic, which
                stabilises the GRPO importance-weight ratio.
        """
        gate_tokens = indices[:, 1:]
        B, L        = gate_tokens.shape
        device      = gate_tokens.device

        total_logp    = torch.zeros(B, L, device=device)
        total_entropy = torch.zeros(B, L, device=device) if return_entropy else None

        for t_int in range(1, self.diffusion_steps + 1):
            t = torch.full((B,), t_int, dtype=torch.long, device=device)

            if masks is not None:
                is_masked = masks[:, t_int - 1, :]
                x_t = torch.where(
                    is_masked,
                    gate_tokens.new_full(gate_tokens.shape, self.mask_token),
                    gate_tokens,
                )
            else:
                x_t, is_clean = self._corrupt(gate_tokens, t)
                is_masked     = ~is_clean

            logits     = self._logits(x_t, t)
            log_probs  = F.log_softmax(-temperature * logits, dim=-1)
            token_logp = torch.gather(
                log_probs, 2, gate_tokens.unsqueeze(-1)
            ).squeeze(-1)

            total_logp += token_logp * is_masked.float()

            if return_entropy:
                entropy        = -(log_probs.exp() * log_probs).sum(dim=-1)
                total_entropy += entropy * is_masked.float()

        avg_logp = total_logp / self.diffusion_steps
        if return_entropy:
            return avg_logp, total_entropy / self.diffusion_steps
        return avg_logp
