"""
QSCI without the quantum circuit -- the direct / ablation arm.

A policy proposes DETERMINANTS, they are diagonalised by the same QSCI path the
circuit arm uses, and the energy is the reward. No operator pool, no gates, no
angles, no CCSD, no sampling shots.

This is deliberately NOT part of gqe_qsci: it is not a generative quantum
eigensolver and must not be mistaken for one in a plot or a thesis chapter. It
exists to measure what routing the same policy through a circuit costs, against
the same molecules, the same budget K and the same diagonalisation. See
direct_qsci/README.md.

It does reuse gqe_qsci for everything that is genuinely shared: the Policy base
class, OperatorScorer, the GRPO loss, the replay buffer, the trainer, and the
determinant/subspace machinery in gqe_qsci.qsci.
"""
from direct_qsci.requirement import required_n_dets  # noqa: F401
from direct_qsci.policy import (  # noqa: F401
    packed_to_tokens,          # noqa: F401
    DeterminantGPT2Policy,
    allowed_mask,
    tokens_to_packed,
)
