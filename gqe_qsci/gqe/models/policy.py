from abc import ABC, abstractmethod
import torch.nn as nn


class Policy(ABC, nn.Module):
    @abstractmethod
    def act(self, state, inv_temperature):
        pass

    @abstractmethod
    def log_prob(self, indices, inv_temperature):
        pass

    def set_molecule(self, bundle):
        """
        Re-point a feature-based policy at a different molecule (Phase 2+).

        `bundle` is a MoleculeBundle (duck-typed here to avoid importing the
        factory). Implemented by feature_scorer models; the default fails loudly
        so a molecule swap on an unsupported model can never silently no-op.

        Implementations MUST swap every per-molecule buffer atomically (menu,
        and for the DAG GNN also footprints / commutation / orbital features /
        vocab_size), and must never be called mid-rollout.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_molecule(). "
            "Cross-molecule training requires a feature_scorer=True model."
        )