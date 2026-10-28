# CLAUDE 2026-10-28. Wiring only: this keeps the direct / no-circuit arm out of
# gqe_qsci.factory, which should not have to know that this arm exists.
"""
Factory for the direct arm.

Subclasses the GQE factory and overrides exactly two things:

    create_model             build the determinant policy, with the budget read
                             from the MEASURED subspace requirement
    create_molecule_bundles  attach that per-molecule budget to each bundle, so
                             a multi-molecule run gives every molecule its own

Everything else -- molecule, QSCI pipeline, loss, scheduler, logger -- is
inherited unchanged, which is the point: the two arms must share every piece
that is not the circuit, or the comparison measures the wrong thing.
"""
from __future__ import annotations

import logging

from hydra.utils import instantiate

from gqe_qsci.factory import Factory
from direct_qsci.requirement import required_n_dets

_log = logging.getLogger(__name__)


class DirectFactory(Factory):
    """Use when cfg.model is a direct_qsci policy."""

    # CLAUDE
    def create_model(self, cfg, op_pool=None):
        target = getattr(cfg.model, "_target_", "") or ""
        if "direct_qsci" not in target:
            return super().create_model(cfg, op_pool=op_pool)

        if op_pool is None:
            op_pool = self.create_operator_pool(cfg)
        molecule = self.create_molecule(cfg)

        n_dets = cfg.model.get("n_dets", "auto")
        if n_dets in ("auto", None):
            n_dets = required_n_dets(molecule, cfg)
            _log.info("determinant budget from subspace_requirement: %d", n_dets)

        n_alpha, n_beta = (int(x) for x in molecule.nelec)
        # The pool is read ONLY for get_orbital_features(); there is no menu,
        # no angle and no CCSD in this arm. Run with
        # operator_pool.ccsd_screening=false so building it stays cheap.
        return instantiate(
            cfg.model,
            orbital_features=op_pool.get_orbital_features(),
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_dets=int(n_dets),
        )

    # CLAUDE
    def create_qsci_pipeline(self, cfg):
        """The determinant pipeline: no sampler, no operator pool."""
        from direct_qsci.pipeline import DirectQSCIPipeline

        molecule = self.create_molecule(cfg)
        return DirectQSCIPipeline(
            molecule,
            max_dim=self._resolve_max_dim(cfg, molecule),
            max_cycle=cfg.qsci.max_cycle,
            enlarge_method=cfg.qsci.enlarge_method,
            eigsh_kwargs=cfg.qsci.eigsh_kwargs,
        )

    # CLAUDE
    def create_molecule_bundles(self, cfg):
        """
        Bundles as usual, plus each molecule's own determinant requirement.

        The budget is per molecule by design: 102 determinants is N2 at 2.5 A,
        not a universal number, and a transfer run spans molecules whose
        requirements differ by an order of magnitude. x_value carries the bond
        length for scan and chain sets; heterogeneous sets have none, and then
        the active space alone must identify the row or the lookup raises.
        """
        bundles = super().create_molecule_bundles(cfg)
        override = cfg.model.get("n_dets", "auto")

        for name, bundle in bundles.items():
            if override not in ("auto", None):
                bundle.n_dets = int(override)
                continue
            bundle.n_dets = required_n_dets(
                bundle.molecule, cfg=None, bond_length=bundle.x_value
            )

        need = max(b.n_dets * sum(b.molecule.nelec) for b in bundles.values())
        _log.info(
            "determinant budgets: %s; longest sequence %d tokens",
            ", ".join(f"{n}={b.n_dets}" for n, b in bundles.items()), need,
        )
        if cfg.model.get("max_tokens", None) in (None, "null"):
            _log.warning(
                "model.max_tokens is unset; set it to at least %d or the "
                "transformer will be built for the first molecule only.", need,
            )
        return bundles
