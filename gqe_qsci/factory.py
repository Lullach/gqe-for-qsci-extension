# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Modifications Copyright (c) 2026 Ryota Kemmoku
# Modified from the original file in NVIDIA CUDA-QX.
# Changes made: Add factory methods for model, molecule, logger,
# operator_pool, and qsci_pipeline construction.

import copy
import logging
from dataclasses import dataclass
from typing import Any

import cudaq
from hydra.utils import instantiate

from gqe_qsci.gqe.loss import GRPOLoss, GSPOLoss
from gqe_qsci.gqe.operator_pool import PauliEvolutionPool, ExcitationPool
from gqe_qsci.gqe.sampler import Sampler
from gqe_qsci.qsci.pipeline import QSCIPipeline
from gqe_qsci.wandb_logger import Logger

_log = logging.getLogger(__name__)


@dataclass
class MoleculeBundle:
    """
    Everything the training loop needs to run one molecule: its pool, its QSCI
    pipeline, and the per-molecule tensors a feature-based policy swaps in via
    set_molecule() (Phase 3 step 3).

    All tensor fields are numpy arrays (torch-free at this layer); the policy
    converts them. Bundles for the same molecule set all share feat_dim (the
    feature columns are fixed), but may differ in vocab_size and — across
    heterogeneous molecules (Phase 4) — in n_qubits.
    """
    name: str
    split: str                      # "train" or "eval"
    molecule: Any
    pool: Any
    qsci_pipeline: Any
    vocab_size: int
    n_qubits: int
    operator_features: Any          # (V, feat_dim)
    orbital_features: Any           # (n_qubits, orbital_feat_dim)
    qubit_footprints: list
    commutation_matrix: Any         # (V, V) bool
    x_value: float | None = None    # scan coordinate (e.g. bond length) for
    x_label: str = "x"              # summary plots; None for non-scan sets

    @property
    def feat_dim(self) -> int:
        return int(self.operator_features.shape[1])


class Factory:
    def __init__(self):
        self.molecule = None
        self.estimator = None
        self.operator_pool = None

    def create_model(self, cfg, op_pool=None):
        # op_pool lets multi-molecule callers size the model from a specific
        # bundle's pool instead of the cached single-molecule pool.
        if op_pool is None:
            op_pool = self.create_operator_pool(cfg)
        vocab_size = op_pool.get_vocab_size()
        cfg.vocab_size = vocab_size
        _log.info("Vocab size: %d", vocab_size)

        # Tensors that cannot be expressed in YAML are passed programmatically.
        # Every model accepts operator_features (the feature "menu"); it is only
        # used when the model config sets feature_scorer: true, so passing it
        # unconditionally is harmless.
        target = getattr(cfg.model, "_target_", "") or ""
        if "CircuitDAGGNNPolicy" in target:
            return instantiate(
                cfg.model,
                vocab_size=vocab_size,
                ngates=cfg.ngates,
                n_qubits=op_pool.n_qubits,
                qubit_footprints=op_pool.get_qubit_footprints(),
                commutation_matrix=op_pool.get_commutation_matrix(),
                operator_features=op_pool.get_operator_features(),
                orbital_features=op_pool.get_orbital_features(),
            )
        return instantiate(
            cfg.model,
            vocab_size=vocab_size,
            ngates=cfg.ngates,
            operator_features=op_pool.get_operator_features(),
        )

    def create_temperature_scheduler(self, cfg):
        return instantiate(cfg.trainer.temperature_scheduler)
    
    def create_molecule(self, cfg):
        if self.molecule is not None:
            return self.molecule
        molecule = instantiate(cfg.molecule)
        self.molecule = molecule
        return molecule
    
    def create_wandb_logger(self, cfg):
        reference_keys = cfg.reference_keys
        molecule = self.create_molecule(cfg)
        reference_energies = {}
        for key in reference_keys:
            if key == "hf_energy":
                reference_energies[key] = molecule.hf.e_tot
            elif key == "R-CASCI":
                reference_energies[key] = molecule.compute_casci()
                _log.info("CASCI Energy: %f", reference_energies[key])
            elif key == "R-CCSD":
                reference_energies[key] = molecule.compute_ccsd()
                _log.info("CCSD Energy: %f", reference_energies[key])
        return Logger(reference_energies=reference_energies)
        
    def create_loss_fn(self, cfg):
        loss_fn_name = cfg.trainer.loss.type
        match loss_fn_name:
            case "grpo":
                assert cfg.trainer.batch_size == cfg.trainer.num_samples, "batch_size must be equal to num_samples for GRPO training"
                return GRPOLoss(cfg.trainer.loss.clip_grpo_low, cfg.trainer.loss.clip_grpo_high)
            case "gspo":
                assert cfg.trainer.batch_size == cfg.trainer.num_samples, "batch_size must be equal to num_samples for GSPO training"
                return GSPOLoss(cfg.trainer.loss.clip_gspo_low, cfg.trainer.loss.clip_gspo_high)
            case _:
                raise ValueError(f"Unknown loss function name: {loss_fn_name}")
    
    def _make_pool(self, cfg, molecule):
        """Build an operator pool for a given molecule (no caching)."""
        dedup = cfg.operator_pool.get("dedup_excitations", False)
        match cfg.operator_pool.spec:
            case "pauli_evolution":
                return PauliEvolutionPool(
                    molecule,
                    params=cfg.operator_pool.params,
                    threshold=cfg.operator_pool.ccsd_threshold,
                    remove_z_ladder=cfg.operator_pool.remove_z_ladder,
                    only_use_first_pauli=cfg.operator_pool.only_use_first_pauli,
                    dedup_excitations=dedup,
                )
            case "excitation":
                return ExcitationPool(
                    molecule,
                    params=cfg.operator_pool.params,
                    threshold=cfg.operator_pool.ccsd_threshold,
                    dedup_excitations=dedup,
                )
            case _:
                raise ValueError(f"Unknown operator pool specification: {cfg.operator_pool.spec}")

    def _make_qsci_pipeline(self, cfg, molecule, operator_pool):
        """Build a QSCI pipeline for a given molecule + pool (no caching)."""
        numQPUs = cudaq.get_target().num_qpus()
        sampler = Sampler(operator_pool, mpi=cfg.sampler.mpi, numQPUs=numQPUs, shots_count=cfg.sampler.shots)
        return QSCIPipeline(
            molecule, operator_pool, sampler,
            max_dim=cfg.qsci.max_dim,
            enlarge_method=cfg.qsci.enlarge_method,
            max_cycle=cfg.qsci.max_cycle,
            eigsh_kwargs=cfg.qsci.eigsh_kwargs,
        )

    def create_operator_pool(self, cfg):
        if self.operator_pool is not None:
            return self.operator_pool
        molecule = self.create_molecule(cfg)
        self.operator_pool = self._make_pool(cfg, molecule)
        return self.operator_pool

    def create_qsci_pipeline(self, cfg):
        molecule = self.create_molecule(cfg)
        operator_pool = self.create_operator_pool(cfg)
        return self._make_qsci_pipeline(cfg, molecule, operator_pool)

    # ------------------------------------------------------------------ #
    # Multi-molecule (Phase 2+)
    # ------------------------------------------------------------------ #

    def _expand_molecule_set(self, cfg):
        """
        Turn a molecule_set config into a list of
        (name, split, molecule_cfg, x_value).

        Two forms are supported:

        1. Bond scan (Phase 3) — one `base` linear-chain molecule plus
           train/eval bond-length lists. x_value is the bond length, so the
           summary charts plot against it.

        2. Heterogeneous molecules (Phase 4) — an explicit `molecules:` list of
           {name, split, molecule: <cfg>} entries, which may differ in atoms,
           active space and QUBIT COUNT. x_value is None (there is no single
           scan coordinate), so the per-molecule metrics are still logged but
           the dissociation-curve summaries are skipped.
        """
        ms = cfg.molecule_set

        if ms.get("molecules") is not None:
            entries = []
            for m in ms.molecules:
                split = m.get("split", "train")
                assert split in ("train", "eval"), (
                    f"molecule '{m.get('name')}' has split='{split}', "
                    "expected 'train' or 'eval'"
                )
                entries.append((str(m.name), split, m.molecule, None))
            if not entries:
                raise ValueError("molecule_set.molecules is empty.")
            return entries

        base = ms.base
        assert base.geometry.type == "linear_chain", (
            "bond-scan molecule_set requires geometry.type == 'linear_chain'"
        )
        atoms = "".join(base.geometry.atoms).lower()

        entries = []
        for split, key in (("train", "train_bond_lengths"), ("eval", "eval_bond_lengths")):
            for r in ms.get(key, []) or []:
                mcfg = copy.deepcopy(base)
                mcfg.geometry.bond_length = float(r)
                entries.append((f"{atoms}_r{float(r):.2f}", split, mcfg, float(r)))
        if not entries:
            raise ValueError("molecule_set expanded to zero molecules.")
        return entries

    def create_molecule_bundles(self, cfg) -> dict[str, MoleculeBundle]:
        """
        Build one MoleculeBundle per molecule in cfg.molecule_set, keyed by name
        and preserving order (train entries first, then eval).

        Each bundle owns a fresh molecule / pool / QSCI pipeline and the feature
        tensors a feature-based policy needs. The singleton caches
        (self.molecule / self.operator_pool) are deliberately NOT touched, so the
        single-molecule path is unaffected.
        """
        if cfg.get("molecule_set") is None:
            raise ValueError(
                "create_molecule_bundles requires cfg.molecule_set "
                "(e.g. run with molecule_set=n2_bond_scan)."
            )

        bundles: dict[str, MoleculeBundle] = {}
        feat_dim = None
        for name, split, mcfg, x_value in self._expand_molecule_set(cfg):
            molecule = instantiate(mcfg)
            pool = self._make_pool(cfg, molecule)
            qsci = self._make_qsci_pipeline(cfg, molecule, pool)
            op_feats = pool.get_operator_features()

            if feat_dim is None:
                feat_dim = op_feats.shape[1]
            elif op_feats.shape[1] != feat_dim:
                # Guaranteed equal (fixed feature columns); guard against a future
                # per-molecule feature change that would silently break the scorer.
                raise ValueError(
                    f"feat_dim mismatch: {name} has {op_feats.shape[1]}, "
                    f"expected {feat_dim}. All bundles must share feature columns."
                )

            bundles[name] = MoleculeBundle(
                name=name,
                split=split,
                molecule=molecule,
                pool=pool,
                qsci_pipeline=qsci,
                vocab_size=pool.get_vocab_size(),
                n_qubits=pool.n_qubits,
                operator_features=op_feats,
                orbital_features=pool.get_orbital_features(),
                qubit_footprints=pool.get_qubit_footprints(),
                commutation_matrix=pool.get_commutation_matrix(),
                x_value=x_value,
                # Heterogeneous sets have no single scan coordinate, so
                # x_value is None there and the summary curves are skipped.
                x_label="bond_length" if x_value is not None else "molecule",
            )
            _log.info(
                "Bundle %-10s (%s): V=%d, n_qubits=%d",
                name, split, bundles[name].vocab_size, bundles[name].n_qubits,
            )
        return bundles