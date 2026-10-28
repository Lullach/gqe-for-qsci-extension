# Lukas wrote DirectQSCIPipeline (moved here verbatim from
# scratch/explore_direct_pipeline.py by Claude 2026-10-28; imports and module
# docstring added, logic untouched).
"""
QSCI for the direct arm: determinants in, energy out, no sampler and no pool.

The circuit arm turns operator indices into a circuit, simulates it for 10k
shots and reads determinants off the counts. Here the policy already emitted
determinants, so process() starts where that finishes -- and everything after
it (post-selection, symmetry completion, diagonalisation) is the SAME code the
circuit arm runs, which is what makes the two comparable.

Tokens stay the single source of truth: state carries only state["idx"], and
the determinants are rebuilt here with tokens_to_packed.
"""
from __future__ import annotations

import numpy as np
import pyci

from direct_qsci.policy import tokens_to_packed
from gqe_qsci.qsci.determinant import Determinant
from gqe_qsci.qsci.pipeline import QSCIPipeline
from gqe_qsci.qsci.schema import QSCIResult, QSCISampleResult
from gqe_qsci.qsci.subspace import DeterminantSubspace


class DirectQSCIPipeline:

    """
    Same role as QSCIPipeline, with no sampler and no operator pool.

    Decisions to make, each of which changes what the numbers mean:
      (a) does `state` carry Determinant objects, or a packed (B, K, 2) uint64
          tensor straight from the policy?
      (b) is K counted before or after symmetry completion?
      (c) what goes in num_sampled_basis: K as emitted, or K after dedup?
          The gap between them IS the coupon-collector measurement, so do not
          collapse it into one number.
      (d) reuse RefinePipeline for local/global refinement, or skip it in v1?
          Skipping changes which columns the comparison scripts can use.
    """

    def __init__(self, molecule, max_dim, max_cycle=1000,
                 enlarge_method="symmetry_completion", eigsh_kwargs=None):
        self.mol = molecule
        self.hamiltonian = molecule.cas_hamiltonian
        self.norb = molecule.norb
        self.spin = molecule.spin
        self.nelec = molecule.nelec
        self.max_dim = max_dim
        self.max_cycle = max_cycle
        self.enlarge_method = enlarge_method
        self.eigsh_kwargs = eigsh_kwargs or {}
        self._pyci_h2 = np.asarray(self.hamiltonian.h2.transpose(0, 2, 1, 3), order="C")
        self._pyci_ham = pyci.hamiltonian(
            self.hamiltonian.e_core, self.hamiltonian.h1, self._pyci_h2
        )
        # CLAUDE: TrainPipeline.on_save_checkpoint reads this off the pipeline.
        # v1 does no global refinement, so it stays None -- but the attribute
        # has to exist or checkpointing dies at the end of the run.
        self.global_refined_scistates = None

    def diagonalize(self, subspace: DeterminantSubspace):
        # same path as the circuit arm (duplicate check, solve, SCIVector), so
        # the two arms differ only in where the determinants come from.
        # It reads self._pyci_ham, self.nelec, self.max_cycle, self.eigsh_kwargs.
        return QSCIPipeline.diagonalize(self, subspace)

    def process(self, state) -> QSCIResult:
        # tokens are the single source of truth for what was sampled; the
        # determinants are derived here rather than published by the policy.
        tokens = state["idx"][:, 1:]                       # (B, K*T), drop BOS
        T = sum(self.nelec)                                # n_alpha + n_beta
        packed = tokens_to_packed(tokens.view(tokens.shape[0], -1, T), self.nelec[0])

        samples = []
        for sample in packed:                              # (K, 2) per sample
            emitted = int(sample.shape[0])

            # Dedup silently: a repeat buys no new determinant, so the subspace
            # shrinks and the energy is worse -- the policy is penalised through
            # the reward rather than by an exception. The two counts below keep
            # the coupon-collector gap visible.
            seen, dets = set(), []
            for a, b in sample.tolist():
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                dets.append(Determinant([np.uint64(a), np.uint64(b)]))

            subspace = DeterminantSubspace(dets)
            valid_subspace = subspace.post_select_by_nelec(self.nelec)
            if valid_subspace.ndet != subspace.ndet:
                raise ValueError(
                    f"post_select_by_nelec dropped "
                    f"{subspace.ndet - valid_subspace.ndet} of {subspace.ndet} "
                    f"determinants, i.e. some do not have exactly "
                    f"{self.nelec[0]} alpha and {self.nelec[1]} beta electrons. "
                    f"Three candidates, in order of likelihood for a "
                    f"multi-molecule run: (1) self.nelec disagrees with the "
                    f"n_alpha/n_beta the policy was built with (a set_molecule "
                    f"mismatch); (2) tokens_to_packed split or packed wrongly "
                    f"(the alpha/beta boundary, or the spatial = q // 2 "
                    f"convention); (3) the occupancy masks are wrong."
                )

            enlarged_subspace = valid_subspace.enlarge(
                max_dim=self.max_dim, method=self.enlarge_method
            )
            energy, ci = self.diagonalize(enlarged_subspace)
            samples.append(
                QSCISampleResult(
                    seq=None,
                    energy=energy,
                    num_sampled_basis=emitted,                    # K as emitted
                    # For this arm the masks make symmetry filtering a no-op, so
                    # this column carries the DISTINCT count instead.
                    num_symmetry_preserving_basis=len(dets),
                    subspace_dim=enlarged_subspace.ndet,
                    cx_count=None,
                    total_gates=None,
                )
            )

        return QSCIResult(
            samples=tuple(samples),
            local_refined=None,
            global_refined=None,
        )
