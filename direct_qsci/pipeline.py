# Lukas wrote DirectQSCIPipeline (moved here verbatim from
# scratch/explore_direct_pipeline.py by Claude 2026-10-28; imports and module
# docstring added, logic untouched).
"""
QSCI for the direct pipeline with no circuit: 
determinants in, energy out, no sampler and no pool.

policy already emitted determinants, so process() starts where that finishes 
and everything after is the same

state carries only state["idx"], and
the determinants are rebuilt here with tokens_to_packed.

CLAUDE 2026-10-30 -- the `seq` column for this pipeline. The circuit pipeline puts its
operator indices in QSCISampleResult.seq, which results_log.py writes out as
space-separated ints. This pipeline has no operators, 
so it puts the EMITTED DETERMINANTS there instead, as the flat word list

    alpha_0 beta_0 alpha_1 beta_1 ... alpha_{K-1} beta_{K-1}

i.e. 2*num_sampled_basis ints, in emission order and BEFORE dedup, so a repeat
shows up as a repeated pair. Read it back by pairing consecutive ints; each
(alpha, beta) is what Determinant([alpha, beta]) takes, bit p of alpha meaning
spatial orbital p holds an alpha electron.

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

    Four decisions, settled -- recorded from the code below (CLAUDE 2026-11-02):
      (a) state carries only state["idx"], the tokens. The determinants are
          rebuilt here with tokens_to_packed, not published by the policy, so
          sampling cannot drift from what log_prob scores.
      (b) K is counted BEFORE symmetry completion; the size after it is
          reported separately, as subspace_dim.
      (c) num_sampled_basis is K as emitted, num_symmetry_preserving_basis the
          distinct count after dedup. Keeping both apart is the
          coupon-collector measurement. The dedup itself is silent -- see
          "Maybe Change later" in process().
      (d) No refinement in v1: local_refined, global_refined and
          global_refined_scistates stay None, so GQE-optimized is the only
          stage this pipeline produces.
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
        # same path as the circuit pipeline
        # It reads self._pyci_ham, self.nelec, self.max_cycle, self.eigsh_kwargs.
        return QSCIPipeline.diagonalize(self, subspace)

    def process(self, state) -> QSCIResult:
        tokens = state["idx"][:, 1:]                       # (B, K*T), drop BOS
        T = sum(self.nelec)                                # n_alpha + n_beta
        packed = tokens_to_packed(tokens.view(tokens.shape[0], -1, T), self.nelec[0])

        samples = []
        for sample in packed:                              # (K, 2) per sample
            emitted = int(sample.shape[0])
            # CLAUDE: the emitted determinants, flattened, for the seq column.
            # Taken before dedup so duplicates stay visible (the
            # coupon-collector gap is the thing worth measuring).
            emitted_words = tuple(int(w) for ab in sample.tolist() for w in ab)

            # Dedup silently: a repeat buys no new determinant, so the subspace
            # shrinks and the energy is worse -- the policy is penalised through
            # the reward rather than by an exception. The two counts below keep
            # the coupon-collector gap visible.
            # Maybe Change later
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
                )

            enlarged_subspace = valid_subspace.enlarge(
                max_dim=self.max_dim, method=self.enlarge_method
            )
            energy, ci = self.diagonalize(enlarged_subspace)
            samples.append(
                QSCISampleResult(
                    seq=emitted_words,
                    energy=energy,
                    num_sampled_basis=emitted,                    # K as emitted
                    # For this pipeline the masks make symmetry filtering a no-op, so
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
