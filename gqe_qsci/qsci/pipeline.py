import numpy as np
import torch
import pyci
from typing import Any

from gqe_qsci.molecule import PySCFMolecule
from gqe_qsci.gqe.sampler import Sampler
from gqe_qsci.gqe.operator_pool import OperatorPool
from gqe_qsci.qsci.schema import QSCIResult, QSCISampleResult
from gqe_qsci.qsci.subspace import DeterminantSubspace
from gqe_qsci.qsci.refine.pipeline import RefinePipeline
from gqe_qsci.qsci.statevector import as_scivector, SCIVector
from gqe_qsci.qsci.diagnostics import boundary_indicators, enpt2_mha

class QSCIPipeline:
    def __init__(
        self,
        molecule: PySCFMolecule,
        operator_pool: OperatorPool,
        sampler: Sampler,
        max_dim: int = 100_000,
        max_cycle: int = 200,
        enlarge_method: str = "symmetry_completion",
        eigsh_kwargs: dict[str, Any] | None = None,
        indicators: bool = True,
    ):
        self.mol = molecule
        self.hamiltonian = molecule.cas_hamiltonian
        self.norb = molecule.norb
        self.spin = molecule.spin
        self.nelec = molecule.nelec
        self.operator_pool = operator_pool
        self.sampler = sampler
        self.max_dim = max_dim
        self.max_cycle = max_cycle
        self.enlarge_method = enlarge_method
        self.indicators = indicators
        self.subspace_refiner = RefinePipeline(hamiltonian=self.hamiltonian, nelec=self.nelec, norb=self.norb)
        self.global_refined_scistates = None

        self.eigsh_kwargs = eigsh_kwargs or {}
        self._pyci_h2 = np.asarray(self.hamiltonian.h2.transpose(0, 2, 1, 3), order="C")
        self._pyci_ham = pyci.hamiltonian(
            self.hamiltonian.e_core, self.hamiltonian.h1, self._pyci_h2
        )


    def _solve(self, subspace: DeterminantSubspace):
        wfn = pyci.fullci_wfn(self._pyci_ham.nbasis, *self.nelec)
        for det in subspace.determinants:
            wfn.add_det(det)

        op = pyci.sparse_op(self._pyci_ham, wfn)
        energy, coeffs = op.solve(maxiter=self.max_cycle)
        return float(energy[0]), coeffs[0], wfn

    def diagonalize(self, subspace: DeterminantSubspace) -> tuple[float, SCIVector]:
        energy, coeffs, _ = self._solve(subspace)
        return energy, as_scivector(coeffs, subspace.determinants)

    def _diagnose(self, subspace: DeterminantSubspace, energy, coeffs, wfn) -> dict:
        """
        Subspace-size indicators for one solved subspace, as QSCISampleResult
        fields. Empty when disabled. A failure here must never cost a training
        step, so it degrades to "not measured" rather than raising.

        ENPT2 is the one to trust: replaying the H10 architecture runs, it was
        hot on every best-so-far circuit, while the two boundary indicators read
        ~100x lower than on importance-ordered subspaces of the same error and
        flickered around their thresholds. See NOTES.md.
        """
        if not self.indicators:
            return {}
        try:
            ind = boundary_indicators(self._pyci_ham, subspace.determinants,
                                      coeffs, energy, self.nelec,
                                      max_cycle=self.max_cycle)
            return {"tail_weight": ind.tail_weight,
                    "boundary_mha": ind.boundary_mha,
                    "pt2_mha": enpt2_mha(self._pyci_ham, wfn, coeffs, energy)}
        except Exception:                                        # noqa: BLE001
            return {}

    def process(
        self,
        states: dict[str, torch.Tensor],
    ) -> QSCIResult:
        sampled_counts = self.sampler.run(states)
        samples = []
        sci_states = []

        for seq, counts in zip(states["idx"].tolist(), sampled_counts):
            circuit_gate_numbers = self.operator_pool.get_gate_count(seq)
            subspace = DeterminantSubspace.from_cudaq_sample_result(counts)
            valid_subspace = subspace.post_select_by_nelec(self.nelec)
            enlarged_subspace = valid_subspace.enlarge(max_dim=self.max_dim, method=self.enlarge_method)
            energy, coeffs, wfn = self._solve(enlarged_subspace)
            ci = as_scivector(coeffs, enlarged_subspace.determinants)
            sci_states.append(ci)

            samples.append(
                QSCISampleResult(
                    seq=tuple(seq),
                    energy=energy,
                    num_sampled_basis=len(counts),
                    num_symmetry_preserving_basis=valid_subspace.ndet,
                    subspace_dim=enlarged_subspace.ndet,
                    cx_count=circuit_gate_numbers.get("cx", 0),
                    total_gates=circuit_gate_numbers.get("total", 0),
                    **self._diagnose(enlarged_subspace, energy, coeffs, wfn),
                )
            )

        # local refinement
        local_refined_subspace = self.subspace_refiner.process(sci_states, max_dim=self.max_dim)
        local_refined_energy, local_coeffs, local_wfn = self._solve(local_refined_subspace)
        local_refined_ci = as_scivector(local_coeffs, local_refined_subspace.determinants)
        local_refined_result = QSCISampleResult(
            seq=None,
            energy=local_refined_energy,
            num_sampled_basis=None,
            num_symmetry_preserving_basis=None,
            subspace_dim=local_refined_subspace.ndet,
            cx_count=None,
            total_gates=None,
            **self._diagnose(local_refined_subspace, local_refined_energy,
                             local_coeffs, local_wfn),
        )

        # global refinement
        global_refined_energy: float | None = None
        if self.global_refined_scistates is None:
            self.global_refined_scistates = local_refined_ci
            global_refined_result = None
        else:
            target_sci_states = [self.global_refined_scistates, local_refined_ci]
            global_refined_subspace = self.subspace_refiner.process(target_sci_states, max_dim=self.max_dim)
            global_refined_energy, global_coeffs, global_wfn = self._solve(global_refined_subspace)
            global_refined_ci = as_scivector(global_coeffs, global_refined_subspace.determinants)
            self.global_refined_scistates = global_refined_ci
            global_refined_result = QSCISampleResult(
                seq=None,
                energy=global_refined_energy,
                num_sampled_basis=None,
                num_symmetry_preserving_basis=None,
                subspace_dim=global_refined_subspace.ndet,
                cx_count=None,
                total_gates=None,
                **self._diagnose(global_refined_subspace, global_refined_energy,
                                 global_coeffs, global_wfn),
            )

        return QSCIResult(
            samples=tuple(samples),
            local_refined=local_refined_result,
            global_refined=global_refined_result,
        )