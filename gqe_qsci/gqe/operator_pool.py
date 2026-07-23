from abc import ABC, abstractmethod
from collections import Counter
import cudaq
import numpy
import tequila as tq
from tequila.circuit import QCircuit
from tequila.quantumchemistry.chemistry_tools import ClosedShellAmplitudes

from gqe_qsci.molecule import PySCFMolecule
from gqe_qsci.gqe.utils import convert_pauli_to_cudaq_spin, get_pauli_evolution_gate_count


class OperatorPool(ABC):
    def __init__(self, molecule: PySCFMolecule, params: list[float] | None, **kwargs):
        self.molecule = molecule
        self.n_qubits = molecule.norb * 2
        self.n_electrons = int(sum(molecule.nelec))
        self.params = params
        self.pool = self.build_operator_pool(**kwargs)

    def __len__(self):
        return len(self.pool)

    def __iter__(self):
        return iter(self.pool)

    def __getitem__(self, idx):
        return self.pool[idx]

    @abstractmethod
    def get_vocab_size(self):
        pass

    @abstractmethod
    def build_operator_pool(self):
        pass

    @abstractmethod
    def get_gate_count(self, seq: list[int]) -> Counter:
        pass

    def get_identity_operator(self):
        In = cudaq.spin.i(0)
        for q in range(1, self.n_qubits):
            In = In * cudaq.spin.i(q)
        return 1.0 * cudaq.SpinOperator(In)


class UCCSDBasedPool(OperatorPool, ABC):
    def __init__(self, molecule: PySCFMolecule, params: list[float] | None, threshold: float =1e-8, **kwargs):
        super().__init__(molecule, params, threshold=threshold, **kwargs)
    
    def get_vocab_size(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def build_operator_pool(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_gate_count(self, seq: list[int]) -> Counter:
        raise NotImplementedError("Subclasses must implement this method")

    def get_qubit_footprints(self) -> list[list[int]]:
        """
        Return the qubit footprint of each operator in the pool.

        footprints[k] = sorted list of qubit indices where operator k acts
        non-trivially (i.e. has a non-identity Pauli on that qubit).
        The identity operator (index 0) returns an empty list.
        """
        footprints = []
        for operator in self.pool:
            qubits = set()
            for term in operator:
                pauli = term.get_pauli_word(self.n_qubits)
                for q, p in enumerate(pauli):
                    if p != 'I':
                        qubits.add(q)
            footprints.append(sorted(qubits))
        return footprints

    # Column names for the feature matrix returned by get_operator_features().
    # See NOTES.md ("Cross-molecule generalization") for the physics behind each.
    FEATURE_NAMES = [
        "arity",            # 1 = single excitation, 2 = double  (0 = identity)
        "eps_occ",          # mean occupied-orbital energy, HOMO-referenced
        "eps_virt",         # mean virtual-orbital energy, HOMO-referenced
        "gap",              # sum(eps) over virtual minus occupied spin-orbitals
        "amplitude",        # |theta| — CCSD amplitude magnitude
        "coupling",         # |<ij||ab>| — Hamiltonian matrix element to HF ref
        "mp2_pair_energy",  # coupling^2 / gap — perturbative importance
        "ladder_span",      # max(xy) - min(xy) + 1: JW string extent (non-locality)
        "n_y",              # number of Y Paulis -> drives gate_cost (s/sdg count)
        "gate_cost",        # compiled primitive gate count (cx+h+s+sdg+rz)
        "spin_frac_beta",   # fraction of beta spin-orbitals (0=aa, 1=bb, 0.5=ab)
    ]

    # Deliberately NOT a feature: the *positions* of the Y Paulis within the
    # footprint. Operators sharing a footprint are mutually-commuting fragments
    # of one excitation generator (all Pauli terms of a JW excitation commute).
    # A per-position code would distinguish them, but it encodes no physics --
    # it is a disguised operator ID, and would not transfer across molecules,
    # defeating the point of the feature-based menu. Operators that agree on
    # every physical column (energy, coupling, cost) are meant to collide and
    # receive equal logits. See NOTES.md, "Duplicate rows are expected".

    # Physical description of each spin-orbital (= qubit wire under JW).
    ORBITAL_FEATURE_NAMES = [
        "eps",       # HOMO-referenced orbital energy
        "occ",       # 1.0 if occupied in the HF reference, else 0.0
        "is_beta",   # 0.0 = alpha (even qubit), 1.0 = beta (odd qubit)
    ]

    def get_orbital_features(self) -> numpy.ndarray:
        """
        (n_qubits, 3) float32 — physical features for each qubit wire.

        Replaces a per-qubit-index learned embedding, which is molecule-specific:
        "qubit 5" is a different physical orbital in every molecule, so an indexed
        embedding cannot transfer. These features can.

        All columns are already O(1) and molecule-comparable (eps is
        HOMO-referenced, the others are indicators), so no normalization is
        applied. Spin-orbital q maps to spatial orbital q // 2 (interleaved JW,
        even = alpha), matching generate_excitations()'s 2*key / 2*key+1.
        """
        eps = numpy.asarray(self.molecule.active_mo_energy, dtype=numpy.float64)
        occ = numpy.asarray(self.molecule.active_mo_occ, dtype=numpy.float64)
        e_homo = float(eps[occ > 0].max())

        out = numpy.zeros((self.n_qubits, len(self.ORBITAL_FEATURE_NAMES)),
                          dtype=numpy.float32)
        for q in range(self.n_qubits):
            spatial = q // 2
            out[q] = [
                eps[spatial] - e_homo,
                1.0 if occ[spatial] > 0 else 0.0,
                float(q % 2),
            ]
        return out

    def get_xy_qubit_footprints(self) -> list[list[int]]:
        """
        Like get_qubit_footprints(), but only counting qubits carrying an X or Y
        Pauli. Under Jordan-Wigner these are exactly the spin-orbitals the
        excitation acts on; Z entries are ladder/phase qubits and are excluded.
        Robust to remove_z_ladder being on or off.
        """
        footprints = []
        for operator in self.pool:
            qubits = set()
            for term in operator:
                pauli = term.get_pauli_word(self.n_qubits)
                for q, p in enumerate(pauli):
                    if p in ('X', 'Y'):
                        qubits.add(q)
            footprints.append(sorted(qubits))
        return footprints

    def get_pauli_words(self) -> list[str]:
        """
        First Pauli term's word (length n_qubits) for each pool operator.

        PauliEvolutionPool operators have exactly one term, so this is exact.
        ExcitationPool operators sum several Pauli strings; only the first is
        used, so its pattern features describe a representative term.

        Two operators can share a qubit footprint but differ in their X/Y
        pattern (e.g. YYYX vs XXYX) — these are physically distinct rotations,
        so the pattern is needed to tell them apart in feature space.
        """
        words = []
        for operator in self.pool:
            word = None
            for term in operator:
                word = term.get_pauli_word(self.n_qubits)
                break
            words.append(word if word is not None else "I" * self.n_qubits)
        return words

    def get_commutation_matrix(self) -> numpy.ndarray:
        """
        (V, V) bool matrix: commutes[i, j] is True iff pool operators i and j
        commute as Pauli strings.

        Two Pauli strings commute iff they anticommute on an EVEN number of
        qubits (a qubit anticommutes when both strings are non-identity there
        and carry different letters).

        Since exp(i a P) and exp(i b Q) commute iff [P, Q] = 0, this is exactly
        the independence relation of the circuit's trace monoid: adjacent gates
        whose operators commute may be swapped without changing the unitary.
        Consumed by CircuitDAGGNNPolicy for canonical-form action masking.
        The identity operator (row/col 0) commutes with everything.
        """
        words = self.get_pauli_words()
        lut = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        code = numpy.zeros((len(words), self.n_qubits), dtype=numpy.int8)
        for i, word in enumerate(words):
            for q, ch in enumerate(word):
                code[i, q] = lut.get(ch, 0)

        n = len(words)
        commutes = numpy.zeros((n, n), dtype=bool)
        for i in range(n):
            a = code[i]                                  # (n_qubits,)
            both_active = (a != 0) & (code != 0)         # (V, n_qubits)
            differ = code != a                           # (V, n_qubits)
            anti = (both_active & differ).sum(axis=1)    # (V,)
            commutes[i] = (anti % 2) == 0
        return commutes

    @staticmethod
    def _hf_coupling(occ_sos: list[int], virt_sos: list[int], h2) -> float:
        """
        |<Phi_0|H|Phi_exc>| for the excitation moving electrons from the given
        occupied spin-orbitals into the given virtual spin-orbitals.

        Singles: exactly 0 for canonical HF orbitals (Brillouin's theorem).
        Doubles: the antisymmetrized two-electron integral <ij||ab>, assembled
        from chemists'-notation integrals h2[p,q,r,s] = (pq|rs):
          opposite spin (abab): (o_a v_a | o_b v_b)              — direct only
          same spin (aaaa/bbbb): (o1 v1 | o2 v2) - (o1 v2 | o2 v1)
        Spin-orbital convention: even = alpha, odd = beta; spatial = so // 2.
        Anything else (spin-flip, unbalanced) returns 0.
        """
        if len(occ_sos) != 2 or len(virt_sos) != 2:
            return 0.0  # singles (Brillouin) and exotic cases

        a_occ  = [q // 2 for q in occ_sos  if q % 2 == 0]
        b_occ  = [q // 2 for q in occ_sos  if q % 2 == 1]
        a_virt = [q // 2 for q in virt_sos if q % 2 == 0]
        b_virt = [q // 2 for q in virt_sos if q % 2 == 1]

        if len(a_occ) == 1 and len(b_occ) == 1 and len(a_virt) == 1 and len(b_virt) == 1:
            # Opposite-spin double (covers pair doubles i_a i_b -> a_a a_b too)
            return abs(float(h2[a_occ[0], a_virt[0], b_occ[0], b_virt[0]]))
        if len(a_occ) == 2 and len(a_virt) == 2:
            (o1, o2), (v1, v2) = a_occ, a_virt
        elif len(b_occ) == 2 and len(b_virt) == 2:
            (o1, o2), (v1, v2) = b_occ, b_virt
        else:
            return 0.0  # spin-flip-like term: does not couple to the reference
        return abs(float(h2[o1, v1, o2, v2] - h2[o1, v2, o2, v1]))

    def get_operator_features(self) -> numpy.ndarray:
        """
        Return a (V, len(FEATURE_NAMES)) float32 matrix describing every pool
        operator by molecule-independent physical properties — the "menu" for
        feature-based policies (NOTES.md, "Cross-molecule generalization").

        Row k is aligned with self.pool[k]; the identity (row 0) is all zeros.
        Returned as numpy so this module stays torch-free; consumers convert
        via torch.from_numpy().
        """
        eps = numpy.asarray(self.molecule.active_mo_energy, dtype=numpy.float64)
        occ = numpy.asarray(self.molecule.active_mo_occ, dtype=numpy.float64)
        h2 = self.molecule.cas_hamiltonian.h2
        e_homo = float(eps[occ > 0].max())

        amplitudes = getattr(self, "_pool_amplitudes", None)
        assert amplitudes is not None and len(amplitudes) == len(self.pool), (
            "Pool amplitudes not threaded: build_operator_pool must populate "
            "self._pool_amplitudes in parallel with the pool."
        )

        words = self.get_pauli_words()
        feats = numpy.zeros((len(self.pool), len(self.FEATURE_NAMES)), dtype=numpy.float32)

        for k, xy in enumerate(self.get_xy_qubit_footprints()):
            if not xy:  # identity (or Z-only operator): keep the all-zero row
                continue

            occ_sos  = [q for q in xy if occ[q // 2] > 0]   # occupied spin-orbitals
            virt_sos = [q for q in xy if occ[q // 2] == 0]  # virtual  spin-orbitals
            occ_orbs  = sorted({q // 2 for q in occ_sos})
            virt_orbs = sorted({q // 2 for q in virt_sos})

            arity    = len(xy) // 2
            eps_occ  = float(eps[occ_orbs].mean() - e_homo) if occ_orbs else 0.0
            eps_virt = float(eps[virt_orbs].mean() - e_homo) if virt_orbs else 0.0
            gap = float(
                sum(eps[q // 2] for q in virt_sos) - sum(eps[q // 2] for q in occ_sos)
            )
            amplitude = abs(float(amplitudes[k]))
            coupling  = self._hf_coupling(occ_sos, virt_sos, h2)
            mp2       = coupling ** 2 / gap if gap > 1e-12 else 0.0
            # Span of the Jordan-Wigner string. With remove_z_ladder the Z's are
            # stripped, but the extent they *would* have covered is recoverable
            # from the X/Y support, and is the real non-locality signal.
            # (A plain footprint size would be exactly 2*arity, i.e. no information.)
            ladder_span = float(max(xy) - min(xy) + 1)
            word        = words[k]
            n_y         = float(sum(1 for q in xy if word[q] == 'Y'))
            gate_cost   = float(self.get_gate_count([k])["total"])
            spin_frac   = sum(q % 2 for q in xy) / len(xy)

            feats[k] = [
                arity, eps_occ, eps_virt, gap, amplitude,
                coupling, mp2, ladder_span, n_y, gate_cost, spin_frac,
            ]
        return feats

    @staticmethod
    def _excitation_key(idx: tuple) -> tuple:
        """
        Canonical identity of an excitation, invariant to the order its index
        pairs are listed in.

        make_excitation_gate() reads the tuple as pairs:
            (a, i, b, j) -> [(a, i), (b, j)]
        so (12,6,10,8) and (10,8,12,6) are the SAME physical excitation. They
        arise as separate dict keys because the CCSD amplitude tensor obeys
        t_ijab = t_jiba, putting both orderings in the amplitude dictionary.
        Sorting the pairs collapses them.
        """
        return tuple(sorted(
            (idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)
        ))

    def generate_excitations(self, threshold: float, dedup_excitations: bool = False):
        """
        Map each screened CCSD amplitude to spin-orbital excitation indices.

        dedup_excitations : drop excitations that are pair-order permutations of
            one already emitted (see _excitation_key). Amplitudes are iterated in
            descending |t|, so the first (largest-amplitude) spelling is kept.

            Without it, a duplicate excitation becomes a *different* pool operator:
            both gates share a generator, so the second gate's first Pauli string
            is already in build_operator_pool's `seen` set, and it contributes its
            SECOND Pauli string instead. That is where the extra commuting
            fragments in the N2 pool come from (ops 4/7/41/44 on footprint
            [6,8,10,12]). See NOTES.md, "Pool redundancy".
        """
        ccsd_amplitudes = ClosedShellAmplitudes(tIjAb=self.molecule.ccsd_amplitude["t2"], tIA=self.molecule.ccsd_amplitude["t1"])
        amplitudes_all = ccsd_amplitudes.make_parameter_dictionary(threshold=0.0, screening=False)
        amplitudes = {
            k: v for k, v in amplitudes_all.items()
            if not numpy.isclose(v, 0.0, atol=threshold)
        }
        amplitudes = dict(sorted(amplitudes.items(), key=lambda x: numpy.fabs(x[1]), reverse=True))

        indices = {}
        seen_excitations = set()

        def add(idx, angle):
            if dedup_excitations:
                canon = self._excitation_key(idx)
                if canon in seen_excitations:
                    return
                seen_excitations.add(canon)
            indices[idx] = angle

        for key, t in amplitudes.items():
            assert (len(key) % 2 == 0)
            if not numpy.isclose(t, 0.0, atol=threshold):
                if len(key) == 2:
                    angle = 2.0 * t
                    idx_a = (2 * key[0], 2 * key[1])
                    idx_b = (2 * key[0] + 1, 2 * key[1] + 1)
                    add(idx_a, angle)
                    add(idx_b, angle)
                else:
                    assert len(key) == 4
                    angle = 2.0 * t
                    idx_abab = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])
                    add(idx_abab, angle)
                    if key[0] != key[2] and key[1] != key[3]:
                        idx_aaaa = (2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3])
                        idx_bbbb = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1)
                        partner = tuple([key[2], key[1], key[0], key[3]])
                        partner_t = amplitudes_all.get(partner, 0.0)
                        anglex = 2.0 * (t - partner_t)
                        add(idx_aaaa, anglex)
                        add(idx_bbbb, anglex)
        return indices

    def make_uccsd_ansatz(self, threshold: float, dedup_excitations: bool = False):
        screened_indices = self.generate_excitations(
            threshold=threshold, dedup_excitations=dedup_excitations
        )
        geometry_lines = [f"{atom_type} {coords[0]} {coords[1]} {coords[2]}" for atom_type, coords in self.molecule.geometry]
        geometry_str = "\n".join(geometry_lines)
        tq_molecule = tq.Molecule(
            geometry=geometry_str, basis_set=self.molecule.basis, active_orbitals=self.molecule.active_indices, transformation="jordan-wigner"
        )
        ansatz = QCircuit()
        for idx, angle in screened_indices.items():
            converted = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
            ansatz += tq_molecule.make_excitation_gate(indices=converted, angle=angle)
        return ansatz


class PauliEvolutionPool(UCCSDBasedPool):
    def __init__(
        self,
        molecule: PySCFMolecule,
        params: list[float] | None,
        threshold: float =1e-8,
        remove_z_ladder: bool = False,
        only_use_first_pauli: bool = False,
        dedup_excitations: bool = False,
    ):
        super().__init__(
            molecule, params, threshold=threshold,
            remove_z_ladder=remove_z_ladder,
            only_use_first_pauli=only_use_first_pauli,
            dedup_excitations=dedup_excitations,
        )

    def get_vocab_size(self):
        return len(self.pool)

    def build_operator_pool(self, threshold, remove_z_ladder=False, only_use_first_pauli=False, dedup_excitations=False):
        uccsd_ansatz = self.make_uccsd_ansatz(
            threshold=threshold, dedup_excitations=dedup_excitations
        )
        seen = set()
        operator_pool = [self.get_identity_operator()]
        # Parallel list of signed source amplitudes, aligned index-for-index with
        # operator_pool (identity = 0.0). Appended exactly where operators are
        # appended so alignment survives dedup / only_use_first_pauli. Consumed
        # by get_operator_features() — see NOTES.md, cross-molecule section.
        self._pool_amplitudes = [0.0]
        for g in uccsd_ansatz.gates:
            coeff = g.parameter
            for p in g.generator.paulistrings:
                if remove_z_ladder:
                    p = {k: v for k, v in p.items() if v.lower() != 'z'}
                term = convert_pauli_to_cudaq_spin(p)
                if str(term) in seen:
                    continue
                seen.add(str(term))
                if self.params is None:
                    operator_pool.append(coeff * cudaq.SpinOperator(term))
                    self._pool_amplitudes.append(float(coeff))
                else:
                    for p in self.params:
                        operator_pool.append(p * cudaq.SpinOperator(term))
                        self._pool_amplitudes.append(float(p))
                if only_use_first_pauli:
                    break
        return operator_pool
    
    def get_gate_count(self, seq: list[int]) -> Counter:
        counts = Counter()
        for i in seq:
            operator = self.pool[i]
            for term in operator:
                pauli = term.get_pauli_word(self.n_qubits)
                count = get_pauli_evolution_gate_count(pauli)
                counts.update(count)
        return counts


class ExcitationPool(UCCSDBasedPool):
    """
    Each pool element is a COMPLETE excitation generator (the sum of all its
    Pauli strings), not a single Pauli term as in PauliEvolutionPool.

    Caveat: get_pauli_words() and get_commutation_matrix() read only an
    operator's FIRST term, which is exact for PauliEvolutionPool but only a
    representative here. The pattern features (n_y, ladder_span) and the
    commutation matrix are therefore approximate for this pool.
    """

    def __init__(
        self,
        molecule: PySCFMolecule,
        params: list[float] | None,
        threshold: float =1e-8,
        dedup_excitations: bool = False,
    ):
        # NOTE: previously called super().__init__(molecule, params), silently
        # dropping `threshold` and falling back to the 1e-8 default — so the
        # configured ccsd_threshold was ignored for this pool.
        super().__init__(
            molecule, params, threshold=threshold,
            dedup_excitations=dedup_excitations,
        )

    def get_vocab_size(self):
        return len(self.pool)

    def build_operator_pool(self, threshold, dedup_excitations=False):
        uccsd_ansatz = self.make_uccsd_ansatz(
            threshold=threshold, dedup_excitations=dedup_excitations
        )
        operator_pool = [self.get_identity_operator()]
        # Parallel amplitude list, same convention as PauliEvolutionPool.
        self._pool_amplitudes = [0.0]
        for g in uccsd_ansatz.gates:
            coeff = g.parameter
            operator = None
            for p in g.generator.paulistrings:
                term = convert_pauli_to_cudaq_spin(p)
                operator = term if operator is None else (operator + term * p._coeff)
            if self.params is None:
                operator_pool.append(coeff * cudaq.SpinOperator(operator))
                self._pool_amplitudes.append(float(coeff))
            else:
                for p in self.params:
                    operator_pool.append(p * cudaq.SpinOperator(operator))
                    self._pool_amplitudes.append(float(p))
        return operator_pool
    
    def get_gate_count(self, seq: list[int]) -> Counter:
        counts = Counter()
        for i in seq:
            operator = self.pool[i]
            for term in operator:
                count = get_pauli_evolution_gate_count(term.get_pauli_word(self.n_qubits))
                counts.update(count)
        return counts