"""
Phase 0 audit: operator feature menu + DAG commutation statistics.

Usage (inside Docker):
    python3 smoke_features.py            # defaults to n2
    python3 smoke_features.py h2o
    python3 smoke_features.py lih

Two questions this answers:
  1. Does every feature column carry information, and are the duplicate rows
     explainable as physics (commuting fragments / degenerate orbitals)?
  2. What fraction of operator pairs share a qubit but COMMUTE? Those are the
     ordering edges the qubit-wire DAG imposes for no physical reason.
     See NOTES.md, "Redesign proposal: the circuit as a trace".

Delete this file once the redesign lands.
"""

import sys
from collections import defaultdict

import numpy
from hydra import compose, initialize

from gqe_qsci.factory import Factory

MOLECULE = sys.argv[1] if len(sys.argv) > 1 else "n2"

with initialize(version_base="1.3", config_path="configs"):
    cfg = compose(config_name="default", overrides=[f"molecule={MOLECULE}"])

factory = Factory()
mol = factory.create_molecule(cfg)
pool = factory.create_operator_pool(cfg)
feats = pool.get_operator_features()
names = pool.FEATURE_NAMES

numpy.set_printoptions(precision=4, suppress=True, linewidth=250)
col = {n: feats[:, i] for i, n in enumerate(names)}
arity = col["arity"]
nonid = arity > 0

print(f"\n############  molecule = {MOLECULE}  ############")

print("\n=== orbital data ===")
eps = numpy.asarray(mol.active_mo_energy, dtype=float)
occ = numpy.asarray(mol.active_mo_occ, dtype=float)
print(f"n_qubits         : {pool.n_qubits}")
print(f"active_mo_energy : {numpy.round(eps, 4)}")
print(f"active_mo_occ    : {occ}")
# Degenerate orbitals drive duplicate feature rows; report them explicitly.
degen = [
    (i, j) for i in range(len(eps)) for j in range(i + 1, len(eps))
    if abs(eps[i] - eps[j]) < 1e-6
]
print(f"degenerate orbital pairs: {degen if degen else 'none'}")

print(f"\n=== menu: {feats.shape[0]} operators x {feats.shape[1]} features ===\n")
hdr = f"{'feature':>16} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'#uniq':>6}"
print(hdr)
print("-" * len(hdr))
for n in names:
    v = col[n]
    print(f"{n:>16} {v.min():>10.4f} {v.max():>10.4f} {v.mean():>10.4f} "
          f"{v.std():>10.4f} {len(numpy.unique(v)):>6d}")

print("\n=== invariants ===")
checks = {
    "row 0 (identity) all zeros": bool(numpy.all(feats[0] == 0)),
    "amplitude > 0 on non-identity": bool(numpy.all(col["amplitude"][nonid] > 0)),
    "coupling == 0 on singles (Brillouin)": bool(numpy.all(col["coupling"][arity == 1] == 0)),
    "gap > 0 on non-identity": bool(numpy.all(col["gap"][nonid] > 0)),
    "eps_occ <= 0 (HOMO-referenced)": bool(numpy.all(col["eps_occ"][nonid] <= 1e-6)),
    "eps_virt >= 0 (HOMO-referenced)": bool(numpy.all(col["eps_virt"][nonid] >= -1e-6)),
    "ladder_span >= 2*arity": bool(numpy.all(col["ladder_span"][nonid] >= 2 * arity[nonid])),
}
for k, v in checks.items():
    print(f"{k:.<42} {v}")

for a in (0, 1, 2):
    label = {0: "identity", 1: "singles", 2: "doubles"}[a]
    print(f"{label:>9}: {int((arity == a).sum()):4d}")


def commutes(w1, w2):
    """Two Pauli strings commute iff they anticommute on an even # of qubits."""
    anti = sum(1 for a, b in zip(w1, w2) if a != "I" and b != "I" and a != b)
    return anti % 2 == 0


words_all = pool.get_pauli_words()
fps_all = pool.get_xy_qubit_footprints()

# ---------------------------------------------------------------------------
# Duplicate rows are EXPECTED: the menu encodes physics, so operators agreeing
# on every physical column should collide and receive equal logits.
# Classify them rather than asserting zero.
# ---------------------------------------------------------------------------
print("\n=== duplicate feature rows: classification ===")
rows = defaultdict(list)
for k in range(1, feats.shape[0]):
    rows[tuple(feats[k])].append(k)
dupe_groups = [ks for ks in rows.values() if len(ks) > 1]
n_dupe = sum(len(ks) - 1 for ks in dupe_groups)
print(f"{n_dupe} duplicate rows in {len(dupe_groups)} groups")
print(f"distinct Pauli words: {len(set(words_all))} / {len(words_all)}")

same_fp = sum(1 for ks in dupe_groups if len({tuple(fps_all[k]) for k in ks}) == 1)
mut_comm = sum(
    1 for ks in dupe_groups
    if all(commutes(words_all[a], words_all[b])
           for i, a in enumerate(ks) for b in ks[i + 1:])
)
print(f"  groups sharing one footprint (generator fragments) : {same_fp}")
print(f"  groups whose members all mutually commute .........: {mut_comm}")
print(f"  groups spanning >1 footprint (degeneracy symmetry) : {len(dupe_groups) - same_fp}")

# ---------------------------------------------------------------------------
# The headline measurement.
# ---------------------------------------------------------------------------
print("\n=== commutation vs qubit-sharing (DAG edge analysis) ===")
idxs = [k for k in range(len(pool.pool)) if fps_all[k]]
fps = [set(f) for f in fps_all]

share_comm = share_anti = disjoint = 0
for i, ki in enumerate(idxs):
    for kj in idxs[i + 1:]:
        if fps[ki] & fps[kj]:
            if commutes(words_all[ki], words_all[kj]):
                share_comm += 1
            else:
                share_anti += 1
        else:
            disjoint += 1

total = share_comm + share_anti + disjoint
shared = share_comm + share_anti
print(f"operators (non-identity) .............. {len(idxs)}")
print(f"total pairs .......................... {total}")
print(f"disjoint support (no edge, correct) .. {disjoint:>7}  ({100*disjoint/total:5.1f}%)")
print(f"share qubit, ANTIcommute (edge real) . {share_anti:>7}  ({100*share_anti/total:5.1f}%)")
print(f"share qubit, COMMUTE (edge spurious) . {share_comm:>7}  ({100*share_comm/total:5.1f}%)")

print(f"\n>>> SUMMARY [{MOLECULE}]  n_qubits={pool.n_qubits}  V={feats.shape[0]}  "
      f"dupe_rows={n_dupe}  degenerate_pairs={len(degen)}")
if shared:
    print(f">>> SUMMARY [{MOLECULE}]  spurious_edges={100*share_comm/total:.1f}% of all pairs, "
          f"{100*share_comm/shared:.1f}% of shared-qubit pairs")
print(">>> If the second number holds across molecules, the spurious-edge")
print(">>> problem is structural, not an N2 artifact.")
