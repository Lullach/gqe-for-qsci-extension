"""
Measure what operator_pool.dedup_excitations actually removes.

Usage (inside Docker):
    python3 smoke_dedup.py            # defaults to n2
    python3 smoke_dedup.py h2o

Shows, for dedup off vs on:
  - number of excitations and pool size V
  - which excitations are pair-order duplicates of each other
  - how the duplicate feature rows change
"""

import sys
from collections import defaultdict

import numpy
from hydra import compose, initialize

from gqe_qsci.factory import Factory
from gqe_qsci.gqe.operator_pool import PauliEvolutionPool

MOLECULE = sys.argv[1] if len(sys.argv) > 1 else "n2"

with initialize(version_base="1.3", config_path="configs"):
    cfg = compose(config_name="default", overrides=[f"molecule={MOLECULE}"])

mol = Factory().create_molecule(cfg)
thr = cfg.operator_pool.ccsd_threshold


def build(dedup):
    return PauliEvolutionPool(
        mol,
        params=cfg.operator_pool.params,
        threshold=thr,
        remove_z_ladder=cfg.operator_pool.remove_z_ladder,
        only_use_first_pauli=cfg.operator_pool.only_use_first_pauli,
        dedup_excitations=dedup,
    )


print(f"\n############  molecule = {MOLECULE}  ############")

pool_off = build(False)
pool_on = build(True)

exc_off = pool_off.generate_excitations(threshold=thr, dedup_excitations=False)
exc_on = pool_off.generate_excitations(threshold=thr, dedup_excitations=True)

print("\n=== excitation list ===")
print(f"dedup off : {len(exc_off):>4} excitations")
print(f"dedup on  : {len(exc_on):>4} excitations   "
      f"(removed {len(exc_off) - len(exc_on)})")

print("\n=== pool size V (incl. identity) ===")
print(f"dedup off : {len(pool_off.pool):>4}")
print(f"dedup on  : {len(pool_on.pool):>4}   "
      f"(removed {len(pool_off.pool) - len(pool_on.pool)})")

# Which excitations collapse together?
groups = defaultdict(list)
for idx in exc_off:
    groups[PauliEvolutionPool._excitation_key(idx)].append(idx)
collapsed = {k: v for k, v in groups.items() if len(v) > 1}

print(f"\n=== {len(collapsed)} canonical excitations listed more than once ===")
for i, (canon, idxs) in enumerate(collapsed.items()):
    if i >= 8:
        print(f"   ... and {len(collapsed) - 8} more")
        break
    angles = [round(exc_off[x], 5) for x in idxs]
    print(f"   pairs {list(canon)}")
    for x, a in zip(idxs, angles):
        print(f"      spelled {x}  angle={a}")

# Effect on feature-row duplicates
for label, pool in (("off", pool_off), ("on", pool_on)):
    feats = pool.get_operator_features()
    rows = defaultdict(list)
    for k in range(1, feats.shape[0]):
        rows[tuple(feats[k])].append(k)
    dupes = sum(len(v) - 1 for v in rows.values() if len(v) > 1)
    fps = pool.get_xy_qubit_footprints()
    shared = len({tuple(f) for f in fps if f})
    print(f"\ndedup {label:>3}: V={feats.shape[0]:>4}  duplicate_feature_rows={dupes:>3}  "
          f"distinct_footprints={shared:>4}")

print("\nNote: dedup REMOVES operators (distinct Pauli rotations), it does not")
print("merely relabel them. Fewer, more principled actions -- may help or hurt.")
