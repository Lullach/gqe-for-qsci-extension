# CLAUDE 2026-10-28. CSV lookup + matching only; the POLICY decision (what to do
# when a molecule has no measured requirement) is deliberately a loud failure
# rather than a guess -- see `required_n_dets`.
"""
How many determinants does this molecule need?

`hpc/subspace_requirement.py` measured it for 181 configurations: the smallest
top-|c| subspace that reaches the target accuracy (1.6 mHa), from the exact
vector. This reads that table back so a run can size its determinant budget per
molecule instead of hard-coding 102 for N2.

It is a MEASUREMENT, not a prediction: it exists only for systems small enough
to have been solved exactly. Nothing here extrapolates -- the entropy law that
claimed to is refuted in hpc/entropy_scan.py's status block (S_cisd has partial
correlation 0.05 with the requirement once system size is controlled for).

The CSV holds THREE schemas, grown one column at a time:

    12  system,basis,nelecas,...,fci_energy,seconds
    15  + family,atoms,bond_length after system
    16  + selfchecked before seconds

Neither end is stable (the prefix grew by 4, the suffix by 1), so rows are
dispatched on field count and read from a known offset. Guessing from one end
is what produced `int('0.03125000')`.
"""
from __future__ import annotations

import csv
import pathlib
import re

DEFAULT_CSV = pathlib.Path("data/subspace/subspace_requirement.csv")

# Where `basis` sits; every later column follows it in the same order in all
# three schemas, so one offset is enough.
_BASIS_AT = {12: 1, 15: 4, 16: 4}


# CLAUDE
def load_requirements(csv_path=DEFAULT_CSV) -> list[dict]:
    """Every measured requirement, as dicts. Tail-anchored, so both CSV
    schemas parse."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for fields in csv.reader(fh):
            if not fields or fields[0] == "system":
                continue
            name = fields[0]
            base = _BASIS_AT.get(len(fields))
            if base is None:
                raise ValueError(
                    f"{csv_path}: {name} has {len(fields)} fields; known "
                    f"schemas are {sorted(_BASIS_AT)}"
                )
            if len(fields) >= 15:
                bond = float(fields[3])
            else:
                m = re.search(r"_r([0-9]*\.?[0-9]+)$", name)
                bond = float(m.group(1)) if m else None
            rows.append(dict(
                system=name,
                bond_length=bond,
                basis=fields[base],
                nelecas=int(fields[base + 1]),
                norbcas=int(fields[base + 2]),
                n_fci=int(fields[base + 4]),
                n_needed=int(fields[base + 5]),
                target_mha=float(fields[base + 8]),
            ))
    return rows


# CLAUDE
def required_n_dets(molecule, cfg=None, csv_path=DEFAULT_CSV, tol=1e-3,
                    bond_length=None) -> int:
    """
    The measured determinant requirement for `molecule`.

    Matched on the physical identity -- basis, active space, and bond length
    read from the molecule config -- not on a name, so a renamed config still
    resolves and a mismatched one fails rather than silently matching.

    Raises LookupError when the molecule was never measured. That is the right
    behaviour: a guessed budget would quietly decide the result of every
    comparison that uses it. Set model.n_dets explicitly to override.
    """
    bond = bond_length
    if bond is None and cfg is not None:
        geom = cfg.molecule.get("geometry", {})
        bond = geom.get("bond_length", None)
    nelecas = int(sum(molecule.nelec))
    norbcas = int(molecule.norb)

    same_space = [r for r in load_requirements(csv_path)
                  if r["basis"] == molecule.basis
                  and r["nelecas"] == nelecas and r["norbcas"] == norbcas]
    if not same_space:
        raise LookupError(
            f"no measured requirement for basis={molecule.basis} "
            f"({nelecas}e, {norbcas}o). Run hpc/subspace_requirement.py for it, "
            f"or set model.n_dets explicitly."
        )

    if bond is None:
        if len({r["n_needed"] for r in same_space}) == 1:
            return same_space[0]["n_needed"]
        raise LookupError(
            f"{len(same_space)} measurements share this active space and differ "
            f"in requirement ({sorted({r['n_needed'] for r in same_space})}); "
            f"the molecule config has no bond_length to disambiguate."
        )

    hits = [r for r in same_space
            if r["bond_length"] is not None and abs(r["bond_length"] - bond) < tol]
    if not hits:
        near = sorted(same_space, key=lambda r: abs((r["bond_length"] or 0) - bond))[:3]
        raise LookupError(
            f"no requirement at bond_length={bond} for ({nelecas}e, {norbcas}o). "
            f"Nearest: " + ", ".join(f"{r['system']}={r['n_needed']}" for r in near)
        )
    return hits[0]["n_needed"]
