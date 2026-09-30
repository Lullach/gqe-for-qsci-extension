"""
Per-circuit results, written to a plain CSV next to each run.

W&B receives aggregates — min / mean / std / best_so_far — which is the right
thing for watching training, but it throws away the per-sample detail. Questions
like "what did the distribution of subspace_dim look like at epoch 200?" or
"which excitations does the policy actually pick?" are then unanswerable without
re-running the experiment. Whether a plot can be made later is a function of what
data was SAVED, not of the plotting tool.

So every evaluated circuit gets a row here, including its operator sequence. One
file per run, `<output>/results.csv`, append-only:

    exp_tag,model,n_params,seed,epoch,molecule,split,stage,sample_idx,energy,
    R-CASCI,R-CCSD,subspace_dim,num_sampled_basis,
    num_symmetry_preserving_basis,cx_count,total_gates,seq

Errors are deliberately NOT precomputed — the reference energies are columns, so
any error convention (signed, absolute, vs CASCI, vs CCSD, in Ha or mHa) can be
derived downstream without another run.

`stage` distinguishes the rollout batch from the running bests:
  GQE-optimized                  every sampled circuit this epoch (sample_idx >= 0)
  GQE-optimized(best_so_far)     the best circuit so far      (sample_idx = -1)
  Local-refined(best_so_far)     after local refinement        (sample_idx = -1)
  Global-refined(best_so_far)    after global refinement       (sample_idx = -1)

Read it back with hpc/plot_results.py, which globs every run's file into one
dataframe.
"""

import csv
import logging
import os

_log = logging.getLogger(__name__)

COLUMNS = [
    "exp_tag", "model", "n_params", "seed", "epoch",
    "molecule", "split", "stage", "sample_idx",
    "energy", "R-CASCI", "R-CCSD",
    "subspace_dim", "num_sampled_basis", "num_symmetry_preserving_basis",
    "cx_count", "total_gates", "seq",
    # subspace-size indicators (gqe_qsci/qsci/diagnostics.py), appended last so
    # every earlier column keeps its position
    "tail_weight", "boundary_mha", "pt2_mha",
]


def _iter_samples(result):
    """
    (sample_idx, QSCISampleResult) pairs.

    A QSCIResult holds the whole rollout batch, so it yields one row per sampled
    circuit with its batch index. A bare QSCISampleResult is a single best-so-far
    or refined circuit, so it yields index -1.
    """
    samples = getattr(result, "samples", None)
    if samples is not None:
        yield from enumerate(samples)
    else:
        yield -1, result


class ResultsWriter:
    """
    Append-only CSV of per-circuit results.

    Deliberately dependency-free (no pandas): it is written from inside the
    training loop, where an import-heavy dependency and a partially flushed
    dataframe would both be liabilities. Each call flushes, so a killed or
    walltime-exceeded job still leaves usable data on disk — which matters on a
    cluster, where that is a normal way for a run to end.
    """

    def __init__(self, path: str, *, exp_tag: str, model: str, seed,
                 n_params: int | None = None):
        self.path = path
        # n_params is recorded because an architecture comparison with unequal
        # capacity is confounded: without it, "GPT-2 won" cannot be separated
        # from "GPT-2 had 100x the parameters".
        self.context = {"exp_tag": exp_tag, "model": model,
                        "n_params": n_params, "seed": seed}
        self._ready = False
        self._columns = COLUMNS

    def _open(self):
        if self._ready:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # Append across restarts, but only write the header for a new file.
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if new:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUMNS)
            _log.info("Per-circuit results: %s", self.path)
        else:
            # A resumed run appends to a file that may predate newer columns.
            # Write in THAT file's layout: appending wider rows under a narrower
            # header silently misaligns every reader downstream.
            with open(self.path, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), None)
            if header and header != COLUMNS:
                self._columns = header
                _log.warning("results.csv has an older header; writing its %d "
                             "columns and dropping the rest", len(header))
        self._ready = True

    def log(self, *, epoch, molecule, split, entries, references=None):
        """
        entries : the same list handed to Logger.log_result, each
                  {"result": ..., "prefix": ..., "stage": ...}
        references : {"R-CASCI": float, ...} for this molecule, or None
        """
        refs = references or {}
        try:
            self._open()
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for entry in entries:
                    stage = entry.get("stage") or entry.get("prefix", "")
                    for idx, s in _iter_samples(entry["result"]):
                        row = {
                            **self.context,
                            "epoch": int(epoch),
                            "molecule": molecule, "split": split,
                            "stage": stage, "sample_idx": idx,
                            "R-CASCI": refs.get("R-CASCI"),
                            "R-CCSD": refs.get("R-CCSD"),
                            "seq": " ".join(str(int(t)) for t in
                                            (getattr(s, "seq", None) or ())),
                        }
                        for name in ("energy", "subspace_dim",
                                     "num_sampled_basis",
                                     "num_symmetry_preserving_basis",
                                     "cx_count", "total_gates", "tail_weight",
                                     "boundary_mha", "pt2_mha"):
                            # getattr: objects restored from checkpoints written
                            # before a field existed simply lack it
                            row[name] = getattr(s, name, None)
                        writer.writerow([row.get(c) for c in self._columns])
        except Exception as exc:                                  # noqa: BLE001
            # Never let bookkeeping kill a training run that is otherwise fine.
            _log.warning("results.csv write failed (%s: %s); training continues.",
                         type(exc).__name__, exc)
