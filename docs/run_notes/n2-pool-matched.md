# n2-pool-matched-s* : several runs stacked in one folder

Every `results.csv` in `outputs/gqe-for-qsci/n2-pool-matched-s<seed>/` holds
more than one run of the SAME experiment (`experiment=n2_pool_matched`, same
config), because each new job wrote into the existing folder and the results
writer appends. The runs differ only in HARDWARE: they were a test of how many
ABCI-Q points CPU vs GPU runs cost, not a change to the experiment.

| job    | date       | seeds | how                                   | per run | indicator columns |
|--------|------------|-------|---------------------------------------|---------|-------------------|
| 215076 | 2026-09-29 | 1-5   | job array, rt_QG, cudaq on GPU        | ~9 min  | no (code predates them) |
| 220006 | by 10-02   | 1-8   | packed.sh: 8 runs on one rt_QG GPU    | 67 min  | seeds 6-8 only |
| 223108 | 2026-10-05 | 1-8   | packed_cpu.sh: rt_QC, numpy sampler   | 10-11 min | seeds 6-8 only |

- In each file the runs appear one after the other; a new run starts where the
  epoch counter goes back to 0.
- Seeds 1-5: the file was created by job 215076 before the indicator columns
  existed, so later jobs kept that layout and DROPPED tail_weight /
  boundary_mha / pt2_mha.
- Seeds 6-8: first written by 220006, so both packed jobs have the indicators.
- 223108 drew shots from numpy, not cudaq: statistically equivalent, not
  bit-identical.
- Result of the test: rt_QC packing costs about a tenth of the points. See
  NOTES.md, "Packing runs into one job, and why on rt_QC".
