# `# CLAUDE` marker backfill

Core functions I (Claude) wrote before the working rules of 2026-10-05.
APPLIED 2026-10-05: Lukas chose to mark all 209 listed functions, the mixed
ones included. Each has a `# CLAUDE` line directly above its `def`. Remove a
marker once you have read or rewritten that function.

## How this list was made

`git blame` on every core file (`.githooks/core_paths`) at `a515bc8`, per
function or method. Each line is attributed to the commit that last changed it:

- **Claude**: a commit with my `Co-Authored-By: Claude` line (83 of your 85 commits).
- **upstream**: the original GQE-QSCI code by moken20 / Ryota Kemmoku. Neither yours nor
  mine; NOT listed here. Whether it needs its own treatment is a separate question.
- **yours**: your two commits without my line. They touched no core file.

Caveats: edits you made inside a commit I co-authored count as mine, and so do
upstream lines I only re-indented or reformatted. So a function can look more
"mine" than it is. The percentages are a guide, not a verdict.

Totals: 303 functions/methods, 5291 lines; 4141 lines last changed by Claude, 1150 upstream.

## Proposal

- **[x] pre-ticked**: 196 functions where >= 50% of the lines are mine.
- **[x] mixed**: 13 functions (20-50% mine), marked too.
- Marker placement: a `# CLAUDE` line directly above the `def` (above decorators),
  so the post-commit count equals the number of marked functions.

Marking everything I wrote gives an honest starting count (~196). It falls
as you read or rewrite functions and remove the markers.

## Suggested first reviews

Highest risk of a plausible-but-wrong convention, i.e. the kind of error this
project has actually had:

1. `hpc/numpy_sampler.py`: `numpy_statevector`, `sample_counts`. qubit/bit order and signs; replaces cudaq in every rt_QC run.
2. `gqe_qsci/gqe/operator_pool.py`: `UCCSDBasedPool.make_excitation_operator`, `UCCSDBasedPool.mp2_angle`, `ExcitationPool.build_operator_pool`. Jordan-Wigner coefficients; the absolute angle convention is still unverified.
3. `gqe_qsci/qsci/baseline.py`: `_determinants_of`, `_full_ci_ordered`, `bisect_minimal`. determinant order must match coefficient order; behind every subspace-requirement number.
4. `hpc/entropy_scan.py`: `marginal_occupations`, `entropy_bits`, `diagonal`. the R_1/4 law; ecore in the diagonal.
5. `gqe_qsci/factory.py`: `Factory._resolve_max_dim`. sets the subspace cap for every run.
6. `gqe_qsci/gqe/models/pointer.py`: `ExcitationRules.__init__`. spin and occupation validity masks.

## Checklist

Columns: line number, lines in the function, % of them last changed by Claude.

### `gqe_qsci/factory.py`  (9 of 14 functions listed)

- [x] `MoleculeBundle.feat_dim` (line 57, 2 lines, 100%)
- [x] `Factory.create_model` (line 67, 39 lines, 97%)
- [x] `Factory._make_pool` (line 144, 25 lines, 60%)
- [x] `Factory._resolve_max_dim` (line 171, 44 lines, 100%)
- [x] `Factory._make_qsci_pipeline` (line 216, 11 lines, 45%)
- [x] `Factory.create_operator_pool` (line 228, 6 lines, 100%)
- [x] `Factory.create_qsci_pipeline` (line 235, 4 lines, 100%)
- [x] `Factory._expand_molecule_set` (line 244, 88 lines, 100%)
- [x] `Factory.create_molecule_bundles` (line 333, 56 lines, 100%)

### `gqe_qsci/gqe/buffer.py`  (4 of 10 functions listed)

- [x] `ReplayBuffer.push` (line 28, 14 lines, 86%)
- [x] `ReplayBuffer.__getitem__` (line 51, 9 lines, 78%)
- [x] `BufferDataset.__len__` (line 80, 2 lines, 50%)
- [x] `buffer_collate_fn` (line 84, 36 lines, 100%)

### `gqe_qsci/gqe/loss.py`  (2 of 9 functions listed)

- [x] `Loss.__call__` (line 20, 2 lines, 100%)
- [x] `GroupRelativeLoss.__call__` (line 38, 2 lines, 50%)

### `gqe_qsci/gqe/models/dag_gnn.py`  (14 of 14 functions listed)

- [x] `_require_pyg` (line 18, 6 lines, 100%)
- [x] `_pack_footprints` (line 26, 12 lines, 100%)
- [x] `CircuitDAGGNNPolicy.__init__` (line 101, 103 lines, 100%)
- [x] `CircuitDAGGNNPolicy.set_molecule` (line 209, 34 lines, 100%)
- [x] `CircuitDAGGNNPolicy._node_table` (line 248, 22 lines, 100%)
- [x] `CircuitDAGGNNPolicy._assemble_node_embeddings` (line 271, 37 lines, 100%)
- [x] `CircuitDAGGNNPolicy._step_forward` (line 309, 67 lines, 100%)
- [x] `CircuitDAGGNNPolicy._advance_dag` (line 377, 25 lines, 100%)
- [x] `CircuitDAGGNNPolicy._canonical_mask` (line 405, 37 lines, 100%)
- [x] `CircuitDAGGNNPolicy._scaled_logits` (line 443, 14 lines, 100%)
- [x] `CircuitDAGGNNPolicy._init_dag_state` (line 458, 18 lines, 100%)
- [x] `CircuitDAGGNNPolicy.act` (line 481, 5 lines, 100%)
- [x] `CircuitDAGGNNPolicy.sample_sequence` (line 487, 38 lines, 100%)
- [x] `CircuitDAGGNNPolicy.log_prob` (line 526, 78 lines, 100%)

### `gqe_qsci/gqe/models/diffusion.py`  (17 of 17 functions listed)

- [x] `_make_alpha_schedule` (line 16, 24 lines, 100%)
- [x] `_CircuitDiffusionBase.__init__` (line 73, 51 lines, 100%)
- [x] `_CircuitDiffusionBase._logits` (line 125, 34 lines, 100%)
- [x] `_CircuitDiffusionBase.act` (line 160, 5 lines, 100%)
- [x] `_CircuitDiffusionBase.set_molecule` (line 166, 14 lines, 100%)
- [x] `_CircuitDiffusionBase._refresh_mask_token` (line 181, 3 lines, 100%)
- [x] `CircuitDiffusionModelSimple.__init__` (line 202, 18 lines, 100%)
- [x] `CircuitDiffusionModelSimple.sample_sequence` (line 221, 15 lines, 100%)
- [x] `CircuitDiffusionModelSimple.log_prob` (line 237, 15 lines, 100%)
- [x] `CircuitDiffusionModelAbsorbing.__init__` (line 301, 27 lines, 100%)
- [x] `CircuitDiffusionModelAbsorbing._refresh_mask_token` (line 329, 3 lines, 100%)
- [x] `CircuitDiffusionModelAbsorbing.sample_sequence` (line 335, 77 lines, 100%)
- [x] `CircuitDiffusionModelAbsorbing.log_prob` (line 415, 77 lines, 100%)
- [x] `CircuitDiffusionModelSingleShot.__init__` (line 531, 29 lines, 100%)
- [x] `CircuitDiffusionModelSingleShot._refresh_mask_token` (line 561, 2 lines, 100%)
- [x] `CircuitDiffusionModelSingleShot.sample_sequence` (line 566, 23 lines, 100%)
- [x] `CircuitDiffusionModelSingleShot.log_prob` (line 592, 31 lines, 100%)

### `gqe_qsci/gqe/models/gnn.py`  (12 of 12 functions listed)

- [x] `_require_pyg` (line 34, 8 lines, 100%)
- [x] `_CircuitGNNBase.__init__` (line 70, 67 lines, 100%)
- [x] `_CircuitGNNBase._batch_edge_index` (line 138, 14 lines, 100%)
- [x] `_CircuitGNNBase._logits` (line 153, 47 lines, 100%)
- [x] `_CircuitGNNBase.act` (line 201, 5 lines, 100%)
- [x] `_CircuitGNNBase.set_molecule` (line 207, 12 lines, 100%)
- [x] `_CircuitGNNBase._refresh_mask_token` (line 220, 2 lines, 100%)
- [x] `_build_edge_index` (line 228, 23 lines, 100%)
- [x] `CircuitGNNModelAbsorbing.__init__` (line 273, 27 lines, 100%)
- [x] `CircuitGNNModelAbsorbing._refresh_mask_token` (line 301, 2 lines, 100%)
- [x] `CircuitGNNModelAbsorbing.sample_sequence` (line 306, 58 lines, 100%)
- [x] `CircuitGNNModelAbsorbing.log_prob` (line 367, 54 lines, 100%)

### `gqe_qsci/gqe/models/gpt2.py`  (6 of 12 functions listed)

- [x] `_FeatureWTE.__init__` (line 37, 3 lines, 100%)
- [x] `_FeatureWTE.forward` (line 41, 2 lines, 100%)
- [x] `_ScorerHead.__init__` (line 56, 3 lines, 100%)
- [x] `_ScorerHead.forward` (line 60, 2 lines, 100%)
- [x] `GPT2Model.__init__` (line 65, 35 lines, 77%)
- [x] `GPT2Model.set_molecule` (line 101, 11 lines, 100%)

### `gqe_qsci/gqe/models/operator_scorer.py`  (9 of 9 functions listed)

- [x] `OperatorScorer.__init__` (line 55, 20 lines, 100%)
- [x] `OperatorScorer._as_tensor` (line 77, 2 lines, 100%)
- [x] `OperatorScorer.set_features` (line 80, 21 lines, 100%)
- [x] `OperatorScorer.set_normalization` (line 102, 13 lines, 100%)
- [x] `OperatorScorer.vocab_size` (line 117, 2 lines, 100%)
- [x] `OperatorScorer.keys` (line 120, 10 lines, 100%)
- [x] `OperatorScorer.forward` (line 131, 10 lines, 100%)
- [x] `SpecialTokenEmbedding.__init__` (line 157, 7 lines, 100%)
- [x] `SpecialTokenEmbedding.forward` (line 165, 16 lines, 100%)

### `gqe_qsci/gqe/models/pointer.py`  (25 of 25 functions listed)

- [x] `build_orbital_inputs` (line 72, 33 lines, 100%)
- [x] `ExcitationRules.__init__` (line 130, 29 lines, 100%)
- [x] `ExcitationRules.to` (line 164, 22 lines, 100%)
- [x] `ExcitationRules._n_beta` (line 189, 5 lines, 100%)
- [x] `ExcitationRules.step_mask` (line 197, 61 lines, 100%)
- [x] `_Block.__init__` (line 267, 12 lines, 100%)
- [x] `_Block.forward` (line 280, 5 lines, 100%)
- [x] `OrbitalEncoder.__init__` (line 304, 17 lines, 100%)
- [x] `OrbitalEncoder.set_normalization` (line 322, 9 lines, 100%)
- [x] `OrbitalEncoder.forward` (line 332, 8 lines, 100%)
- [x] `ExcitationPointer.__init__` (line 364, 12 lines, 100%)
- [x] `ExcitationPointer._pair_context` (line 378, 16 lines, 100%)
- [x] `ExcitationPointer.forward` (line 395, 56 lines, 100%)
- [x] `gate_embedding` (line 457, 18 lines, 100%)
- [x] `excitation_qubits` (line 477, 12 lines, 100%)
- [x] `excitation_pairs` (line 491, 10 lines, 100%)
- [x] `PointerActionSpace.__init__` (line 540, 25 lines, 100%)
- [x] `PointerActionSpace._attach` (line 568, 5 lines, 100%)
- [x] `PointerActionSpace.set_molecule` (line 574, 12 lines, 100%)
- [x] `PointerActionSpace.keys` (line 589, 7 lines, 100%)
- [x] `PointerActionSpace.decode` (line 597, 17 lines, 100%)
- [x] `PointerActionSpace.embed` (line 615, 3 lines, 100%)
- [x] `PointerActionSpace.footprint` (line 619, 3 lines, 100%)
- [x] `PointerActionSpace.to_indices` (line 625, 16 lines, 100%)
- [x] `PointerActionSpace.to_picks` (line 642, 17 lines, 100%)

### `gqe_qsci/gqe/models/pointer_dag.py`  (12 of 12 functions listed)

- [x] `_require_pyg` (line 56, 6 lines, 100%)
- [x] `PointerDAGGNNPolicy.__init__` (line 80, 40 lines, 100%)
- [x] `PointerDAGGNNPolicy.n_qubits` (line 124, 3 lines, 100%)
- [x] `PointerDAGGNNPolicy.set_molecule` (line 128, 3 lines, 100%)
- [x] `PointerDAGGNNPolicy._init_dag_state` (line 134, 4 lines, 100%)
- [x] `PointerDAGGNNPolicy._node_embeddings` (line 139, 12 lines, 100%)
- [x] `PointerDAGGNNPolicy._pool_frontier` (line 152, 26 lines, 100%)
- [x] `PointerDAGGNNPolicy._advance` (line 179, 7 lines, 100%)
- [x] `PointerDAGGNNPolicy._rollout` (line 189, 33 lines, 100%)
- [x] `PointerDAGGNNPolicy.act` (line 225, 5 lines, 100%)
- [x] `PointerDAGGNNPolicy.sample_sequence` (line 231, 7 lines, 100%)
- [x] `PointerDAGGNNPolicy.log_prob` (line 239, 18 lines, 100%)

### `gqe_qsci/gqe/models/policy.py`  (3 of 3 functions listed)

- [x] `Policy.act` (line 7, 2 lines, 50%)
- [x] `Policy.log_prob` (line 11, 2 lines, 100%)
- [x] `Policy.set_molecule` (line 14, 16 lines, 100%)

### `gqe_qsci/gqe/operator_pool.py`  (18 of 34 functions listed)

- [x] `UCCSDBasedPool.get_qubit_footprints` (line 62, 18 lines, 100%)
- [x] `UCCSDBasedPool.get_orbital_features` (line 113, 27 lines, 100%)
- [x] `UCCSDBasedPool.get_xy_qubit_footprints` (line 141, 17 lines, 100%)
- [x] `UCCSDBasedPool.get_pauli_words` (line 159, 20 lines, 100%)
- [x] `UCCSDBasedPool.get_commutation_matrix` (line 180, 31 lines, 100%)
- [x] `UCCSDBasedPool._tq_molecule` (line 224, 14 lines, 100%)
- [x] `UCCSDBasedPool.mp2_angle` (line 239, 27 lines, 100%)
- [x] `UCCSDBasedPool.make_excitation_operator` (line 267, 51 lines, 100%)
- [x] `UCCSDBasedPool.ensure_excitation` (line 319, 33 lines, 100%)
- [x] `UCCSDBasedPool._hf_coupling` (line 354, 34 lines, 100%)
- [x] `UCCSDBasedPool.get_operator_features` (line 389, 57 lines, 100%)
- [x] `UCCSDBasedPool._excitation_key` (line 448, 15 lines, 100%)
- [x] `UCCSDBasedPool.generate_excitations` (line 464, 57 lines, 54%)
- [x] `UCCSDBasedPool.make_uccsd_ansatz` (line 522, 14 lines, 29%)
- [x] `PauliEvolutionPool.__init__` (line 539, 17 lines, 53%)
- [x] `PauliEvolutionPool.build_operator_pool` (line 560, 50 lines, 62%)
- [x] `ExcitationPool.__init__` (line 633, 16 lines, 100%)
- [x] `ExcitationPool.build_operator_pool` (line 653, 44 lines, 75%)

### `gqe_qsci/gqe/sampler.py`  (1 of 6 functions listed)

- [x] `Sampler.run` (line 27, 63 lines, 41%)

### `gqe_qsci/gqe/scheduler.py`  (10 of 11 functions listed)

- [x] `TemperatureScheduler.get_inverse_temperature` (line 35, 3 lines, 33%)
- [x] `TemperatureScheduler.update` (line 40, 7 lines, 29%)
- [x] `DefaultScheduler.__init__` (line 64, 4 lines, 25%)
- [x] `DefaultScheduler.get_inverse_temperature` (line 69, 2 lines, 50%)
- [x] `DefaultScheduler.update` (line 72, 7 lines, 43%)
- [x] `CosineScheduler.get_inverse_temperature` (line 100, 2 lines, 50%)
- [x] `CosineScheduler.update` (line 103, 10 lines, 30%)
- [x] `VarBasedScheduler.__init__` (line 147, 4 lines, 25%)
- [x] `VarBasedScheduler.get_inverse_temperature` (line 152, 2 lines, 50%)
- [x] `VarBasedScheduler.update` (line 155, 15 lines, 33%)

### `gqe_qsci/molecule.py`  (4 of 13 functions listed)

- [x] `PySCFMolecule._save_cache` (line 48, 13 lines, 100%)
- [x] `PySCFMolecule.active_mo_energy` (line 69, 3 lines, 100%)
- [x] `PySCFMolecule.active_mo_occ` (line 74, 3 lines, 100%)
- [x] `PySCFMolecule.n_determinants` (line 79, 14 lines, 100%)

### `gqe_qsci/qsci/baseline.py`  (13 of 13 functions listed)

- [x] `_ndet` (line 29, 6 lines, 100%)
- [x] `build_pyci_hamiltonian` (line 37, 6 lines, 100%)
- [x] `fci_dimension` (line 45, 3 lines, 100%)
- [x] `hf_determinant` (line 50, 4 lines, 100%)
- [x] `diagonalize` (line 56, 9 lines, 100%)
- [x] `random_determinants` (line 71, 24 lines, 100%)
- [x] `random_curve` (line 97, 13 lines, 100%)
- [x] `hci_curve` (line 116, 26 lines, 100%)
- [x] `_determinants_of` (line 148, 29 lines, 100%)
- [x] `oracle_curve` (line 179, 56 lines, 100%)
- [x] `_full_ci_ordered` (line 237, 22 lines, 100%)
- [x] `bisect_minimal` (line 261, 38 lines, 100%)
- [x] `minimal_subspace` (line 301, 19 lines, 100%)

### `gqe_qsci/qsci/pipeline.py`  (1 of 3 functions listed)

- [x] `QSCIPipeline.diagonalize` (line 45, 8 lines, 25%)

### `gqe_qsci/train_pipeline.py`  (22 of 26 functions listed)

- [x] `TrainPipeline.__init__` (line 33, 21 lines, 48%)
- [x] `TrainPipeline._init_single_molecule` (line 59, 6 lines, 100%)
- [x] `TrainPipeline._init_multi_molecule` (line 66, 63 lines, 100%)
- [x] `TrainPipeline._operator_scorer` (line 130, 5 lines, 100%)
- [x] `TrainPipeline._orbital_encoder` (line 136, 5 lines, 100%)
- [x] `TrainPipeline._reference_energies` (line 142, 19 lines, 100%)
- [x] `TrainPipeline._activate_bundle` (line 162, 9 lines, 100%)
- [x] `TrainPipeline._next_train_bundle` (line 172, 4 lines, 100%)
- [x] `TrainPipeline._tracker_key` (line 178, 2 lines, 100%)
- [x] `TrainPipeline._make_results_writer` (line 181, 10 lines, 100%)
- [x] `TrainPipeline._metric_prefix` (line 193, 3 lines, 100%)
- [x] `TrainPipeline.on_fit_start` (line 197, 10 lines, 30%)
- [x] `TrainPipeline._apply_warm_start` (line 208, 52 lines, 100%)
- [x] `TrainPipeline.on_train_epoch_start` (line 261, 24 lines, 75%)
- [x] `TrainPipeline.on_train_epoch_end` (line 286, 10 lines, 80%)
- [x] `TrainPipeline._zeroshot_eval` (line 302, 46 lines, 100%)
- [x] `TrainPipeline._log_dissociation_curve` (line 349, 70 lines, 100%)
- [x] `TrainPipeline.collect_rollout` (line 420, 50 lines, 50%)
- [x] `TrainPipeline._update_bests` (line 471, 12 lines, 100%)
- [x] `TrainPipeline.training_step` (line 485, 54 lines, 61%)
- [x] `TrainPipeline.on_save_checkpoint` (line 561, 22 lines, 82%)
- [x] `TrainPipeline.on_load_checkpoint` (line 584, 16 lines, 62%)

### `hpc/energy_vs_overlap.py`  (3 of 3 functions listed)

- [x] `already_done` (line 67, 5 lines, 100%)
- [x] `compare` (line 74, 29 lines, 100%)
- [x] `main` (line 105, 74 lines, 100%)

### `hpc/entropy_scan.py`  (15 of 15 functions listed)

- [x] `_binary_entropy_bits` (line 97, 9 lines, 100%)
- [x] `marginal_occupations` (line 108, 18 lines, 100%)
- [x] `_renyi_bits` (line 140, 6 lines, 100%)
- [x] `_spectral` (line 148, 6 lines, 100%)
- [x] `entropy_bits` (line 156, 40 lines, 100%)
- [x] `_solve` (line 202, 4 lines, 100%)
- [x] `diagonal` (line 208, 11 lines, 100%)
- [x] `_entropies` (line 221, 13 lines, 100%)
- [x] `cisd_entropy` (line 236, 7 lines, 100%)
- [x] `fci_entropy` (line 245, 6 lines, 100%)
- [x] `hf_entropy` (line 253, 7 lines, 100%)
- [x] `_ndet` (line 268, 5 lines, 100%)
- [x] `measured_requirements` (line 291, 20 lines, 100%)
- [x] `already_done` (line 313, 5 lines, 100%)
- [x] `main` (line 328, 106 lines, 100%)

### `hpc/numpy_sampler.py`  (4 of 4 functions listed)

- [x] `circuit_terms` (line 42, 7 lines, 100%)
- [x] `numpy_statevector` (line 51, 34 lines, 100%)
- [x] `sample_counts` (line 87, 12 lines, 100%)
- [x] `verify` (line 101, 11 lines, 100%)

### `hpc/subspace_requirement.py`  (5 of 5 functions listed)

- [x] `G.__init__` (line 60, 2 lines, 100%)
- [x] `_r` (line 64, 3 lines, 100%)
- [x] `configurations` (line 112, 8 lines, 100%)
- [x] `already_done` (line 122, 5 lines, 100%)
- [x] `main` (line 129, 87 lines, 100%)
