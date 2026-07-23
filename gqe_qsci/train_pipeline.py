# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Modifications Copyright (c) 2026 Ryota Kemmoku
# Modified from the original file in NVIDIA CUDA-QX.
# Changes made: add refinement post-processing and Weights & Biases logging.


import logging
import os

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
import random
import pytorch_lightning as pl

_log = logging.getLogger(__name__)

from gqe_qsci.gqe.buffer import ReplayBuffer, BufferDataset, buffer_collate_fn
from gqe_qsci.gqe.models.operator_scorer import OperatorScorer
from gqe_qsci.qsci.schema import QSCISampleResult
from gqe_qsci.qsci.pipeline import as_scivector
from gqe_qsci.wandb_logger import Logger


class TrainPipeline(pl.LightningModule):
    def __init__(self, factory, config):
        super().__init__()
        self.config = config
        self.factory = factory
        self.loss_fn = self.factory.create_loss_fn(config)
        self.scheduler = self.factory.create_temperature_scheduler(self.config)
        self.warmup_size = config.trainer.warmup_size
        self.ngates = config.ngates
        self.num_samples = config.trainer.num_samples
        self.buffer = ReplayBuffer(size=config.trainer.buffer_size)

        # best-so-far trackers are keyed by a "tracker key" — the molecule name
        # in multi-molecule mode, a fixed key otherwise — so energies from
        # different molecules (e.g. -7.9 vs -107 Ha) are never compared.
        self._best: dict[str, dict[str, QSCISampleResult | None]] = {}

        self.multi_molecule = config.get("molecule_set") is not None
        if self.multi_molecule:
            self._init_multi_molecule()
        else:
            self._init_single_molecule()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _init_single_molecule(self):
        self.qsci_pipeline = self.factory.create_qsci_pipeline(self.config)
        self.model = self.factory.create_model(self.config)
        self.metric_logger = self.factory.create_wandb_logger(self.config)
        self.current_bundle = None

    def _init_multi_molecule(self):
        """
        Build one bundle per molecule, create the model ONCE from the first
        training bundle, and freeze feature-normalization stats over the whole
        training set (Phase 2 steps 4-5). Molecules are then swapped in per
        rollout group via _activate_bundle().
        """
        assert self.config.trainer.buffer_size == self.num_samples, (
            "multi-molecule training requires buffer_size == num_samples so the "
            "replay buffer never mixes molecules across rollout groups."
        )
        bundles = self.factory.create_molecule_bundles(self.config)
        self.bundles = bundles
        self.train_bundles = [b for b in bundles.values() if b.split == "train"]
        self.eval_bundles = [b for b in bundles.values() if b.split == "eval"]
        assert self.train_bundles, "molecule_set expanded to no training molecules"

        first = self.train_bundles[0]
        self.model = self.factory.create_model(self.config, op_pool=first.pool)

        scorer = self._operator_scorer()
        if scorer is None:
            raise ValueError(
                "molecule_set requires a feature-based model "
                "(e.g. model=dag_gnn_features); the integer-ID head cannot "
                "transfer across molecules."
            )

        # Step 5: global normalization over the pooled training-set features.
        train_feats = np.concatenate(
            [b.operator_features for b in self.train_bundles], axis=0
        )
        scorer.set_normalization(
            train_feats.mean(axis=0, keepdims=True),
            train_feats.std(axis=0, keepdims=True),
        )

        self._refs: dict[str, dict] = {}     # molecule name -> reference energies
        self.metric_logger = Logger(reference_energies=None)
        self.current_bundle = None
        self._rr = 0                         # round-robin pointer over train set
        _log.info(
            "Multi-molecule: %d train, %d eval; feature stats frozen over %d "
            "operators.",
            len(self.train_bundles), len(self.eval_bundles), train_feats.shape[0],
        )

    def _operator_scorer(self):
        """The model's single OperatorScorer (one instance even when wte/lm_head
        share it), or None for an integer-ID model."""
        scorers = [m for m in self.model.modules() if isinstance(m, OperatorScorer)]
        return scorers[0] if scorers else None

    def _reference_energies(self, bundle):
        """Per-molecule reference energies, computed once and cached (disk-cached
        by molecule.py, so cheap on re-runs)."""
        if bundle.name in self._refs:
            return self._refs[bundle.name]
        refs = {}
        for key in self.config.reference_keys:
            if key == "hf_energy":
                refs[key] = bundle.molecule.hf.e_tot
            elif key == "R-CASCI":
                refs[key] = bundle.molecule.compute_casci()
            elif key == "R-CCSD":
                refs[key] = bundle.molecule.compute_ccsd()
        # compute_casci() returns a 0-d numpy array (loaded from the .npz cache);
        # cast to plain float so wandb custom charts (which reject ndarrays) and
        # arithmetic downstream both behave.
        refs = {k: (float(v) if v is not None else None) for k, v in refs.items()}
        self._refs[bundle.name] = refs
        return refs

    def _activate_bundle(self, bundle):
        """Atomically point the whole pipeline at one molecule: swap the model's
        buffers, the QSCI pipeline, and the logger's reference energies."""
        if self.current_bundle is not None and self.current_bundle.name == bundle.name:
            return
        self.model.set_molecule(bundle)
        self.qsci_pipeline = bundle.qsci_pipeline
        self.metric_logger.reference_energies = self._reference_energies(bundle)
        self.current_bundle = bundle

    def _next_train_bundle(self):
        b = self.train_bundles[self._rr % len(self.train_bundles)]
        self._rr += 1
        return b

    @property
    def _tracker_key(self) -> str:
        return self.current_bundle.name if self.multi_molecule else "_single_"

    @property
    def _metric_prefix(self) -> str:
        """Namespace metrics by molecule so per-molecule curves stay separate."""
        return f"{self.current_bundle.name}/" if self.multi_molecule else ""

    def on_fit_start(self):
        run = self.logger.experiment
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        self._apply_warm_start()
        while len(self.buffer) < self.warmup_size:
            if self.multi_molecule:
                self._activate_bundle(self._next_train_bundle())
            self.collect_rollout(log=False)
        super().on_fit_start()

    def _apply_warm_start(self):
        """
        Optionally load model weights from a previous checkpoint before training
        begins, without restoring optimizer state or epoch counters.

        This is the classical analogue of consistency-model distillation: a
        trained absorbing-diffusion checkpoint can warm-start the single-shot
        model because both share the same _CircuitDiffusionBase architecture.
        Weights that don't exist in the target model (e.g. the 'alpha' schedule
        buffer present in the absorbing model but not in the single-shot model)
        are silently skipped via strict=False.

        Set  trainer.warm_start_checkpoint  in the experiment config to enable.
        The path should point to a PyTorch Lightning .ckpt file.
        """
        path = getattr(self.config.trainer, "warm_start_checkpoint", None)
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"warm_start_checkpoint not found: {path}"
            )
        _log.info(f"Warm-starting model weights from {path}")
        ckpt = torch.load(path, map_location="cpu")

        # PL checkpoints store the full LightningModule state dict under
        # "state_dict" with keys prefixed "model." (e.g. "model.denoiser.…").
        full_sd = ckpt.get("state_dict", ckpt)
        model_sd = {
            k[len("model."):]: v
            for k, v in full_sd.items()
            if k.startswith("model.")
        }
        if not model_sd:
            _log.warning(
                "Warm-start: no 'model.*' keys found in checkpoint — "
                "trying to load the dict directly."
            )
            model_sd = full_sd

        missing, unexpected = self.model.load_state_dict(model_sd, strict=False)
        if missing:
            _log.info(
                f"Warm-start: {len(missing)} missing key(s) "
                f"(expected when architectures differ, e.g. 'alpha' buffer): "
                f"{missing}"
            )
        if unexpected:
            _log.warning(
                f"Warm-start: {len(unexpected)} unexpected key(s): {unexpected}"
            )
        _log.info("Warm-start complete.")

    def on_train_epoch_start(self):
        if self.multi_molecule:
            self._activate_bundle(self._next_train_bundle())
        qsci_result = self.collect_rollout(log=True)

        best = self._best[self._tracker_key]
        p = self._metric_prefix
        log_inputs = [
            {"result": qsci_result, "prefix": f"{p}GQE-optimized"},
            {"result": best["sample"], "prefix": f"{p}GQE-optimized(best_so_far)"},
        ]
        if best["local"] is not None:
            log_inputs.append({"result": best["local"], "prefix": f"{p}Local-refined(best_so_far)"})
        if best["global"] is not None:
            log_inputs.append({"result": best["global"], "prefix": f"{p}Global-refined(best_so_far)"})
        self.metric_logger.log_result(self, log_inputs)
        super().on_train_epoch_start()
    
    def on_train_epoch_end(self):
        if self.multi_molecule:
            eval_every = int(self.config.trainer.get("eval_every", 10))
            is_last = self.current_epoch >= (self.config.trainer.max_iters - 1)
            due = ((self.current_epoch + 1) % eval_every == 0) or is_last
            if due:
                if self.eval_bundles:
                    self._zeroshot_eval()
                self._log_dissociation_curve()
        super().on_train_epoch_end()

    # ------------------------------------------------------------------ #
    # Zero-shot evaluation on held-out molecules (Phase 2 step 6)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _zeroshot_eval(self):
        """
        Generate + QSCI for each held-out (split=='eval') molecule with NO
        gradient update and NO buffer push — the generalization measurement.
        Runs in eval mode (dropout off) for a clean, deterministic-ish number.
        """
        was_training = self.model.training
        self.model.eval()
        beta = self.scheduler.get_inverse_temperature()
        for bundle in self.eval_bundles:
            self._activate_bundle(bundle)
            state = {
                "idx": torch.zeros(
                    (self.num_samples, 1), dtype=torch.long, device=self.device
                )
            }
            if hasattr(self.model, "sample_sequence"):
                state = self.model.sample_sequence(state, beta)
            else:
                for _ in range(self.ngates):
                    state = self.update_state(state, self.model.act(state, beta))

            qsci_result = self.qsci_pipeline.process(state)
            energies = torch.tensor(qsci_result.energies, device=self.device)
            self._update_bests(qsci_result, energies)   # keyed by eval-molecule name

            best = self._best[bundle.name]
            p = f"zeroshot/{bundle.name}/"
            log_inputs = [
                {"result": qsci_result, "prefix": f"{p}GQE-optimized"},
                {"result": best["sample"], "prefix": f"{p}GQE-optimized(best_so_far)"},
            ]
            if best["global"] is not None:
                log_inputs.append(
                    {"result": best["global"], "prefix": f"{p}Global-refined(best_so_far)"}
                )
            self.metric_logger.log_result(self, log_inputs)   # refs already the eval molecule's
        if was_training:
            self.model.train()

    def _log_dissociation_curve(self):
        """
        Summary charts vs scan coordinate (bond length), one point per molecule
        (train + zero-shot eval combined, ordered by bond length):

          summary/energy_vs_bond_length   absolute energy, policy vs CASCI/CCSD
          summary/error_vs_bond_length    |policy - CASCI| in mHa, split into
                                          train vs zero-shot series, with the
                                          1.6 mHa chemical-accuracy line
          summary/subspace_dim_vs_bond    QSCI subspace size vs geometry
          summary/energy_table            raw values for custom charts

        None is used to leave gaps in a series (e.g. a train-only point in the
        zero-shot series); wandb renders those as breaks.
        """
        rows = []
        for name, bundle in self.bundles.items():
            if bundle.x_value is None:
                continue
            best = self._best.get(name)
            if not best or best["global"] is None:
                continue
            g = best["global"]
            refs = self._reference_energies(bundle)
            sd = getattr(g, "subspace_dim", None)
            # Coerce every numeric to a plain Python scalar — wandb custom charts
            # reject numpy 0-d arrays / np.float64.
            rows.append({
                "x": float(bundle.x_value),
                "split": bundle.split,
                "energy": float(g.energy),
                "casci": refs.get("R-CASCI"),          # already float from _reference_energies
                "ccsd": refs.get("R-CCSD"),
                "subspace_dim": int(sd) if sd is not None else None,
            })
        if not rows:
            return
        rows.sort(key=lambda r: r["x"])

        def err_mha(r):   # |policy - CASCI| in milli-Hartree
            return abs(r["energy"] - r["casci"]) * 1000.0 if r["casci"] is not None else None

        # Everything logged in ONE run.log so it lands on a single step with epoch.
        payload = {"epoch": int(self.current_epoch)}

        # --- per-molecule SCATTER charts -----------------------------------
        # Scatter (not line_series): plots points only — no misleading linear
        # interpolation between the 7 irregular bond lengths — and auto-scales
        # the y-axis to the data (line_series forces the y-domain toward 0,
        # which squashed the near -107 Ha energies flat). Toggle a panel's
        # y-axis to log scale in the UI to inspect sub-mHa detail.
        def scatter(col, fn, title):
            t = wandb.Table(columns=["bond_length", col])
            for r in rows:
                y = fn(r)
                if y is not None:
                    t.add_data(r["x"], y)
            return wandb.plot.scatter(t, "bond_length", col, title=title)

        # No absolute-energy chart: wandb custom charts force the y-axis to
        # include 0, squashing the near -107.5 Ha energies flat, and the absolute
        # offset is chemically irrelevant anyway. The error-vs-reference scatters
        # below carry the signal on a natural scale; absolute energies remain in
        # summary/energy_table for anyone who wants them.
        payload["summary/err_vs_CASCI_mHa"] = scatter(
            "policy_minus_CASCI_mHa",
            lambda r: (r["energy"] - r["casci"]) * 1000.0 if r["casci"] is not None else None,
            "policy - CASCI per bond length (mHa; chemical accuracy = 1.6)")
        payload["summary/err_vs_CCSD_mHa"] = scatter(
            "policy_minus_CCSD_mHa",
            lambda r: (r["energy"] - r["ccsd"]) * 1000.0 if r["ccsd"] is not None else None,
            "policy - CCSD per bond length (mHa; below 0 beats CCSD)")
        if any(r["subspace_dim"] is not None for r in rows):
            payload["summary/subspace_dim_vs_bond"] = scatter(
                "subspace_dim", lambda r: r["subspace_dim"],
                "QSCI subspace dim vs bond length")

        # --- full table: for custom UI panels (colour by split, reference
        #     line at 1.6 mHa, references overlaid, etc.) --------------------
        table = wandb.Table(columns=[
            "bond_length", "split", "policy_energy", "R-CASCI", "R-CCSD",
            "policy - CASCI (mHa)", "policy - CCSD (mHa)", "abs err CASCI (mHa)",
            "subspace_dim",
        ])
        for r in rows:
            table.add_data(
                r["x"], r["split"], r["energy"], r["casci"], r["ccsd"],
                (r["energy"] - r["casci"]) * 1000.0 if r["casci"] is not None else None,
                (r["energy"] - r["ccsd"]) * 1000.0 if r["ccsd"] is not None else None,
                err_mha(r), r["subspace_dim"],
            )
        payload["summary/energy_table"] = table

        # --- scalar headline metrics (plain wandb line charts, x=epoch; these
        #     DO overlay across runs in the workspace — the model comparison) --
        train_errs = [err_mha(r) for r in rows if r["split"] == "train" and err_mha(r) is not None]
        eval_errs = [err_mha(r) for r in rows if r["split"] == "eval" and err_mha(r) is not None]
        if train_errs:
            payload["summary/train_mean_err_CASCI_mHa"] = float(np.mean(train_errs))
            payload["summary/train_min_err_CASCI_mHa"] = float(np.min(train_errs))
        if eval_errs:
            payload["summary/zeroshot_mean_err_CASCI_mHa"] = float(np.mean(eval_errs))
            payload["summary/zeroshot_max_err_CASCI_mHa"] = float(np.max(eval_errs))
        # generalization gap: how much worse the held-out geometries are
        if train_errs and eval_errs:
            payload["summary/generalization_gap_mHa"] = float(np.mean(eval_errs) - np.mean(train_errs))

        self.logger.experiment.log(payload)

    def collect_rollout(self, log=False):
        state = {
            "idx": torch.zeros(
                (self.config.trainer.num_samples, 1),
                dtype=torch.long,
                device=self.device,
            )
        }
        with torch.no_grad():
            if hasattr(self.model, "sample_sequence"):
                state = self.model.sample_sequence(
                    state, self.scheduler.get_inverse_temperature()
                )
            else:
                for _ in range(self.ngates):
                    next_tokens = self.model.act(
                        state, self.scheduler.get_inverse_temperature()
                    )
                    state = self.update_state(state, next_tokens)

            qsci_result = self.qsci_pipeline.process(state)
            energies = torch.tensor(qsci_result.energies, device=self.device)

            # Pre-sample diffusion masks if the model supports it (Fix 4).
            # Storing masks in the buffer makes log_prob() deterministic during
            # the GRPO training step, stabilising the importance-weight ratio.
            masks_all = None
            if hasattr(self.model, "sample_masks"):
                masks_all = self.model.sample_masks(state["idx"][:, 1:])  # (B, T, L)

            # log-probs under the behavior policy at rollout time
            lp_kwargs = {} if masks_all is None else {"masks": masks_all}
            old_log_probs = self.model.log_prob(
                state["idx"], self.scheduler.get_inverse_temperature(), **lp_kwargs
            )

            masks_iter = masks_all if masks_all is not None else [None] * len(energies)
            for seq, energy, olp, msk in zip(state["idx"], energies, old_log_probs, masks_iter):
                self.buffer.push(
                    seq.detach().cpu(),
                    energy.detach().cpu(),
                    olp.detach().cpu(),
                    msk.detach().cpu() if msk is not None else None,
                )
            self._update_bests(qsci_result, energies)

        self.scheduler.update(energies=energies)
        return qsci_result

    def _update_bests(self, qsci_result, energies):
        """Update the best-so-far trackers for the CURRENT molecule (keyed so
        molecules never compare energies against each other)."""
        best = self._best.setdefault(
            self._tracker_key, {"sample": None, "local": None, "global": None}
        )
        if best["sample"] is None or energies.min() < best["sample"].energy:
            best["sample"] = qsci_result.best_sample
        if best["local"] is None or qsci_result.local_refined.energy < best["local"].energy:
            best["local"] = qsci_result.local_refined
        if best["global"] is None or qsci_result.global_refined.energy < best["global"].energy:
            best["global"] = qsci_result.global_refined


    def training_step(self, batch, _):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)

        # Pass pre-sampled masks if available (Fix 4: deterministic ELBO)
        batch_masks = batch.get("masks")   # None for GPT-2 / Simple
        lp_kwargs = {} if batch_masks is None else {"masks": batch_masks}

        # Entropy regularization coefficient (Fix 5). 0.0 disables it.
        entropy_coeff = getattr(self.config.trainer, "entropy_coeff", 0.0)

        if entropy_coeff > 0.0:
            full_log_probs, entropy = self.model.log_prob(
                batch["idx"], self.scheduler.get_inverse_temperature(),
                return_entropy=True, **lp_kwargs,
            )
        else:
            full_log_probs = self.model.log_prob(
                batch["idx"], self.scheduler.get_inverse_temperature(),
                **lp_kwargs,
            )
            entropy = None

        gate_seqs = batch["idx"][:, 1:]
        energies = batch["energy"]
        context = {
            "old_log_probs": batch["old_log_probs"],
            "energies": energies,
            "gate_seqs": gate_seqs,
        }
        loss = self.loss_fn(full_log_probs, context)

        if entropy_coeff > 0.0 and entropy is not None:
            entropy_mean = entropy.mean()
            self.log(
                "trainer/entropy", entropy_mean,
                on_step=True, on_epoch=True, prog_bar=False, logger=True,
            )
            # Subtract entropy bonus: encourages diversity, acts as regulariser
            loss = loss - entropy_coeff * entropy_mean

        self.log("trainer/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            "trainer/inv_temperature",
            self.scheduler.get_inverse_temperature(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def update_state(self, state, next_token):
        state["idx"] = torch.cat((state["idx"], next_token.unsqueeze(1)), dim=1)
        return state

    def train_dataloader(self):
        return DataLoader(
            BufferDataset(self.buffer, self.config.trainer.step_per_epoch),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
            collate_fn=buffer_collate_fn,
        )

    def configure_optimizers(self):
        base_lr = self.config.trainer.optimizer.lr
        weight_decay = self.config.trainer.optimizer.weight_decay
        optimizer_class = getattr(torch.optim, self.config.trainer.optimizer.cls)
        optimizer = optimizer_class(
            self.model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
        return {"optimizer": optimizer}

    def on_save_checkpoint(self, checkpoint):
        extra = {
            "inverse_temperature": self.scheduler.get_inverse_temperature(),
            "best": self._best,
        }
        if not self.multi_molecule:
            scistate = self.qsci_pipeline.global_refined_scistates
            extra["global_refined_scistates"] = {
                "coeffs": np.asarray(scistate),
                "strs": getattr(scistate, "_strs", None),
            }
        else:
            # Per-molecule QSCI refinement state is not persisted, and the model
            # buffers are sized to whichever molecule is active at save time, so
            # resuming a shape-changed state_dict would mismatch. Multi-molecule
            # runs are expected to use trainer.load_checkpoint=false.
            _log.warning(
                "Multi-molecule checkpoint saves model weights + bests only; "
                "resume is not supported (use load_checkpoint=false)."
            )
        checkpoint["extra_info"] = extra
        self.buffer.save(f"{self.config.output}/buffer.pkl")

    def on_load_checkpoint(self, checkpoint):
        extra_info = checkpoint.get("extra_info", {})
        if "inverse_temperature" in extra_info:
            self.scheduler.current_beta = extra_info["inverse_temperature"]
        if "best" in extra_info:
            self._best = extra_info["best"]
        elif "best_sample" in extra_info:   # legacy single-molecule checkpoint
            self._best["_single_"] = {
                "sample": extra_info.get("best_sample"),
                "local": extra_info.get("best_local_refined"),
                "global": extra_info.get("best_global_refined"),
            }
        if "global_refined_scistates" in extra_info and not self.multi_molecule:
            data = extra_info["global_refined_scistates"]
            self.qsci_pipeline.global_refined_scistates = as_scivector(data["coeffs"], data["strs"])
        self.buffer.load(f"{self.config.output}/buffer.pkl")

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
