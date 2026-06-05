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
from torch.utils.data import DataLoader
import random
import pytorch_lightning as pl

_log = logging.getLogger(__name__)

from gqe_qsci.gqe.buffer import ReplayBuffer, BufferDataset, buffer_collate_fn
from gqe_qsci.qsci.schema import QSCISampleResult
from gqe_qsci.qsci.pipeline import as_scivector


class TrainPipeline(pl.LightningModule):
    def __init__(self, factory, config):
        super().__init__()
        self.config = config
        self.factory = factory
        self.loss_fn = self.factory.create_loss_fn(config)
        self.qsci_pipeline = self.factory.create_qsci_pipeline(config)
        self.model = self.factory.create_model(config)
        self.scheduler = self.factory.create_temperature_scheduler(self.config)
        self.metric_logger = self.factory.create_wandb_logger(config)
        self.warmup_size = config.trainer.warmup_size
        self.ngates = config.ngates
        self.num_samples = config.trainer.num_samples
        self.best_sample: QSCISampleResult | None = None
        self.best_local_refined: QSCISampleResult | None = None
        self.best_global_refined: QSCISampleResult | None = None
        self.buffer = ReplayBuffer(size=config.trainer.buffer_size)

    def on_fit_start(self):
        run = self.logger.experiment
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        self._apply_warm_start()
        while len(self.buffer) < self.warmup_size:
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
        qsci_result = self.collect_rollout(log=True)
        log_inputs = [
            {"result": qsci_result, "prefix": "GQE-optimized"},
            {"result": self.best_sample, "prefix": "GQE-optimized(best_so_far)"},
        ]
        if self.best_local_refined is not None:
            log_inputs.append({"result": self.best_local_refined, "prefix": "Local-refined(best_so_far)"})
        if self.best_global_refined is not None:
            log_inputs.append({"result": self.best_global_refined, "prefix": "Global-refined(best_so_far)"})
        self.metric_logger.log_result(self, log_inputs)
        super().on_train_epoch_start()
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()

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
            if self.best_sample is None or energies.min() < self.best_sample.energy:
                self.best_sample = qsci_result.best_sample
            if self.best_local_refined is None or qsci_result.local_refined.energy < self.best_local_refined.energy:
                self.best_local_refined = qsci_result.local_refined
            if self.best_global_refined is None or qsci_result.global_refined.energy < self.best_global_refined.energy:
                self.best_global_refined = qsci_result.global_refined

        self.scheduler.update(energies=energies)
        return qsci_result


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
            "trainer/inv temperature",
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
        scistate = self.qsci_pipeline.global_refined_scistates
        scistate_data = {
            "coeffs": np.asarray(scistate),
            "strs": getattr(scistate, "_strs", None)
        }
        checkpoint["extra_info"] = {
            "inverse_temperature": self.scheduler.get_inverse_temperature(),
            "best_sample": self.best_sample,
            "best_local_refined": self.best_local_refined,
            "best_global_refined": self.best_global_refined,
            "global_refined_scistates": scistate_data,
        }
        self.buffer.save(f"{self.config.output}/buffer.pkl")

    def on_load_checkpoint(self, checkpoint):
        extra_info = checkpoint.get("extra_info", {})
        if "inverse_temperature" in extra_info:
            self.scheduler.current_temperature = extra_info["inverse_temperature"]
        if "best_sample" in extra_info:
            self.best_sample = extra_info["best_sample"]
        if "best_local_refined" in extra_info:
            self.best_local_refined = extra_info["best_local_refined"]
        if "best_global_refined" in extra_info:
            self.best_global_refined = extra_info["best_global_refined"]
        if "global_refined_scistates" in extra_info:
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
