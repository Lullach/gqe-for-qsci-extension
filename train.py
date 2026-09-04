import os
import functools

import torch
import hydra, wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from gqe_qsci.train_pipeline import TrainPipeline
from gqe_qsci.factory import Factory

torch.load = functools.partial(torch.load, weights_only=False)

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_float32_matmul_precision("medium")


def setup_callbacks(cfg):
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{cfg.output}/models",
            filename="{epoch:02d}",
            save_top_k=1,
            monitor="epoch",
            mode="max",
            save_last=True,
            every_n_epochs=cfg.trainer.checkpoint_every_n_iters,
            enable_version_counter=False
        ),
    ]
    return callbacks


def get_checkpoint_path(cfg):
    if not cfg.trainer.load_checkpoint:
        print("No checkpoint loading requested. Training from scratch.")
        return None
    path = f"{cfg.output}/models/last.ckpt"
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        return path
    print(f"Checkpoint {path} does not exist. Training from scratch.")
    return None

def run_provenance(cfg):
    """
    (choices, setup) — what this run WAS, in a form readable at a glance.

    The full resolved config already reaches wandb.config, but that is ~100
    nested keys and nobody scans it to remember what an experiment did. These
    two flat fields are meant to be shown as columns in the W&B runs table:

      choices  which config groups were composed (experiment / model /
               molecule_set / molecule) — the thing you actually chose
      setup    one line with the knobs that change a result

    Hydra records the composition in runtime.choices, so this stays correct
    when configs are overridden on the command line — unlike anything written
    by hand into the config files.
    """
    try:
        choices = {k: str(v) for k, v in HydraConfig.get().runtime.choices.items()
                   if not k.startswith("hydra/")}
    except ValueError:      # not inside @hydra.main (e.g. imported for tests)
        choices = {}

    where = (f"molecule_set={choices['molecule_set']}"
             if choices.get("molecule_set") not in (None, "None")
             else f"molecule={choices.get('molecule')}")
    setup = " | ".join([
        f"experiment={choices.get('experiment')}",
        f"model={choices.get('model')}",
        where,
        f"L={cfg.ngates}",
        f"iters={cfg.trainer.max_iters}",
        f"samples={cfg.trainer.num_samples}",
        f"shots={cfg.sampler.shots}",
        f"dmax={cfg.qsci.max_dim}",
        f"seed={cfg.trainer.seed}",
    ])
    return choices, setup


def setup_logger(cfg):
    save_dir = cfg.output
    run_id_file = os.path.join(save_dir, "run_id")
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        print(f"Resuming training from run_id: {run_id}")
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    choices, setup = run_provenance(cfg)
    wandb_config["choices"] = choices
    wandb_config["setup"] = setup
    print(f"setup: {setup}")

    logger = WandbLogger(
        project=cfg.wandb.project,
        settings=wandb.Settings(init_timeout=300),
        config=wandb_config,
        save_dir=save_dir,
        name=cfg.wandb.name,
        group=cfg.wandb.get("group", None),
        tags=list(cfg.wandb.get("tags", [])),
        # Free-text description of what the experiment is FOR; set it in the
        # experiment config's wandb.notes. Shows in the W&B run overview.
        notes=cfg.wandb.get("notes", None),
        id=run_id if run_id is not None else None,
        resume='allow'
    )
    if not os.path.exists(run_id_file): 
        run_id = logger.experiment.id
        with open(run_id_file, "w") as f:
            f.write(str(run_id))

    return logger


def train(cfg):
    # Seed everything before any model weights are initialised so runs are reproducible.
    seed_everything(cfg.trainer.seed, workers=True)
    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)
    training = TrainPipeline(Factory(), cfg)
    trainer = Trainer(
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_iters,
        deterministic=True, 
        reload_dataloaders_every_n_epochs=1,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=10
    )
    trainer.fit(training, ckpt_path=get_checkpoint_path(cfg))

@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
    print("training finished")
