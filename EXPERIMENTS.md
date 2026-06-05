# Experiment Commands

These configs keep W&B runs grouped so GPT-2 and diffusion can be overlaid in the
same project charts.

## N2 comparison

```powershell
python train.py experiment=n2_gpt2
python train.py experiment=n2_diffusion
```

## N2 L10 15-iteration comparison

These are the first more serious CPU runs. They use the original-style serious
settings with circuit length `L=10`, 15 training iterations, 10 samples per
rollout, 30 policy updates per epoch, 100,000 shots, and QSCI `max_dim=2000`.

```powershell
python train.py experiment=n2_l10_gpt2_15iter
python train.py experiment=n2_l10_diffusion_small_15iter
python train.py experiment=n2_l10_diffusion_absorbing_15iter
```

All three share the W&B group `n2-L10-policy-comparison` so their charts overlay
automatically. The absorbing diffusion run uses `model=diffusion_absorbing`
(proper forward/reverse process with cosine noise schedule).

## H2 comparison

```powershell
python train.py experiment=h2_gpt2
python train.py experiment=h2_diffusion
```

## Phenylene prepared-only config

```powershell
python train.py experiment=phenylene_gpt2
python train.py experiment=phenylene_diffusion
```

The phenylene config is present for later use, but it is much more expensive than
the N2 and H2 smoke runs.

## Tiny smoke-test overrides

Use these to check plumbing before a real run:

```powershell
python train.py experiment=n2_gpt2 trainer.max_iters=1 trainer.num_samples=2 trainer.batch_size=2 trainer.warmup_size=2 trainer.buffer_size=2 trainer.step_per_epoch=1 sampler.shots=100 qsci.max_dim=100 trainer.load_checkpoint=false exp_tag=n2-gpt2-smoke
python train.py experiment=n2_diffusion trainer.max_iters=1 trainer.num_samples=2 trainer.batch_size=2 trainer.warmup_size=2 trainer.buffer_size=2 trainer.step_per_epoch=1 sampler.shots=100 qsci.max_dim=100 trainer.load_checkpoint=false exp_tag=n2-diffusion-smoke
```

## Docker on Windows

CUDA-Q 0.12.0 is not available as a native Windows Python package, so the
working local path is Docker:

```powershell
docker build -t gqe_qsci_cpu .
```

Future NVIDIA GPU build:

```powershell
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t gqe_qsci_gpu .
```

**PowerShell path note**: `${PWD}` expands incorrectly in PowerShell (Git Bash
path). Always use an explicit variable:

```powershell
$workdir = "C:\Users\Lukas\Documents\Codex\2026-05-24\i-would-like-to-modify-a\gqe-for-qsci"
```

then `-v "${workdir}:/workspace"` in every docker run command below.

Run smoke tests in W&B offline mode:

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_MODE=offline -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_gpt2 trainer.max_iters=1 trainer.num_samples=2 trainer.batch_size=2 trainer.warmup_size=2 trainer.buffer_size=2 trainer.step_per_epoch=1 sampler.shots=100 qsci.max_dim=100 trainer.load_checkpoint=false exp_tag=n2-gpt2-smoke"
docker run --rm --entrypoint /bin/bash -e WANDB_MODE=offline -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_diffusion trainer.max_iters=1 trainer.num_samples=2 trainer.batch_size=2 trainer.warmup_size=2 trainer.buffer_size=2 trainer.step_per_epoch=1 sampler.shots=100 qsci.max_dim=100 trainer.load_checkpoint=false exp_tag=n2-diffusion-smoke"
```

Run the N2 L10 15-iteration comparison in W&B offline mode:

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_MODE=offline -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_gpt2_15iter"
docker run --rm --entrypoint /bin/bash -e WANDB_MODE=offline -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_small_15iter"
docker run --rm --entrypoint /bin/bash -e WANDB_MODE=offline -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_absorbing_15iter"
```

## N2 L10 — GPT-2 and diffusion, 30 iterations (shots=100k, dmax=2000)

GPT-2 and capacity-matched absorbing diffusion (T=16, 256-dim) at 30 iterations
with the high-shot settings. Both join the `n2-L10-policy-comparison` group.

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_gpt2_30iter"
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_absorbing_matched_30iter"
```

## N2 L10 — GNN absorbing diffusion, 30 iterations (shots=100k, no dmax cap)

GAT-based denoiser on a chain graph over gate positions. Same forward/reverse
process as the absorbing diffusion model but with a GNN instead of Transformer.
Joins `n2-L10-policy-comparison` so charts overlay with GPT-2 and diffusion runs.

**Requires torch_geometric** (not in the base image — installed inline below).

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "pip install torch_geometric && python3 train.py experiment=n2_l10_gnn_absorbing_30iter"
```

## N2 L10 — single-shot diffusion, 30 iterations (shots=100k, no dmax cap)

Direct comparison to the GPT-2 and absorbing diffusion 30-iter runs.
Identical settings (shots=100k, max_dim=2000, 30 iters) — only the model changes.
Joins `n2-L10-policy-comparison` so charts overlay automatically.

**Without warm-start (train from scratch):**

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_singleshot_30iter"
```

**With warm-start from a trained absorbing checkpoint:**

First find available checkpoints:

```powershell
docker run --rm -v "${workdir}:/workspace" gqe_qsci_cpu find /workspace/outputs -name "*.ckpt"
```

Then run with the checkpoint path:

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_singleshot_30iter trainer.warm_start_checkpoint=/workspace/outputs/gqe-for-qsci/<run-tag>/checkpoints/<epoch>.ckpt"
```

## N2 L10 — single-shot diffusion, dmax=170 constrained (two comparison points)

### 30 iterations, shots=100k — joins `n2-L10-dmax170-comparison`

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_singleshot_30iter_dmax170"
```

With warm-start from a trained absorbing checkpoint:

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_singleshot_30iter_dmax170 trainer.warm_start_checkpoint=/workspace/outputs/gqe-for-qsci/<run-tag>/checkpoints/<epoch>.ckpt"
```

### 100 iterations, shots=1000 — joins `n2-L10-paper-settings`

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_singleshot_paper"
```

With warm-start:

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_singleshot_paper trainer.warm_start_checkpoint=/workspace/outputs/gqe-for-qsci/<run-tag>/checkpoints/<epoch>.ckpt"
```

Find available checkpoints:

```powershell
docker run --rm -v "${workdir}:/workspace" gqe_qsci_cpu find /workspace/outputs -name "*.ckpt"
```

## N2 L10 — dmax=170 constrained, 30 iterations (shots=100k)

Same 30-iteration runs but with the paper's d_max=170 cap. Note: with 100k shots
the subspace is saturated at 170 from iteration 1 (see NOTES.md for explanation).
Both join the `n2-L10-dmax170-comparison` group.

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_gpt2_30iter_dmax170"
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_absorbing_matched_30iter_dmax170"
```

Future NVIDIA GPU run shape:

```powershell
docker run --rm --gpus all --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -v "${workdir}:/workspace" -w /workspace gqe_qsci_gpu -lc "python3 train.py experiment=n2_l10_gpt2_15iter trainer.precision=16-mixed"
```

## N2 L10 — single-shot diffusion, 100 iterations (shots=1000, no dmax cap)

Classical analogue of the USS (Unitary Single-Sampling) architecture from
"Quantum Denoising Diffusion Models" (Kölle et al., 2024, arXiv:2401.07049).
The entire reverse process collapses to one transformer forward pass:
all-[MASK] → clean gate sequence in a single step.

Same architecture as diffusion_absorbing_matched (256-dim, 8 layers, T=16).
Joins `n2-L10-paper-settings` so charts overlay with GPT-2 and absorbing runs.
No max_dim cap (default 2000).

**Option A — train from scratch:**

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_singleshot_100iter"
```

**Option B — warm-start from a trained absorbing checkpoint:**

The single-shot and absorbing models share the same architecture, so weights
transfer directly. Replace the checkpoint path with the actual `.ckpt` file
from a previous absorbing run (found under `outputs/…/checkpoints/`).

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_singleshot_100iter trainer.warm_start_checkpoint=/workspace/outputs/gqe-for-qsci/<run-tag>/checkpoints/<epoch>.ckpt"
```

To find available checkpoints from inside Docker:

```powershell
docker run --rm -v "${workdir}:/workspace" gqe_qsci_cpu find /workspace/outputs -name "*.ckpt"
```

## N2 L10 — paper-reproduction settings (shots=1000, dmax=170, 100 iterations)

Matches Figure 3 of the GQE-QSCI paper as closely as possible:
- `shots=1000` per circuit — low enough that random/early-training circuits do NOT
  saturate d_max, so the subspace dimension grows with training quality (the growing
  curve in Figure 3b).
- `max_dim=170` — paper's hard cap on QSCI subspace size.
- `max_iters=100` — paper's iteration count.

Both runs share the W&B group `n2-L10-paper-settings`.

```powershell
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_gpt2_paper"
docker run --rm --entrypoint /bin/bash -e WANDB_API_KEY=<KEY> -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp -e OMPI_MCA_opal_warn_on_missing_libcuda=0 -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu -lc "python3 train.py experiment=n2_l10_diffusion_paper"
```

To sync an offline W&B run, use the `wandb sync ...` path printed at the end of
the run from inside the same Docker image, or run online with `WANDB_MODE=online`
after logging in.
