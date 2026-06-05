# Generative Circuit Design for Quantum-Selected Configuration Interaction

Training pipeline for **GQE (Generative Quantum Eigensolver) + QSCI (Quantum-Selected Configuration Interaction)** on molecular Hamiltonians.  
The entry point is `train.py` (Hydra-based), and quantum-circuit sampling is performed with **CUDA-Q**.

For full details, see the paper: *Generative Circuit Design for Quantum-Selected Configuration Interaction*.

![Workflow](figs/workflow.jpg)

---

## Table of Contents

1. [Overview](#overview)
2. [Methods and Models](#methods-and-models)
   - [Policy Models](#policy-models)
   - [Operator Pool](#operator-pool)
   - [Policy Optimization: GRPO and GSPO](#policy-optimization-grpo-and-gspo)
   - [Temperature Scheduling](#temperature-scheduling)
   - [QSCI: Quantum-Selected Configuration Interaction](#qsci-quantum-selected-configuration-interaction)
   - [GEVP Refinement](#gevp-refinement)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Running the Code](#running-the-code)
7. [Configuration](#configuration)
8. [Outputs and Resuming](#outputs-and-resuming)
9. [Upstream Reference and Attribution](#upstream-reference-and-attribution)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)

---

## Overview

This project trains a generative model to **design quantum circuits** that prepare low-energy electronic states of molecules. The workflow is:

1. A **policy model** (GPT-2, discrete diffusion, or GNN) generates gate-index sequences selecting operators from a chemistry-informed pool.
2. The resulting quantum circuit is simulated with **CUDA-Q**, producing a probability distribution over computational basis states (bitstrings).
3. The most probable bitstrings are interpreted as Slater determinants, and a **QSCI diagonalization** refines the energy estimate by building and diagonalizing the molecular Hamiltonian in the sampled subspace.
4. The energy serves as a reward signal: a **policy gradient loss** (GRPO or GSPO) updates the model to generate circuits that yield lower energies.

The feedback loop connects classical deep learning with quantum chemistry simulation, iteratively improving both the circuits and the energy estimates.

---

## Methods and Models

### Policy Models

Four policy architectures are available, all implementing the same `Policy` interface (`gqe_qsci/gqe/models/policy.py`). GPT-2 generates circuits autoregressively; the diffusion and GNN models generate full sequences non-autoregressively via `sample_sequence()`.

#### GPT-2 Policy Network

**Theory.** The Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) processes sequences via stacked self-attention and feed-forward blocks. GPT-2 is a decoder-only variant that performs **autoregressive generation**: at each step, it attends to all previous tokens and produces a probability distribution over the next token. This makes it well-suited for sequential decision-making, where each "token" is a gate selected from a discrete operator vocabulary.

**Implementation.** The policy (`gqe_qsci/gqe/models/gpt2.py`) wraps Hugging Face's `GPT2LMHeadModel`:

- **Architecture**: 6 transformer layers, 6 attention heads, hidden dimension 384 (the "small" GPT-2 variant). The vocabulary size equals the number of operators in the pool, determined at runtime from the molecular Hamiltonian.
- **Repetition penalty**: A multiplicative penalty (default 1.2) is applied to the logits of already-selected operators at each decoding step, discouraging the model from repeating gates and encouraging diverse circuits.
- **KV caching**: Incremental decoding uses key/value caches for efficient autoregressive generation.
- **`act(state, temperature)`**: Generates a complete gate-index sequence of length `ngates` by sampling from the softmax distribution with the given temperature.
- **`log_prob(indices, temperature)`**: Computes per-step log-probabilities for a batch of existing sequences, used during the policy gradient update.

Select with `model=gpt2`.

---

#### Discrete Diffusion Policy Networks

All diffusion models use a **Transformer Encoder** (bidirectional, non-causal) as the denoiser, making them naturally suited for predicting masked positions from full context. Three variants are available:

##### CircuitDiffusionModelSimple (`model=diffusion`)

The original simplified model, kept for comparison and backward compatibility. Sampling starts from uniformly random tokens and replaces all positions at every denoising step — no principled forward process. `log_prob` is a simplified proxy evaluated at an all-zero context. **Not recommended for new experiments.**

##### CircuitDiffusionModelAbsorbing (`model=diffusion_absorbing`)

**Theory.** Implements absorbing diffusion ([D3PM, Austin et al., 2021](https://arxiv.org/abs/2107.03006) / MDLM-style). The forward process independently masks each gate with probability $(1 - \alpha_t)$ under a noise schedule:

$$q(x_t | x_0): \text{each token masked with prob } (1 - \alpha_t), \quad \alpha_0 = 1,\ \alpha_T = 0$$

The reverse process uses the exact closed-form posterior $q(x_{t-1} | x_t, \hat{x}_0)$, revealing masked positions progressively from $t = T$ down to $t = 1$.

**log_prob.** Estimated via the denoising ELBO averaged over all $T$ timesteps. Corruption masks are **pre-sampled** once during rollout collection (`sample_masks()`) and stored in the replay buffer, making the GRPO importance-weight ratio $\exp(\log p_\text{new} - \log p_\text{old})$ deterministic and stable across policy updates.

**Noise schedule.** Configurable as `cosine` (recommended) or `linear` via `noise_schedule`.

**Architecture.** Transformer Encoder, default hidden_size=256, 8 layers, 8 heads, $T=16$ steps (see `configs/model/diffusion_absorbing_matched.yaml`).

##### CircuitDiffusionModelSingleShot (`model=diffusion_singleshot`)

**Theory.** Classical analogue of the USS (Unitary Single-Sampling) architecture from *Quantum Denoising Diffusion Models* ([Kölle et al., 2024](https://arxiv.org/abs/2401.07049)). The entire reverse process collapses into **one transformer forward pass**: the model predicts all gate positions simultaneously from the fully-masked sequence at $t = T$, without iterative unmasking.

**log_prob.** Exact (no ELBO averaging), fully deterministic — the best possible importance-weight stability for GRPO.

**Trade-offs.**

| | Absorbing (T=16) | Single-Shot |
|---|---|---|
| Inference | T forward passes | 1 forward pass (T× faster) |
| log_prob | ELBO approximation | Exact |
| Task difficulty | Easier (step-by-step) | Harder (direct mapping) |

Weights are architecturally compatible with the absorbing model; warm-starting from a trained absorbing checkpoint is supported via `trainer.warm_start_checkpoint`.

---

#### GNN Policy Network (`model=gnn_absorbing`)

**Theory.** Replaces the Transformer Encoder denoiser with **Graph Attention Network** ([GAT, Veličković et al., 2018](https://arxiv.org/abs/1710.10903)) layers. Gate positions are nodes; edges encode positional structure via a configurable graph topology.

**Motivation.** A GNN allows domain knowledge to be encoded directly into the graph structure, rather than relying solely on positional embeddings. Two graph types are supported:

| Graph type | Edges | Character |
|---|---|---|
| `chain` (default) | bidirectional $i \leftrightarrow i+1$ | Encodes gate ordering; 6 layers covers the full L=10 sequence |
| `full` | all pairs $i \to j$, $i \neq j$ | Equivalent to Transformer attention without softmax; ablation baseline |

**Implementation** (`gqe_qsci/gqe/models/gnn.py`). The edge index is computed once in `__init__` and registered as a buffer (no per-forward-pass overhead). GAT layers use `concat=False` (heads averaged, not concatenated) with residual connections and per-layer LayerNorm to prevent over-smoothing in deep networks. The diffusion logic (forward process, reverse process, ELBO log_prob, mask pre-sampling) is identical to `CircuitDiffusionModelAbsorbing`.

**Requirement.** Needs `torch_geometric` (`pip install torch_geometric`).

---

### Operator Pool

**Theory.** UCCSD (Unitary Coupled Cluster with Singles and Doubles) is a standard quantum chemistry ansatz that parameterizes the wavefunction as a product of Pauli evolution operators:

$$|\psi\rangle = \prod_k e^{i\theta_k \hat{P}_k} |\phi_0\rangle$$

where each $\hat{P}_k$ is a product of Pauli matrices (X, Y, Z, I) on different qubits, and $|\phi_0\rangle$ is the Hartree-Fock reference state. The operator amplitudes $\theta_k$ are taken from a classical CCSD calculation, providing a physics-informed initialization.

**Implementation.** Two pool variants are available (`gqe_qsci/gqe/operator_pool.py`):

| Pool | Description |
|------|-------------|
| `pauli_evolution` | Each pool element is a single Pauli product $e^{i\theta_k \hat{P}_k}$. Larger vocabulary, finer-grained control. |
| `excitation` | Each pool element sums the Pauli products corresponding to one full fermionic excitation. Smaller vocabulary, coarser control. |

Both pools use CCSD amplitudes to define operator coefficients (filtered by `ccsd_threshold`). Options `remove_z_ladder` and `only_use_first_pauli` further control the pool size.

### Policy Optimization: GRPO and GSPO

**Theory.** Standard REINFORCE-style policy gradient updates the policy by:

$$\nabla_\theta \mathcal{L} = -\mathbb{E}\left[A \cdot \nabla_\theta \log \pi_\theta(a|s)\right]$$

where $A$ is an advantage estimate. GRPO (Group Relative Policy Optimization, [Shao et al., 2024](https://arxiv.org/abs/2402.03300)) computes advantages relative to a group of samples generated from the same prompt, eliminating the need for a separate value network. A PPO-style clipped objective prevents large policy updates:

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(r_t A_t,\; \text{clip}(r_t, 1-\epsilon_l, 1+\epsilon_h) A_t\right)\right]$$

where $r_t = \pi_\theta / \pi_{\theta_\text{old}}$ is the importance ratio and advantages are energy-based: lower energy → higher advantage.

An **entropy regularization** bonus (coefficient `entropy_coeff`, default 0.01) is subtracted from the loss to encourage sequence diversity.

**Implementation.** Both losses are in `gqe_qsci/gqe/loss.py`:

| Loss | Formula detail |
|------|---------------|
| `GRPOLoss` | Advantage computed from raw energies: $A = (E_\text{mean} - E) / (E_\text{std} + \epsilon)$ |
| `GSPOLoss` | Same as GRPO but log-probabilities normalized by sequence length, improving stability for variable-length circuits |

Default clipping range: `clip_grpo_low=0.2`, `clip_grpo_high=0.28` (asymmetric to allow larger improvements than penalties).

### Temperature Scheduling

**Theory.** Sampling temperature $T$ controls exploration vs. exploitation in the policy: high $T$ flattens the distribution (explore), low $T$ sharpens it (exploit). Adaptive temperature scheduling adjusts $T$ based on the diversity of sampled energies.

**Implementation.** Three schedulers are available (`gqe_qsci/gqe/scheduler.py`):

| Scheduler | Behavior |
|-----------|----------|
| `DefaultScheduler` | Linear increment: $T \mathrel{+}= \delta$ per epoch |
| `CosineScheduler` | Oscillates between $T_\text{min}$ and $T_\text{max}$ (cosine annealing with warm restarts) |
| `VarBasedScheduler` *(default)* | Increases $T$ if energy variance exceeds target, decreases otherwise. Keeps the policy in a productive exploration regime |

The current inverse temperature is logged each epoch as `trainer/inv_temperature`.

### QSCI: Quantum-Selected Configuration Interaction

**Theory.** Configuration Interaction (CI) methods diagonalize the many-body Hamiltonian in a subspace of Slater determinants. QSCI ([Kanno et al., 2023](https://arxiv.org/abs/2302.11320)) selects this subspace using a quantum device: the most probable measurement outcomes of a parametrized quantum state define the set of determinants. This leverages the quantum state's ability to concentrate amplitude on chemically relevant configurations, reducing the CI subspace size compared to conventional selected-CI methods.

**Implementation.** The QSCI pipeline (`gqe_qsci/qsci/pipeline.py`) proceeds as follows:

1. **Quantum sampling**: CUDA-Q simulates the circuit and returns bitstring counts over `shots` measurements.
2. **Determinant extraction**: Bitstrings are parsed into alpha/beta electron occupation strings (Slater determinants).
3. **Post-selection**: Determinants that do not conserve electron number are discarded.
4. **Subspace enlargement**: Symmetry-related determinants are added to restore spin symmetry, up to `max_dim` total determinants.
5. **Hamiltonian diagonalization**: The second-quantized Hamiltonian (built via PySCF + Jordan-Wigner) is projected onto the subspace and diagonalized using PyCI.
6. **Local refinement**: CI vectors from the current batch of samples are merged and re-diagonalized to improve the energy estimate within the epoch.
7. **Global refinement**: The best CI vectors from all previous epochs are accumulated and re-diagonalized via GEVP, tracking the best energy found so far.

### GEVP Refinement

**Theory.** When CI vectors from multiple batches are merged, the union basis is non-orthogonal. The refined energy is obtained by solving a **Generalized Eigenvalue Problem** (GEVP):

$$H \mathbf{c} = E\, S \mathbf{c}$$

where $H$ is the Hamiltonian matrix projected onto the union basis and $S$ is the overlap matrix. This is more rigorous than re-diagonalizing in the full union, as it properly accounts for linear dependencies.

**Implementation.** `gqe_qsci/qsci/refine/gevp.py` constructs $H$ and $S$ from stored CI vectors, solves the GEVP via `scipy.linalg.eigh`, and selects the top determinants by squared coefficient weight $|c_i|^2$ for further refinement iterations.

---

## Repository Structure

```
.
├── train.py                         # Hydra entry point
├── pyproject.toml                   # Dependencies
├── dockerfile                       # Docker build (based on ghcr.io/nvidia/cudaqx:0.4.0)
├── configs/
│   ├── default.yaml                 # Top-level config (ngates, qsci, sampler, …)
│   ├── model/
│   │   ├── gpt2.yaml                # GPT-2 (autoregressive, 6L-6H-384d)
│   │   ├── diffusion.yaml           # Simple diffusion (legacy, not recommended)
│   │   ├── diffusion_absorbing.yaml         # Absorbing diffusion (small)
│   │   ├── diffusion_absorbing_matched.yaml # Absorbing diffusion (matched to GPT-2 capacity)
│   │   ├── diffusion_singleshot.yaml        # Single-shot diffusion
│   │   └── gnn_absorbing.yaml       # GNN absorbing diffusion (requires torch_geometric)
│   ├── molecule/
│   │   ├── h2.yaml
│   │   ├── n2.yaml                  # N2, STO-3G, 10e/8o active space
│   │   └── phenylene-1_4-dinitrene.yaml
│   ├── trainer/
│   │   └── default.yaml             # Optimizer, loss, scheduler, batch sizes
│   └── experiment/                  # Pre-configured experiment overrides
│       ├── n2_l10_gpt2_paper.yaml
│       ├── n2_l10_diffusion_paper.yaml
│       ├── n2_l10_diffusion_absorbing_matched_30iter.yaml
│       ├── n2_l10_diffusion_singleshot_30iter.yaml
│       ├── n2_l10_gnn_absorbing_30iter.yaml
│       └── …
├── gqe_qsci/
│   ├── train_pipeline.py            # PyTorch Lightning TrainPipeline
│   ├── factory.py                   # Component instantiation
│   ├── molecule.py                  # PySCFMolecule (Hamiltonian construction)
│   ├── wandb_logger.py              # W&B logging utilities
│   ├── gqe/
│   │   ├── models/
│   │   │   ├── policy.py            # Abstract Policy base class
│   │   │   ├── gpt2.py              # GPT-2 autoregressive policy
│   │   │   ├── diffusion.py         # Diffusion policies (Simple, Absorbing, SingleShot)
│   │   │   └── gnn.py               # GNN absorbing diffusion policy
│   │   ├── buffer.py               # ReplayBuffer & BufferDataset
│   │   ├── loss.py                 # GRPOLoss, GSPOLoss
│   │   ├── operator_pool.py        # PauliEvolutionPool, ExcitationPool
│   │   ├── sampler.py              # CUDA-Q quantum sampler
│   │   ├── scheduler.py            # Temperature schedulers
│   │   └── utils.py                # Pauli string utilities
│   └── qsci/
│       ├── pipeline.py             # QSCIPipeline (sample → diagonalize → refine)
│       ├── schema.py               # Result dataclasses
│       ├── subspace.py             # DeterminantSubspace
│       ├── determinant.py          # Bitstring determinant representation
│       ├── statevector.py          # SCIVector (CI wavefunction)
│       └── refine/
│           ├── pipeline.py         # Refinement subspace construction
│           └── gevp.py             # GEVP solver
└── figs/
    └── workflow.jpg
```

---

## Requirements

- Python `>= 3.10`
- NVIDIA GPU with CUDA 12.8 (for GPU-accelerated simulation)
- Docker is the recommended setup (see below)

Key dependencies (see `pyproject.toml` for exact versions):

| Package | Role |
|---------|------|
| `cuda-quantum` | Quantum circuit simulation |
| `pyscf` | Molecular Hamiltonian construction |
| `torch` + `lightning` | Neural network training |
| `transformers` | GPT-2 model |
| `hydra-core` | Configuration management |
| `tequila-basic` | UCCSD ansatz / operator pool |
| `pyci` | CI Hamiltonian diagonalization |
| `mpi4py` | MPI parallelization across QPUs |
| `wandb` | Experiment tracking |
| `torch_geometric` | GNN models only (optional) |

---

## Installation

The `Dockerfile` is based on `ghcr.io/nvidia/cudaqx:0.4.0`.

### Build

CPU-only:

```bash
docker build -t gqe_qsci .
```

GPU (CUDA 12.8):

```bash
docker build \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  -t gqe_qsci .
```

---

## Running the Code

All commands below assume execution inside the Docker container (or a matching local environment). The entry point is `train.py`, configured via [Hydra](https://hydra.cc/).

### Basic training (N2 molecule, GPT-2)

```bash
python train.py molecule=n2
```

### Train with absorbing diffusion

```bash
python train.py molecule=n2 model=diffusion_absorbing_matched
```

### Train with single-shot diffusion

```bash
python train.py molecule=n2 model=diffusion_singleshot
```

### Train with GNN absorbing diffusion

```bash
pip install torch_geometric
python train.py molecule=n2 model=gnn_absorbing
```

### Reproduce paper experiments

```bash
# GPT-2 paper run
python train.py experiment=n2_l10_gpt2_paper

# Diffusion paper run
python train.py experiment=n2_l10_diffusion_paper
```

### Override training duration and circuit length

```bash
python train.py molecule=n2 trainer.max_iters=200 ngates=15
```

### Pin an experiment tag (enables easy resuming)

```bash
python train.py molecule=n2 exp_tag=my-n2-run
```

### Start fresh (ignore existing checkpoints)

```bash
python train.py molecule=n2 trainer.load_checkpoint=false
```

### Warm-start a single-shot model from an absorbing checkpoint

```bash
python train.py molecule=n2 model=diffusion_singleshot \
  trainer.warm_start_checkpoint=outputs/my-project/my-absorbing-run/models/last.ckpt
```

### Run with a different operator pool

```bash
python train.py molecule=n2 operator_pool.spec=excitation
```

### Run with GSPO loss instead of GRPO

```bash
python train.py molecule=n2 trainer.loss.type=gspo
```

### Run with MPI (multi-GPU/QPU sampling)

```bash
mpirun -n 4 python train.py molecule=n2
```

### Docker: run with GPU

```bash
docker run -it --rm --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  gqe_qsci \
  python train.py molecule=n2
```

### Docker: run CPU-only

```bash
docker run -it --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  gqe_qsci \
  python train.py molecule=n2
```

---

## Configuration

Hydra configs live under `configs/`. Key parameters:

| Config key | Default | Description |
|------------|---------|-------------|
| `molecule` | `n2` | Molecule config group (`n2`, `h2`, `phenylene-1_4-dinitrene`) |
| `model` | `gpt2` | Policy model (`gpt2`, `diffusion_absorbing_matched`, `diffusion_singleshot`, `gnn_absorbing`, …) |
| `ngates` | `10` | Number of gates in each generated circuit |
| `operator_pool.spec` | `pauli_evolution` | Pool type: `pauli_evolution` or `excitation` |
| `operator_pool.ccsd_threshold` | `1e-6` | Minimum CCSD amplitude to include an operator |
| `qsci.max_dim` | `2000` | Maximum QSCI subspace dimension |
| `sampler.shots` | `100000` | Quantum circuit measurement shots |
| `trainer.max_iters` | `15` | Total training epochs |
| `trainer.num_samples` | `10` | Rollout circuits per epoch |
| `trainer.step_per_epoch` | `30` | Gradient update steps per epoch |
| `trainer.warmup_size` | `10` | Buffer size before training starts |
| `trainer.loss.type` | `grpo` | Loss function: `grpo` or `gspo` |
| `trainer.entropy_coeff` | `0.01` | Entropy regularization coefficient (0 to disable) |
| `trainer.warm_start_checkpoint` | `null` | Path to a `.ckpt` to load model weights from before training |
| `model.repetition_penalty` | `1.2` | GPT-2 only: repetition penalty on already-selected gates |
| `model.noise_schedule` | `cosine` | Diffusion models: `cosine` or `linear` |
| `model.diffusion_steps` | varies | Diffusion models: number of denoising steps T |
| `scheduler.target_var` | `1e-5` | Energy variance target for `VarBasedScheduler` |
| `exp_tag` | auto | Experiment tag (determines output directory) |

Pre-configured experiment files in `configs/experiment/` bundle molecule, model, and trainer settings for reproducible runs.

---

## Outputs and Resuming

Outputs are written to `outputs/${project.name}/${exp_tag}/`:

| File | Contents |
|------|---------|
| `models/last.ckpt` | PyTorch Lightning checkpoint |
| `buffer.pkl` | Replay buffer (trajectories from all epochs) |
| `run_id` | W&B run ID (used to resume the same W&B run) |
| `.hydra/config.yaml` | Full resolved Hydra configuration |

**Tracked W&B metrics:**

- `trainer/loss` — policy gradient loss
- `trainer/inv_temperature` — current inverse temperature
- `GQE-optimized/energy/*` — best energy from the current epoch
- `GQE-optimized(best_so_far)/energy/*` — best energy across all epochs
- `Local-refined(best_so_far)/energy/*` — best local-refinement energy
- `Global-refined(best_so_far)/energy/*` — best global (GEVP) refinement energy

**Resuming**: re-run with the same `exp_tag` and `trainer.load_checkpoint=true` (default). The model checkpoint, replay buffer, W&B run, and temperature scheduler state are all restored automatically.

---

## Upstream Reference and Attribution

This implementation is based in part on the CUDA-QX contribution proposed in
[NVIDIA/cudaqx PR #373](https://github.com/NVIDIA/cudaqx/pull/373).

In particular, the present codebase was informed by the upstream work on:

- GRPO-based policy training
- Replay-buffer-based training flow
- Variance-based temperature scheduling
- A modular training pipeline built on PyTorch Lightning

We gratefully acknowledge **NVIDIA** and the **CUDA-QX** contributors for making that work publicly available.

---

## License

This repository is distributed under the **Apache License 2.0**.

Because this project was developed with reference to, and may include derivative work from, **NVIDIA/cudaqx**, the repository keeps the corresponding license and attribution information in the top-level `LICENSE` and `NOTICE` files. If you redistribute or modify this code, please preserve those notices and clearly indicate your changes in modified files.

---

## Acknowledgments

A part of this work was performed for the Council for Science, Technology and Innovation (CSTI), Cross-ministerial Strategic Innovation Promotion Program (SIP), "Promoting the application of advanced quantum technology platforms to social issues" (funding agency: QST). The results presented in this work were obtained using the ABCI-Q of AIST G-QuAT.
RK would like to express gratitude to Kenji Sugisaki for insightful discussion.
