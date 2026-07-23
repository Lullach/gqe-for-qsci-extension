# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from abc import ABC, abstractmethod
import math


class TemperatureScheduler(ABC):
    """Abstract base class for inverse-temperature scheduling in GQE.

    Naming convention (fixed 2026 — previously "temperature" and "inverse
    temperature" were mixed up across this codebase; see NOTES.md, "BUG
    (fixed): temperature / inverse-temperature naming"):

    Every subclass tracks and returns beta, the INVERSE temperature, never T.
    Sampling always applies it as ``Categorical(logits = -beta * raw_logits)``
    (see e.g. gpt2.py, diffusion.py, dag_gnn.py) — the standard Boltzmann form
    p(a) ~ exp(-beta * E(a)), with the model's raw output acting as a learned
    pseudo-energy (trained, via the policy gradient sign, to be LOW for good
    actions). Consequences of this convention, easy to get backwards:

      - INCREASING beta SHARPENS the policy (more exploitation, less random).
      - DECREASING beta FLATTENS the policy (more exploration, more random).
      - This is the opposite of what "increasing temperature" would do if T
        (not beta) were the tracked quantity — hence the renamed attributes
        below (`current_beta`, not `current_temperature`).
    """

    @abstractmethod
    def get_inverse_temperature(self):
        """Return the current inverse temperature beta (float)."""
        pass

    @abstractmethod
    def update(self, **kwargs):
        """Advance scheduler state by one training iteration.

        Args:
            **kwargs: Optional keyword arguments (e.g., energies, loss, iteration)
        """
        pass


class DefaultScheduler(TemperatureScheduler):
    """Linear ramp: beta increases by a fixed delta every iteration.

    Standard deterministic annealing — the policy monotonically sharpens
    (explores less, exploits more) over the course of training, independent of
    any per-batch signal. This is the current DEFAULT (see
    configs/trainer/default.yaml): it has no dependence on a noisy per-batch
    variance estimate and cannot mis-fire in the wrong direction, unlike
    VarBasedScheduler below.

    Args:
        start: Initial beta
        delta: Amount to increase beta each iteration
    """

    def __init__(self, start, delta) -> None:
        self.start = start
        self.delta = delta
        self.current_beta = start

    def get_inverse_temperature(self):
        return self.current_beta

    def update(self, **kwargs):
        """Increment beta by delta (sharpens the policy over training).

        Args:
            **kwargs: Unused, but accepts any keyword arguments for interface compatibility
        """
        self.current_beta += self.delta


class CosineScheduler(TemperatureScheduler):
    """Cosine-based scheduler: beta oscillates between min and max.

    Useful for periodic exploration/exploitation cycles (warm restarts) during
    training.

    Args:
        minimum: Minimum beta
        maximum: Maximum beta
        frequency: Number of iterations for one complete cycle
    """

    def __init__(self, minimum, maximum, frequency) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.frequency = frequency
        self.current_iter = 0
        self.current_beta = (maximum + minimum) / 2

    def get_inverse_temperature(self):
        return self.current_beta

    def update(self, **kwargs):
        """Advance beta along the cosine cycle.

        Args:
            **kwargs: Unused, but accepts any keyword arguments for interface compatibility
        """
        self.current_iter += 1
        self.current_beta = (self.maximum + self.minimum) / 2 - (
            self.maximum - self.minimum) / 2 * math.cos(
                2 * math.pi * self.current_iter / self.frequency)


class VarBasedScheduler(TemperatureScheduler):
    """Variance-based adaptive beta scheduler. NOT the default — see caveat.

    Adjusts beta based on the variance of energies in the current rollout batch.

    Args:
        initial: Initial beta
        delta: Amount to adjust beta each iteration
        target_var: Variance threshold that decides the adjustment direction.
            Must be on the scale of the actual per-batch energy variance for
            this molecule, or one branch never fires and this degenerates into
            DefaultScheduler. See configs/trainer/default.yaml for how to
            calibrate it (target_var ~ std^2 of GQE-optimized/energy/std).

    CAVEAT — direction is not settled science, read before using:
    this implementation SHARPENS (beta += delta) when the batch variance is
    HIGH, and FLATTENS (beta -= delta) when it is LOW:

        var > target_var:  current_beta += delta   (sharpen)
        var < target_var:  current_beta -= delta   (flatten)

    That is "sharpen when the signal looks noisy, relax once it looks stable".
    An earlier version of this docstring claimed the opposite ("high variance
    -> more exploration") — a leftover of the same temperature/inverse-
    temperature confusion this whole module was renamed to fix — which never
    matched the code. Whether "sharpen on noise" or "reheat on collapse
    (opposite direction)" is the better policy for this problem is an open
    design question (see NOTES.md, "BUG (fixed): VarBasedScheduler..."), not
    resolved here. Only 10 samples/batch back the variance estimate, so treat
    either direction as a rough heuristic, not a principled signal.
    """

    def __init__(self, initial, delta, target_var) -> None:
        self.delta = delta
        self.current_beta = initial
        self.target_var = target_var

    def get_inverse_temperature(self):
        return self.current_beta

    def update(self, **kwargs):
        """Adjust beta based on energy variance (see class docstring for direction).

        Args:
            **kwargs: Must contain 'energies' key with a tensor of energy values
        """
        energies = kwargs["energies"]
        current_var = energies.var().item()

        if current_var > self.target_var:
            self.current_beta += self.delta  # sharpen
        else:
            self.current_beta -= self.delta  # flatten

        self.current_beta = max(self.current_beta, 0.01)  # keep positive