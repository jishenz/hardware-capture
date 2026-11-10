"""Total-loss SPSA with deterministic, paired command perturbations.

Numerical code only: the caller supplies a scalar-loss measurement callback.
See Spall (1998), https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_An_Overview.PDF.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SPSAConfig:
    directions: int = 1
    seed: int = 20260820
    amplitude_logit_perturbation: float = 0.08
    phase_perturbation_rad: float = 0.05
    perturbation_decay: float = 0.101

    def validate(self) -> None:
        for name, minimum in (("directions", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"spsa.{name} must be an integer >= {minimum}")
        for name in ("amplitude_logit_perturbation", "phase_perturbation_rad"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"spsa.{name} must be finite and positive")
        if (
            isinstance(self.perturbation_decay, bool)
            or not isinstance(self.perturbation_decay, (float, int))
            or not np.isfinite(self.perturbation_decay)
            or not 0 <= self.perturbation_decay <= 1
        ):
            raise ValueError("spsa.perturbation_decay must be in [0, 1]")

    def steps(self, count: int, epoch: int) -> np.ndarray:
        """Logit then phase scales; epoch is the zero-based update index."""
        self.validate()
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("SPSA epoch must be a nonnegative integer")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("SPSA coefficient count must be a positive integer")
        return np.repeat(
            [self.amplitude_logit_perturbation, self.phase_perturbation_rad], count
        ) * ((epoch + 1.0) ** -self.perturbation_decay)


@dataclass(frozen=True)
class SPSAEstimate:
    gradient: np.ndarray
    directions: np.ndarray
    steps: np.ndarray
    plus_parameters: np.ndarray
    minus_parameters: np.ndarray
    plus_losses: np.ndarray
    minus_losses: np.ndarray
    capture_order: np.ndarray


def spsa_gradient_estimate(loss_plus, loss_minus, direction, steps) -> np.ndarray:
    """Estimate dL/d(parameters) from the complete scalar losses of one pair."""
    for loss in (loss_plus, loss_minus):
        if np.ndim(loss) != 0 or not np.isrealobj(loss) or not np.isfinite(loss):
            raise ValueError("SPSA losses must be finite real scalars")
    direction = np.asarray(direction, dtype=float)
    steps = np.asarray(steps, dtype=float)
    if (
        direction.ndim != 1
        or not direction.size
        or not np.all(np.isin(direction, [-1, 1]))
    ):
        raise ValueError("SPSA direction must be a nonempty vector of -1 or +1")
    if (
        steps.shape != direction.shape
        or not np.all(np.isfinite(steps))
        or np.any(steps <= 0)
    ):
        raise ValueError("SPSA steps must be a matching finite positive vector")
    gradient = (float(loss_plus) - float(loss_minus)) / (2 * steps * direction)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("SPSA gradient is not finite")
    return gradient


def estimate_spsa(
    parameters: np.ndarray,
    measure_loss: Callable[[np.ndarray, int, str], float],
    *,
    config: SPSAConfig,
    epoch: int,
) -> SPSAEstimate:
    """Measure complete +/- pairs about one center, then average gradients.

    The vector contains amplitude logits followed by phase coefficients. Each
    callback decodes its own command and evaluates the full measured loss. No
    GRAPE, spline, sigmoid, or hardware Jacobian is multiplied into this estimate.
    Independent directions are reconstructed from seed/update/direction indices.
    Capture order alternates to avoid a fixed association of sign with time.
    Failure of any probe propagates before returning an updateable gradient.
    """
    parameters = np.asarray(parameters, dtype=float)
    if (
        parameters.ndim != 1
        or not parameters.size
        or parameters.size % 2
        or not np.all(np.isfinite(parameters))
    ):
        raise ValueError("SPSA requires finite equal-length logit and phase vectors")
    steps = config.steps(parameters.size // 2, epoch)
    signs, plus, minus, losses_plus, losses_minus, orders, gradients = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index in range(config.directions):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, epoch, index]))
        delta = rng.choice([-1.0, 1.0], size=parameters.size)
        positive = parameters + steps * delta
        negative = parameters - steps * delta
        separation = positive - negative
        if (
            not np.all(np.isfinite(positive))
            or not np.all(np.isfinite(negative))
            or not np.all(np.isfinite(separation))
            or not np.allclose(separation, 2 * steps * delta, rtol=1e-6, atol=0)
        ):
            raise ValueError(
                "SPSA probe separation cannot represent the requested steps"
            )
        order = ("plus", "minus") if (epoch + index) % 2 == 0 else ("minus", "plus")
        probes = {"plus": positive, "minus": negative}
        losses = {}
        for label in order:
            loss = measure_loss(probes[label].copy(), index, label)
            if np.ndim(loss) != 0 or not np.isrealobj(loss) or not np.isfinite(loss):
                raise ValueError(f"SPSA {label} loss must be a finite real scalar")
            losses[label] = float(loss)
        gradients.append(
            spsa_gradient_estimate(losses["plus"], losses["minus"], delta, steps)
        )
        signs.append(delta)
        plus.append(positive)
        minus.append(negative)
        losses_plus.append(losses["plus"])
        losses_minus.append(losses["minus"])
        orders.append("_then_".join(order))
    gradient = np.mean(gradients, axis=0)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("SPSA mean gradient is not finite")
    return SPSAEstimate(
        gradient,
        np.asarray(signs),
        steps,
        np.asarray(plus),
        np.asarray(minus),
        np.asarray(losses_plus),
        np.asarray(losses_minus),
        np.asarray(orders),
    )
