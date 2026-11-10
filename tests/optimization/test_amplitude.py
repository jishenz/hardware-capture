"""Pure numerical tests for sigmoid coefficients and their GRAPE pullback."""

import math

import numpy as np
import pytest

from hardware_capture.optimization.amplitude import (
    INITIAL_INTERIOR_FRACTION,
    amplitude_logits,
    sigmoid_amplitude,
)
from hardware_capture.optimization.core import (
    RobustnessConfig,
    adam_update,
    clamped_bspline_design_matrix,
    polar_control_gradients,
    robustness_grape_gradient,
)

pytestmark = pytest.mark.unit


def test_seed_roundtrip_and_explicit_endpoint_initialization():
    maximum = 0.6
    seed = maximum * np.array([0, 0.1, 0.2604166666667, 0.9, 1])
    logits = amplitude_logits(seed, maximum)
    amplitude, _ = sigmoid_amplitude(logits, maximum)
    assert np.all(np.isfinite(logits))
    assert amplitude[1:-1] == pytest.approx(seed[1:-1], abs=1e-15)
    assert amplitude[0] == pytest.approx(maximum * INITIAL_INTERIOR_FRACTION)
    assert amplitude[-1] == pytest.approx(maximum * (1 - INITIAL_INTERIOR_FRACTION))
    for invalid in ([-0.01], [maximum + 0.01], [np.nan], [np.inf], []):
        with pytest.raises(ValueError):
            amplitude_logits(np.array(invalid), maximum)


def test_extreme_logits_are_stable_and_bounded_without_projection():
    with np.errstate(over="raise", invalid="raise"):
        amplitude, derivative = sigmoid_amplitude(
            np.array([-1000, -40, 0, 40, 1000]), 0.7
        )
    assert np.all(np.isfinite(amplitude))
    assert np.all((amplitude >= 0) & (amplitude <= 0.7))
    assert np.all(derivative >= 0)
    assert derivative[2] == pytest.approx(0.7 / 4)
    assert derivative[3] > 0  # Avoid cancellation in 1-sigmoid(40).
    for maximum in (0, -1, 1.01, np.nan, np.inf):
        with pytest.raises(ValueError):
            sigmoid_amplitude(np.array([0.0]), maximum)
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            sigmoid_amplitude(np.array([bad]), 1)


@pytest.mark.parametrize("maximum", [1.0, 0.65])
def test_full_sigmoid_spline_grape_gradient_matches_finite_difference(maximum):
    logits = np.array([-3, -1.2, -0.3, 0.5, 1.7, 3.0])
    spline = clamped_bspline_design_matrix(24, len(logits))
    phase = np.linspace(-0.4, 0.9, 24)
    dt = np.full(24, 0.2e-6)
    objective = RobustnessConfig(
        omega_reference_rad_s=2 * math.pi * 160000,
        amplitude_errors=np.array([-0.1, 0.0, 0.1]),
        detuning_ratios=np.array([-0.05, 0.0, 0.05]),
        mean_weight=1.0,
        threshold_weight=0.0,
        cvar_weight=0.0,
    )

    def evaluate(z):
        coefficients, derivative = sigmoid_amplitude(z, maximum)
        envelope = np.asarray(spline @ coefficients)
        i, q = envelope * np.cos(phase), envelope * np.sin(phase)
        result = robustness_grape_gradient(i, q, dt, objective)
        ga, _ = polar_control_gradients(i, q, result.i_gradient, result.q_gradient)
        return result.objective.loss, np.asarray(spline.T @ ga) * derivative

    _, analytical = evaluate(logits)
    numerical = []
    epsilon = 1e-6
    for index in range(len(logits)):
        perturbation = np.zeros(len(logits))
        perturbation[index] = epsilon
        numerical.append(
            (evaluate(logits + perturbation)[0] - evaluate(logits - perturbation)[0])
            / (2 * epsilon)
        )
    assert analytical == pytest.approx(numerical, rel=2e-5, abs=2e-8)


def test_adam_optimizes_logits_toward_boundary_without_clipping_amplitudes():
    maximum = 0.6
    logits = amplitude_logits(np.array([0.1]), maximum)
    state = None
    losses = []
    for _ in range(100):
        amplitude, derivative = sigmoid_amplitude(logits, maximum)
        losses.append(float((amplitude[0] - 0.8) ** 2))
        logits, state = adam_update(
            logits, 2 * (amplitude - 0.8) * derivative, state, learning_rate=0.2
        )
        assert 0 < amplitude[0] < maximum
    assert losses[-1] < losses[0] / 8
    assert logits[0] > 0
