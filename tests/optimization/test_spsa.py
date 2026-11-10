"""Pure numerical full-loss SPSA checks; no hardware operations."""

from itertools import product

import numpy as np
import pytest
from scipy.special import expit

from hardware_capture.optimization.spsa import (
    SPSAConfig,
    estimate_spsa,
    spsa_gradient_estimate,
)

pytestmark = pytest.mark.unit


def test_full_nonlinear_loss_includes_additional_penalty_and_sigmoid_once():
    parameters = np.array([-0.7, 0.4, 0.2, -0.5])
    mixing = np.array([[1.3, 0.4j], [-0.2, 0.8 - 0.1j]])

    def loss(p):
        amplitude = 0.7 * expit(p[:2])
        measured = mixing @ (amplitude**2 * np.exp(1j * p[2:]))
        # A coupled nonlinear response and an additional command penalty.
        return float(np.sum(abs(measured - (0.2 + 0.1j)) ** 2) + 3 * np.sum(p[2:] ** 2))

    steps = np.array([1e-5, 2e-5, 1e-5, 2e-5])
    gradients = []
    for signs in product([-1.0, 1.0], repeat=4):
        direction = np.asarray(signs)
        gradients.append(
            spsa_gradient_estimate(
                loss(parameters + steps * direction),
                loss(parameters - steps * direction),
                direction,
                steps,
            )
        )
    expected = []
    for index in range(4):
        step = np.eye(4)[index] * 1e-6
        expected.append((loss(parameters + step) - loss(parameters - step)) / 2e-6)
    assert np.mean(gradients, axis=0) == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_reproducible_pairs_order_decay_and_mean_gradient():
    config = SPSAConfig(directions=3, seed=17)
    parameters = np.array([0.3, -0.7, 0.2, 0.5])
    calls = []

    def measure(probe, index, label):
        calls.append((index, label, probe.copy()))
        result = np.dot(probe, probe)
        probe[:] = 0  # A callback cannot mutate the center or saved probes.
        return result

    result = estimate_spsa(parameters, measure, config=config, epoch=0)
    assert [(index, label) for index, label, _ in calls] == [
        (0, "plus"),
        (0, "minus"),
        (1, "minus"),
        (1, "plus"),
        (2, "plus"),
        (2, "minus"),
    ]
    assert (result.plus_parameters + result.minus_parameters) / 2 == pytest.approx(
        np.tile(parameters, (3, 1))
    )
    expected = np.mean(
        (result.plus_losses - result.minus_losses)[:, None]
        / (2 * result.steps * result.directions),
        axis=0,
    )
    assert result.gradient == pytest.approx(expected)
    again = estimate_spsa(parameters, measure, config=config, epoch=0)
    np.testing.assert_array_equal(result.gradient, again.gradient)
    later = estimate_spsa(parameters, measure, config=config, epoch=1)
    assert later.capture_order[0] == "minus_then_plus"
    assert later.steps == pytest.approx(result.steps / 2**config.perturbation_decay)
    assert not np.array_equal(later.directions, result.directions)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j, np.ones(2)])
def test_invalid_loss_aborts_without_another_probe(bad):
    calls = []

    def measure(p, index, label):
        calls.append(label)
        return bad

    with pytest.raises(ValueError, match="loss"):
        estimate_spsa(np.ones(4), measure, config=SPSAConfig(directions=3), epoch=0)
    assert calls == ["plus"]


def test_incomplete_pair_propagates_failure():
    def measure(p, index, label):
        if label == "minus":
            raise RuntimeError("missing record")
        return 0.1

    with pytest.raises(RuntimeError, match="missing record"):
        estimate_spsa(np.zeros(4), measure, config=SPSAConfig(), epoch=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"directions": 0},
        {"directions": True},
        {"directions": 1.5},
        {"seed": -1},
        {"seed": True},
        {"amplitude_logit_perturbation": 0},
        {"amplitude_logit_perturbation": -1},
        {"phase_perturbation_rad": float("inf")},
        {"perturbation_decay": -0.1},
        {"perturbation_decay": 1.1},
        {"phase_perturbation_rad": True},
    ],
)
def test_invalid_spsa_config(settings):
    with pytest.raises(ValueError, match="spsa"):
        SPSAConfig(**settings).validate()


def test_steps_use_explicit_logit_and_phase_units():
    config = SPSAConfig(perturbation_decay=0.5)
    assert config.steps(2, 3) == pytest.approx([0.04, 0.04, 0.025, 0.025])


def test_unrepresentable_step_fails_before_measurement():
    with pytest.raises(ValueError, match="separation"):
        estimate_spsa(
            np.full(2, 1e30),
            lambda *_: pytest.fail("Measured"),
            config=SPSAConfig(),
            epoch=0,
        )


@pytest.mark.parametrize(
    "direction,steps", [([0, 1], [0.1, 0.1]), ([1, -1], [0.1, 0]), ([1, -1], [0.1])]
)
def test_invalid_direction_or_scales_rejected(direction, steps):
    with pytest.raises(ValueError):
        spsa_gradient_estimate(0.2, 0.1, direction, steps)
