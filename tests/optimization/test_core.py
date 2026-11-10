"""No-hardware tests for the closed-loop numerical core."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hardware_capture.optimization.core import (
    RobustnessConfig,
    apply_phase_reference,
    bb1_initialization,
    clamped_bspline_design_matrix,
    command_tables,
    estimate_phase_reference,
    polar_control_gradients,
    robustness_grape_gradient,
    robustness_objective,
    x_pi_target,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "entries,scenarios", [(1, 3), (7, 3), (256, 3), (257, 515), (6000, 3)]
)
def test_batched_propagation_preserves_order_and_block_boundaries(entries, scenarios):
    from hardware_capture.optimization.core import (
        _forward_unitary,
        _step_and_control_derivatives,
    )

    rng = np.random.default_rng(63)
    i, q = rng.uniform(-0.8, 0.8, (2, entries))
    durations = rng.uniform(1e-9, 2e-6, entries)
    scale = np.linspace(0, 1.5, scenarios)
    detuning = np.linspace(0, 0.5, scenarios)
    omega = 2 * math.pi * 160000
    expected = np.broadcast_to(np.eye(2), (scenarios, 2, 2)).astype(complex).copy()
    for iv, qv, dt in zip(i, q, durations):
        step, _, _ = _step_and_control_derivatives(iv, qv, scale, detuning, dt, omega)
        expected = np.einsum("gij,gjk->gik", step, expected)
    actual = _forward_unitary(i, q, durations, scale, detuning, omega)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)


def test_forward_only_objective_matches_complete_grape_loss():
    rng = np.random.default_rng(14)
    i, q = rng.normal(0, 0.2, size=(2, 30))
    durations = np.full(30, 1e-6)
    config = RobustnessConfig(
        omega_reference_rad_s=2 * math.pi * 160000,
        amplitude_errors=np.array([-0.2, 0, 0.2]),
        detuning_ratios=np.array([-0.1, 0, 0.1]),
    )
    analytical = robustness_grape_gradient(i, q, durations, config)
    forward = robustness_objective(i, q, durations, config)
    assert forward.objective.loss == pytest.approx(analytical.objective.loss, abs=1e-15)
    assert forward.objective.fidelity == pytest.approx(
        analytical.objective.fidelity, abs=1e-15
    )
    assert not hasattr(forward, "i_gradient")
    assert not hasattr(forward, "q_gradient")


def test_spline_is_partition_of_unity_and_has_requested_shape() -> None:
    spline = clamped_bspline_design_matrix(600, 150)

    assert spline.shape == (600, 150)
    assert np.allclose(np.asarray(spline.sum(axis=1)).reshape(-1), 1.0)
    assert np.min(spline.data) >= 0.0


def test_bb1_initialization_has_five_segments_and_zero_phase_ends() -> None:
    duration_s = 60e-6
    omega_100 = 2.0 * math.pi * 125e3
    initialization = bb1_initialization(
        150,
        pulse_duration_s=duration_s,
        omega_at_amplitude_100_rad_s=omega_100,
    )
    spline = clamped_bspline_design_matrix(600, 150)
    amplitude_percent, phase_degrees, phase_rad = command_tables(
        initialization.amplitude_coefficients,
        initialization.phase_coefficients_rad,
        spline,
    )

    assert initialization.amplitude_fraction == pytest.approx(1.0 / 3.0)
    assert np.allclose(amplitude_percent, 100.0 / 3.0)
    sample_indices = [30, 120, 300, 480, 570]
    expected_degrees = [0.0, 104.4775121859, 313.4325365578, 104.4775121859, 0.0]
    assert phase_degrees[sample_indices] == pytest.approx(expected_degrees, abs=1e-8)
    assert np.max(np.abs(phase_rad[10:50])) < 1e-12
    assert np.max(np.abs(phase_rad[550:590])) < 1e-12


def _bb1_capture(entry_count: int, offset_rad: float) -> np.ndarray:
    phi = math.acos(-0.25)
    phases = np.empty(entry_count)
    boundaries = [
        0,
        int(0.1 * entry_count),
        int(0.3 * entry_count),
        int(0.7 * entry_count),
        int(0.9 * entry_count),
        entry_count,
    ]
    for start, stop, phase in zip(
        boundaries[:-1],
        boundaries[1:],
        [0.0, phi, 3.0 * phi, phi, 0.0],
    ):
        phases[start:stop] = phase
    return np.exp(1.0j * (phases + offset_rad))


def test_phase_reference_uses_bb1_first_and_last_segments_without_table_change() -> (
    None
):
    offsets = np.asarray([0.397, 0.401, 0.402, 0.400])
    captures = np.asarray([_bb1_capture(600, offset) for offset in offsets])

    reference = estimate_phase_reference(captures)
    corrected = apply_phase_reference(captures[0], reference)

    assert reference.offset_rad == pytest.approx(float(np.mean(offsets)), abs=2e-5)
    assert reference.segment_disagreement_rad < 1e-12
    assert np.angle(np.mean(corrected[20:40])) == pytest.approx(-0.003, abs=2e-5)
    assert np.allclose(np.abs(corrected), np.abs(captures[0]))


def test_phase_reference_rejects_disagreeing_zero_phase_segments() -> None:
    captures = np.asarray([_bb1_capture(600, 0.2) for _ in range(3)])
    captures[:, 540:] *= np.exp(1.0j * math.radians(30.0))

    with pytest.raises(ValueError, match="First/last BB1 phase references disagree"):
        estimate_phase_reference(captures)


def test_phase_reference_rejects_wrong_middle_segment_convention() -> None:
    captures = np.asarray([np.conj(_bb1_capture(600, 0.2)) for _ in range(3)])

    with pytest.raises(ValueError, match="five-segment BB1 phase pattern"):
        estimate_phase_reference(captures)


def _mean_only_config() -> RobustnessConfig:
    return RobustnessConfig(
        omega_reference_rad_s=2.0 * math.pi * 50e3,
        amplitude_errors=np.asarray([-0.15, 0.0, 0.15]),
        detuning_ratios=np.asarray([-0.10, 0.0, 0.10]),
        mean_weight=1.0,
        threshold_weight=0.0,
        cvar_weight=0.0,
    )


def test_exact_grape_gradient_matches_finite_difference() -> None:
    normalized_i = np.asarray([0.55, -0.20, 0.35, 0.10])
    normalized_q = np.asarray([0.10, 0.45, -0.15, 0.25])
    dt_s = np.full(4, 0.7e-6)
    config = _mean_only_config()
    evaluation = robustness_grape_gradient(
        normalized_i,
        normalized_q,
        dt_s,
        config,
    )
    epsilon = 1e-6
    plus = normalized_i.copy()
    minus = normalized_i.copy()
    plus[2] += epsilon
    minus[2] -= epsilon
    finite_difference = (
        robustness_grape_gradient(plus, normalized_q, dt_s, config).objective.loss
        - robustness_grape_gradient(minus, normalized_q, dt_s, config).objective.loss
    ) / (2.0 * epsilon)

    assert evaluation.i_gradient[2] == pytest.approx(
        finite_difference,
        rel=2e-6,
        abs=2e-8,
    )
    assert np.allclose(evaluation.target_unitary, x_pi_target())


def test_threshold_cvar_grape_gradient_matches_finite_difference() -> None:
    normalized_i = np.asarray([0.45, -0.15, 0.30])
    normalized_q = np.asarray([0.20, 0.35, -0.10])
    dt_s = np.full(3, 0.6e-6)
    config = RobustnessConfig(
        omega_reference_rad_s=2.0 * math.pi * 45e3,
        amplitude_errors=np.linspace(-0.2, 0.2, 5),
        detuning_ratios=np.linspace(-0.2, 0.2, 5),
    )
    evaluation = robustness_grape_gradient(
        normalized_i,
        normalized_q,
        dt_s,
        config,
    )
    epsilon = 1e-6
    plus = normalized_q.copy()
    minus = normalized_q.copy()
    plus[1] += epsilon
    minus[1] -= epsilon
    finite_difference = (
        robustness_grape_gradient(normalized_i, plus, dt_s, config).objective.loss
        - robustness_grape_gradient(normalized_i, minus, dt_s, config).objective.loss
    ) / (2.0 * epsilon)

    assert evaluation.q_gradient[1] == pytest.approx(
        finite_difference,
        rel=2e-6,
        abs=2e-8,
    )


def test_polar_gradient_obeys_iq_chain_rule() -> None:
    amplitude = np.asarray([0.2, 0.7])
    phase = np.asarray([0.3, -1.2])
    i_values = amplitude * np.cos(phase)
    q_values = amplitude * np.sin(phase)
    gradient_i = np.asarray([2.0, -0.4])
    gradient_q = np.asarray([-0.5, 1.7])

    amplitude_gradient, phase_gradient = polar_control_gradients(
        i_values,
        q_values,
        gradient_i,
        gradient_q,
    )

    assert amplitude_gradient == pytest.approx(
        gradient_i * np.cos(phase) + gradient_q * np.sin(phase)
    )
    assert phase_gradient == pytest.approx(
        amplitude * (-gradient_i * np.sin(phase) + gradient_q * np.cos(phase))
    )
