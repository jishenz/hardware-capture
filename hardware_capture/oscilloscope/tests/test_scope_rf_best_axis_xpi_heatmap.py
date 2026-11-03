import math

import numpy as np
import pytest

from hardware_capture.oscilloscope.examples import scope_rf_best_axis_xpi_heatmap
from hardware_capture.oscilloscope.examples.scope_rf_best_axis_xpi_heatmap import (
    best_equatorial_pi_target,
    scan_best_axis_xpi,
)
from hardware_capture.oscilloscope.pulse_robustness import PulseModel
from hardware_capture.oscilloscope.rf_demodulation import PulseWindow


pytestmark = pytest.mark.unit


def _constant_phase_pulse(rotation_angle: float, phase: float) -> PulseModel:
    steps = 25
    dt = 2e-8
    omega = rotation_angle / (steps * dt)
    omega_x = np.full(steps, omega * math.cos(phase))
    omega_y = np.full(steps, omega * math.sin(phase))
    return PulseModel(
        time_mid=(np.arange(steps) + 0.5) * dt,
        dt=np.full(steps, dt),
        omega_x=omega_x,
        omega_y=omega_y,
        envelope=np.full(steps, omega),
        phase=np.full(steps, phase),
        raw_envelope_area=rotation_angle,
        scale=1.0,
        omega_ref=omega,
        pulse_window=PulseWindow(0.0, steps * dt, 0, steps),
    )


def test_best_axis_recovers_constant_nominal_phase():
    phase = 0.73
    analysis = scan_best_axis_xpi(
        _constant_phase_pulse(math.pi, phase),
        grid_size=5,
    )

    assert analysis.target_phase_rad == pytest.approx(phase)
    assert analysis.result.center_fidelity == pytest.approx(1.0, abs=1e-14)


def test_center_fidelity_is_not_forced_to_one():
    rotation_angle = 0.8 * math.pi
    analysis = scan_best_axis_xpi(
        _constant_phase_pulse(rotation_angle, -0.41),
        grid_size=5,
    )

    assert analysis.result.center_fidelity == pytest.approx(
        math.sin(0.5 * rotation_angle) ** 2,
        abs=1e-14,
    )
    assert analysis.result.center_fidelity < 1.0


def test_one_constant_phase_choice_does_not_change_best_axis_fidelity():
    first = scan_best_axis_xpi(
        _constant_phase_pulse(0.93 * math.pi, 0.1),
        grid_size=5,
    )
    second = scan_best_axis_xpi(
        _constant_phase_pulse(0.93 * math.pi, 1.2),
        grid_size=5,
    )

    assert second.result.fidelity == pytest.approx(first.result.fidelity, abs=1e-14)


def test_analytic_and_scipy_expm_scans_agree():
    pulse = _constant_phase_pulse(0.91 * math.pi, -0.37)

    analytic = scan_best_axis_xpi(pulse, grid_size=3, method="analytic")
    scipy = scan_best_axis_xpi(pulse, grid_size=3, method="scipy_expm")

    assert analytic.target_phase_rad == pytest.approx(scipy.target_phase_rad, abs=1e-14)
    assert analytic.result.fidelity == pytest.approx(scipy.result.fidelity, abs=1e-13)


def test_parser_defaults_come_from_this_scripts_configuration(monkeypatch):
    monkeypatch.setattr(scope_rf_best_axis_xpi_heatmap, "CHANNEL", "CHAN3")
    monkeypatch.setattr(scope_rf_best_axis_xpi_heatmap, "GRID_SIZE", 17)
    monkeypatch.setattr(scope_rf_best_axis_xpi_heatmap, "Q_CONVENTION", "I-iQ")

    args = scope_rf_best_axis_xpi_heatmap.build_parser().parse_args([])

    assert args.channel == "CHAN3"
    assert args.grid_size == 17
    assert args.q_convention == "I-iQ"


def test_best_axis_target_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="2x2"):
        best_equatorial_pi_target(np.ones(2))


@pytest.mark.parametrize("grid_size", [1, 2, 4])
def test_scan_requires_an_odd_grid_with_a_true_center(grid_size):
    with pytest.raises(ValueError, match="odd integer"):
        scan_best_axis_xpi(
            _constant_phase_pulse(math.pi, 0.0),
            grid_size=grid_size,
        )
