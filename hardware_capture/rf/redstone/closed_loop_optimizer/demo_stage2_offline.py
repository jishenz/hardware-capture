"""Fast, hardware-free smoke demo of one complete Stage-2 feedback loop."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_capture.rf.redstone.closed_loop_optimizer.core import (
    RobustnessConfig,
    adam_update,
    bb1_initialization,
    clamped_bspline_design_matrix,
    command_tables,
    polar_control_gradients,
    robustness_grape_gradient,
)


COEFFICIENT_COUNT = 40
ENTRY_COUNT = 160
UPDATE_COUNT = 3
PULSE_DURATION_S = 60e-6
OMEGA_100_RAD_S = 2.0 * math.pi * 125e3


def main() -> int:
    """Optimize repeated synthetic captures and report the measured improvement."""

    started = time.perf_counter()
    spline = clamped_bspline_design_matrix(ENTRY_COUNT, COEFFICIENT_COUNT)
    initial = bb1_initialization(
        COEFFICIENT_COUNT,
        pulse_duration_s=PULSE_DURATION_S,
        omega_at_amplitude_100_rad_s=OMEGA_100_RAD_S,
    )
    amplitude_coefficients = initial.amplitude_coefficients.copy()
    phase_coefficients = initial.phase_coefficients_rad.copy()
    amplitude_adam = None
    phase_adam = None
    position = (np.arange(ENTRY_COUNT) + 0.5) / ENTRY_COUNT
    dt_s = np.full(ENTRY_COUNT, PULSE_DURATION_S / ENTRY_COUNT)
    config = RobustnessConfig(
        omega_reference_rad_s=OMEGA_100_RAD_S,
        amplitude_errors=np.linspace(-0.25, 0.25, 7),
        detuning_ratios=np.linspace(-0.25, 0.25, 7),
    )

    history = []
    print("epoch      loss       center      mean       worst")
    for epoch in range(UPDATE_COUNT + 1):
        amplitude_percent, _, command_phase = command_tables(
            amplitude_coefficients,
            phase_coefficients,
            spline,
        )
        command_amplitude = amplitude_percent / 100.0

        # A deterministic stand-in for a fresh oscilloscope capture.
        measured_amplitude = 0.82 * command_amplitude
        measured_phase = command_phase + 0.14 * np.sin(2.0 * math.pi * position)
        measured_i = measured_amplitude * np.cos(measured_phase)
        measured_q = measured_amplitude * np.sin(measured_phase)
        evaluation = robustness_grape_gradient(measured_i, measured_q, dt_s, config)
        objective = evaluation.objective
        history.append(objective)
        print(
            f"{epoch:5d}  {objective.loss:10.6f}  "
            f"{objective.center_fidelity:10.6f}  "
            f"{objective.mean_fidelity:10.6f}  "
            f"{objective.worst_fidelity:10.6f}"
        )
        if epoch == UPDATE_COUNT:
            break

        amplitude_gradient, phase_gradient = polar_control_gradients(
            measured_i,
            measured_q,
            evaluation.i_gradient,
            evaluation.q_gradient,
        )
        amplitude_coefficients, amplitude_adam = adam_update(
            amplitude_coefficients,
            np.asarray(spline.T @ amplitude_gradient),
            amplitude_adam,
            learning_rate=0.01,
            maximum_step=0.01,
        )
        phase_coefficients, phase_adam = adam_update(
            phase_coefficients,
            np.asarray(spline.T @ phase_gradient),
            phase_adam,
            learning_rate=0.02,
            maximum_step=0.04,
        )
        amplitude_coefficients = np.clip(amplitude_coefficients, 0.0, 1.0)

    first, last = history[0], history[-1]
    passed = (
        last.loss < first.loss
        and last.center_fidelity > first.center_fidelity
        and last.mean_fidelity > first.mean_fidelity
        and last.worst_fidelity > first.worst_fidelity
    )
    print(f"\nStage-2 smoke demo: {'PASS' if passed else 'FAIL'}")
    print(f"loss change: {first.loss:.6f} -> {last.loss:.6f}")
    print(f"runtime: {time.perf_counter() - started:.3f} s")
    print("No TNMR or oscilloscope command was sent.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
