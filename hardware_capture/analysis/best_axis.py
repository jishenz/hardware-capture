"""Reusable analysis best_axis algorithms. Use hardware_capture.api for workflows."""

from __future__ import annotations

import math

from dataclasses import dataclass

from pathlib import Path
from hardware_capture.figures import subplots

import numpy as np

from hardware_capture.analysis.robustness import I2, SIGMA_X, SIGMA_Y, PulseModel, RobustnessResult, process_fidelity, process_fidelity_grid, simulate_unitary, step_unitary_analytic

@dataclass(frozen=True)
class BestAxisAnalysis:
    """Best-axis target metadata together with the fixed-target scan."""

    result: RobustnessResult
    nominal_unitary: np.ndarray
    target_phase_rad: float

def best_equatorial_pi_target(nominal: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the equatorial pi gate closest to a nominal SU(2) unitary.

    For ``U = a I - i (b_x X + b_y Y + b_z Z)``, the closest equatorial pi
    target has phase ``atan2(b_y, b_x)``.  Its axis is meaningful modulo pi
    because opposite axes give pi gates that differ only by global phase.
    """

    unitary = np.asarray(nominal, dtype=complex)
    if unitary.shape != (2, 2):
        raise ValueError("nominal must be a 2x2 unitary.")
    if not np.all(np.isfinite(unitary)):
        raise ValueError("nominal must contain only finite values.")

    b_x = float(np.real(0.5j * np.trace(SIGMA_X @ unitary)))
    b_y = float(np.real(0.5j * np.trace(SIGMA_Y @ unitary)))
    if math.hypot(b_x, b_y) <= 10.0 * np.finfo(float).eps:
        phase = 0.0
    else:
        phase = math.atan2(b_y, b_x)
    target = -1j * (math.cos(phase) * SIGMA_X + math.sin(phase) * SIGMA_Y)
    return target, phase

def _propagate_analytic_grid(
    pulse: PulseModel,
    amplitude_scale: np.ndarray,
    detuning_rad_s: np.ndarray,
) -> np.ndarray:
    grid_shape = np.broadcast_shapes(amplitude_scale.shape, detuning_rad_s.shape)
    unitary = np.broadcast_to(I2, grid_shape + (2, 2)).copy()
    for omega_x, omega_y, dt in zip(pulse.omega_x, pulse.omega_y, pulse.dt):
        step = step_unitary_analytic(
            amplitude_scale * float(omega_x),
            amplitude_scale * float(omega_y),
            detuning_rad_s,
            float(dt),
        )
        unitary = np.einsum("...ij,...jk->...ik", step, unitary)
    return unitary

def _result_from_grid(
    amplitude_error: np.ndarray,
    detuning_error: np.ndarray,
    fidelity: np.ndarray,
    target: np.ndarray,
) -> RobustnessResult:
    center_column = int(np.argmin(np.abs(amplitude_error)))
    center_row = int(np.argmin(np.abs(detuning_error)))
    worst = np.unravel_index(int(np.argmin(fidelity)), fidelity.shape)
    return RobustnessResult(
        amplitude_error=amplitude_error,
        detuning_error=detuning_error,
        fidelity=fidelity,
        target_unitary=target,
        center_fidelity=float(fidelity[center_row, center_column]),
        min_fidelity=float(np.min(fidelity)),
        max_fidelity=float(np.max(fidelity)),
        worst_amplitude_error=float(amplitude_error[worst[1]]),
        worst_detuning_error=float(detuning_error[worst[0]]),
    )

def scan_best_axis_xpi(
    pulse: PulseModel,
    *,
    grid_size: int = 101,
    amplitude_error_min: float = -0.5,
    amplitude_error_max: float = 0.5,
    detuning_error_min: float = -0.5,
    detuning_error_max: float = 0.5,
    method: str = "analytic",
) -> BestAxisAnalysis:
    """Scan fidelity to one best-axis equatorial pi target."""

    if grid_size < 3 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least 3.")
    if method not in {"analytic", "scipy_expm"}:
        raise ValueError('method must be "analytic" or "scipy_expm".')

    amplitude_error = np.linspace(amplitude_error_min, amplitude_error_max, grid_size)
    detuning_error = np.linspace(detuning_error_min, detuning_error_max, grid_size)
    nominal = simulate_unitary(pulse, 0.0, 0.0, method=method)
    target, target_phase_rad = best_equatorial_pi_target(nominal)

    if method == "analytic":
        amplitude_scale = 1.0 + amplitude_error[None, :]
        detuning_rad_s = pulse.omega_ref * detuning_error[:, None]
        actual = _propagate_analytic_grid(pulse, amplitude_scale, detuning_rad_s)
        fidelity = process_fidelity_grid(target, actual)
    else:
        fidelity = np.empty((grid_size, grid_size), dtype=float)
        for row, detuning in enumerate(detuning_error):
            for column, amplitude in enumerate(amplitude_error):
                actual = simulate_unitary(
                    pulse,
                    float(amplitude),
                    float(detuning),
                    method="scipy_expm",
                )
                fidelity[row, column] = process_fidelity(target, actual)

    return BestAxisAnalysis(
        result=_result_from_grid(
            amplitude_error,
            detuning_error,
            fidelity,
            target,
        ),
        nominal_unitary=nominal,
        target_phase_rad=target_phase_rad,
    )

def _phase_mod_180_degrees(phase_rad: float) -> float:
    return float(np.mod(np.degrees(phase_rad), 180.0))

def plot_best_axis_heatmap(
    analysis: BestAxisAnalysis,
    *,
    save_figure: bool,
    figure_path: Path,
    show_plot: bool,
    vmin: float = 0.99,
    vmax: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    result = analysis.result
    extent = (
        float(result.detuning_error[0]),
        float(result.detuning_error[-1]),
        float(result.amplitude_error[0]),
        float(result.amplitude_error[-1]),
    )
    fig, ax = subplots(figsize=(8, 6), show=show_plot)
    image = ax.imshow(
        result.fidelity.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.plot(0.0, 0.0, marker="+", markersize=14, markeredgewidth=2.5, color="red")
    ax.set_xlabel("Normalized detuning error (Delta / Omega_ref)")
    ax.set_ylabel("Amplitude error (fractional; amplitude scale = 1 + error)")
    ax.set_title(
        "Best-axis equatorial X(pi) process fidelity\n"
        "one fixed nominal axis, phase = "
        f"{_phase_mod_180_degrees(analysis.target_phase_rad):.3f} deg (mod 180 deg)"
    )
    fig.colorbar(image, ax=ax).set_label("Process fidelity")
    fig.tight_layout()
    if save_figure:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=220)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
