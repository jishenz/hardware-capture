"""Measured RF pulse scaling and robustness simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import expm

from hardware_capture.signal.demodulation import DemodulatedWaveform, PulseWindow

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class PulseModel:
    """Resampled measured pulse used as Hamiltonian controls."""

    time_mid: np.ndarray
    dt: np.ndarray
    omega_x: np.ndarray
    omega_y: np.ndarray
    envelope: np.ndarray
    phase: np.ndarray
    raw_envelope_area: float
    scale: float
    omega_ref: float
    pulse_window: PulseWindow


@dataclass(frozen=True)
class RobustnessResult:
    """Fidelity grid versus amplitude and detuning error."""

    amplitude_error: np.ndarray
    detuning_error: np.ndarray
    fidelity: np.ndarray
    target_unitary: np.ndarray
    center_fidelity: float
    min_fidelity: float
    max_fidelity: float
    worst_amplitude_error: float
    worst_detuning_error: float


def bb1_nominal_area(target_flip_angle_rad: float = 0.5 * math.pi) -> float:
    """Return the total transverse area for a BB1 pulse."""

    return 4.0 * math.pi + float(target_flip_angle_rad)


def process_fidelity(target: np.ndarray, actual: np.ndarray) -> float:
    dim = target.shape[0]
    return float(abs(np.trace(target.conj().T @ actual)) ** 2 / dim**2)


def process_fidelity_grid(target: np.ndarray, actual_grid: np.ndarray) -> np.ndarray:
    overlap = np.einsum("ij,...ij->...", np.conj(target), actual_grid)
    return np.abs(overlap) ** 2 / target.shape[0] ** 2


def _active_iq(
    demod: DemodulatedWaveform,
    window: PulseWindow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = slice(window.start_index, window.stop_index)
    return demod.time[active], demod.i[active], demod.q[active]


def build_pulse_model(
    demod: DemodulatedWaveform,
    window: PulseWindow,
    *,
    nominal_pulse_area_rad: float,
    simulation_steps: int = 900,
    omega_rad_s_per_volt: float | None = None,
) -> PulseModel:
    """Scale measured I/Q voltage and resample it for simulation."""

    if simulation_steps < 1:
        raise ValueError("simulation_steps must be at least 1.")

    active_time, active_i, active_q = _active_iq(demod, window)
    start = float(active_time[0])
    stop = float(active_time[-1])
    if stop <= start:
        raise ValueError("Active pulse duration must be positive.")

    edges = np.linspace(start, stop, simulation_steps + 1)
    time_mid = 0.5 * (edges[:-1] + edges[1:])
    dt = np.diff(edges)
    i_mid = np.interp(time_mid, active_time, active_i)
    q_mid = np.interp(time_mid, active_time, active_q)
    envelope = np.hypot(i_mid, q_mid)
    area = float(np.sum(envelope * dt))
    if area <= 0.0:
        raise ValueError("Active pulse envelope area is zero.")

    if omega_rad_s_per_volt is None:
        scale = float(nominal_pulse_area_rad / area)
    else:
        scale = float(omega_rad_s_per_volt)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("omega_rad_s_per_volt must be finite and positive.")
    return PulseModel(
        time_mid=time_mid,
        dt=dt,
        omega_x=scale * i_mid,
        omega_y=scale * q_mid,
        envelope=envelope,
        phase=np.unwrap(np.arctan2(q_mid, i_mid)),
        raw_envelope_area=area,
        scale=scale,
        omega_ref=float(scale * area / (stop - start)),
        pulse_window=window,
    )


def step_unitary_analytic(
    omega_x: np.ndarray,
    omega_y: np.ndarray,
    delta: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Exact exp(-i H dt) for H = 1/2 * (Ox sx + Oy sy + D sz)."""

    h_norm = np.sqrt(omega_x * omega_x + omega_y * omega_y + delta * delta)
    half_angle = 0.5 * h_norm * dt
    ratio = np.zeros_like(h_norm, dtype=float)
    np.divide(np.sin(half_angle), h_norm, out=ratio, where=h_norm > 0.0)

    out = np.empty(h_norm.shape + (2, 2), dtype=complex)
    out[..., 0, 0] = np.cos(half_angle) - 1j * ratio * delta
    out[..., 0, 1] = -1j * ratio * (omega_x - 1j * omega_y)
    out[..., 1, 0] = -1j * ratio * (omega_x + 1j * omega_y)
    out[..., 1, 1] = np.cos(half_angle) + 1j * ratio * delta
    return out


def step_unitary_expm(omega_x: float, omega_y: float, delta: float, dt: float) -> np.ndarray:
    hamiltonian = 0.5 * (omega_x * SIGMA_X + omega_y * SIGMA_Y + delta * SIGMA_Z)
    return expm(-1j * hamiltonian * dt)


def _propagate_analytic(pulse: PulseModel, amp_scale: np.ndarray, delta: np.ndarray) -> np.ndarray:
    grid_shape = np.broadcast_shapes(np.shape(amp_scale), np.shape(delta))
    unitary = np.zeros(grid_shape + (2, 2), dtype=complex)
    unitary[..., 0, 0] = 1.0
    unitary[..., 1, 1] = 1.0
    for omega_x, omega_y, dt in zip(pulse.omega_x, pulse.omega_y, pulse.dt):
        step = step_unitary_analytic(
            amp_scale * float(omega_x),
            amp_scale * float(omega_y),
            delta,
            float(dt),
        )
        unitary = np.einsum("...ij,...jk->...ik", step, unitary)
    return unitary


def simulate_unitary(
    pulse: PulseModel,
    amplitude_error: float,
    detuning_error: float,
    *,
    method: str = "analytic",
) -> np.ndarray:
    """Propagate one pulse at one amplitude/detuning error point."""

    amp_scale = np.asarray(1.0 + float(amplitude_error))
    delta = np.asarray(float(detuning_error) * pulse.omega_ref)
    if method == "analytic":
        return _propagate_analytic(pulse, amp_scale, delta)
    if method != "scipy_expm":
        raise ValueError('method must be "analytic" or "scipy_expm".')

    unitary = I2.copy()
    for omega_x, omega_y, dt in zip(pulse.omega_x, pulse.omega_y, pulse.dt):
        unitary = step_unitary_expm(
            float(amp_scale) * float(omega_x),
            float(amp_scale) * float(omega_y),
            float(delta),
            float(dt),
        ) @ unitary
    return unitary


def _scan_expm(
    pulse: PulseModel,
    amplitude_error: np.ndarray,
    detuning_error: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    fidelity = np.empty((len(detuning_error), len(amplitude_error)), dtype=float)
    for row, det_err in enumerate(detuning_error):
        for col, amp_err in enumerate(amplitude_error):
            actual = simulate_unitary(pulse, float(amp_err), float(det_err), method="scipy_expm")
            fidelity[row, col] = process_fidelity(target, actual)
    return fidelity


def _result(
    amplitude_error: np.ndarray,
    detuning_error: np.ndarray,
    fidelity: np.ndarray,
    target: np.ndarray,
) -> RobustnessResult:
    amp_center = int(np.argmin(np.abs(amplitude_error)))
    det_center = int(np.argmin(np.abs(detuning_error)))
    worst = np.unravel_index(int(np.argmin(fidelity)), fidelity.shape)
    return RobustnessResult(
        amplitude_error=amplitude_error,
        detuning_error=detuning_error,
        fidelity=fidelity,
        target_unitary=target,
        center_fidelity=float(fidelity[det_center, amp_center]),
        min_fidelity=float(np.min(fidelity)),
        max_fidelity=float(np.max(fidelity)),
        worst_amplitude_error=float(amplitude_error[worst[1]]),
        worst_detuning_error=float(detuning_error[worst[0]]),
    )


def scan_robustness(
    pulse: PulseModel,
    *,
    grid_size: int = 101,
    amplitude_error_min: float = -0.5,
    amplitude_error_max: float = 0.5,
    detuning_error_min: float = -0.5,
    detuning_error_max: float = 0.5,
    method: str = "analytic",
) -> RobustnessResult:
    """Scan process fidelity over amplitude and detuning error."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    amplitude_error = np.linspace(amplitude_error_min, amplitude_error_max, grid_size)
    detuning_error = np.linspace(detuning_error_min, detuning_error_max, grid_size)
    target = simulate_unitary(pulse, 0.0, 0.0, method=method)

    if method == "analytic":
        amp_scale = 1.0 + amplitude_error[None, :]
        delta = pulse.omega_ref * detuning_error[:, None]
        fidelity = process_fidelity_grid(target, _propagate_analytic(pulse, amp_scale, delta))
    elif method == "scipy_expm":
        fidelity = _scan_expm(pulse, amplitude_error, detuning_error, target)
    else:
        raise ValueError('method must be "analytic" or "scipy_expm".')
    return _result(amplitude_error, detuning_error, fidelity, target)


def threshold_region_summary(
    result: RobustnessResult,
    levels: Sequence[float] = (0.999, 0.995, 0.99),
) -> list[str]:
    """Return text summaries of fidelity contour extents."""

    lines = []
    amp_grid, det_grid = np.meshgrid(result.amplitude_error, result.detuning_error)
    for level in levels:
        keep = result.fidelity >= level
        if not np.any(keep):
            lines.append(f"Region F >= {level:g}: no grid points")
            continue
        lines.append(
            f"Region F >= {level:g}: "
            f"amp_error {np.min(amp_grid[keep]):+.4g} to {np.max(amp_grid[keep]):+.4g}, "
            f"detuning_error {np.min(det_grid[keep]):+.4g} to {np.max(det_grid[keep]):+.4g}"
        )
    return lines
