"""Shared pulse analysis with explicit target and physical detuning axes."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from hardware_capture.figures import subplots

import numpy as np

from hardware_capture.configuration import Experiment, positive
from hardware_capture.pulses.conditions import generate
from hardware_capture.analysis.robustness import PulseModel, I2, SIGMA_X, build_pulse_model, simulate_unitary, step_unitary_analytic, process_fidelity_grid
from hardware_capture.analysis.best_axis import best_equatorial_pi_target
from hardware_capture.analysis.plotting import plot_pulse_diagnostics
from hardware_capture.analysis import overlays
from hardware_capture.signal.demodulation import PulseWindow
from .common import acquire, demodulate, pulse_window, write_json


def model(config: Experiment, *, scope=None):
    options = config.section("analysis")
    if config.section("input").get("kind") == "pulse":
        pulse = generate(config)
        amplitude = 2 * math.pi * pulse.rabi_hz
        window = PulseWindow(0, pulse.duration_s, 0, len(amplitude))
        result = PulseModel(pulse.time_s, pulse.dt_s, amplitude * np.cos(pulse.phase_rad), amplitude * np.sin(pulse.phase_rad), pulse.rabi_hz, pulse.phase_rad, float(np.sum(pulse.rabi_hz * pulse.dt_s)), 2 * math.pi, float(np.mean(amplitude)), window)
        return result, None, window
    raw = acquire(config, scope=scope)
    demod = demodulate(config, raw)
    offset = float(options.get("phase_offset_rad", 0))
    z = demod.z * np.exp(-1j * offset)
    demod = replace(demod, i=z.real, q=z.imag, z=z, phase=np.unwrap(np.angle(z)))
    window = pulse_window(config, demod)
    scaling = options.get("scaling", "fixed")
    if scaling not in {"fixed", "bb1_area"}:
        raise ValueError("analysis.scaling must be fixed or bb1_area")
    scale = None if scaling == "bb1_area" else positive(config.calibration().get("omega_rad_s_per_volt", 0), "calibration.omega_rad_s_per_volt")
    pulse = build_pulse_model(demod, window, nominal_pulse_area_rad=4 * math.pi + math.radians(config.section("pulse").get("angle_deg", 180)), simulation_steps=options.get("steps", 900), omega_rad_s_per_volt=scale)
    return pulse, demod, window


def scan(config: Experiment, pulse: PulseModel) -> dict:
    options = config.section("analysis")
    mode = options.get("target", "fixed_xpi")
    nominal = simulate_unitary(pulse, 0, 0)
    phase = 0.0
    if mode == "nominal":
        target = nominal
    elif mode == "best_axis_pi":
        target, phase = best_equatorial_pi_target(nominal)
    elif mode == "fixed_xpi":
        target = -1j * SIGMA_X
    else:
        raise ValueError("analysis.target must be fixed_xpi, nominal or best_axis_pi")
    amp, detuning = config.scan_axes()
    method = options.get("method", "analytic")
    if method not in {"analytic", "scipy_expm"}:
        raise ValueError("analysis.method must be analytic or scipy_expm")
    if method == "analytic":
        actual = np.broadcast_to(I2, (len(detuning), len(amp), 2, 2)).copy()
        for x, y, dt in zip(pulse.omega_x, pulse.omega_y, pulse.dt):
            step = step_unitary_analytic((1 + amp[None, :]) * x, (1 + amp[None, :]) * y, 2 * math.pi * detuning[:, None], float(dt))
            actual = np.einsum("...ij,...jk->...ik", step, actual)
    else:
        actual = np.array([[simulate_unitary(pulse, a, 2 * math.pi * d / pulse.omega_ref, method="scipy_expm") for a in amp] for d in detuning])
    fidelity = process_fidelity_grid(target, actual)
    center = process_fidelity_grid(target, nominal)
    return {"amplitude_errors": amp, "detuning_hz": detuning, "fidelity": fidelity, "center_fidelity": float(center), "target": mode, "target_phase_rad": phase, "worst_fidelity": float(np.min(fidelity))}


def analyze(config: Experiment, directory: Path, *, scope=None) -> dict:
    pulse, demod, window = model(config, scope=scope)
    result = scan(config, pulse)
    np.savez_compressed(directory / "analysis.npz", **{k: result[k] for k in ["amplitude_errors", "detuning_hz", "fidelity"]}, time_s=pulse.time_mid, omega_x_rad_s=pulse.omega_x, omega_y_rad_s=pulse.omega_y)
    import matplotlib.pyplot as plt
    fig, ax = subplots(show=config.section("output").get("show_plots", False))
    mesh = ax.pcolormesh(result["amplitude_errors"], result["detuning_hz"] / 1000, result["fidelity"], shading="nearest", vmin=0, vmax=1)
    ax.set(xlabel="Fractional amplitude error", ylabel="Detuning (kHz)", title=f"Target: {result['target']}")
    fig.colorbar(mesh, ax=ax, label="Process fidelity")
    fig.tight_layout(); fig.savefig(directory / "heatmap.png", dpi=150)
    show = config.section("output").get("show_plots", False)
    if show:
        plt.show()
    plt.close(fig)
    if demod is not None:
        plot_pulse_diagnostics(demod, window, pulse, save_figure=True, figure_path=directory / "pulse.png", show_plot=show)
        options = config.section("analysis")
        if options.get("amplitude_path"):
            overlays._plot_amplitude_comparison(demod, window, pulse, Path(options["amplitude_path"]), amplitude_unit=options.get("amplitude_unit", "auto"), show_plot=show, figure_path=directory / "amplitude_comparison.png")
        if options.get("phase_path"):
            overlays._plot_phase_comparison(demod, window, Path(options["phase_path"]), phase_unit=options.get("phase_unit", "auto"), show_plot=show, figure_path=directory / "phase_comparison.png")
    write_json(directory / "analysis.json", result)
    return {k: result[k] for k in ["target", "center_fidelity", "worst_fidelity", "target_phase_rad"]}
