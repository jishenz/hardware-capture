"""Shared pulse conditions for simulated and hardware X(pi) optimization."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hardware_capture.configuration import Experiment, positive
from hardware_capture.pulses.conditions import Pulse, generate
from hardware_capture.optimization.core import RobustnessConfig, bb1_initialization, clamped_bspline_design_matrix, _coefficient_locations, command_tables, robustness_grape_gradient, polar_control_gradients, adam_update
from hardware_capture.optimization.pipeline import ClosedLoopConfig, run_closed_loop_optimizer
from .common import write_json


def settings(config: Experiment):
    pulse = config.section("pulse")
    options = config.section("optimization")
    reference = positive(config.calibration().get("rabi_at_100_hz", 0), "calibration.rabi_at_100_hz")
    duration = positive(config.require("pulse", "duration_s"), "pulse.duration_s")
    entries = pulse.get("samples", 6000)
    coefficients = options.get("coefficients", min(1500, entries))
    updates = options.get("updates", 0)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in [entries, coefficients, updates]) or not 4 <= coefficients <= entries or updates < 0:
        raise ValueError("Require integer 4 <= optimization.coefficients <= pulse.samples and updates >= 0")
    if config.section("analysis").get("target", "fixed_xpi") != "fixed_xpi":
        raise ValueError("Optimization currently supports fixed_xpi only")
    maximum = positive(pulse.get("max_rabi_hz", reference), "pulse.max_rabi_hz") / reference
    if maximum > 1 + 1e-12:
        raise ValueError("pulse.max_rabi_hz exceeds the calibrated amplitude-100 limit")
    for key in ["amplitude_learning_rate", "phase_learning_rate", "amplitude_step", "phase_step_rad"]:
        if key in options:
            positive(options[key], "optimization." + key)
    return pulse, options, reference, duration, entries, coefficients, updates, min(maximum, 1.0)


def initial_coefficients(config: Experiment):
    pulse, _, reference, duration, entries, count, _, maximum = settings(config)
    if pulse.get("kind", "bb1") == "bb1" and pulse.get("angle_deg", 180) == 180 and pulse.get("axis_phase_deg", 0) == 0:
        initial = bb1_initialization(count, pulse_duration_s=duration, omega_at_amplitude_100_rad_s=2 * math.pi * reference)
        amplitude, phase = initial.amplitude_coefficients, initial.phase_coefficients_rad
    else:
        samples = generate(config)
        locations = _coefficient_locations(count, 3)
        times = samples.time_s / duration
        amplitude = np.interp(locations, times, samples.rabi_hz / reference)
        phase = np.interp(locations, times, samples.phase_rad)
    if np.max(amplitude) > maximum + 1e-12:
        raise ValueError("Initial pulse exceeds pulse.max_rabi_hz")
    return amplitude, phase


def objective(config: Experiment) -> RobustnessConfig:
    _, opts, reference, *_ = settings(config)
    amp, detuning_hz = config.scan_axes()
    keys = ["fidelity_threshold", "threshold_smoothing", "mean_weight", "threshold_weight", "cvar_weight", "cvar_confidence"]
    result = RobustnessConfig(omega_reference_rad_s=2 * math.pi * reference, amplitude_errors=amp, detuning_ratios=detuning_hz / reference, **{key: opts[key] for key in keys if key in opts})
    result.validate()
    return result


def offline(config: Experiment, directory: Path) -> dict:
    _, opts, reference, duration, entries, count, updates, maximum = settings(config)
    amplitudes, phases = initial_coefficients(config)
    spline = clamped_bspline_design_matrix(entries, count)
    dt = np.full(entries, duration / entries)
    model = config.section("simulation")
    gain = positive(model.get("amplitude_scale", 1.0), "simulation.amplitude_scale")
    ripple = float(model.get("phase_ripple_rad", 0))
    position = (np.arange(entries) + 0.5) / entries
    loss_config = objective(config)
    amplitude_state = phase_state = None
    history = []
    best = None
    for epoch in range(updates + 1):
        percent, _, command_phase = command_tables(amplitudes, phases, spline)
        measured_amp = gain * percent / 100
        measured_phase = command_phase + ripple * np.sin(2 * math.pi * position)
        i, q = measured_amp * np.cos(measured_phase), measured_amp * np.sin(measured_phase)
        evaluation = robustness_grape_gradient(i, q, dt, loss_config)
        row = {"epoch": epoch, "loss": evaluation.objective.loss, "center_fidelity": evaluation.objective.center_fidelity, "mean_fidelity": evaluation.objective.mean_fidelity, "worst_fidelity": evaluation.objective.worst_fidelity}
        history.append(row)
        if best is None or row["loss"] < best[0]:
            best = (row["loss"], epoch, percent.copy(), command_phase.copy())
        if epoch == updates:
            break
        ag, pg = polar_control_gradients(i, q, evaluation.i_gradient, evaluation.q_gradient)
        # This is the same identity-response approximation used by the live loop.
        amplitudes, amplitude_state = adam_update(amplitudes, np.asarray(spline.T @ ag), amplitude_state, learning_rate=opts.get("amplitude_learning_rate", 0.02), maximum_step=opts.get("amplitude_step", 0.02))
        phases, phase_state = adam_update(phases, np.asarray(spline.T @ pg), phase_state, learning_rate=opts.get("phase_learning_rate", 0.02), maximum_step=opts.get("phase_step_rad", 0.1))
        amplitudes = np.clip(amplitudes, 0, maximum)
    best_pulse = Pulse(duration, best[2] * reference / 100, best[3])
    artifacts = best_pulse.save(directory, rabi_at_100_hz=reference)
    write_json(directory / "history.json", history)
    np.savez_compressed(directory / "checkpoint.npz", amplitude_coefficients=amplitudes, phase_coefficients_rad=phases)
    return {"best_epoch": best[1], "best_loss": best[0], "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "pulse_path": artifacts["pulse_path"], "backend": "simulation", "target": "fixed_xpi"}


def hardware_config(config: Experiment, directory: Path, *, phase_only=False) -> ClosedLoopConfig:
    pulse, opts, reference, duration, entries, coefficients, updates, maximum = settings(config)
    if 2.5 / duration > reference * maximum * (1 + 1e-12):
        raise ValueError("BB1 phase reference exceeds pulse.max_rabi_hz; increase pulse.duration_s")
    calibration = config.calibration()
    scope, demod, tnmr = (config.section(name) for name in ("scope", "demodulation", "tnmr"))
    for key, value, minimum in [
        ("optimization.phase_reference_captures", opts.get("phase_reference_captures", 5), 2),
        ("demodulation.filter_order", demod.get("filter_order", 4), 1),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    positive(demod.get("cutoff_hz", 25e6), "demodulation.cutoff_hz")
    objective(config)
    amp, detuning_hz = config.scan_axes()
    initial_amp = initial_phase = None
    if not phase_only and opts.get("use_configured_seed", True):
        a, p = initial_coefficients(config)
        initial_amp, initial_phase = tuple(a), tuple(p)
    kwargs = dict(
        tnt_path=Path(config.require("tnmr", "path")), amplitude_table_name=config.require("tnmr", "amplitude_table"), phase_table_name=config.require("tnmr", "phase_table"),
        visa_resource=config.require("scope", "resource"), results_root=directory,
        scope_channel=scope.get("channel", "CHAN1"), acquisition_window_start_s=scope.get("window_start_s", -10e-6), acquisition_window_end_s=scope.get("window_end_s", 100e-6),
        pulse_duration_s=duration, pulse_start_s=config.section("window").get("start_s"),
        demodulation_frequency_hz=positive(config.require("demodulation", "frequency_hz"), "demodulation.frequency_hz"), baseband_cutoff_hz=demod.get("cutoff_hz", 25e6), demodulation_filter_order=demod.get("filter_order", 4), q_sign=demod.get("q_sign", -1),
        scope_omega_rad_s_per_volt=positive(calibration.get("omega_rad_s_per_volt", 0), "calibration.omega_rad_s_per_volt"), omega_at_amplitude_100_rad_s=2 * math.pi * reference,
        coefficient_count=coefficients, table_entry_count=entries, update_count=0 if phase_only else updates, phase_reference_captures=opts.get("phase_reference_captures", 5),
        amplitude_errors=amp, detuning_ratios=detuning_hz / reference, launch_tnmr_if_needed=tnmr.get("launch", False),
        amplitude_learning_rate=opts.get("amplitude_learning_rate", 0.02), phase_learning_rate=opts.get("phase_learning_rate", 0.02), maximum_amplitude_coefficient_step=opts.get("amplitude_step", 0.02), maximum_phase_coefficient_step_rad=opts.get("phase_step_rad", 0.1),
        scope_timeout_ms=scope.get("timeout_ms", 30000), trigger_timeout_s=scope.get("trigger_timeout_s", 30), armed_timeout_s=scope.get("armed_timeout_s", 2), trigger_poll_s=scope.get("poll_s", 0.05),
        maximum_amplitude_fraction=maximum, initial_amplitude_coefficients=initial_amp, initial_phase_coefficients_rad=initial_phase,
    )
    for key in ["fidelity_threshold", "threshold_smoothing", "mean_weight", "threshold_weight", "cvar_weight", "cvar_confidence"]:
        if key in opts:
            kwargs[key] = opts[key]
    result = ClosedLoopConfig(**kwargs)
    result.validate()
    return result


def live(config: Experiment, directory: Path, *, phase_only=False, tnmr=None, scope=None):
    result = run_closed_loop_optimizer(hardware_config(config, directory, phase_only=phase_only), tnmr=tnmr, scope=scope)
    reference = config.calibration()["rabi_at_100_hz"]
    with np.load(result.run_directory / "best_measured.npz") as best:
        pulse = Pulse(config.require("pulse", "duration_s"), best["amplitude_percent"] * reference / 100, best["command_phase_unwrapped_rad"])
    artifacts = pulse.save(directory, rabi_at_100_hz=reference)
    return {"run_directory": str(result.run_directory), "best_epoch": result.best_epoch, "best_loss": result.best_loss, "target": "fixed_xpi", "pulse_path": artifacts["pulse_path"]}
