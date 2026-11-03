"""TNMR upload, RIGOL capture, and closed-loop optimization orchestration."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from hardware_capture.figures import subplots
from typing import Any

import numpy as np

from hardware_capture.signal.demodulation import (
    DemodulatedWaveform,
    PulseWindow,
    demodulate_raw_rf,
    detect_pulse_window,
)
from hardware_capture.devices.rigol import (
    RigolDHO,
    ScopeWaveform,
    WaveformCaptureConfig,
)
from hardware_capture.devices.tnmr import TNMRControl

from .core import (
    AdamState,
    GrapeEvaluation,
    PhaseReference,
    RobustnessConfig,
    adam_update,
    apply_phase_reference,
    bb1_initialization,
    clamped_bspline_design_matrix,
    command_tables,
    estimate_phase_reference,
    polar_control_gradients,
    robustness_grape_gradient,
)


_NUMBER_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


@dataclass(frozen=True)
class ClosedLoopConfig:
    """All settings needed for one live two-stage optimization run."""

    tnt_path: Path
    amplitude_table_name: str
    phase_table_name: str
    visa_resource: str
    results_root: Path
    scope_channel: str = "CHAN1"
    acquisition_window_start_s: float = -10e-6
    acquisition_window_end_s: float = 100e-6
    pulse_duration_s: float = 60e-6
    pulse_start_s: float | None = None
    demodulation_frequency_hz: float = 601.0100113e6
    baseband_cutoff_hz: float = 25e6
    demodulation_filter_order: int = 4
    q_sign: int = -1
    scope_omega_rad_s_per_volt: float = 1_776_212.654495664
    omega_at_amplitude_100_rad_s: float = 2.0 * math.pi * 123_968.40469314181
    coefficient_count: int = 1500
    table_entry_count: int = 6000
    spline_degree: int = 3
    phase_reference_captures: int = 5
    phase_reference_edge_trim_fraction: float = 0.20
    phase_reference_minimum_amplitude_fraction: float = 0.20
    phase_reference_maximum_segment_disagreement_rad: float = math.radians(10.0)
    phase_reference_maximum_capture_spread_rad: float = math.radians(5.0)
    phase_reference_maximum_bb1_pattern_residual_rad: float = math.radians(15.0)
    amplitude_errors: np.ndarray = field(
        default_factory=lambda: np.linspace(-0.5, 0.5, 21)
    )
    detuning_ratios: np.ndarray = field(
        default_factory=lambda: np.linspace(-0.5, 0.5, 21)
    )
    fidelity_threshold: float = 0.99
    threshold_smoothing: float = 1e-3
    mean_weight: float = 0.25
    threshold_weight: float = 0.60
    cvar_weight: float = 0.15
    cvar_confidence: float = 0.95
    update_count: int = 100
    amplitude_learning_rate: float = 0.02
    phase_learning_rate: float = 0.02
    maximum_amplitude_coefficient_step: float = 0.02
    maximum_phase_coefficient_step_rad: float = 0.10
    scope_timeout_ms: int = 30000
    trigger_timeout_s: float = 30.0
    armed_timeout_s: float = 2.0
    trigger_poll_s: float = 0.05
    launch_tnmr_if_needed: bool = False
    maximum_amplitude_fraction: float = 1.0
    initial_amplitude_coefficients: tuple[float, ...] | None = None
    initial_phase_coefficients_rad: tuple[float, ...] | None = None

    def validate(self) -> None:
        if not self.tnt_path.exists():
            raise ValueError(f"TNMR document does not exist: {self.tnt_path}")
        if not self.amplitude_table_name.strip() or not self.phase_table_name.strip():
            raise ValueError("TNMR table names must not be empty.")
        if self.amplitude_table_name == self.phase_table_name:
            raise ValueError("Amplitude and phase table names must be different.")
        if self.acquisition_window_end_s <= self.acquisition_window_start_s:
            raise ValueError("Scope acquisition window end must follow its start.")
        if self.pulse_duration_s <= 0.0:
            raise ValueError("pulse_duration_s must be positive.")
        if self.pulse_duration_s > (
            self.acquisition_window_end_s - self.acquisition_window_start_s
        ):
            raise ValueError("Pulse duration does not fit inside the scope window.")
        if self.q_sign not in (-1, 1):
            raise ValueError("q_sign must be +1 for I+iQ or -1 for I-iQ.")
        if self.scope_omega_rad_s_per_volt <= 0.0:
            raise ValueError("scope_omega_rad_s_per_volt must be positive.")
        if self.omega_at_amplitude_100_rad_s <= 0.0:
            raise ValueError("omega_at_amplitude_100_rad_s must be positive.")
        if self.coefficient_count <= self.spline_degree:
            raise ValueError("coefficient_count must exceed spline_degree.")
        if self.table_entry_count < self.coefficient_count:
            raise ValueError("table_entry_count must not be smaller than coefficient_count.")
        if self.phase_reference_captures < 2:
            raise ValueError("At least two phase-reference captures are required.")
        if self.update_count < 0:
            raise ValueError("update_count must be nonnegative.")
        if self.pulse_start_s is not None:
            pulse_end_s = self.pulse_start_s + self.pulse_duration_s
            if self.pulse_start_s < self.acquisition_window_start_s:
                raise ValueError("Configured pulse start precedes the scope window.")
            if pulse_end_s > self.acquisition_window_end_s:
                raise ValueError("Configured pulse end exceeds the scope window.")
        if not 0 < self.maximum_amplitude_fraction <= 1:
            raise ValueError("maximum_amplitude_fraction must be in (0, 1]")
        if (self.initial_amplitude_coefficients is None) != (self.initial_phase_coefficients_rad is None):
            raise ValueError("Provide both initial amplitude and phase coefficients")
        if self.initial_amplitude_coefficients is not None:
            a = np.asarray(self.initial_amplitude_coefficients, dtype=float)
            p = np.asarray(self.initial_phase_coefficients_rad, dtype=float)
            if a.shape != (self.coefficient_count,) or p.shape != a.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(p)):
                raise ValueError("Initial coefficient arrays must be finite and match coefficient_count")
            if np.min(a) < 0 or np.max(a) > self.maximum_amplitude_fraction:
                raise ValueError("Initial coefficients exceed the configured amplitude limit")
        self.robustness_config().validate()

    def robustness_config(self) -> RobustnessConfig:
        return RobustnessConfig(
            omega_reference_rad_s=self.omega_at_amplitude_100_rad_s,
            amplitude_errors=np.asarray(self.amplitude_errors, dtype=float),
            detuning_ratios=np.asarray(self.detuning_ratios, dtype=float),
            fidelity_threshold=self.fidelity_threshold,
            threshold_smoothing=self.threshold_smoothing,
            mean_weight=self.mean_weight,
            threshold_weight=self.threshold_weight,
            cvar_weight=self.cvar_weight,
            cvar_confidence=self.cvar_confidence,
        )


@dataclass(frozen=True)
class CapturedControls:
    """One demodulated capture resampled onto the TNMR table grid."""

    raw: ScopeWaveform
    demodulated: DemodulatedWaveform
    pulse_window: PulseWindow
    entry_time_s: np.ndarray
    entry_dt_s: np.ndarray
    complex_voltage: np.ndarray


@dataclass(frozen=True)
class ClosedLoopResult:
    """Summary of a completed run whose best measured controls were restored."""

    run_directory: Path
    best_epoch: int
    best_loss: float
    phase_reference: PhaseReference
    pulse_start_s: float
    history: tuple[dict[str, float | int], ...]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _create_run_directory(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"xpi_closed_loop_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"xpi_closed_loop_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def format_table_entries(values: np.ndarray) -> list[str]:
    """Format TNMR entries as fixed-point decimals with stable readback."""

    numeric = np.asarray(values, dtype=float).reshape(-1)
    if numeric.size < 1 or not np.all(np.isfinite(numeric)):
        raise ValueError("TNMR table entries must be finite and nonempty.")
    return [f"{value:.12f}" for value in numeric]


def parse_numeric_table(value: Any) -> np.ndarray:
    """Parse TNMR GetTable output into a flat numeric array."""

    if isinstance(value, str):
        _prefix, separator, table_text = value.partition("=")
        tokens = _NUMBER_RE.findall(table_text if separator else value)
    else:
        try:
            tokens = list(value)
        except TypeError:
            tokens = [value]
    if not tokens:
        raise ValueError("TNMR returned an empty table.")
    try:
        parsed = np.asarray([float(token) for token in tokens], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("TNMR returned a nonnumeric table.") from exc
    if not np.all(np.isfinite(parsed)):
        raise ValueError("TNMR returned nonfinite table entries.")
    return parsed


def upload_tables_with_readback(
    tnmr: Any,
    *,
    amplitude_table_name: str,
    phase_table_name: str,
    amplitude_percent: np.ndarray,
    phase_degrees: np.ndarray,
) -> None:
    """Upload both TNMR tables and require numeric round-trip agreement."""

    amplitude = np.asarray(amplitude_percent, dtype=float).reshape(-1)
    phase = np.asarray(phase_degrees, dtype=float).reshape(-1)
    if amplitude.shape != phase.shape:
        raise ValueError("TNMR amplitude and phase tables must have the same length.")
    if np.min(amplitude) < 0.0 or np.max(amplitude) > 100.0:
        raise ValueError("TNMR amplitude table lies outside [0, 100].")
    available = set(tnmr.get_table_list())
    missing = {amplitude_table_name, phase_table_name} - available
    if missing:
        raise RuntimeError(f"TNMR document is missing tables: {sorted(missing)}")

    amplitude_lines = format_table_entries(amplitude)
    phase_lines = format_table_entries(phase)
    tnmr.set_table(amplitude_table_name, amplitude_lines)
    tnmr.set_table(phase_table_name, phase_lines)
    amplitude_readback = parse_numeric_table(tnmr.get_table(amplitude_table_name))
    phase_readback = parse_numeric_table(tnmr.get_table(phase_table_name))
    if amplitude_readback.shape != amplitude.shape or not np.allclose(
        amplitude_readback,
        amplitude,
        rtol=0.0,
        atol=5e-12,
    ):
        raise RuntimeError("TNMR amplitude-table readback does not match the upload.")
    if phase_readback.shape != phase.shape or not np.allclose(
        phase_readback,
        phase,
        rtol=0.0,
        atol=5e-12,
    ):
        raise RuntimeError("TNMR phase-table readback does not match the upload.")


def _capture_and_demodulate(
    scope: Any,
    tnmr: Any,
    config: ClosedLoopConfig,
    *,
    frozen_pulse_start_s: float | None,
) -> tuple[CapturedControls, float]:
    raw = scope.capture_single_waveform(
        WaveformCaptureConfig(
            channel=config.scope_channel,
            window_start_s=config.acquisition_window_start_s,
            window_end_s=config.acquisition_window_end_s,
            allow_clamp=False,
        ),
        start_shot=tnmr.zero_and_go,
        trigger_timeout_s=config.trigger_timeout_s,
        armed_timeout_s=config.armed_timeout_s,
        poll_s=config.trigger_poll_s,
        confirm_armed=True,
        rearm_after_read=False,
    )
    demodulated = demodulate_raw_rf(
        raw,
        demodulation_freq_hz=config.demodulation_frequency_hz,
        baseband_cutoff_hz=config.baseband_cutoff_hz,
        filter_order=config.demodulation_filter_order,
        q_sign=config.q_sign,
    )
    pulse_start_s = frozen_pulse_start_s
    if pulse_start_s is None:
        detected = detect_pulse_window(demodulated)
        detected_center_s = 0.5 * (detected.start_time + detected.end_time)
        pulse_start_s = detected_center_s - 0.5 * config.pulse_duration_s
    pulse_end_s = pulse_start_s + config.pulse_duration_s
    if pulse_start_s < demodulated.time[0] or pulse_end_s > demodulated.time[-1]:
        raise RuntimeError(
            "The fixed pulse interval is not fully contained in the captured waveform."
        )
    window = detect_pulse_window(
        demodulated,
        manual_start=pulse_start_s,
        manual_end=pulse_end_s,
    )
    edges = np.linspace(
        pulse_start_s,
        pulse_end_s,
        config.table_entry_count + 1,
    )
    entry_time_s = 0.5 * (edges[:-1] + edges[1:])
    complex_voltage = np.interp(
        entry_time_s,
        demodulated.time,
        demodulated.i,
    ) + 1.0j * np.interp(
        entry_time_s,
        demodulated.time,
        demodulated.q,
    )
    return (
        CapturedControls(
            raw=raw,
            demodulated=demodulated,
            pulse_window=window,
            entry_time_s=entry_time_s,
            entry_dt_s=np.diff(edges),
            complex_voltage=complex_voltage,
        ),
        float(pulse_start_s),
    )


def _normalized_measured_controls(
    capture: CapturedControls,
    phase_reference: PhaseReference,
    config: ClosedLoopConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrected_voltage = apply_phase_reference(
        capture.complex_voltage,
        phase_reference,
    )
    normalized = (
        config.scope_omega_rad_s_per_volt
        * corrected_voltage
        / config.omega_at_amplitude_100_rad_s
    )
    return normalized.real, normalized.imag, corrected_voltage


def _history_row(epoch: int, evaluation: GrapeEvaluation) -> dict[str, float | int]:
    objective = evaluation.objective
    return {
        "epoch": epoch,
        "loss": objective.loss,
        "mean_fidelity": objective.mean_fidelity,
        "cvar_fidelity": objective.cvar_fidelity,
        "worst_fidelity": objective.worst_fidelity,
        "center_fidelity": objective.center_fidelity,
        "fraction_at_or_above_threshold": (
            objective.fraction_at_or_above_threshold
        ),
    }


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _plot_heatmap(
    path: Path,
    evaluation: GrapeEvaluation,
    config: ClosedLoopConfig,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    amplitude_axis = np.asarray(config.amplitude_errors, dtype=float)
    detuning_axis = np.asarray(config.detuning_ratios, dtype=float) * config.omega_at_amplitude_100_rad_s / (2 * math.pi * 1000)
    amplitude_extent = (
        (float(amplitude_axis[0] - 0.5), float(amplitude_axis[0] + 0.5))
        if len(amplitude_axis) == 1
        else (float(amplitude_axis[0]), float(amplitude_axis[-1]))
    )
    detuning_extent = (
        (float(detuning_axis[0] - 0.5), float(detuning_axis[0] + 0.5))
        if len(detuning_axis) == 1
        else (float(detuning_axis[0]), float(detuning_axis[-1]))
    )
    fig, ax = subplots(figsize=(7.5, 6.0))
    image = ax.imshow(
        evaluation.objective.fidelity,
        origin="lower",
        aspect="auto",
        extent=(
            *amplitude_extent,
            *detuning_extent,
        ),
        vmin=min(0.99, float(np.min(evaluation.objective.fidelity))),
        vmax=1.0,
        cmap="viridis",
    )
    ax.set(
        xlabel="Amplitude error",
        ylabel="Detuning (kHz)",
        title=title,
    )
    fig.colorbar(image, ax=ax, label="Process fidelity to fixed X(pi)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _aligned_unwrapped_phase(
    values: np.ndarray,
    reference_phase_rad: np.ndarray,
) -> np.ndarray:
    """Unwrap measured phase and choose the nearest 2*pi branch to a reference."""

    phase = np.unwrap(np.angle(np.asarray(values, dtype=np.complex128)))
    reference = np.asarray(reference_phase_rad, dtype=float)
    branch_shift = 2.0 * math.pi * round(
        float(np.median(reference - phase)) / (2.0 * math.pi)
    )
    return phase + branch_shift


def _phase_residual_degrees(
    measured_phase_rad: np.ndarray,
    reference_phase_rad: np.ndarray,
) -> np.ndarray:
    residual = np.angle(
        np.exp(
            1.0j
            * (
                np.asarray(measured_phase_rad, dtype=float)
                - np.asarray(reference_phase_rad, dtype=float)
            )
        )
    )
    return np.rad2deg(residual)


def _draw_bb1_boundaries(ax: Any, time_us: np.ndarray) -> None:
    start = float(time_us[0])
    duration = float(time_us[-1] - time_us[0])
    for fraction in (0.1, 0.3, 0.7, 0.9):
        ax.axvline(start + fraction * duration, color="0.65", linestyle="--", linewidth=0.8)


def _save_stage1_diagnostics(
    run_directory: Path,
    *,
    reference_captures: np.ndarray,
    phase_reference: PhaseReference,
    amplitude_percent: np.ndarray,
    command_phase_unwrapped_rad: np.ndarray,
    capture: CapturedControls,
    corrected_voltage: np.ndarray,
    normalized_i: np.ndarray,
    normalized_q: np.ndarray,
    evaluation: GrapeEvaluation,
    config: ClosedLoopConfig,
) -> None:
    """Save the standalone Stage-1 phase, final-pulse, heatmap, and summary graphs."""

    import matplotlib.pyplot as plt

    reference_values = np.asarray(reference_captures, dtype=np.complex128)
    command_phase = np.asarray(command_phase_unwrapped_rad, dtype=float)
    time_us = (capture.entry_time_s - capture.entry_time_s[0]) * 1e6
    raw_phase = _aligned_unwrapped_phase(capture.complex_voltage, command_phase)
    corrected_phase = _aligned_unwrapped_phase(corrected_voltage, command_phase)
    corrected_reference_phases = np.asarray(
        [
            _aligned_unwrapped_phase(
                apply_phase_reference(reference_capture, phase_reference),
                command_phase,
            )
            for reference_capture in reference_values
        ]
    )

    fig, axes = subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(
        time_us,
        np.rad2deg(command_phase),
        color="black",
        linewidth=1.6,
        label="Commanded BB1",
    )
    axes[0].plot(
        time_us,
        np.rad2deg(raw_phase),
        color="tab:orange",
        linewidth=1.0,
        label="Measured before correction",
    )
    axes[0].plot(
        time_us,
        np.rad2deg(corrected_phase),
        color="tab:blue",
        linewidth=1.0,
        label="Measured after correction",
    )
    axes[0].set_ylabel("Unwrapped phase (deg)")
    axes[0].set_title(
        "Stage 1 phase calibration: "
        f"measured offset {math.degrees(phase_reference.offset_rad):+.6g} deg, "
        "analysis-frame correction "
        f"{-math.degrees(phase_reference.offset_rad):+.6g} deg"
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    for reference_phase in corrected_reference_phases:
        axes[1].plot(
            time_us,
            _phase_residual_degrees(reference_phase, command_phase),
            color="0.75",
            linewidth=0.6,
            alpha=0.7,
        )
    axes[1].plot(
        time_us,
        _phase_residual_degrees(raw_phase, command_phase),
        color="tab:orange",
        linewidth=1.0,
        label="Before correction residual",
    )
    axes[1].plot(
        time_us,
        _phase_residual_degrees(corrected_phase, command_phase),
        color="tab:blue",
        linewidth=1.0,
        label="After correction residual",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Time within pulse (us)",
        ylabel="Phase residual (deg)",
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    for axis in axes:
        _draw_bb1_boundaries(axis, time_us)
    fig.tight_layout()
    fig.savefig(run_directory / "stage1_phase_change.png", dpi=180)
    plt.close(fig)

    measured_amplitude_percent = 100.0 * np.hypot(normalized_i, normalized_q)
    reference_amplitude_percent = (
        100.0
        * config.scope_omega_rad_s_per_volt
        * np.abs(reference_values)
        / config.omega_at_amplitude_100_rad_s
    )
    reference_amplitude_mean = np.mean(reference_amplitude_percent, axis=0)
    reference_amplitude_std = np.std(reference_amplitude_percent, axis=0)
    measured_area_rad = float(
        np.sum(
            np.hypot(normalized_i, normalized_q)
            * config.omega_at_amplitude_100_rad_s
            * capture.entry_dt_s
        )
    )
    area_ratio = measured_area_rad / (5.0 * math.pi)

    fig, axes = subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].fill_between(
        time_us,
        reference_amplitude_mean - reference_amplitude_std,
        reference_amplitude_mean + reference_amplitude_std,
        color="tab:blue",
        alpha=0.18,
        label="Reference captures +/- 1 std",
    )
    axes[0].plot(
        time_us,
        reference_amplitude_mean,
        color="tab:blue",
        linestyle="--",
        linewidth=0.9,
        label="Reference-capture mean",
    )
    axes[0].plot(
        time_us,
        amplitude_percent,
        color="black",
        linewidth=1.5,
        label="Commanded amplitude",
    )
    axes[0].plot(
        time_us,
        measured_amplitude_percent,
        color="tab:blue",
        linewidth=1.0,
        label="Independent measured amplitude",
    )
    axes[0].set(
        ylabel="Amplitude (% of TNMR amplitude 100)",
        title=f"Stage 1 calibrated final pulse; measured area / (5*pi) = {area_ratio:.9f}",
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    for reference_phase in corrected_reference_phases:
        axes[1].plot(
            time_us,
            np.rad2deg(reference_phase),
            color="0.75",
            linewidth=0.6,
            alpha=0.7,
        )
    axes[1].plot(
        time_us,
        np.rad2deg(command_phase),
        color="black",
        linewidth=1.5,
        label="Commanded BB1 phase",
    )
    axes[1].plot(
        time_us,
        np.rad2deg(corrected_phase),
        color="tab:blue",
        linewidth=1.0,
        label="Independent corrected phase",
    )
    axes[1].set(
        xlabel="Time within pulse (us)",
        ylabel="Unwrapped phase (deg)",
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    for axis in axes:
        _draw_bb1_boundaries(axis, time_us)
    fig.tight_layout()
    fig.savefig(run_directory / "stage1_final_pulse.png", dpi=180)
    plt.close(fig)

    _plot_heatmap(
        run_directory / "stage1_final_heatmap.png",
        evaluation,
        config,
        title="Stage 1 independent measured pulse vs fixed X(pi)",
    )

    reference_areas_rad = np.sum(
        reference_amplitude_percent
        / 100.0
        * config.omega_at_amplitude_100_rad_s
        * capture.entry_dt_s[None, :],
        axis=1,
    )
    reference_area_cv = float(
        np.std(reference_areas_rad) / np.mean(reference_areas_rad)
    )
    _write_json(
        run_directory / "stage1_summary.json",
        {
            "automatic_phase_checks_passed": True,
            "full_stage1_acceptance_requires_graph_review": True,
            "measured_phase_offset_rad": phase_reference.offset_rad,
            "measured_phase_offset_deg": math.degrees(phase_reference.offset_rad),
            "applied_analysis_phase_change_deg": -math.degrees(phase_reference.offset_rad),
            "first_last_disagreement_deg": math.degrees(
                phase_reference.segment_disagreement_rad
            ),
            "capture_phase_spread_deg": math.degrees(
                phase_reference.capture_spread_rad
            ),
            "bb1_segment_residuals_deg": np.rad2deg(
                phase_reference.segment_residuals_rad
            ),
            "maximum_allowed_first_last_disagreement_deg": math.degrees(
                config.phase_reference_maximum_segment_disagreement_rad
            ),
            "maximum_allowed_capture_phase_spread_deg": math.degrees(
                config.phase_reference_maximum_capture_spread_rad
            ),
            "maximum_allowed_bb1_pattern_residual_deg": math.degrees(
                config.phase_reference_maximum_bb1_pattern_residual_rad
            ),
            "reference_area_rad": reference_areas_rad,
            "reference_area_coefficient_of_variation": reference_area_cv,
            "independent_measured_area_rad": measured_area_rad,
            "independent_measured_area_over_5pi": area_ratio,
            "center_fidelity_to_fixed_xpi": evaluation.objective.center_fidelity,
            "mean_fidelity_to_fixed_xpi": evaluation.objective.mean_fidelity,
            "worst_fidelity_to_fixed_xpi": evaluation.objective.worst_fidelity,
            "fraction_at_or_above_threshold": (
                evaluation.objective.fraction_at_or_above_threshold
            ),
            "fidelity_threshold": config.fidelity_threshold,
        },
    )


def _save_candidate(
    run_directory: Path,
    label: str,
    *,
    epoch: int,
    amplitude_coefficients: np.ndarray,
    phase_coefficients_rad: np.ndarray,
    amplitude_percent: np.ndarray,
    phase_degrees: np.ndarray,
    command_phase_unwrapped_rad: np.ndarray,
    capture: CapturedControls,
    corrected_voltage: np.ndarray,
    normalized_i: np.ndarray,
    normalized_q: np.ndarray,
    evaluation: GrapeEvaluation,
    config: ClosedLoopConfig,
) -> None:
    np.savez_compressed(
        run_directory / f"{label}.npz",
        epoch=epoch,
        amplitude_coefficients=amplitude_coefficients,
        phase_coefficients_rad=phase_coefficients_rad,
        amplitude_percent=amplitude_percent,
        phase_degrees=phase_degrees,
        command_phase_unwrapped_rad=command_phase_unwrapped_rad,
        entry_time_s=capture.entry_time_s,
        entry_dt_s=capture.entry_dt_s,
        captured_complex_voltage=capture.complex_voltage,
        corrected_complex_voltage=corrected_voltage,
        normalized_i=normalized_i,
        normalized_q=normalized_q,
        amplitude_errors=config.amplitude_errors,
        detuning_ratios=config.detuning_ratios,
        detuning_hz=np.asarray(config.detuning_ratios) * config.omega_at_amplitude_100_rad_s / (2 * math.pi),
        fidelity=evaluation.objective.fidelity,
        loss=evaluation.objective.loss,
    )
    np.savetxt(
        run_directory / f"{label}_amplitude_percent.csv",
        amplitude_percent,
        delimiter=",",
        header="amplitude_percent",
        comments="",
        fmt="%.12f",
    )
    np.savetxt(
        run_directory / f"{label}_phase_degrees.csv",
        phase_degrees,
        delimiter=",",
        header="phase_degrees",
        comments="",
        fmt="%.12f",
    )
    _plot_heatmap(
        run_directory / f"{label}_heatmap.png",
        evaluation,
        config,
        title=f"{label.replace('_', ' ').title()} (epoch {epoch})",
    )


def _write_checkpoint(
    path: Path,
    *,
    epoch: int,
    amplitude_coefficients: np.ndarray,
    phase_coefficients_rad: np.ndarray,
    amplitude_adam: AdamState | None,
    phase_adam: AdamState | None,
    best_epoch: int,
    best_loss: float,
    best_amplitude_coefficients: np.ndarray,
    best_phase_coefficients_rad: np.ndarray,
    phase_reference: PhaseReference,
    pulse_start_s: float,
) -> None:
    np.savez_compressed(
        path,
        epoch=epoch,
        amplitude_coefficients=amplitude_coefficients,
        phase_coefficients_rad=phase_coefficients_rad,
        amplitude_adam_first=(
            np.asarray([]) if amplitude_adam is None else amplitude_adam.first_moment
        ),
        amplitude_adam_second=(
            np.asarray([]) if amplitude_adam is None else amplitude_adam.second_moment
        ),
        amplitude_adam_step=(0 if amplitude_adam is None else amplitude_adam.step),
        phase_adam_first=(
            np.asarray([]) if phase_adam is None else phase_adam.first_moment
        ),
        phase_adam_second=(
            np.asarray([]) if phase_adam is None else phase_adam.second_moment
        ),
        phase_adam_step=(0 if phase_adam is None else phase_adam.step),
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_amplitude_coefficients=best_amplitude_coefficients,
        best_phase_coefficients_rad=best_phase_coefficients_rad,
        phase_offset_rad=phase_reference.offset_rad,
        pulse_start_s=pulse_start_s,
    )


def _close_quietly(device: Any) -> None:
    try:
        device.close()
    except Exception:
        pass


def run_closed_loop_optimizer(
    config: ClosedLoopConfig,
    *,
    tnmr: Any | None = None,
    scope: Any | None = None,
) -> ClosedLoopResult:
    """Run BB1 phase referencing, then measured-waveform GRAPE/Adam updates.

    The GRAPE gradient is exact with respect to the captured I/Q waveform. Its
    conversion into the next commanded amplitude/phase table deliberately uses
    a straight-through (identity) hardware Jacobian. The next capture closes
    that modeling gap before another update is accepted.
    """

    config.validate()
    run_directory = _create_run_directory(config.results_root)
    _write_json(run_directory / "run_config.json", asdict(config))
    print(f"Closed-loop run directory: {run_directory}")

    spline = clamped_bspline_design_matrix(
        config.table_entry_count,
        config.coefficient_count,
        config.spline_degree,
    )
    initialization = bb1_initialization(
        config.coefficient_count,
        pulse_duration_s=config.pulse_duration_s,
        omega_at_amplitude_100_rad_s=config.omega_at_amplitude_100_rad_s,
        degree=config.spline_degree,
    )
    if initialization.amplitude_fraction > config.maximum_amplitude_fraction + 1e-12:
        raise ValueError("BB1 phase reference exceeds the configured amplitude limit; increase duration")
    amplitude_coefficients = initialization.amplitude_coefficients.copy()
    phase_coefficients_rad = initialization.phase_coefficients_rad.copy()
    amplitude_percent, phase_degrees, command_phase_rad = command_tables(
        amplitude_coefficients,
        phase_coefficients_rad,
        spline,
    )

    owns_tnmr = tnmr is None
    owns_scope = scope is None
    original_amplitude_table: Any | None = None
    original_phase_table: Any | None = None
    completed = False
    try:
        if tnmr is None:
            tnmr = TNMRControl(
                connect=True,
                launch=config.launch_tnmr_if_needed,
            )
            tnmr.open(config.tnt_path)
        if scope is None:
            scope = RigolDHO(resource=config.visa_resource, timeout_ms=config.scope_timeout_ms)
            scope.open()

        available_tables = set(tnmr.get_table_list())
        required_tables = {config.amplitude_table_name, config.phase_table_name}
        missing_tables = required_tables - available_tables
        if missing_tables:
            raise RuntimeError(
                f"TNMR document is missing tables: {sorted(missing_tables)}"
            )
        original_amplitude_table = tnmr.get_table(config.amplitude_table_name)
        original_phase_table = tnmr.get_table(config.phase_table_name)
        upload_tables_with_readback(
            tnmr,
            amplitude_table_name=config.amplitude_table_name,
            phase_table_name=config.phase_table_name,
            amplitude_percent=amplitude_percent,
            phase_degrees=phase_degrees,
        )
        print(
            "Uploaded split-BB1 initialization and verified "
            f"{config.table_entry_count} amplitude/phase entries."
        )

        frozen_pulse_start_s = config.pulse_start_s
        reference_captures: list[np.ndarray] = []
        for capture_index in range(config.phase_reference_captures):
            print(
                "Phase-reference capture "
                f"{capture_index + 1}/{config.phase_reference_captures}..."
            )
            captured, frozen_pulse_start_s = _capture_and_demodulate(
                scope,
                tnmr,
                config,
                frozen_pulse_start_s=frozen_pulse_start_s,
            )
            reference_captures.append(captured.complex_voltage)
        phase_reference = estimate_phase_reference(
            np.asarray(reference_captures),
            edge_trim_fraction=config.phase_reference_edge_trim_fraction,
            minimum_amplitude_fraction=(
                config.phase_reference_minimum_amplitude_fraction
            ),
            maximum_segment_disagreement_rad=(
                config.phase_reference_maximum_segment_disagreement_rad
            ),
            maximum_capture_spread_rad=(
                config.phase_reference_maximum_capture_spread_rad
            ),
            maximum_bb1_pattern_residual_rad=(
                config.phase_reference_maximum_bb1_pattern_residual_rad
            ),
        )
        _write_json(
            run_directory / "stage1_phase_reference.json",
            asdict(phase_reference),
        )
        np.savez_compressed(
            run_directory / "stage1_reference_captures.npz",
            complex_voltage=np.asarray(reference_captures),
            pulse_start_s=frozen_pulse_start_s,
            pulse_duration_s=config.pulse_duration_s,
            phase_offset_rad=phase_reference.offset_rad,
        )
        print(
            "Frozen BB1 analysis-frame offset: "
            f"{phase_reference.offset_rad:+.12g} rad; first/last disagreement "
            f"{math.degrees(phase_reference.segment_disagreement_rad):.6g} deg."
        )

        # Verify the reference independently before switching to a requested seed.
        if config.initial_amplitude_coefficients is not None:
            reference, _ = _capture_and_demodulate(scope, tnmr, config, frozen_pulse_start_s=frozen_pulse_start_s)
            ref_i, ref_q, corrected = _normalized_measured_controls(reference, phase_reference, config)
            ref_evaluation = robustness_grape_gradient(ref_i, ref_q, reference.entry_dt_s, config.robustness_config())
            _save_stage1_diagnostics(run_directory, reference_captures=np.asarray(reference_captures), phase_reference=phase_reference, amplitude_percent=amplitude_percent, command_phase_unwrapped_rad=command_phase_rad, capture=reference, corrected_voltage=corrected, normalized_i=ref_i, normalized_q=ref_q, evaluation=ref_evaluation, config=config)
            amplitude_coefficients = np.asarray(config.initial_amplitude_coefficients).copy()
            phase_coefficients_rad = np.asarray(config.initial_phase_coefficients_rad).copy()
            amplitude_percent, phase_degrees, command_phase_rad = command_tables(amplitude_coefficients, phase_coefficients_rad, spline)
            upload_tables_with_readback(tnmr, amplitude_table_name=config.amplitude_table_name, phase_table_name=config.phase_table_name, amplitude_percent=amplitude_percent, phase_degrees=phase_degrees)

        robustness_config = config.robustness_config()
        history: list[dict[str, float | int]] = []
        amplitude_adam: AdamState | None = None
        phase_adam: AdamState | None = None
        best_epoch = -1
        best_loss = math.inf
        best_amplitude_coefficients = amplitude_coefficients.copy()
        best_phase_coefficients_rad = phase_coefficients_rad.copy()

        for epoch in range(config.update_count + 1):
            captured, measured_pulse_start_s = _capture_and_demodulate(
                scope,
                tnmr,
                config,
                frozen_pulse_start_s=frozen_pulse_start_s,
            )
            if not math.isclose(
                measured_pulse_start_s,
                frozen_pulse_start_s,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError("The frozen pulse start changed during optimization.")
            normalized_i, normalized_q, corrected_voltage = (
                _normalized_measured_controls(captured, phase_reference, config)
            )
            evaluation = robustness_grape_gradient(
                normalized_i,
                normalized_q,
                captured.entry_dt_s,
                robustness_config,
            )
            history.append(_history_row(epoch, evaluation))
            _write_history(run_directory / "history.csv", history)
            print(
                f"Epoch {epoch:04d}/{config.update_count:04d}: "
                f"loss={evaluation.objective.loss:.12g}, "
                f"center={evaluation.objective.center_fidelity:.9f}, "
                f"mean={evaluation.objective.mean_fidelity:.9f}, "
                f"worst={evaluation.objective.worst_fidelity:.9f}, "
                f">={config.fidelity_threshold:.3f}="
                f"{evaluation.objective.fraction_at_or_above_threshold:.3%}"
            )

            if epoch == 0:
                _save_candidate(
                    run_directory,
                    "initial_measured",
                    epoch=epoch,
                    amplitude_coefficients=amplitude_coefficients,
                    phase_coefficients_rad=phase_coefficients_rad,
                    amplitude_percent=amplitude_percent,
                    phase_degrees=phase_degrees,
                    command_phase_unwrapped_rad=command_phase_rad,
                    capture=captured,
                    corrected_voltage=corrected_voltage,
                    normalized_i=normalized_i,
                    normalized_q=normalized_q,
                    evaluation=evaluation,
                    config=config,
                )
                if config.initial_amplitude_coefficients is None:
                    _save_stage1_diagnostics(
                        run_directory,
                        reference_captures=np.asarray(reference_captures),
                        phase_reference=phase_reference,
                        amplitude_percent=amplitude_percent,
                        command_phase_unwrapped_rad=command_phase_rad,
                        capture=captured,
                        corrected_voltage=corrected_voltage,
                        normalized_i=normalized_i,
                        normalized_q=normalized_q,
                        evaluation=evaluation,
                        config=config,
                    )
            if evaluation.objective.loss < best_loss:
                best_epoch = epoch
                best_loss = evaluation.objective.loss
                best_amplitude_coefficients = amplitude_coefficients.copy()
                best_phase_coefficients_rad = phase_coefficients_rad.copy()
                _save_candidate(
                    run_directory,
                    "best_measured",
                    epoch=epoch,
                    amplitude_coefficients=amplitude_coefficients,
                    phase_coefficients_rad=phase_coefficients_rad,
                    amplitude_percent=amplitude_percent,
                    phase_degrees=phase_degrees,
                    command_phase_unwrapped_rad=command_phase_rad,
                    capture=captured,
                    corrected_voltage=corrected_voltage,
                    normalized_i=normalized_i,
                    normalized_q=normalized_q,
                    evaluation=evaluation,
                    config=config,
                )
            if epoch == config.update_count:
                _save_candidate(
                    run_directory,
                    "final_measured",
                    epoch=epoch,
                    amplitude_coefficients=amplitude_coefficients,
                    phase_coefficients_rad=phase_coefficients_rad,
                    amplitude_percent=amplitude_percent,
                    phase_degrees=phase_degrees,
                    command_phase_unwrapped_rad=command_phase_rad,
                    capture=captured,
                    corrected_voltage=corrected_voltage,
                    normalized_i=normalized_i,
                    normalized_q=normalized_q,
                    evaluation=evaluation,
                    config=config,
                )
                break

            amplitude_entry_gradient, phase_entry_gradient = polar_control_gradients(
                normalized_i,
                normalized_q,
                evaluation.i_gradient,
                evaluation.q_gradient,
            )
            amplitude_coefficient_gradient = np.asarray(
                spline.T @ amplitude_entry_gradient,
                dtype=float,
            )
            phase_coefficient_gradient = np.asarray(
                spline.T @ phase_entry_gradient,
                dtype=float,
            )
            amplitude_coefficients, amplitude_adam = adam_update(
                amplitude_coefficients,
                amplitude_coefficient_gradient,
                amplitude_adam,
                learning_rate=config.amplitude_learning_rate,
                maximum_step=config.maximum_amplitude_coefficient_step,
            )
            phase_coefficients_rad, phase_adam = adam_update(
                phase_coefficients_rad,
                phase_coefficient_gradient,
                phase_adam,
                learning_rate=config.phase_learning_rate,
                maximum_step=config.maximum_phase_coefficient_step_rad,
            )
            amplitude_coefficients = np.clip(amplitude_coefficients, 0.0, config.maximum_amplitude_fraction)
            amplitude_percent, phase_degrees, command_phase_rad = command_tables(
                amplitude_coefficients,
                phase_coefficients_rad,
                spline,
            )
            upload_tables_with_readback(
                tnmr,
                amplitude_table_name=config.amplitude_table_name,
                phase_table_name=config.phase_table_name,
                amplitude_percent=amplitude_percent,
                phase_degrees=phase_degrees,
            )
            _write_checkpoint(
                run_directory / "checkpoint_latest.npz",
                epoch=epoch + 1,
                amplitude_coefficients=amplitude_coefficients,
                phase_coefficients_rad=phase_coefficients_rad,
                amplitude_adam=amplitude_adam,
                phase_adam=phase_adam,
                best_epoch=best_epoch,
                best_loss=best_loss,
                best_amplitude_coefficients=best_amplitude_coefficients,
                best_phase_coefficients_rad=best_phase_coefficients_rad,
                phase_reference=phase_reference,
                pulse_start_s=frozen_pulse_start_s,
            )

        best_amplitude_percent, best_phase_degrees, _ = command_tables(
            best_amplitude_coefficients,
            best_phase_coefficients_rad,
            spline,
        )
        upload_tables_with_readback(
            tnmr,
            amplitude_table_name=config.amplitude_table_name,
            phase_table_name=config.phase_table_name,
            amplitude_percent=best_amplitude_percent,
            phase_degrees=best_phase_degrees,
        )
        _write_json(
            run_directory / "summary.json",
            {
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "phase_reference": asdict(phase_reference),
                "pulse_start_s": frozen_pulse_start_s,
                "restored_tnmr_tables": "best measured candidate",
                "gradient_hardware_jacobian": "straight-through identity",
                "target": "fixed ideal X(pi)",
            },
        )
        print(
            f"Restored best measured candidate from epoch {best_epoch} "
            f"(loss {best_loss:.12g}) to TNMR."
        )
        completed = True
        return ClosedLoopResult(
            run_directory=run_directory,
            best_epoch=best_epoch,
            best_loss=best_loss,
            phase_reference=phase_reference,
            pulse_start_s=frozen_pulse_start_s,
            history=tuple(history),
        )
    finally:
        if not completed and original_amplitude_table is not None:
            try:
                tnmr.set_table(config.amplitude_table_name, original_amplitude_table)
            except Exception:
                pass
        if not completed and original_phase_table is not None:
            try:
                tnmr.set_table(config.phase_table_name, original_phase_table)
            except Exception:
                pass
        if not completed:
            try:
                tnmr.stop()
            except Exception:
                pass
        if owns_scope:
            _close_quietly(scope)
        if owns_tnmr:
            _close_quietly(tnmr)
