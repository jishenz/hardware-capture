"""Reusable calibration bb1 algorithms. Use hardware_capture.api for workflows."""

from __future__ import annotations

import csv

import json

import math

from dataclasses import dataclass, replace

from datetime import datetime, timezone

from pathlib import Path
from hardware_capture.figures import subplots

from typing import Any, Mapping

import numpy as np

from hardware_capture.signal.demodulation import DemodulatedWaveform, PulseWindow, demodulate_raw_rf, detect_pulse_window

from hardware_capture.devices.rigol import ScopePreamble, ScopeWaveform, normalize_channel

DEFAULT_PULSE_THRESHOLD_FRACTION = 0.10

DEFAULT_PULSE_PADDING_SECONDS = 0.25e-6

MIN_BASELINE_SAMPLES = 20

@dataclass(frozen=True)
class DurationCrossCheck:
    """Optional constant-envelope check inferred from a primitive pi/2 duration."""

    pi_over_2_duration_s: float
    nominal_rabi_khz: float
    plateau_usable: bool
    measured_plateau_voltage_mv: float | None
    plateau_rabi_khz_per_mv: float | None
    relative_difference: float | None

@dataclass(frozen=True)
class BB1Calibration:
    """Area-derived calibration and the corrected baseband data behind it."""

    demodulated: DemodulatedWaveform
    initial_window: PulseWindow
    pulse_window: PulseWindow
    i_baseline_v: float
    q_baseline_v: float
    target_angle_rad: float
    bb1_total_area_rad: float
    voltage_envelope_area_v_s: float
    omega_rad_s_per_volt: float
    rabi_khz_per_mv: float
    peak_envelope_v: float
    median_envelope_v: float
    cumulative_area_v_s: np.ndarray
    duration_cross_check: DurationCrossCheck | None

def _validate_positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number

def _actual_sample_window(window: PulseWindow, time: np.ndarray) -> PulseWindow:
    return PulseWindow(
        start_time=float(time[window.start_index]),
        end_time=float(time[window.stop_index - 1]),
        start_index=window.start_index,
        stop_index=window.stop_index,
    )

def _validate_raw_waveform(raw: ScopeWaveform) -> None:
    time = np.asarray(raw.time, dtype=float)
    voltage = np.asarray(raw.voltage, dtype=float)
    if time.ndim != 1 or voltage.ndim != 1 or len(time) != len(voltage):
        raise ValueError("Scope time and voltage must be one-dimensional and equal length.")
    if len(time) < 3:
        raise ValueError("Scope waveform must contain at least three samples.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(voltage)):
        raise ValueError("Scope waveform contains non-finite time or voltage values.")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("Scope time values must be strictly increasing.")

def _trapezoid(values: np.ndarray, time: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(time)))

def _cumulative_trapezoid(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(time)
    return np.concatenate((np.asarray([0.0]), np.cumsum(increments)))

def _plateau_voltage_v(envelope_v: np.ndarray) -> float | None:
    """Return a robust plateau estimate only when most active samples are flat."""

    if len(envelope_v) < 20:
        return None
    peak = float(np.max(envelope_v))
    if not math.isfinite(peak) or peak <= 0.0:
        return None

    high = envelope_v[envelope_v >= 0.8 * peak]
    if len(high) < max(10, math.ceil(0.5 * len(envelope_v))):
        return None
    median = float(np.median(high))
    relative_spread = float(np.percentile(high, 90) - np.percentile(high, 10)) / median
    if not math.isfinite(median) or median <= 0.0 or relative_spread > 0.10:
        return None
    return median

def _duration_cross_check(
    envelope_v: np.ndarray,
    pi_over_2_duration_s: float | None,
    area_rabi_khz_per_mv: float,
) -> DurationCrossCheck | None:
    if pi_over_2_duration_s is None:
        return None

    duration_s = _validate_positive_finite(
        pi_over_2_duration_s,
        "pi_over_2_duration_s",
    )
    nominal_rabi_khz = 1.0 / (4.0 * duration_s) / 1_000.0
    plateau_v = _plateau_voltage_v(envelope_v)
    if plateau_v is None:
        return DurationCrossCheck(
            pi_over_2_duration_s=duration_s,
            nominal_rabi_khz=nominal_rabi_khz,
            plateau_usable=False,
            measured_plateau_voltage_mv=None,
            plateau_rabi_khz_per_mv=None,
            relative_difference=None,
        )

    plateau_mv = plateau_v * 1_000.0
    plateau_coefficient = nominal_rabi_khz / plateau_mv
    relative_difference = abs(
        plateau_coefficient - area_rabi_khz_per_mv
    ) / area_rabi_khz_per_mv
    return DurationCrossCheck(
        pi_over_2_duration_s=duration_s,
        nominal_rabi_khz=nominal_rabi_khz,
        plateau_usable=True,
        measured_plateau_voltage_mv=plateau_mv,
        plateau_rabi_khz_per_mv=plateau_coefficient,
        relative_difference=relative_difference,
    )

def calibrate_bb1_waveform(
    raw: ScopeWaveform,
    *,
    demodulation_freq_hz: float,
    baseband_cutoff_hz: float,
    q_sign: int = 1,
    filter_order: int = 4,
    target_angle_rad: float = 0.5 * math.pi,
    threshold_fraction: float = DEFAULT_PULSE_THRESHOLD_FRACTION,
    padding_seconds: float = DEFAULT_PULSE_PADDING_SECONDS,
    manual_start_s: float | None = None,
    manual_end_s: float | None = None,
    pi_over_2_duration_s: float | None = None,
) -> BB1Calibration:
    """Derive the RF conversion from a measured, complete BB1 envelope area."""

    _validate_raw_waveform(raw)
    target_angle_rad = _validate_positive_finite(target_angle_rad, "target_angle_rad")
    demodulation_freq_hz = _validate_positive_finite(
        demodulation_freq_hz,
        "demodulation_freq_hz",
    )
    baseband_cutoff_hz = _validate_positive_finite(
        baseband_cutoff_hz,
        "baseband_cutoff_hz",
    )
    if not math.isfinite(padding_seconds) or padding_seconds < 0.0:
        raise ValueError("padding_seconds must be finite and non-negative.")

    demodulated = demodulate_raw_rf(
        raw,
        demodulation_freq_hz=demodulation_freq_hz,
        baseband_cutoff_hz=baseband_cutoff_hz,
        q_sign=q_sign,
        filter_order=filter_order,
    )
    initial_window = detect_pulse_window(
        demodulated,
        threshold_fraction=threshold_fraction,
        padding_seconds=padding_seconds,
        manual_start=manual_start_s,
        manual_end=manual_end_s,
    )
    initial_window = _actual_sample_window(initial_window, demodulated.time)

    outside = np.ones(len(demodulated.time), dtype=bool)
    outside[initial_window.start_index : initial_window.stop_index] = False
    outside_samples = int(np.count_nonzero(outside))
    if outside_samples < MIN_BASELINE_SAMPLES:
        raise ValueError(
            "Insufficient out-of-pulse baseline data: "
            f"found {outside_samples} samples, need at least {MIN_BASELINE_SAMPLES}."
        )

    i_baseline_v = float(np.median(demodulated.i[outside]))
    q_baseline_v = float(np.median(demodulated.q[outside]))
    corrected_i = demodulated.i - i_baseline_v
    corrected_q = demodulated.q - q_baseline_v
    corrected_z = corrected_i + 1j * corrected_q
    corrected = replace(
        demodulated,
        i=corrected_i,
        q=corrected_q,
        z=corrected_z,
        amplitude=np.abs(corrected_z),
        phase=np.unwrap(np.angle(corrected_z)),
    )

    pulse_window = detect_pulse_window(
        corrected,
        threshold_fraction=threshold_fraction,
        padding_seconds=padding_seconds,
        manual_start=manual_start_s,
        manual_end=manual_end_s,
    )
    pulse_window = _actual_sample_window(pulse_window, corrected.time)
    active = slice(pulse_window.start_index, pulse_window.stop_index)
    active_time_s = corrected.time[active]
    active_envelope_v = corrected.amplitude[active]
    if len(active_time_s) < 2:
        raise ValueError("Final pulse window contains fewer than two samples.")

    envelope_area_v_s = _trapezoid(active_envelope_v, active_time_s)
    if not math.isfinite(envelope_area_v_s) or envelope_area_v_s <= 0.0:
        raise ValueError("Integrated corrected envelope area must be finite and positive.")

    bb1_total_area_rad = 4.0 * math.pi + target_angle_rad
    omega_rad_s_per_volt = bb1_total_area_rad / envelope_area_v_s
    rabi_khz_per_mv = omega_rad_s_per_volt / (2.0 * math.pi * 1e6)
    if not math.isfinite(omega_rad_s_per_volt) or omega_rad_s_per_volt <= 0.0:
        raise ValueError("omega_rad_s_per_volt must be finite and positive.")
    if not math.isfinite(rabi_khz_per_mv) or rabi_khz_per_mv <= 0.0:
        raise ValueError("rabi_khz_per_mv must be finite and positive.")

    cumulative_area_v_s = _cumulative_trapezoid(active_envelope_v, active_time_s)
    cross_check = _duration_cross_check(
        active_envelope_v,
        pi_over_2_duration_s,
        rabi_khz_per_mv,
    )
    return BB1Calibration(
        demodulated=corrected,
        initial_window=initial_window,
        pulse_window=pulse_window,
        i_baseline_v=i_baseline_v,
        q_baseline_v=q_baseline_v,
        target_angle_rad=target_angle_rad,
        bb1_total_area_rad=bb1_total_area_rad,
        voltage_envelope_area_v_s=envelope_area_v_s,
        omega_rad_s_per_volt=omega_rad_s_per_volt,
        rabi_khz_per_mv=rabi_khz_per_mv,
        peak_envelope_v=float(np.max(active_envelope_v)),
        median_envelope_v=float(np.median(active_envelope_v)),
        cumulative_area_v_s=cumulative_area_v_s,
        duration_cross_check=cross_check,
    )

def load_waveform_csv(path: Path | str) -> tuple[ScopeWaveform, dict[str, str]]:
    """Load a repository capture CSV, preserving leading metadata rows."""

    input_path = Path(path)
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[str]] = []
    with input_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or not any(item.strip() for item in row):
                continue
            first = row[0].strip()
            if header is None and first.startswith("#"):
                key = first[1:].strip()
                metadata[key] = ",".join(row[1:]).strip()
                continue
            if header is None:
                header = [item.strip() for item in row]
            else:
                rows.append(row)

    if header is None:
        raise ValueError(f"Waveform CSV {input_path} has no header row.")
    if "time_s" not in header:
        raise ValueError("Waveform CSV must contain a time_s column.")
    voltage_columns = [
        name for name in header if name.lower().endswith("voltage_v")
    ]
    if not voltage_columns:
        raise ValueError("Waveform CSV must contain a voltage column such as chan1_voltage_V.")
    if not rows:
        raise ValueError(f"Waveform CSV {input_path} contains no waveform samples.")

    time_index = header.index("time_s")
    voltage_name = voltage_columns[0]
    voltage_index = header.index(voltage_name)
    try:
        time = np.asarray([float(row[time_index]) for row in rows], dtype=float)
        voltage = np.asarray([float(row[voltage_index]) for row in rows], dtype=float)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse waveform samples from {input_path}.") from exc

    sample_rate = 1.0 / float(np.median(np.diff(time)))
    channel_text = metadata.get("channel") or voltage_name[: -len("_voltage_V")]
    try:
        channel = normalize_channel(channel_text)
    except ValueError:
        channel = str(channel_text).strip().upper() or "UNKNOWN"
    start_point = int(metadata.get("start_point", "1"))
    source = metadata.get("scope") or str(input_path)
    preamble = ScopePreamble(
        data_format="CSV",
        data_type="OFFLINE",
        points=len(time),
        count=1,
        x_increment=1.0 / sample_rate,
        x_origin=float(time[0]),
        x_reference=0.0,
        y_increment=1.0,
        y_origin=0.0,
        y_reference=0.0,
    )
    raw = ScopeWaveform(
        time=time,
        voltage=voltage,
        sample_rate=sample_rate,
        source=source,
        channel=channel,
        preamble=preamble,
        start_point=start_point,
        stop_point=start_point + len(time) - 1,
    )
    _validate_raw_waveform(raw)
    return raw, metadata

def calibration_to_dict(
    calibration: BB1Calibration,
    raw: ScopeWaveform,
    *,
    input_mode: str,
    q_convention: str,
    scope_resource: str | None = None,
    input_csv: Path | None = None,
    csv_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the versioned, reusable calibration artifact."""

    window = calibration.pulse_window
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "known_bb1_envelope_area",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": input_mode,
        "target_angle_rad": calibration.target_angle_rad,
        "bb1_total_area_rad": calibration.bb1_total_area_rad,
        "voltage_envelope_area_v_s": calibration.voltage_envelope_area_v_s,
        "voltage_envelope_area_mv_us": (
            calibration.voltage_envelope_area_v_s * 1e9
        ),
        "omega_rad_s_per_volt": calibration.omega_rad_s_per_volt,
        "rabi_khz_per_mv": calibration.rabi_khz_per_mv,
        "peak_envelope_voltage_mv": calibration.peak_envelope_v * 1_000.0,
        "median_envelope_voltage_mv": calibration.median_envelope_v * 1_000.0,
        "pulse_window": {
            "start_s": window.start_time,
            "end_s": window.end_time,
            "duration_s": window.duration,
        },
        "baseline": {
            "i_v": calibration.i_baseline_v,
            "q_v": calibration.q_baseline_v,
        },
        "demodulation": {
            "frequency_hz": calibration.demodulated.demodulation_freq_hz,
            "baseband_cutoff_hz": calibration.demodulated.baseband_cutoff_hz,
            "q_convention": q_convention,
        },
        "scope": {
            "source": raw.source,
            "resource": scope_resource,
            "channel": raw.channel,
        },
        "units": {
            "voltage_envelope_area_v_s": "V*s",
            "voltage_envelope_area_mv_us": "mV*us",
            "omega_rad_s_per_volt": "rad/s/V",
            "rabi_khz_per_mv": "kHz/mV for Omega/(2*pi)",
        },
        "voltage_definition": (
            "baseline-corrected demodulated peak-envelope voltage"
        ),
    }
    if input_csv is not None:
        payload["input_csv"] = str(input_csv)
        payload["csv_metadata"] = dict(csv_metadata or {})
    if calibration.duration_cross_check is not None:
        cross_check = calibration.duration_cross_check
        payload["pi_over_2_duration_cross_check"] = {
            "pi_over_2_duration_s": cross_check.pi_over_2_duration_s,
            "nominal_rabi_khz": cross_check.nominal_rabi_khz,
            "plateau_usable": cross_check.plateau_usable,
            "measured_plateau_voltage_mv": cross_check.measured_plateau_voltage_mv,
            "plateau_rabi_khz_per_mv": cross_check.plateau_rabi_khz_per_mv,
            "relative_difference": cross_check.relative_difference,
        }
    return payload

def write_calibration_json(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path:
    output_path = Path(path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Calibration output {output_path} already exists; pass --force to overwrite it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    with output_path.open(mode) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path

def _standard_bb1_boundaries(calibration: BB1Calibration) -> list[float]:
    cross_check = calibration.duration_cross_check
    if cross_check is None:
        return []
    pi_over_2_s = cross_check.pi_over_2_duration_s
    target_duration_s = (
        calibration.target_angle_rad / (0.5 * math.pi) * pi_over_2_s
    )
    offsets = np.cumsum(
        [target_duration_s, 2.0 * pi_over_2_s, 4.0 * pi_over_2_s, 2.0 * pi_over_2_s]
    )
    return [calibration.pulse_window.start_time + float(value) for value in offsets]

def plot_diagnostics(
    raw: ScopeWaveform,
    calibration: BB1Calibration,
    *,
    save_figure: bool,
    figure_path: Path,
    show_plot: bool,
) -> None:
    """Plot the raw trace, corrected baseband, area, and calibrated rotation."""

    import matplotlib.pyplot as plt

    demodulated = calibration.demodulated
    window = calibration.pulse_window
    active = slice(window.start_index, window.stop_index)
    active_time_s = demodulated.time[active]
    cumulative_rotation_rad = (
        calibration.omega_rad_s_per_volt * calibration.cumulative_area_v_s
    )
    figure, axes = subplots(5, 1, figsize=(11, 13), sharex=True, show=show_plot)
    time_us = raw.time * 1e6
    axes[0].plot(time_us, raw.voltage * 1e3, linewidth=0.8)
    axes[0].set_ylabel("Raw RF (mV)")
    axes[1].plot(time_us, demodulated.i * 1e3, label="I")
    axes[1].plot(time_us, demodulated.q * 1e3, label="Q")
    axes[1].set_ylabel("Baseband (mV)")
    axes[1].legend()
    axes[2].plot(time_us, demodulated.amplitude * 1e3, color="tab:purple")
    axes[2].axvspan(window.start_time * 1e6, window.end_time * 1e6, alpha=0.2)
    axes[2].set_ylabel("Envelope (mV)")
    axes[3].plot(active_time_s * 1e6, calibration.cumulative_area_v_s * 1e9)
    axes[3].set_ylabel("Cumulative area\n(mV*us)")
    axes[4].plot(active_time_s * 1e6, cumulative_rotation_rad)
    axes[4].axhline(
        calibration.bb1_total_area_rad,
        color="black",
        linestyle="--",
        label="Known BB1 area",
    )
    axes[4].set_ylabel("Cumulative rotation (rad)")
    axes[4].set_xlabel("Scope time (us)")
    axes[4].legend()

    for boundary_s in _standard_bb1_boundaries(calibration):
        for axis in axes:
            axis.axvline(boundary_s * 1e6, color="gray", linestyle=":", alpha=0.65)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    if save_figure:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=180)
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
