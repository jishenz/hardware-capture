"""Calibrate demodulated RF voltage from a TNMR amplitude-100 square pulse.

Generate a 10-20 us square pulse, set its TNMR amplitude to 100, and keep the
RF repetition rate low. This script auto-detects the pulse duration, discards
the head and tail, and uses the median of the middle plateau.

The printed ``SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT`` assignment is intended to be
pasted into ``scope_rf_robustness_heatmap.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_capture.oscilloscope.rf_demodulation import (
    DemodulatedWaveform,
    PulseWindow,
    demodulate_raw_rf,
)
from hardware_capture.oscilloscope.rigol_dho import (
    DEFAULT_ARMED_TIMEOUT_S,
    DEFAULT_TRIGGER_POLL_S,
    DEFAULT_TRIGGER_TIMEOUT_S,
    RigolDHO,
    ScopeWaveform,
    WaveformCaptureConfig,
    parse_time_seconds,
)

VISA_RESOURCE = "TCPIP::192.168.1.152::INSTR"
DEMODULATION_FREQ_HZ = 601.0100113e6
DEFAULT_WINDOW_START = "-10us"
DEFAULT_WINDOW_END = "80us"
DEFAULT_BASEBAND_CUTOFF_HZ = 25e6

# Omega/(2*pi) at TNMR amplitude 100. This is the tweakable physical reference.
# Use 123.96840469314181 kHz for the saved pulse whose actual optimized maximum
# was 123.96840469314181 kHz; 125 kHz was only that optimizer's configured cap.
PEAK_OMEGA_OVER_2PI_KHZ = 123.96840469314181  # 150.600407  #

# Analyze only the central 60% of the automatically detected square pulse.
MIDDLE_FRACTION = 0.60
DETECTION_SMOOTHING_SECONDS = 0.25e-6
DETECTION_THRESHOLD_FRACTION = 0.50
BASELINE_GUARD_SECONDS = 0.50e-6
MIN_BASELINE_SAMPLES = 20

OSCILLOSCOPE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = OSCILLOSCOPE_DIR / "results"
DEFAULT_OUTPUT_JSON = DEFAULT_RESULTS_DIR / "rf_square_calibration.json"
DEFAULT_FIGURE_PATH = DEFAULT_RESULTS_DIR / "rf_square_calibration_diagnostics.png"
DEFAULT_SAVE_FIGURE = True


@dataclass(frozen=True)
class SquarePulseCalibration:
    """Fixed voltage-to-Omega calibration derived from a square-pulse plateau."""

    demodulated: DemodulatedWaveform
    pulse_window: PulseWindow
    middle_window: PulseWindow
    i_baseline_v: float
    q_baseline_v: float
    peak_omega_over_2pi_khz: float
    middle_envelope_v: float
    middle_envelope_mad_v: float
    middle_relative_mad: float
    omega_rad_s_per_volt: float


def q_sign_from_convention(value: str) -> int:
    if value == "I+iQ":
        return 1
    if value == "I-iQ":
        return -1
    raise ValueError(f"Unsupported Q convention {value!r}.")


def _validate_positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _validate_raw_waveform(raw: ScopeWaveform) -> None:
    time = np.asarray(raw.time, dtype=float)
    voltage = np.asarray(raw.voltage, dtype=float)
    if time.ndim != 1 or voltage.ndim != 1 or len(time) != len(voltage):
        raise ValueError(
            "Scope time and voltage must be one-dimensional and equal length."
        )
    if len(time) < 3:
        raise ValueError("Scope waveform must contain at least three samples.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(voltage)):
        raise ValueError("Scope waveform contains non-finite time or voltage values.")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("Scope time values must be strictly increasing.")


def _smoothing_samples(sample_rate: float, smoothing_seconds: float) -> int:
    rate = _validate_positive_finite(sample_rate, "sample_rate")
    duration = _validate_positive_finite(
        smoothing_seconds,
        "detection_smoothing_seconds",
    )
    return max(1, int(round(rate * duration)))


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    if len(starts) == 0:
        raise ValueError("Could not detect a square-pulse interval.")
    longest = int(np.argmax(stops - starts))
    return int(starts[longest]), int(stops[longest])


def detect_square_pulse_window(
    demodulated: DemodulatedWaveform,
    *,
    smoothing_seconds: float = DETECTION_SMOOTHING_SECONDS,
    threshold_fraction: float = DETECTION_THRESHOLD_FRACTION,
) -> PulseWindow:
    """Detect the longest sustained high-envelope interval without using a raw peak."""

    fraction = float(threshold_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("threshold_fraction must be finite and in (0, 1).")

    envelope = np.asarray(demodulated.amplitude, dtype=float)
    if len(envelope) < 3 or not np.all(np.isfinite(envelope)):
        raise ValueError("Demodulated envelope must contain finite samples.")
    smooth_samples = _smoothing_samples(
        demodulated.sample_rate,
        smoothing_seconds,
    )
    smoothed = uniform_filter1d(envelope, size=smooth_samples, mode="nearest")

    baseline_level = float(np.percentile(smoothed, 20.0))
    high_level = float(np.percentile(smoothed, 95.0))
    contrast = high_level - baseline_level
    if not math.isfinite(contrast) or contrast <= 0.0:
        raise ValueError("Demodulated envelope has no positive square-pulse contrast.")

    threshold = baseline_level + fraction * contrast
    start_index, stop_index = _longest_true_run(smoothed >= threshold)
    if stop_index - start_index < 3:
        raise ValueError("Detected square pulse contains fewer than three samples.")

    time = demodulated.time
    return PulseWindow(
        start_time=float(time[start_index]),
        end_time=float(time[stop_index - 1]),
        start_index=start_index,
        stop_index=stop_index,
    )


def _baseline_correct(
    demodulated: DemodulatedWaveform,
    pulse_window: PulseWindow,
    *,
    guard_seconds: float,
) -> tuple[DemodulatedWaveform, float, float]:
    guard = float(guard_seconds)
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError("baseline_guard_seconds must be finite and non-negative.")
    guard_samples = int(round(guard * demodulated.sample_rate))
    excluded_start = max(0, pulse_window.start_index - guard_samples)
    excluded_stop = min(len(demodulated.time), pulse_window.stop_index + guard_samples)
    outside = np.ones(len(demodulated.time), dtype=bool)
    outside[excluded_start:excluded_stop] = False
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
    return corrected, i_baseline_v, q_baseline_v


def middle_pulse_window(
    pulse_window: PulseWindow,
    time: np.ndarray,
    *,
    middle_fraction: float = MIDDLE_FRACTION,
) -> PulseWindow:
    """Return the centered portion remaining after equal head/tail discards."""

    fraction = float(middle_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("middle_fraction must be finite and in (0, 1).")
    discard_each = int(round(0.5 * (1.0 - fraction) * pulse_window.samples))
    start_index = pulse_window.start_index + discard_each
    stop_index = pulse_window.stop_index - discard_each
    if stop_index - start_index < 3:
        raise ValueError(
            "Middle square-pulse window contains fewer than three samples."
        )
    return PulseWindow(
        start_time=float(time[start_index]),
        end_time=float(time[stop_index - 1]),
        start_index=start_index,
        stop_index=stop_index,
    )


def calibrate_square_waveform(
    raw: ScopeWaveform,
    *,
    demodulation_freq_hz: float,
    baseband_cutoff_hz: float,
    peak_omega_over_2pi_khz: float = PEAK_OMEGA_OVER_2PI_KHZ,
    q_sign: int = 1,
    middle_fraction: float = MIDDLE_FRACTION,
    detection_smoothing_seconds: float = DETECTION_SMOOTHING_SECONDS,
    detection_threshold_fraction: float = DETECTION_THRESHOLD_FRACTION,
    baseline_guard_seconds: float = BASELINE_GUARD_SECONDS,
) -> SquarePulseCalibration:
    """Derive a fixed rad/s/V scale from the middle of a square pulse."""

    _validate_raw_waveform(raw)
    demodulation_freq_hz = _validate_positive_finite(
        demodulation_freq_hz,
        "demodulation_freq_hz",
    )
    baseband_cutoff_hz = _validate_positive_finite(
        baseband_cutoff_hz,
        "baseband_cutoff_hz",
    )
    peak_khz = _validate_positive_finite(
        peak_omega_over_2pi_khz,
        "peak_omega_over_2pi_khz",
    )

    initial = demodulate_raw_rf(
        raw,
        demodulation_freq_hz=demodulation_freq_hz,
        baseband_cutoff_hz=baseband_cutoff_hz,
        q_sign=q_sign,
    )
    initial_window = detect_square_pulse_window(
        initial,
        smoothing_seconds=detection_smoothing_seconds,
        threshold_fraction=detection_threshold_fraction,
    )
    corrected, i_baseline_v, q_baseline_v = _baseline_correct(
        initial,
        initial_window,
        guard_seconds=baseline_guard_seconds,
    )
    pulse_window = detect_square_pulse_window(
        corrected,
        smoothing_seconds=detection_smoothing_seconds,
        threshold_fraction=detection_threshold_fraction,
    )
    middle_window = middle_pulse_window(
        pulse_window,
        corrected.time,
        middle_fraction=middle_fraction,
    )
    middle_envelope = corrected.amplitude[
        middle_window.start_index : middle_window.stop_index
    ]
    middle_envelope_v = float(np.median(middle_envelope))
    if not math.isfinite(middle_envelope_v) or middle_envelope_v <= 0.0:
        raise ValueError("Middle square-pulse envelope must be finite and positive.")
    middle_mad_v = float(np.median(np.abs(middle_envelope - middle_envelope_v)))
    relative_mad = middle_mad_v / middle_envelope_v
    omega_rad_s = 2.0 * math.pi * peak_khz * 1_000.0
    omega_rad_s_per_volt = omega_rad_s / middle_envelope_v

    return SquarePulseCalibration(
        demodulated=corrected,
        pulse_window=pulse_window,
        middle_window=middle_window,
        i_baseline_v=i_baseline_v,
        q_baseline_v=q_baseline_v,
        peak_omega_over_2pi_khz=peak_khz,
        middle_envelope_v=middle_envelope_v,
        middle_envelope_mad_v=middle_mad_v,
        middle_relative_mad=relative_mad,
        omega_rad_s_per_volt=omega_rad_s_per_volt,
    )


def calibration_to_dict(
    calibration: SquarePulseCalibration,
    raw: ScopeWaveform,
    *,
    q_convention: str,
    scope_resource: str,
) -> dict[str, Any]:
    pulse = calibration.pulse_window
    middle = calibration.middle_window
    return {
        "schema_version": 1,
        "method": "tnmr_100_square_middle_median",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "peak_omega_over_2pi_khz": calibration.peak_omega_over_2pi_khz,
        "middle_envelope_v": calibration.middle_envelope_v,
        "middle_envelope_mad_v": calibration.middle_envelope_mad_v,
        "middle_relative_mad": calibration.middle_relative_mad,
        "omega_rad_s_per_volt": calibration.omega_rad_s_per_volt,
        "pulse_window": {
            "start_s": pulse.start_time,
            "end_s": pulse.end_time,
            "duration_s": pulse.duration,
        },
        "middle_window": {
            "start_s": middle.start_time,
            "end_s": middle.end_time,
            "duration_s": middle.duration,
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
        "voltage_definition": (
            "median baseline-corrected demodulated envelope in the centered "
            "square-pulse analysis window"
        ),
        "units": {
            "peak_omega_over_2pi_khz": "kHz",
            "middle_envelope_v": "V",
            "middle_envelope_mad_v": "V",
            "omega_rad_s_per_volt": "rad/s/V",
        },
    }


def write_calibration_json(
    path: Path | str,
    payload: dict[str, Any],
    *,
    force: bool = False,
) -> Path:
    output_path = Path(path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists; pass --force to overwrite it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def plot_diagnostics(
    raw: ScopeWaveform,
    calibration: SquarePulseCalibration,
    *,
    save_figure: bool,
    figure_path: Path,
    show_plot: bool,
) -> None:
    demodulated = calibration.demodulated
    pulse = calibration.pulse_window
    middle = calibration.middle_window
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    time_us = raw.time * 1e6

    axes[0].plot(time_us, raw.voltage * 1e3, linewidth=0.8)
    axes[0].set_ylabel("Raw RF (mV)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        demodulated.time * 1e6,
        demodulated.amplitude * 1e3,
        linewidth=1.0,
        label="Corrected envelope",
    )
    axes[1].axvspan(
        pulse.start_time * 1e6,
        pulse.end_time * 1e6,
        color="tab:blue",
        alpha=0.10,
        label="Detected square pulse",
    )
    axes[1].axvspan(
        middle.start_time * 1e6,
        middle.end_time * 1e6,
        color="tab:green",
        alpha=0.20,
        label="Analyzed middle",
    )
    axes[1].axhline(
        calibration.middle_envelope_v * 1e3,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="Middle median",
    )
    axes[1].set_xlabel("Scope time (us)")
    axes[1].set_ylabel("Envelope (mV)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    figure.tight_layout()

    if save_figure:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=180)
    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate RF rad/s/V from a TNMR amplitude-100 square pulse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visa-resource", default=VISA_RESOURCE)
    parser.add_argument("--channel", default="CHAN1")
    parser.add_argument("--window-start", default=DEFAULT_WINDOW_START)
    parser.add_argument("--window-end", default=DEFAULT_WINDOW_END)
    parser.add_argument("--demod-freq", type=float, default=DEMODULATION_FREQ_HZ)
    parser.add_argument(
        "--baseband-cutoff",
        type=float,
        default=DEFAULT_BASEBAND_CUTOFF_HZ,
    )
    parser.add_argument("--q-convention", choices=("I+iQ", "I-iQ"), default="I+iQ")
    parser.add_argument(
        "--peak-khz",
        type=float,
        default=PEAK_OMEGA_OVER_2PI_KHZ,
        help="Omega/(2*pi) at TNMR amplitude 100.",
    )
    parser.add_argument(
        "--middle-fraction",
        type=float,
        default=MIDDLE_FRACTION,
    )
    parser.add_argument(
        "--detection-smoothing-seconds",
        type=float,
        default=DETECTION_SMOOTHING_SECONDS,
    )
    parser.add_argument(
        "--detection-threshold-fraction",
        type=float,
        default=DETECTION_THRESHOLD_FRACTION,
    )
    parser.add_argument(
        "--baseline-guard-seconds",
        type=float,
        default=BASELINE_GUARD_SECONDS,
    )
    parser.add_argument("--read-current", action="store_true")
    parser.add_argument("--shot-command")
    parser.add_argument("--no-arm-confirm", action="store_true")
    parser.add_argument("--no-rearm-after-capture", action="store_true")
    parser.add_argument(
        "--trigger-timeout-s",
        type=float,
        default=DEFAULT_TRIGGER_TIMEOUT_S,
    )
    parser.add_argument(
        "--armed-timeout-s",
        type=float,
        default=DEFAULT_ARMED_TIMEOUT_S,
    )
    parser.add_argument("--poll-s", type=float, default=DEFAULT_TRIGGER_POLL_S)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
    )
    parser.add_argument("--save-figure", dest="save_figure", action="store_true")
    parser.add_argument("--no-save-figure", dest="save_figure", action="store_false")
    parser.set_defaults(save_figure=DEFAULT_SAVE_FIGURE)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _window_seconds(args: argparse.Namespace) -> tuple[float, float]:
    start = parse_time_seconds(args.window_start)
    end = parse_time_seconds(args.window_end)
    if start is None or end is None or end <= start:
        raise SystemExit("--window-end must be greater than --window-start.")
    return start, end


def _start_shot(args: argparse.Namespace) -> None:
    print("Scope armed. Starting or waiting for the amplitude-100 square shot...")
    if args.shot_command:
        subprocess.run(args.shot_command, shell=True, check=True)


def _capture_raw(
    scope: RigolDHO,
    config: WaveformCaptureConfig,
    args: argparse.Namespace,
) -> ScopeWaveform:
    if args.read_current:
        return scope.capture_waveform(config)
    return scope.capture_single_waveform(
        config,
        start_shot=lambda: _start_shot(args),
        trigger_timeout_s=args.trigger_timeout_s,
        armed_timeout_s=args.armed_timeout_s,
        poll_s=args.poll_s,
        confirm_armed=not args.no_arm_confirm,
        rearm_after_read=not args.no_rearm_after_capture,
    )


def _print_summary(
    calibration: SquarePulseCalibration,
    raw: ScopeWaveform,
    *,
    q_convention: str,
    output_json: Path,
) -> None:
    pulse = calibration.pulse_window
    middle = calibration.middle_window
    print("TNMR amplitude-100 square-pulse RF calibration")
    print(f"Scope: {raw.source}")
    print(f"Channel: {raw.channel}")
    print(f"Q convention: {q_convention}")
    print(f"Assigned peak Omega/(2*pi): {calibration.peak_omega_over_2pi_khz:.12g} kHz")
    print(
        "Detected square pulse: "
        f"{pulse.start_time:.12g} s to {pulse.end_time:.12g} s, "
        f"duration {pulse.duration:.12g} s"
    )
    print(
        "Analyzed middle: "
        f"{middle.start_time:.12g} s to {middle.end_time:.12g} s, "
        f"duration {middle.duration:.12g} s"
    )
    print(f"Middle median envelope: {calibration.middle_envelope_v:.12g} V")
    print(f"Middle envelope MAD: {calibration.middle_envelope_mad_v:.12g} V")
    print(f"Relative middle MAD: {100.0 * calibration.middle_relative_mad:.6g}%")
    print(
        f"Voltage-to-Omega calibration: {calibration.omega_rad_s_per_volt:.12g} rad/s/V"
    )
    print(f"Saved calibration: {output_json}")
    print()
    print("Paste this into scope_rf_robustness_heatmap.py:")
    print(
        f"SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT = {calibration.omega_rad_s_per_volt:.16g}"
    )


def main() -> int:
    args = build_parser().parse_args()
    start, end = _window_seconds(args)
    config = WaveformCaptureConfig(
        channel=args.channel,
        window_start_s=start,
        window_end_s=end,
        allow_clamp=True,
    )
    with RigolDHO(resource=args.visa_resource, timeout_ms=30_000) as scope:
        raw = _capture_raw(scope, config, args)

    calibration = calibrate_square_waveform(
        raw,
        demodulation_freq_hz=args.demod_freq,
        baseband_cutoff_hz=args.baseband_cutoff,
        peak_omega_over_2pi_khz=args.peak_khz,
        q_sign=q_sign_from_convention(args.q_convention),
        middle_fraction=args.middle_fraction,
        detection_smoothing_seconds=args.detection_smoothing_seconds,
        detection_threshold_fraction=args.detection_threshold_fraction,
        baseline_guard_seconds=args.baseline_guard_seconds,
    )
    payload = calibration_to_dict(
        calibration,
        raw,
        q_convention=args.q_convention,
        scope_resource=args.visa_resource,
    )
    output_json = write_calibration_json(
        args.output_json,
        payload,
        force=args.force,
    )
    _print_summary(
        calibration,
        raw,
        q_convention=args.q_convention,
        output_json=output_json,
    )
    plot_diagnostics(
        raw,
        calibration,
        save_figure=args.save_figure,
        figure_path=args.figure_path,
        show_plot=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
