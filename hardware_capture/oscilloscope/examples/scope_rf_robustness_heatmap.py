"""Capture a complex or regular BB1 RF waveform and plot robustness diagnostics."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_capture.oscilloscope.plotting import plot_heatmap, plot_pulse_diagnostics
from hardware_capture.oscilloscope.pulse_robustness import (
    PulseModel,
    RobustnessResult,
    bb1_nominal_area,
    build_pulse_model,
    scan_robustness,
    threshold_region_summary,
)
from hardware_capture.oscilloscope.rf_demodulation import (
    DemodulatedWaveform,
    PulseWindow,
    demodulate_raw_rf,
    detect_pulse_window,
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
DEFAULT_WINDOW_END = "100us"
DEFAULT_PULSE_THRESHOLD_FRACTION = 0.10
DEFAULT_PULSE_PADDING_SECONDS = 0.25e-6
PHASE_ALIGNMENT_MIN_AMPLITUDE_FRACTION = 0.25

COMPLEX_PULSE = "complex"
REGULAR_BB1_PULSE = "regular-bb1"
PULSE_TYPES = (COMPLEX_PULSE, REGULAR_BB1_PULSE)

# Select "complex" for a normalized-amplitude optimized pulse or "regular-bb1"
# for a standard BB1 waveform captured by the oscilloscope.
PULSE_TYPE = COMPLEX_PULSE
BB1_ROTATION_ANGLE_DEG = 90.0

# Paste the assignment printed by calibrate_rf_from_square.py.
# This fixed scale preserves real amplitude differences between scope captures.
SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT: float | None = 1776212.654495664  # 1703986.53476382

# Optional one-column CSV containing the intended pulse amplitudes.
# "auto" recognizes Hz or kHz in the filename or CSV header. Set the unit
# explicitly for a headerless file whose filename does not include it.
AMPLITUDE_CSV_PATH: str | Path | None = r"C:\Users\ctcheung\Downloads\160_IQ_khz.csv"
AMPLITUDE_CSV_UNIT = "auto"  # "auto", "Hz", or "kHz"

# Optional one-column CSV containing the intended pulse phase.
# "auto" recognizes degrees or radians in the filename or CSV header.
PHASE_CSV_PATH: str | Path | None = (
    r"C:\Users\ctcheung\Downloads\IQ_160_phase_degree.csv"
)
PHASE_CSV_UNIT = "auto"  # "auto", "deg", or "rad"


def q_sign_from_convention(value: str) -> int:
    if value == "I+iQ":
        return 1
    if value == "I-iQ":
        return -1
    raise ValueError(f"Unsupported Q convention {value!r}.")


def _validate_fixed_omega_scale(
    omega_rad_s_per_volt: float | None,
) -> float:
    if omega_rad_s_per_volt is None:
        raise ValueError(
            "Set SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT from "
            "calibrate_rf_from_square.py before running a complex pulse."
        )
    scale = float(omega_rad_s_per_volt)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT must be finite and positive."
        )
    return scale


def _pulse_scaling_for_type(
    pulse_type: str,
    bb1_rotation_angle_deg: float,
) -> tuple[float, float | None]:
    """Return the nominal pulse area and optional fixed voltage-to-Omega scale."""

    if pulse_type == COMPLEX_PULSE:
        return (
            bb1_nominal_area(),
            _validate_fixed_omega_scale(
                SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT,
            ),
        )
    if pulse_type == REGULAR_BB1_PULSE:
        angle_deg = float(bb1_rotation_angle_deg)
        if not math.isfinite(angle_deg) or angle_deg <= 0.0 or angle_deg > 720.0:
            raise ValueError(
                "BB1_ROTATION_ANGLE_DEG must be finite and in the range "
                "(0, 720] degrees."
            )
        return bb1_nominal_area(math.radians(angle_deg)), None
    raise ValueError(f"Unsupported pulse type {pulse_type!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture scope RF data and plot a robustness heatmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visa-resource", default=VISA_RESOURCE)
    parser.add_argument("--channel", default="CHAN1")
    parser.add_argument("--window-start", default=DEFAULT_WINDOW_START)
    parser.add_argument("--window-end", default=DEFAULT_WINDOW_END)
    parser.add_argument("--demod-freq", type=float, default=DEMODULATION_FREQ_HZ)
    parser.add_argument("--baseband-cutoff", type=float, default=25e6)
    parser.add_argument("--q-convention", choices=("I+iQ", "I-iQ"), default="I-iQ")
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--simulation-steps", type=int, default=900)
    parser.add_argument("--pulse-type", choices=PULSE_TYPES, default=PULSE_TYPE)
    parser.add_argument(
        "--bb1-rotation-angle-deg",
        type=float,
        default=BB1_ROTATION_ANGLE_DEG,
    )
    parser.add_argument(
        "--method", choices=("analytic", "scipy_expm"), default="analytic"
    )
    parser.add_argument("--read-current", action="store_true")
    parser.add_argument("--shot-command")
    parser.add_argument("--no-arm-confirm", action="store_true")
    parser.add_argument("--no-rearm-after-capture", action="store_true")
    parser.add_argument(
        "--trigger-timeout-s", type=float, default=DEFAULT_TRIGGER_TIMEOUT_S
    )
    parser.add_argument(
        "--armed-timeout-s", type=float, default=DEFAULT_ARMED_TIMEOUT_S
    )
    parser.add_argument("--poll-s", type=float, default=DEFAULT_TRIGGER_POLL_S)
    parser.add_argument("--save-figure", action="store_true")
    parser.add_argument(
        "--figure-path", type=Path, default=Path("scope_rf_robustness_heatmap.png")
    )
    parser.add_argument("--save-pulse-figure", action="store_true")
    parser.add_argument(
        "--pulse-figure-path", type=Path, default=Path("scope_rf_demodulated_pulse.png")
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def _window_seconds(args: argparse.Namespace) -> tuple[float, float]:
    start = parse_time_seconds(args.window_start)
    end = parse_time_seconds(args.window_end)
    if start is None or end is None or end <= start:
        raise SystemExit("--window-end must be greater than --window-start.")
    return start, end


def _resolve_amplitude_csv_unit(csv_path: Path, unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized in {"hz", "hertz"}:
        return "hz"
    if normalized in {"khz", "kilohertz"}:
        return "khz"
    if normalized != "auto":
        raise ValueError('Amplitude CSV unit must be "auto", "Hz", or "kHz".')

    with csv_path.open(encoding="utf-8-sig", errors="replace") as csv_file:
        header = "".join(csv_file.readline() for _ in range(5))
    unit_text = f"{csv_path.name}\n{header}".lower()
    has_khz = bool(re.search(r"(?<![a-z])(?:khz|kilohertz)(?![a-z])", unit_text))
    has_hz = bool(re.search(r"(?<![a-z])(?:hz|hertz)(?![a-z])", unit_text))
    if has_khz and has_hz:
        raise ValueError("Amplitude CSV has conflicting Hz and kHz unit labels.")
    if has_khz:
        return "khz"
    if has_hz:
        return "hz"
    raise ValueError(
        "Could not determine the amplitude CSV unit from its filename or header. "
        'Set AMPLITUDE_CSV_UNIT to "Hz" or "kHz".'
    )


def _load_amplitude_csv_khz(csv_path: Path, unit: str = "auto") -> np.ndarray:
    values = np.asarray(np.genfromtxt(csv_path, delimiter=","), dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError("Amplitude CSV must contain exactly one column.")

    if values.size > 0 and np.isnan(values[0]):
        values = values[1:]
    if values.size == 0:
        raise ValueError("Amplitude CSV contains no amplitude values.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Amplitude CSV values must be finite and nonnegative.")
    if _resolve_amplitude_csv_unit(csv_path, unit) == "hz":
        values = values / 1e3
    return values


def _resolve_phase_csv_unit(csv_path: Path, unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized in {"rad", "radian", "radians"}:
        return "rad"
    if normalized in {"deg", "degree", "degrees"}:
        return "deg"
    if normalized != "auto":
        raise ValueError('Phase CSV unit must be "auto", "deg", or "rad".')

    with csv_path.open(encoding="utf-8-sig", errors="replace") as csv_file:
        header = "".join(csv_file.readline() for _ in range(5))
    unit_text = f"{csv_path.name}\n{header}".lower()
    has_degrees = bool(
        re.search(r"(?<![a-z])(?:deg|degree|degrees)(?![a-z])", unit_text)
    )
    has_radians = bool(
        re.search(r"(?<![a-z])(?:rad|radian|radians)(?![a-z])", unit_text)
    )
    if has_degrees and has_radians:
        raise ValueError("Phase CSV has conflicting degree and radian unit labels.")
    if has_degrees:
        return "deg"
    if has_radians:
        return "rad"
    raise ValueError(
        "Could not determine the phase CSV unit from its filename or header. "
        'Set PHASE_CSV_UNIT to "deg" or "rad".'
    )


def _load_phase_csv_radians(csv_path: Path, unit: str = "auto") -> np.ndarray:
    values = np.asarray(np.genfromtxt(csv_path, delimiter=","), dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError("Phase CSV must contain exactly one column.")

    if values.size > 0 and np.isnan(values[0]):
        values = values[1:]
    if values.size == 0:
        raise ValueError("Phase CSV contains no phase values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Phase CSV values must be finite.")
    if _resolve_phase_csv_unit(csv_path, unit) == "deg":
        values = np.deg2rad(values)
    return np.unwrap(values)


def _align_overall_phase(
    demod_phase_radians: np.ndarray,
    csv_phase_radians: np.ndarray,
    demod_amplitude: np.ndarray,
) -> tuple[np.ndarray, float]:
    if not (
        len(demod_phase_radians) == len(csv_phase_radians) == len(demod_amplitude)
    ):
        raise ValueError("Phase-alignment arrays must have the same length.")
    peak_amplitude = float(np.max(demod_amplitude))
    reliable = demod_amplitude >= (
        PHASE_ALIGNMENT_MIN_AMPLITUDE_FRACTION * peak_amplitude
    )
    phase_offset = float(
        np.median(csv_phase_radians[reliable] - demod_phase_radians[reliable])
    )
    return demod_phase_radians + phase_offset, phase_offset


def _plot_amplitude_comparison(
    demod: DemodulatedWaveform,
    window: PulseWindow,
    pulse: PulseModel,
    csv_path: Path,
    *,
    amplitude_unit: str,
    show_plot: bool,
) -> None:
    import matplotlib.pyplot as plt

    csv_amplitude_khz = _load_amplitude_csv_khz(csv_path, amplitude_unit)
    active = slice(window.start_index, window.stop_index)
    active_time = demod.time[active]
    time_us = (active_time - active_time[0]) * 1e6
    csv_time_us = np.linspace(0.0, float(time_us[-1]), len(csv_amplitude_khz))
    demod_amplitude_khz = demod.amplitude[active] * pulse.scale / (2.0 * math.pi * 1e3)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        time_us,
        demod_amplitude_khz,
        label="Demodulated scope amplitude",
        linewidth=1.2,
    )
    ax.plot(
        csv_time_us,
        csv_amplitude_khz,
        label="CSV amplitude",
        linewidth=1.2,
    )
    ax.set(
        xlabel="Time within detected pulse (us)",
        ylabel="Amplitude (kHz)",
        title="Demodulated scope amplitude vs CSV amplitude",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_phase_comparison(
    demod: DemodulatedWaveform,
    window: PulseWindow,
    csv_path: Path,
    *,
    phase_unit: str,
    show_plot: bool,
) -> None:
    import matplotlib.pyplot as plt

    csv_phase_radians = _load_phase_csv_radians(csv_path, phase_unit)
    active = slice(window.start_index, window.stop_index)
    active_time = demod.time[active]
    time_us = (active_time - active_time[0]) * 1e6
    csv_time_us = np.linspace(0.0, float(time_us[-1]), len(csv_phase_radians))
    demod_phase_radians = np.unwrap(demod.phase[active])
    csv_phase_on_scope_grid = np.interp(time_us, csv_time_us, csv_phase_radians)
    demod_phase_radians, phase_offset = _align_overall_phase(
        demod_phase_radians,
        csv_phase_on_scope_grid,
        demod.amplitude[active],
    )
    q_convention = "I+iQ" if demod.q_sign == 1 else "I-iQ"
    print(
        "Phase comparison offset applied to demodulated trace: "
        f"{phase_offset:+.12g} rad"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        time_us,
        demod_phase_radians,
        label=f"Demodulated scope phase ({q_convention}, offset aligned)",
        linewidth=1.2,
    )
    ax.plot(
        csv_time_us,
        csv_phase_radians,
        label="CSV phase",
        linewidth=1.2,
    )
    ax.set(
        xlabel="Time within detected pulse (us)",
        ylabel="Unwrapped phase (rad)",
        title="Offset-aligned demodulated scope phase vs CSV phase",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _start_shot(args: argparse.Namespace) -> None:
    pulse_label = "regular BB1" if args.pulse_type == REGULAR_BB1_PULSE else "TNMR"
    print(f"Scope armed. Starting or waiting for {pulse_label} shot...")
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
    raw: ScopeWaveform,
    demod: DemodulatedWaveform,
    window: PulseWindow,
    pulse: PulseModel,
    result: RobustnessResult,
    q_convention: str,
    pulse_type: str,
    nominal_pulse_area_rad: float,
    bb1_rotation_angle_deg: float,
) -> None:
    print("Scope RF robustness heatmap summary")
    print(f"Scope: {raw.source}")
    print(f"Channel: {raw.channel}")
    print(f"Raw points: {len(raw.voltage)}")
    print(f"Sample rate: {raw.sample_rate:.12g} Sa/s")
    print(f"Demodulation frequency: {demod.demodulation_freq_hz:.12g} Hz")
    print(f"Q convention: {q_convention}")
    print(f"Pulse type: {pulse_type}")
    if pulse_type == REGULAR_BB1_PULSE:
        print("Calibration mode: known regular-BB1 envelope area")
        print(f"BB1 target rotation angle: {bb1_rotation_angle_deg:.12g} deg")
        print(f"BB1 total transverse area: {nominal_pulse_area_rad:.12g} rad")
    else:
        print("Calibration mode: fixed amplitude-100 square-pulse calibration")
    print(f"Voltage-to-Omega scale: {pulse.scale:.12g} rad/s/V")
    print(
        "Calibrated transverse pulse area: "
        f"{pulse.scale * pulse.raw_envelope_area:.12g} rad"
    )
    print(
        "Detected pulse: "
        f"{window.start_time:.12g} s to {window.end_time:.12g} s, "
        f"duration {window.duration:.12g} s"
    )
    print(f"Simulation steps: {len(pulse.dt)}")
    print(f"Fidelity at center: {result.center_fidelity:.12f}")
    print(f"Minimum fidelity: {result.min_fidelity:.12f}")
    for line in threshold_region_summary(result):
        print(line)


def main() -> int:
    args = build_parser().parse_args()
    nominal_pulse_area_rad, omega_rad_s_per_volt = _pulse_scaling_for_type(
        args.pulse_type,
        args.bb1_rotation_angle_deg,
    )
    start, end = _window_seconds(args)
    config = WaveformCaptureConfig(
        channel=args.channel,
        window_start_s=start,
        window_end_s=end,
        allow_clamp=True,
    )
    with RigolDHO(resource=args.visa_resource, timeout_ms=30_000) as scope:
        raw = _capture_raw(scope, config, args)

    demod = demodulate_raw_rf(
        raw,
        demodulation_freq_hz=args.demod_freq,
        baseband_cutoff_hz=args.baseband_cutoff,
        q_sign=q_sign_from_convention(args.q_convention),
    )
    window = detect_pulse_window(
        demod,
        threshold_fraction=DEFAULT_PULSE_THRESHOLD_FRACTION,
        padding_seconds=DEFAULT_PULSE_PADDING_SECONDS,
    )
    pulse = build_pulse_model(
        demod,
        window,
        nominal_pulse_area_rad=nominal_pulse_area_rad,
        simulation_steps=args.simulation_steps,
        omega_rad_s_per_volt=omega_rad_s_per_volt,
    )
    result = scan_robustness(pulse, grid_size=args.grid_size, method=args.method)

    _print_summary(
        raw,
        demod,
        window,
        pulse,
        result,
        args.q_convention,
        args.pulse_type,
        nominal_pulse_area_rad,
        args.bb1_rotation_angle_deg,
    )
    show_plot = not args.no_show
    plot_pulse_diagnostics(
        demod,
        window,
        pulse,
        save_figure=args.save_pulse_figure,
        figure_path=args.pulse_figure_path,
        show_plot=show_plot,
    )
    if AMPLITUDE_CSV_PATH is not None:
        _plot_amplitude_comparison(
            demod,
            window,
            pulse,
            Path(AMPLITUDE_CSV_PATH),
            amplitude_unit=AMPLITUDE_CSV_UNIT,
            show_plot=show_plot,
        )
    if PHASE_CSV_PATH is not None:
        _plot_phase_comparison(
            demod,
            window,
            Path(PHASE_CSV_PATH),
            phase_unit=PHASE_CSV_UNIT,
            show_plot=show_plot,
        )
    plot_heatmap(
        result,
        save_figure=args.save_figure,
        figure_path=args.figure_path,
        show_plot=show_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
