"""Capture an RF waveform and compare it with one fixed best-axis X(pi) gate.

This is a dedicated sibling of ``scope_rf_robustness_heatmap.py``.  The
original script intentionally compares the measured pulse with its own nominal
unitary.  This script instead:

1. propagates the measured waveform at zero amplitude error and zero detuning;
2. extracts its nominal transverse rotation-axis phase;
3. defines one ideal equatorial pi target about that fixed axis; and
4. uses that same target for every point in the robustness heatmap.

Only one constant I/Q-frame phase is removed.  The target phase is never fitted
separately at different amplitude-error or detuning points, so rotation-angle
errors and unwanted Z rotation remain visible in the fidelity.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_capture.oscilloscope.examples import scope_rf_robustness_heatmap as base
from hardware_capture.oscilloscope.plotting import plot_pulse_diagnostics
from hardware_capture.oscilloscope.pulse_robustness import (
    I2,
    SIGMA_X,
    SIGMA_Y,
    PulseModel,
    RobustnessResult,
    build_pulse_model,
    process_fidelity,
    process_fidelity_grid,
    simulate_unitary,
    step_unitary_analytic,
)
from hardware_capture.oscilloscope.rf_demodulation import (
    demodulate_raw_rf,
    detect_pulse_window,
)
from hardware_capture.oscilloscope.rigol_dho import (
    DEFAULT_ARMED_TIMEOUT_S,
    DEFAULT_TRIGGER_POLL_S,
    DEFAULT_TRIGGER_TIMEOUT_S,
    RigolDHO,
    WaveformCaptureConfig,
    parse_time_seconds,
)


# =============================================================================
# User configuration
#
# Edit these values and run this file directly.  Matching command-line options
# can still override the most commonly changed settings for one-off runs.
# =============================================================================

# Scope acquisition.
VISA_RESOURCE = "TCPIP::192.168.1.152::INSTR"
CHANNEL = "CHAN1"
WINDOW_START = "-10us"
WINDOW_END = "100us"
READ_CURRENT_TRACE = False
SHOT_COMMAND: str | None = None
CONFIRM_SCOPE_ARMED = True
REARM_AFTER_CAPTURE = True
TRIGGER_TIMEOUT_S = DEFAULT_TRIGGER_TIMEOUT_S
ARMED_TIMEOUT_S = DEFAULT_ARMED_TIMEOUT_S
TRIGGER_POLL_S = DEFAULT_TRIGGER_POLL_S

# RF demodulation and pulse-window detection.
DEMODULATION_FREQ_HZ = 601.0100113e6
BASEBAND_CUTOFF_HZ = 25e6
Q_CONVENTION = "I-iQ"  # Choose "I+iQ" or "I-iQ".
PULSE_THRESHOLD_FRACTION = 0.10
PULSE_PADDING_SECONDS = 0.25e-6
PHASE_ALIGNMENT_MIN_AMPLITUDE_FRACTION = 0.25

# Fixed square-pulse calibration.  Paste the value printed by
# calibrate_rf_from_square.py.  Units are rad/s/V.
SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT: float | None = 1776212.654495664

# Robustness simulation.
GRID_SIZE = 101  # Must be odd so (amplitude error, detuning) = (0, 0) is sampled.
SIMULATION_STEPS = 900
PROPAGATION_METHOD = "analytic"  # Choose "analytic" or "scipy_expm".
AMPLITUDE_ERROR_MIN = -0.5
AMPLITUDE_ERROR_MAX = 0.5
DETUNING_ERROR_MIN = -0.5
DETUNING_ERROR_MAX = 0.5

# Figures and optional commanded-amplitude comparison.
SHOW_PLOTS = True
SAVE_HEATMAP = False
HEATMAP_PATH = Path("scope_rf_best_axis_xpi_heatmap.png")
HEATMAP_VMIN = 0.99
HEATMAP_VMAX = 1.0
SAVE_PULSE_FIGURE = False
PULSE_FIGURE_PATH = Path("scope_rf_best_axis_xpi_demodulated_pulse.png")
AMPLITUDE_CSV_PATH: str | Path | None = None
AMPLITUDE_CSV_UNIT = "auto"  # Choose "auto", "Hz", or "kHz".
PHASE_CSV_PATH: str | Path | None = None
PHASE_CSV_UNIT = "auto"  # Choose "auto", "deg", or "rad".


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
    fig, ax = plt.subplots(figsize=(8, 6))
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


def _validate_fixed_omega_scale(value: float | None) -> float:
    if value is None:
        raise ValueError(
            "Set SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT from "
            "calibrate_rf_from_square.py before running this script."
        )
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT must be finite and positive."
        )
    return scale


def _q_sign_from_convention(value: str) -> int:
    if value == "I+iQ":
        return 1
    if value == "I-iQ":
        return -1
    raise ValueError(f"Unsupported Q convention {value!r}.")


def _window_seconds(args: argparse.Namespace) -> tuple[float, float]:
    start = parse_time_seconds(args.window_start)
    end = parse_time_seconds(args.window_end)
    if start is None or end is None or end <= start:
        raise SystemExit("--window-end must be greater than --window-start.")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture scope RF data and compare it with one fixed best-axis "
            "equatorial X(pi) target."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visa-resource", default=VISA_RESOURCE)
    parser.add_argument("--channel", default=CHANNEL)
    parser.add_argument("--window-start", default=WINDOW_START)
    parser.add_argument("--window-end", default=WINDOW_END)
    parser.add_argument("--demod-freq", type=float, default=DEMODULATION_FREQ_HZ)
    parser.add_argument("--baseband-cutoff", type=float, default=BASEBAND_CUTOFF_HZ)
    parser.add_argument(
        "--q-convention",
        choices=("I+iQ", "I-iQ"),
        default=Q_CONVENTION,
    )
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument(
        "--simulation-steps",
        type=int,
        default=SIMULATION_STEPS,
    )
    parser.add_argument(
        "--amplitude-error-min",
        type=float,
        default=AMPLITUDE_ERROR_MIN,
    )
    parser.add_argument(
        "--amplitude-error-max",
        type=float,
        default=AMPLITUDE_ERROR_MAX,
    )
    parser.add_argument(
        "--detuning-error-min",
        type=float,
        default=DETUNING_ERROR_MIN,
    )
    parser.add_argument(
        "--detuning-error-max",
        type=float,
        default=DETUNING_ERROR_MAX,
    )
    parser.add_argument(
        "--method",
        choices=("analytic", "scipy_expm"),
        default=PROPAGATION_METHOD,
    )
    parser.add_argument(
        "--read-current",
        action="store_true",
        default=READ_CURRENT_TRACE,
    )
    parser.add_argument("--shot-command", default=SHOT_COMMAND)
    parser.add_argument(
        "--no-arm-confirm",
        action="store_true",
        default=not CONFIRM_SCOPE_ARMED,
    )
    parser.add_argument(
        "--no-rearm-after-capture",
        action="store_true",
        default=not REARM_AFTER_CAPTURE,
    )
    parser.add_argument(
        "--trigger-timeout-s",
        type=float,
        default=TRIGGER_TIMEOUT_S,
    )
    parser.add_argument(
        "--armed-timeout-s",
        type=float,
        default=ARMED_TIMEOUT_S,
    )
    parser.add_argument("--poll-s", type=float, default=TRIGGER_POLL_S)
    parser.add_argument(
        "--save-figure",
        action="store_true",
        default=SAVE_HEATMAP,
    )
    parser.add_argument("--figure-path", type=Path, default=HEATMAP_PATH)
    parser.add_argument(
        "--save-pulse-figure",
        action="store_true",
        default=SAVE_PULSE_FIGURE,
    )
    parser.add_argument(
        "--pulse-figure-path",
        type=Path,
        default=PULSE_FIGURE_PATH,
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=not SHOW_PLOTS,
    )
    # The reused capture/summary helpers inspect these fields.  They are fixed
    # here because this sibling script intentionally supports only complex X(pi).
    parser.set_defaults(
        pulse_type=base.COMPLEX_PULSE,
        bb1_rotation_angle_deg=180.0,
    )
    return parser


def _print_best_axis_summary(analysis: BestAxisAnalysis) -> None:
    phase_deg = _phase_mod_180_degrees(analysis.target_phase_rad)
    frame_rotation_deg = float(np.mod(-np.degrees(analysis.target_phase_rad), 180.0))
    print("Fidelity target: one fixed ideal best-axis equatorial X(pi) gate")
    print(f"Nominal best-axis phase: {phase_deg:.12g} deg (mod 180 deg)")
    print(
        "Equivalent constant I/Q frame rotation that calls this axis X: "
        f"{frame_rotation_deg:.12g} deg (mod 180 deg)"
    )
    print(
        "The target phase was selected once from the nominal center and held "
        "fixed across the full grid."
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    nominal_pulse_area_rad = math.pi
    omega_rad_s_per_volt = _validate_fixed_omega_scale(
        SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT
    )
    start, end = _window_seconds(args)
    config = WaveformCaptureConfig(
        channel=args.channel,
        window_start_s=start,
        window_end_s=end,
        allow_clamp=True,
    )
    with RigolDHO(resource=args.visa_resource, timeout_ms=30_000) as scope:
        raw = base._capture_raw(scope, config, args)

    demod = demodulate_raw_rf(
        raw,
        demodulation_freq_hz=args.demod_freq,
        baseband_cutoff_hz=args.baseband_cutoff,
        q_sign=_q_sign_from_convention(args.q_convention),
    )
    window = detect_pulse_window(
        demod,
        threshold_fraction=PULSE_THRESHOLD_FRACTION,
        padding_seconds=PULSE_PADDING_SECONDS,
    )
    pulse = build_pulse_model(
        demod,
        window,
        nominal_pulse_area_rad=nominal_pulse_area_rad,
        simulation_steps=args.simulation_steps,
        omega_rad_s_per_volt=omega_rad_s_per_volt,
    )
    analysis = scan_best_axis_xpi(
        pulse,
        grid_size=args.grid_size,
        amplitude_error_min=args.amplitude_error_min,
        amplitude_error_max=args.amplitude_error_max,
        detuning_error_min=args.detuning_error_min,
        detuning_error_max=args.detuning_error_max,
        method=args.method,
    )

    base._print_summary(
        raw,
        demod,
        window,
        pulse,
        analysis.result,
        args.q_convention,
        args.pulse_type,
        nominal_pulse_area_rad,
        args.bb1_rotation_angle_deg,
    )
    _print_best_axis_summary(analysis)

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
        base._plot_amplitude_comparison(
            demod,
            window,
            pulse,
            Path(AMPLITUDE_CSV_PATH),
            amplitude_unit=AMPLITUDE_CSV_UNIT,
            show_plot=show_plot,
        )
    if PHASE_CSV_PATH is not None:
        previous_phase_threshold = base.PHASE_ALIGNMENT_MIN_AMPLITUDE_FRACTION
        base.PHASE_ALIGNMENT_MIN_AMPLITUDE_FRACTION = (
            PHASE_ALIGNMENT_MIN_AMPLITUDE_FRACTION
        )
        try:
            base._plot_phase_comparison(
                demod,
                window,
                Path(PHASE_CSV_PATH),
                phase_unit=PHASE_CSV_UNIT,
                show_plot=show_plot,
            )
        finally:
            base.PHASE_ALIGNMENT_MIN_AMPLITUDE_FRACTION = previous_phase_threshold
    plot_best_axis_heatmap(
        analysis,
        save_figure=args.save_figure,
        figure_path=args.figure_path,
        show_plot=show_plot,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
