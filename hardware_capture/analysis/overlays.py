"""Reusable analysis overlays algorithms. Use hardware_capture.api for workflows."""

from __future__ import annotations

import math

import re

from pathlib import Path
from hardware_capture.figures import subplots

import numpy as np

from hardware_capture.analysis.robustness import PulseModel

from hardware_capture.signal.demodulation import DemodulatedWaveform, PulseWindow

PHASE_ALIGNMENT_MIN_AMPLITUDE_FRACTION = 0.25

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
    figure_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    csv_amplitude_khz = _load_amplitude_csv_khz(csv_path, amplitude_unit)
    active = slice(window.start_index, window.stop_index)
    active_time = demod.time[active]
    time_us = (active_time - active_time[0]) * 1e6
    csv_time_us = np.linspace(0.0, float(time_us[-1]), len(csv_amplitude_khz))
    demod_amplitude_khz = demod.amplitude[active] * pulse.scale / (2.0 * math.pi * 1e3)

    fig, ax = subplots(figsize=(10, 5), show=show_plot)
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
    if figure_path is not None:
        fig.savefig(figure_path, dpi=150)
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
    figure_path: Path | None = None,
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

    fig, ax = subplots(figsize=(10, 5), show=show_plot)
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
    if figure_path is not None:
        fig.savefig(figure_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
