"""RF demodulation and pulse-window extraction for scope waveforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from hardware_capture.devices.rigol import ScopeWaveform


@dataclass(frozen=True)
class DemodulatedWaveform:
    """Baseband I/Q waveform derived from a raw RF trace."""

    time: np.ndarray
    i: np.ndarray
    q: np.ndarray
    z: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    sample_rate: float
    demodulation_freq_hz: float
    baseband_cutoff_hz: float
    filter_order: int
    raw_baseline: float
    q_sign: int = 1


@dataclass(frozen=True)
class PulseWindow:
    """Exclusive sample bounds and inclusive display times for a pulse."""

    start_time: float
    end_time: float
    start_index: int
    stop_index: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def samples(self) -> int:
        return self.stop_index - self.start_index


def _sample_interval(time: np.ndarray) -> float:
    if len(time) < 3:
        raise ValueError("time must contain at least three samples.")
    return float(np.median(np.diff(time)))


def estimate_demodulation_frequency(time: np.ndarray, voltage: np.ndarray) -> float:
    """Estimate the strongest non-DC carrier in a raw RF trace."""

    if len(time) != len(voltage):
        raise ValueError("time and voltage must have the same length.")
    centered = np.asarray(voltage, dtype=float) - float(np.median(voltage))
    spectrum = np.fft.rfft(centered * np.hanning(len(centered)))
    frequencies = np.fft.rfftfreq(len(centered), d=_sample_interval(time))
    if len(frequencies) < 2:
        raise ValueError("At least two FFT bins are needed to estimate a carrier.")
    return float(frequencies[int(np.argmax(np.abs(spectrum[1:]))) + 1])


def lowpass_baseband(
    values: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Low-pass filter a mixed RF waveform into baseband."""

    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be positive.")
    if order < 1:
        raise ValueError("filter order must be at least 1.")
    nyquist = 0.5 * sample_rate
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff {cutoff_hz:g} Hz must be below Nyquist {nyquist:g} Hz.")

    sos = butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    try:
        return sosfiltfilt(sos, values)
    except ValueError:
        return sosfilt(sos, values)


def demodulate_raw_rf(
    raw: ScopeWaveform,
    demodulation_freq_hz: float | None = None,
    *,
    baseband_cutoff_hz: float = 25e6,
    filter_order: int = 4,
    q_sign: int = 1,
) -> DemodulatedWaveform:
    """Demodulate a raw RF scope trace into I/Q baseband."""

    if q_sign not in (1, -1):
        raise ValueError("q_sign must be +1 for I+iQ or -1 for I-iQ.")

    baseline = float(np.median(raw.voltage))
    rf = raw.voltage - baseline
    lo_freq = float(demodulation_freq_hz) if demodulation_freq_hz else estimate_demodulation_frequency(raw.time, rf)
    phase = 2.0 * math.pi * lo_freq * raw.time
    i = 2.0 * lowpass_baseband(
        rf * np.cos(phase),
        raw.sample_rate,
        baseband_cutoff_hz,
        filter_order,
    )
    q = q_sign * 2.0 * lowpass_baseband(
        -rf * np.sin(phase),
        raw.sample_rate,
        baseband_cutoff_hz,
        filter_order,
    )
    z = i + 1j * q
    return DemodulatedWaveform(
        time=raw.time,
        i=i,
        q=q,
        z=z,
        amplitude=np.abs(z),
        phase=np.unwrap(np.angle(z)),
        sample_rate=raw.sample_rate,
        demodulation_freq_hz=lo_freq,
        baseband_cutoff_hz=float(baseband_cutoff_hz),
        filter_order=int(filter_order),
        raw_baseline=baseline,
        q_sign=int(q_sign),
    )


def _manual_window(
    time: np.ndarray,
    manual_start: float | None,
    manual_end: float | None,
) -> PulseWindow:
    start_time = float(time[0] if manual_start is None else manual_start)
    end_time = float(time[-1] if manual_end is None else manual_end)
    if end_time <= start_time:
        raise ValueError("Manual pulse window end must be greater than start.")
    start_index = int(np.searchsorted(time, start_time, side="left"))
    stop_index = int(np.searchsorted(time, end_time, side="right"))
    return PulseWindow(start_time, end_time, start_index, stop_index)


def _auto_window(
    demod: DemodulatedWaveform,
    threshold_fraction: float,
    padding_seconds: float,
) -> PulseWindow:
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must be in (0, 1).")
    peak = float(np.max(demod.amplitude))
    if peak <= 0.0:
        raise ValueError("Demodulated envelope has zero peak amplitude.")

    active = np.flatnonzero(demod.amplitude >= threshold_fraction * peak)
    if len(active) == 0:
        raise ValueError("Could not detect an active pulse window.")

    pad = int(round(padding_seconds * demod.sample_rate))
    start_index = max(0, int(active[0]) - pad)
    stop_index = min(len(demod.time), int(active[-1]) + 1 + pad)
    return PulseWindow(
        start_time=float(demod.time[start_index]),
        end_time=float(demod.time[stop_index - 1]),
        start_index=start_index,
        stop_index=stop_index,
    )


def detect_pulse_window(
    demod: DemodulatedWaveform,
    *,
    threshold_fraction: float = 0.10,
    padding_seconds: float = 0.25e-6,
    manual_start: float | None = None,
    manual_end: float | None = None,
) -> PulseWindow:
    """Detect the active pulse window from demodulated amplitude."""

    if manual_start is None and manual_end is None:
        window = _auto_window(demod, threshold_fraction, padding_seconds)
    else:
        window = _manual_window(demod.time, manual_start, manual_end)

    if window.samples < 2:
        raise ValueError("Pulse window contains fewer than two samples.")
    return window
