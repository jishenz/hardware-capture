"""One sampled pulse contract shared by generation, analysis and optimization."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from hardware_capture.configuration import Experiment, positive
from . import waveforms


@dataclass(frozen=True)
class Pulse:
    """Uniform midpoint samples; Rabi amplitude in Hz, phase in the command frame."""
    duration_s: float
    rabi_hz: np.ndarray
    phase_rad: np.ndarray

    def __post_init__(self):
        positive(self.duration_s, "pulse.duration_s")
        if self.rabi_hz.ndim != 1 or len(self.rabi_hz) < 2 or self.phase_rad.shape != self.rabi_hz.shape:
            raise ValueError("Pulse amplitude and phase must be matching 1D arrays with at least two samples")
        if not np.all(np.isfinite(self.rabi_hz)) or not np.all(np.isfinite(self.phase_rad)) or np.any(self.rabi_hz < 0):
            raise ValueError("Pulse values must be finite and amplitudes nonnegative")

    @property
    def time_s(self):
        return (np.arange(len(self.rabi_hz)) + 0.5) * self.duration_s / len(self.rabi_hz)

    @property
    def dt_s(self):
        return np.full(len(self.rabi_hz), self.duration_s / len(self.rabi_hz))

    def save(self, directory: Path, *, rabi_at_100_hz: float | None = None) -> dict:
        if rabi_at_100_hz is not None and np.max(self.rabi_hz) > positive(rabi_at_100_hz, "rabi_at_100_hz") * (1 + 1e-12):
            raise ValueError("Pulse exceeds the calibrated amplitude-100 Rabi rate")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "pulse.csv"
        np.savetxt(path, np.column_stack((self.time_s, self.rabi_hz, self.phase_rad)), delimiter=",", header="time_s,rabi_hz,phase_rad", comments="")
        np.savetxt(directory / "amplitude_khz.csv", self.rabi_hz / 1000, delimiter=",", header="amplitude_kHz", comments="")
        np.savetxt(directory / "phase_deg.csv", np.rad2deg(self.phase_rad), delimiter=",", header="phase_deg", comments="")
        metadata = {"duration_s": self.duration_s, "samples": len(self.rabi_hz), "peak_rabi_hz": float(np.max(self.rabi_hz)), "phase_convention": "command frame; match the demodulation Q convention", "pulse_path": str(path)}
        if rabi_at_100_hz is not None:
            reference = positive(rabi_at_100_hz, "rabi_at_100_hz")
            if np.max(self.rabi_hz) > reference * (1 + 1e-12):
                raise ValueError("Pulse exceeds the calibrated amplitude-100 Rabi rate")
            np.savetxt(directory / "tnmr_amplitude.txt", 100 * self.rabi_hz / reference, fmt="%.12g")
            np.savetxt(directory / "tnmr_phase_deg.txt", np.mod(np.rad2deg(self.phase_rad), 360), fmt="%.12g")
        (directory / "pulse.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata


def generate(config: Experiment) -> Pulse:
    settings = config.section("pulse")
    kind = settings.get("kind", "bb1").lower()
    duration = positive(config.require("pulse", "duration_s"), "pulse.duration_s")
    samples = settings.get("samples", 6000)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("pulse.samples must be an integer >= 2")
    phase_deg = float(settings.get("axis_phase_deg", 0))
    angle_deg = positive(settings.get("angle_deg", 180), "pulse.angle_deg")
    params = dict(settings.get("parameters", {}))
    if kind == "csv":
        if "path" in settings:
            rows = np.genfromtxt(settings["path"], delimiter=",", names=True)
            if rows.dtype.names != ("time_s", "rabi_hz", "phase_rad"):
                raise ValueError("Pulse CSV header must be time_s,rabi_hz,phase_rad")
            times = np.atleast_1d(rows["time_s"])
            expected = (np.arange(len(times)) + 0.5) * duration / len(times)
            if len(times) < 2 or not np.allclose(times, expected, rtol=1e-6, atol=1e-15):
                raise ValueError("Pulse CSV must use uniform midpoint times matching pulse.duration_s")
            amplitude = np.atleast_1d(rows["rabi_hz"])
            phase = np.unwrap(np.atleast_1d(rows["phase_rad"]))
        else:
            from hardware_capture.analysis.overlays import _load_amplitude_csv_khz, _load_phase_csv_radians
            amplitude = 1000 * _load_amplitude_csv_khz(Path(config.require("pulse", "amplitude_path")), settings.get("amplitude_unit", "auto"))
            phase = _load_phase_csv_radians(Path(config.require("pulse", "phase_path")), settings.get("phase_unit", "auto"))
            if len(amplitude) != len(phase):
                raise ValueError("Amplitude/phase CSVs must have matching sample counts")
        old = (np.arange(len(amplitude)) + 0.5) / len(amplitude)
        new = (np.arange(samples) + 0.5) / samples
        amplitude = np.interp(new, old, amplitude)
        phase = np.interp(new, old, phase)
    elif kind == "bb1":
        if params:
            raise ValueError("BB1 pulse generation uses angle_deg/axis_phase_deg and duration, not pulse.parameters")
        segments = waveforms.BB1(angle=angle_deg, phase=phase_deg).segments()
        weights = np.array([s["flip_angle_deg"] for s in segments])
        boundaries = np.cumsum(weights) / weights.sum()
        indices = np.minimum(np.searchsorted(boundaries, (np.arange(samples) + 0.5) / samples), len(segments) - 1)
        phase = np.deg2rad([segments[i]["phase_deg"] for i in indices])
        amplitude = np.full(samples, weights.sum() / 360 / duration)
    elif kind in {"square", "constant"}:
        if params:
            raise ValueError("Square pulses use pulse.peak_rabi_hz or angle_deg/duration_s")
        amplitude = np.full(samples, float(settings.get("peak_rabi_hz", angle_deg / 360 / duration)))
        phase = np.full(samples, math.radians(phase_deg))
    else:
        classes = {"sine": waveforms.Sine, "triangle": waveforms.Triangle, "linear": waveforms.Linear, "chirp": waveforms.Chirp, "hs1": waveforms.HS1, "sahs1": waveforms.SAHS1, "wurst": waveforms.WURST, "wursttanh": waveforms.WURSTtanh}
        if kind not in classes:
            raise ValueError(f"Unknown pulse kind: {kind}")
        peak = positive(config.require("pulse", "peak_rabi_hz"), "pulse.peak_rabi_hz")
        reserved = {"points", "dwell_s", "amp", "amp_min", "amp_max", "phase", "nutation_hz"} & set(params)
        if reserved:
            raise ValueError(f"Set shared pulse conditions instead of waveform parameters: {sorted(reserved)}")
        defaults = {"points": samples}
        if kind in {"sine", "triangle", "linear"}:
            defaults.update(amp_min=0, amp_max=1)
        else:
            defaults.update(dwell_s=duration / samples, phase=phase_deg)
            if kind != "chirp":
                defaults["amp"] = 1.0
            if kind == "sahs1":
                defaults["nutation_hz"] = peak
        shape = classes[kind](**defaults, **params)
        amp_values = shape.amplitude_values()
        phase_values = shape.phase_values()
        amplitude = peak * (np.asarray(amp_values) if amp_values else np.ones(samples))
        phase = np.deg2rad(phase_values) if phase_values else np.full(samples, math.radians(phase_deg))
        phase = np.unwrap(phase)
    pulse = Pulse(duration, np.asarray(amplitude, float), np.asarray(phase, float))
    ceiling = settings.get("max_rabi_hz")
    if ceiling is not None and np.max(pulse.rabi_hz) > positive(ceiling, "pulse.max_rabi_hz") * (1 + 1e-12):
        raise ValueError("Requested pulse exceeds pulse.max_rabi_hz; adjust duration, amplitude or shape")
    return pulse
