"""Shared acquisition, validation and artifact helpers."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np

from hardware_capture.configuration import Experiment, positive
from hardware_capture.devices.rigol import RigolDHO, WaveformCaptureConfig, save_waveform_csv, normalize_channel
from hardware_capture.calibration.bb1 import load_waveform_csv
from hardware_capture.signal.demodulation import demodulate_raw_rf, detect_pulse_window


def json_ready(value):
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_directory(config: Experiment, project: str) -> Path:
    base = Path(config.section("output").get("directory", "outputs"))
    directory = base / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{project}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "resolved_config.json", config.data)
    return directory


def capture_config(config: Experiment) -> WaveformCaptureConfig:
    scope = config.section("scope")
    start, end = scope.get("window_start_s", -10e-6), scope.get("window_end_s", 100e-6)
    if not start < end:
        raise ValueError("scope.window_start_s must precede window_end_s")
    return WaveformCaptureConfig(channel=normalize_channel(scope.get("channel", "CHAN1")), window_start_s=start, window_end_s=end, allow_clamp=scope.get("allow_clamp", False))


def acquire(config: Experiment, *, scope=None):
    source = config.section("input")
    if source.get("kind", "csv") == "csv":
        return load_waveform_csv(config.require("input", "path"))[0]
    if source.get("kind") != "scope":
        raise ValueError("Raw RF input.kind must be csv or scope")
    settings = config.section("scope")
    owned = scope is None
    if owned:
        scope = RigolDHO(resource=config.require("scope", "resource"), timeout_ms=settings.get("timeout_ms", 30000))
    try:
        if owned:
            scope.open()
        capture = capture_config(config)
        if source.get("read_current", False):
            return scope.capture_waveform(capture)
        command = source.get("shot_command")
        callback = (lambda: subprocess.run(command, shell=True, check=True)) if command else None
        return scope.capture_single_waveform(capture, start_shot=callback, trigger_timeout_s=settings.get("trigger_timeout_s", 30), armed_timeout_s=settings.get("armed_timeout_s", 2), poll_s=settings.get("poll_s", 0.05), confirm_armed=True, rearm_after_read=settings.get("rearm", False))
    finally:
        if owned:
            scope.close()


def demodulate(config: Experiment, raw):
    settings = config.section("demodulation")
    return demodulate_raw_rf(raw, demodulation_freq_hz=settings.get("frequency_hz"), baseband_cutoff_hz=settings.get("cutoff_hz", 25e6), filter_order=settings.get("filter_order", 4), q_sign=settings.get("q_sign", -1))


def pulse_window(config: Experiment, demod):
    settings = config.section("window")
    return detect_pulse_window(demod, threshold_fraction=settings.get("threshold_fraction", 0.1), padding_seconds=settings.get("padding_s", 0.25e-6), manual_start=settings.get("start_s"), manual_end=settings.get("end_s"))


def capture(config: Experiment, directory: Path, *, scope=None) -> dict:
    shots = config.section("input").get("shots", 1)
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("input.shots must be a positive integer")
    paths = []
    for i in range(shots):
        raw = acquire(config, scope=scope)
        paths.append(str(save_waveform_csv(directory / f"capture_{i + 1:04d}.csv", raw)))
    return {"captures": paths}
