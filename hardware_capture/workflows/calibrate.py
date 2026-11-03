"""The same calibration operation accepts an offline CSV or a live capture."""
from __future__ import annotations

import math
from pathlib import Path

from hardware_capture.configuration import Experiment, positive
from hardware_capture.calibration import bb1, square
from .common import acquire, write_json


def calibrate(config: Experiment, directory: Path, method: str, *, scope=None) -> dict:
    raw = acquire(config, scope=scope)
    demod = config.section("demodulation")
    settings = config.calibration()
    frequency = positive(config.require("demodulation", "frequency_hz"), "demodulation.frequency_hz")
    kwargs = dict(demodulation_freq_hz=frequency, baseband_cutoff_hz=demod.get("cutoff_hz", 25e6), q_sign=demod.get("q_sign", -1), filter_order=demod.get("filter_order", 4))
    if method == "square":
        reference = positive(settings.get("rabi_at_100_hz", 0), "calibration.rabi_at_100_hz")
        result = square.calibrate_square_waveform(raw, **kwargs, peak_omega_over_2pi_khz=reference / 1000, middle_fraction=settings.get("middle_fraction", 0.6), detection_smoothing_seconds=settings.get("smoothing_s", 0.25e-6), detection_threshold_fraction=settings.get("threshold_fraction", 0.5), baseline_guard_seconds=settings.get("baseline_guard_s", 0.5e-6))
        payload = square.calibration_to_dict(result, raw, q_convention="I+iQ" if kwargs["q_sign"] == 1 else "I-iQ", scope_resource=config.section("scope").get("resource", "offline"))
        payload["rabi_at_100_hz"] = reference
        square.plot_diagnostics(raw, result, figure_path=directory / "diagnostics.png", save_figure=True, show_plot=config.section("output").get("show_plots", False))
    else:
        window = config.section("window")
        result = bb1.calibrate_bb1_waveform(raw, **kwargs, target_angle_rad=math.radians(config.section("pulse").get("angle_deg", 180)), threshold_fraction=window.get("threshold_fraction", 0.1), padding_seconds=window.get("padding_s", 0.25e-6), manual_start_s=window.get("start_s"), manual_end_s=window.get("end_s"), pi_over_2_duration_s=settings.get("pi_over_2_duration_s"))
        payload = bb1.calibration_to_dict(result, raw, input_mode=config.section("input").get("kind", "csv"), q_convention="I+iQ" if kwargs["q_sign"] == 1 else "I-iQ")
        bb1.plot_diagnostics(raw, result, figure_path=directory / "diagnostics.png", save_figure=True, show_plot=config.section("output").get("show_plots", False))
    payload["input"] = config.section("input")
    write_json(directory / "calibration.json", payload)
    return {"calibration_path": str(directory / "calibration.json"), "omega_rad_s_per_volt": result.omega_rad_s_per_volt}
