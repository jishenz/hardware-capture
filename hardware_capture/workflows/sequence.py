"""Offline sequence editing with shared pulse generation and named templates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from hardware_capture.configuration import Experiment, positive
from hardware_capture.pulses.conditions import generate
from hardware_capture.pulses.waveforms import Waveform
from hardware_capture.sequences.tps import Sequence
from hardware_capture.sequences.tnt import replace_sequence
from .common import write_json

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources/tnmr"
TEMPLATES = {
    "pulse_phmod": "20260630_pulse_phmod_template",
    "inversion_phase": "20260630_NVNMR_AdiabaticInversionTrain",
    "inversion_ampm": "20260704_NVNMR_AdiabaticInversionTrain_AMPM",
    "nvnmr": "20260620_NVNMR_template",
}


@dataclass
class SampledWaveform(Waveform):
    phases: list | None = None
    amplitudes: list | None = None

    def _raw_phase_values(self):
        return self.phases

    def _raw_amplitude_values(self):
        return self.amplitudes


def duration_seconds(value) -> float:
    match = re.fullmatch(r"\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*([num]?)(?:s)?\s*", str(value))
    if not match:
        raise ValueError(f"Unsupported TNMR duration: {value!r}")
    return float(match[1]) * {"": 1, "n": 1e-9, "u": 1e-6, "m": 1e-3}[match[2]]


def _sampled(config: Experiment, quantity: str) -> SampledWaveform:
    pulse = generate(config)
    amplitudes = None
    if quantity == "amplitude":
        reference = positive(config.calibration().get("rabi_at_100_hz", 0), "calibration.rabi_at_100_hz")
        amplitudes = (100 * pulse.rabi_hz / reference).tolist()
        if max(amplitudes) > 100 + 1e-9:
            raise ValueError("Generated pulse exceeds TNMR amplitude 100")
    return SampledWaveform(phase_steps=65536, phases=np.rad2deg(pulse.phase_rad).tolist(), amplitudes=amplitudes)


def prepare(config: Experiment) -> tuple[Sequence, Path, Path | None]:
    settings = config.section("sequence")
    template = settings.get("template")
    if template and template not in TEMPLATES:
        raise ValueError(f"Unknown sequence.template {template!r}")
    input_path = Path(settings.get("input_path") or (RESOURCE_DIR / (TEMPLATES[template] + ".tps") if template else config.require("sequence", "input_path")))
    tnt_path = settings.get("tnt_path")
    if template and template != "inversion_phase" and tnt_path is None:
        tnt_path = RESOURCE_DIR / (TEMPLATES[template] + ".tnt")
    sequence = Sequence.load_json_file(input_path) if settings.get("json_input", False) else Sequence.load(input_path)
    if template in {"pulse_phmod", "inversion_phase", "inversion_ampm"}:
        pulse_config = config.section("pulse")
        params = pulse_config.get("parameters", {})
        duration = positive(config.require("pulse", "duration_s"), "pulse.duration_s")
        offset = duration_seconds(sequence.parameter_values()["t_mod_offset"])
        if duration <= offset:
            raise ValueError("pulse.duration_s must exceed the template's t_mod_offset")
        if template in {"pulse_phmod", "inversion_phase"}:
            sampled = generate(config)
            if not np.allclose(sampled.rabi_hz, sampled.rabi_hz[0], rtol=1e-10, atol=1e-10):
                raise ValueError("This phase-only template requires constant amplitude; use inversion_ampm for shaped envelopes")
            reference = positive(config.calibration().get("rabi_at_100_hz", 0), "calibration.rabi_at_100_hz")
            percent = 100 * sampled.rabi_hz[0] / reference
            if percent > 100 + 1e-9:
                raise ValueError("Generated pulse exceeds TNMR amplitude 100")
            sequence.set_parameter_value("f1_amp", f"{percent:.12g}")
        # Timing and offsets derive from this pulse profile, never duplicated script constants.
        if template == "pulse_phmod":
            phase_table, time_name = "phmod_pulse", "t_pulse"
            sequence.set_parameter_values({time_name: f"{(duration-offset)*1e6:.12g}u", "freq_offset": str(params.get("offset_hz", 0))})
            sequence.set_waveform_to_table(phase_table, _sampled(config, "phase"), quantity="phase", allow_shared=True)
        else:
            for suffix, direction in [("p180", 1), ("pn180", -1)]:
                reverse = config.with_updates(pulse={"parameters": {"sweep_width_hz": direction * abs(params.get("sweep_width_hz", 200000.0))}})
                sequence.set_parameter_value("t_" + suffix, f"{(duration-offset)*1e6:.12g}u")
                sequence.set_waveform_to_table("phmod_" + suffix, _sampled(reverse, "phase"), quantity="phase", allow_shared=True)
                if template == "inversion_ampm":
                    sequence.set_parameter_value("freq_" + suffix, str(params.get("offset_hz", 0)))
                    sequence.set_waveform_to_table("ampmod_" + suffix, _sampled(reverse, "amplitude"), quantity="amplitude", allow_shared=True)
            if template == "inversion_phase":
                sequence.set_parameter_values({"freq_neg": str(params.get("offset_hz", 0)), "freq_pos": str(params.get("offset_hz", 0))})
            else:
                sequence.set_parameter_value("f1_amp", 100)
    sequence.set_parameter_values(settings.get("parameter_updates", {}))
    for update in settings.get("table_updates", []):
        allowed = {"name", "quantity", "waveform", "pulse", "allow_shared", "phase_units", "precision"}
        if set(update) - allowed:
            raise ValueError(f"Unknown sequence table-update keys: {sorted(set(update)-allowed)}")
        quantity = update["quantity"]
        waveform = Waveform.from_dict(update["waveform"]) if "waveform" in update else _sampled(config.with_updates(pulse=update.get("pulse", {})), quantity)
        sequence.set_waveform_to_table(update["name"], waveform, quantity=quantity, precision=update.get("precision", 3), phase_units=update.get("phase_units", "deg"), allow_shared=update.get("allow_shared", False))
    for assignment in settings.get("assignments", []):
        sequence.assign_table_to_event(**assignment)
    if settings.get("remove_unused", False):
        sequence.remove_unused_tables()
    return sequence, input_path, Path(tnt_path) if tnt_path else None


def edit(config: Experiment, directory: Path) -> dict:
    sequence, input_path, tnt = prepare(config)
    output = directory / "sequence.tps"
    sequence.write_tps(output)
    sequence.write_json(directory / "sequence.json", include_raw_bytes=True)
    if tnt:
        if config.section("sequence").get("json_input", False):
            raise ValueError("Matched TNT output requires the original binary TPS as input")
        replace_sequence(tnt, input_path, output.read_bytes(), directory / "experiment.tnt")
    for table in sequence.tables:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", table.get("name", "table"))
        (directory / f"table_{name}.txt").write_text(str(table.get("entries", "")) + "\n", encoding="utf-8")
    summary = sequence.summary().to_dict()
    write_json(directory / "sequence_summary.json", summary)
    template = config.section("sequence").get("template")
    live_source = tnt or (RESOURCE_DIR / (TEMPLATES[template] + ".tnt") if template else None)
    update_path = None
    if live_source:
        # Some historical TPS/TNT files are not byte-matched. Provide the same
        # explicit COM load/edit/save operation as their original notebooks.
        update_path = directory / "tnmr_update.json"
        write_json(update_path, {"tnmr": {
            "path": str(live_source), "action": "update",
            "sequence_path": str(output), "save_path": str(directory / "tnmr_saved.tnt"),
            "parameters": sequence.parameter_values(),
            "tables": {table["name"]: table.get("entries", "") for table in sequence.tables},
        }})
    return {"sequence_path": str(output), "tnt_path": str(directory / "experiment.tnt") if tnt else None, "tnmr_update_config": str(update_path) if update_path else None, "summary": summary}
