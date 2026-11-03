"""Fake-backed tests for TNMR table and acquisition orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hardware_capture.devices.rigol import ScopePreamble, ScopeWaveform
from hardware_capture.optimization.pipeline import (
    ClosedLoopConfig,
    parse_numeric_table,
    run_closed_loop_optimizer,
    upload_tables_with_readback,
)


pytestmark = pytest.mark.unit


class FakeTNMR:
    def __init__(self) -> None:
        self.tables = {
            "amp": "10\r\n",
            "phase": "20\r\n",
        }
        self.events: list[str] = []

    def get_table_list(self):
        return list(self.tables)

    def get_table(self, name):
        return self.tables[name]

    def set_table(self, name, entries):
        self.events.append(f"set:{name}")
        if isinstance(entries, str):
            self.tables[name] = entries
        else:
            self.tables[name] = "\r\n".join(str(entry) for entry in entries) + "\r\n"

    def zero_and_go(self):
        self.events.append("zg")

    def stop(self):
        self.events.append("stop")


class FakeScope:
    def __init__(self, tnmr: FakeTNMR, *, carrier_hz: float, phase_offset_rad: float) -> None:
        self.tnmr = tnmr
        self.carrier_hz = carrier_hz
        self.phase_offset_rad = phase_offset_rad
        self.capture_count = 0

    def capture_single_waveform(self, config, *, start_shot, **_kwargs):
        self.tnmr.events.append("armed")
        start_shot()
        self.capture_count += 1
        sample_rate = 5e6
        time = np.arange(-20e-6, 121e-6, 1.0 / sample_rate)
        amplitude_table = parse_numeric_table(self.tnmr.tables["amp"]) / 100.0
        phase_table = np.deg2rad(parse_numeric_table(self.tnmr.tables["phase"]))
        relative = np.clip(time / 100e-6, 0.0, 1.0 - np.finfo(float).eps)
        indices = np.floor(relative * len(amplitude_table)).astype(int)
        active = (time >= 0.0) & (time <= 100e-6)
        amplitude = np.zeros_like(time)
        phase = np.zeros_like(time)
        amplitude[active] = amplitude_table[indices[active]]
        phase[active] = phase_table[indices[active]] + self.phase_offset_rad
        i_values = amplitude * np.cos(phase)
        q_values = amplitude * np.sin(phase)
        carrier_phase = 2.0 * math.pi * self.carrier_hz * time
        voltage = i_values * np.cos(carrier_phase) - q_values * np.sin(carrier_phase)
        preamble = ScopePreamble(
            data_format="0",
            data_type="0",
            points=len(time),
            count=1,
            x_increment=1.0 / sample_rate,
            x_origin=float(time[0]),
            x_reference=0.0,
            y_increment=1.0,
            y_origin=0.0,
            y_reference=0.0,
        )
        return ScopeWaveform(
            time=time,
            voltage=voltage,
            sample_rate=sample_rate,
            source="FAKE,RIGOL",
            channel="CHAN1",
            preamble=preamble,
            start_point=1,
            stop_point=len(time),
        )


def test_upload_tables_requires_full_numeric_readback() -> None:
    tnmr = FakeTNMR()
    amplitude = np.asarray([0.0, 12.5, 100.0])
    phase = np.asarray([0.0, 104.4775121859, 313.4325365578])

    upload_tables_with_readback(
        tnmr,
        amplitude_table_name="amp",
        phase_table_name="phase",
        amplitude_percent=amplitude,
        phase_degrees=phase,
    )

    assert parse_numeric_table(tnmr.tables["amp"]) == pytest.approx(amplitude)
    assert parse_numeric_table(tnmr.tables["phase"]) == pytest.approx(phase)


@pytest.mark.parametrize("custom_seed", [False, True])
def test_fake_end_to_end_stage_one_and_one_update_run(tmp_path: Path, custom_seed) -> None:
    tnt_path = tmp_path / "working_copy.tnt"
    tnt_path.write_bytes(b"fake")
    tnmr = FakeTNMR()
    carrier_hz = 500e3
    scope = FakeScope(tnmr, carrier_hz=carrier_hz, phase_offset_rad=0.31)
    omega_100 = 5.0 * math.pi / 100e-6
    config = ClosedLoopConfig(
        tnt_path=tnt_path,
        amplitude_table_name="amp",
        phase_table_name="phase",
        visa_resource="FAKE",
        results_root=tmp_path / "results",
        acquisition_window_start_s=-20e-6,
        acquisition_window_end_s=120e-6,
        pulse_duration_s=100e-6,
        pulse_start_s=0.0,
        demodulation_frequency_hz=carrier_hz,
        baseband_cutoff_hz=200e3,
        q_sign=1,
        scope_omega_rad_s_per_volt=omega_100,
        omega_at_amplitude_100_rad_s=omega_100,
        coefficient_count=25,
        table_entry_count=100,
        phase_reference_captures=2,
        amplitude_errors=np.asarray([0.0]),
        detuning_ratios=np.asarray([0.0]),
        mean_weight=1.0,
        threshold_weight=0.0,
        cvar_weight=0.0,
        update_count=1,
        initial_amplitude_coefficients=tuple([.2] * 25) if custom_seed else None,
        initial_phase_coefficients_rad=tuple([0.] * 25) if custom_seed else None,
    )

    result = run_closed_loop_optimizer(config, tnmr=tnmr, scope=scope)

    assert result.best_epoch in {0, 1}
    assert result.phase_reference.offset_rad == pytest.approx(0.31, abs=0.08)
    assert scope.capture_count == (5 if custom_seed else 4)
    assert tnmr.events[:4] == ["set:amp", "set:phase", "armed", "zg"]
    assert (result.run_directory / "stage1_phase_reference.json").exists()
    assert (result.run_directory / "initial_measured_heatmap.png").exists()
    assert (result.run_directory / "stage1_phase_change.png").exists()
    assert (result.run_directory / "stage1_final_pulse.png").exists()
    assert (result.run_directory / "stage1_final_heatmap.png").exists()
    stage1_summary = json.loads(
        (result.run_directory / "stage1_summary.json").read_text(encoding="utf-8")
    )
    assert stage1_summary["automatic_phase_checks_passed"] is True
    assert stage1_summary["full_stage1_acceptance_requires_graph_review"] is True
    assert math.isfinite(stage1_summary["independent_measured_area_over_5pi"])
    assert math.isfinite(stage1_summary["center_fidelity_to_fixed_xpi"])
    assert (result.run_directory / "checkpoint_latest.npz").exists()
    assert (result.run_directory / "final_measured_heatmap.png").exists()
    assert (result.run_directory / "summary.json").exists()


def test_failure_restores_original_tables_and_stops(tmp_path):
    source = tmp_path / "working.tnt"
    source.write_bytes(b"fake")
    tnmr = FakeTNMR()
    original = tnmr.tables.copy()

    class BrokenScope:
        def capture_single_waveform(self, config, **kwargs):
            raise RuntimeError("synthetic trigger failure")

    config = ClosedLoopConfig(
        tnt_path=source, amplitude_table_name="amp", phase_table_name="phase",
        visa_resource="FAKE", results_root=tmp_path / "output",
        coefficient_count=25, table_entry_count=100,
        phase_reference_captures=2, update_count=0,
    )
    with pytest.raises(RuntimeError, match="synthetic trigger failure"):
        run_closed_loop_optimizer(config, tnmr=tnmr, scope=BrokenScope())
    assert tnmr.tables == original
    assert tnmr.events[-1] == "stop"


def test_public_live_api_uses_shared_conditions_and_injected_devices(tmp_path):
    from hardware_capture.api import run
    from hardware_capture.configuration import Experiment

    source = tmp_path / "working.tnt"
    source.write_bytes(b"fake")
    tnmr = FakeTNMR()
    scope = FakeScope(tnmr, carrier_hz=500000, phase_offset_rad=.31)
    config = Experiment({
        "pulse": {"kind": "square", "duration_s": 100e-6, "samples": 100},
        "calibration": {"rabi_at_100_hz": 25000, "omega_rad_s_per_volt": 2 * math.pi * 25000},
        "scope": {"resource": "FAKE", "window_start_s": -20e-6, "window_end_s": 120e-6},
        "demodulation": {"frequency_hz": 500000, "cutoff_hz": 200000, "q_sign": 1},
        "window": {"start_s": 0},
        "tnmr": {"path": str(source), "amplitude_table": "amp", "phase_table": "phase"},
        "optimization": {"coefficients": 25, "updates": 0, "phase_reference_captures": 2},
        "scan": {"amplitude_errors": [0], "detuning_hz": [0]},
        "output": {"directory": str(tmp_path / "output")},
    })
    result = run("optimize-live", config, live=True, scope=scope, tnmr_device=tnmr)
    assert result["target"] == "fixed_xpi"
    assert scope.capture_count == 4
    assert Path(result["run_directory"]).is_dir()
    assert Path(result["pulse_path"]).is_file()
