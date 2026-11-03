"""Contracts across configuration, offline workflows and explicit device boundaries."""
import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hardware_capture.api import plan, run
from hardware_capture.configuration import Experiment
from hardware_capture.pulses.conditions import generate
from hardware_capture.workflows import analyze, common, sequence

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = json.loads((ROOT / "docs/projects.json").read_text())


def preset(folder, tmp_path):
    return Experiment.load(ROOT / "projects" / folder / "config.json").with_updates(output={"directory": str(tmp_path)})


@pytest.mark.parametrize("row", PROJECTS, ids=lambda row: row["folder"])
def test_every_project_has_valid_config_and_thin_notebook(row):
    folder = ROOT / "projects" / row["folder"]
    plan(row["workflow"], folder / "config.json")
    readme = (folder / "README.md").read_text()
    for heading in ["## Input", "## Output", "## Assumptions", "## Run"]:
        assert heading in readme
    notebook = json.loads((folder / "notebook.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            tree = ast.parse(source)
            assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(tree))
            assert cell["outputs"] == []


def test_inheritance_resolves_paths_where_declared_and_overrides_values(tmp_path):
    parent = tmp_path / "profiles"
    parent.mkdir()
    (parent / "base.json").write_text(json.dumps({"input": {"kind": "csv", "path": "data.csv"}, "pulse": {"duration_s": 1, "samples": 40}}))
    child = tmp_path / "child.toml"
    child.write_text('extends = "profiles/base.json"\n[pulse]\nsamples = 80\n')
    config = Experiment.load(child)
    assert config.require("input", "path") == str(parent / "data.csv")
    assert config.section("pulse") == {"duration_s": 1, "samples": 80}
    child.write_text('extends = "child.toml"')
    with pytest.raises(ValueError, match="cycle"):
        Experiment.load(child)


def test_unknown_keys_and_nonfinite_conditions_are_rejected():
    with pytest.raises(ValueError, match="Unknown pulse"):
        Experiment({"pulse": {"duraton_s": 1}})
    with pytest.raises(ValueError, match="finite"):
        Experiment({"pulse": {"duration_s": float("nan")}})


def test_live_preview_never_connects_or_creates_output(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preview connected to hardware")
    monkeypatch.setattr(common, "RigolDHO", forbidden)
    config = preset("acquisition/capture", tmp_path / "absent")
    assert run("capture", config)["dry_run"]
    assert not (tmp_path / "absent").exists()
    invalid = config.with_updates(input={"shots": 0})
    with pytest.raises(ValueError, match="positive integer"):
        run("capture", invalid, live=True)


@pytest.mark.parametrize("kind", ["square", "bb1", "sine", "triangle", "linear", "chirp", "hs1", "sahs1", "wurst", "wursttanh"])
def test_waveform_families_share_units_and_csv_roundtrip(kind, tmp_path):
    config = Experiment({"pulse": {"kind": kind, "duration_s": 60e-6, "samples": 80, "peak_rabi_hz": 50000, "max_rabi_hz": 125000}})
    pulse = generate(config)
    saved = pulse.save(tmp_path, rabi_at_100_hz=125000)
    loaded = generate(config.with_updates(pulse={"kind": "csv", "path": saved["pulse_path"]}))
    assert np.sum(loaded.dt_s) == pytest.approx(60e-6)
    assert loaded.rabi_hz == pytest.approx(pulse.rabi_hz)
    assert np.exp(1j * loaded.phase_rad) == pytest.approx(np.exp(1j * pulse.phase_rad))
    assert np.loadtxt(tmp_path / "tnmr_amplitude.txt").max() <= 100


def test_over_limit_pulse_fails_before_writing_files(tmp_path):
    pulse = generate(Experiment({"pulse": {"kind": "square", "duration_s": 1e-6}}))
    with pytest.raises(ValueError, match="amplitude-100"):
        pulse.save(tmp_path / "not_created", rabi_at_100_hz=1)
    assert not (tmp_path / "not_created").exists()


def test_analysis_target_meanings_and_physical_detuning():
    config = Experiment({"input": {"kind": "pulse"}, "pulse": {"kind": "square", "duration_s": 10e-6, "samples": 40, "axis_phase_deg": 90}, "scan": {"amplitude_errors": [0], "detuning_hz": [0, 50000]}})
    pulse, _, _ = analyze.model(config)
    fixed = analyze.scan(config, pulse)
    nominal = analyze.scan(config.with_updates(analysis={"target": "nominal"}), pulse)
    best = analyze.scan(config.with_updates(analysis={"target": "best_axis_pi"}), pulse)
    assert fixed["center_fidelity"] == pytest.approx(0, abs=1e-12)
    assert nominal["center_fidelity"] == pytest.approx(1)
    assert best["center_fidelity"] == pytest.approx(1)
    assert nominal["fidelity"][1, 0] < 1
    exact = analyze.scan(config.with_updates(analysis={"target": "nominal", "method": "scipy_expm"}), pulse)
    assert exact["fidelity"] == pytest.approx(nominal["fidelity"], abs=1e-12)


def test_csv_and_fake_scope_calibrate_identically_and_artifact_is_reusable(tmp_path):
    config = preset("calibration/square", tmp_path)
    raw = common.acquire(config)

    class Scope:
        def capture_waveform(self, capture):
            return raw

    csv_result = run("calibrate-square", config)
    live_config = config.with_updates(input={"kind": "scope", "read_current": True}, scope={"resource": "FAKE"})
    live_result = run("calibrate-square", live_config, live=True, scope=Scope())
    expected = 2 * math.pi * 100000 / .03
    assert csv_result["omega_rad_s_per_volt"] == pytest.approx(expected, rel=.005)
    assert live_result["omega_rad_s_per_volt"] == csv_result["omega_rad_s_per_volt"]
    calibration = Experiment({"calibration": {"path": csv_result["calibration_path"]}}).calibration()
    assert calibration["rabi_at_100_hz"] == 100000
    assert calibration["omega_rad_s_per_volt"] == csv_result["omega_rad_s_per_volt"]


def test_offline_optimization_improves_demo_and_accepts_other_seeds(tmp_path):
    config = preset("optimization/offline", tmp_path)
    result = run("optimize-offline", config)
    assert result["best_loss"] < result["initial_loss"]
    assert Path(result["pulse_path"]).is_file()
    square = run("optimize-offline", config.with_updates(pulse={"kind": "square"}, optimization={"updates": 0}))
    assert square["target"] == "fixed_xpi"
    with pytest.raises(ValueError, match="fixed_xpi only"):
        plan("optimize-offline", config.with_updates(analysis={"target": "nominal"}))


@pytest.mark.parametrize("folder", ["bb1", "chirp", "inversion-chirp", "inversion-hs1", "inversion-wurst", "inversion-wursttanh", "nvnmr"])
def test_matched_template_edits_preserve_sources_and_roundtrip(folder, tmp_path):
    config = preset("sequences/" + folder, tmp_path)
    prepared, source, tnt = sequence.prepare(config)
    original_tps, original_tnt = source.read_bytes(), tnt.read_bytes() if tnt else None
    result = run("sequence", config)
    assert source.read_bytes() == original_tps
    if tnt:
        assert tnt.read_bytes() == original_tnt
    edited = Path(result["sequence_path"])
    exported = sequence.Sequence.load(edited)
    assert exported.parameter_values() == prepared.parameter_values()
    loaded = sequence.Sequence.load_json_file(edited.with_suffix(".json"))
    assert loaded.parameter_values() == exported.parameter_values()
    if tnt:
        assert Path(result["tnt_path"]).is_file()
    else:
        assert result["tnt_path"] is None
    assert Path(result["tnmr_update_config"]).is_file()


def test_live_reference_limit_is_checked_without_connecting(tmp_path):
    config = preset("optimization/live", tmp_path).with_updates(pulse={"duration_s": 1e-6})
    with pytest.raises(ValueError, match="BB1 phase reference"):
        plan("optimize-live", config)


def test_owned_scope_is_opened_and_closed_even_on_failure(tmp_path, monkeypatch):
    events = []

    class Scope:
        def __init__(self, **kwargs):
            pass

        def open(self):
            events.append("open")

        def capture_waveform(self, config):
            raise RuntimeError("capture failed")

        def close(self):
            events.append("close")

    monkeypatch.setattr(common, "RigolDHO", Scope)
    config = Experiment({"input": {"kind": "scope", "read_current": True}, "scope": {"resource": "FAKE"}})
    with pytest.raises(RuntimeError, match="capture failed"):
        common.acquire(config)
    assert events == ["open", "close"]


def test_generated_tnmr_update_config_reuses_sequence_parameters_and_tables(tmp_path):
    from hardware_capture.workflows import tnmr

    result = run("sequence", preset("sequences/inversion-chirp", tmp_path))
    config = Experiment.load(result["tnmr_update_config"])
    events = []

    class Control:
        def open(self, path): events.append(("open", path))
        def load_sequence(self, path): events.append(("load", path))
        def set_parameter(self, name, value): events.append(("parameter", name, value))
        def set_table(self, name, values): events.append(("table", name, values))
        def snapshot(self): return {"state": "fake"}
        def save_as(self, path): events.append(("save", path))

    tnmr.control(config, tmp_path, tnmr=Control())
    assert events[0][0] == "open"
    assert events[1][0] == "load"
    assert events[-1][0] == "save"
    assert any(e[0:2] == ("parameter", "f1_amp") for e in events)
    assert any(e[0:2] == ("table", "phmod_p180") for e in events)


def test_tnmr_clear_refuses_used_tables_before_save_or_mutation(tmp_path):
    from hardware_capture.workflows import tnmr

    source = sequence.RESOURCE_DIR / "20260630_pulse_phmod_template.tps"

    class Control:
        def open(self, path): pass
        def save_sequence_as(self, path): Path(path).write_bytes(source.read_bytes())
        def save_as(self, path): pytest.fail("Must reject used table before saving")
        def set_table(self, *args): pytest.fail("Must reject used table before mutation")

    config = preset("tnmr/clear-tables", tmp_path).with_updates(tnmr={"clear_tables": ["phmod_pulse"]})
    with pytest.raises(ValueError, match="referenced"):
        tnmr.control(config, tmp_path, tnmr=Control())
