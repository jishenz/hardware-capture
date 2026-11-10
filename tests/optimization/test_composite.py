"""Synthetic reduction and fake-device tests: no instrument connections."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hardware_capture.configuration import Experiment
from hardware_capture.optimization.amplitude import sigmoid_amplitude
from hardware_capture.optimization.composite import (
    CompositeSchedule,
    capture_schedule,
    reduce_composite,
    schedule_from_sequence,
)
from hardware_capture.optimization.pipeline import parse_numeric_table
from hardware_capture.workflows import composite_optimize as workflow

pytestmark = pytest.mark.unit


def schedule(repetitions=20):
    return CompositeSchedule(
        repetitions,
        5e-6,
        20e-6,
        35e-6,
        40e-6,
        90e-6,
        "N_repeat",
        {"N_repeat": str(repetitions)},
        ("ph0",),
        "value",
    )


def signal(plan, *, noise=0.0, missing=None, target=None):
    rng = np.random.default_rng(12)
    time = np.arange(-5e-6, plan.end_s + 10e-6, 0.05e-6)
    iq = np.zeros(time.shape, dtype=complex)
    gains = np.linspace(0.7, 1.4, plan.repetitions)
    offsets = np.linspace(-3.0, 3.0, plan.repetitions)
    shifts = np.linspace(-0.35e-6, 0.35e-6, plan.repetitions)
    for rep in range(plan.repetitions):
        origin = rep * plan.period_s + shifts[rep]
        ref = (time >= origin + plan.reference_start_s) & (
            time < origin + plan.reference_start_s + plan.reference_duration_s
        )
        active = (time >= origin + plan.target_start_s) & (
            time < origin + plan.target_start_s + plan.target_duration_s
        )
        if rep != missing:
            iq[ref] = gains[rep] * np.exp(1j * offsets[rep])
        position = (
            time[active] - origin - plan.target_start_s
        ) / plan.target_duration_s
        desired = (0.3 + 0.1 * np.sin(2 * math.pi * position)) * np.exp(
            1j * (0.2 + 0.8 * position)
        )
        if target is not None:
            grid = (np.arange(len(target)) + 0.5) / len(target)
            desired = np.interp(position, grid, target.real) + 1j * np.interp(
                position, grid, target.imag
            )
        iq[active] = desired * gains[rep] * np.exp(1j * offsets[rep])
    iq += noise * (rng.normal(size=len(time)) + 1j * rng.normal(size=len(time)))
    iq += 0.025 + 0.012j
    return SimpleNamespace(time=time, z=iq)


def test_per_pair_gain_phase_and_timing_removed_before_mean():
    plan = schedule()
    reduced = reduce_composite(signal(plan, noise=0.008), plan, samples=200)
    position = reduced.time_s / plan.target_duration_s
    expected = (0.3 + 0.1 * np.sin(2 * math.pi * position)) * np.exp(
        1j * (0.2 + 0.8 * position)
    )
    corrected = reduced.calibrated_iq_rad_s / (2 * math.pi * 160000)
    mean = reduced.mean_iq_rad_s / (2 * math.pi * 160000)
    assert mean == pytest.approx(np.mean(corrected, axis=0))
    assert np.sqrt(np.mean(abs(mean - expected) ** 2)) < 0.004
    assert np.sqrt(np.mean(abs(mean - expected) ** 2)) < 0.4 * np.sqrt(
        np.mean(abs(corrected - expected) ** 2)
    )
    assert reduced.phase_offsets_rad == pytest.approx(np.linspace(-3, 3, 20), abs=0.01)
    assert reduced.omega_rad_s_per_volt == pytest.approx(
        2 * math.pi * 160000 / np.linspace(0.7, 1.4, 20), rel=0.01
    )
    assert reduced.timing_offsets_s == pytest.approx(
        np.linspace(-0.35e-6, 0.35e-6, 20), abs=0.06e-6
    )


@pytest.mark.parametrize("anchor", [0.0, 25e-6])
def test_scope_origin_override_preserves_relative_timing_and_calibration(anchor):
    plan = schedule(6)
    data = signal(plan, noise=0.001)
    expected = reduce_composite(data, plan, samples=200)
    offset = anchor - plan.reference_start_s
    shifted = SimpleNamespace(time=data.time + offset, z=data.z)
    actual = reduce_composite(
        shifted, plan, samples=200, capture_reference_start_s=anchor
    )
    assert actual.mean_iq_rad_s == pytest.approx(expected.mean_iq_rad_s)
    assert actual.timing_offsets_s == pytest.approx(expected.timing_offsets_s + offset)
    assert actual.schedule_offset_s == pytest.approx(offset)
    assert plan.reference_start_s == 5e-6
    missing = signal(plan, missing=3)
    with pytest.raises(ValueError, match="Pair 3"):
        reduce_composite(
            SimpleNamespace(time=missing.time + offset, z=missing.z),
            plan,
            samples=200,
            capture_reference_start_s=anchor,
        )
    with pytest.raises(ValueError, match="complete"):
        reduce_composite(
            SimpleNamespace(time=shifted.time[:-300], z=shifted.z[:-300]),
            plan,
            samples=200,
            capture_reference_start_s=anchor,
        )


@pytest.mark.parametrize("anchor", [float("nan"), float("inf"), True, "0"])
def test_invalid_capture_origin_rejected(anchor):
    with pytest.raises(ValueError, match="finite timestamp"):
        capture_schedule(schedule(), anchor)


def test_incomplete_record_and_missing_square_rejected():
    plan = schedule()
    data = signal(plan)
    with pytest.raises(ValueError, match="complete"):
        reduce_composite(
            SimpleNamespace(time=data.time[:-300], z=data.z[:-300]), plan, samples=100
        )
    with pytest.raises(ValueError, match="Pair 3"):
        reduce_composite(signal(plan, missing=3), plan, samples=100)


def test_reference_quality_rejects_incoherent_phase():
    plan = schedule(2)
    data = signal(plan)
    ref = (data.time > plan.reference_start_s) & (
        data.time < plan.reference_start_s + plan.reference_duration_s
    )
    data.z[ref] *= np.exp(1j * np.linspace(0, 4 * math.pi, np.count_nonzero(ref)))
    with pytest.raises(ValueError, match="unstable"):
        reduce_composite(data, plan, samples=100)


def sequence_fixture():
    def event(value, table=""):
        return {"data": value, "table_slots": [{"name": table}]}

    rows = {
        "Delay": ["5u", "2u", "=P_aux", "220n", "10u", "=P(90)", "220n", "10u"],
        "Loop": ["0", "=N_repeat", "0", "0", "0", "0", "0", "-1"],
        "Scope_Trig": ["1", "0", "0", "0", "0", "0", "0", "0"],
        "F1_TxGate": ["0", "0", "1", "1", "0", "1", "1", "0"],
        "F1_Ampl": ["0", "0", "100", "100", "0", "100", "100", "0"],
        "F1_Ph": ["0"] * 8,
        "F1_PhMod": ["-1"] * 8,
    }
    result = []
    for label, values in rows.items():
        events = [event(label)] + [event(v) for v in values]
        if label == "F1_Ampl":
            events[6] = event("100", "amp")
        if label == "F1_PhMod":
            events[6] = event("-1", "phase")
        result.append({"label": label, "events": events})
    parameters = [
        {"name": name, "value": value, "current_value": current}
        for name, value, current in (
            ("N_repeat", "6", "20"),
            ("P_aux", "59.78u", "10u"),
            ("P(90)", "59.78u", "2u"),
        )
    ]
    return SimpleNamespace(rows=result, parameters=parameters)


def test_current_repetition_count_with_explicit_duration_requirements():
    plan = schedule_from_sequence(
        sequence_fixture(),
        "amp",
        "phase",
        parameter_source="current_value",
        parameter_overrides={"P_aux": "59.78u", "P(90)": "59.78u"},
    )
    assert plan.repetitions == 20
    assert plan.reference_duration_s == pytest.approx(60e-6)
    assert plan.target_duration_s == pytest.approx(60e-6)
    assert plan.period_s == pytest.approx(142e-6)
    with pytest.raises(ValueError, match="repetition count"):
        schedule_from_sequence(
            sequence_fixture(), "amp", "phase", parameter_overrides={"N_repeat": "1"}
        )


@pytest.mark.parametrize("repetitions", [1, 6, 20])
@pytest.mark.parametrize("anchor", [0.0, 25e-6])
def test_sequence_window_follows_repetitions_and_capture_origin(
    monkeypatch, repetitions, anchor
):
    sequence = sequence_fixture()
    next(p for p in sequence.parameters if p["name"] == "N_repeat")["value"] = str(
        repetitions
    )
    monkeypatch.setattr(workflow, "load_sequence_file", lambda _: sequence)
    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"},
        scope={"window_start_s": -0.001, "window_end_s": 0.004},
        composite={"capture_window": "sequence", "capture_reference_start_s": anchor},
    )
    effective, plan = workflow.prepare(config)
    captured = capture_schedule(plan, anchor)
    assert effective.section("scope")["window_start_s"] == pytest.approx(anchor - 10e-6)
    assert effective.section("scope")["window_end_s"] == pytest.approx(
        captured.end_s + 20e-6
    )
    assert config.section("scope")["window_end_s"] == 0.004
    assert plan.repetitions == repetitions


@pytest.mark.parametrize(
    "option,value",
    [
        ("capture_window", "unknown"),
        ("capture_pre_margin_s", 0),
        ("capture_pre_margin_s", True),
        ("capture_post_margin_s", -1e-6),
        ("capture_post_margin_s", 1e-6),
    ],
)
def test_invalid_sequence_window_rejected_before_devices(monkeypatch, option, value):
    monkeypatch.setattr(workflow, "load_sequence_file", lambda _: sequence_fixture())
    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"},
        composite={"capture_window": "sequence", option: value},
    )
    with pytest.raises(ValueError, match=f"composite.{option}"):
        workflow.prepare(config)


def test_configured_window_remains_available(monkeypatch):
    monkeypatch.setattr(workflow, "load_sequence_file", lambda _: sequence_fixture())
    config = config_for_test().with_updates(tnmr={"path": "not-opened.tnt"})
    effective, _ = workflow.prepare(config)
    assert effective.section("scope") == config.section("scope")


class FakeTNMR:
    def __init__(self, reps=3, entries=100):
        self.tables = {"amp": [30.0] * entries, "phase": [15.0] * entries, "ph0": [0.0]}
        self.reps = reps
        self.shots = 0
        self.shot_tables = []
        self.stopped = False

    def get_parameter(self, name):
        return str(self.reps)

    def snapshot(self):
        return {"acquisition_complete": True}

    def get_table_list(self):
        return list(self.tables)

    def get_table(self, name):
        return self.tables[name]

    def set_table(self, name, entries):
        self.tables[name] = entries

    def zero_and_go(self):
        self.shot_tables.append(
            {
                name: parse_numeric_table(value).copy()
                for name, value in self.tables.items()
            }
        )
        self.shots += 1

    def stop(self):
        self.stopped = True


class FakeScope:
    def __init__(self, tnmr, plan):
        self.tnmr, self.plan = tnmr, plan

    def capture_single_waveform(self, config, *, start_shot, **kwargs):
        assert config.allow_clamp is False
        assert kwargs["confirm_armed"] is True
        start_shot()
        target = (
            parse_numeric_table(self.tnmr.tables["amp"])
            / 100
            * np.exp(1j * np.deg2rad(parse_numeric_table(self.tnmr.tables["phase"])))
        )
        measured = signal(self.plan, noise=0.003, target=target)
        return SimpleNamespace(time=measured.time, voltage=measured.z, sample_rate=20e6)


def config_for_test():
    return Experiment(
        {
            "pulse": {"duration_s": 40e-6, "samples": 100, "max_rabi_hz": 160000},
            "optimization": {"coefficients": 10, "updates": 1},
            "calibration": {"rabi_at_100_hz": 160000},
            "tnmr": {"amplitude_table": "amp", "phase_table": "phase"},
            "scope": {"resource": "FAKE", "window_end_s": 0.001},
            "demodulation": {"frequency_hz": 1e6},
            "scan": {"amplitude_errors": [0.0], "detuning_hz": [0.0]},
        }
    )


def test_each_spsa_probe_uses_fresh_calibration_and_one_mean_loss(
    tmp_path, monkeypatch
):
    plan = schedule(3)
    config = config_for_test()
    tnmr = FakeTNMR()
    scope = FakeScope(tnmr, plan)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    original_objective = workflow.robustness_objective
    evaluations = []

    def evaluate(i, q, dt, objective):
        evaluations.append(i + 1j * q)
        return original_objective(i, q, dt, objective)

    monkeypatch.setattr(workflow, "robustness_objective", evaluate)
    result = workflow.run_composite(config, tmp_path, tnmr=tnmr, scope=scope)
    assert tnmr.shots == 4
    assert len(evaluations) == 4
    labels = [
        "epoch_0000",
        "epoch_0001_direction_000_plus",
        "epoch_0001_direction_000_minus",
        "final_verification",
    ]
    for epoch, label in enumerate(labels):
        with np.load(tmp_path / f"{label}.npz") as saved:
            assert saved["calibrated_iq_rad_s"].shape == (3, 100)
            assert evaluations[epoch] == pytest.approx(
                saved["mean_iq_rad_s"] / (2 * math.pi * 160000)
            )
    with np.load(result["best_measurement_path"]) as best:
        assert parse_numeric_table(tnmr.tables["amp"]) == pytest.approx(
            best["amplitude_percent"]
        )
    assert (tmp_path / "tnmr_amplitude.txt").exists()
    assert (tmp_path / "checkpoint_latest.npz").exists()
    assert result["gradient_method"] == "spsa"
    assert result["candidate_evaluations"] == 4
    assert result["final_verification_loss"] == pytest.approx(result["best_loss"])


def test_scope_zero_anchor_reaches_every_candidate(tmp_path, monkeypatch):
    plan = schedule(3)
    config = config_for_test().with_updates(
        composite={"capture_reference_start_s": 0.0}
    )
    tnmr = FakeTNMR()
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    result = workflow.run_composite(
        config, tmp_path, tnmr=tnmr, scope=FakeScope(tnmr, capture_schedule(plan, 0.0))
    )
    assert result["candidate_evaluations"] == tnmr.shots == 4
    for label in (
        "epoch_0000",
        "epoch_0001_direction_000_plus",
        "epoch_0001_direction_000_minus",
        "final_verification",
    ):
        with np.load(tmp_path / f"{label}.npz") as measured:
            assert measured["schedule_offset_s"] == pytest.approx(-5e-6)
            assert measured["timing_offsets_s"] == pytest.approx(
                -5e-6 + np.linspace(-0.35e-6, 0.35e-6, 3), abs=0.06e-6
            )


def test_sequence_window_is_passed_to_every_scope_read(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(workflow, "load_sequence_file", lambda _: sequence_fixture())
    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"},
        scope={"window_start_s": -0.001, "window_end_s": 0.004, "waveform_format": "ASC"},
        composite={"capture_window": "sequence", "capture_reference_start_s": 0.0},
    )
    effective, plan = workflow.prepare(config)
    tnmr = FakeTNMR(reps=6)
    monkeypatch.setattr(tnmr, "get_parameter", lambda name: plan.parameters[name])
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    requests = []

    class CheckedScope(FakeScope):
        def capture_single_waveform(self, capture_config, **kwargs):
            requests.append(capture_config)
            assert capture_config.waveform_format == "ASC"
            assert (
                capture_config.window_start_s
                == effective.section("scope")["window_start_s"]
            )
            assert (
                capture_config.window_end_s
                == effective.section("scope")["window_end_s"]
            )
            return super().capture_single_waveform(capture_config, **kwargs)

    workflow.run_composite(
        config,
        tmp_path,
        tnmr=tnmr,
        scope=CheckedScope(tnmr, capture_schedule(plan, 0.0)),
    )
    assert len(requests) == 4  # Initial, both SPSA probes, final verification.
    saved = json.loads((tmp_path / "effective_config.json").read_text())
    assert saved["scope"]["window_end_s"] == effective.section("scope")["window_end_s"]


def test_failure_restores_tables_and_stops(tmp_path, monkeypatch):
    plan = schedule(3)
    tnmr = FakeTNMR()
    original = dict(tnmr.tables)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))

    class BrokenScope:
        def capture_single_waveform(self, *args, **kwargs):
            raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        workflow.run_composite(
            config_for_test(), tmp_path, tnmr=tnmr, scope=BrokenScope()
        )
    assert tnmr.tables == original
    assert tnmr.stopped


def test_stale_repetition_parameter_fails_before_upload(tmp_path, monkeypatch):
    tnmr = FakeTNMR(reps=2)
    original = dict(tnmr.tables)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, schedule(3)))
    with pytest.raises(ValueError, match="differs"):
        workflow.run_composite(config_for_test(), tmp_path, tnmr=tnmr, scope=object())
    assert tnmr.shots == 0
    assert tnmr.tables == original


def test_required_pulse_duration_mismatch_fails_before_upload(tmp_path, monkeypatch):
    plan = schedule_from_sequence(
        sequence_fixture(),
        "amp",
        "phase",
        parameter_overrides={"P_aux": "59.78u", "P(90)": "59.78u"},
    )
    tnmr = FakeTNMR(reps=6)
    original = dict(tnmr.tables)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        tnmr, "get_parameter", lambda name: "6" if name == "N_repeat" else "10u"
    )
    with pytest.raises(ValueError, match="P_aux differs"):
        workflow.run_composite(config_for_test(), tmp_path, tnmr=tnmr, scope=object())
    assert tnmr.shots == 0
    assert tnmr.tables == original


def test_target_table_cannot_control_square_reference_phase():
    sequence = sequence_fixture()
    row = next(row for row in sequence.rows if row["label"] == "F1_Ph")
    row["events"][3]["table_slots"][0]["name"] = "phase"
    with pytest.raises(ValueError, match="reference phase"):
        schedule_from_sequence(sequence, "amp", "phase")


def test_unknown_acquisition_state_fails_before_upload(tmp_path, monkeypatch):
    tnmr = FakeTNMR()
    original = dict(tnmr.tables)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, schedule(3)))
    monkeypatch.setattr(tnmr, "snapshot", lambda: {"acquisition_complete": None})
    with pytest.raises(RuntimeError, match="state unknown"):
        workflow.run_composite(config_for_test(), tmp_path, tnmr=tnmr, scope=object())
    assert tnmr.shots == 0
    assert tnmr.tables == original
    assert not tnmr.stopped


def test_cleanup_does_not_restore_while_state_unknown(tmp_path, monkeypatch):
    tnmr = FakeTNMR()
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, schedule(3)))

    class BrokenScope:
        def capture_single_waveform(self, *args, **kwargs):
            monkeypatch.setattr(
                tnmr, "snapshot", lambda: {"acquisition_complete": None}
            )
            monkeypatch.setattr(
                tnmr, "set_table", lambda *args: pytest.fail("Unsafe restore")
            )
            raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        workflow.run_composite(
            config_for_test(), tmp_path, tnmr=tnmr, scope=BrokenScope()
        )
    assert tnmr.stopped
    assert "restoration skipped" in (tmp_path / "cleanup_errors.json").read_text()


def test_public_preview_resolves_timing_without_devices(monkeypatch):
    from hardware_capture.api import run

    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"},
        composite={"parameter_overrides": {"P_aux": "59.78u", "P(90)": "59.78u"}},
        scope={"window_end_s": 0.004},
    )
    monkeypatch.setattr(workflow, "load_sequence_file", lambda path: sequence_fixture())
    monkeypatch.setattr(
        workflow, "TNMRControl", lambda **kwargs: pytest.fail("Device opened")
    )
    monkeypatch.setattr(
        workflow, "RigolDHO", lambda **kwargs: pytest.fail("Device opened")
    )
    result = run("optimize-composite", config)
    assert result["dry_run"]
    plan = result["plan"]["composite_schedule"]
    assert plan["repetitions"] == 6
    assert plan["parameter_source"] == "value"
    assert plan["target_duration_s"] == pytest.approx(60e-6)
    optimizer = result["plan"]["optimizer"]
    assert optimizer["gradient_method"] == "spsa"
    assert optimizer["parameters"] == 20
    assert optimizer["table_samples"] == 100
    assert optimizer["candidate_evaluations"] == 4


def test_raw_rf_demodulation_then_pair_reduction():
    from hardware_capture.signal.demodulation import demodulate_raw_rf

    plan = schedule(3)
    baseband = signal(plan, noise=0.001)
    carrier = 2e6
    raw = SimpleNamespace(
        time=baseband.time,
        voltage=np.real(baseband.z * np.exp(2j * math.pi * carrier * baseband.time)),
        sample_rate=20e6,
    )
    measured = demodulate_raw_rf(raw, carrier, baseband_cutoff_hz=0.8e6, q_sign=1)
    reduced = reduce_composite(measured, plan, samples=200)
    position = reduced.time_s / plan.target_duration_s
    expected = (0.3 + 0.1 * np.sin(2 * math.pi * position)) * np.exp(
        1j * (0.2 + 0.8 * position)
    )
    # Exclude filter transients at the discontinuous target edges.
    mean = reduced.mean_iq_rad_s / (2 * math.pi * 160000)
    assert mean[10:-10] == pytest.approx(expected[10:-10], abs=0.005)


def test_value_fields_are_default_authority_without_overrides():
    plan = schedule_from_sequence(sequence_fixture(), "amp", "phase")
    assert plan.parameter_source == "value"
    assert plan.repetitions == 6
    assert plan.parameters["P_aux"] == "59.78u"
    assert plan.parameters["P(90)"] == "59.78u"
    assert plan.reference_duration_s == pytest.approx(60e-6)
    assert plan.target_duration_s == pytest.approx(60e-6)
    assert plan.period_s == pytest.approx(142e-6)
    with pytest.raises(ValueError, match="literal parser fields"):
        schedule_from_sequence(
            sequence_fixture(), "amp", "phase", parameter_source="default"
        )


def bb1_config():
    return config_for_test().with_updates(
        pulse={
            "kind": "bb1",
            "angle_deg": 180,
            "axis_phase_deg": 0,
            "duration_s": 60e-6,
            "samples": 6000,
        },
        optimization={"coefficients": 1500},
        composite={"seed": "configured", "parameter_source": "value"},
    )


def test_exact_symmetric_bb1_is_uploaded_before_first_capture(tmp_path, monkeypatch):
    config = bb1_config()
    plan = replace(schedule(6), target_duration_s=60e-6, period_s=110e-6)
    tnmr = FakeTNMR(reps=6, entries=2000)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    result = workflow.run_composite(
        config, tmp_path, tnmr=tnmr, scope=FakeScope(tnmr, plan)
    )
    first = tnmr.shot_tables[0]
    phi = math.degrees(math.acos(-0.25))
    expected_phases = np.repeat([0, phi, 3 * phi, phi, 0], [600, 1200, 2400, 1200, 600])
    assert len(first["amp"]) == len(first["phase"]) == 6000
    assert first["amp"] == pytest.approx(np.full(6000, 100 * (2.5 / 60e-6) / 160000))
    assert first["phase"] == pytest.approx(expected_phases, abs=1e-10)
    assert first["ph0"] == pytest.approx([0])
    assert tnmr.shots == 4
    assert len(tnmr.shot_tables[1]["phase"]) == 6000
    assert not np.allclose(tnmr.shot_tables[1]["phase"], first["phase"])
    ideal = (first["amp"] / 100) * np.exp(1j * np.deg2rad(first["phase"]))
    evaluation = workflow.robustness_objective(
        ideal.real, ideal.imag, np.full(6000, 10e-9), workflow.objective(config)
    )
    assert evaluation.objective.center_fidelity == pytest.approx(1, abs=1e-10)
    assert np.loadtxt(tmp_path / "initial_seed/tnmr_phase_deg.txt") == pytest.approx(
        expected_phases
    )
    with np.load(tmp_path / "epoch_0000.npz") as saved:
        assert saved["calibrated_iq_rad_s"].shape == (6, 6000)
        assert saved["phase_coefficients_rad"] == pytest.approx(np.zeros(1500))
        assert np.mod(np.rad2deg(saved["phase_baseline_rad"]), 360) == pytest.approx(
            expected_phases, abs=1e-10
        )
    with np.load(result["best_measurement_path"]) as best:
        assert parse_numeric_table(tnmr.tables["phase"]) == pytest.approx(
            np.mod(np.rad2deg(best["command_phase_unwrapped_rad"]), 360), abs=1e-10
        )


def test_configured_seed_restores_original_table_lengths_on_failure(
    tmp_path, monkeypatch
):
    tnmr = FakeTNMR(reps=6, entries=2000)
    original = dict(tnmr.tables)
    plan = replace(schedule(6), target_duration_s=60e-6, period_s=110e-6)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))

    class BrokenScope:
        def capture_single_waveform(self, *args, **kwargs):
            assert len(parse_numeric_table(tnmr.tables["amp"])) == 6000
            assert len(parse_numeric_table(tnmr.tables["phase"])) == 6000
            raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        workflow.run_composite(bb1_config(), tmp_path, tnmr=tnmr, scope=BrokenScope())
    assert tnmr.tables == original
    assert tnmr.stopped


def test_bb1_seed_limits_and_segment_grid_rejected_offline():
    with pytest.raises(ValueError, match="too small"):
        workflow._configured_seed_controls(
            bb1_config().with_updates(pulse={"duration_s": 60e-9})
        )
    with pytest.raises(ValueError, match="divisible by 10"):
        workflow._configured_seed_controls(
            bb1_config().with_updates(pulse={"samples": 6001})
        )


def test_logit_adam_chain_checkpoint_and_next_upload(tmp_path, monkeypatch):
    config = config_for_test().with_updates(
        pulse={"max_rabi_hz": 80000},
        composite={"amplitude_logit_learning_rate": 0.13, "amplitude_logit_step": 0.07},
    )
    plan = schedule(3)
    tnmr = FakeTNMR()
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    actual_adam = workflow.adam_update
    calls = []

    def adam(parameters, gradient, state, **kwargs):
        output, state = actual_adam(parameters, gradient, state, **kwargs)
        calls.append((parameters.copy(), gradient.copy(), output.copy(), kwargs))
        return output, state

    monkeypatch.setattr(workflow, "adam_update", adam)
    result = workflow.run_composite(
        config, tmp_path, tnmr=tnmr, scope=FakeScope(tnmr, plan)
    )
    assert result["amplitude_parameterization"] == "sigmoid"
    assert len(calls) == 2
    z, gz, next_z, options = calls[0]
    assert z == pytest.approx(np.full(10, math.log(0.3 / (0.5 - 0.3))))
    assert options == {"learning_rate": 0.13, "maximum_step": 0.07}
    spline = workflow.clamped_bspline_design_matrix(100, 10)
    with np.load(tmp_path / "epoch_0001_spsa.npz") as pair:
        expected = np.mean(
            (pair["plus_losses"] - pair["minus_losses"])[:, None]
            / (2 * pair["steps"] * pair["directions"]),
            axis=0,
        )
        assert gz == pytest.approx(expected[:10])
        assert calls[1][1] == pytest.approx(expected[10:])
        for shot, key in ((1, "plus_parameters"), (2, "minus_parameters")):
            probe_coefficients, _ = sigmoid_amplitude(pair[key][0, :10], 0.5)
            assert tnmr.shot_tables[shot]["amp"] == pytest.approx(
                100 * np.asarray(spline @ probe_coefficients)
            )
    with np.load(tmp_path / "epoch_0000.npz") as saved:
        assert saved["amplitude_logits"] == pytest.approx(z)
    with np.load(tmp_path / "checkpoint_latest.npz") as checkpoint:
        assert str(checkpoint["amplitude_parameterization"]) == "sigmoid"
        assert checkpoint["amplitude_maximum"] == 0.5
        assert checkpoint["amplitude_logits"] == pytest.approx(next_z)
        assert checkpoint["amplitude_logit_first"] == pytest.approx(0.1 * gz)
        assert checkpoint["amplitude_logit_second"] == pytest.approx(0.001 * gz**2)
        coefficients, _ = sigmoid_amplitude(next_z, 0.5)
        assert checkpoint["amplitude_coefficients"] == pytest.approx(coefficients)
        assert not bool(checkpoint["center_is_measured"])
        assert str(checkpoint["gradient_method"]) == "spsa"
    for capture in tnmr.shot_tables:
        assert np.all((capture["amp"] > 0) & (capture["amp"] < 50))


@pytest.mark.parametrize(
    "bad_settings",
    [
        {"amplitude_logit_learning_rate": 0},
        {"amplitude_logit_step": -0.01},
        {"amplitude_logit_step": True},
    ],
)
def test_logit_settings_rejected_before_devices(bad_settings, monkeypatch):
    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"}, composite=bad_settings
    )
    monkeypatch.setattr(workflow, "load_sequence_file", lambda path: sequence_fixture())
    with pytest.raises(ValueError, match="finite and positive"):
        workflow.prepare(config)


def test_legacy_amplitude_step_units_not_silently_reinterpreted(monkeypatch):
    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"}, optimization={"amplitude_step": 0.02}
    )
    monkeypatch.setattr(workflow, "load_sequence_file", lambda path: sequence_fixture())
    with pytest.raises(ValueError, match="replace optimization"):
        workflow.prepare(config)


@pytest.mark.parametrize("failed_capture,completed_updates", [(3, 0), (5, 1)])
def test_failed_probe_preserves_last_complete_checkpoint(
    tmp_path, monkeypatch, failed_capture, completed_updates
):
    plan = schedule(3)
    config = config_for_test().with_updates(optimization={"updates": 2})
    tnmr = FakeTNMR()
    original = dict(tnmr.tables)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    actual_adam = workflow.adam_update
    updates = []

    def adam(*args, **kwargs):
        updates.append(1)
        return actual_adam(*args, **kwargs)

    class MissingSquare(FakeScope):
        def capture_single_waveform(self, *args, **kwargs):
            raw = super().capture_single_waveform(*args, **kwargs)
            if tnmr.shots == failed_capture:
                raw.voltage[:] = 0
            return raw

    monkeypatch.setattr(workflow, "adam_update", adam)
    with pytest.raises(ValueError, match="square reference"):
        workflow.run_composite(
            config, tmp_path, tnmr=tnmr, scope=MissingSquare(tnmr, plan)
        )
    assert len(updates) == 2 * completed_updates
    checkpoint = tmp_path / "checkpoint_latest.npz"
    assert checkpoint.exists() == bool(completed_updates)
    if completed_updates:
        with np.load(checkpoint) as state:
            assert state["adam_step"] == completed_updates
    failed_epoch = completed_updates + 1
    failed_member = "minus" if completed_updates == 0 else "plus"
    assert (
        tmp_path / f"epoch_{failed_epoch:04d}_direction_000_{failed_member}_raw.npz"
    ).exists()
    assert not (tmp_path / f"epoch_{failed_epoch:04d}_spsa.npz").exists()
    assert tnmr.tables == original


@pytest.mark.parametrize("updates,directions", [(0, 1), (2, 2)])
def test_capture_counts_order_and_complete_scalar_loss(
    tmp_path, monkeypatch, updates, directions
):
    from hardware_capture.optimization import core

    plan = schedule(3)
    config = config_for_test().with_updates(
        optimization={"updates": updates}, spsa={"directions": directions}
    )
    tnmr = FakeTNMR()
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    monkeypatch.setattr(
        core, "robustness_grape_gradient", lambda *_: pytest.fail("GRAPE backward pass")
    )
    actual_objective = workflow.robustness_objective

    def full_loss(i, q, dt, objective):
        evaluation = actual_objective(i, q, dt, objective)
        # Deliberately add a dominant term without implementing its derivative.
        loss = evaluation.objective.loss + 10 * float(np.mean(i**2 + q**2))
        return replace(evaluation, objective=replace(evaluation.objective, loss=loss))

    monkeypatch.setattr(workflow, "robustness_objective", full_loss)
    result = workflow.run_composite(
        config, tmp_path, tnmr=tnmr, scope=FakeScope(tnmr, plan)
    )
    assert tnmr.shots == result["candidate_evaluations"] == 2 + 2 * directions * updates
    if updates:
        with np.load(tmp_path / "epoch_0002_spsa.npz") as pair:
            assert pair["capture_order"].tolist() == [
                "minus_then_plus",
                "plus_then_minus",
            ]
            for direction in range(directions):
                for member in ("plus", "minus"):
                    with np.load(
                        tmp_path / f"epoch_0002_direction_{direction:03d}_{member}.npz"
                    ) as measured:
                        assert pair[f"{member}_losses"][direction] == measured["loss"]
                        assert measured["calibrated_iq_rad_s"].shape == (3, 100)
            expected = np.mean(
                (pair["plus_losses"] - pair["minus_losses"])[:, None]
                / (2 * pair["steps"] * pair["directions"]),
                axis=0,
            )
            assert pair["gradient"] == pytest.approx(expected)
        with np.load(tmp_path / "checkpoint_latest.npz") as checkpoint:
            assert checkpoint["adam_step"] == 2
    else:
        assert not (tmp_path / "checkpoint_latest.npz").exists()


def test_invalid_spsa_is_rejected_in_preview(monkeypatch):
    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"}, spsa={"phase_perturbation_rad": 0}
    )
    monkeypatch.setattr(workflow, "load_sequence_file", lambda path: sequence_fixture())
    with pytest.raises(ValueError, match="spsa.phase_perturbation_rad"):
        workflow.prepare(config)


def _operator_script():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "projects/optimization/composite/run_15_epochs.py"
    )
    spec = importlib.util.spec_from_file_location("composite_operator_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_1000_epoch_script_preview_cannot_enter_hardware(tmp_path, monkeypatch, capsys):
    import json

    script = _operator_script()
    config = bb1_config().with_updates(
        tnmr={"path": "not-opened.tnt"}, optimization={"updates": 1000}
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.data))
    monkeypatch.setattr(workflow, "load_sequence_file", lambda _: sequence_fixture())
    monkeypatch.setattr(
        script, "run", lambda *_args, **_kwargs: pytest.fail("Hardware branch entered")
    )
    assert script.main(["--config", str(path)]) == 0
    output = capsys.readouterr().out
    assert '"updates": 1000' in output
    assert '"parameters": 3000' in output
    assert '"table_samples": 6000' in output
    assert '"candidate_evaluations": 2002' in output
    assert "Preview only" in output


def test_15_update_bb1_run_writes_real_loss_history_with_fake_devices(
    tmp_path, monkeypatch
):
    import csv
    import json

    script = _operator_script()
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(bb1_config().with_updates(optimization={"updates": 15}).data)
    )
    config = script.load_config(path)
    plan = replace(schedule(6), target_duration_s=60e-6, period_s=110e-6)
    tnmr = FakeTNMR(reps=6, entries=2000)
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    result = workflow.run_composite(
        config, tmp_path, tnmr=tnmr, scope=FakeScope(tnmr, plan)
    )
    assert tnmr.shots == result["candidate_evaluations"] == 32
    with (tmp_path / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    training = [row for row in rows if row["kind"] != "final_verification"]
    assert [int(row["epoch"]) for row in training] == list(range(16))
    assert [row["kind"] for row in training] == ["initial"] + ["spsa"] * 15
    assert rows[-1]["kind"] == "final_verification"
    assert all(math.isfinite(float(row["spsa_plus_loss_mean"])) for row in training[1:])
    assert (tmp_path / "loss_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result["loss_curve_path"] == str(tmp_path / "loss_curve.png")
    with np.load(tmp_path / "checkpoint_latest.npz") as checkpoint:
        assert checkpoint["adam_step"] == 15
        assert checkpoint["amplitude_logits"].shape == (1500,)
    from hardware_capture.optimization.history import plot_saved_history

    assert plot_saved_history(tmp_path) == tmp_path / "loss_curve.png"


def scripted_acceptance_run(monkeypatch, values, *, updates=1):
    """Use real acquisition/update plumbing with controlled (loss, fidelity) outcomes."""
    config = config_for_test().with_updates(
        optimization={"updates": updates},
        composite={"minimum_center_fidelity": 0.99},
    )
    plan = schedule(3)
    tnmr = FakeTNMR()
    monkeypatch.setattr(workflow, "prepare", lambda config: (config, plan))
    monkeypatch.setattr(
        workflow,
        "demodulate_raw_rf",
        lambda raw, **kwargs: SimpleNamespace(time=raw.time, z=raw.voltage),
    )
    monkeypatch.setattr(workflow, "_heatmap", lambda *args, **kwargs: None)
    real_objective = workflow.robustness_objective
    outcomes = iter(values)

    def evaluate(*args):
        result = real_objective(*args)
        loss, fidelity = next(outcomes)
        return replace(
            result,
            objective=replace(result.objective, loss=loss, center_fidelity=fidelity),
        )

    monkeypatch.setattr(workflow, "robustness_objective", evaluate)
    return config, tnmr, FakeScope(tnmr, plan)


def test_fidelity_floor_backtracks_and_excludes_better_infeasible_probe(
    tmp_path, monkeypatch
):
    import csv
    import json

    config, tnmr, scope = scripted_acceptance_run(
        monkeypatch,
        [
            (0.5, 0.995),  # Seed.
            (0.1, 0.95),
            (0.6, 0.995),  # Lower-loss probe must not win.
            (0.2, 0.98),
            (0.4, 0.99),  # Full step fails; half step passes exactly.
            (0.4, 0.99),
            (0.4, 0.991),
            (0.4, 0.992),
        ],
    )
    result = workflow.run_composite(config, tmp_path, tnmr=tnmr, scope=scope)
    assert result["best_loss"] == 0.4
    assert result["best_center_fidelity"] == 0.99
    assert result["minimum_verified_center_fidelity"] == 0.99
    assert len(result["final_verifications"]) == 3
    assert result["candidate_evaluations"] == tnmr.shots == 8
    with np.load(tmp_path / "epoch_0001_spsa.npz") as pair:
        # SPSA sees robustness loss + 10 * ideal-fidelity shortfall.
        assert pair["plus_losses"][0] == pytest.approx(0.5)
        center, gradient = pair["center_parameters"], pair["gradient"]
    z, _ = workflow.adam_update(
        center[:10], gradient[:10], None, learning_rate=0.02, maximum_step=0.02
    )
    phase, _ = workflow.adam_update(
        center[10:], gradient[10:], None, learning_rate=0.02, maximum_step=0.1
    )
    with np.load(tmp_path / "checkpoint_latest.npz") as checkpoint:
        assert checkpoint["amplitude_logits"] == pytest.approx((center[:10] + z) / 2)
        assert checkpoint["phase_coefficients_rad"] == pytest.approx(
            (center[10:] + phase) / 2
        )
        assert checkpoint["adam_step"] == 1
        assert checkpoint["center_is_measured"]
        assert checkpoint["center_fidelity"] == 0.99
    decisions = json.loads((tmp_path / "epoch_0001_acceptance.json").read_text())
    assert [a["accepted"] for a in decisions["attempts"]] == [False, True]
    rows = list(csv.DictReader((tmp_path / "history.csv").open()))
    assert rows[1]["kind"] == "accepted_center"
    assert float(rows[1]["spsa_plus_loss_mean"]) == 0.1
    assert float(rows[1]["spsa_plus_search_loss_mean"]) == pytest.approx(0.5)
    assert float(rows[1]["best_feasible_loss"]) == 0.4


def test_rejected_updates_retain_parameters_and_adam_state(tmp_path, monkeypatch):
    import json

    # Both probes and all four trial sizes fail in each of two updates.
    bad_update = [(0.1, 0.95), (0.2, 0.96)] + [(0.1, 0.98)] * 4
    config, tnmr, scope = scripted_acceptance_run(
        monkeypatch, [(0.5, 0.995)] + bad_update * 2 + [(0.5, 0.995)] * 3, updates=2
    )
    result = workflow.run_composite(config, tmp_path, tnmr=tnmr, scope=scope)
    assert result["best_epoch"] == 0
    assert tnmr.shots == 16
    with (
        np.load(tmp_path / "epoch_0001_spsa.npz") as first,
        np.load(tmp_path / "epoch_0002_spsa.npz") as second,
    ):
        assert second["center_parameters"] == pytest.approx(first["center_parameters"])
    with np.load(tmp_path / "checkpoint_latest.npz") as checkpoint:
        assert checkpoint["adam_step"] == 0
        assert not checkpoint["update_accepted"]
        assert np.all(checkpoint["amplitude_logit_first"] == 0)
        assert np.all(checkpoint["phase_second"] == 0)
        assert checkpoint["center_fidelity"] == 0.995
    decisions = json.loads((tmp_path / "epoch_0002_acceptance.json").read_text())
    assert not decisions["accepted"]
    assert [a["step_scale"] for a in decisions["attempts"]] == [1, 0.5, 0.25, 0.125]
    assert parse_numeric_table(tnmr.tables["amp"]) == pytest.approx(
        tnmr.shot_tables[0]["amp"]
    )


@pytest.mark.parametrize("failure", ["seed", "verification"])
def test_fidelity_failure_cannot_export_success(tmp_path, monkeypatch, failure):
    values = (
        [(0.1, 0.98)]
        if failure == "seed"
        else [(0.5, 0.995), (0.5, 0.995), (0.1, 0.98)]
    )
    config, tnmr, scope = scripted_acceptance_run(monkeypatch, values, updates=0)
    original = dict(tnmr.tables)
    with pytest.raises(ValueError, match="fidelity floor"):
        workflow.run_composite(config, tmp_path, tnmr=tnmr, scope=scope)
    assert tnmr.tables == original
    assert tnmr.stopped
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "pulse.csv").exists()
    if failure == "verification":
        assert (tmp_path / "verification_failure.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"minimum_center_fidelity": 99},
        {"minimum_center_fidelity": True},
        {"minimum_center_fidelity": 0},
        {"center_penalty_weight": 0},
        {"maximum_backtracks": -1},
        {"maximum_backtracks": 1.5},
        {"verification_captures": 0},
        {"verification_captures": True},
    ],
)
def test_acceptance_options_fail_in_preview(monkeypatch, options):
    monkeypatch.setattr(workflow, "load_sequence_file", lambda _: sequence_fixture())
    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"},
        composite={"minimum_center_fidelity": 0.99, **options},
    )
    with pytest.raises(ValueError, match="composite"):
        workflow.prepare(config)


def test_acceptance_preview_requires_actual_ideal_point(monkeypatch):
    from hardware_capture.api import plan

    monkeypatch.setattr(workflow, "load_sequence_file", lambda _: sequence_fixture())
    config = config_for_test().with_updates(
        tnmr={"path": "not-opened.tnt"}, composite={"minimum_center_fidelity": 0.99}
    )
    preview = plan("optimize-composite", config)["optimizer"]
    assert preview["acceptance"]["minimum_center_fidelity"] == 0.99
    assert preview["candidate_evaluations_min"] == 7
    assert preview["candidate_evaluations_max"] == 10
    with pytest.raises(ValueError, match="exact zero"):
        workflow.prepare(config.with_updates(scan={"amplitude_errors": [-0.1, 0.1]}))


def test_composite_preset_enables_99_percent_floor():
    import json
    from pathlib import Path

    preset = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "projects/optimization/composite/config.json"
        ).read_text()
    )
    assert preset["composite"]["minimum_center_fidelity"] == 0.99
