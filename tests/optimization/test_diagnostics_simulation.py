"""Numerical/file-only tests; no device construction or workflow entrypoints."""

import csv
import json
import math

import numpy as np
import pytest

from hardware_capture.optimization.core import RobustnessConfig, robustness_objective
from hardware_capture.optimization.diagnostics import saved_diagnostics, step_metrics
from hardware_capture.optimization.simulation import compare_saved_run, simulated_objective

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", [1, 7, 32, 65])
def test_quaternion_propagation_matches_reference_including_time_order(size):
    rng = np.random.default_rng(123)
    i, q = rng.normal(size=(2, size))
    durations = rng.uniform(0.01, 0.1, size)
    config = RobustnessConfig(2*math.pi, np.array([-0.5, 0, 0.2]), np.array([-0.3, 0, 0.4]))
    actual = simulated_objective(i, q, durations, config)
    reference = robustness_objective(i, q, durations, config).objective
    np.testing.assert_allclose(actual.fidelity, reference.fidelity, atol=1e-13, rtol=1e-12)
    assert actual.loss == pytest.approx(reference.loss, abs=1e-13)


def test_zero_drive_is_finite():
    config = RobustnessConfig(2*math.pi, np.array([0.0]), np.array([0.0]))
    assert simulated_objective(np.zeros(3), np.zeros(3), np.ones(3), config).loss > 0.9


def test_step_metrics_distinguishes_backtracking_and_rejection():
    before = np.zeros(4)
    proposed = np.array([1, -1, 2, -2])
    half = step_metrics(before, proposed, proposed/2)
    assert half["amplitude_logit_step_rms"] == 0.5
    assert half["phase_step_rms_rad"] == 1
    rejected = step_metrics(before, proposed, before)
    assert rejected["amplitude_logit_step_rms"] == rejected["phase_step_rms_rad"] == 0
    assert rejected["proposed_phase_step_rms_rad"] == 2


def test_saved_diagnostics_preserve_source_and_reconstruct_steps(tmp_path):
    source, output = tmp_path / "source", tmp_path / "diagnostics"
    source.mkdir()
    rows = [dict(epoch=0, kind="initial", loss=.5, center_fidelity=.995,
                 minimum_center_fidelity=.99, measurement_path="epoch_0000.npz"),
            dict(epoch=1, kind="accepted_center", loss=.6, center_fidelity=.98,
                 minimum_center_fidelity=.99, measurement_path=r"Z:\old\epoch_0001_trial_001.npz"),
            dict(epoch=2, kind="retained_center", loss=.6, center_fidelity=.98,
                 minimum_center_fidelity=.99, measurement_path="epoch_0001_trial_001.npz")]
    with (source / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (source / "effective_config.json").write_text(json.dumps({"composite": {
        "minimum_center_fidelity": .99, "center_penalty_weight": 10}}))
    np.savez(source / "epoch_0001_spsa.npz", center_parameters=np.zeros(4))
    np.savez(source / "epoch_0001_trial_000.npz", amplitude_logits=[1, 1], phase_coefficients_rad=[2, 2])
    np.savez(source / "epoch_0001_trial_001.npz", amplitude_logits=[.5, .5], phase_coefficients_rad=[1, 1])
    np.savez(source / "epoch_0002_spsa.npz", center_parameters=[.5, .5, 1, 1])
    np.savez(source / "epoch_0002_trial_000.npz", amplitude_logits=[1, 1], phase_coefficients_rad=[2, 2])
    original = (source / "history.csv").read_bytes()
    report = saved_diagnostics(source, output)
    assert report["missing_step_epochs"] == []
    assert (source / "history.csv").read_bytes() == original
    with (output / "diagnostics.csv").open(newline="") as f:
        actual = list(csv.DictReader(f))
    assert float(actual[1]["search_loss"]) == pytest.approx(.7)
    assert float(actual[1]["amplitude_logit_step_rms"]) == .5
    assert float(actual[2]["amplitude_logit_step_rms"]) == 0
    assert float(actual[2]["proposed_amplitude_logit_step_rms"]) == .5
    assert (output / "optimizer_diagnostics.png").read_bytes().startswith(b"\x89PNG")


def test_simulation_is_reproducible_and_only_changes_learning_rates(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"optimization": {}, "composite": {"minimum_center_fidelity": .99},
              "pulse": {"duration_s": 1}, "calibration": {"rabi_at_100_hz": 1},
              "spsa": {"directions": 2}}
    (source / "effective_config.json").write_text(json.dumps(config))
    np.savez(source / "epoch_0000.npz", amplitude_logits=np.zeros(4),
             phase_coefficients_rad=np.zeros(4), phase_baseline_rad=np.zeros(20),
             amplitude_maximum=1.0, amplitude_errors=[-.1, 0, .1], detuning_hz=[-.1, 0, .1])
    first = compare_saved_run(source, tmp_path / "first", updates=3, seeds=(11, 12), samples=20)
    second = compare_saved_run(source, tmp_path / "second", updates=3, seeds=(11, 12), samples=20)
    assert len(first) == 4
    for a, b in zip(first, second):
        assert {k: v for k, v in a.items() if k != "elapsed_s"} == {
            k: v for k, v in b.items() if k != "elapsed_s"}
        assert a["full_resolution_best_center_fidelity"] >= .99
    assert [row["learning_rate_factor"] for row in first] == [1, .5, 1, .5]
    manifest = json.loads((tmp_path / "first/manifest.json").read_text())
    assert manifest["step_caps"] == "Unchanged in both arms"
    assert first[0]["initial_loss"] == first[1]["initial_loss"]


def test_cap_comparison_removes_both_caps_and_preserves_rates_and_fidelity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "optimization": {"phase_learning_rate": .02, "phase_step_rad": .0001},
        "composite": {"minimum_center_fidelity": .99,
                      "amplitude_logit_learning_rate": .02, "amplitude_logit_step": .0001},
        "pulse": {"duration_s": 1}, "calibration": {"rabi_at_100_hz": 1},
        "spsa": {"directions": 2},
    }
    (source / "effective_config.json").write_text(json.dumps(config))
    np.savez(source / "epoch_0000.npz", amplitude_logits=np.zeros(4),
             phase_coefficients_rad=np.zeros(4), phase_baseline_rad=np.zeros(20),
             amplitude_maximum=1.0, amplitude_errors=[-.1, 0, .1], detuning_hz=[-.1, 0, .1])
    output = tmp_path / "comparison"
    summaries = compare_saved_run(source, output, updates=4, seeds=(11,), samples=20,
                                  comparison="step-caps")
    capped, uncapped = summaries
    assert capped["learning_rate_factor"] == uncapped["learning_rate_factor"] == 1
    assert capped["initial_loss"] == uncapped["initial_loss"]
    assert capped["maximum_proposed_amplitude_logit_step"] <= .0001+1e-12
    assert capped["maximum_proposed_phase_step_rad"] <= .0001+1e-12
    assert uncapped["maximum_proposed_amplitude_logit_step"] > .0001
    assert uncapped["maximum_proposed_phase_step_rad"] > .0001
    assert capped["epochs_with_amplitude_clipping"] > 0
    assert capped["epochs_with_phase_clipping"] > 0
    assert uncapped["epochs_with_amplitude_clipping"] == uncapped["epochs_with_phase_clipping"] == 0
    for summary in summaries:
        with (output / f"seed_11_{summary['comparison_arm']}" / "history.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert all(float(row["center_fidelity"]) >= .99 for row in rows)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["arms"][1]["amplitude_logit_step_cap"] is None
    assert manifest["arms"][1]["phase_step_cap_rad"] is None
    assert manifest["base_amplitude_logit_learning_rate"] == .02
    assert manifest["base_phase_learning_rate"] == .02


def test_larger_learning_rates_scale_proposals_and_keep_caps(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"optimization": {"phase_step_rad": .1},
              "composite": {"minimum_center_fidelity": .99, "amplitude_logit_step": .02},
              "pulse": {"duration_s": 1}, "calibration": {"rabi_at_100_hz": 1},
              "spsa": {"directions": 2}}
    (source / "effective_config.json").write_text(json.dumps(config))
    np.savez(source / "epoch_0000.npz", amplitude_logits=np.zeros(4),
             phase_coefficients_rad=np.zeros(4), phase_baseline_rad=np.zeros(20),
             amplitude_maximum=1.0, amplitude_errors=[-.1, 0, .1], detuning_hz=[-.1, 0, .1])
    output = tmp_path / "comparison"
    summaries = compare_saved_run(source, output, updates=1, seeds=(11,), samples=20,
                                  learning_rate_factors=(1, 1.5, 2))
    assert [row["learning_rate_factor"] for row in summaries] == [1, 1.5, 2]
    first = None
    for summary in summaries:
        folder = output / f"seed_11_{summary['comparison_arm']}"
        with (folder / "history.csv").open(newline="") as f:
            row = list(csv.DictReader(f))[1]
        if first is None:
            first = row
        factor = summary["learning_rate_factor"]
        for key in ("max_uncapped_amplitude_logit_step", "max_uncapped_phase_step_rad"):
            assert float(row[key]) == pytest.approx(factor * float(first[key]))
        assert float(row["max_proposed_amplitude_logit_step"]) <= .02 + 1e-12
        assert float(row["max_proposed_phase_step_rad"]) <= .1 + 1e-12
        with (folder / "checkpoints.csv").open(newline="") as f:
            checkpoint = list(csv.DictReader(f))[0]
        assert float(checkpoint["full_resolution_best_loss"]) == summary["full_resolution_best_loss"]
        assert float(checkpoint["full_resolution_best_center_fidelity"]) >= .99
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["learning_rate_factors"] == [1, 1.5, 2]
    assert all(arm["amplitude_logit_step_cap"] == .02 for arm in manifest["arms"])


@pytest.mark.parametrize("factors", [(), (0,), (-1,), (float("nan"),), (float("inf"),), (1, 1)])
def test_invalid_learning_rate_factors_fail_before_reading_source(tmp_path, factors):
    with pytest.raises(ValueError, match="Learning-rate factors"):
        compare_saved_run(tmp_path / "missing", tmp_path / "out", learning_rate_factors=factors)


def test_amplitude_cap_sweep_preserves_rates_phase_caps_and_raw_first_proposal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"optimization": {"phase_step_rad": .1},
              "composite": {"minimum_center_fidelity": .99, "amplitude_logit_step": .02},
              "pulse": {"duration_s": 1}, "calibration": {"rabi_at_100_hz": 1},
              "spsa": {"directions": 2}}
    (source / "effective_config.json").write_text(json.dumps(config))
    np.savez(source / "epoch_0000.npz", amplitude_logits=np.zeros(4),
             phase_coefficients_rad=np.zeros(4), phase_baseline_rad=np.zeros(20),
             amplitude_maximum=1.0, amplitude_errors=[-.1, 0, .1], detuning_hz=[-.1, 0, .1])
    output = tmp_path / "comparison"
    summaries = compare_saved_run(source, output, updates=2, seeds=(11,), samples=20,
                                  comparison="amplitude-caps", learning_rate_factors=(2,),
                                  amplitude_step_caps=(.02, .03, .04))
    proposals = []
    for summary, cap in zip(summaries, (.02, .03, .04)):
        assert summary["amplitude_logit_learning_rate"] == summary["phase_learning_rate"] == .04
        assert summary["amplitude_logit_step_cap"] == cap
        assert summary["phase_step_cap_rad"] == .1
        assert summary["maximum_proposed_amplitude_logit_step"] <= cap + 1e-12
        assert summary["maximum_proposed_phase_step_rad"] <= .1 + 1e-12
        with (output / f"seed_11_cap_{cap:g}" / "history.csv").open(newline="") as stream:
            first = list(csv.DictReader(stream))[1]
        proposals.append([float(first[key]) for key in (
            "max_uncapped_amplitude_logit_step", "max_uncapped_phase_step_rad",
            "spsa_plus_search_loss_mean", "spsa_minus_search_loss_mean")])
    np.testing.assert_array_equal(proposals, [proposals[0]] * 3)
    assert summaries[0]["epochs_with_amplitude_clipping"] > 0
    assert summaries[1]["maximum_proposed_amplitude_logit_step"] > .02
    assert summaries[2]["maximum_proposed_amplitude_logit_step"] > .03
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["learning_rate_factors"] == [2]
    assert [arm["amplitude_logit_step_cap"] for arm in manifest["arms"]] == [.02, .03, .04]
    assert all(arm["phase_step_cap_rad"] == .1 for arm in manifest["arms"])


@pytest.mark.parametrize("caps", [(), (0,), (-1,), (float("nan"),), (float("inf"),), (.02, .02)])
def test_invalid_amplitude_caps_fail_before_reading_source(tmp_path, caps):
    with pytest.raises(ValueError, match="Amplitude step caps"):
        compare_saved_run(tmp_path / "missing", tmp_path / "out", comparison="amplitude-caps",
                          amplitude_step_caps=caps)
