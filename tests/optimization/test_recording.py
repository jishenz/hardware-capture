"""File-only recording checks; never access instruments."""

import json
import time

import numpy as np
import pytest

from hardware_capture.optimization.recording import measure_stage, save_raw_capture

pytestmark = pytest.mark.unit


def test_fast_raw_archive_is_lossless_and_numpy_readable(tmp_path):
    path = tmp_path / "raw.npz"
    payload = {
        "time_s": np.linspace(-1e-5, 0.004, 10000),
        "voltage_v": np.random.default_rng(4).normal(size=10000),
        "sample_rate_hz": 4e9,
    }
    save_raw_capture(path, **payload)
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        for name, expected in payload.items():
            np.testing.assert_array_equal(saved[name], expected)


def test_stage_record_survives_failure_and_exposes_in_progress_work(tmp_path):
    timings = {}
    path = tmp_path / "candidate_timing.json"
    with measure_stage(tmp_path, "candidate", timings, "first"):
        assert json.loads(path.read_text())["first"]["status"] == "running"
    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        measure_stage(tmp_path, "candidate", timings, "second"),
    ):
        raise RuntimeError("synthetic failure")
    saved = json.loads(path.read_text())
    assert saved["first"]["status"] == "complete"
    assert saved["second"]["status"] == "failed"
    assert all(stage["elapsed_s"] >= 0 for stage in saved.values())


def test_heartbeat_updates_during_blocked_stage_and_stops_after_exit(tmp_path, capsys):
    path = tmp_path / "progress.json"
    with measure_stage(
        tmp_path,
        "probe",
        {},
        "acquire_and_transfer",
        description="Epoch 03/15, plus probe | capture 6/32",
        heartbeat_s=0.01,
    ):
        initial = json.loads(path.read_text())
        deadline = time.monotonic() + 2
        while True:
            progress = json.loads(path.read_text())
            if progress["updated_at"] != initial["updated_at"]:
                break
            assert time.monotonic() < deadline, "Heartbeat did not update"
            time.sleep(0.005)
        assert progress["status"] == "running"
        assert progress["elapsed_s"] > initial["elapsed_s"]
    completed = path.read_text()
    time.sleep(0.03)
    assert path.read_text() == completed
    assert json.loads(completed)["status"] == "complete"
    output = capsys.readouterr().out
    assert "Epoch 03/15, plus probe | capture 6/32" in output
    assert "Python heartbeat, stage has not returned" in output


def test_keyboard_interrupt_records_interrupted_stage(tmp_path):
    with pytest.raises(KeyboardInterrupt), measure_stage(tmp_path, "probe", {}, "wait"):
        raise KeyboardInterrupt
    assert (
        json.loads((tmp_path / "progress.json").read_text())["status"] == "interrupted"
    )
