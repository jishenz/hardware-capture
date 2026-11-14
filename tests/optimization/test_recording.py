"""File-only recording checks; never access instruments."""

import json

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


def test_progress_updates_only_at_stage_boundaries(tmp_path, capsys, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "hardware_capture.optimization.recording.perf_counter", lambda: clock[0]
    )
    path = tmp_path / "progress.json"
    with measure_stage(
        tmp_path,
        "probe",
        {},
        "acquire_and_transfer",
        description="Epoch 03/15, plus probe | capture 6/32",
    ):
        initial = path.read_text()
        progress = json.loads(initial)
        assert progress["status"] == "running"
        assert progress["elapsed_s"] == 0.0
        clock[0] = 30.0
        assert path.read_text() == initial
        assert "waiting/working" not in capsys.readouterr().out
    completed = path.read_text()
    assert json.loads(completed)["status"] == "complete"
    assert json.loads(completed)["elapsed_s"] == 30.0
    timing = json.loads((tmp_path / "probe_timing.json").read_text())[
        "acquire_and_transfer"
    ]
    progress = json.loads(completed)
    for field in ("status", "elapsed_s", "updated_at"):
        assert timing[field] == progress[field]
    output = capsys.readouterr().out
    assert "Epoch 03/15, plus probe | capture 6/32" in output
    assert "complete (30.00 s)" in output
    assert "heartbeat" not in output


def test_keyboard_interrupt_records_interrupted_stage(tmp_path):
    with pytest.raises(KeyboardInterrupt), measure_stage(tmp_path, "probe", {}, "wait"):
        raise KeyboardInterrupt
    assert (
        json.loads((tmp_path / "progress.json").read_text())["status"] == "interrupted"
    )


def test_compact_stage_keeps_progress_and_exception_without_candidate_file(tmp_path):
    with pytest.raises(OSError), measure_stage(
        tmp_path, "probe", {}, "objective", candidate_timing=False
    ):
        raise OSError(28, "No space left on device")
    assert not (tmp_path / "probe_timing.json").exists()
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["error_type"] == "OSError"
    assert "No space left" in progress["error"]


def test_checkpoint_write_failure_preserves_previous_archive(tmp_path, monkeypatch):
    from hardware_capture.optimization.retention import atomic_npz

    path = tmp_path / "checkpoint_latest.npz"
    atomic_npz(path, next_epoch=3)
    previous = path.read_bytes()

    def fail(stream, **arrays):
        stream.write(b"incomplete archive")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="No space left"):
        atomic_npz(path, next_epoch=4)
    assert path.read_bytes() == previous
    assert not (tmp_path / "checkpoint_latest.npz.tmp").exists()
    with np.load(path, allow_pickle=False) as saved:
        assert saved["next_epoch"] == 3


def test_evidence_write_failure_preserves_original_processing_error(tmp_path, monkeypatch):
    from hardware_capture.optimization.retention import CaptureRecorder

    recorder = CaptureRecorder(tmp_path, {})

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(recorder, "save_raw", full_disk)
    with pytest.raises(ValueError, match="bad calibration"), recorder.capture("probe", {}) as candidate:
        candidate["raw"] = object()
        raise ValueError("bad calibration")
    event = json.loads((tmp_path / "captures.jsonl").read_text())
    assert event["error_type"] == "ValueError"
    assert event["error"] == "bad calibration"
    assert "No space left" in event["evidence_error"]
    assert "ValueError: bad calibration" in event["traceback"]


def test_progress_write_failure_does_not_mask_processing_error(tmp_path, monkeypatch):
    from pathlib import Path

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    with (
        pytest.raises(ValueError, match="bad calibration") as failure,
        measure_stage(tmp_path, "probe", {}, "calibration", candidate_timing=False),
    ):
        monkeypatch.setattr(Path, "write_text", full_disk)
        raise ValueError("bad calibration")
    assert any("Could not persist stage failure" in note for note in failure.value.__notes__)
