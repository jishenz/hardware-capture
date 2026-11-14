"""File-only, bounded evidence retention for the composite optimizer."""

import json
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .recording import measure_stage, save_raw_capture


def capture_retention(options):
    mode = options.get("capture_retention", "selected")
    if mode not in ("selected", "all"):
        raise ValueError("composite.capture_retention must be selected or all")
    return mode


def atomic_npz(path, **arrays):
    """Replace an archive only after its replacement has been fully written."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def release_waveforms(candidate):
    """Drop bulky arrays once selection and any required persistence finish."""
    candidate.pop("raw", None)
    candidate.pop("archive", None)


class CaptureRecorder:
    """Never reload measurements; persist only the selected evidence slots."""

    def __init__(self, directory, options):
        self.directory = Path(directory)
        self.mode = capture_retention(options)
        self.best_id = self.center_id = None
        self.best_path = self.center_path = None

    @property
    def selected(self):
        return self.mode == "selected"

    def append_event(self, event, filename="captures.jsonl"):
        with (self.directory / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def save_raw(self, label, raw):
        path = self.directory / f"{label}_raw.npz"
        temporary = path.with_name(path.name + ".tmp")
        try:
            save_raw_capture(
                temporary, time_s=raw.time, voltage_v=raw.voltage,
                sample_rate_hz=raw.sample_rate,
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, candidate, label, *, raw=True):
        if raw:
            self.save_raw(label, candidate["raw"])
        path = self.directory / f"{label}.npz"
        atomic_npz(path, **candidate["archive"])
        return path

    def keep_best(self, candidate):
        if not self.selected:
            return candidate["path"]
        if self.best_id != candidate["measurement_id"]:
            # Leave the slot referenced by the previous checkpoint untouched.
            slot = 1 if self.best_path == self.directory / "best_measured_0.npz" else 0
            self.best_path = (
                candidate["path"] if candidate["measurement_id"] == "epoch_0000"
                else self.save(candidate, f"best_measured_{slot}")
            )
            self.best_id = candidate["measurement_id"]
        return self.best_path

    def keep_center(self, candidate):
        if not self.selected:
            return candidate["path"]
        if self.center_id != candidate["measurement_id"]:
            # Commit the checkpoint only after writing the alternate evidence slot.
            slot = 1 if self.center_path == self.directory / "latest_center_0.npz" else 0
            self.center_path = (
                candidate["path"] if candidate["measurement_id"] == "epoch_0000"
                else self.save(candidate, f"latest_center_{slot}", raw=False)
            )
            self.center_id = candidate["measurement_id"]
        return self.center_path

    @contextmanager
    def capture(self, label, timings):
        candidate = {"measurement_id": label, "path": None}
        try:
            yield candidate
            if self.selected and (
                label == "epoch_0000" or label.startswith("final_verification")
            ):
                with measure_stage(
                    self.directory, label, timings, "save_selected",
                    candidate_timing=False,
                ):
                    candidate["path"] = self.save(candidate, label)
            archive = candidate["archive"]
            calibration = {
                name: np.asarray(archive[name]).tolist()
                for name in (
                    "reference_amplitude_v", "phase_offsets_rad",
                    "omega_rad_s_per_volt", "timing_offsets_s", "schedule_offset_s",
                )
            }
            self.append_event({
                "measurement_id": label, "status": "complete",
                "updated_at": datetime.now(UTC).isoformat(),
                "measurement_path": str(candidate["path"] or ""),
                "loss": float(archive["loss"]),
                "search_loss": float(archive["search_loss"]),
                "center_fidelity": float(archive["center_fidelity"]),
                "calibration": calibration, "timings": timings,
            })
            if not self.selected or label.startswith("final_verification"):
                release_waveforms(candidate)
        except BaseException as exc:
            # A recording failure must not replace the original processing error.
            event = {
                "measurement_id": label,
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "updated_at": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "timings": timings,
            }
            if self.selected and "raw" in candidate:
                try:
                    self.save_raw("failed_capture", candidate["raw"])
                    if "archive" in candidate:
                        self.save(candidate, "failed_capture", raw=False)
                except Exception as recording_error:  # noqa: BLE001 - preserve the primary failure
                    event["evidence_error"] = str(recording_error)
            try:
                self.append_event(event)
            except Exception as recording_error:  # noqa: BLE001 - preserve the primary failure
                exc.add_note(f"Could not persist failure log: {recording_error}")
            raise
