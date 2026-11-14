"""File-only recording and timing helpers for candidate measurements."""

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def save_raw_capture(path: Path, **arrays) -> None:
    """Write a lossless, standard NPZ using fast level-1 compression."""
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as entry:
                np.save(entry, np.asarray(array), allow_pickle=False)


@contextmanager
def measure_stage(
    directory: Path,
    label: str,
    timings: dict,
    name: str,
    *,
    description: str | None = None,
    candidate_timing: bool = True,
):
    """Record stage transitions and elapsed time without a background reporter."""
    path = directory / f"{label}_timing.json"
    display = description or label
    started = perf_counter()
    error = {}

    def persist(status):
        elapsed = perf_counter() - started
        updated = datetime.now(UTC).isoformat()
        timings[name] = {
            "status": status, "elapsed_s": elapsed, "updated_at": updated, **error,
        }
        payloads = (
            (
                directory / "progress.json",
                {
                    "pid": os.getpid(),
                    "candidate": label,
                    "description": display,
                    "stage": name,
                    **timings[name],
                },
            ),
        )
        if candidate_timing:
            payloads = ((path, timings), *payloads)
        for destination, payload in payloads:
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(destination)
        return elapsed

    persist("running")
    print(f"{display} | {name}: starting", flush=True)
    status = "failed"
    failure = None
    try:
        yield
        status = "complete"
    except BaseException as exc:
        failure = exc
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        error.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            elapsed = persist(status)
        except Exception as recording_error:
            if failure is None:
                raise
            failure.add_note(f"Could not persist stage failure: {recording_error}")
            elapsed = perf_counter() - started
        print(f"{display} | {name}: {status} ({elapsed:.2f} s)", flush=True)
