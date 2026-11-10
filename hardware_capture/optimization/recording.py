"""File-only recording and timing helpers for candidate measurements."""

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
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
    heartbeat_s: float = 5.0,
):
    """Report a live software heartbeat; never poll or command an instrument.

    A heartbeat means this Python process is responsive, not that a blocking
    device call is making progress. A stale file cannot establish device state.
    """
    if not np.isfinite(heartbeat_s) or heartbeat_s <= 0:
        raise ValueError("heartbeat_s must be finite and positive")
    path = directory / f"{label}_timing.json"
    display = description or label
    started = perf_counter()
    stopped = Event()

    def persist(status):
        elapsed = perf_counter() - started
        updated = datetime.now(UTC).isoformat()
        timings[name] = {"status": status, "elapsed_s": elapsed, "updated_at": updated}
        payloads = (
            (path, timings),
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
        for destination, payload in payloads:
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(destination)
        return elapsed

    def heartbeat():
        while not stopped.wait(heartbeat_s):
            elapsed = perf_counter() - started
            print(
                f"{display} | {name}: waiting/working ({elapsed:.0f} s elapsed; "
                "Python heartbeat, stage has not returned)",
                flush=True,
            )
            try:
                persist("running")
            except OSError as exc:
                print(f"Progress file update failed: {exc}", flush=True)

    persist("running")
    print(f"{display} | {name}: starting", flush=True)
    reporter = Thread(target=heartbeat, name="optimizer-progress", daemon=True)
    reporter.start()
    status = "failed"
    try:
        yield
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        stopped.set()
        reporter.join()
        elapsed = persist(status)
        print(f"{display} | {name}: {status} ({elapsed:.2f} s)", flush=True)
