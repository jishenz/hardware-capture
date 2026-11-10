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
):
    """Record stage transitions and elapsed time without a background reporter."""
    path = directory / f"{label}_timing.json"
    display = description or label
    started = perf_counter()

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

    persist("running")
    print(f"{display} | {name}: starting", flush=True)
    status = "failed"
    try:
        yield
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        elapsed = persist(status)
        print(f"{display} | {name}: {status} ({elapsed:.2f} s)", flush=True)
