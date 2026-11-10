"""Per-pair square referencing and coherent reduction of repeated RF pulses.

This module only reads sequence files and processes arrays. It never opens devices.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from hardware_capture.sequences.tps import Sequence, parse_tps_bytes
from hardware_capture.signal.demodulation import DemodulatedWaveform


def _seconds(value: str) -> float:
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*([num]?)s?\s*", str(value)
    )
    if not match:
        raise ValueError(f"Unsupported sequence duration {value!r}")
    result = (
        float(match[1] + (match[2] or ""))
        * {"": 1, "m": 1e-3, "u": 1e-6, "n": 1e-9}[match[3]]
    )
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Sequence event durations must be finite and positive")
    return result


@dataclass(frozen=True)
class CompositeSchedule:
    repetitions: int
    reference_start_s: float
    reference_duration_s: float
    target_start_s: float
    target_duration_s: float
    period_s: float
    repetition_parameter: str
    parameters: dict[str, str]
    reference_phase_tables: tuple[str, ...]
    parameter_source: str

    @property
    def end_s(self) -> float:
        return (
            self.target_start_s
            + self.target_duration_s
            + (self.repetitions - 1) * self.period_s
        )


def load_sequence_file(path: Path) -> Sequence:
    """Read TPS or the supported single-PSEQ/TMG4 TNT variant conservatively."""
    data = Path(path).read_bytes()
    if data.startswith(b"PSEQ1.001.18 BIN"):
        sequence = parse_tps_bytes(data, source_path=str(path))
    else:
        signature = b"PSEQ\x01\x00\x00\x001.18 BIN"
        if data.count(signature) != 1:
            raise ValueError("Expected exactly one supported embedded PSEQ in the TNT")
        start = data.index(signature)
        marker = b"TMG4\x01\x00\x00\x00"
        if data[start:].count(marker) != 1:
            raise ValueError(
                "Cannot unambiguously bound the embedded sequence; supply an exported TPS"
            )
        end = data.index(marker, start)
        sequence = parse_tps_bytes(
            b"PSEQ1.001.18 BIN" + data[start + 16 : end], source_path=str(path)
        )
    if sequence.rows_parse_error or not sequence.rows or not sequence.parameters:
        raise ValueError("Sequence rows and parameters must parse completely")
    return sequence


def schedule_from_sequence(
    sequence: Sequence,
    amplitude_table: str,
    phase_table: str,
    *,
    parameter_source: str = "value",
    parameter_overrides: dict | None = None,
) -> CompositeSchedule:
    """Extract one simple loop with a square followed by a modulated RF gate.

    Event indices and durations describe the saved script, not physical timing.
    Unsupported/nested loop structures and ambiguous table ownership are errors.
    """
    if parameter_source not in {"value", "current_value"}:
        raise ValueError(
            "parameter_source must be value or current_value (literal parser fields)"
        )
    parameters: dict[str, str] = {}
    for parameter in sequence.parameters:
        value = str(
            parameter.get("current_value", parameter["value"])
            if parameter_source == "current_value"
            else parameter["value"]
        )
        for name in (
            parameter["name"],
            parameter.get("current_name", parameter["name"]),
        ):
            if name in parameters and parameters[name] != value:
                raise ValueError(f"Conflicting sequence values for {name}")
            parameters[name] = value
    for name, value in (parameter_overrides or {}).items():
        if name not in parameters:
            raise ValueError(f"Unknown override parameter {name}")
        if name.lower() in {"n_repeat", "n_reps"}:
            raise ValueError("Read repetition count from the script, not an override")
        parameters[name] = str(value)
    used: dict[str, str] = {}

    def resolve(value):
        if value.startswith("="):
            name = value[1:]
            if name not in parameters:
                raise ValueError(f"Unknown sequence parameter {name}")
            used[name] = parameters[name]
            return parameters[name]
        return value

    rows = {row["label"]: row["events"][1:] for row in sequence.rows}
    required = {
        "Delay",
        "Loop",
        "F1_TxGate",
        "F1_Ampl",
        "F1_Ph",
        "F1_PhMod",
        "Scope_Trig",
    }
    if not required <= rows.keys():
        raise ValueError(f"Missing sequence rows: {sorted(required - rows.keys())}")
    count = len(rows["Delay"])
    if any(len(rows[name]) != count for name in required):
        raise ValueError("Sequence row lengths disagree")
    loops = [event["data"] for event in rows["Loop"]]
    starts = [i for i, value in enumerate(loops) if value.startswith("=")]
    stops = [i for i, value in enumerate(loops) if value == "-1"]
    if (
        len(starts) != 1
        or len(stops) != 1
        or stops[0] <= starts[0]
        or any(v not in {"0", "-1"} and not v.startswith("=") for v in loops)
    ):
        raise ValueError("Require one parameter-controlled loop with one end")
    first, last = starts[0], stops[0]
    repetition_parameter = loops[first][1:]
    if repetition_parameter in (parameter_overrides or {}):
        raise ValueError("Read repetition count from the script, not an override")
    repetition_text = resolve(loops[first])
    if not re.fullmatch(r"[1-9]\d*", repetition_text):
        raise ValueError("Sequence repetition count must be a positive integer")
    repetitions = int(repetition_text)
    duration = np.array([_seconds(resolve(event["data"])) for event in rows["Delay"]])
    edges = np.r_[0.0, np.cumsum(duration)]
    trigger = [i for i, event in enumerate(rows["Scope_Trig"]) if event["data"] == "1"]
    if len(trigger) != 1 or trigger[0] >= first:
        raise ValueError("Require one scope trigger preceding the repetition loop")
    gate = np.array([event["data"] == "1" for event in rows["F1_TxGate"]])
    if any(event["data"] not in {"0", "1"} for event in rows["F1_TxGate"]):
        raise ValueError("Unsupported RF gate values")
    transitions = np.diff(np.r_[False, gate, False].astype(int))
    groups = list(
        zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))
    )
    if len(groups) != 2 or any(a < first or b - 1 > last for a, b in groups):
        raise ValueError("Require exactly square then target RF gates within the loop")
    (ra, rb), (ta, tb) = groups
    phase_tables: set[str] = set()
    for index in range(ra, rb):
        amplitude_event = rows["F1_Ampl"][index]
        if (
            any(slot["name"] for slot in amplitude_event["table_slots"])
            or float(resolve(amplitude_event["data"])) != 100
        ):
            raise ValueError("Square reference must be unmodulated at amplitude 100")
        if any(slot["name"] for slot in rows["F1_PhMod"][index]["table_slots"]):
            raise ValueError("Square reference must not have phase modulation")
        event = rows["F1_Ph"][index]
        names = {slot["name"] for slot in event["table_slots"] if slot["name"]}
        if not names and float(resolve(event["data"])) % 360 != 0:
            raise ValueError(
                "Square reference must have an explicit zero phase or phase table"
            )
        phase_tables.update(names)
    if phase_tables & {amplitude_table, phase_table}:
        raise ValueError("Target tables must not also control the reference phase")
    for row_name, table_name in (
        ("F1_Ampl", amplitude_table),
        ("F1_PhMod", phase_table),
    ):
        locations = [
            i
            for i, event in enumerate(rows[row_name])
            if any(slot["name"] == table_name for slot in event["table_slots"])
        ]
        if (
            not locations
            or not any(ta <= i < tb for i in locations)
            or any(i < rb or i >= tb for i in locations)
        ):
            raise ValueError(f"{table_name} must belong only to the target pulse")
    origin = edges[trigger[0]]
    return CompositeSchedule(
        repetitions,
        float(edges[ra] - origin),
        float(edges[rb] - edges[ra]),
        float(edges[ta] - origin),
        float(edges[tb] - edges[ta]),
        float(edges[last + 1] - edges[first]),
        repetition_parameter,
        used,
        tuple(sorted(phase_tables)),
        parameter_source,
    )


def capture_schedule(
    schedule: CompositeSchedule, reference_start_s: float | None = None
) -> CompositeSchedule:
    """Anchor sequence-relative timing to an explicit scope timestamp."""
    if reference_start_s is None:
        return schedule
    if (
        isinstance(reference_start_s, bool)
        or not isinstance(reference_start_s, (float, int))
        or not math.isfinite(reference_start_s)
    ):
        raise ValueError("capture_reference_start_s must be a finite timestamp")
    offset = reference_start_s - schedule.reference_start_s
    return replace(
        schedule,
        reference_start_s=float(reference_start_s),
        target_start_s=schedule.target_start_s + offset,
    )


@dataclass(frozen=True)
class CompositeReduction:
    time_s: np.ndarray
    calibrated_iq_rad_s: np.ndarray
    mean_iq_rad_s: np.ndarray
    std_iq_rad_s: np.ndarray
    phase_offsets_rad: np.ndarray
    omega_rad_s_per_volt: np.ndarray
    timing_offsets_s: np.ndarray
    reference_amplitude_v: np.ndarray
    baseline_v: complex
    schedule_offset_s: float = 0.0


def reduce_composite(
    demodulated: DemodulatedWaveform,
    schedule: CompositeSchedule,
    *,
    samples: int,
    reference_rabi_hz: float = 160000.0,
    alignment_search_s: float = 1e-6,
    middle_fraction: float = 0.6,
    minimum_coherence: float = 0.9,
    maximum_relative_spread: float = 0.2,
    capture_reference_start_s: float | None = None,
) -> CompositeReduction:
    """Calibrate every pair, then average complex targets on a common time grid.

    Timing comes from the square's rising half-height crossing, constrained near
    its scheduled edge in capture coordinates. An explicit first-reference
    timestamp shifts both pulses together without changing sequence intervals.
    timing_offsets_s includes this common offset plus measured per-pair shifts.
    No target phase fit or target area normalization occurs.
    """
    time = np.asarray(demodulated.time, dtype=float)
    voltage = np.asarray(demodulated.z, dtype=complex)
    if (
        time.ndim != 1
        or time.shape != voltage.shape
        or len(time) < 4
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(voltage))
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError(
            "Demodulated time/IQ arrays must be finite, ordered and matching"
        )
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 4:
        raise ValueError("samples must be an integer >= 4")
    if (
        not math.isfinite(reference_rabi_hz)
        or reference_rabi_hz <= 0
        or not math.isfinite(alignment_search_s)
        or alignment_search_s <= 0
    ):
        raise ValueError(
            "Reference Rabi frequency and alignment search must be positive"
        )
    if (
        not 0 < middle_fraction < 1
        or not 0 < minimum_coherence <= 1
        or not 0 < maximum_relative_spread < 1
    ):
        raise ValueError("Invalid reference quality settings")
    if (
        schedule.repetitions < 1
        or schedule.target_start_s
        < schedule.reference_start_s + schedule.reference_duration_s
        or schedule.period_s
        < schedule.target_start_s
        + schedule.target_duration_s
        - schedule.reference_start_s
    ):
        raise ValueError("Invalid composite schedule")
    if alignment_search_s >= min(
        schedule.reference_duration_s * 0.5,
        (
            schedule.target_start_s
            - schedule.reference_start_s
            - schedule.reference_duration_s
        )
        * 0.5,
    ):
        raise ValueError(
            "Alignment search overlaps another pulse or the square interior"
        )
    original_start = schedule.reference_start_s
    schedule = capture_schedule(schedule, capture_reference_start_s)
    schedule_offset = schedule.reference_start_s - original_start
    if (
        time[0] > schedule.reference_start_s - alignment_search_s
        or time[-1] < schedule.end_s + alignment_search_s
    ):
        raise ValueError(
            f"Capture must cover all {schedule.repetitions} complete square/target pairs plus alignment margins"
        )
    baseline_mask = np.ones(len(time), dtype=bool)
    for rep in range(schedule.repetitions):
        for start, duration in (
            (schedule.reference_start_s, schedule.reference_duration_s),
            (schedule.target_start_s, schedule.target_duration_s),
        ):
            start += rep * schedule.period_s
            first = np.searchsorted(time, start - alignment_search_s, side="left")
            last = np.searchsorted(
                time, start + duration + alignment_search_s, side="right"
            )
            baseline_mask[first:last] = False
    if np.count_nonzero(baseline_mask) < 8:
        raise ValueError("Insufficient off-pulse samples for I/Q baseline")
    baseline = complex(
        np.median(voltage.real[baseline_mask]), np.median(voltage.imag[baseline_mask])
    )
    voltage = voltage - baseline
    envelope = np.abs(voltage)
    noise = float(np.median(envelope[baseline_mask]))
    relative_time = (np.arange(samples) + 0.5) * schedule.target_duration_s / samples
    targets, phases, scales, shifts, amplitudes = [], [], [], [], []
    trim = 0.5 * (1 - middle_fraction) * schedule.reference_duration_s
    for rep in range(schedule.repetitions):
        start = schedule.reference_start_s + rep * schedule.period_s
        central = slice(
            np.searchsorted(time, start + trim, side="left"),
            np.searchsorted(
                time, start + schedule.reference_duration_s - trim, side="right"
            ),
        )
        if central.stop - central.start < 8:
            raise ValueError(f"Pair {rep}: insufficient square plateau samples")
        level = float(np.median(envelope[central]))
        if level <= max(10 * noise, np.finfo(float).tiny):
            raise ValueError(f"Pair {rep}: missing or low-SNR square reference")
        first = np.searchsorted(time, start - alignment_search_s, side="left")
        last = np.searchsorted(time, start + alignment_search_s, side="right")
        local_envelope = envelope[first:last]
        crossings = first + np.flatnonzero(
            (local_envelope[:-1] < 0.5 * level) & (local_envelope[1:] >= 0.5 * level)
        )
        if len(crossings) != 1:
            raise ValueError(
                f"Pair {rep}: ambiguous or missing square rising edge: "
                f"{len(crossings)} crossings in "
                f"[{(start - alignment_search_s) * 1e6:.6g}, "
                f"{(start + alignment_search_s) * 1e6:.6g}] us of capture time"
            )
        index = crossings[0]
        crossing = time[index] + (0.5 * level - envelope[index]) / (
            envelope[index + 1] - envelope[index]
        ) * (time[index + 1] - time[index])
        shift = float(crossing - start)
        central = slice(
            np.searchsorted(time, start + shift + trim, side="left"),
            np.searchsorted(
                time, start + shift + schedule.reference_duration_s - trim, side="right"
            ),
        )
        plateau = voltage[central]
        if len(plateau) < 8:
            raise ValueError(f"Pair {rep}: insufficient aligned reference samples")
        amplitude = float(np.median(np.abs(plateau)))
        coherence = abs(
            np.mean(plateau / np.maximum(np.abs(plateau), np.finfo(float).tiny))
        )
        spread = (
            float(
                np.percentile(np.abs(plateau), 90) - np.percentile(np.abs(plateau), 10)
            )
            / amplitude
        )
        if coherence < minimum_coherence or spread > maximum_relative_spread:
            raise ValueError(f"Pair {rep}: unstable square phase/amplitude")
        phase = float(np.angle(np.sum(plateau)))
        scale = 2 * math.pi * reference_rabi_hz / amplitude
        target_times = (
            schedule.target_start_s + rep * schedule.period_s + shift + relative_time
        )
        # Retain one bracketing sample on each side, without copying the full
        # multi-million-sample record for each real/imaginary interpolation.
        first = max(0, np.searchsorted(time, target_times[0], side="left") - 1)
        last = min(len(time), np.searchsorted(time, target_times[-1], side="right") + 1)
        target = np.interp(
            target_times, time[first:last], voltage.real[first:last]
        ) + 1j * np.interp(target_times, time[first:last], voltage.imag[first:last])
        targets.append(target * np.exp(-1j * phase) * scale)
        phases.append(phase)
        scales.append(scale)
        shifts.append(schedule_offset + shift)
        amplitudes.append(amplitude)
    values = np.asarray(targets)
    return CompositeReduction(
        relative_time,
        values,
        np.mean(values, axis=0),
        np.std(values, axis=0),
        np.asarray(phases),
        np.asarray(scales),
        np.asarray(shifts),
        np.asarray(amplitudes),
        baseline,
        schedule_offset,
    )
