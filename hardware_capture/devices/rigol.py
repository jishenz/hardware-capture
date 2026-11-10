"""RIGOL DHO oscilloscope capture helpers."""

from __future__ import annotations

import csv
import re
import time as time_module
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

DEFAULT_IP = "192.168.1.152"
DEFAULT_TIMEOUT_MS = 20_000
DEFAULT_CHUNK_POINTS = 100_000
DEFAULT_WAVEFORM_MODE = "RAW"
DEFAULT_WAVEFORM_FORMAT = "WORD"
DEFAULT_TRIGGER_TIMEOUT_S = 30.0
DEFAULT_ARMED_TIMEOUT_S = 2.0
DEFAULT_TRIGGER_POLL_S = 0.05

ARMED_TRIGGER_STATUSES = {"WAIT", "RUN", "TD", "AUTO"}
STOPPED_TRIGGER_STATUSES = {"STOP", "STOPPED"}

TIME_UNITS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ns": 1e-9,
    "nanosecond": 1e-9,
    "nanoseconds": 1e-9,
}

_TIME_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)?"
)


@dataclass(frozen=True)
class ScopePreamble:
    """Waveform preamble fields used for axis conversion."""

    data_format: str
    data_type: str
    points: int
    count: int
    x_increment: float
    x_origin: float
    x_reference: float
    y_increment: float
    y_origin: float
    y_reference: float


@dataclass(frozen=True)
class ScopeWaveform:
    """Waveform captured from the scope."""

    time: np.ndarray
    voltage: np.ndarray
    sample_rate: float
    source: str
    channel: str
    preamble: ScopePreamble
    start_point: int
    stop_point: int


@dataclass(frozen=True)
class WaveformCaptureConfig:
    """Configuration for one waveform read."""

    channel: str = "CHAN1"
    stop_before_read: bool = True
    run_after_read: bool = False
    waveform_mode: str = DEFAULT_WAVEFORM_MODE
    waveform_format: str = DEFAULT_WAVEFORM_FORMAT
    chunk_points: int = DEFAULT_CHUNK_POINTS
    window_start_s: float | None = None
    window_end_s: float | None = None
    start_point: int | None = None
    stop_point: int | None = None
    allow_clamp: bool = True


class ScopeTimeoutError(TimeoutError):
    """Raised when the scope does not reach the requested state in time."""


def parse_time_seconds(value: str | float | int | None) -> float | None:
    """Parse values like ``-1us``, ``3e-6``, ``2 ms``, or ``0.0001s``."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace("\u00b5", "u").replace("\u03bc", "u")
    match = _TIME_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"Could not parse time {value!r}. Try -1us or 2ms.")

    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if not unit:
        return number
    if unit not in TIME_UNITS:
        raise ValueError(f"Unknown time unit {unit!r}. Supported units: s, ms, us, ns.")
    return number * TIME_UNITS[unit]


def normalize_channel(channel: str | int) -> str:
    """Return a RIGOL channel token such as ``CHAN1``."""

    if isinstance(channel, int):
        if channel < 1:
            raise ValueError("channel must be 1 or greater.")
        return f"CHAN{channel}"

    text = str(channel).strip().upper()
    if text.isdigit():
        return f"CHAN{int(text)}"
    if text.startswith("CH") and text[2:].isdigit():
        return f"CHAN{text[2:]}"
    if text.startswith("CHAN") and text[4:].isdigit():
        return text
    raise ValueError(f"Unsupported channel {channel!r}. Use values like 1 or CHAN1.")


def normalize_waveform_format(value: str) -> str:
    """Return a supported transfer token (WORD is the 16-bit DHO default)."""

    if isinstance(value, str):
        token = value.strip().upper()
        if token in {"ASC", "ASCII"}:
            return "ASC"
        if token in {"WORD", "BYTE"}:
            return token
    raise ValueError("scope.waveform_format must be WORD, BYTE, or ASC")


def parse_preamble(raw: str) -> ScopePreamble:
    """Parse the comma-separated ``:WAV:PRE?`` response."""

    values = [item.strip() for item in raw.strip().split(",")]
    if len(values) < 10:
        raise ValueError(f"Expected at least 10 preamble values, got {len(values)}.")
    return ScopePreamble(
        data_format=values[0],
        data_type=values[1],
        points=int(float(values[2])),
        count=int(float(values[3])),
        x_increment=float(values[4]),
        x_origin=float(values[5]),
        x_reference=float(values[6]),
        y_increment=float(values[7]),
        y_origin=float(values[8]),
        y_reference=float(values[9]),
    )


def time_to_point(time_s: float, preamble: ScopePreamble) -> int:
    """Convert seconds on the scope time axis to a 1-based point."""

    offset = (time_s - preamble.x_origin) / preamble.x_increment
    return int(round(offset + preamble.x_reference)) + 1


def point_to_time(point: int, preamble: ScopePreamble) -> float:
    """Convert a 1-based waveform point to seconds."""

    return preamble.x_origin + (point - 1 - preamble.x_reference) * preamble.x_increment


def strip_ieee_block(raw: str | bytes) -> str:
    """Remove an IEEE 488.2 definite-length block header when present."""

    text = (
        raw.decode("latin1", errors="replace").strip()
        if isinstance(raw, bytes)
        else raw.strip()
    )
    if len(text) < 2 or text[0] != "#" or not text[1].isdigit():
        return text

    digit_count = int(text[1])
    length_start = 2
    length_stop = length_start + digit_count
    payload_length = int(text[length_start:length_stop])
    payload_start = length_stop
    return text[payload_start : payload_start + payload_length].strip()


def parse_ascii_waveform(raw: str | bytes) -> np.ndarray:
    """Parse an ASCII ``:WAV:DATA?`` payload."""

    payload = strip_ieee_block(raw)
    if not payload:
        return np.asarray([], dtype=float)
    try:
        return np.asarray(
            [float(item) for item in payload.split(",") if item.strip()],
            dtype=float,
        )
    except ValueError as exc:
        preview = payload[:120].replace("\n", "\\n")
        raise ValueError(f"Could not parse waveform data. First bytes: {preview!r}") from exc


def chunk_ranges(start: int, stop: int, chunk_points: int) -> Iterable[tuple[int, int]]:
    """Yield inclusive point ranges for chunked waveform reads."""

    if chunk_points < 1:
        raise ValueError("chunk_points must be 1 or greater.")
    current = int(start)
    while current <= stop:
        chunk_stop = min(int(stop), current + int(chunk_points) - 1)
        yield current, chunk_stop
        current = chunk_stop + 1


def default_visa_resource(ip: str = DEFAULT_IP, *, socket: bool = False) -> str:
    """Build a TCPIP VISA resource for a RIGOL DHO scope."""

    return f"TCPIP::{ip}::5555::SOCKET" if socket else f"TCPIP::{ip}::INSTR"


def query_int(session: Any, command: str) -> int | None:
    """Best-effort integer query for optional scope metadata."""

    try:
        return int(float(str(session.query(command)).strip()))
    except Exception:
        return None


def next_available_path(path: Path) -> Path:
    """Return ``path`` or a numbered sibling without replacing an existing file."""

    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused output path based on {path}")


def timestamped_capture_path(directory: Path | str = "captures") -> Path:
    """Return a timestamped CSV path under ``directory``."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(directory) / f"rigol_window_{timestamp}.csv"


def save_waveform_csv(
    path: Path | str,
    waveform: ScopeWaveform,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a captured waveform as ``time_s, voltage_V`` CSV."""

    output_path = next_available_path(Path(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        for key, value in (metadata or {}).items():
            writer.writerow([f"# {key}", value])
        writer.writerow(["time_s", f"{waveform.channel.lower()}_voltage_V"])
        writer.writerows(zip(waveform.time, waveform.voltage))
    return output_path


def _validate_capture_config(config: WaveformCaptureConfig) -> None:
    normalize_waveform_format(config.waveform_format)
    if config.chunk_points < 1:
        raise ValueError("chunk_points must be 1 or greater.")
    if config.window_start_s is None or config.window_end_s is None:
        return
    if config.window_end_s <= config.window_start_s:
        raise ValueError("window_end_s must be greater than window_start_s.")


def _requested_points(
    config: WaveformCaptureConfig,
    preamble: ScopePreamble,
    available_stop: int,
) -> tuple[int, int]:
    if config.start_point is not None or config.stop_point is not None:
        start = 1 if config.start_point is None else int(config.start_point)
        stop = available_stop if config.stop_point is None else int(config.stop_point)
    elif config.window_start_s is not None and config.window_end_s is not None:
        start = time_to_point(float(config.window_start_s), preamble)
        stop = time_to_point(float(config.window_end_s), preamble)
    else:
        start, stop = 1, available_stop
    return start, stop


def _clamp_points(
    start: int,
    stop: int,
    *,
    available_stop: int,
    preamble: ScopePreamble,
    allow_clamp: bool,
) -> tuple[int, int]:
    if start < 1 or stop > available_stop:
        if not allow_clamp:
            available_time = (point_to_time(1, preamble), point_to_time(available_stop, preamble))
            raise ValueError(
                "Requested waveform window is outside the acquired record: "
                f"points {start} to {stop}; available 1 to {available_stop}; "
                f"available time {available_time[0]:.12g} s to {available_time[1]:.12g} s."
            )
        start = max(start, 1)
        stop = min(stop, available_stop)

    if stop < start:
        raise ValueError(f"Invalid point window: {start} to {stop}")
    return start, stop


def _time_axis(preamble: ScopePreamble, start_point: int, sample_count: int) -> np.ndarray:
    points = start_point + np.arange(sample_count)
    return preamble.x_origin + (points - 1 - preamble.x_reference) * preamble.x_increment


class RigolDHO:
    """Small RIGOL DHO waveform-capture controller."""

    def __init__(
        self,
        resource: str | None = None,
        *,
        ip: str = DEFAULT_IP,
        use_socket: bool = False,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        backend: str | None = None,
        resource_manager: Any | None = None,
    ) -> None:
        self.resource = resource or default_visa_resource(ip, socket=use_socket)
        self.timeout_ms = int(timeout_ms)
        self.backend = backend
        self._external_rm = resource_manager
        self._rm: Any | None = None
        self._session: Any | None = None

    @property
    def session(self) -> Any:
        if self._session is None:
            raise RuntimeError("oscilloscope session is not open")
        return self._session

    def open(self) -> None:
        """Open the VISA resource if needed."""

        if self._session is not None:
            return
        if self._external_rm is None:
            import pyvisa

            try:
                self._rm = (
                    pyvisa.ResourceManager()
                    if self.backend is None
                    else pyvisa.ResourceManager(self.backend)
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not create a PyVISA resource manager. "
                    "Check the VISA backend installation."
                ) from exc
        else:
            self._rm = self._external_rm

        try:
            self._session = self._rm.open_resource(self.resource)
        except Exception as exc:
            raise RuntimeError(
                f"Could not open oscilloscope VISA resource {self.resource!r}."
            ) from exc
        self._session.timeout = self.timeout_ms
        if self.resource.endswith("::SOCKET"):
            self._session.read_termination = "\n"
            self._session.write_termination = "\n"

    connect = open

    def close(self) -> None:
        """Close the scope session and owned ResourceManager."""

        session = self._session
        self._session = None
        if session is not None:
            session.close()
        if self._external_rm is None and self._rm is not None:
            self._rm.close()
        self._rm = None

    disconnect = close

    def __enter__(self) -> "RigolDHO":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_session(self) -> Any:
        if self._session is None:
            self.open()
        return self.session

    def query(self, command: str) -> str:
        """Send a SCPI query and return stripped text."""

        return str(self._ensure_session().query(command)).strip()

    def write(self, command: str) -> None:
        """Send a SCPI command."""

        self._ensure_session().write(command)

    def identify(self) -> str:
        """Return the instrument identity string."""

        return self.query("*IDN?")

    def get_preamble(self) -> ScopePreamble:
        """Query and parse the current waveform preamble."""

        return parse_preamble(self.query(":WAV:PRE?"))

    def trigger_status(self) -> str:
        """Return the scope trigger/acquisition status in uppercase."""

        return self.query(":TRIGger:STATus?").upper()

    def is_stopped(self) -> bool:
        """Return True when the scope reports a stopped acquisition state."""

        return self.trigger_status() in STOPPED_TRIGGER_STATUSES

    def _wait_for_status(
        self,
        accepted: set[str],
        *,
        timeout_s: float,
        poll_s: float,
        message: str,
    ) -> str:
        deadline = time_module.monotonic() + float(timeout_s)
        last_status = "UNKNOWN"
        while time_module.monotonic() <= deadline:
            last_status = self.trigger_status()
            if last_status in accepted:
                return last_status
            time_module.sleep(float(poll_s))
        raise ScopeTimeoutError(f"{message}; last trigger status was {last_status!r}.")

    def wait_until_armed(
        self,
        *,
        timeout_s: float = DEFAULT_ARMED_TIMEOUT_S,
        poll_s: float = DEFAULT_TRIGGER_POLL_S,
    ) -> str:
        """Wait until SINGLE mode is armed and waiting/running for a trigger."""

        return self._wait_for_status(
            ARMED_TRIGGER_STATUSES,
            timeout_s=timeout_s,
            poll_s=poll_s,
            message="Timed out waiting for scope to arm after :SINGle",
        )

    def wait_until_stopped(
        self,
        *,
        timeout_s: float = DEFAULT_TRIGGER_TIMEOUT_S,
        poll_s: float = DEFAULT_TRIGGER_POLL_S,
    ) -> str:
        """Wait until the scope reports STOP after a completed acquisition."""

        return self._wait_for_status(
            STOPPED_TRIGGER_STATUSES,
            timeout_s=timeout_s,
            poll_s=poll_s,
            message="Timed out waiting for SINGLE acquisition to complete",
        )

    def arm_single(
        self,
        *,
        wait_until_armed: bool = True,
        armed_timeout_s: float = DEFAULT_ARMED_TIMEOUT_S,
        poll_s: float = DEFAULT_TRIGGER_POLL_S,
    ) -> str | None:
        """Arm the scope for one trigger using RIGOL ``:SINGle`` mode."""

        self.write(":SINGle")
        if not wait_until_armed:
            return None
        return self.wait_until_armed(timeout_s=armed_timeout_s, poll_s=poll_s)

    def _resolve_point_window(
        self,
        config: WaveformCaptureConfig,
        preamble: ScopePreamble,
    ) -> tuple[int, int]:
        available_stop = query_int(self._ensure_session(), ":ACQ:MDEP?") or preamble.points
        start, stop = _requested_points(config, preamble, int(available_stop))
        return _clamp_points(
            start,
            stop,
            available_stop=int(available_stop),
            preamble=preamble,
            allow_clamp=config.allow_clamp,
        )

    def _configure_waveform_read(
        self,
        config: WaveformCaptureConfig,
    ) -> tuple[str, ScopePreamble]:
        channel = normalize_channel(config.channel)
        if config.stop_before_read:
            self.write(":STOP")
        self.write(f":WAV:SOUR {channel}")
        self.write(f":WAV:MODE {config.waveform_mode}")
        waveform_format = normalize_waveform_format(config.waveform_format)
        self.write(f":WAV:FORM {waveform_format}")
        # Y reference/scaling depend on the selected transfer format.
        preamble = self.get_preamble()
        actual_format = {"0": "BYTE", "1": "WORD", "2": "ASC"}.get(
            preamble.data_format
        )
        if actual_format != waveform_format:
            raise ValueError(
                f"Scope preamble reports format {preamble.data_format!r}; "
                f"requested {waveform_format}. Refusing to decode mismatched data."
            )
        scales = [
            preamble.x_increment, preamble.x_origin, preamble.x_reference,
            preamble.y_increment, preamble.y_origin, preamble.y_reference,
        ]
        if (
            not np.all(np.isfinite(scales))
            or preamble.x_increment <= 0
            or preamble.y_increment <= 0
        ):
            raise ValueError("Scope preamble contains invalid timing or voltage scaling")
        return channel, preamble

    def _read_voltage(
        self, start: int, stop: int, chunk_points: int, preamble: ScopePreamble
    ) -> np.ndarray:
        chunks = []
        waveform_format = {"0": "BYTE", "1": "WORD", "2": "ASC"}[
            preamble.data_format
        ]
        for chunk_start, chunk_stop in chunk_ranges(start, stop, chunk_points):
            self.write(f":WAV:STAR {chunk_start}")
            self.write(f":WAV:STOP {chunk_stop}")
            if waveform_format == "ASC":
                voltage = parse_ascii_waveform(self.query(":WAV:DATA?"))
            else:
                # DHO WORD samples are unsigned little-endian 16-bit codes.
                # Let VISA consume the IEEE block header and trailing terminator;
                # never decode/strip binary payloads as text (0x0A is valid data).
                codes = np.asarray(
                    self._ensure_session().query_binary_values(
                        ":WAV:DATA?",
                        datatype="H" if waveform_format == "WORD" else "B",
                        is_big_endian=False,
                        container=np.array,
                        header_fmt="ieee",
                        expect_termination=True,
                        chunk_size=1024 * 1024,
                    ),
                    dtype=np.float64,
                )
                voltage = (
                    codes - preamble.y_origin - preamble.y_reference
                ) * preamble.y_increment
            expected = chunk_stop - chunk_start + 1
            if voltage.shape != (expected,):
                raise ValueError(
                    f"Scope returned {voltage.size} samples for points "
                    f"{chunk_start} to {chunk_stop}; expected {expected}"
                )
            if not np.all(np.isfinite(voltage)):
                raise ValueError("Scope returned nonfinite waveform voltages")
            chunks.append(voltage)
        return np.concatenate(chunks) if chunks else np.asarray([], dtype=float)

    def capture_waveform(self, config: WaveformCaptureConfig) -> ScopeWaveform:
        """Read a waveform from the scope into memory."""

        _validate_capture_config(config)
        source = self.identify()
        channel, preamble = self._configure_waveform_read(config)
        start_point, stop_point = self._resolve_point_window(config, preamble)
        voltage = self._read_voltage(start_point, stop_point, config.chunk_points, preamble)
        if len(voltage) < 2:
            raise ValueError("Scope returned fewer than two waveform samples.")

        time = _time_axis(preamble, start_point, len(voltage))
        sample_rate = 1.0 / float(np.median(np.diff(time)))
        if config.run_after_read:
            self.write(":RUN")

        return ScopeWaveform(
            time=time,
            voltage=voltage,
            sample_rate=sample_rate,
            source=source,
            channel=channel,
            preamble=preamble,
            start_point=start_point,
            stop_point=start_point + len(voltage) - 1,
        )

    def capture_single_waveform(
        self,
        config: WaveformCaptureConfig,
        *,
        start_shot: Callable[[], None] | None = None,
        trigger_timeout_s: float = DEFAULT_TRIGGER_TIMEOUT_S,
        armed_timeout_s: float = DEFAULT_ARMED_TIMEOUT_S,
        poll_s: float = DEFAULT_TRIGGER_POLL_S,
        confirm_armed: bool = True,
        rearm_after_read: bool = False,
    ) -> ScopeWaveform:
        """Arm SINGLE mode, optionally start a shot, wait for STOP, then read."""

        _validate_capture_config(config)
        self.write(f":WAV:SOUR {normalize_channel(config.channel)}")
        self.arm_single(
            wait_until_armed=confirm_armed,
            armed_timeout_s=armed_timeout_s,
            poll_s=poll_s,
        )
        try:
            if start_shot is not None:
                start_shot()
            self.wait_until_stopped(timeout_s=trigger_timeout_s, poll_s=poll_s)
        except Exception:
            self.write(":STOP")
            raise

        waveform = self.capture_waveform(
            replace(config, stop_before_read=False, run_after_read=False)
        )
        if rearm_after_read:
            self.arm_single(
                wait_until_armed=confirm_armed,
                armed_timeout_s=armed_timeout_s,
                poll_s=poll_s,
            )
        return waveform

    def capture_csv(
        self,
        path: Path | str,
        config: WaveformCaptureConfig,
    ) -> Path:
        """Capture a waveform and save it as CSV."""

        return save_waveform_csv(path, self.capture_waveform(config))
