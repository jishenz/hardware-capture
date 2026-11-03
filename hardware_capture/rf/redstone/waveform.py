"""Waveform models for TNMR-friendly phase and amplitude tables."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar


def _format_number(value: float, precision: int) -> str:
    """Return a compact decimal string."""

    if math.isclose(value, round(value), abs_tol=10 ** (-(precision + 1))):
        return str(int(round(value)))
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def _normalized_points(points: int) -> list[float]:
    """Return evenly spaced points from 0 to 1."""

    if points <= 0:
        raise ValueError("points must be greater than zero.")
    if points == 1:
        return [0.0]
    return [point / (points - 1) for point in range(points)]


def _symmetric_points(points: int) -> list[float]:
    """Return evenly spaced points from -1 to 1."""

    if points <= 0:
        raise ValueError("points must be greater than zero.")
    if points == 1:
        return [0.0]
    return [2.0 * point / (points - 1) - 1.0 for point in range(points)]


def _scale(x: float, low: float, high: float) -> float:
    """Scale a normalized value into the requested range."""

    return low + x * (high - low)


def _wrap_phase(value: float) -> float:
    """Wrap a phase into the range [0, 360)."""

    wrapped = value % 360.0
    if math.isclose(wrapped, 360.0):
        return 0.0
    return wrapped


def _discretize_phase(values: list[float], steps: int) -> list[float]:
    """Wrap and quantize phase values for hardware output."""

    if steps <= 0:
        raise ValueError("phase_steps must be greater than zero.")
    phase_step = 360.0 / steps
    return [
        _wrap_phase(round(_wrap_phase(value) / phase_step) * phase_step)
        for value in values
    ]


def _discretize_amplitude(values: list[float], step: float | None) -> list[float]:
    """Quantize amplitude values for hardware output."""

    if step is None:
        return values
    if step <= 0:
        raise ValueError("amp_step must be greater than zero.")
    return [round(value / step) * step for value in values]


def _format_column(values: list[float], precision: int) -> list[str]:
    """Return formatted output rows."""

    return [_format_number(value, precision) for value in values]


@dataclass(frozen=True)
class WaveformSummary:
    """Compact metadata about a waveform's current outputs."""

    waveform_type: str
    point_count: int
    has_phase: bool
    has_amplitude: bool
    phase_min: float | None
    phase_max: float | None
    amplitude_min: float | None
    amplitude_max: float | None
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary payload."""

        return {
            "waveform_type": self.waveform_type,
            "point_count": self.point_count,
            "has_phase": self.has_phase,
            "has_amplitude": self.has_amplitude,
            "phase_min": self.phase_min,
            "phase_max": self.phase_max,
            "amplitude_min": self.amplitude_min,
            "amplitude_max": self.amplitude_max,
            "parameters": dict(self.parameters),
        }


@dataclass
class Waveform:
    """Base class for TNMR waveform output and lightweight serialization."""

    phase_steps: int = 1024
    amp_step: float | None = None

    _REGISTRY: ClassVar[dict[str, type["Waveform"]]] = {}

    def _raw_phase_values(self) -> list[float] | None:
        """Return unquantized phase values in degrees."""

        return None

    def _raw_amplitude_values(self) -> list[float] | None:
        """Return unquantized amplitude values."""

        return None

    @property
    def waveform_type(self) -> str:
        """Return the lowercase registry key for this waveform."""

        return type(self).__name__.lower()

    def phase_values(self) -> list[float] | None:
        """Return phase values in degrees."""

        values = self._raw_phase_values()
        if values is None:
            return None
        return _discretize_phase(values, self.phase_steps)

    def amplitude_values(self) -> list[float] | None:
        """Return amplitude values."""

        values = self._raw_amplitude_values()
        if values is None:
            return None
        return _discretize_amplitude(values, self.amp_step)

    def phase_column(self, precision: int = 3) -> list[str] | None:
        """Return formatted phase values."""

        values = self.phase_values()
        if values is None:
            return None
        return _format_column(values, precision)

    def amplitude_column(self, precision: int = 3) -> list[str] | None:
        """Return formatted amplitude values."""

        values = self.amplitude_values()
        if values is None:
            return None
        return _format_column(values, precision)

    def sample_records(self, *, precision: int = 6) -> list[dict[str, Any]]:
        """Return GUI-friendly per-sample rows."""

        phase = self.phase_values()
        amplitude = self.amplitude_values()
        point_count = max(
            len(phase or []),
            len(amplitude or []),
        )
        if point_count == 0:
            return []

        normalized = _normalized_points(point_count)
        rows: list[dict[str, Any]] = []
        for index in range(point_count):
            rows.append(
                {
                    "index": index,
                    "fraction": round(normalized[index], precision),
                    "phase_deg": None if phase is None else phase[index],
                    "phase_text": None
                    if phase is None
                    else _format_number(phase[index], precision),
                    "amplitude": None if amplitude is None else amplitude[index],
                    "amplitude_text": None
                    if amplitude is None
                    else _format_number(amplitude[index], precision),
                }
            )
        return rows

    def summary(self) -> WaveformSummary:
        """Return compact current-output metadata."""

        phase = self.phase_values()
        amplitude = self.amplitude_values()
        point_count = max(
            len(phase or []),
            len(amplitude or []),
        )
        return WaveformSummary(
            waveform_type=self.waveform_type,
            point_count=point_count,
            has_phase=phase is not None,
            has_amplitude=amplitude is not None,
            phase_min=min(phase) if phase else None,
            phase_max=max(phase) if phase else None,
            amplitude_min=min(amplitude) if amplitude else None,
            amplitude_max=max(amplitude) if amplitude else None,
            parameters=self.parameter_dict(),
        )

    def parameter_dict(self) -> dict[str, Any]:
        """Return the dataclass fields that define this waveform."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    def to_dict(self, *, include_summary: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly waveform description."""

        payload: dict[str, Any] = {
            "waveform_type": self.waveform_type,
            "parameters": self.parameter_dict(),
        }
        if include_summary:
            payload["summary"] = self.summary().to_dict()
        return payload

    def to_json(self, *, include_summary: bool = False, indent: int = 2) -> str:
        """Return the waveform configuration as JSON text."""

        return json.dumps(
            self.to_dict(include_summary=include_summary),
            indent=indent,
        )

    def write_phase_column(
        self,
        path: str | Path,
        precision: int = 3,
    ) -> Path:
        """Write formatted phase values to a text file."""

        lines = self.phase_column(precision=precision)
        if lines is None:
            raise ValueError("Phase output is not available.")
        output_path = Path(path).expanduser().resolve()
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def write_amplitude_column(
        self,
        path: str | Path,
        precision: int = 3,
    ) -> Path:
        """Write formatted amplitude values to a text file."""

        lines = self.amplitude_column(precision=precision)
        if lines is None:
            raise ValueError("Amplitude output is not available.")
        output_path = Path(path).expanduser().resolve()
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def write_json(
        self,
        path: str | Path,
        *,
        include_summary: bool = False,
        indent: int = 2,
    ) -> Path:
        """Write the waveform configuration to disk."""

        output_path = Path(path).expanduser().resolve()
        output_path.write_text(
            self.to_json(include_summary=include_summary, indent=indent) + "\n",
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def register(cls, waveform_cls: type["Waveform"]) -> type["Waveform"]:
        """Register a waveform subclass for JSON dispatch."""

        cls._REGISTRY[waveform_cls.__name__.lower()] = waveform_cls
        return waveform_cls

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Waveform":
        """Build a waveform from a JSON-like payload."""

        waveform_type = str(
            payload.get("waveform_type")
            or payload.get("type")
            or payload.get("kind")
            or ""
        ).strip().lower()
        if not waveform_type:
            raise ValueError("waveform_type is required.")

        parameters = payload.get("parameters")
        if parameters is None:
            parameters = {
                key: value
                for key, value in payload.items()
                if key not in {"waveform_type", "type", "kind", "summary"}
            }
        if not isinstance(parameters, dict):
            raise ValueError("Waveform parameters must be a JSON object.")

        waveform_cls = cls._REGISTRY.get(waveform_type)
        if waveform_cls is None:
            supported = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(
                f"Unsupported waveform_type {waveform_type!r}. Supported types: {supported}."
            )
        return waveform_cls(**parameters)

    @classmethod
    def from_json(cls, text: str) -> "Waveform":
        """Build a waveform from JSON text."""

        return cls.from_dict(json.loads(text))

    @classmethod
    def load_json_file(cls, path: str | Path) -> "Waveform":
        """Load a waveform JSON file from disk."""

        resolved = Path(path).expanduser().resolve()
        return cls.from_json(resolved.read_text(encoding="utf-8"))


@dataclass
class MappedWaveform(Waveform):
    """Base class for waveforms driven by normalized sample values."""

    points: int = 128
    amp_min: float | None = 0.0
    amp_max: float | None = 1.0
    phase_min: float | None = None
    phase_max: float | None = None

    def _normalized_shape_values(self) -> list[float]:
        """Return normalized values in the range [0, 1]."""

        raise NotImplementedError

    def _scaled_values(self, low: float | None, high: float | None) -> list[float] | None:
        """Scale normalized samples into a requested range."""

        if low is None or high is None:
            return None
        return [
            _scale(value, low, high)
            for value in self._normalized_shape_values()
        ]

    def _raw_amplitude_values(self) -> list[float] | None:
        """Return unquantized amplitude values."""

        return self._scaled_values(self.amp_min, self.amp_max)

    def _raw_phase_values(self) -> list[float] | None:
        """Return unquantized phase values."""

        return self._scaled_values(self.phase_min, self.phase_max)


@dataclass
class PeriodicWaveform(MappedWaveform):
    """Mapped waveform driven by cycles and a phase offset."""

    cycles: float = 1.0
    offset_deg: float = 0.0


@Waveform.register
@dataclass
class Sine(PeriodicWaveform):
    """Sine-shaped waveform."""

    def _normalized_shape_values(self) -> list[float]:
        """Return normalized sine-wave values."""

        values = []
        for point in _normalized_points(self.points):
            angle = 2.0 * math.pi * self.cycles * point
            angle += math.radians(self.offset_deg)
            values.append(0.5 + 0.5 * math.sin(angle))
        return values


@Waveform.register
@dataclass
class Triangle(PeriodicWaveform):
    """Triangle-shaped waveform."""

    def _normalized_shape_values(self) -> list[float]:
        """Return normalized triangle-wave values."""

        values = []
        for point in _normalized_points(self.points):
            shifted = (self.cycles * point + self.offset_deg / 360.0) % 1.0
            values.append(1.0 - abs(2.0 * shifted - 1.0))
        return values


@Waveform.register
@dataclass
class Linear(MappedWaveform):
    """Linear ramp waveform."""

    def _normalized_shape_values(self) -> list[float]:
        """Return normalized linear-ramp values."""

        return _normalized_points(self.points)


@Waveform.register
@dataclass
class Constant(Waveform):
    """Constant waveform."""

    points: int = 128
    amp: float | None = 0.0
    phase: float | None = None

    def _raw_amplitude_values(self) -> list[float] | None:
        """Return unquantized constant amplitude values."""

        if self.amp is None:
            return None
        return [self.amp] * self.points

    def _raw_phase_values(self) -> list[float] | None:
        """Return unquantized constant phase values."""

        if self.phase is None:
            return None
        return [self.phase] * self.points


@Waveform.register
@dataclass
class Chirp(Waveform):
    """TNMR PhMod phase table for a requested physical RF chirp.

    ``center_hz`` is the center of the requested absolute RF sweep, while
    ``offset_hz`` is the RF offset already applied by the sequence before the
    PhMod table. The TNMR PhMod phase table is the negative integral of
    ``requested_offset - offset_hz``.

    Warning: Redstone/TNMR PhMod table phase uses the opposite sign from the
    physical RF frequency offset. An increasing table phase slope lowers the
    RF frequency, so this waveform writes the negative integral of the
    requested offset.
    """

    points: int = 1024
    sweep_width_hz: float = 5000.0
    dwell_s: float = 5.0e-6
    center_hz: float = 0.0
    offset_hz: float = 0.0
    phase: float = 0.0

    def frequency_values(self) -> list[float]:
        """Return requested physical RF offsets across the linear sweep."""

        if self.dwell_s <= 0:
            raise ValueError("dwell_s must be greater than zero.")
        half_width = self.sweep_width_hz / 2.0
        return [
            self.center_hz + _scale(point, -half_width, half_width)
            for point in _normalized_points(self.points)
        ]

    def _raw_phase_values(self) -> list[float]:
        """Return the unquantized TNMR PhMod phase table."""

        phase_deg = self.phase
        values: list[float] = []
        for frequency_hz in self.frequency_values():
            phase_deg -= 360.0 * (frequency_hz - self.offset_hz) * self.dwell_s
            values.append(phase_deg)
        return values


@Waveform.register
@dataclass
class HS1(Waveform):
    """Hyperbolic-secant adiabatic pulse for Redstone TNMR tables.

    The amplitude follows ``sech(x)`` and the requested physical RF offset
    follows a finite-duration normalized ``tanh(x)`` sweep. Like ``Chirp``,
    ``center_hz`` is the center of the requested absolute RF sweep, while
    ``offset_hz`` is the RF offset already applied by the sequence before the
    PhMod table. The TNMR PhMod phase table is the negative integral of
    ``requested_offset - offset_hz``.
    """

    points: int = 1024
    sweep_width_hz: float = 5000.0
    dwell_s: float = 5.0e-6
    truncation: float = 5.0
    amp: float = 1.0
    center_hz: float = 0.0
    offset_hz: float = 0.0
    phase: float = 0.0

    def _axis_values(self) -> list[float]:
        """Return the finite HS1 axis from ``-truncation`` to ``+truncation``."""

        if self.dwell_s <= 0:
            raise ValueError("dwell_s must be greater than zero.")
        if self.truncation <= 0:
            raise ValueError("truncation must be greater than zero.")
        if self.amp < 0:
            raise ValueError("amp must be greater than or equal to zero.")
        return [
            self.truncation * point
            for point in _symmetric_points(self.points)
        ]

    def amplitude_envelope_values(self) -> list[float]:
        """Return the requested HS1 amplitude envelope."""

        return [self.amp / math.cosh(axis) for axis in self._axis_values()]

    def frequency_values(self) -> list[float]:
        """Return requested physical RF offsets across the finite HS1 sweep."""

        half_width = self.sweep_width_hz / 2.0
        scale = math.tanh(self.truncation)
        return [
            self.center_hz + half_width * math.tanh(axis) / scale
            for axis in self._axis_values()
        ]

    def _raw_amplitude_values(self) -> list[float]:
        """Return the unquantized HS1 amplitude table."""

        return self.amplitude_envelope_values()

    def _raw_phase_values(self) -> list[float]:
        """Return the unquantized TNMR PhMod phase table."""

        phase_deg = self.phase
        values: list[float] = []
        for frequency_hz in self.frequency_values():
            phase_deg -= 360.0 * (frequency_hz - self.offset_hz) * self.dwell_s
            values.append(phase_deg)
        return values


@Waveform.register
@dataclass
class SAHS1(HS1):
    """Power-matched superadiabatic HS1 pulse for Redstone TNMR tables.

    ``nutation_hz`` is the RF nutation frequency produced by an amplitude-table
    value of one. ``amp`` is the requested amplitude-table ceiling. The base
    HS1 amplitude is reduced until the combined HS1 and exact counterdiabatic
    quadratures fit below that ceiling.

    The amplitude table contains the combined transverse magnitude. The phase
    table adds ``atan2(counterdiabatic, HS1)`` to the usual HS1 chirp phase.
    """

    points: int = 4000
    nutation_hz: float = 160000.0

    def _quadrature_nutation_values(
        self,
        base_nutation_hz: float,
    ) -> tuple[list[float], list[float]]:
        """Return the HS1 and exact counterdiabatic quadratures in hertz."""

        axes = self._axis_values()
        if self.nutation_hz <= 0:
            raise ValueError("nutation_hz must be greater than zero.")

        duration_s = self.points * self.dwell_s
        dx_dt = 2.0 * self.truncation / duration_s
        detuning_scale_hz = (
            self.sweep_width_hz
            / (2.0 * math.tanh(self.truncation))
        )

        x_values: list[float] = []
        cd_values: list[float] = []
        for axis in axes:
            sech = 1.0 / math.cosh(axis)
            x_nutation_hz = base_nutation_hz * sech
            detuning_hz = detuning_scale_hz * math.tanh(axis)
            denominator_hz2 = (
                x_nutation_hz * x_nutation_hz
                + detuning_hz * detuning_hz
            )
            if math.isclose(denominator_hz2, 0.0, abs_tol=1e-30):
                cd_nutation_hz = 0.0
            else:
                cd_nutation_hz = (
                    -dx_dt
                    * base_nutation_hz
                    * detuning_scale_hz
                    * sech
                    / (2.0 * math.pi * denominator_hz2)
                )
            x_values.append(x_nutation_hz)
            cd_values.append(cd_nutation_hz)
        return x_values, cd_values

    def _peak_transverse_nutation_hz(self, base_nutation_hz: float) -> float:
        """Return the peak combined transverse nutation for a base amplitude."""

        x_values, cd_values = self._quadrature_nutation_values(base_nutation_hz)
        return max(
            math.hypot(x_nutation_hz, cd_nutation_hz)
            for x_nutation_hz, cd_nutation_hz in zip(x_values, cd_values)
        )

    def base_nutation_hz(self) -> float:
        """Return the highest exact-CD HS1 base amplitude below the RF ceiling."""

        self._axis_values()
        if self.nutation_hz <= 0:
            raise ValueError("nutation_hz must be greater than zero.")

        ceiling_hz = self.amp * self.nutation_hz
        if math.isclose(ceiling_hz, 0.0, abs_tol=1e-30):
            return 0.0

        trial_values = [
            ceiling_hz * (1.0e-4 + index * (1.0 - 1.0e-4) / 500.0)
            for index in range(501)
        ]
        feasible_indices = [
            index
            for index, trial_hz in enumerate(trial_values)
            if self._peak_transverse_nutation_hz(trial_hz)
            <= ceiling_hz * (1.0 + 1.0e-12)
        ]
        if not feasible_indices:
            raise ValueError(
                "No exact counterdiabatic HS1 solution fits the RF amplitude "
                "ceiling; increase points, dwell_s, amp, or nutation_hz."
            )

        index = feasible_indices[-1]
        if index == len(trial_values) - 1:
            return trial_values[index]

        lower_hz = trial_values[index]
        upper_hz = trial_values[index + 1]
        for _ in range(60):
            midpoint_hz = 0.5 * (lower_hz + upper_hz)
            if (
                self._peak_transverse_nutation_hz(midpoint_hz)
                <= ceiling_hz
            ):
                lower_hz = midpoint_hz
            else:
                upper_hz = midpoint_hz
        return lower_hz

    def quadrature_nutation_values(self) -> tuple[list[float], list[float]]:
        """Return the power-matched HS1 and counterdiabatic fields in hertz."""

        return self._quadrature_nutation_values(self.base_nutation_hz())

    def counterdiabatic_phase_values(self) -> list[float]:
        """Return the center-referenced counterdiabatic phase in degrees."""

        x_values, cd_values = self.quadrature_nutation_values()
        corrections = [
            math.degrees(math.atan2(cd_nutation_hz, x_nutation_hz))
            for x_nutation_hz, cd_nutation_hz in zip(x_values, cd_values)
        ]
        reference = corrections[len(corrections) // 2]
        return [correction - reference for correction in corrections]

    def amplitude_envelope_values(self) -> list[float]:
        """Return the power-matched combined transverse amplitude table."""

        x_values, cd_values = self.quadrature_nutation_values()
        return [
            math.hypot(x_nutation_hz, cd_nutation_hz) / self.nutation_hz
            for x_nutation_hz, cd_nutation_hz in zip(x_values, cd_values)
        ]

    def _raw_phase_values(self) -> list[float]:
        """Return the HS1 chirp phase plus counterdiabatic phase correction."""

        chirp_values = super()._raw_phase_values()
        return [
            chirp_deg + correction_deg
            for chirp_deg, correction_deg in zip(
                chirp_values,
                self.counterdiabatic_phase_values(),
            )
        ]


@Waveform.register
@dataclass
class WURST(Waveform):
    """WURST-n adiabatic pulse for Redstone TNMR tables.

    The amplitude follows ``amp * (1 - abs(cos(pi * t / T)) ** order)`` and
    the requested physical RF offset follows a linear sweep. As with ``HS1``,
    ``center_hz`` is the center of the requested absolute RF sweep, while
    ``offset_hz`` is the RF offset already applied by the sequence before the
    PhMod table. The TNMR PhMod phase table is the negative integral of
    ``requested_offset - offset_hz``.
    """

    points: int = 1024
    sweep_width_hz: float = 5000.0
    dwell_s: float = 5.0e-6
    order: float = 20.0
    amp: float = 1.0
    center_hz: float = 0.0
    offset_hz: float = 0.0
    phase: float = 0.0

    def _shape_points(self) -> list[float]:
        """Return normalized WURST sample positions from 0 to 1."""

        if self.dwell_s <= 0:
            raise ValueError("dwell_s must be greater than zero.")
        if self.order <= 0:
            raise ValueError("order must be greater than zero.")
        if self.amp < 0:
            raise ValueError("amp must be greater than or equal to zero.")
        if self.points == 1:
            return [0.5]
        return _normalized_points(self.points)

    def amplitude_envelope_values(self) -> list[float]:
        """Return the requested WURST-n amplitude envelope."""

        return [
            self.amp * (1.0 - abs(math.cos(math.pi * point)) ** self.order)
            for point in self._shape_points()
        ]

    def frequency_values(self) -> list[float]:
        """Return requested physical RF offsets across the linear sweep."""

        half_width = self.sweep_width_hz / 2.0
        return [
            self.center_hz + _scale(point, -half_width, half_width)
            for point in self._shape_points()
        ]

    def _raw_amplitude_values(self) -> list[float]:
        """Return the unquantized WURST amplitude table."""

        return self.amplitude_envelope_values()

    def _raw_phase_values(self) -> list[float]:
        """Return the unquantized TNMR PhMod phase table."""

        phase_deg = self.phase
        values: list[float] = []
        for frequency_hz in self.frequency_values():
            phase_deg -= 360.0 * (frequency_hz - self.offset_hz) * self.dwell_s
            values.append(phase_deg)
        return values


@Waveform.register
@dataclass
class WURSTtanh(Waveform):
    """WURST amplitude with hyperbolic-secant-style tanh frequency sweep.

    The amplitude follows ``amp * (1 - abs(cos(pi * t / T)) ** order)`` and
    the requested physical RF offset follows a normalized ``tanh(beta * x)``
    sweep from ``-sweep_width_hz / 2`` to ``+sweep_width_hz / 2``. ``center_hz``
    is the center of the requested absolute RF sweep, while ``offset_hz`` is
    the RF offset already applied by the sequence before the PhMod table.
    """

    points: int = 1024
    sweep_width_hz: float = 5000.0
    dwell_s: float = 5.0e-6
    order: float = 20.0
    beta: float = 5.0
    amp: float = 1.0
    center_hz: float = 0.0
    offset_hz: float = 0.0
    phase: float = 0.0

    def _shape_points(self) -> list[float]:
        """Return normalized WURST sample positions from 0 to 1."""

        if self.dwell_s <= 0:
            raise ValueError("dwell_s must be greater than zero.")
        if self.order <= 0:
            raise ValueError("order must be greater than zero.")
        if self.beta <= 0:
            raise ValueError("beta must be greater than zero.")
        if self.amp < 0:
            raise ValueError("amp must be greater than or equal to zero.")
        if self.points == 1:
            return [0.5]
        return _normalized_points(self.points)

    def _sweep_axis_values(self) -> list[float]:
        """Return normalized HS sweep positions from -1 to 1."""

        if self.points == 1:
            return [0.0]
        return _symmetric_points(self.points)

    def amplitude_envelope_values(self) -> list[float]:
        """Return the requested WURST-n amplitude envelope."""

        return [
            self.amp * (1.0 - abs(math.cos(math.pi * point)) ** self.order)
            for point in self._shape_points()
        ]

    def frequency_values(self) -> list[float]:
        """Return requested physical RF offsets across the tanh sweep."""

        self._shape_points()
        half_width = self.sweep_width_hz / 2.0
        scale = math.tanh(self.beta)
        return [
            self.center_hz + half_width * math.tanh(self.beta * axis) / scale
            for axis in self._sweep_axis_values()
        ]

    def _raw_amplitude_values(self) -> list[float]:
        """Return the unquantized WURST amplitude table."""

        return self.amplitude_envelope_values()

    def _raw_phase_values(self) -> list[float]:
        """Return the unquantized TNMR PhMod phase table."""

        phase_deg = self.phase
        values: list[float] = []
        for frequency_hz in self.frequency_values():
            phase_deg -= 360.0 * (frequency_hz - self.offset_hz) * self.dwell_s
            values.append(phase_deg)
        return values


@Waveform.register
@dataclass
class BB1(Waveform):
    """BB1 composite-pulse phase list."""

    angle: float = 90.0
    phase: float = 0.0
    split: bool = True
    correction_only: bool = False
    duration_units: str = "subpulse"
    duration_step_deg: float | None = None
    repeats: int = 1

    def segments(self) -> list[dict[str, float]]:
        """Return GUI-friendly BB1 segment records."""

        return [
            {
                "index": index,
                "phase_deg": phase_deg,
                "flip_angle_deg": flip_angle_deg,
            }
            for index, (phase_deg, flip_angle_deg) in enumerate(self._segments())
        ]

    def _segments(self) -> list[tuple[float, float]]:
        """Return ``(phase, flip_angle_deg)`` BB1 segments."""

        theta_rad = math.radians(self.angle)
        argument = -theta_rad / (4.0 * math.pi)
        if not -1.0 <= argument <= 1.0:
            raise ValueError("angle is outside the simple BB1 range.")
        phi = math.degrees(math.acos(argument))
        correction = [
            (self.phase + phi, 180.0),
            (self.phase + 3.0 * phi, 360.0),
            (self.phase + phi, 180.0),
        ]
        if self.correction_only:
            return correction
        if self.split:
            return [
                (self.phase, self.angle / 2.0),
                *correction,
                (self.phase, self.angle / 2.0),
            ]
        return [(self.phase, self.angle), *correction]

    def _raw_phase_values(self) -> list[float]:
        """Return unquantized BB1 phases."""

        if self.repeats <= 0:
            raise ValueError("repeats must be greater than zero.")
        segments = self._segments()
        if self.duration_step_deg is not None:
            if self.duration_step_deg <= 0:
                raise ValueError("duration_step_deg must be greater than zero.")
            values: list[float] = []
            for phase_deg, flip_angle in segments:
                units = flip_angle / self.duration_step_deg
                if not math.isclose(units, round(units), abs_tol=1e-9):
                    raise ValueError(
                        "duration_step_deg must evenly divide all BB1 segment angles."
                    )
                values.extend([phase_deg] * int(round(units)))
        elif self.duration_units == "subpulse":
            values = [phase_deg for phase_deg, _flip_angle in segments]
        elif self.duration_units == "pi":
            values = []
            for phase_deg, flip_angle in segments:
                units = flip_angle / 180.0
                if not math.isclose(units, round(units), abs_tol=1e-9):
                    raise ValueError(
                        "duration_units='pi' requires all BB1 segments to be whole-pi rotations."
                    )
                values.extend([phase_deg] * int(round(units)))
        else:
            raise ValueError("duration_units must be 'subpulse' or 'pi'.")
        return values * self.repeats


def load_waveform_json(path: str | Path) -> Waveform:
    """Load a waveform JSON file from disk."""

    return Waveform.load_json_file(path)


def available_waveform_types() -> list[str]:
    """Return the registered waveform types."""

    return sorted(Waveform._REGISTRY)


__all__ = [
    "Waveform",
    "WaveformSummary",
    "Sine",
    "Triangle",
    "Linear",
    "Constant",
    "Chirp",
    "HS1",
    "SAHS1",
    "WURST",
    "WURSTtanh",
    "BB1",
    "available_waveform_types",
    "load_waveform_json",
]
