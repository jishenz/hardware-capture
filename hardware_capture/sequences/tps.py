"""Lossless TPS sequence parsing and JSON/TPS round-tripping.

The parser stays conservative:

- the default JSON export mirrors only byte-backed TPS structure
- debug data is opt-in so parser bookkeeping does not pollute the schema
- unknown TPS chunks are preserved raw instead of being guessed

The row/event portion of the format is still based on repeated structures seen
across real TPS files rather than formal Tecmag documentation, so this module
prefers preserving bytes over inventing semantics.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from hardware_capture.pulses.waveforms import Waveform

HEADER_LENGTH = 16
EVENT_TABLE_SLOT_COUNT = 7
TABLE_TRAILING_U32_COUNT = 7
EXTENDED_TABLE_TRAILING_U32_COUNT = 15
F1_PHASE_MOD_LINE = "F1_PhMod"
F1_PHASE_MOD_TABLE_SLOT = 0
F1_PHASE_MOD_TABLE_TYPE_TEXT = "3P"
F1_PHASE_MOD_TABLE_TYPE = int.from_bytes(
    F1_PHASE_MOD_TABLE_TYPE_TEXT.encode("latin1").ljust(4, b"\x00"),
    "little",
)
F1_PHASE_MOD_ADDRESS_LOW_BYTE = 0x1F
F1_PHASE_MOD_BIT_LENGTH = 16
PRINTABLE_RE = re.compile(rb"[ -~]{3,}")
TPS_HEADER_RE = re.compile(r"^PSEQ(?P<version>[\d.]+)\s+(?P<kind>[A-Z]+)$")
STRUCTURED_JSON_KEYS = (
    "header",
    "stored_sequence",
    "sequence_rows",
    "tail_sections",
    "debug",
    "analysis",
    "serialization",
)
TAIL_SECTION_ORDER = ("tables", "parameter_pages", "parameters", "comment")
KNOWN_TAIL_SECTION_KINDS = frozenset(TAIL_SECTION_ORDER)

KNOWN_CONTROL_LINES = {
    "F0_PhRst",
    "F1_PhRst",
    "F1_Ampl",
    "F1_Atten",
    "F1_PhMod",
    "F1_Ph",
    "F1_Freq",
    "F1_TxGate",
    "F1_UnBlank",
    "F2_PhRst",
    "F2_Ampl",
    "F2_Atten",
    "F2_PhMod",
    "F2_Ph",
    "F2_Freq",
    "F2_TxGate",
    "F2_UnBlank",
    "RX_PhRst",
    "RX_Phase",
    "RX_Freq",
    "RX_Blank",
    "Acq",
    "Acq_phase",
    "Scope_Trig",
    "TTl_1",
    "TTl_2",
    "TTl_3",
    "TTl_4",
    "TTl_5",
    "TTl_6",
    "TTl_7",
    "TTl_8",
    "TTl_9",
    "TTl_10",
    "TTl_11",
    "TTl_12",
    "TTl_13",
    "Loop",
    "CBranch",
    "CTest",
    "Ext_Trig",
    "RT_Update",
    "F1_Freq_I",
    "F1_Freq_Q",
    "F2_Freq_I",
    "F2_Freq_Q",
    "RX_Freq_I",
    "RX_Freq_Q",
    "F1_Freq_dummy",
    "F2_Freq_dummy",
    "RX_Freq_dummy",
    "F1_H/L",
    "F2_H/L",
}


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class _SectionParseError(ValueError):
    """Raised when a documented TPS section cannot be parsed at an offset."""


class SequenceParameterError(ValueError):
    """Raised when an offline sequence parameter edit is invalid.

    Parameter edits are intentionally conservative: v1 only changes existing
    TNMR parameter entries by exact name, preserving the parameter's type,
    limits, and read-only metadata.
    """


class SequenceWaveformError(ValueError):
    """Raised when a waveform cannot be applied to a sequence table."""


@dataclass(frozen=True)
class SequenceTableReference:
    """One visible sequence-event reference to a named TNMR table.

    ``quantity`` is inferred from the sequence line name for inspection only.
    """

    line: str
    event_number: int
    table_name: str
    data: str
    quantity: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly table-reference payload."""

        return {
            "line": self.line,
            "event_number": self.event_number,
            "table_name": self.table_name,
            "data": self.data,
            "quantity": self.quantity,
        }


@dataclass(frozen=True)
class SequenceSummary:
    """Compact current-state metadata for a parsed TPS sequence."""

    source_path: str | None
    stored_sequence_path: str | None
    signature: str
    version: str | None
    container_kind: str | None
    row_count: int | None
    column_count: int | None
    event_count: int
    control_lines: tuple[str, ...]
    parameter_references: tuple[str, ...]
    table_count: int
    parameter_page_count: int
    parameter_count: int
    raw_tail_chunk_count: int
    comment_present: bool
    rows_parse_error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary payload."""

        return {
            "source_path": self.source_path,
            "stored_sequence_path": self.stored_sequence_path,
            "signature": self.signature,
            "version": self.version,
            "container_kind": self.container_kind,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "event_count": self.event_count,
            "control_lines": list(self.control_lines),
            "parameter_references": list(self.parameter_references),
            "table_count": self.table_count,
            "parameter_page_count": self.parameter_page_count,
            "parameter_count": self.parameter_count,
            "raw_tail_chunk_count": self.raw_tail_chunk_count,
            "comment_present": self.comment_present,
            "rows_parse_error": self.rows_parse_error,
        }


@dataclass(frozen=True)
class SequenceTableCell:
    """One visible cell in the TNMR-style sequence table."""

    event_index: int
    data: str
    table_references: tuple[str, ...]
    extra_strings: tuple[str, ...]
    extra_raw_present: bool

    @property
    def is_parameter_reference(self) -> bool:
        """Return whether the cell references a named parameter."""

        return self.data.startswith("=")

    @property
    def is_blank(self) -> bool:
        """Return whether the visible cell has no decoded content."""

        return (
            not self.data
            and not self.table_references
            and not self.extra_strings
            and not self.extra_raw_present
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly cell payload."""

        return {
            "event_index": self.event_index,
            "data": self.data,
            "table_references": list(self.table_references),
            "extra_strings": list(self.extra_strings),
            "extra_raw_present": self.extra_raw_present,
            "is_parameter_reference": self.is_parameter_reference,
            "is_blank": self.is_blank,
        }


@dataclass(frozen=True)
class SequenceTableRow:
    """One visible row in the TNMR-style sequence table."""

    row_index: int
    section: str
    label: str
    default_text: str
    group: int
    visible_flag: int
    cells: tuple[SequenceTableCell, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row payload."""

        return {
            "row_index": self.row_index,
            "section": self.section,
            "label": self.label,
            "default_text": self.default_text,
            "group": self.group,
            "visible_flag": self.visible_flag,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True)
class SequenceTableView:
    """TNMR-editor-style sequence matrix derived from parsed rows."""

    event_count: int
    rows: tuple[SequenceTableRow, ...]

    @property
    def event_numbers(self) -> tuple[int, ...]:
        """Return one-based event numbers for the visible matrix columns."""

        return tuple(range(1, self.event_count + 1))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly table payload."""

        return {
            "event_count": self.event_count,
            "event_numbers": list(self.event_numbers),
            "rows": [row.to_dict() for row in self.rows],
        }


class Sequence:
    """One TNMR sequence as a mutable Python domain object.

    Public attributes map directly to the major TNMR sequence sections:
    ``rows``, ``tables``, ``parameter_pages``, ``parameters``, and
    ``sequence_comment``. Raw byte offsets and tail chunks are kept alongside
    those sections so parsed files can still round-trip when some suffix bytes
    are opaque.

    Offline edits update the structured model and are reflected by JSON and
    rebuilt TPS output. Edit provenance is kept in ``analysis.edits`` for JSON
    consumers only; it is never serialized back into the binary TPS payload.
    """

    def __init__(
        self,
        source_path: str | None,
        *,
        raw_bytes: bytes = b"",
        signature: str = "",
        version: str | None = None,
        container_kind: str | None = None,
        stored_sequence_path: str | None = None,
        stored_file_name: str | None = None,
        row_count: int | None = None,
        column_count: int | None = None,
        header_extra_bytes: bytes = b"",
        header_end_offset: int = 0,
        rows_raw_bytes: bytes = b"",
        rows: list[dict[str, Any]] | None = None,
        rows_parse_error: str | None = None,
        tables_offset: int | None = None,
        parameter_pages_offset: int | None = None,
        parameters_offset: int | None = None,
        comment_offset: int | None = None,
        tail_layout: list[dict[str, Any]] | None = None,
        control_lines: list[str] | None = None,
        parameter_references: list[str] | None = None,
        tables: list[dict[str, Any]] | None = None,
        parameter_pages: list[dict[str, Any]] | None = None,
        parameters: list[dict[str, Any]] | None = None,
        sequence_comment: str | None = None,
        analysis: dict[str, Any] | None = None,
        strings: list[dict[str, Any]] | None = None,
    ) -> None:
        """Create a sequence from parsed TNMR sections."""

        self.source_path = source_path
        self.raw_bytes = raw_bytes
        self.signature = signature
        self.version = version
        self.container_kind = container_kind
        self.stored_sequence_path = stored_sequence_path
        self.stored_file_name = stored_file_name
        self.row_count = row_count
        self.column_count = column_count
        self.header_extra_bytes = header_extra_bytes
        self.header_end_offset = header_end_offset
        self.rows_raw_bytes = rows_raw_bytes
        self.rows_parse_error = rows_parse_error
        self.tables_offset = tables_offset
        self.parameter_pages_offset = parameter_pages_offset
        self.parameters_offset = parameters_offset
        self.comment_offset = comment_offset
        self.sequence_comment = sequence_comment
        self.strings = copy.deepcopy(strings or [])

        self.rows = [_normalize_row(row) for row in (rows or [])]
        self.tables = [_normalize_table(table) for table in (tables or [])]
        self.parameter_pages = [
            _normalize_parameter_page(page) for page in (parameter_pages or [])
        ]
        self.parameters = [
            _normalize_parameter(parameter) for parameter in (parameters or [])
        ]
        self.analysis = _normalize_analysis(analysis)
        self.tail_layout = _normalize_tail_layout(tail_layout or [])
        self.control_lines = _ordered_unique(
            [str(value) for value in (control_lines or [])]
        )
        self.parameter_references = _ordered_unique(
            [str(value) for value in (parameter_references or [])]
        )
        self._sync_derived_fields()

    @property
    def file_size(self) -> int:
        """Return the current TPS payload size in bytes."""

        return len(self.to_bytes())

    @property
    def sha256(self) -> str:
        """Return a SHA-256 digest of the current TPS payload."""

        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def sequence_rows_end_offset(self) -> int:
        """Return the end offset of the sequence-row section."""

        return self.header_end_offset + len(self.sequence_rows_raw)

    @property
    def sequence_rows_raw(self) -> bytes:
        """Return the row/event bytes between the header and tail sections."""

        if self.rows:
            return b"".join(_build_row(row) for row in self.rows)
        return self.rows_raw_bytes

    @property
    def current_row_count(self) -> int | None:
        """Return the row count implied by the current structured state."""

        if self.rows:
            return len(self.rows)
        return self.row_count

    @property
    def current_column_count(self) -> int | None:
        """Return the current global column count."""

        if self.rows:
            return max((row["column_count"] for row in self.rows), default=0)
        return self.column_count

    @property
    def current_control_lines(self) -> list[str]:
        """Return control-line labels derived from the current rows."""

        if self.rows:
            return _derive_control_lines(self.rows)
        return list(self.control_lines)

    @property
    def current_parameter_references(self) -> list[str]:
        """Return ``=parameter`` references derived from the current rows."""

        if self.rows:
            return _derive_parameter_references(self.rows)
        return list(self.parameter_references)

    @property
    def event_count(self) -> int:
        """Return the total number of decoded row events."""

        return sum(len(row.get("events", [])) for row in self.rows)

    @property
    def visible_event_count(self) -> int:
        """Return the maximum number of visible event cells across all rows."""

        return max(
            (len(_visible_row_events(row)) for row in self.rows),
            default=0,
        )

    def to_dict(
        self,
        *,
        include_strings: bool = False,
        include_raw_bytes: bool = False,
        include_sequence_rows_raw: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-serializable dictionary.

        The canonical JSON mirrors the TPS layout. ``analysis`` is included
        only when provenance metadata exists, keeping unedited round-trips
        compact and stable.
        """

        payload: dict[str, Any] = {
            "header": {
                "signature": self.signature,
                "version": self.version,
                "container_kind": self.container_kind,
            },
            "stored_sequence": {
                "path": self.stored_sequence_path,
            },
            "sequence_rows": {
                "row_count": self.current_row_count,
                "column_count": self.current_column_count,
                "rows": [_row_to_json(row) for row in self.rows],
            },
            "tail_sections": _tail_sections_to_json(self),
        }
        debug: dict[str, Any] = {}
        if self.rows_parse_error is not None:
            debug["rows_parse_error"] = self.rows_parse_error
        if self.header_extra_bytes:
            payload["header"]["extra_bytes_base64"] = _encode_base64(
                self.header_extra_bytes
            )
        if include_strings:
            debug["printable_strings"] = copy.deepcopy(self.strings)
        if include_sequence_rows_raw:
            payload["sequence_rows"]["raw_bytes_base64"] = _encode_base64(
                self.sequence_rows_raw
            )
        if include_raw_bytes:
            debug["raw_bytes_base64"] = _encode_base64(self.to_bytes())
            debug["file_size"] = self.file_size
            debug["sha256"] = self.sha256
            debug["byte_offsets"] = self._current_offsets()
        if self.analysis:
            payload["analysis"] = copy.deepcopy(self.analysis)
        if debug:
            payload["debug"] = debug
        return payload

    def to_json(
        self,
        *,
        include_strings: bool = False,
        include_raw_bytes: bool = False,
        include_sequence_rows_raw: bool = False,
        indent: int = 2,
    ) -> str:
        """Return the parsed data as JSON text."""

        return json.dumps(
            self.to_dict(
                include_strings=include_strings,
                include_raw_bytes=include_raw_bytes,
                include_sequence_rows_raw=include_sequence_rows_raw,
            ),
            indent=indent,
        )

    def to_bytes(self, *, rebuild: bool = False) -> bytes:
        """Return TPS bytes for this object.

        The current structured state is always serialized. The ``rebuild`` flag
        is kept for backward compatibility.
        """

        return _build_tps_bytes(self)

    def write_tps(self, path: str | Path, *, rebuild: bool = False) -> Path:
        """Write the TPS payload to disk."""

        path = Path(path).expanduser().resolve()
        path.write_bytes(self.to_bytes(rebuild=rebuild))
        return path

    def write_json(
        self,
        path: str | Path,
        *,
        include_strings: bool = False,
        include_raw_bytes: bool = False,
        include_sequence_rows_raw: bool = False,
        indent: int = 2,
    ) -> Path:
        """Write the JSON dump to disk."""

        path = Path(path).expanduser().resolve()
        path.write_text(
            self.to_json(
                include_strings=include_strings,
                include_raw_bytes=include_raw_bytes,
                include_sequence_rows_raw=include_sequence_rows_raw,
                indent=indent,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Sequence:
        """Build a ``Sequence`` object from a JSON-like dictionary."""

        if _is_structured_tps_json(payload):
            return _from_structured_json_payload(payload)

        return _from_legacy_json_payload(payload)

    @classmethod
    def from_json(cls, text: str) -> Sequence:
        """Build a ``Sequence`` object from JSON text."""

        return cls.from_dict(json.loads(text))

    @classmethod
    def from_tps_bytes(cls, data: bytes, *, source_path: str | None = None) -> Sequence:
        """Parse raw TPS bytes into a ``Sequence``."""

        return parse_tps_bytes(data, source_path=source_path)

    @classmethod
    def load(cls, path: str | Path) -> Sequence:
        """Load a TPS file from disk into a ``Sequence``."""

        return load_tps(path)

    @classmethod
    def load_json_file(cls, path: str | Path) -> Sequence:
        """Load a JSON sequence dump from disk into a ``Sequence``."""

        return load_tps_json(path)

    def _current_offsets(self) -> dict[str, int | None]:
        """Return current section offsets for the serialized TPS payload."""

        return _calculate_offsets(self)

    def parameter_values(self) -> dict[str, str]:
        """Return sequence parameter values keyed by exact TNMR parameter name."""

        return {parameter["name"]: parameter["value"] for parameter in self.parameters}

    def set_parameter_value(self, name: str, value: object) -> "Sequence":
        """Update one existing sequence parameter by exact name.

        Python values are converted to compact strings before assignment:
        floats use ``.15g`` formatting and booleans become ``"1"`` or
        ``"0"``. Missing parameter names raise :class:`SequenceParameterError`
        instead of creating a new parameter.
        """

        new_value = _format_parameter_value(value)
        for parameter in self.parameters:
            if parameter["name"] == name:
                old_value = parameter["value"]
                parameter["value"] = new_value
                if "current_value" in parameter:
                    parameter["current_value"] = new_value
                self._record_edit(
                    {
                        "kind": "parameter",
                        "name": name,
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                )
                return self
        raise SequenceParameterError(f"Sequence parameter {name!r} was not found.")

    def set_parameter_values(self, values: Mapping[str, object]) -> "Sequence":
        """Update multiple existing parameter values atomically by name.

        All names are checked before any write occurs. If one name is missing,
        no parameter values are changed.
        """

        existing = self.parameter_values()
        missing = [name for name in values if name not in existing]
        if missing:
            joined = ", ".join(repr(name) for name in missing)
            raise SequenceParameterError(f"Sequence parameter(s) not found: {joined}.")
        for name, value in values.items():
            self.set_parameter_value(name, value)
        return self

    def table_references(self) -> list[SequenceTableReference]:
        """Return named table references from visible TNMR sequence cells.

        Event numbers in the returned objects are one-based visible TNMR event
        numbers, not zero-based Python indexes.
        """

        references: list[SequenceTableReference] = []
        for row in self.table_view().rows:
            quantity = _infer_table_quantity(row.label)
            for cell in row.cells:
                for table_name in cell.table_references:
                    references.append(
                        SequenceTableReference(
                            line=row.label,
                            event_number=cell.event_index + 1,
                            table_name=table_name,
                            data=cell.data,
                            quantity=quantity,
                        )
                    )
        return references

    def remove_unused_tables(self) -> dict[str, list[str]]:
        """Remove table definitions that are not referenced by sequence events."""

        used_names = {
            reference.table_name
            for reference in self.table_references()
            if reference.table_name
        }
        before_names = [table["name"] for table in self.tables if table.get("name")]
        self.tables = [
            table for table in self.tables if table.get("name") in used_names
        ]
        kept_names = [table["name"] for table in self.tables if table.get("name")]
        removed_names = [name for name in before_names if name not in set(kept_names)]
        if removed_names:
            self._record_edit(
                {
                    "kind": "remove_unused_tables",
                    "kept_tables": kept_names,
                    "removed_tables": removed_names,
                }
            )
        return {
            "used_tables": sorted(used_names),
            "kept_tables": kept_names,
            "removed_tables": removed_names,
        }

    def set_waveform_to_table(
        self,
        table_name: str,
        waveform: Waveform,
        *,
        quantity: Literal["phase", "amplitude"],
        precision: int = 3,
        phase_units: Literal["deg", "step"] = "deg",
        allow_shared: bool = False,
    ) -> "Sequence":
        """Write one waveform output column into a named TNMR table.

        This edits only the table definition. It does not assign the table to a
        sequence event; call :meth:`assign_table_to_event` for that separate step.
        The table must already exist so its TNMR metadata is preserved.
        """

        table_name = _normalize_table_name(table_name)
        quantity = _normalize_waveform_quantity(quantity)
        value_type: int | None = None
        if quantity == "phase":
            phase_units, value_type = _normalize_phase_units(phase_units)
            column = waveform.phase_column(precision=precision)
        else:
            column = waveform.amplitude_column(precision=precision)
        if column is None:
            raise SequenceWaveformError(
                f"Waveform {waveform.waveform_type!r} has no {quantity} output."
            )

        references = [
            reference
            for reference in self.table_references()
            if reference.table_name == table_name
        ]
        if len(references) > 1 and not allow_shared:
            raise SequenceWaveformError(
                f"Table {table_name!r} is referenced by {len(references)} "
                "events. Set allow_shared=True to update all shared uses."
            )

        entries = "\n".join(column)
        table = self._find_table(table_name)
        table["entries"] = entries
        if value_type is not None:
            table["value_type"] = value_type

        self._record_edit(
            {
                "kind": "set_waveform_to_table",
                "table_name": table_name,
                "quantity": quantity,
                "waveform": waveform.to_dict(),
                "point_count": len(column),
                "precision": precision,
                "phase_units": phase_units if quantity == "phase" else None,
                "allow_shared": allow_shared,
            }
        )
        return self

    def assign_table_to_event(
        self,
        line: str,
        event_number: int,
        table_name: str,
        *,
        slot_index: int = F1_PHASE_MOD_TABLE_SLOT,
    ) -> "Sequence":
        """Assign an existing TNMR table to one visible sequence event."""

        line = str(line).strip()
        if not line:
            raise SequenceWaveformError("line must not be empty.")
        try:
            event_number = int(event_number)
        except (TypeError, ValueError) as exc:
            raise SequenceWaveformError("event_number must be an integer.") from exc
        if event_number <= 0:
            raise SequenceWaveformError("event_number must be one-based and positive.")
        try:
            slot_index = int(slot_index)
        except (TypeError, ValueError) as exc:
            raise SequenceWaveformError("slot_index must be an integer.") from exc
        if slot_index < 0 or slot_index >= EVENT_TABLE_SLOT_COUNT:
            raise SequenceWaveformError(
                f"slot_index must be between 0 and {EVENT_TABLE_SLOT_COUNT - 1}."
            )

        table_name = _normalize_table_name(table_name)
        self._find_table(table_name)
        event, row_upgraded = self._resolve_visible_event(
            line,
            event_number,
            upgrade_phase_mod=(
                line == F1_PHASE_MOD_LINE and slot_index == F1_PHASE_MOD_TABLE_SLOT
            ),
        )
        _assign_event_table(event, slot_index, table_name)
        self._record_edit(
            {
                "kind": "assign_table_to_event",
                "line": line,
                "event_number": event_number,
                "table_name": table_name,
                "slot_index": slot_index,
                "phase_mod_row_upgraded": row_upgraded,
            }
        )
        return self

    def _resolve_visible_event(
        self,
        line: str,
        event_number: int,
        *,
        upgrade_phase_mod: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Return one mutable visible event dictionary."""

        for row in self.rows:
            if row["label"] != line:
                continue
            events = row["events"]
            event_index = event_number - 1
            if events and events[0]["data"] == row["label"]:
                event_index += 1
            if event_index < 0 or event_index >= len(events):
                break
            row_upgraded = (
                _ensure_phase_mod_waveform_row(row) if upgrade_phase_mod else False
            )
            event = _normalize_event(events[event_index])
            events[event_index] = event
            return event, row_upgraded
        raise SequenceWaveformError(
            f"No visible {line!r} event {event_number} was found."
        )

    def _find_table(self, table_name: str) -> dict[str, Any]:
        """Return an existing table by exact name."""

        for table in self.tables:
            if table["name"] == table_name:
                return table
        raise SequenceWaveformError(f"Sequence table {table_name!r} was not found.")

    def _record_edit(self, edit: dict[str, Any]) -> None:
        """Append JSON-only edit provenance."""

        edits = self.analysis.setdefault("edits", [])
        if not isinstance(edits, list):
            edits = []
            self.analysis["edits"] = edits
        edits.append(copy.deepcopy(edit))

    def _sync_derived_fields(self) -> None:
        """Refresh fields that are derived from the structured row state."""

        if self.stored_file_name is None:
            self.stored_file_name = _stored_file_name(self.stored_sequence_path)
        if self.rows:
            self.row_count = self.current_row_count
            self.column_count = self.current_column_count
            self.control_lines = self.current_control_lines
            self.parameter_references = self.current_parameter_references

    def summary(self) -> SequenceSummary:
        """Return compact current-state metadata."""

        return SequenceSummary(
            source_path=self.source_path,
            stored_sequence_path=self.stored_sequence_path,
            signature=self.signature,
            version=self.version,
            container_kind=self.container_kind,
            row_count=self.current_row_count,
            column_count=self.current_column_count,
            event_count=self.event_count,
            control_lines=tuple(self.current_control_lines),
            parameter_references=tuple(self.current_parameter_references),
            table_count=len(self.tables),
            parameter_page_count=len(self.parameter_pages),
            parameter_count=len(self.parameters),
            raw_tail_chunk_count=sum(
                1 for chunk in self.tail_layout if chunk.get("kind") == "raw"
            ),
            comment_present=self.sequence_comment is not None,
            rows_parse_error=self.rows_parse_error,
        )

    def row_records(self) -> list[dict[str, Any]]:
        """Return GUI-friendly per-row records."""

        records: list[dict[str, Any]] = []
        for index, row in enumerate(self.rows):
            normalized = _normalize_row(row)
            records.append(
                {
                    "index": index,
                    "label": normalized["label"],
                    "default_text": normalized["default_text"],
                    "column_count": normalized["column_count"],
                    "event_count": len(normalized["events"]),
                    "address_raw": normalized["address_raw"],
                    "address_text": normalized.get("address_text")
                    or _decode_loose_code(normalized["address_raw"]),
                    "bit_length": normalized["bit_length"],
                    "icon_library_type_raw": normalized["icon_library_type_raw"],
                    "icon_library_type_text": normalized.get("icon_library_type_text")
                    or _decode_loose_code(normalized["icon_library_type_raw"]),
                    "visible_flag": normalized["visible_flag"],
                    "private_data": normalized["private_data"],
                    "group": normalized["group"],
                }
            )
        return records

    def event_records(self) -> list[dict[str, Any]]:
        """Return GUI-friendly flattened event rows."""

        records: list[dict[str, Any]] = []
        for row_index, row in enumerate(self.rows):
            normalized_row = _normalize_row(row)
            for event_index, event in enumerate(normalized_row["events"]):
                normalized_event = _normalize_event(event)
                table_references = [
                    slot["name"]
                    for slot in normalized_event["table_slots"]
                    if slot["name"]
                ]
                records.append(
                    {
                        "row_index": row_index,
                        "row_label": normalized_row["label"],
                        "event_index": event_index,
                        "data": normalized_event["data"],
                        "table_references": ", ".join(table_references),
                        "table_reference_count": len(table_references),
                        "extra_string_count": len(normalized_event["extra_strings"]),
                        "extra_strings": list(normalized_event["extra_strings"]),
                        "extra_raw_present": bool(
                            normalized_event.get("extra_raw_base64")
                        ),
                    }
                )
        return records

    def table_view(self) -> SequenceTableView:
        """Return a TNMR-editor-style row/event matrix."""

        table_rows: list[SequenceTableRow] = []
        for row_index, row in enumerate(self.rows):
            normalized_row = _normalize_row(row)
            visible_events = _visible_row_events(normalized_row)
            cells = tuple(
                SequenceTableCell(
                    event_index=event_index,
                    data=event["data"],
                    table_references=tuple(
                        slot["name"] for slot in event["table_slots"] if slot["name"]
                    ),
                    extra_strings=tuple(event["extra_strings"]),
                    extra_raw_present=bool(event.get("extra_raw_base64")),
                )
                for event_index, event in enumerate(visible_events)
            )
            table_rows.append(
                SequenceTableRow(
                    row_index=row_index,
                    section=_sequence_section(normalized_row["label"]),
                    label=normalized_row["label"],
                    default_text=normalized_row["default_text"],
                    group=normalized_row["group"],
                    visible_flag=normalized_row["visible_flag"],
                    cells=cells,
                )
            )

        return SequenceTableView(
            event_count=max((len(row.cells) for row in table_rows), default=0),
            rows=tuple(table_rows),
        )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _encode_base64(data: bytes) -> str:
    """Encode raw bytes as ASCII base64."""

    return base64.b64encode(data).decode("ascii")


def _decode_base64(value: Any) -> bytes:
    """Decode a base64 payload when present."""

    if not value:
        return b""
    return base64.b64decode(value)


def _is_structured_tps_json(payload: dict[str, Any]) -> bool:
    """Check whether a JSON payload uses the canonical TPS block layout."""

    return any(key in payload for key in STRUCTURED_JSON_KEYS)


def _tail_layout_to_json(layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tail-layout chunks into JSON-safe data."""

    payload: list[dict[str, Any]] = []
    for chunk in layout:
        if chunk["kind"] == "raw":
            payload.append(
                {
                    "kind": "raw",
                    "raw_base64": _encode_base64(chunk["raw_bytes"]),
                }
            )
        else:
            payload.append({"kind": chunk["kind"]})
    return payload


def _normalize_tail_layout(
    layout: Any,
    *,
    legacy_trailing_raw: bytes = b"",
) -> list[dict[str, Any]]:
    """Normalize JSON-like tail-layout data."""

    normalized: list[dict[str, Any]] = []
    for chunk in layout or []:
        kind = chunk.get("kind")
        if kind == "raw":
            raw_bytes = chunk.get("raw_bytes")
            if raw_bytes is None:
                raw_bytes = _decode_base64(chunk.get("raw_base64"))
            normalized.append(
                {
                    "kind": "raw",
                    "raw_bytes": bytes(raw_bytes),
                }
            )
        elif kind in {"tables", "parameter_pages", "parameters", "comment"}:
            normalized.append({"kind": kind})

    if legacy_trailing_raw:
        normalized.append({"kind": "raw", "raw_bytes": legacy_trailing_raw})
    return normalized


def _stored_file_name(stored_sequence_path: str | None) -> str | None:
    """Return the final path segment from the stored TPS sequence path."""

    if not stored_sequence_path:
        return None
    return re.split(r"[\\/]", stored_sequence_path)[-1]


def _format_parameter_value(value: object) -> str:
    """Return a compact TNMR parameter value string."""

    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.15g}"
    return str(value)


def _normalize_analysis(value: Any) -> dict[str, Any]:
    """Return a copied JSON provenance payload."""

    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _infer_table_quantity(line: str) -> str | None:
    """Infer whether a control-line table is phase or amplitude data."""

    normalized = line.strip()
    if (
        normalized.endswith("_Ph")
        or normalized.endswith("_PhMod")
        or normalized in {"RX_Phase", "Acq_phase"}
    ):
        return "phase"
    if normalized.endswith("_Ampl"):
        return "amplitude"
    return None


def _normalize_table_name(value: object) -> str:
    """Return a non-empty TNMR table name."""

    table_name = str(value).strip()
    if not table_name:
        raise SequenceWaveformError("table_name must not be empty.")
    return table_name


def _normalize_waveform_quantity(value: str) -> Literal["phase", "amplitude"]:
    """Return the waveform output column to write."""

    normalized = str(value).strip().lower()
    if normalized in {"phase", "ph"}:
        return "phase"
    if normalized in {"amplitude", "amp"}:
        return "amplitude"
    raise SequenceWaveformError("quantity must be 'phase' or 'amplitude'.")


def _normalize_phase_units(value: str) -> tuple[str, int]:
    """Return TNMR phase-table units and the matching table value type."""

    normalized = str(value).strip().lower()
    if normalized == "deg":
        return normalized, 0
    if normalized == "step":
        return normalized, 1
    raise SequenceWaveformError("phase_units must be 'deg' or 'step'.")


def _ensure_phase_mod_waveform_row(row: dict[str, Any]) -> bool:
    """Upgrade old F1 phase-modulation rows to TNMR's 3P waveform form."""

    changed = False
    if row.get("icon_library_type_raw") != F1_PHASE_MOD_TABLE_TYPE:
        row["icon_library_type_raw"] = F1_PHASE_MOD_TABLE_TYPE
        changed = True
    if int(row.get("bit_length", 0)) < F1_PHASE_MOD_BIT_LENGTH:
        row["bit_length"] = F1_PHASE_MOD_BIT_LENGTH
        changed = True
    address_raw = int(row.get("address_raw", 0))
    if address_raw and address_raw & 0xFF != F1_PHASE_MOD_ADDRESS_LOW_BYTE:
        row["address_raw"] = (address_raw & ~0xFF) | F1_PHASE_MOD_ADDRESS_LOW_BYTE
        changed = True
    return changed


def _assign_event_table(
    event: dict[str, Any],
    slot_index: int,
    table_name: str,
) -> None:
    """Assign a named table to a TNMR event table slot."""

    slots = _normalize_table_slots(event.get("table_slots"))
    slots[slot_index]["name"] = table_name
    slots[slot_index]["flag"] = 1
    event["table_slots"] = slots


def _infer_row_count(rows: list[dict[str, Any]], value: Any) -> int | None:
    """Prefer an explicit row count, then fall back to decoded row length."""

    if value is not None:
        return int(value)
    return len(rows) if rows else None


def _infer_column_count(rows: list[dict[str, Any]], value: Any) -> int | None:
    """Prefer an explicit column count, then fall back to the first decoded row."""

    if value is not None:
        return int(value)
    return rows[0]["column_count"] if rows else None


def _slice_rows_raw_bytes(
    raw_bytes: bytes,
    header_end_offset: int,
    sequence_rows_end_offset: int | None,
) -> bytes:
    """Extract the raw row section from a full TPS payload when possible."""

    if not raw_bytes:
        return b""
    if sequence_rows_end_offset is None:
        sequence_rows_end_offset = len(raw_bytes)
    return raw_bytes[header_end_offset:sequence_rows_end_offset]


def _offset_lookup(offsets: dict[str, Any]) -> dict[str, int | None]:
    """Normalize the small byte-offset map stored in JSON debug payloads."""

    return {
        "header_end": offsets.get("header_end", 0),
        "tables": offsets.get("tables"),
        "parameter_pages": offsets.get("parameter_pages"),
        "parameters": offsets.get("parameters"),
        "comment": offsets.get("comment"),
        "sequence_rows_end": offsets.get("sequence_rows_end"),
    }


def _derive_control_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Derive known control-line labels from parsed rows."""

    return _ordered_unique(
        [row["label"] for row in rows if row["label"] in KNOWN_CONTROL_LINES]
    )


def _derive_parameter_references(rows: list[dict[str, Any]]) -> list[str]:
    """Derive ``=parameter`` references from parsed row events."""

    return _ordered_unique(
        [
            event["data"][1:].strip()
            for row in rows
            for event in row.get("events", [])
            if event.get("data", "").startswith("=")
        ]
    )


def _sequence_section(label: str) -> str:
    """Infer the TNMR editor section name for one row label."""

    normalized = label.strip()
    if normalized in {"Ext_Trig", "Loop", "CBranch", "CTest", "RT_Update"}:
        return "CTRL"
    if normalized.endswith("_PhRst"):
        return "RST"
    if normalized.startswith("F1_") or normalized == "F1_H/L":
        return "F1"
    if normalized.startswith("F2_") or normalized == "F2_H/L":
        return "F2"
    if normalized.startswith("RX_") or normalized in {"Acq", "Acq_phase", "Scope_Trig"}:
        return "RX"
    if normalized.startswith("TTl_") or normalized.startswith("TTL_"):
        return "TTL"
    return "Sequence"


def _visible_row_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return row events that map to visible TNMR sequence columns."""

    normalized_row = _normalize_row(row)
    events = [_normalize_event(event) for event in normalized_row["events"]]
    if events and events[0]["data"] == normalized_row["label"]:
        return events[1:]
    return events


def _parameter_page_to_json(page: dict[str, Any]) -> dict[str, Any]:
    """Convert a parameter page into the canonical JSON structure."""

    normalized = _normalize_parameter_page(page)
    return {
        "name": normalized["name"],
        "parameter_name_refs": list(normalized["parameter_names"]),
    }


def _table_slot_to_json(slot: dict[str, Any]) -> dict[str, Any]:
    """Convert one table slot into canonical JSON."""

    return {
        "name": str(slot.get("name", "")),
        "flag": int(slot.get("flag", 0)),
    }


def _event_to_json(event: dict[str, Any]) -> dict[str, Any]:
    """Convert one sequence event into canonical JSON."""

    normalized = _normalize_event(event)
    payload = {
        "data": normalized["data"],
        "table_slots": [
            _table_slot_to_json(slot) for slot in normalized["table_slots"]
        ],
        "extra_strings": list(normalized["extra_strings"]),
    }
    if normalized["extra_raw_base64"] is not None:
        payload["extra_raw_base64"] = normalized["extra_raw_base64"]
    return payload


def _row_to_json(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one sequence row into canonical JSON."""

    normalized = _normalize_row(row)
    return {
        "column_count": normalized["column_count"],
        "address_raw": normalized["address_raw"],
        "bit_length": normalized["bit_length"],
        "icon_library_type_raw": normalized["icon_library_type_raw"],
        "visible_flag": normalized["visible_flag"],
        "private_data": normalized["private_data"],
        "group": normalized["group"],
        "default_text": normalized["default_text"],
        "label": normalized["label"],
        "events": [_event_to_json(event) for event in normalized["events"]],
    }


def _table_to_json(table: dict[str, Any]) -> dict[str, Any]:
    """Convert one table entry into canonical JSON."""

    normalized = _normalize_table(table)
    payload = {
        "name": normalized["name"],
        "entries": normalized["entries"],
        "increment_operation": normalized["increment_operation"],
        "increment_value": normalized["increment_value"],
        "increment_scheme": normalized["increment_scheme"],
        "repeat_time": normalized["repeat_time"],
        "table_type": normalized["table_type"],
        "dimension": normalized["dimension"],
        "steps_per_360_cycle": normalized["steps_per_360_cycle"],
        "use_as_increment_list": normalized["use_as_increment_list"],
        "value_type": normalized["value_type"],
        "unknown_0": normalized["unknown_0"],
    }
    if normalized["extra_trailing_u32"]:
        payload["extra_trailing_u32"] = list(normalized["extra_trailing_u32"])
    return payload


def _parameter_to_json(parameter: dict[str, Any]) -> dict[str, Any]:
    """Convert one parameter definition into canonical JSON."""

    normalized = _normalize_parameter(parameter)
    payload = {
        "name": normalized["name"],
        "value": normalized["value"],
        "parameter_type": normalized["parameter_type"],
        "minimum": normalized["minimum"],
        "maximum": normalized["maximum"],
        "read_only": normalized["read_only"],
    }
    if normalized.get("extended_format"):
        payload.update(
            {
                "extended_format": True,
                "unknown_0": normalized["unknown_0"],
                "unknown_1": list(normalized["unknown_1"]),
                "current_name": normalized["current_name"],
                "current_value": normalized["current_value"],
                "unknown_2": list(normalized["unknown_2"]),
            }
        )
    return payload


def _from_structured_json_payload(payload: dict[str, Any]) -> Sequence:
    """Build ``Sequence`` from the canonical TPS JSON representation."""

    header = payload.get("header", {})
    stored_sequence = payload.get("stored_sequence", {})
    sequence_rows = payload.get("sequence_rows", {})
    debug = payload.get("debug", {})
    analysis = _normalize_analysis(payload.get("analysis", {}))
    serialization = payload.get("serialization", {})
    offsets = _offset_lookup(
        debug.get("byte_offsets", serialization.get("byte_offsets", {}))
    )

    raw_bytes = _decode_base64(
        debug.get("raw_bytes_base64") or serialization.get("raw_bytes_base64")
    )
    rows_raw_bytes = _decode_base64(sequence_rows.get("raw_bytes_base64"))
    rows = [_normalize_row(row) for row in sequence_rows.get("rows", [])]
    tables, parameter_pages, parameters, sequence_comment, tail_layout = (
        _decode_tail_sections(payload.get("tail_sections", []))
    )

    row_count = _infer_row_count(rows, sequence_rows.get("row_count"))
    column_count = _infer_column_count(rows, sequence_rows.get("column_count"))

    if not rows_raw_bytes:
        rows_raw_bytes = _slice_rows_raw_bytes(
            raw_bytes,
            offsets["header_end"] or 0,
            offsets["sequence_rows_end"],
        )

    stored_sequence_path = stored_sequence.get("path")
    control_lines = list(analysis.get("control_lines", _derive_control_lines(rows)))
    parameter_references = list(
        analysis.get("parameter_references", _derive_parameter_references(rows))
    )

    return Sequence(
        source_path=payload.get("source_path"),
        raw_bytes=raw_bytes,
        signature=header.get("signature", ""),
        version=header.get("version"),
        container_kind=header.get("container_kind"),
        stored_sequence_path=stored_sequence_path,
        stored_file_name=_stored_file_name(stored_sequence_path),
        row_count=row_count,
        column_count=column_count,
        header_extra_bytes=_decode_base64(header.get("extra_bytes_base64")),
        header_end_offset=offsets["header_end"] or 0,
        rows_raw_bytes=rows_raw_bytes,
        rows=rows,
        rows_parse_error=debug.get("rows_parse_error")
        or sequence_rows.get("rows_parse_error"),
        tables_offset=offsets["tables"],
        parameter_pages_offset=offsets["parameter_pages"],
        parameters_offset=offsets["parameters"],
        comment_offset=offsets["comment"],
        tail_layout=tail_layout,
        control_lines=control_lines,
        parameter_references=parameter_references,
        tables=tables,
        parameter_pages=parameter_pages,
        parameters=parameters,
        sequence_comment=sequence_comment,
        analysis=analysis,
        strings=copy.deepcopy(
            debug.get("printable_strings", analysis.get("strings", []))
        ),
    )


def _from_legacy_json_payload(payload: dict[str, Any]) -> Sequence:
    """Build ``Sequence`` from the older flat JSON representation."""

    offsets = _offset_lookup(payload.get("offsets", {}))
    raw_bytes = _decode_base64(payload.get("raw_bytes_base64"))
    rows_raw_bytes = _decode_base64(payload.get("sequence_rows_raw_base64"))
    rows = [_normalize_row(row) for row in payload.get("rows", [])]
    tables = [_normalize_table(table) for table in payload.get("tables", [])]
    parameter_pages = [
        _normalize_parameter_page(page) for page in payload.get("parameter_pages", [])
    ]
    parameters = [
        _normalize_parameter(parameter) for parameter in payload.get("parameters", [])
    ]

    row_count = _infer_row_count(rows, payload.get("row_count"))
    column_count = _infer_column_count(rows, payload.get("column_count"))

    if not rows_raw_bytes:
        rows_raw_bytes = _slice_rows_raw_bytes(
            raw_bytes,
            payload.get("header_end_offset", offsets["header_end"] or 0),
            offsets["sequence_rows_end"],
        )

    stored_sequence_path = payload.get("stored_sequence_path")
    control_lines = list(payload.get("control_lines", _derive_control_lines(rows)))
    parameter_references = list(
        payload.get("parameter_references", _derive_parameter_references(rows))
    )
    analysis = _normalize_analysis(payload.get("analysis", {}))

    return Sequence(
        source_path=payload.get("source_path"),
        raw_bytes=raw_bytes,
        signature=payload.get("signature", ""),
        version=payload.get("version"),
        container_kind=payload.get("container_kind"),
        stored_sequence_path=stored_sequence_path,
        stored_file_name=payload.get("stored_file_name")
        or _stored_file_name(stored_sequence_path),
        row_count=row_count,
        column_count=column_count,
        header_extra_bytes=_decode_base64(payload.get("header_extra_bytes_base64")),
        header_end_offset=payload.get("header_end_offset", offsets["header_end"] or 0),
        rows_raw_bytes=rows_raw_bytes,
        rows=rows,
        rows_parse_error=payload.get("rows_parse_error"),
        tables_offset=offsets["tables"],
        parameter_pages_offset=offsets["parameter_pages"],
        parameters_offset=offsets["parameters"],
        comment_offset=offsets["comment"],
        tail_layout=_normalize_tail_layout(
            payload.get("tail_layout"),
            legacy_trailing_raw=_decode_base64(payload.get("trailing_raw_base64")),
        ),
        control_lines=control_lines,
        parameter_references=parameter_references,
        tables=tables,
        parameter_pages=parameter_pages,
        parameters=parameters,
        sequence_comment=payload.get("sequence_comment"),
        analysis=analysis,
        strings=copy.deepcopy(payload.get("strings", [])),
    )


def _decode_tail_sections(
    sections: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str | None,
    list[dict[str, Any]],
]:
    """Decode canonical tail sections into object fields."""

    tables: list[dict[str, Any]] = []
    parameter_pages: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    sequence_comment: str | None = None
    tail_layout: list[dict[str, Any]] = []

    for section in sections or []:
        kind = section.get("kind")
        if kind == "raw":
            tail_layout.append(
                {"kind": "raw", "raw_bytes": _decode_base64(section.get("raw_base64"))}
            )
        elif kind == "tables":
            tables = [_normalize_table(entry) for entry in section.get("entries", [])]
            tail_layout.append({"kind": "tables"})
        elif kind == "parameter_pages":
            parameter_pages = [
                _normalize_parameter_page(entry) for entry in section.get("entries", [])
            ]
            tail_layout.append({"kind": "parameter_pages"})
        elif kind == "parameters":
            parameters = [
                _normalize_parameter(entry) for entry in section.get("entries", [])
            ]
            tail_layout.append({"kind": "parameters"})
        elif kind == "comment":
            sequence_comment = section.get("text")
            tail_layout.append({"kind": "comment"})

    return tables, parameter_pages, parameters, sequence_comment, tail_layout


def _build_tail_section_bytes(tps: Sequence, kind: str) -> bytes:
    """Serialize one known tail section from the current TPS object."""

    if kind == "tables":
        return _build_tables(tps.tables)
    if kind == "parameter_pages":
        return _build_parameter_pages(tps.parameter_pages)
    if kind == "parameters":
        return _build_parameters(tps.parameters)
    if kind == "comment":
        return _build_comment_section(tps.sequence_comment)
    raise ValueError(f"Unsupported TPS tail section kind: {kind}")


def _effective_tail_layout(tps: Sequence) -> list[dict[str, Any]]:
    """Return tail chunks in the current serialized order."""

    layout = copy.deepcopy(tps.tail_layout)
    emitted = {chunk["kind"] for chunk in layout if chunk["kind"] != "raw"}
    for kind in TAIL_SECTION_ORDER:
        if kind in emitted:
            continue
        if _build_tail_section_bytes(tps, kind):
            layout.append({"kind": kind})
    return layout


def _tail_sections_to_json(tps: Sequence) -> list[dict[str, Any]]:
    """Convert TPS tail chunks into canonical JSON sections."""

    payload: list[dict[str, Any]] = []
    for chunk in _effective_tail_layout(tps):
        kind = chunk["kind"]
        if kind == "raw":
            payload.append(
                {"kind": "raw", "raw_base64": _encode_base64(chunk["raw_bytes"])}
            )
        elif kind == "tables":
            payload.append(
                {
                    "kind": "tables",
                    "entries": [_table_to_json(table) for table in tps.tables],
                }
            )
        elif kind == "parameter_pages":
            payload.append(
                {
                    "kind": "parameter_pages",
                    "entries": [
                        _parameter_page_to_json(page) for page in tps.parameter_pages
                    ],
                }
            )
        elif kind == "parameters":
            payload.append(
                {
                    "kind": "parameters",
                    "entries": [
                        _parameter_to_json(parameter) for parameter in tps.parameters
                    ],
                }
            )
        elif kind == "comment":
            payload.append({"kind": "comment", "text": tps.sequence_comment})
    return payload


# ---------------------------------------------------------------------------
# Binary readers and low-level helpers
# ---------------------------------------------------------------------------


def _read_u32(data: bytes, offset: int, *, end: int | None = None) -> int:
    """Read a little-endian ``uint32`` from the source bytes."""

    limit = len(data) if end is None else min(end, len(data))
    if offset + 4 > limit:
        raise _SectionParseError("Not enough bytes left for uint32.")
    return struct.unpack_from("<I", data, offset)[0]


def _read_i32(data: bytes, offset: int, *, end: int | None = None) -> int:
    """Read a little-endian ``int32`` from the source bytes."""

    limit = len(data) if end is None else min(end, len(data))
    if offset + 4 > limit:
        raise _SectionParseError("Not enough bytes left for int32.")
    return struct.unpack_from("<i", data, offset)[0]


def _pack_i32(value: int) -> bytes:
    """Pack a TNMR signed flag while preserving positive raw values."""

    if value < 0:
        return struct.pack("<i", value)
    return struct.pack("<I", value)


def _read_ascii(
    data: bytes, offset: int, length: int, *, allow_empty: bool = True
) -> tuple[str, int]:
    """Read a printable ASCII/latin1 string."""

    if length < 0 or offset + length > len(data):
        raise _SectionParseError("String length exceeds available bytes.")
    if length == 0:
        if allow_empty:
            return "", offset
        raise _SectionParseError("Unexpected empty string.")

    raw = data[offset : offset + length]
    if any(byte == 0 or byte < 32 or byte > 126 for byte in raw):
        raise _SectionParseError("Encountered non-printable string content.")
    return raw.decode("latin1"), offset + length


def _read_comment_text(data: bytes, offset: int, length: int) -> tuple[str, int]:
    """Read a comment string that may contain CR/LF line breaks."""

    return _read_multiline_ascii(data, offset, length, context="Comment")


def _read_multiline_ascii(
    data: bytes,
    offset: int,
    length: int,
    *,
    context: str = "Text",
) -> tuple[str, int]:
    """Read a Latin-1 text block that may contain tab/CR/LF line breaks."""

    if length < 0 or offset + length > len(data):
        raise _SectionParseError(f"{context} length exceeds available bytes.")

    raw = data[offset : offset + length]
    if any(
        byte == 0 or (byte < 32 and byte not in {9, 10, 13}) or byte > 126
        for byte in raw
    ):
        raise _SectionParseError(f"Encountered invalid {context.lower()} content.")
    return raw.decode("latin1"), offset + length


def _read_lp_text(
    data: bytes,
    offset: int,
    end: int,
    *,
    allow_empty: bool = True,
) -> tuple[str, int]:
    """Read a length-prefixed printable text string."""

    length = _read_u32(data, offset, end=end)
    offset += 4
    text, offset = _read_ascii(data, offset, length, allow_empty=allow_empty)
    if offset > end:
        raise _SectionParseError("Section overran its expected end.")
    return text, offset


def _read_lp_table_entries(data: bytes, offset: int, end: int) -> tuple[str, int]:
    """Read a length-prefixed TNMR table body, allowing row line breaks."""

    length = _read_u32(data, offset, end=end)
    offset += 4
    text, offset = _read_multiline_ascii(data, offset, length, context="Table entries")
    if offset > end:
        raise _SectionParseError("Section overran its expected end.")
    return text, offset


def _pack_text(text: str) -> bytes:
    """Pack a printable ASCII/latin1 string with a 32-bit length prefix."""

    encoded = text.encode("latin1")
    return struct.pack("<I", len(encoded)) + encoded


def _is_zero_padding(data: bytes, start: int, end: int) -> bool:
    """Return whether the remaining bytes are all zero padding."""

    return start <= end and all(byte == 0 for byte in data[start:end])


def _ordered_unique(values: list[str]) -> list[str]:
    """Return unique values while preserving order."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _extract_strings(data: bytes) -> list[dict[str, Any]]:
    """Extract printable strings from the TPS bytes for debugging and indexing."""

    return [
        {
            "index": index,
            "offset": match.start(),
            "text": match.group().decode("latin1"),
        }
        for index, match in enumerate(PRINTABLE_RE.finditer(data))
    ]


def _decode_type_code(value: int) -> str | None:
    """Decode a compact four-byte ASCII code when present."""

    raw = struct.pack("<I", value).rstrip(b"\x00")
    if raw and all(32 <= byte <= 126 for byte in raw):
        return raw.decode("latin1")
    return None


def _decode_loose_code(value: int) -> str | None:
    """Decode printable bytes from a compact integer code."""

    printable = bytes(byte for byte in struct.pack("<I", value) if 32 <= byte <= 126)
    return printable.decode("latin1") if printable else None


def _read_header_fields(data: bytes) -> dict[str, Any]:
    """Read the fixed TPS header and stored path fields."""

    if len(data) < HEADER_LENGTH + 12:
        raise ValueError("TPS data is too short to contain a valid header.")

    signature = data[:HEADER_LENGTH].decode("latin1")
    match = TPS_HEADER_RE.match(signature)
    if match is None:
        raise ValueError("TPS data does not start with a recognized PSEQ header.")

    path_len = _read_u32(data, HEADER_LENGTH)
    path_start = HEADER_LENGTH + 4
    path_end = path_start + path_len
    if path_end + 8 > len(data):
        raise ValueError("TPS stored path overran the available bytes.")

    stored_sequence_path, _ = _read_ascii(data, path_start, path_len)
    row_count_offset = _find_header_row_count_offset(data, path_end)
    row_count = _read_u32(data, row_count_offset)
    column_count = _read_u32(data, row_count_offset + 4)
    return {
        "signature": signature,
        "version": match.group("version"),
        "container_kind": match.group("kind"),
        "stored_sequence_path": stored_sequence_path,
        "stored_file_name": _stored_file_name(stored_sequence_path),
        "row_count": row_count,
        "column_count": column_count,
        "header_extra_bytes": data[path_end:row_count_offset],
        "header_end_offset": row_count_offset + 8,
    }


def _find_header_row_count_offset(data: bytes, path_end: int) -> int:
    """Find row-count and column-count fields after optional header metadata."""

    scan_end = min(len(data) - 8, path_end + 512)
    for offset in range(path_end, scan_end + 1):
        try:
            row_count = _read_u32(data, offset)
            column_count = _read_u32(data, offset + 4)
        except _SectionParseError:
            break
        if not (0 < row_count <= 4096 and 0 < column_count <= 4096):
            continue
        if _looks_like_row_start(data, offset + 8, len(data), column_count):
            return offset
    return path_end


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _parse_comment_section(data: bytes) -> tuple[int | None, str | None, bool]:
    """Locate the last valid ``SEQC`` comment block in the TPS payload.

    Some Tecmag files carry trailing raw bytes after the comment, so the parser
    searches backward and validates the block rather than assuming the comment
    must end at EOF.
    """

    stop = len(data)
    while True:
        offset = data.rfind(b"SEQC", 0, stop)
        if offset < 0:
            return None, None, False
        try:
            length = _read_u32(data, offset + 4)
            comment, end = _read_comment_text(data, offset + 8, length)
            return offset, comment, end == len(data)
        except _SectionParseError:
            stop = offset


def _try_parse_tables(data: bytes, start: int, end: int) -> list[dict[str, Any]] | None:
    """Try to parse a TPS table block at a specific offset."""

    for trailing_count in (
        TABLE_TRAILING_U32_COUNT,
        EXTENDED_TABLE_TRAILING_U32_COUNT,
    ):
        try:
            count = _read_u32(data, start, end=end)
            if count <= 0 or count > 4096:
                return None

            offset = start + 4
            tables: list[dict[str, Any]] = []
            for _ in range(count):
                name, offset = _read_lp_text(data, offset, end, allow_empty=False)
                entries, offset = _read_lp_table_entries(data, offset, end)
                increment_operation, offset = _read_lp_text(data, offset, end)
                increment_value, offset = _read_lp_text(data, offset, end)
                increment_scheme, offset = _read_lp_text(data, offset, end)

                trailing = [
                    _read_u32(data, offset + 4 * index, end=end)
                    for index in range(trailing_count)
                ]
                offset += 4 * trailing_count

                table = {
                    "name": name,
                    "entries": entries,
                    "increment_operation": increment_operation,
                    "increment_value": increment_value,
                    "increment_scheme": increment_scheme,
                    "repeat_time": trailing[0],
                    "table_type": trailing[1],
                    "table_type_text": _decode_type_code(trailing[1]),
                    "dimension": trailing[2],
                    "steps_per_360_cycle": trailing[3],
                    "use_as_increment_list": trailing[4],
                    "value_type": trailing[5],
                    "unknown_0": trailing[6],
                }
                if trailing_count > TABLE_TRAILING_U32_COUNT:
                    table["extra_trailing_u32"] = trailing[TABLE_TRAILING_U32_COUNT:]
                tables.append(table)

            if offset == end or _is_zero_padding(data, offset, end):
                return tables
        except _SectionParseError:
            continue
    return None


def _try_parse_parameter_pages(
    data: bytes,
    start: int,
    end: int,
) -> list[dict[str, Any]] | None:
    """Try to parse a TPS parameter-page block at a specific offset."""

    try:
        count = _read_u32(data, start, end=end)
        if count <= 0 or count > 512:
            return None

        offset = start + 4
        pages: list[dict[str, Any]] = []
        for _ in range(count):
            name, offset = _read_lp_text(data, offset, end, allow_empty=False)
            item_count = _read_u32(data, offset, end=end)
            if item_count > 8192:
                raise _SectionParseError("Unreasonable parameter page size.")
            offset += 4

            parameter_names: list[str] = []
            for _ in range(item_count):
                parameter_name, offset = _read_lp_text(
                    data, offset, end, allow_empty=False
                )
                parameter_names.append(parameter_name)

            pages.append(
                {
                    "name": name,
                    "parameter_names": parameter_names,
                }
            )

        return pages if offset == end else None
    except _SectionParseError:
        return None


def _try_parse_parameters(
    data: bytes, start: int, end: int
) -> list[dict[str, Any]] | None:
    """Try to parse a TPS parameter-definition block at a specific offset."""

    legacy = _try_parse_legacy_parameters(data, start, end)
    if legacy is not None:
        return legacy
    return _try_parse_extended_parameters(data, start, end)


def _try_parse_legacy_parameters(
    data: bytes, start: int, end: int
) -> list[dict[str, Any]] | None:
    """Try to parse the compact TPS parameter-definition block."""

    try:
        count = _read_u32(data, start, end=end)
        if count <= 0 or count > 8192:
            return None

        offset = start + 4
        parameters: list[dict[str, Any]] = []
        for _ in range(count):
            name, offset = _read_lp_text(data, offset, end, allow_empty=False)
            value, offset = _read_lp_text(data, offset, end)
            parameter_type = _read_u32(data, offset, end=end)
            offset += 4
            minimum, offset = _read_lp_text(data, offset, end)
            maximum, offset = _read_lp_text(data, offset, end)
            read_only = _read_u32(data, offset, end=end)
            offset += 4

            parameters.append(
                {
                    "name": name,
                    "value": value,
                    "parameter_type": parameter_type,
                    "minimum": minimum,
                    "maximum": maximum,
                    "read_only": read_only,
                }
            )

        return parameters if offset == end else None
    except _SectionParseError:
        return None


def _try_parse_extended_parameters(
    data: bytes, start: int, end: int
) -> list[dict[str, Any]] | None:
    """Try to parse the extended parameter records used by newer TPS files."""

    try:
        count = _read_u32(data, start, end=end)
        if count <= 0 or count > 8192:
            return None

        offset = start + 4
        parameters: list[dict[str, Any]] = []
        for _ in range(count):
            name, offset = _read_lp_text(data, offset, end, allow_empty=False)
            unknown_0 = _read_u32(data, offset, end=end)
            offset += 4
            value, offset = _read_lp_text(data, offset, end)
            parameter_type = _read_u32(data, offset, end=end)
            offset += 4
            minimum, offset = _read_lp_text(data, offset, end)
            maximum, offset = _read_lp_text(data, offset, end)
            unknown_1 = [
                _read_u32(data, offset + 4 * index, end=end) for index in range(3)
            ]
            offset += 12
            current_name, offset = _read_lp_text(data, offset, end, allow_empty=False)
            current_value, offset = _read_lp_text(data, offset, end)
            unknown_2 = [
                _read_u32(data, offset + 4 * index, end=end) for index in range(3)
            ]
            offset += 12
            read_only = _read_i32(data, offset, end=end)
            offset += 4

            parameters.append(
                {
                    "name": name,
                    "value": value,
                    "parameter_type": parameter_type,
                    "minimum": minimum,
                    "maximum": maximum,
                    "read_only": read_only,
                    "extended_format": True,
                    "unknown_0": unknown_0,
                    "unknown_1": unknown_1,
                    "current_name": current_name,
                    "current_value": current_value,
                    "unknown_2": unknown_2,
                }
            )

        return parameters if offset == end else None
    except _SectionParseError:
        return None


def _find_section_start(
    data: bytes,
    floor: int,
    end: int,
    parser: Callable[[bytes, int, int], Any],
) -> tuple[int | None, Any]:
    """Find the earliest offset where a section parser cleanly reaches ``end``."""

    best_offset: int | None = None
    best_payload: Any = None
    for start in range(end - 4, floor - 1, -1):
        payload = parser(data, start, end)
        if payload is not None:
            best_offset = start
            best_payload = payload
    return best_offset, best_payload


def _looks_like_row_start(data: bytes, start: int, end: int, column_count: int) -> bool:
    """Check whether an offset looks like the start of a row header."""

    try:
        fields = struct.unpack_from("<8I", data, start)
        if fields[0] != column_count or fields[7] > 64:
            return False

        offset = start + 32
        _, offset = _read_ascii(data, offset, fields[7])
        label_length = _read_u32(data, offset, end=end)
        if label_length <= 0 or label_length > 64:
            return False

        label, event0_offset = _read_ascii(
            data, offset + 4, label_length, allow_empty=False
        )
        event0_length = _read_u32(data, event0_offset, end=end)
        if event0_length != label_length:
            return False

        event0_text, _ = _read_ascii(
            data, event0_offset + 4, event0_length, allow_empty=False
        )
        return event0_text == label
    except (struct.error, _SectionParseError):
        return False


def _find_row_starts(
    data: bytes,
    start: int,
    end: int,
    row_count: int,
    column_count: int,
) -> list[int]:
    """Locate all sequence-row headers inside the row/event section."""

    row_starts = [
        offset
        for offset in range(start, end - 48)
        if _looks_like_row_start(data, offset, end, column_count)
    ]
    if len(row_starts) != row_count:
        raise _SectionParseError(
            f"Expected {row_count} row headers, but found {len(row_starts)}."
        )
    return row_starts


def _find_scanned_last_row_end(data: bytes, row_start: int, end: int) -> int | None:
    """Find the shortest clean end offset for a scanned final row."""

    candidates = _last_row_end_candidates(data, row_start, end)
    base_end = _last_row_sequential_base_end(data, row_start, end)
    if base_end is not None:
        candidates = [base_end] + [
            candidate for candidate in candidates if candidate != base_end
        ]

    for candidate_end in candidates:
        try:
            rows = _parse_rows(data, [row_start], candidate_end)
        except _SectionParseError:
            continue
        if not rows:
            continue
        events = rows[0].get("events", [])
        if events and events[-1].get("extra_raw_base64") is None:
            return candidate_end
    return None


def _last_row_end_candidates(data: bytes, row_start: int, end: int) -> list[int]:
    """Return plausible final-row end offsets without a byte-by-byte scan."""

    minimum = row_start + 48
    candidates = {end}
    for match in PRINTABLE_RE.finditer(data[minimum:end]):
        text_start = minimum + match.start()
        for candidate in (text_start - 8, text_start - 4):
            if minimum <= candidate <= end:
                candidates.add(candidate)
    return sorted(candidates)


def _last_row_sequential_base_end(
    data: bytes,
    row_start: int,
    end: int,
) -> int | None:
    """Return the end of the last row when its events parse sequentially."""

    try:
        fields = struct.unpack_from("<8I", data, row_start)
        offset = row_start + 32
        _, offset = _read_ascii(data, offset, fields[7])
        _, offset = _read_lp_text(data, offset, end, allow_empty=False)
        for _ in range(fields[0]):
            _, offset = _parse_event_base(data, offset, end)
        return offset if offset <= end else None
    except (struct.error, _SectionParseError):
        return None


def _scan_row_layout(
    data: bytes,
    start: int,
    end: int,
) -> tuple[list[dict[str, Any]], int, int, int] | None:
    """Recover a row grid when v1.18 header counts are stored indirectly."""

    row_starts_by_column: dict[int, list[int]] = {}
    for offset in range(start, end - 48):
        column_count = struct.unpack_from("<I", data, offset)[0]
        if column_count < 1 or column_count > 64:
            continue
        if _looks_like_row_start(data, offset, end, column_count):
            row_starts_by_column.setdefault(column_count, []).append(offset)

    best: tuple[list[dict[str, Any]], int, int, int] | None = None
    best_score: tuple[int, int] = (0, 0)
    groups = sorted(
        row_starts_by_column.items(),
        key=lambda item: (-len(item[1]), item[1][0]),
    )
    for column_count, row_starts in groups:
        if not row_starts:
            continue
        if len(row_starts) <= best_score[0]:
            continue

        row_end = _find_scanned_last_row_end(data, row_starts[-1], end)
        if row_end is None:
            continue
        try:
            rows = _parse_rows(data, row_starts, row_end)
        except _SectionParseError:
            continue

        score = (len(rows), -row_starts[0])
        if score > best_score:
            best = (rows, row_starts[0], row_end, column_count)
            best_score = score
    return best


def _scan_printable_text_starts(data: bytes, start: int, end: int) -> list[int]:
    """Find plausible length-prefixed text starts inside a row."""

    starts = [start]
    for offset in range(start + 1, end - 4):
        try:
            length = _read_u32(data, offset, end=end)
        except _SectionParseError:
            continue
        if length <= 0 or length > 64 or offset + 4 + length > end:
            continue
        raw = data[offset + 4 : offset + 4 + length]
        if all(32 <= byte <= 126 for byte in raw):
            starts.append(offset)
    return sorted(set(starts))


def _parse_event_base(data: bytes, start: int, end: int) -> tuple[dict[str, Any], int]:
    """Parse the repeated base payload for one event cell.

    In every successfully decoded TPS example so far, an event stores one text
    value followed by ``EVENT_TABLE_SLOT_COUNT`` repeated ``(name, flag)``
    pairs. The parser treats that repeated structure as data, not as named
    semantics.
    """

    offset = start
    event_text, offset = _read_lp_text(data, offset, end)
    table_slots: list[dict[str, Any]] = []
    for slot in range(EVENT_TABLE_SLOT_COUNT):
        name, offset = _read_lp_text(data, offset, end)
        flag = _read_u32(data, offset, end=end)
        offset += 4
        table_slots.append({"slot": slot, "name": name, "flag": flag})

    return {"data": event_text, "table_slots": table_slots}, offset


def _parse_extra_payload(data: bytes, start: int, end: int) -> dict[str, Any] | None:
    """Parse any trailing payload after a base event block.

    This is usually zero or more printable strings. Some live TNMR rows also
    carry a short raw trailer after those strings, so we preserve that too.
    """

    offset = start
    strings: list[str] = []
    while offset < end:
        try:
            text, next_offset = _read_lp_text(data, offset, end, allow_empty=False)
        except _SectionParseError:
            break
        strings.append(text)
        offset = next_offset

    return {
        "extra_strings": strings,
        "extra_raw_base64": _encode_base64(data[offset:end]) if offset < end else None,
    }


def _finalize_event(
    event: dict[str, Any],
    *,
    extra_payload: dict[str, Any],
) -> dict[str, Any]:
    """Attach optional trailing payload to a parsed event."""

    finalized = copy.deepcopy(event)
    finalized["extra_strings"] = extra_payload["extra_strings"]
    finalized["extra_raw_base64"] = extra_payload["extra_raw_base64"]
    return finalized


def _parse_row_events_sequential(
    data: bytes,
    event0_start: int,
    row_end: int,
    event_count: int,
) -> list[dict[str, Any]]:
    """Parse row events assuming the row contains no embedded extra payload."""

    events: list[dict[str, Any]] = []
    offset = event0_start
    for _ in range(event_count):
        event, offset = _parse_event_base(data, offset, row_end)
        events.append(
            _finalize_event(
                event,
                extra_payload={"extra_strings": [], "extra_raw_base64": None},
            )
        )

    if offset != row_end:
        raise _SectionParseError(
            "Row contains embedded payload between base event blocks."
        )
    return events


def _parse_row_events_search(
    data: bytes,
    event0_start: int,
    row_end: int,
    event_count: int,
) -> list[dict[str, Any]]:
    """Parse row events with a search fallback for ambiguous embedded payloads.

    The common case is simple sequential parsing. This slower path exists so
    we can still preserve rows that carry embedded payload between event bases.
    """

    candidate_starts = _scan_printable_text_starts(data, event0_start, row_end)
    if not candidate_starts or candidate_starts[0] != event0_start:
        raise _SectionParseError("Could not identify the first event in the row.")

    @lru_cache(maxsize=None)
    def solve(start: int, remaining: int) -> tuple[dict[str, Any], ...] | None:
        try:
            event, base_end = _parse_event_base(data, start, row_end)
        except _SectionParseError:
            return None

        if remaining == 1:
            extra_payload = _parse_extra_payload(data, base_end, row_end)
            if extra_payload is None:
                return None
            return (
                _finalize_event(
                    event,
                    extra_payload=extra_payload,
                ),
            )

        next_starts = [base_end]
        next_starts.extend(
            candidate for candidate in candidate_starts if candidate > base_end
        )
        for next_start in next_starts:
            if next_start < base_end:
                continue
            extra_payload = _parse_extra_payload(data, base_end, next_start)
            if extra_payload is None:
                continue
            remainder = solve(next_start, remaining - 1)
            if remainder is None:
                continue

            return (
                _finalize_event(
                    event,
                    extra_payload=extra_payload,
                ),
            ) + remainder

        return None

    parsed_events = solve(event0_start, event_count)
    if parsed_events is None:
        raise _SectionParseError("Could not segment the row into sequence events.")
    return [copy.deepcopy(event) for event in parsed_events]


def _parse_row_events(
    data: bytes,
    event0_start: int,
    row_end: int,
    event_count: int,
) -> list[dict[str, Any]]:
    """Parse one row of sequence events.

    Most TNMR rows are simple sequential event blocks. A smaller subset embeds
    extra payload between base events, so we keep a search fallback for those
    ambiguous layouts.
    """

    try:
        return _parse_row_events_sequential(data, event0_start, row_end, event_count)
    except _SectionParseError:
        return _parse_row_events_search(data, event0_start, row_end, event_count)


def _parse_rows(
    data: bytes,
    row_starts: list[int],
    row_end: int,
) -> list[dict[str, Any]]:
    """Parse all sequence rows between the fixed header and tail sections."""

    rows: list[dict[str, Any]] = []
    for index, start in enumerate(row_starts):
        end = row_starts[index + 1] if index + 1 < len(row_starts) else row_end
        fields = struct.unpack_from("<8I", data, start)

        offset = start + 32
        default_text, offset = _read_ascii(data, offset, fields[7])
        label, event0_start = _read_lp_text(data, offset, end, allow_empty=False)
        events = _parse_row_events(data, event0_start, end, fields[0])

        # The compact row header is still only partly understood, so we keep
        # the raw integer fields for faithful round-tripping and a couple of
        # loose text decodes for debugging and local inspection.
        rows.append(
            {
                "index": index,
                "offset": start,
                "end_offset": end,
                "column_count": fields[0],
                "address_raw": fields[1],
                "address_text": _decode_loose_code(fields[1]),
                "bit_length": fields[2],
                "icon_library_type_raw": fields[3],
                "icon_library_type_text": _decode_loose_code(fields[3]),
                "visible_flag": fields[4],
                "private_data": fields[5],
                "group": fields[6],
                "default_text": default_text,
                "label": label,
                "events": events,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


def _normalize_table_slots(slots: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize event table-slot dictionaries."""

    slots = copy.deepcopy(slots or [])
    by_slot: dict[int, dict[str, Any]] = {}
    for index, slot in enumerate(slots):
        slot_index = int(slot.get("slot", index))
        by_slot[slot_index] = {
            "slot": slot_index,
            "name": str(slot.get("name", "")),
            "flag": int(slot.get("flag", 0)),
        }

    normalized: list[dict[str, Any]] = []
    for slot_index in range(EVENT_TABLE_SLOT_COUNT):
        normalized.append(
            by_slot.get(
                slot_index,
                {"slot": slot_index, "name": "", "flag": 0},
            )
        )
    return normalized


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize a sequence-event dictionary."""

    normalized = copy.deepcopy(event)
    normalized["data"] = str(normalized.get("data", ""))
    normalized["table_slots"] = _normalize_table_slots(normalized.get("table_slots"))
    normalized["extra_strings"] = [
        str(value) for value in normalized.get("extra_strings", [])
    ]
    normalized["extra_raw_base64"] = normalized.get("extra_raw_base64")
    return normalized


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a sequence-row dictionary."""

    normalized = copy.deepcopy(row)
    normalized["column_count"] = int(normalized.get("column_count", 0))
    normalized["address_raw"] = int(normalized.get("address_raw", 0))
    normalized["address_text"] = normalized.get("address_text")
    normalized["bit_length"] = int(normalized.get("bit_length", 0))
    normalized["icon_library_type_raw"] = int(
        normalized.get("icon_library_type_raw", 0)
    )
    normalized["icon_library_type_text"] = normalized.get("icon_library_type_text")
    normalized["visible_flag"] = int(normalized.get("visible_flag", 0))
    normalized["private_data"] = int(normalized.get("private_data", 0))
    normalized["group"] = int(normalized.get("group", 0))
    normalized["default_text"] = str(normalized.get("default_text", ""))
    normalized["label"] = str(normalized.get("label", ""))
    normalized["events"] = [
        _normalize_event(event) for event in normalized.get("events", [])
    ]
    return normalized


def _normalize_table(table: dict[str, Any]) -> dict[str, Any]:
    """Normalize a sequence-table dictionary."""

    normalized = copy.deepcopy(table)
    normalized["name"] = str(normalized.get("name", ""))
    normalized["entries"] = str(normalized.get("entries", ""))
    normalized["increment_operation"] = str(normalized.get("increment_operation", ""))
    normalized["increment_value"] = str(normalized.get("increment_value", ""))
    normalized["increment_scheme"] = str(normalized.get("increment_scheme", ""))
    normalized["repeat_time"] = int(normalized.get("repeat_time", 0))
    normalized["table_type"] = int(normalized.get("table_type", 0))
    normalized["table_type_text"] = normalized.get(
        "table_type_text"
    ) or _decode_type_code(normalized["table_type"])
    normalized["dimension"] = int(normalized.get("dimension", 0))
    normalized["steps_per_360_cycle"] = int(normalized.get("steps_per_360_cycle", 0))
    normalized["use_as_increment_list"] = int(
        normalized.get("use_as_increment_list", 0)
    )
    normalized["value_type"] = int(normalized.get("value_type", 0))
    normalized["unknown_0"] = int(normalized.get("unknown_0", 0))
    normalized["extra_trailing_u32"] = [
        int(value) for value in normalized.get("extra_trailing_u32", [])
    ]
    return normalized


def _normalize_parameter_page(page: dict[str, Any]) -> dict[str, Any]:
    """Normalize a parameter-page dictionary."""

    normalized = copy.deepcopy(page)
    normalized["name"] = str(normalized.get("name", ""))
    parameter_names = normalized.get("parameter_names")
    if parameter_names is None:
        parameter_names = normalized.get("parameter_name_refs", [])
    normalized["parameter_names"] = [
        str(parameter_name) for parameter_name in parameter_names
    ]
    return normalized


def _normalize_parameter(parameter: dict[str, Any]) -> dict[str, Any]:
    """Normalize a parameter-definition dictionary."""

    normalized = copy.deepcopy(parameter)
    normalized["name"] = str(normalized.get("name", ""))
    normalized["value"] = str(normalized.get("value", ""))
    normalized["parameter_type"] = int(normalized.get("parameter_type", 0))
    normalized["minimum"] = str(normalized.get("minimum", ""))
    normalized["maximum"] = str(normalized.get("maximum", ""))
    normalized["read_only"] = int(normalized.get("read_only", 0))
    if normalized.get("extended_format"):
        normalized["extended_format"] = True
        normalized["unknown_0"] = int(normalized.get("unknown_0", 0))
        normalized["unknown_1"] = [
            int(value) for value in normalized.get("unknown_1", [0, 0, 0])
        ][:3]
        while len(normalized["unknown_1"]) < 3:
            normalized["unknown_1"].append(0)
        normalized["current_name"] = str(
            normalized.get("current_name", normalized["name"])
        )
        normalized["current_value"] = str(
            normalized.get("current_value", normalized["value"])
        )
        normalized["unknown_2"] = [
            int(value) for value in normalized.get("unknown_2", [0, 0, 0])
        ][:3]
        while len(normalized["unknown_2"]) < 3:
            normalized["unknown_2"].append(0)
    return normalized


# ---------------------------------------------------------------------------
# Binary builders
# ---------------------------------------------------------------------------


def _build_signature(tps: Sequence) -> str:
    """Build the 16-byte TPS signature string."""

    if tps.signature:
        return tps.signature
    if tps.version and tps.container_kind:
        return f"PSEQ{tps.version} {tps.container_kind}"
    raise ValueError("TPS signature information is incomplete.")


def _build_row(row: dict[str, Any]) -> bytes:
    """Serialize one sequence row."""

    row = _normalize_row(row)
    default_bytes = row["default_text"].encode("latin1")
    chunks = [
        struct.pack(
            "<8I",
            row["column_count"],
            row["address_raw"],
            row["bit_length"],
            row["icon_library_type_raw"],
            row["visible_flag"],
            row["private_data"],
            row["group"],
            len(default_bytes),
        ),
        default_bytes,
        _pack_text(row["label"]),
    ]
    chunks.extend(_build_event(event) for event in row["events"])
    return b"".join(chunks)


def _build_event(event: dict[str, Any]) -> bytes:
    """Serialize one sequence event."""

    event = _normalize_event(event)
    chunks = [_pack_text(event["data"])]
    for slot in event["table_slots"]:
        chunks.append(_pack_text(slot["name"]))
        chunks.append(struct.pack("<I", slot["flag"]))
    chunks.extend(_pack_text(text) for text in event["extra_strings"])
    if event["extra_raw_base64"]:
        chunks.append(_decode_base64(event["extra_raw_base64"]))
    return b"".join(chunks)


def _build_tables(tables: list[dict[str, Any]]) -> bytes:
    """Serialize sequence tables."""

    if not tables:
        return b""

    chunks = [struct.pack("<I", len(tables))]
    for table in tables:
        normalized = _normalize_table(table)
        chunks.extend(
            [
                _pack_text(normalized["name"]),
                _pack_text(normalized["entries"]),
                _pack_text(normalized["increment_operation"]),
                _pack_text(normalized["increment_value"]),
                _pack_text(normalized["increment_scheme"]),
                struct.pack(
                    "<7I",
                    normalized["repeat_time"],
                    normalized["table_type"],
                    normalized["dimension"],
                    normalized["steps_per_360_cycle"],
                    normalized["use_as_increment_list"],
                    normalized["value_type"],
                    normalized["unknown_0"],
                ),
                (
                    struct.pack(
                        f"<{len(normalized['extra_trailing_u32'])}I",
                        *normalized["extra_trailing_u32"],
                    )
                    if normalized["extra_trailing_u32"]
                    else b""
                ),
            ]
        )
    return b"".join(chunks)


def _build_parameter_pages(parameter_pages: list[dict[str, Any]]) -> bytes:
    """Serialize sequence parameter pages."""

    if not parameter_pages:
        return b""

    chunks = [struct.pack("<I", len(parameter_pages))]
    for page in parameter_pages:
        normalized = _normalize_parameter_page(page)
        chunks.append(_pack_text(normalized["name"]))
        chunks.append(struct.pack("<I", len(normalized["parameter_names"])))
        chunks.extend(_pack_text(name) for name in normalized["parameter_names"])
    return b"".join(chunks)


def _build_parameters(parameters: list[dict[str, Any]]) -> bytes:
    """Serialize sequence parameters."""

    if not parameters:
        return b""

    chunks = [struct.pack("<I", len(parameters))]
    for parameter in parameters:
        normalized = _normalize_parameter(parameter)
        if normalized.get("extended_format"):
            chunks.extend(
                [
                    _pack_text(normalized["name"]),
                    struct.pack("<I", normalized["unknown_0"]),
                    _pack_text(normalized["value"]),
                    struct.pack("<I", normalized["parameter_type"]),
                    _pack_text(normalized["minimum"]),
                    _pack_text(normalized["maximum"]),
                    struct.pack("<3I", *normalized["unknown_1"]),
                    _pack_text(normalized["current_name"]),
                    _pack_text(normalized["current_value"]),
                    struct.pack("<3I", *normalized["unknown_2"]),
                    _pack_i32(normalized["read_only"]),
                ]
            )
        else:
            chunks.extend(
                [
                    _pack_text(normalized["name"]),
                    _pack_text(normalized["value"]),
                    struct.pack("<I", normalized["parameter_type"]),
                    _pack_text(normalized["minimum"]),
                    _pack_text(normalized["maximum"]),
                    _pack_i32(normalized["read_only"]),
                ]
            )
    return b"".join(chunks)


def _build_comment_section(sequence_comment: str | None) -> bytes:
    """Serialize the optional sequence comment."""

    if sequence_comment is None:
        return b""
    return b"SEQC" + _pack_text(sequence_comment)


def _build_tail_layout_from_source(
    data: bytes,
    row_section_end: int,
    *,
    tables_offset: int | None,
    tables: list[dict[str, Any]],
    parameter_pages_offset: int | None,
    parameter_pages: list[dict[str, Any]],
    parameters_offset: int | None,
    parameters: list[dict[str, Any]],
    comment_offset: int | None,
    sequence_comment: str | None,
) -> list[dict[str, Any]]:
    """Build tail-layout chunks from the original TPS bytes.

    Preserving the observed order of known sections and raw gaps is what keeps
    round-tripping lossless even while parts of the suffix remain opaque.
    """

    sections: list[tuple[int, int, str]] = []
    if tables_offset is not None:
        section_bytes = _build_tables(tables)
        sections.append((tables_offset, tables_offset + len(section_bytes), "tables"))
    if parameter_pages_offset is not None:
        section_bytes = _build_parameter_pages(parameter_pages)
        sections.append(
            (
                parameter_pages_offset,
                parameter_pages_offset + len(section_bytes),
                "parameter_pages",
            )
        )
    if parameters_offset is not None:
        section_bytes = _build_parameters(parameters)
        sections.append(
            (parameters_offset, parameters_offset + len(section_bytes), "parameters")
        )
    if comment_offset is not None and sequence_comment is not None:
        section_bytes = _build_comment_section(sequence_comment)
        sections.append(
            (comment_offset, comment_offset + len(section_bytes), "comment")
        )

    layout: list[dict[str, Any]] = []
    cursor = row_section_end
    for start, end, kind in sorted(sections):
        if cursor < start:
            raw = data[cursor:start]
            if raw:
                layout.append({"kind": "raw", "raw_bytes": raw})
        layout.append({"kind": kind})
        cursor = max(cursor, end)

    if cursor < len(data):
        raw = data[cursor:]
        if raw:
            layout.append({"kind": "raw", "raw_bytes": raw})

    return layout


def _build_tail_bytes(tps: Sequence) -> bytes:
    """Serialize tail sections in the stored layout order."""

    chunks: list[bytes] = []
    for chunk in _effective_tail_layout(tps):
        kind = chunk["kind"]
        if kind == "raw":
            chunks.append(chunk["raw_bytes"])
            continue
        chunks.append(_build_tail_section_bytes(tps, kind))

    return b"".join(chunks)


def _calculate_offsets(tps: Sequence) -> dict[str, int | None]:
    """Calculate current section offsets from the structured state."""

    offset = tps.header_end_offset + len(tps.sequence_rows_raw)
    offsets: dict[str, int | None] = {
        "header_end": tps.header_end_offset,
        "sequence_rows_end": offset,
        "tables": None,
        "parameter_pages": None,
        "parameters": None,
        "comment": None,
    }

    for chunk in _effective_tail_layout(tps):
        kind = chunk["kind"]
        if kind == "raw":
            offset += len(chunk["raw_bytes"])
            continue
        section_bytes = _build_tail_section_bytes(tps, kind)
        if section_bytes:
            offsets[kind] = offset
            offset += len(section_bytes)

    return offsets


def _build_tps_bytes(tps: Sequence) -> bytes:
    """Rebuild a TPS payload from the structured fields."""

    tps._sync_derived_fields()
    signature = _build_signature(tps)
    if len(signature.encode("latin1")) != HEADER_LENGTH:
        raise ValueError("TPS signatures must be exactly 16 bytes long.")

    stored_sequence_path = tps.stored_sequence_path or tps.stored_file_name or ""
    stored_path_bytes = stored_sequence_path.encode("latin1")
    rows_bytes = tps.sequence_rows_raw

    return b"".join(
        [
            signature.encode("latin1"),
            struct.pack("<I", len(stored_path_bytes)),
            stored_path_bytes,
            tps.header_extra_bytes,
            struct.pack(
                "<II",
                int(
                    tps.current_row_count
                    if tps.current_row_count is not None
                    else len(tps.rows)
                ),
                int(
                    tps.current_column_count
                    if tps.current_column_count is not None
                    else (tps.rows[0]["column_count"] if tps.rows else 0)
                ),
            ),
            rows_bytes,
            _build_tail_bytes(tps),
        ]
    )


# ---------------------------------------------------------------------------
# Public loading API
# ---------------------------------------------------------------------------


def parse_tps_bytes(data: bytes, *, source_path: str | None = None) -> Sequence:
    """Parse a TNMR ``.tps`` file into a lossless ``Sequence``.

    The parser decodes the fixed header, row grid, and known tail sections. If
    any suffix bytes are not yet understood, they are preserved as raw tail
    chunks so the file can still be round-tripped without loss.
    """

    header = _read_header_fields(data)
    strings = _extract_strings(data)

    comment_offset, sequence_comment, _comment_reaches_eof = _parse_comment_section(
        data
    )
    section_scan_end = comment_offset if comment_offset is not None else len(data)

    parameters_offset, parameters = _find_section_start(
        data,
        header["header_end_offset"],
        section_scan_end,
        _try_parse_parameters,
    )
    parameter_pages_end = (
        parameters_offset if parameters_offset is not None else section_scan_end
    )
    parameter_pages_offset, parameter_pages = _find_section_start(
        data,
        header["header_end_offset"],
        parameter_pages_end,
        _try_parse_parameter_pages,
    )
    tables_end = (
        parameter_pages_offset
        if parameter_pages_offset is not None
        else parameter_pages_end
    )
    tables_offset, tables = _find_section_start(
        data,
        header["header_end_offset"],
        tables_end,
        _try_parse_tables,
    )

    row_section_end = min(
        offset
        for offset in (
            tables_offset,
            parameter_pages_offset,
            parameters_offset,
            comment_offset,
            len(data),
        )
        if offset is not None
    )
    rows_raw_start = header["header_end_offset"]

    rows: list[dict[str, Any]] = []
    rows_parse_error: str | None = None
    try:
        row_starts = _find_row_starts(
            data,
            header["header_end_offset"],
            row_section_end,
            header["row_count"],
            header["column_count"],
        )
        rows = _parse_rows(data, row_starts, row_section_end)
    except _SectionParseError as exc:
        rows_parse_error = str(exc)
        scanned = _scan_row_layout(data, header["header_end_offset"], row_section_end)
        if scanned is not None:
            rows, rows_raw_start, row_section_end, column_count = scanned
            header["row_count"] = len(rows)
            header["column_count"] = column_count
            rows_parse_error = None

    rows_raw_bytes = data[rows_raw_start:row_section_end]

    control_lines = _derive_control_lines(rows)
    parameter_references = _derive_parameter_references(rows)

    tail_layout = _build_tail_layout_from_source(
        data,
        row_section_end,
        tables_offset=tables_offset,
        tables=tables or [],
        parameter_pages_offset=parameter_pages_offset,
        parameter_pages=parameter_pages or [],
        parameters_offset=parameters_offset,
        parameters=parameters or [],
        comment_offset=comment_offset,
        sequence_comment=sequence_comment,
    )

    return Sequence(
        source_path=source_path,
        raw_bytes=data,
        rows_raw_bytes=rows_raw_bytes,
        rows=rows,
        rows_parse_error=rows_parse_error,
        tail_layout=tail_layout,
        control_lines=control_lines,
        parameter_references=parameter_references,
        tables=tables or [],
        parameter_pages=parameter_pages or [],
        parameters=parameters or [],
        sequence_comment=sequence_comment,
        strings=strings,
        tables_offset=tables_offset,
        parameter_pages_offset=parameter_pages_offset,
        parameters_offset=parameters_offset,
        comment_offset=comment_offset,
        **header,
    )


def load_tps(path: str | Path) -> Sequence:
    """Load and parse a TNMR ``.tps`` file."""

    path = Path(path).expanduser().resolve()
    return parse_tps_bytes(path.read_bytes(), source_path=str(path))


def load_tps_json(path: str | Path) -> Sequence:
    """Load a TPS JSON dump."""

    path = Path(path).expanduser().resolve()
    return Sequence.from_json(path.read_text(encoding="utf-8"))
