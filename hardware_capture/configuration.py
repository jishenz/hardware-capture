"""Shared, file-relative configuration for offline and live workflows."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tomllib
from typing import Any

import numpy as np


FIELDS = {
    "spsa": {"directions", "seed", "amplitude_logit_perturbation", "phase_perturbation_rad", "perturbation_decay"},
    "composite": {
        "sequence_path",
        "parameter_source",
        "parameter_overrides",
        "seed",
        "alignment_search_s",
        "capture_reference_start_s",
        "capture_window",
        "capture_pre_margin_s",
        "capture_post_margin_s",
        "capture_window",
        "capture_pre_margin_s",
        "capture_post_margin_s",
        "middle_fraction",
        "minimum_coherence",
        "maximum_relative_spread",
        "amplitude_logit_learning_rate",
        "amplitude_logit_step",
    },
    "input": {"kind", "path", "read_current", "shots", "shot_command"},
    "output": {"directory", "show_plots"},
    "scope": {"resource", "channel", "window_start_s", "window_end_s", "timeout_ms", "trigger_timeout_s", "armed_timeout_s", "poll_s", "allow_clamp", "rearm"},
    "demodulation": {"frequency_hz", "cutoff_hz", "filter_order", "q_sign"},
    "window": {"start_s", "end_s", "threshold_fraction", "padding_s"},
    "pulse": {"kind", "duration_s", "samples", "angle_deg", "axis_phase_deg", "peak_rabi_hz", "max_rabi_hz", "parameters", "path", "amplitude_path", "phase_path", "amplitude_unit", "phase_unit"},
    "calibration": {"path", "omega_rad_s_per_volt", "rabi_at_100_hz", "middle_fraction", "smoothing_s", "threshold_fraction", "baseline_guard_s", "pi_over_2_duration_s"},
    "analysis": {"target", "scaling", "steps", "method", "amplitude_path", "phase_path", "amplitude_unit", "phase_unit", "phase_offset_rad"},
    "scan": {"amplitude_min", "amplitude_max", "amplitude_count", "detuning_min_hz", "detuning_max_hz", "detuning_count", "amplitude_errors", "detuning_hz"},
    "optimization": {"coefficients", "updates", "amplitude_learning_rate", "phase_learning_rate", "amplitude_step", "amplitude_logit_learning_rate", "amplitude_logit_step", "phase_step_rad", "phase_reference_captures", "use_configured_seed", "fidelity_threshold", "threshold_smoothing", "mean_weight", "threshold_weight", "cvar_weight", "cvar_confidence"},
    "simulation": {"amplitude_scale", "phase_ripple_rad"},
    "sequence": {"template", "input_path", "tnt_path", "parameter_updates", "table_updates", "assignments", "remove_unused", "json_input"},
    "tnmr": {"path", "amplitude_table", "phase_table", "launch", "action", "parameters", "tables", "save_path", "clear_tables", "sequence_path"},
}
PATH_KEYS = {"path", "input_path", "tnt_path", "amplitude_path", "phase_path", "save_path", "sequence_path"}


def merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def _resolve_paths(data: dict, directory: Path) -> dict:
    data = copy.deepcopy(data)
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if name in PATH_KEYS or (section == "output" and name == "directory"):
                if value is not None:
                    path = Path(value).expanduser()
                    values[name] = str((directory / path).resolve() if not path.is_absolute() else path.resolve())
    return data


class Experiment:
    """One reusable set of instrument, pulse, calibration, and analysis conditions.

    TOML/JSON files may extend other files. Relative paths belong to the file
    declaring them, even when a child profile overrides another setting.
    """

    def __init__(self, data: dict[str, Any], *, directory: Path | str | None = None):
        self.data = _resolve_paths(data, Path(directory or Path.cwd()))
        unknown = set(self.data) - set(FIELDS)
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        for section, values in self.data.items():
            if not isinstance(values, dict):
                raise ValueError(f"{section} must be a table/object")
            extra = set(values) - FIELDS[section]
            if extra:
                raise ValueError(f"Unknown {section} settings: {sorted(extra)}")
        self._validate_numbers(self.data)

    @classmethod
    def load(cls, path: Path | str, _stack: tuple[Path, ...] = ()) -> "Experiment":
        path = Path(path).expanduser().resolve()
        if path in _stack:
            raise ValueError(f"Configuration inheritance cycle: {path}")
        text = path.read_text(encoding="utf-8-sig")
        own = json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
        parents = own.pop("extends", [])
        if isinstance(parents, str):
            parents = [parents]
        data = {}
        for parent in parents:
            data = merge(data, cls.load(path.parent / parent, (*_stack, path)).data)
        return cls(merge(data, _resolve_paths(own, path.parent)), directory=path.parent)

    def section(self, name: str) -> dict:
        return copy.deepcopy(self.data.get(name, {}))

    def with_updates(self, **sections: dict) -> "Experiment":
        return Experiment(merge(self.data, sections))

    def require(self, section: str, key: str):
        value = self.data.get(section, {}).get(key)
        if value is None or value == "":
            raise ValueError(f"Set [{section}].{key}")
        return value

    @staticmethod
    def _validate_numbers(value):
        if isinstance(value, dict):
            for item in value.values():
                Experiment._validate_numbers(item)
        elif isinstance(value, list):
            for item in value:
                Experiment._validate_numbers(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Configuration values must be finite")

    def calibration(self) -> dict:
        values = self.section("calibration")
        path = values.pop("path", None)
        if path:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
            inherited = {key: artifact[key] for key in ("omega_rad_s_per_volt", "rabi_at_100_hz") if key in artifact}
            # Accept the original square-calibration artifact as well.
            if "peak_omega_over_2pi_khz" in artifact:
                inherited["rabi_at_100_hz"] = artifact["peak_omega_over_2pi_khz"] * 1000.0
            values = merge(inherited, values)
        return values

    def scan_axes(self) -> tuple[np.ndarray, np.ndarray]:
        values = self.section("scan")
        axes = []
        for name, low, high, count, default_low, default_high in [
            ("amplitude_errors", "amplitude_min", "amplitude_max", "amplitude_count", -0.5, 0.5),
            ("detuning_hz", "detuning_min_hz", "detuning_max_hz", "detuning_count", -62500.0, 62500.0),
        ]:
            if name in values:
                axis = np.asarray(values[name], dtype=float)
            else:
                n = values.get(count, 21)
                if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                    raise ValueError(f"scan.{count} must be a positive integer")
                axis = np.linspace(values.get(low, default_low), values.get(high, default_high), n)
            if axis.ndim != 1 or not len(axis) or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                raise ValueError(f"scan.{name} must be a finite increasing one-dimensional axis")
            axes.append(axis)
        if np.min(axes[0]) < -1:
            raise ValueError("Fractional amplitude error cannot be below -1")
        return axes[0], axes[1]


def positive(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value
