"""Explicit live TNMR operations; offline sequence preparation stays separate."""
from pathlib import Path

from hardware_capture.configuration import Experiment
from hardware_capture.devices.tnmr import TNMRControl
from hardware_capture.sequences.tps import Sequence
from .common import write_json


def control(config: Experiment, directory: Path, *, tnmr=None) -> dict:
    settings = config.section("tnmr")
    action = settings.get("action", "inspect")
    owned = tnmr is None
    if owned:
        tnmr = TNMRControl()
    try:
        if owned:
            tnmr.connect(launch=settings.get("launch", False))
        tnmr.open(config.require("tnmr", "path"))
        if action == "inspect":
            result = tnmr.snapshot()
            result["tables"] = {name: tnmr.get_table(name) for name in tnmr.get_table_list()}
            result["dashboard"] = {page: {name: _read_parameter(tnmr, name) for name in tnmr.parameter_list_in_page(page)} for page in tnmr.parameter_page_list()}
        elif action == "update":
            if settings.get("sequence_path"):
                tnmr.load_sequence(settings["sequence_path"])
            for name, value in settings.get("parameters", {}).items():
                tnmr.set_parameter(name, value)
            for name, values in settings.get("tables", {}).items():
                tnmr.set_table(name, values)
            result = tnmr.snapshot()
        elif action == "clear_tables":
            # Review used table names from the current sequence before clearing.
            exported = directory / "original_sequence.tps"
            tnmr.save_sequence_as(exported)
            used = {r.table_name for r in Sequence.load(exported).table_references()}
            requested = set(settings.get("clear_tables", []))
            if requested & used:
                raise ValueError(f"Cannot clear tables referenced by sequence: {sorted(requested & used)}")
            if not requested <= set(tnmr.get_table_list()):
                raise ValueError("Requested clear_tables includes unknown table names")
            tnmr.save_as(config.require("tnmr", "save_path"))
            for name in sorted(requested):
                tnmr.set_table(name, "0\r\n")
            tnmr.save()
            result = {"cleared_tables": sorted(requested), "used_tables": sorted(used), "table_names_deleted": False}
        elif action in {"zg", "go", "stop", "abort", "reset"}:
            getattr(tnmr, action)()
            result = {"action": action}
        else:
            raise ValueError(f"Unknown TNMR action {action}")
        if action == "update" and settings.get("save_path"):
            tnmr.save_as(settings["save_path"])
        write_json(directory / "tnmr.json", result)
        return result
    finally:
        if owned:
            tnmr.close()


def _read_parameter(tnmr, name):
    try:
        return tnmr.get_parameter(name)
    except RuntimeError as error:
        return {"unavailable": str(error)}
