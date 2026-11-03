"""Tests for repo-local Redstone TPS sequence table editing."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from hardware_capture import HS1, Sequence, Sine


def _example_tps_path() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "hardware_capture/resources/tnmr"
        / "references"
        / "examples"
        / "tnmr-sequences"
        / "common"
        / "1pulse - f1.tps"
    )


def _nvnmr_template_tps_path() -> Path:
    return Path(__file__).resolve().parents[2] / "hardware_capture/resources/tnmr" / "20260620_NVNMR_template.tps"


def _nvnmr_adiabatic_20260704_tps_path() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "hardware_capture/resources/tnmr"
        / "20260704_NVNMR_AdiabaticInversionTrain_AMPM.tps"
    )


def _type_code(value: int) -> str:
    return value.to_bytes(4, "little").rstrip(b"\x00").decode("latin1")


def test_remove_unused_tables_writes_reloadable_tps(tmp_path: Path) -> None:
    sequence = Sequence.load(_example_tps_path())
    assert sequence.tables

    unused = copy.deepcopy(sequence.tables[0])
    unused["name"] = "__unused_test_table__"
    sequence.tables.append(unused)

    result = sequence.remove_unused_tables()

    assert "__unused_test_table__" in result["removed_tables"]
    output_path = tmp_path / "used_tables_only.tps"
    sequence.write_tps(output_path)

    reloaded = Sequence.load(output_path)
    table_names = {table["name"] for table in reloaded.tables if table.get("name")}
    referenced_names = {
        reference.table_name
        for reference in reloaded.table_references()
        if reference.table_name
    }
    assert "__unused_test_table__" not in table_names
    assert referenced_names <= table_names


def test_set_waveform_to_table_and_assign_event_update_existing_table(
    tmp_path: Path,
) -> None:
    sequence = Sequence.load(_example_tps_path())

    sequence.set_waveform_to_table(
        "ph1",
        Sine(
            points=4,
            phase_min=0.0,
            phase_max=180.0,
            amp_min=None,
            amp_max=None,
        ),
        quantity="phase",
    )
    sequence.assign_table_to_event("F1_PhMod", 5, "ph1")

    edited = next(table for table in sequence.tables if table["name"] == "ph1")
    assert len(edited["entries"].splitlines()) == 4
    assert edited["value_type"] == 0

    row = next(row for row in sequence.rows if row["label"] == "F1_PhMod")
    assert _type_code(row["icon_library_type_raw"]) == "3P"
    assert row["bit_length"] == 16
    assert row["address_raw"] & 0xFF == 0x1F

    output_path = tmp_path / "phase_mod_3p.tps"
    sequence.write_tps(output_path)
    reloaded = Sequence.load(output_path)
    reloaded_table = next(table for table in reloaded.tables if table["name"] == "ph1")
    reloaded_row = next(row for row in reloaded.rows if row["label"] == "F1_PhMod")

    assert len(reloaded_table["entries"].splitlines()) == 4
    assert reloaded_table["value_type"] == 0
    assert _type_code(reloaded_row["icon_library_type_raw"]) == "3P"
    assert reloaded_row["bit_length"] == 16


def test_set_waveform_to_table_can_write_step_metadata() -> None:
    sequence = Sequence.load(_example_tps_path())
    waveform = Sine(
        points=4,
        phase_min=0.0,
        phase_max=180.0,
        amp_min=None,
        amp_max=None,
    )

    sequence.set_waveform_to_table(
        "ph1",
        waveform,
        quantity="phase",
        phase_units="step",
    )

    table = next(table for table in sequence.tables if table["name"] == "ph1")
    assert table["value_type"] == 1

    sequence.set_waveform_to_table(
        "ph1",
        waveform,
        quantity="phase",
    )

    assert table["value_type"] == 0


def test_set_waveform_to_table_can_write_amplitude_to_existing_table() -> None:
    sequence = Sequence.load(_example_tps_path())
    table = copy.deepcopy(sequence.tables[0])
    table["name"] = "amp_wf"
    table["entries"] = "0"
    original_value_type = table["value_type"]
    sequence.tables.append(table)

    sequence.set_waveform_to_table(
        "amp_wf",
        Sine(
            points=4,
            amp_min=0.0,
            amp_max=1.0,
            phase_min=None,
            phase_max=None,
        ),
        quantity="amplitude",
    )

    edited = next(table for table in sequence.tables if table["name"] == "amp_wf")
    assert len(edited["entries"].splitlines()) == 4
    assert edited["value_type"] == original_value_type


def test_nvnmr_template_tail_sections_round_trip_when_available() -> None:
    path = _nvnmr_template_tps_path()
    if not path.exists():
        pytest.skip("NV-NMR template TPS is not available in this checkout.")

    sequence = Sequence.load(path)

    assert sequence.to_bytes() == path.read_bytes()
    assert len(sequence.header_extra_bytes) == 43
    assert [table["name"] for table in sequence.tables] == [
        "ph_cycle",
        "phmod_p180",
        "phmod_pn180",
    ]
    assert sequence.parameter_values()["Acq. Time"] == "102.4u"
    assert sequence.parameter_values()["N_4fcycle"] == "200"


def test_nvnmr_template_existing_phase_table_edit_when_available(
    tmp_path: Path,
) -> None:
    path = _nvnmr_template_tps_path()
    if not path.exists():
        pytest.skip("NV-NMR template TPS is not available in this checkout.")

    sequence = Sequence.load(path)
    sequence.set_parameter_values({"N_4fcycle": 12, "t_pn180": "12u"})
    sequence.set_waveform_to_table(
        "phmod_pn180",
        Sine(
            points=8,
            phase_min=0.0,
            phase_max=180.0,
            amp_min=None,
            amp_max=None,
            phase_steps=1024,
        ),
        quantity="phase",
        allow_shared=True,
    )
    sequence.assign_table_to_event("F1_PhMod", 5, "phmod_pn180")

    output_path = tmp_path / "nvnmr_template_waveform_edit.tps"
    sequence.write_tps(output_path)
    reloaded = Sequence.load(output_path)

    n_4fcycle = next(
        parameter
        for parameter in reloaded.parameters
        if parameter["name"] == "N_4fcycle"
    )
    edited_table = next(
        table for table in reloaded.tables if table["name"] == "phmod_pn180"
    )

    assert n_4fcycle["value"] == "12"
    assert n_4fcycle["current_value"] == "12"
    assert len(edited_table["entries"].splitlines()) == 8


def test_nvnmr_adiabatic_ampm_hs1_phase_and_amplitude_tables_when_available(
    tmp_path: Path,
) -> None:
    path = _nvnmr_adiabatic_20260704_tps_path()
    if not path.exists():
        pytest.skip(
            "NV-NMR adiabatic inversion AMPM TPS is not available in this checkout."
        )

    sequence = Sequence.load(path)
    existing_references = {
        (reference.line, reference.event_number, reference.table_name)
        for reference in sequence.table_references()
    }
    expected_p180_references = {
        ("F1_PhMod", 5, "phmod_p180"),
        ("F1_PhMod", 6, "phmod_p180"),
        ("F1_Ampl", 5, "ampmod_p180"),
        ("F1_Ampl", 6, "ampmod_p180"),
    }
    assert expected_p180_references <= existing_references

    waveform = HS1(
        points=16,
        sweep_width_hz=200000.0,
        dwell_s=0.2e-6,
        amp=100.0,
        offset_hz=-100000.0,
        phase_steps=2**16,
    )

    sequence.set_waveform_to_table(
        "phmod_p180",
        waveform,
        quantity="phase",
        allow_shared=True,
    )
    sequence.set_waveform_to_table(
        "ampmod_p180",
        waveform,
        quantity="amplitude",
        allow_shared=True,
    )

    output_path = tmp_path / "nvnmr_adiabatic_hs1_waveform_edit.tps"
    sequence.write_tps(output_path)
    reloaded = Sequence.load(output_path)
    reloaded_tables = {table["name"]: table for table in reloaded.tables}
    reloaded_references = {
        (reference.line, reference.event_number, reference.table_name)
        for reference in reloaded.table_references()
    }

    assert len(reloaded_tables["phmod_p180"]["entries"].splitlines()) == 16
    assert len(reloaded_tables["ampmod_p180"]["entries"].splitlines()) == 16
    assert reloaded_tables["ampmod_p180"]["table_type_text"] == "3R"
    assert expected_p180_references <= reloaded_references
