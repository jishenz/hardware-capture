"""Tests for Redstone waveform generation conventions."""

from __future__ import annotations

import math

import pytest

from hardware_capture.rf.redstone import (
    Chirp,
    HS1,
    SAHS1,
    WURST,
    WURSTtanh,
    Waveform,
    available_waveform_types,
)


def test_chirp_positive_physical_offset_decreases_tnmr_phmod_phase() -> None:
    waveform = Chirp(
        points=3,
        sweep_width_hz=0.0,
        dwell_s=1.0e-6,
        offset_hz=-1000.0,
        phase_steps=360000,
    )

    assert waveform.frequency_values() == [0.0, 0.0, 0.0]
    assert waveform.phase_values() == pytest.approx([359.64, 359.28, 358.92])


def test_chirp_center_hz_moves_frequency_sweep() -> None:
    waveform = Chirp(
        points=3,
        sweep_width_hz=2000.0,
        center_hz=500.0,
    )

    assert waveform.frequency_values() == pytest.approx([-500.0, 500.0, 1500.0])


def test_chirp_phase_integrates_modulation_relative_to_sequence_offset() -> None:
    waveform = Chirp(
        points=3,
        sweep_width_hz=2000.0,
        dwell_s=1.0e-6,
        center_hz=500.0,
        offset_hz=500.0,
        phase_steps=360000,
    )

    assert waveform.frequency_values() == pytest.approx([-500.0, 500.0, 1500.0])
    assert waveform.phase_values() == pytest.approx([0.36, 0.36, 0.0])


def test_hs1_frequency_uses_finite_tanh_sweep() -> None:
    waveform = HS1(points=5, sweep_width_hz=2000.0, truncation=2.0)

    frequency = waveform.frequency_values()

    assert frequency[0] == pytest.approx(-1000.0)
    assert frequency[2] == pytest.approx(0.0)
    assert frequency[-1] == pytest.approx(1000.0)
    assert frequency[1] == pytest.approx(-frequency[-2])


def test_hs1_center_hz_moves_frequency_sweep() -> None:
    waveform = HS1(
        points=5,
        sweep_width_hz=2000.0,
        center_hz=500.0,
        truncation=2.0,
    )

    frequency = waveform.frequency_values()

    assert frequency[0] == pytest.approx(-500.0)
    assert frequency[2] == pytest.approx(500.0)
    assert frequency[-1] == pytest.approx(1500.0)


def test_hs1_offset_is_phase_reference_not_sweep_center() -> None:
    waveform = HS1(
        points=5,
        sweep_width_hz=2000.0,
        offset_hz=500.0,
        truncation=2.0,
    )

    frequency = waveform.frequency_values()

    assert frequency[0] == pytest.approx(-1000.0)
    assert frequency[2] == pytest.approx(0.0)
    assert frequency[-1] == pytest.approx(1000.0)


def test_hs1_amplitude_uses_sech_envelope() -> None:
    waveform = HS1(points=5, amp=2.0, truncation=2.0)

    assert waveform.amplitude_values() == pytest.approx(
        [
            2.0 / math.cosh(2.0),
            2.0 / math.cosh(1.0),
            2.0,
            2.0 / math.cosh(1.0),
            2.0 / math.cosh(2.0),
        ]
    )


def test_hs1_phase_integrates_centered_physical_offset() -> None:
    waveform = HS1(
        points=5,
        sweep_width_hz=2000.0,
        dwell_s=1.0e-6,
        truncation=2.0,
        phase_steps=360000,
    )

    phase = waveform.phase_values()

    assert phase[2] == pytest.approx(phase[1])
    assert phase[-1] == pytest.approx(0.0)


def test_hs1_phase_integrates_modulation_relative_to_sequence_offset() -> None:
    waveform = HS1(
        points=3,
        sweep_width_hz=2000.0,
        dwell_s=1.0e-6,
        truncation=2.0,
        offset_hz=-1000.0,
        phase_steps=360000,
    )

    assert waveform.frequency_values() == pytest.approx([-1000.0, 0.0, 1000.0])
    assert waveform.phase_values() == pytest.approx([0.0, 359.64, 358.92])


def test_hs1_registers_for_json_dispatch() -> None:
    assert "hs1" in available_waveform_types()

    loaded = Waveform.from_dict(
        {
            "waveform_type": "hs1",
            "parameters": {"points": 1, "amp": 0.5},
        }
    )

    assert isinstance(loaded, HS1)
    assert loaded.amplitude_values() == pytest.approx([0.5])


def test_sahs1_matches_notebook_counterdiabatic_equation() -> None:
    waveform = SAHS1(
        points=5,
        sweep_width_hz=160000.0,
        dwell_s=16.0e-6,
        truncation=2.0,
        amp=1.0,
        nutation_hz=160000.0,
        phase_steps=360000,
    )

    base_hz = waveform.base_nutation_hz()
    x_values, cd_values = waveform.quadrature_nutation_values()
    center = 2
    dx_dt = 2.0 * waveform.truncation / (
        waveform.points * waveform.dwell_s
    )
    detuning_scale_hz = (
        waveform.sweep_width_hz
        / (2.0 * math.tanh(waveform.truncation))
    )

    assert x_values[center] == pytest.approx(base_hz)
    assert cd_values[center] == pytest.approx(
        -dx_dt * detuning_scale_hz / (2.0 * math.pi * base_hz)
    )


def test_sahs1_defaults_to_4000_points() -> None:
    assert SAHS1().points == 4000


def test_sahs1_power_matches_total_transverse_amplitude() -> None:
    waveform = SAHS1(
        points=101,
        sweep_width_hz=160000.0,
        dwell_s=80.0e-6 / 101,
        truncation=8.0,
        amp=0.75,
        nutation_hz=160000.0,
    )

    x_values, cd_values = waveform.quadrature_nutation_values()
    transverse_values = [
        math.hypot(x_nutation_hz, cd_nutation_hz)
        for x_nutation_hz, cd_nutation_hz in zip(x_values, cd_values)
    ]

    assert max(transverse_values) == pytest.approx(
        waveform.amp * waveform.nutation_hz,
        rel=1e-10,
    )
    assert max(waveform.amplitude_values()) == pytest.approx(
        waveform.amp,
        rel=1e-10,
    )


def test_sahs1_phase_adds_center_referenced_cd_correction() -> None:
    parameters = {
        "points": 5,
        "sweep_width_hz": 160000.0,
        "dwell_s": 16.0e-6,
        "truncation": 2.0,
        "amp": 1.0,
        "nutation_hz": 160000.0,
        "phase_steps": 360000,
    }
    waveform = SAHS1(**parameters)
    bare = HS1(
        **{
            key: value
            for key, value in parameters.items()
            if key != "nutation_hz"
        }
    )
    correction = waveform.counterdiabatic_phase_values()

    assert correction[2] == pytest.approx(0.0)
    assert waveform._raw_phase_values() == pytest.approx(
        [
            phase_deg + correction_deg
            for phase_deg, correction_deg in zip(
                bare._raw_phase_values(),
                correction,
            )
        ]
    )


def test_sahs1_registers_for_json_dispatch() -> None:
    assert "sahs1" in available_waveform_types()

    loaded = Waveform.from_dict(
        {
            "waveform_type": "sahs1",
            "parameters": {
                "points": 5,
                "dwell_s": 20.0e-6,
                "amp": 0.5,
                "nutation_hz": 160000.0,
            },
        }
    )

    assert isinstance(loaded, SAHS1)
    assert max(loaded.amplitude_values()) <= 0.5 * (1.0 + 1.0e-12)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"points": 0}, "points"),
        ({"dwell_s": 0.0}, "dwell_s"),
        ({"truncation": 0.0}, "truncation"),
        ({"amp": -1.0}, "amp"),
        ({"nutation_hz": 0.0}, "nutation_hz"),
    ],
)
def test_sahs1_rejects_bad_shape_parameters(
    parameters: dict[str, float],
    message: str,
) -> None:
    waveform = SAHS1(**parameters)

    with pytest.raises(ValueError, match=message):
        waveform.summary()


def test_wurst_amplitude_uses_wurst_n_envelope() -> None:
    waveform = WURST(points=5, amp=2.0, order=2.0)

    assert waveform.amplitude_values() == pytest.approx([0.0, 1.0, 2.0, 1.0, 0.0])


def test_wurst_frequency_uses_linear_centered_sweep() -> None:
    waveform = WURST(
        points=5,
        sweep_width_hz=2000.0,
        center_hz=500.0,
    )

    assert waveform.frequency_values() == pytest.approx(
        [-500.0, 0.0, 500.0, 1000.0, 1500.0]
    )


def test_wurst_phase_integrates_modulation_relative_to_sequence_offset() -> None:
    waveform = WURST(
        points=3,
        sweep_width_hz=2000.0,
        dwell_s=1.0e-6,
        center_hz=500.0,
        offset_hz=500.0,
        phase_steps=360000,
    )

    assert waveform.frequency_values() == pytest.approx([-500.0, 500.0, 1500.0])
    assert waveform.phase_values() == pytest.approx([0.36, 0.36, 0.0])


def test_wurst_registers_for_json_dispatch() -> None:
    assert "wurst" in available_waveform_types()

    loaded = Waveform.from_dict(
        {
            "waveform_type": "wurst",
            "parameters": {"points": 1, "amp": 0.5},
        }
    )

    assert isinstance(loaded, WURST)
    assert loaded.amplitude_values() == pytest.approx([0.5])


def test_wursttanh_amplitude_uses_wurst_n_envelope() -> None:
    waveform = WURSTtanh(points=5, amp=2.0, order=2.0, beta=2.0)

    assert waveform.amplitude_values() == pytest.approx([0.0, 1.0, 2.0, 1.0, 0.0])


def test_wursttanh_frequency_uses_finite_tanh_sweep() -> None:
    waveform = WURSTtanh(
        points=5,
        sweep_width_hz=2000.0,
        center_hz=500.0,
        beta=2.0,
    )

    frequency = waveform.frequency_values()

    assert frequency[0] == pytest.approx(-500.0)
    assert frequency[2] == pytest.approx(500.0)
    assert frequency[-1] == pytest.approx(1500.0)
    assert frequency[1] - 500.0 == pytest.approx(-(frequency[-2] - 500.0))


def test_wursttanh_phase_integrates_modulation_relative_to_sequence_offset() -> None:
    waveform = WURSTtanh(
        points=3,
        sweep_width_hz=2000.0,
        dwell_s=1.0e-6,
        beta=2.0,
        center_hz=500.0,
        offset_hz=500.0,
        phase_steps=360000,
    )

    assert waveform.frequency_values() == pytest.approx([-500.0, 500.0, 1500.0])
    assert waveform.phase_values() == pytest.approx([0.36, 0.36, 0.0])


def test_wursttanh_registers_for_json_dispatch() -> None:
    assert "wursttanh" in available_waveform_types()

    loaded = Waveform.from_dict(
        {
            "waveform_type": "wursttanh",
            "parameters": {"points": 1, "amp": 0.5, "beta": 2.0},
        }
    )

    assert isinstance(loaded, WURSTtanh)
    assert loaded.amplitude_values() == pytest.approx([0.5])
    assert loaded.frequency_values() == pytest.approx([0.0])


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"points": 0}, "points"),
        ({"dwell_s": 0.0}, "dwell_s"),
        ({"truncation": 0.0}, "truncation"),
        ({"amp": -1.0}, "amp"),
    ],
)
def test_hs1_rejects_bad_shape_parameters(
    parameters: dict[str, float],
    message: str,
) -> None:
    waveform = HS1(**parameters)

    with pytest.raises(ValueError, match=message):
        waveform.summary()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"points": 0}, "points"),
        ({"dwell_s": 0.0}, "dwell_s"),
        ({"order": 0.0}, "order"),
        ({"amp": -1.0}, "amp"),
    ],
)
def test_wurst_rejects_bad_shape_parameters(
    parameters: dict[str, float],
    message: str,
) -> None:
    waveform = WURST(**parameters)

    with pytest.raises(ValueError, match=message):
        waveform.summary()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"points": 0}, "points"),
        ({"dwell_s": 0.0}, "dwell_s"),
        ({"order": 0.0}, "order"),
        ({"beta": 0.0}, "beta"),
        ({"amp": -1.0}, "amp"),
    ],
)
def test_wursttanh_rejects_bad_shape_parameters(
    parameters: dict[str, float],
    message: str,
) -> None:
    waveform = WURSTtanh(**parameters)

    with pytest.raises(ValueError, match=message):
        waveform.summary()
