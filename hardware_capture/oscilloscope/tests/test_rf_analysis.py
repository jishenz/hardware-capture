import math

import numpy as np
import pytest

from hardware_capture.oscilloscope.examples import scope_rf_robustness_heatmap
from hardware_capture.oscilloscope.examples.scope_rf_robustness_heatmap import (
    COMPLEX_PULSE,
    REGULAR_BB1_PULSE,
    _pulse_scaling_for_type,
    _validate_fixed_omega_scale,
)
from hardware_capture.oscilloscope.pulse_robustness import (
    bb1_nominal_area,
    build_pulse_model,
    scan_robustness,
)
from hardware_capture.oscilloscope.rf_demodulation import (
    DemodulatedWaveform,
    demodulate_raw_rf,
    detect_pulse_window,
    estimate_demodulation_frequency,
)
from hardware_capture.oscilloscope.rigol_dho import ScopePreamble, ScopeWaveform

pytestmark = pytest.mark.unit


def test_estimate_demodulation_frequency_finds_sine_peak():
    sample_rate = 10_000.0
    freq = 1_000.0
    time = np.arange(0, 0.1, 1 / sample_rate)
    voltage = np.sin(2 * np.pi * freq * time)

    assert estimate_demodulation_frequency(time, voltage) == pytest.approx(freq, rel=0.02)


def test_demodulate_raw_rf_applies_q_sign():
    sample_rate = 100_000.0
    freq = 1_000.0
    time = np.arange(0, 0.01, 1 / sample_rate)
    voltage = np.sin(2 * np.pi * freq * time)
    preamble = ScopePreamble("0", "0", len(time), 1, 1 / sample_rate, 0.0, 0.0, 1, 0, 0)
    raw = ScopeWaveform(
        time=time,
        voltage=voltage,
        sample_rate=sample_rate,
        source="synthetic",
        channel="CHAN1",
        preamble=preamble,
        start_point=1,
        stop_point=len(time),
    )

    plus = demodulate_raw_rf(
        raw,
        demodulation_freq_hz=freq,
        baseband_cutoff_hz=5_000.0,
        q_sign=1,
    )
    minus = demodulate_raw_rf(
        raw,
        demodulation_freq_hz=freq,
        baseband_cutoff_hz=5_000.0,
        q_sign=-1,
    )

    center = slice(100, -100)
    assert np.allclose(plus.i[center], minus.i[center])
    assert np.allclose(plus.q[center], -minus.q[center])
    assert plus.q_sign == 1
    assert minus.q_sign == -1


def test_detect_pulse_window_and_scan_center_fidelity():
    time = np.linspace(0.0, 1e-6, 201)
    amp = np.zeros_like(time)
    amp[60:140] = 1.0
    demod = DemodulatedWaveform(
        time=time,
        i=amp,
        q=np.zeros_like(amp),
        z=amp.astype(complex),
        amplitude=amp,
        phase=np.zeros_like(amp),
        sample_rate=1.0 / float(np.median(np.diff(time))),
        demodulation_freq_hz=1.0e6,
        baseband_cutoff_hz=100e3,
        filter_order=2,
        raw_baseline=0.0,
    )

    window = detect_pulse_window(demod, threshold_fraction=0.5, padding_seconds=0.0)
    pulse = build_pulse_model(
        demod,
        window,
        nominal_pulse_area_rad=bb1_nominal_area(0.5 * math.pi),
        simulation_steps=20,
    )
    result = scan_robustness(pulse, grid_size=5)

    assert window.samples == 80
    assert result.center_fidelity == pytest.approx(1.0)
    assert result.fidelity.shape == (5, 5)


def test_build_pulse_model_accepts_fixed_square_pulse_calibration():
    time = np.linspace(0.0, 2e-6, 101)
    i = np.linspace(0.01, 0.02, len(time))
    demod = DemodulatedWaveform(
        time=time,
        i=i,
        q=np.zeros_like(i),
        z=i.astype(complex),
        amplitude=i,
        phase=np.zeros_like(i),
        sample_rate=1.0 / float(np.median(np.diff(time))),
        demodulation_freq_hz=1.0e6,
        baseband_cutoff_hz=100e3,
        filter_order=2,
        raw_baseline=0.0,
    )
    window = detect_pulse_window(
        demod,
        threshold_fraction=0.5,
        padding_seconds=0.0,
        manual_start=float(time[0]),
        manual_end=float(time[-1]),
    )
    omega_rad_s_per_volt = _validate_fixed_omega_scale(1_500_000.0)

    pulse = build_pulse_model(
        demod,
        window,
        nominal_pulse_area_rad=bb1_nominal_area(),
        simulation_steps=20,
        omega_rad_s_per_volt=omega_rad_s_per_volt,
    )

    assert pulse.scale == pytest.approx(omega_rad_s_per_volt)
    assert pulse.omega_x == pytest.approx(pulse.scale * np.interp(pulse.time_mid, time, i))
    assert pulse.omega_y == pytest.approx(0.0)
    assert pulse.omega_ref == pytest.approx(
        pulse.scale * pulse.raw_envelope_area / window.duration
    )


def test_complex_pulse_scaling_uses_top_level_square_calibration(monkeypatch):
    expected = 1_500_000.0
    monkeypatch.setattr(
        scope_rf_robustness_heatmap,
        "SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT",
        expected,
    )

    _, scale = _pulse_scaling_for_type(COMPLEX_PULSE, 90.0)

    assert scale == pytest.approx(expected)


def test_regular_bb1_scaling_remains_area_based_without_square_calibration():
    nominal_area, scale = _pulse_scaling_for_type(REGULAR_BB1_PULSE, 90.0)

    assert nominal_area == pytest.approx(bb1_nominal_area(0.5 * math.pi))
    assert scale is None


@pytest.mark.parametrize("value", [None, 0.0, -1.0, math.inf, math.nan])
def test_fixed_square_calibration_rejects_invalid_scale(value):
    with pytest.raises(ValueError, match="SQUARE_PULSE_OMEGA_RAD_S_PER_VOLT"):
        _validate_fixed_omega_scale(value)
