import json
import math

import numpy as np
import pytest

from hardware_capture.oscilloscope.examples.calibrate_rf_from_square import (
    DEFAULT_FIGURE_PATH,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_RESULTS_DIR,
    PEAK_OMEGA_OVER_2PI_KHZ,
    build_parser,
    calibrate_square_waveform,
    calibration_to_dict,
    write_calibration_json,
)
from hardware_capture.oscilloscope.rigol_dho import ScopePreamble, ScopeWaveform

pytestmark = pytest.mark.unit

CARRIER_HZ = 2.0e6
SAMPLE_RATE_HZ = 20.0e6
BASEBAND_CUTOFF_HZ = 400.0e3
PULSE_DURATION_S = 20.0e-6
PLATEAU_V = 40.0e-3


def _synthetic_square(
    *,
    plateau_v: float = PLATEAU_V,
    pulse_duration_s: float = PULSE_DURATION_S,
    leakage_i_v: float = 0.0,
    leakage_q_v: float = 0.0,
    distorted_edges: bool = False,
    isolated_spike: bool = False,
) -> ScopeWaveform:
    baseline_duration_s = 20.0e-6
    time = np.arange(
        -baseline_duration_s,
        pulse_duration_s + baseline_duration_s,
        1.0 / SAMPLE_RATE_HZ,
    )
    envelope = np.zeros_like(time)
    active = (time >= 0.0) & (time < pulse_duration_s)
    envelope[active] = plateau_v
    if distorted_edges:
        envelope[(time >= 0.0) & (time < 2.0e-6)] = 1.8 * plateau_v
        envelope[
            (time >= pulse_duration_s - 2.0e-6) & (time < pulse_duration_s)
        ] = 0.55 * plateau_v
    if isolated_spike:
        envelope[(time >= -10.0e-6) & (time < -9.9e-6)] = 4.0 * plateau_v

    complex_baseband = (
        complex(leakage_i_v, leakage_q_v)
        + envelope * np.exp(1j * 0.37)
    )
    voltage = np.real(
        complex_baseband * np.exp(1j * 2.0 * math.pi * CARRIER_HZ * time)
    )
    preamble = ScopePreamble(
        "0",
        "0",
        len(time),
        1,
        1.0 / SAMPLE_RATE_HZ,
        float(time[0]),
        0.0,
        1.0,
        0.0,
        0.0,
    )
    return ScopeWaveform(
        time=time,
        voltage=voltage,
        sample_rate=SAMPLE_RATE_HZ,
        source="synthetic-square",
        channel="CHAN1",
        preamble=preamble,
        start_point=1,
        stop_point=len(time),
    )


def _calibrate(waveform: ScopeWaveform, **kwargs):
    return calibrate_square_waveform(
        waveform,
        demodulation_freq_hz=CARRIER_HZ,
        baseband_cutoff_hz=BASEBAND_CUTOFF_HZ,
        detection_smoothing_seconds=0.5e-6,
        baseline_guard_seconds=1.0e-6,
        **kwargs,
    )


def test_default_outputs_and_peak_frequency_are_operator_configurable():
    args = build_parser().parse_args([])

    assert args.peak_khz == PEAK_OMEGA_OVER_2PI_KHZ
    assert args.output_json == DEFAULT_OUTPUT_JSON
    assert args.figure_path == DEFAULT_FIGURE_PATH
    assert args.output_json.parent == DEFAULT_RESULTS_DIR
    assert args.save_figure
    assert build_parser().parse_args(["--peak-khz", "125"]).peak_khz == 125.0


def test_auto_detection_and_middle_median_recover_square_calibration():
    waveform = _synthetic_square(
        leakage_i_v=1.2e-3,
        leakage_q_v=-0.8e-3,
        distorted_edges=True,
        isolated_spike=True,
    )

    result = _calibrate(waveform)
    expected_scale = (
        2.0 * math.pi * PEAK_OMEGA_OVER_2PI_KHZ * 1_000.0 / PLATEAU_V
    )

    assert result.pulse_window.duration == pytest.approx(PULSE_DURATION_S, abs=1.0e-6)
    assert result.middle_window.start_time > result.pulse_window.start_time
    assert result.middle_window.end_time < result.pulse_window.end_time
    assert result.middle_envelope_v == pytest.approx(PLATEAU_V, rel=0.02)
    assert result.omega_rad_s_per_volt == pytest.approx(expected_scale, rel=0.02)
    assert result.i_baseline_v == pytest.approx(1.2e-3 * math.cos(0.0), abs=2e-4)
    assert result.q_baseline_v == pytest.approx(-0.8e-3, abs=2e-4)


def test_peak_frequency_changes_scale_without_changing_measured_voltage():
    waveform = _synthetic_square()

    lower = _calibrate(waveform, peak_omega_over_2pi_khz=100.0)
    higher = _calibrate(waveform, peak_omega_over_2pi_khz=125.0)

    assert higher.middle_envelope_v == pytest.approx(lower.middle_envelope_v)
    assert higher.omega_rad_s_per_volt / lower.omega_rad_s_per_volt == pytest.approx(
        1.25
    )


def test_json_artifact_contains_pasteable_fixed_scale(tmp_path):
    waveform = _synthetic_square()
    result = _calibrate(waveform)
    payload = calibration_to_dict(
        result,
        waveform,
        q_convention="I+iQ",
        scope_resource="TCPIP::synthetic::INSTR",
    )
    output_path = tmp_path / "square_calibration.json"

    write_calibration_json(output_path, payload)
    saved = json.loads(output_path.read_text())

    assert saved["method"] == "tnmr_100_square_middle_median"
    assert saved["middle_envelope_v"] == pytest.approx(result.middle_envelope_v)
    assert saved["omega_rad_s_per_volt"] == pytest.approx(
        result.omega_rad_s_per_volt
    )
    assert saved["units"]["omega_rad_s_per_volt"] == "rad/s/V"
    with pytest.raises(FileExistsError, match="--force"):
        write_calibration_json(output_path, payload)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_peak_frequency_is_rejected(value):
    with pytest.raises(ValueError, match="peak_omega_over_2pi_khz"):
        _calibrate(_synthetic_square(), peak_omega_over_2pi_khz=value)
