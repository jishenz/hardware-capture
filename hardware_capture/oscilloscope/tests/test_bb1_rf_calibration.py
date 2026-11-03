import json
import math

import numpy as np
import pytest

from hardware_capture.oscilloscope.examples.calibrate_rf_from_bb1 import (
    DEFAULT_FIGURE_PATH,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_RESULTS_DIR,
    INPUT_CSV,
    MIN_BASELINE_SAMPLES,
    build_parser,
    calibrate_bb1_waveform,
    calibration_to_dict,
    load_waveform_csv,
    write_calibration_json,
)
from hardware_capture.oscilloscope.rigol_dho import ScopePreamble, ScopeWaveform

pytestmark = pytest.mark.unit

CARRIER_HZ = 2.0e6
SAMPLE_RATE_HZ = 20.0e6
BASEBAND_CUTOFF_HZ = 400.0e3
TEST_RABI_KHZ_PER_MV = 0.23
TEST_AMPLITUDE_MV = 30.0


def _synthetic_bb1(
    *,
    rabi_khz_per_mv=TEST_RABI_KHZ_PER_MV,
    amplitude_mv=TEST_AMPLITUDE_MV,
    target_angle_rad=0.5 * math.pi,
    phase_changes=True,
    leakage_i_v=0.0,
    leakage_q_v=0.0,
    baseline_duration_s=50e-6,
):
    amplitude_v = amplitude_mv / 1_000.0
    rabi_hz = rabi_khz_per_mv * amplitude_mv * 1_000.0
    omega_rad_s = 2.0 * math.pi * rabi_hz
    segment_angles = np.asarray(
        [target_angle_rad, math.pi, 2.0 * math.pi, math.pi]
    )
    segment_durations_s = segment_angles / omega_rad_s
    pulse_duration_s = float(np.sum(segment_durations_s))
    time = np.arange(
        -baseline_duration_s,
        pulse_duration_s + baseline_duration_s,
        1.0 / SAMPLE_RATE_HZ,
    )

    correction_phase = math.acos(-target_angle_rad / (4.0 * math.pi))
    phases = (
        np.asarray([0.0, correction_phase, 3.0 * correction_phase, correction_phase])
        if phase_changes
        else np.zeros(4)
    )
    complex_baseband = np.full(
        len(time),
        complex(leakage_i_v, leakage_q_v),
        dtype=complex,
    )
    boundaries = np.concatenate(([0.0], np.cumsum(segment_durations_s)))
    for index, phase in enumerate(phases):
        active = (time >= boundaries[index]) & (time < boundaries[index + 1])
        complex_baseband[active] += amplitude_v * np.exp(1j * phase)

    voltage = np.real(complex_baseband * np.exp(1j * 2.0 * math.pi * CARRIER_HZ * time))
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
    waveform = ScopeWaveform(
        time=time,
        voltage=voltage,
        sample_rate=SAMPLE_RATE_HZ,
        source="synthetic-bb1",
        channel="CHAN1",
        preamble=preamble,
        start_point=1,
        stop_point=len(time),
    )
    return waveform, segment_durations_s


def _calibrate(waveform, **kwargs):
    return calibrate_bb1_waveform(
        waveform,
        demodulation_freq_hz=CARRIER_HZ,
        baseband_cutoff_hz=BASEBAND_CUTOFF_HZ,
        padding_seconds=3e-6,
        **kwargs,
    )


def test_diagnostic_figure_is_saved_by_default():
    parser = build_parser()

    assert parser.parse_args([]).save_figure
    assert not parser.parse_args(["--no-save-figure"]).save_figure


def test_default_calibration_artifacts_are_saved_under_oscilloscope_results():
    args = build_parser().parse_args([])

    assert args.output_json == DEFAULT_OUTPUT_JSON
    assert args.figure_path == DEFAULT_FIGURE_PATH
    assert args.output_json.parent == DEFAULT_RESULTS_DIR
    assert args.figure_path.parent == DEFAULT_RESULTS_DIR
    assert DEFAULT_RESULTS_DIR.parent.name == "oscilloscope"


def test_input_csv_is_a_top_level_configuration_not_a_cli_argument():
    assert INPUT_CSV is not None
    assert INPUT_CSV.name == "pulse1_008.csv"
    assert "--input-csv" not in build_parser().format_help()


def test_synthetic_bb1_recovers_deliberately_chosen_coefficient():
    waveform, _ = _synthetic_bb1()

    result = _calibrate(waveform)

    assert result.rabi_khz_per_mv == pytest.approx(TEST_RABI_KHZ_PER_MV, rel=0.02)
    assert result.omega_rad_s_per_volt == pytest.approx(
        TEST_RABI_KHZ_PER_MV * 2.0 * math.pi * 1e6,
        rel=0.02,
    )


def test_bb1_phase_changes_do_not_cancel_envelope_area():
    waveform, _ = _synthetic_bb1(phase_changes=True)
    result = _calibrate(waveform)
    active = slice(result.pulse_window.start_index, result.pulse_window.stop_index)
    time = result.demodulated.time[active]
    complex_values = result.demodulated.z[active]
    complex_area = abs(
        np.sum(0.5 * (complex_values[:-1] + complex_values[1:]) * np.diff(time))
    )

    assert result.rabi_khz_per_mv == pytest.approx(TEST_RABI_KHZ_PER_MV, rel=0.02)
    assert complex_area < 0.5 * result.voltage_envelope_area_v_s


def test_voltage_scaling_with_inverse_duration_preserves_coefficient():
    original, _ = _synthetic_bb1(amplitude_mv=TEST_AMPLITUDE_MV)
    scaled, _ = _synthetic_bb1(amplitude_mv=2.0 * TEST_AMPLITUDE_MV)

    original_result = _calibrate(original)
    scaled_result = _calibrate(scaled)

    assert scaled_result.rabi_khz_per_mv == pytest.approx(
        original_result.rabi_khz_per_mv,
        rel=0.01,
    )
    assert scaled_result.rabi_khz_per_mv == pytest.approx(
        TEST_RABI_KHZ_PER_MV,
        rel=0.02,
    )


def test_constant_complex_leakage_is_removed_by_baseline_correction():
    clean, _ = _synthetic_bb1()
    leaked, _ = _synthetic_bb1(leakage_i_v=1.2e-3, leakage_q_v=-0.8e-3)

    clean_result = _calibrate(clean)
    leaked_result = _calibrate(leaked)

    assert leaked_result.i_baseline_v == pytest.approx(1.2e-3, abs=2e-5)
    assert leaked_result.q_baseline_v == pytest.approx(-0.8e-3, abs=2e-5)
    assert leaked_result.rabi_khz_per_mv == pytest.approx(
        clean_result.rabi_khz_per_mv,
        rel=0.005,
    )


def test_q_convention_does_not_change_envelope_calibration():
    waveform, _ = _synthetic_bb1()

    plus = _calibrate(waveform, q_sign=1)
    minus = _calibrate(waveform, q_sign=-1)

    assert minus.rabi_khz_per_mv == pytest.approx(plus.rabi_khz_per_mv, rel=1e-12)
    assert np.allclose(minus.demodulated.amplitude, plus.demodulated.amplitude)


def test_zero_area_waveform_is_rejected():
    waveform, _ = _synthetic_bb1()
    zero = ScopeWaveform(
        time=waveform.time,
        voltage=np.zeros_like(waveform.voltage),
        sample_rate=waveform.sample_rate,
        source=waveform.source,
        channel=waveform.channel,
        preamble=waveform.preamble,
        start_point=waveform.start_point,
        stop_point=waveform.stop_point,
    )

    with pytest.raises(ValueError, match="zero peak|positive"):
        _calibrate(zero)


def test_empty_manual_window_is_rejected():
    waveform, _ = _synthetic_bb1()

    with pytest.raises(ValueError, match="end must be greater than start"):
        _calibrate(waveform, manual_start_s=10e-6, manual_end_s=10e-6)


def test_insufficient_out_of_pulse_baseline_is_rejected():
    waveform, _ = _synthetic_bb1()
    sample_interval_s = 1.0 / waveform.sample_rate
    outside_samples_per_side = MIN_BASELINE_SAMPLES // 4

    with pytest.raises(ValueError, match="Insufficient out-of-pulse baseline"):
        _calibrate(
            waveform,
            manual_start_s=float(
                waveform.time[outside_samples_per_side] - 0.25 * sample_interval_s
            ),
            manual_end_s=float(
                waveform.time[-outside_samples_per_side - 1] + 0.25 * sample_interval_s
            ),
        )


def test_json_artifact_contains_units_values_and_offline_metadata(tmp_path):
    waveform, _ = _synthetic_bb1()
    result = _calibrate(waveform)
    input_path = tmp_path / "capture.csv"
    payload = calibration_to_dict(
        result,
        waveform,
        input_mode="offline",
        q_convention="I+iQ",
        input_csv=input_path,
        csv_metadata={"shot_index": "1"},
    )
    output_path = tmp_path / "calibration.json"

    write_calibration_json(output_path, payload)
    saved = json.loads(output_path.read_text())

    assert saved["schema_version"] == 1
    assert saved["method"] == "known_bb1_envelope_area"
    assert saved["rabi_khz_per_mv"] == pytest.approx(result.rabi_khz_per_mv)
    assert saved["omega_rad_s_per_volt"] == pytest.approx(result.omega_rad_s_per_volt)
    assert saved["units"]["omega_rad_s_per_volt"] == "rad/s/V"
    assert saved["units"]["rabi_khz_per_mv"] == "kHz/mV for Omega/(2*pi)"
    assert saved["input_csv"] == str(input_path)
    assert saved["csv_metadata"] == {"shot_index": "1"}
    with pytest.raises(FileExistsError, match="--force"):
        write_calibration_json(output_path, payload)


def test_pi_over_2_duration_cross_check_agrees_for_flat_synthetic_pulse():
    waveform, segment_durations_s = _synthetic_bb1()
    result = _calibrate(
        waveform,
        pi_over_2_duration_s=float(segment_durations_s[1] / 2.0),
    )

    cross_check = result.duration_cross_check
    assert cross_check is not None
    assert cross_check.plateau_usable
    assert cross_check.measured_plateau_voltage_mv == pytest.approx(
        TEST_AMPLITUDE_MV,
        rel=0.01,
    )
    assert cross_check.plateau_rabi_khz_per_mv == pytest.approx(
        TEST_RABI_KHZ_PER_MV,
        rel=0.01,
    )
    assert cross_check.relative_difference < 0.02


def test_repository_capture_csv_is_loaded_with_metadata(tmp_path):
    csv_path = tmp_path / "capture.csv"
    csv_path.write_text(
        "# scope,RIGOL synthetic\n"
        "# channel,CHAN1\n"
        "# shot_index,7\n"
        "time_s,chan1_voltage_V\n"
        "0.0,0.1\n"
        "0.000001,0.2\n"
        "0.000002,0.3\n"
    )

    waveform, metadata = load_waveform_csv(csv_path)

    assert waveform.source == "RIGOL synthetic"
    assert waveform.channel == "CHAN1"
    assert waveform.sample_rate == pytest.approx(1e6)
    assert np.allclose(waveform.voltage, [0.1, 0.2, 0.3])
    assert metadata["shot_index"] == "7"
