"""
test_chirp_ao.py

Tests for FMChirpAOcontrol (full class with:
- HARD clamp V_MIN=0.0, V_MAX=2.5
- waveforms: sawtooth, triangle, raised_cosine_monotonic, cosine_periodic
- direction: up/down
- finite/continuous modes (logic checks only; hardware tests optional)

By default: NO HARDWARE REQUIRED.
Optional hardware smoke tests can be enabled by setting RUN_HARDWARE_TESTS=True.

Run:
  python -m pytest hardware_capture/rf/windfreak/tests/test_chirp_ao_control.py
"""

import os

import pytest
import numpy as np

pytest.importorskip("nidaqmx")

from hardware_capture.rf.windfreak.chirp import FMChirpAOcontrol, V_MAX, V_MIN

pytestmark = pytest.mark.unit

RUN_HARDWARE_TESTS = False

AO_CH = os.getenv("NI_AO_CH", "Dev1/ao3")
TRIG_SRC = os.getenv("NI_TRIG_SRC", "/Dev1/PFI1")


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def assert_close(a: float, b: float, tol: float, msg: str) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} vs {b} (tol={tol})")


def assert_in_range(w: np.ndarray, lo: float, hi: float, name: str) -> None:
    assert_true(np.min(w) >= lo - 1e-12, f"{name}: min {np.min(w)} < {lo}")
    assert_true(np.max(w) <= hi + 1e-12, f"{name}: max {np.max(w)} > {hi}")


def is_monotonic_nonincreasing(w: np.ndarray) -> bool:
    return bool(np.all(np.diff(w) <= 1e-12))


def is_monotonic_nondecreasing(w: np.ndarray) -> bool:
    return bool(np.all(np.diff(w) >= -1e-12))


# ============================================================
# Pure waveform/safety tests (NO HARDWARE)
# ============================================================


def test_raised_cosine_monotonic_down():
    ao = FMChirpAOcontrol(AO_CH)
    ao.set_waveform_raised_cosine_monotonic(direction="down")
    ao.set_timing(sweep_time_s=3e-3, sample_rate=500_000)

    w = ao.build_waveform()
    assert_in_range(w, V_MIN, V_MAX, "raised_cosine_monotonic_down")
    assert_close(float(w[0]), V_MAX, 1e-5, "raised_cosine_down start != V_MAX")
    assert_true(float(w[-1]) <= 0.05, "raised_cosine_down end not near 0V")
    assert_true(
        is_monotonic_nonincreasing(w), "raised_cosine_down not monotonic decreasing"
    )

    print("PASS: test_raised_cosine_monotonic_down")


def test_raised_cosine_monotonic_up():
    ao = FMChirpAOcontrol(AO_CH)
    ao.set_waveform_raised_cosine_monotonic(direction="up")
    ao.set_timing(sweep_time_s=3e-3, sample_rate=500_000)

    w = ao.build_waveform()
    assert_in_range(w, V_MIN, V_MAX, "raised_cosine_monotonic_up")
    assert_true(float(w[0]) <= 0.05, "raised_cosine_up start not near 0V")
    assert_close(float(np.max(w)), V_MAX, 1e-5, "raised_cosine_up max != V_MAX")
    assert_true(
        is_monotonic_nondecreasing(w), "raised_cosine_up not monotonic increasing"
    )

    print("PASS: test_raised_cosine_monotonic_up")


def test_cosine_periodic_wrap_smooth():
    ao = FMChirpAOcontrol(AO_CH)
    ao.set_waveform_cosine(direction="down")
    ao.set_timing(sweep_time_s=2e-3, sample_rate=500_000)

    w = ao.build_waveform()
    assert_in_range(w, V_MIN, V_MAX, "cosine_periodic_down")

    # Wraparound smooth means first and last are approximately equal (continuous periodic)
    assert_close(
        float(w[0]), float(w[-1]), 5e-3, "cosine_periodic wrap not smooth (w[0]~w[-1])"
    )

    # Starts high for direction="down"
    assert_close(float(w[0]), V_MAX, 1e-5, "cosine_periodic_down start != V_MAX")

    print("PASS: test_cosine_periodic_wrap_smooth")


def test_triangle_wrap_smooth():
    ao = FMChirpAOcontrol(AO_CH)
    ao.set_waveform_triangle(direction="up")
    ao.set_timing(sweep_time_s=2e-3, sample_rate=500_000)

    w = ao.build_waveform()
    assert_in_range(w, V_MIN, V_MAX, "triangle_up")

    # Wraparound continuity: triangle should also have w[0]~w[-1] (approximately)
    assert_close(
        float(w[0]), float(w[-1]), 6e-3, "triangle wrap not smooth (w[0]~w[-1])"
    )

    print("PASS: test_triangle_wrap_smooth")


def test_sawtooth_monotonic_and_not_wrap_smooth():
    ao = FMChirpAOcontrol(AO_CH)
    ao.set_waveform_sawtooth(direction="down")
    ao.set_timing(sweep_time_s=3e-3, sample_rate=200_000)

    w = ao.build_waveform()
    assert_in_range(w, V_MIN, V_MAX, "sawtooth_down")
    assert_true(is_monotonic_nonincreasing(w), "sawtooth_down not monotonic decreasing")

    # Sawtooth is NOT wraparound-smooth; first and last should differ significantly
    assert_true(
        abs(float(w[0]) - float(w[-1])) > 0.5,
        "sawtooth appears wrap-smooth (unexpected)",
    )

    print("PASS: test_sawtooth_monotonic_and_not_wrap_smooth")


def test_hard_clamp_never_violated_even_weird_params():
    ao = FMChirpAOcontrol(AO_CH)
    ao.set_waveform_cosine(direction="down")
    ao.set_timing(sweep_time_s=1e-5, sample_rate=1_000_000)  # tiny duration

    w = ao.build_waveform()
    assert_in_range(w, V_MIN, V_MAX, "hard_clamp_weird_params")

    print("PASS: test_hard_clamp_never_violated_even_weird_params")


def test_summary_keys_present():
    ao = FMChirpAOcontrol(AO_CH)
    ao.set_waveform_raised_cosine_monotonic(direction="down")
    ao.set_timing(sweep_time_s=3e-3, sample_rate=500_000)
    ao.build_waveform()

    s = ao.summary()
    needed = [
        "ao_chan",
        "trigger_mode",
        "trig_src",
        "run_mode",
        "waveform",
        "direction",
        "sample_rate_Sa_s",
        "sweep_time_s",
        "N",
        "t_step_s",
        "v_wave_min",
        "v_wave_max",
        "state",
    ]
    for k in needed:
        assert_true(k in s, f"summary missing key: {k}")

    print("PASS: test_summary_keys_present")


# ============================================================
# Optional hardware smoke tests
# ============================================================


def hw_test_finite_software_once():
    ao = FMChirpAOcontrol(AO_CH)
    ao.set_software_start()
    ao.set_run_mode_finite()
    ao.set_waveform_raised_cosine_monotonic(direction="down")
    ao.set_timing(sweep_time_s=3e-3, sample_rate=500_000)
    ao.set_prehold(True)

    try:
        ao.start()
        ao.wait_done(timeout_s=5.0)
        print("PASS: hw_test_finite_software_once")
    finally:
        ao.close()


def hw_test_continuous_software_start_stop():
    import time

    ao = FMChirpAOcontrol(AO_CH)
    ao.set_software_start()
    ao.set_run_mode_continuous()
    ao.set_waveform_cosine(direction="down")
    ao.set_timing(sweep_time_s=2e-3, sample_rate=500_000)
    ao.set_prehold(True)

    try:
        ao.start()
        time.sleep(0.25)
        ao.stop()
        print("PASS: hw_test_continuous_software_start_stop")
    finally:
        ao.close()


def hw_test_finite_hardware_trigger_once():
    from nidaqmx.constants import Edge

    ao = FMChirpAOcontrol(AO_CH)
    ao.set_hardware_trigger(TRIG_SRC, edge=Edge.RISING)
    ao.set_run_mode_finite()
    ao.set_waveform_raised_cosine_monotonic(direction="down")
    ao.set_timing(sweep_time_s=3e-3, sample_rate=500_000)
    ao.set_prehold(True)

    try:
        ao.start()
        print(f"ARMED: send TTL to {TRIG_SRC} AFTER this message.")
        ao.wait_done(timeout_s=10.0)
        print("PASS: hw_test_finite_hardware_trigger_once")
    finally:
        ao.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Pure tests
    test_raised_cosine_monotonic_down()
    test_raised_cosine_monotonic_up()
    test_cosine_periodic_wrap_smooth()
    test_triangle_wrap_smooth()
    test_sawtooth_monotonic_and_not_wrap_smooth()
    test_hard_clamp_never_violated_even_weird_params()
    test_summary_keys_present()

    print("\nAll non-hardware tests passed.")

    if RUN_HARDWARE_TESTS:
        print("\nRunning hardware smoke tests...")
        hw_test_finite_software_once()
        hw_test_continuous_software_start_stop()
        # hw_test_finite_hardware_trigger_once()
        print("\nHardware smoke tests completed.")
    else:
        print("\nHardware tests disabled. Set RUN_HARDWARE_TESTS=True to enable.")
