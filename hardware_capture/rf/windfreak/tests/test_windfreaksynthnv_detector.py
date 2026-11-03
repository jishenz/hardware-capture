"""
UNIT TEST 5 — Detector only (set_ampersand / get_ampersand / read_w)

Covers ONLY:
- set_ampersand(mode)   # detector mode: 0=instant, 1=avg, 2=peak
- get_ampersand()
- read_w(samples)       # detector readback in dBm (burst; EOM. terminator for N>1)

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Does not touch frequency/power/reference.
- Restores original detector mode at end.

Run:
    python ut5_detector_only.py
"""

from __future__ import annotations

import sys
import time
import traceback

import pytest

pytest.importorskip("serial")

from hardware_capture.config import WF_NV_PORT
from hardware_capture.rf.windfreak.windfreaksynthnv import SynthNVPro

PORT = WF_NV_PORT
pytestmark = pytest.mark.hardware

DETECTOR_MODES = [0, 1, 2]
SAMPLES_SINGLE = 1
SAMPLES_BURST = 5


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    print(f"[UT5] Detector-only on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=1.2, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    orig_mode = None
    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        # Snapshot original detector mode
        orig_mode = dev.get_ampersand()
        print(f"[INFO] Original detector mode (&) = {orig_mode}")

        # Verify set/get modes
        for m in DETECTOR_MODES:
            dev.set_ampersand(m)
            time.sleep(0.05)
            rb = dev.get_ampersand()
            print(f"  set_&({m}) -> get_&() = {rb}")
            assert_true(rb == m, f"Detector mode mismatch: set {m}, got {rb}")

            # Single sample
            v1 = dev.read_w(SAMPLES_SINGLE)
            assert_true(len(v1) == 1, f"read_w(1) length mismatch: got {len(v1)}")
            assert_true(isinstance(v1[0], float), "read_w(1) did not return float")
            print(f"    read_w(1) = {v1[0]:.2f} dBm")

            # Burst samples
            vb = dev.read_w(SAMPLES_BURST)
            # Some firmware can return fewer values; accept >=1, but report if short.
            assert_true(len(vb) >= 1, "read_w(N) returned empty list")
            if len(vb) != SAMPLES_BURST:
                print(
                    f"    [WARN] read_w({SAMPLES_BURST}) returned {len(vb)} samples (expected {SAMPLES_BURST})"
                )
            print(f"    read_w({SAMPLES_BURST}) = {[round(x, 2) for x in vb]} dBm")

        print("[PASS] UT5 detector wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT5 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Restore original detector mode
        try:
            if orig_mode is not None:
                dev.set_ampersand(int(orig_mode))
                time.sleep(0.05)
        except Exception:
            pass

        # Safety baseline again
        try:
            dev.write_raw("h0")
        except Exception:
            pass

        try:
            dev.close()
        except Exception:
            pass


def test_detector() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
