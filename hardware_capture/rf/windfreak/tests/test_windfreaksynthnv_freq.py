"""
UNIT TEST 1 — Frequency only (set_f / get_f)

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Restores original frequency.

Run:
    python ut1_freq_only.py
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
TEST_FREQS_MHZ = [2870.00000000, 2870.10000000, 2869.90000000]
TOL_MHZ = 1e-4  # 0.1 kHz


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print(f"[UT1] Frequency-only on {PORT}")
    try:
        dev = SynthNVPro(PORT, timeout_s=0.9, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(e)
        print(traceback.format_exc())
        return 1

    orig = None
    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        orig = dev.get_f()
        print(f"[INFO] Original f = {orig:.8f} MHz")

        for f in TEST_FREQS_MHZ:
            dev.set_f(f)
            time.sleep(0.05)
            rb = dev.get_f()
            print(f"  set {f:.8f} -> got {rb:.8f}")
            assert_true(approx(rb, f, TOL_MHZ), f"Mismatch: set {f:.8f}, got {rb:.8f}")

        print("[PASS] UT1 frequency wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT1 failed")
        print(e)
        print(traceback.format_exc())
        return 2

    finally:
        try:
            dev.write_raw("h0")
        except Exception:
            pass
        try:
            if orig is not None:
                dev.set_f(orig)
        except Exception:
            pass
        try:
            dev.close()
        except Exception:
            pass


def test_freq() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
