"""
UNIT TEST 2 — Power only (set_W / get_W)

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Does NOT enable RF output.
- Restores original power.

Run:
    python ut2_power_only.py
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
TEST_POWERS_DBM = [-30.000, -20.000, -10.000]  # keep conservative
TOL_DBM = 0.02


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print(f"[UT2] Power-only on {PORT}")
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

        orig = dev.get_W()
        print(f"[INFO] Original W = {orig:.3f} dBm")

        for p in TEST_POWERS_DBM:
            dev.set_W(p)
            time.sleep(0.05)
            rb = dev.get_W()
            print(f"  set {p:.3f} -> got {rb:.3f}")
            assert_true(approx(rb, p, TOL_DBM), f"Mismatch: set {p:.3f}, got {rb:.3f}")

        print("[PASS] UT2 power wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT2 failed")
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
                dev.set_W(orig)
        except Exception:
            pass
        try:
            dev.close()
        except Exception:
            pass


def test_power() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
