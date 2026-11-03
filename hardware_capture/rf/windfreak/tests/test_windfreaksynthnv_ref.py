"""
UNIT TEST 9 — Reference select + reference frequency ONLY (set_x/get_x and set_star/get_star)

Covers ONLY:
- set_x(ref_sel) / get_x()
- set_star(mhz)  / get_star()

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Does NOT touch RF frequency/power/output gate (beyond raw safety baseline).
- Restores original x and * at the end.

Important notes:
- On some firmwares, '*' is meaningful primarily when x=0 (external ref). We still test
  readback for x=1/2, but don't over-interpret the hardware meaning—only that the device
  accepts the command and reports a sensible value.

Run:
    python ut9_ref_select_freq_only.py
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

# Common reference frequencies (MHz)
REF_EXT_MHZ = 10.0
REF_INT27_MHZ = 27.0
REF_INT10_MHZ = 10.0

# Delays
SETTLE_S = 0.05

# Tolerances for '*' readback (MHz)
TOL_REF_MHZ = 0.02


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print(f"[UT9] Reference select x + reference frequency * on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=1.2, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    orig_x = None
    orig_star = None

    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        # Snapshot original settings
        orig_x = dev.get_x()
        orig_star = dev.get_star()
        print(f"[INFO] Original: x={orig_x}, *={orig_star:.6f} MHz")

        # 1) External ref (x=0) and set * to 10 MHz
        dev.set_x(0)
        time.sleep(SETTLE_S)
        rb_x0 = dev.get_x()
        assert_true(rb_x0 == 0, f"x mismatch: expected 0 got {rb_x0}")
        print("  set_x(0) -> OK")

        dev.set_star(REF_EXT_MHZ)
        time.sleep(SETTLE_S)
        rb_star0 = dev.get_star()
        print(f"  set_star({REF_EXT_MHZ}) -> get_star()={rb_star0:.6f}")
        assert_true(
            approx(rb_star0, REF_EXT_MHZ, TOL_REF_MHZ),
            f"* mismatch: expected ~{REF_EXT_MHZ}, got {rb_star0}",
        )

        # 2) Internal 27 MHz (x=1), set * to 27 for consistency
        dev.set_x(1)
        time.sleep(SETTLE_S)
        rb_x1 = dev.get_x()
        assert_true(rb_x1 == 1, f"x mismatch: expected 1 got {rb_x1}")
        print("  set_x(1) -> OK")

        dev.set_star(REF_INT27_MHZ)
        time.sleep(SETTLE_S)
        rb_star1 = dev.get_star()
        print(f"  set_star({REF_INT27_MHZ}) -> get_star()={rb_star1:.6f}")
        assert_true(
            approx(rb_star1, REF_INT27_MHZ, TOL_REF_MHZ),
            f"* mismatch: expected ~{REF_INT27_MHZ}, got {rb_star1}",
        )

        # 3) Internal 10 MHz (x=2), set * to 10 for consistency
        dev.set_x(2)
        time.sleep(SETTLE_S)
        rb_x2 = dev.get_x()
        assert_true(rb_x2 == 2, f"x mismatch: expected 2 got {rb_x2}")
        print("  set_x(2) -> OK")

        dev.set_star(REF_INT10_MHZ)
        time.sleep(SETTLE_S)
        rb_star2 = dev.get_star()
        print(f"  set_star({REF_INT10_MHZ}) -> get_star()={rb_star2:.6f}")
        assert_true(
            approx(rb_star2, REF_INT10_MHZ, TOL_REF_MHZ),
            f"* mismatch: expected ~{REF_INT10_MHZ}, got {rb_star2}",
        )

        print("[PASS] UT9 reference select + reference frequency wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT9 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Restore original x and *
        try:
            if orig_x is not None:
                dev.set_x(int(orig_x))
                time.sleep(SETTLE_S)
        except Exception:
            pass

        try:
            if orig_star is not None:
                dev.set_star(float(orig_star))
                time.sleep(SETTLE_S)
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


def test_ref() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
