"""
UNIT TEST 8 — Reference doubler only (set_D / get_D)

Covers ONLY:
- set_D(on) / get_D()

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Does not touch frequency/power/reference selection.
- Restores original D at the end.

Run:
    python ut8_ref_doubler_only.py
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
SETTLE_S = 0.03


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    print(f"[UT8] Ref doubler (D) only on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=1.0, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    orig_D = None

    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        # Snapshot original D
        orig_D = dev.get_D()
        print(f"[INFO] Original D = {orig_D}")

        # Toggle off -> on -> off
        dev.set_D(0)
        time.sleep(SETTLE_S)
        rb0 = dev.get_D()
        print(f"  set_D(0) -> get_D() = {rb0}")
        assert_true(rb0 == 0, f"D mismatch: expected 0 got {rb0}")

        dev.set_D(1)
        time.sleep(SETTLE_S)
        rb1 = dev.get_D()
        print(f"  set_D(1) -> get_D() = {rb1}")
        assert_true(rb1 == 1, f"D mismatch: expected 1 got {rb1}")

        dev.set_D(0)
        time.sleep(SETTLE_S)
        rb0b = dev.get_D()
        print(f"  set_D(0) -> get_D() = {rb0b}")
        assert_true(rb0b == 0, f"D mismatch: expected 0 got {rb0b}")

        print("[PASS] UT8 ref doubler wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT8 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Restore original D
        try:
            if orig_D is not None:
                dev.set_D(int(orig_D))
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


def test_doubler() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
