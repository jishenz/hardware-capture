"""
UNIT TEST 7 — PLL enable (E) + charge pump current (U) ONLY

Covers ONLY:
- set_E / get_E    (PLL chip enable)
- set_U / get_U    (charge pump current, allowed 1..15)

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Does NOT touch frequency/power/reference.
- Restores original E and U at the end (best-effort).
- Adds required settle delay after enabling PLL.

Run:
    python ut7_pll_enable_charge_pump.py
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

# Valid U range per programming guide: 1..15 (avoid 0)
U_VALUES = [1, 3, 8, 15]

# Delays
BOOT_DELAY_S = 0.05  # doc says ~20 ms; use 50 ms margin
SETTLE_S = 0.03


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    print(f"[UT7] PLL enable (E) + charge pump (U) on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=1.2, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    orig_E = None
    orig_U = None

    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        # Snapshot original states (wrappers under test)
        orig_E = dev.get_E()
        orig_U = dev.get_U()
        print(f"[INFO] Original: E={orig_E}, U={orig_U}")

        # --- Test E toggling ---
        # Turn PLL off
        dev.set_E(0)
        time.sleep(SETTLE_S)
        rb0 = dev.get_E()
        print(f"  set_E(0) -> get_E() = {rb0}")
        assert_true(rb0 == 0, f"E mismatch: expected 0 got {rb0}")

        # Turn PLL on (needs boot delay)
        dev.set_E(1)
        time.sleep(BOOT_DELAY_S)
        rb1 = dev.get_E()
        print(f"  set_E(1) -> get_E() = {rb1}")
        assert_true(rb1 == 1, f"E mismatch: expected 1 got {rb1}")

        # --- Test U values (only valid 1..15) ---
        for u in U_VALUES:
            dev.set_U(u)
            time.sleep(SETTLE_S)
            rbu = dev.get_U()
            print(f"  set_U({u}) -> get_U() = {rbu}")
            assert_true(rbu == u, f"U mismatch: set {u} got {rbu}")

        print("[PASS] UT7 PLL enable + charge pump wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT7 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Restore original U then E (best effort)
        try:
            if orig_U is not None:
                dev.set_U(int(orig_U))
                time.sleep(SETTLE_S)
        except Exception:
            pass

        try:
            if orig_E is not None:
                dev.set_E(int(orig_E))
                if int(orig_E) == 1:
                    time.sleep(BOOT_DELAY_S)
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


def test_pll() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
