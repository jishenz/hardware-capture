"""
UNIT TEST 12 — Misc / Status / Info only

Covers:
- get_p()          PLL lock status
- get_z()          temperature
- get_v(0/1)       firmware / hardware version
- get_plus()       model
- get_minus()      serial number
- set_b / get_b    speaker feedback mode
- get_help()       '?' multi-line help

Safety:
- Forces RF low using RAW command "h0"
- Restores original b at end

Run:
    python ut12_misc_status_info.py
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


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    print(f"[UT12] Misc / status / info on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=1.2, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    orig_b = None

    try:
        # --------------------
        # Safety baseline
        # --------------------
        dev.write_raw("h0")
        time.sleep(0.05)

        # --------------------
        # Identity / versions
        # --------------------
        model = dev.get_plus()
        serial = dev.get_minus()
        fw = dev.get_v(0)
        hw = dev.get_v(1)

        print("[INFO] Identity")
        print(f"  Model : {model}")
        print(f"  Serial: {serial}")
        print(f"  FW    : {fw}")
        print(f"  HW    : {hw}")

        assert_true(len(model) > 0, "Empty model string")
        assert_true(len(serial) > 0, "Empty serial string")
        assert_true(len(fw) > 0, "Empty firmware version")
        assert_true(len(hw) > 0, "Empty hardware version")

        # --------------------
        # Status reads
        # --------------------
        pll = dev.get_p()
        temp_c = dev.get_z()

        print("[INFO] Status")
        print(f"  PLL lock : {pll}")
        print(f"  Temp (C) : {temp_c:.2f}")

        assert_true(pll in (0, 1), "PLL lock status not 0/1")
        assert_true(-40.0 < temp_c < 100.0, f"Temperature looks implausible: {temp_c}")

        # --------------------
        # Speaker feedback b
        # --------------------
        orig_b = dev.get_b()
        print(f"[INFO] Original b = {orig_b}")

        # We only test simple round-trip; semantics depend on firmware
        for v in (0, 1):
            dev.set_b(v)
            time.sleep(0.03)
            rb = dev.get_b()
            print(f"  set_b({v}) -> get_b() = {rb}")
            assert_true(rb == v, f"b mismatch: set {v}, got {rb}")

        # --------------------
        # Help text '?'
        # --------------------
        help_txt = dev.get_help()
        print(f"[INFO] Help text length = {len(help_txt)} chars")

        assert_true(len(help_txt) > 50, "Help text suspiciously short")
        # very loose sanity check
        assert_true(
            "f" in help_txt or "W" in help_txt,
            "Help text doesn't look like command list",
        )

        print("[PASS] UT12 misc / status / info wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT12 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Restore speaker feedback mode
        try:
            if orig_b is not None:
                dev.set_b(int(orig_b))
                time.sleep(0.03)
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


def test_info() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
