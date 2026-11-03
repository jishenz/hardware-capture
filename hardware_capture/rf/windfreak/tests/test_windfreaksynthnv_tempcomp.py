"""
UNIT TEST 4 — Temperature compensation only (set_Z / get_Z)

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Only tests Z (no frequency/power/gate wrappers beyond the raw safety baseline).
- Restores original Z mode at the end.

Run:
    python ut4_tempcomp_only.py
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

# Valid Z modes per API
Z_MODES = [0, 1, 2, 3]


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    print(f"[UT4] Temperature compensation Z-only on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=0.9, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    orig_z = None
    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        # Snapshot original Z
        orig_z = dev.get_Z()
        print(f"[INFO] Original Z = {orig_z}")

        # Cycle through modes and verify readback
        for z in Z_MODES:
            dev.set_Z(z)
            time.sleep(0.05)  # allow internal update
            rb = dev.get_Z()
            print(f"  set_Z({z}) -> get_Z() = {rb}")
            assert_true(rb == z, f"Z mismatch: set {z}, got {rb}")

        print("[PASS] UT4 temperature compensation wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT4 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Restore original Z
        try:
            if orig_z is not None:
                dev.set_Z(orig_z)
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


def test_tempcomp() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
