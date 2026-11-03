"""
UNIT TEST 10 — Channel spacing only (set_i / get_i)

Covers ONLY:
- set_i(hz) / get_i()

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Does not touch frequency/power/reference selection.
- Restores original i at the end.

Notes:
- Very small spacing can slow USB responsiveness; we use a small set of values and
  allow generous readback tolerance and extra settle time for small spacings.

Run:
    python ut10_channel_spacing_only.py
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

# Choose representative values within 0.1..1000 Hz
TEST_I_HZ = [1000.0, 100.0, 10.0, 1.0]  # avoid 0.1 unless you really want it
TOL_HZ = 2.0  # readback tolerance (Hz)

# Delays: add more for small spacings
SETTLE_BIG_S = 0.03
SETTLE_SMALL_S = 0.10


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    print(f"[UT10] Channel spacing i-only on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=1.5, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    orig_i = None

    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        # Snapshot original i
        orig_i = dev.get_i()
        print(f"[INFO] Original i = {orig_i:.3f} Hz")

        for val in TEST_I_HZ:
            dev.set_i(val)
            time.sleep(SETTLE_SMALL_S if val < 10 else SETTLE_BIG_S)

            rb = dev.get_i()
            print(f"  set_i({val:.3f}) -> get_i() = {rb:.3f}")

            assert_true(
                abs(rb - val) <= TOL_HZ,
                f"i mismatch: set {val:.3f} Hz, got {rb:.3f} Hz",
            )

        print("[PASS] UT10 channel spacing wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT10 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Restore original i
        try:
            if orig_i is not None:
                dev.set_i(float(orig_i))
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


def test_chspacing() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
