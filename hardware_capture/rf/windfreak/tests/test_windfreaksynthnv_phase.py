"""
UNIT TEST 6 — Phase step only (set_tilde)

We intentionally DO NOT test get_tilde (~?) because the programming guide says "~?"
has no real meaning / is not a stable state readback. So this unit test verifies:
- set_tilde(...) executes without error across a few representative values.

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Does not touch frequency/power/reference.

Run:
    python ut6_phase_step_only.py
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

# Representative phase step values (degrees)
PHASE_STEPS_DEG = [0.0, 1.0, 10.0, 90.0, 180.0, 359.9]


def main() -> int:
    print(f"[UT6] Phase step (~) only on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=0.9, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        for deg in PHASE_STEPS_DEG:
            dev.set_tilde(deg)
            time.sleep(0.02)
            print(f"  set_tilde({deg:.4f}) -> OK")

        print("[PASS] UT6 phase step set_tilde() OK")
        return 0

    except Exception as e:
        print("[FAIL] UT6 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Safety baseline again
        try:
            dev.write_raw("h0")
        except Exception:
            pass

        try:
            dev.close()
        except Exception:
            pass


def test_phase() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
