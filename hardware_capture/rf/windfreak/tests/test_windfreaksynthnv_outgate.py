"""
UNIT TEST 3 — Output gate only (set_h / get_h)

Safety:
- Starts by forcing RF low using RAW command "h0".
- Then tests wrapper toggles set_h(0/1/0) with brief dwell.
- Restores original h at end (by default keeps muted unless originally high).

Run:
    python ut3_gate_only.py
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
    print(f"[UT3] Output gate-only on {PORT}")
    try:
        dev = SynthNVPro(PORT, timeout_s=0.9, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(e)
        print(traceback.format_exc())
        return 1

    orig_h = None
    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        orig_h = dev.get_h()
        print(f"[INFO] Original h = {orig_h}")

        # Ensure we can set/read 0
        dev.set_h(0)
        time.sleep(0.03)
        rb0 = dev.get_h()
        print(f"  set_h(0) -> {rb0}")
        assert_true(rb0 == 0, f"Expected h=0, got {rb0}")

        # Briefly set 1 then back to 0
        dev.set_h(1)
        time.sleep(0.05)
        rb1 = dev.get_h()
        print(f"  set_h(1) -> {rb1}")
        assert_true(rb1 == 1, f"Expected h=1, got {rb1}")

        dev.set_h(0)
        time.sleep(0.03)
        rb0b = dev.get_h()
        print(f"  set_h(0) -> {rb0b}")
        assert_true(rb0b == 0, f"Expected h=0, got {rb0b}")

        print("[PASS] UT3 gate wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT3 failed")
        print(e)
        print(traceback.format_exc())
        return 2

    finally:
        # Default: mute at end
        try:
            dev.set_h(0)
        except Exception:
            try:
                dev.write_raw("h0")
            except Exception:
                pass

        # If you truly want to restore original gate state, uncomment:
        # try:
        #     if orig_h == 1:
        #         dev.set_h(1)
        # except Exception:
        #     pass

        try:
            dev.close()
        except Exception:
            pass


def test_outgate() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
