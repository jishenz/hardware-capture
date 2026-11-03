"""
API Test 4: helper APIs (initialize / soft_reset / wait_for_pll_lock)

Run:
    python api_test_soft_reset_wait_lock.py
"""

import sys
import traceback

import pytest

pytest.importorskip("serial")

from hardware_capture.config import WF_NV_PORT
from hardware_capture.rf.windfreak.windfreaksynthnv import SynthNVPro

PORT = WF_NV_PORT
pytestmark = pytest.mark.hardware


def main():
    print("[API4] helper APIs")
    dev = None
    try:
        dev = SynthNVPro(PORT, timeout_s=1.6, command_delay_s=0.01, query_retries=2)
        dev.safe_mute()

        if hasattr(dev, "initialize"):
            dev.initialize(reference="ext", ext_ref_mhz=10.0, pll_on=True, mute=True)
            print("initialize OK")

        if hasattr(dev, "soft_reset"):
            dev.soft_reset(level="light", restore_h=False)
            print("soft_reset(light) OK")

        if hasattr(dev, "wait_for_pll_lock"):
            locked = dev.wait_for_pll_lock(timeout_s=1.0, poll_s=0.05)
            print("wait_for_pll_lock:", locked)

        # (Optional) more aggressive reset:
        # dev.soft_reset(level="pll_cycle", restore_h=False)

        print("[PASS] API4")
        return 0

    except Exception as e:
        print("[FAIL] API4", type(e).__name__, e)
        print(traceback.format_exc())
        return 1
    finally:
        try:
            if dev:
                dev.safe_mute()
        except Exception:
            pass
        try:
            if dev:
                dev.close()
        except Exception:
            pass


def test_api_soft_reset_wait_lock() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
