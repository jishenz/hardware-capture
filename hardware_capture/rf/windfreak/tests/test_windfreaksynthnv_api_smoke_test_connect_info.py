"""
API Smoke Test 1: connect + info/status helpers (safe)

Run:
    python api_smoke_test_connect_info.py
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
    print("[API1] connect/info/status")
    try:
        dev = SynthNVPro(PORT, timeout_s=1.2, command_delay_s=0.01, query_retries=2)
        dev.safe_mute()

        info = (
            dev.get_device_info()
            if hasattr(dev, "get_device_info")
            else {
                "model": dev.get_plus(),
                "serial": dev.get_minus(),
                "fw": dev.get_v(0),
                "hw": dev.get_v(1),
            }
        )
        print("Device info:", info)

        ref = dev.get_reference_clock()
        print("Ref clock:", ref)

        st = dev.get_status() if hasattr(dev, "get_status") else dev.snapshot_state()
        print("Status keys:", sorted(list(st.keys())))

        print("[PASS] API1")
        return 0
    except Exception as e:
        print("[FAIL] API1", type(e).__name__, e)
        print(traceback.format_exc())
        return 1
    finally:
        try:
            dev.safe_mute()
        except Exception:
            pass
        try:
            dev.close()
        except Exception:
            pass


def test_api_smoke_test_connect_info() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
