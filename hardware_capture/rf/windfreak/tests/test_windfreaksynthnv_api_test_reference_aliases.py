"""
API Test 3: reference aliases (safe)

Run:
    python api_test_reference_aliases.py
"""

import sys
import time
import traceback

import pytest

pytest.importorskip("serial")

from hardware_capture.config import WF_NV_PORT
from hardware_capture.rf.windfreak.windfreaksynthnv import SynthNVPro

PORT = WF_NV_PORT
pytestmark = pytest.mark.hardware
TOL_MHZ = 0.05


def assert_true(c, m):
    if not c:
        raise AssertionError(m)


def main():
    print("[API3] reference aliases")
    dev = None
    try:
        dev = SynthNVPro(PORT, timeout_s=1.4, command_delay_s=0.01, query_retries=2)
        dev.safe_mute()

        orig = dev.get_reference_clock()
        print("orig:", orig)

        dev.set_reference_clock("ext", 10.0)
        time.sleep(0.1)
        rb = dev.get_reference_clock()
        print("ext:", rb)
        assert_true(rb["x"] == 0, "expected x=0")
        assert_true(abs(rb["ref_mhz"] - 10.0) <= TOL_MHZ, "expected *~10.0")

        dev.set_reference_clock("int27")
        time.sleep(0.1)
        rb = dev.get_reference_clock()
        print("int27:", rb)
        assert_true(rb["x"] == 1, "expected x=1")
        assert_true(abs(rb["ref_mhz"] - 27.0) <= TOL_MHZ, "expected *~27.0")

        dev.set_reference_clock("int10")
        time.sleep(0.1)
        rb = dev.get_reference_clock()
        print("int10:", rb)
        assert_true(rb["x"] == 2, "expected x=2")
        assert_true(abs(rb["ref_mhz"] - 10.0) <= TOL_MHZ, "expected *~10.0")

        # restore
        if "x" in orig and "ref_mhz" in orig:
            dev.set_ref_source(int(orig["x"]))
            dev.set_ref_frequency(float(orig["ref_mhz"]))
        dev.safe_mute()

        print("[PASS] API3")
        return 0
    except Exception as e:
        print("[FAIL] API3", type(e).__name__, e)
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


def test_api_reference_aliases() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
