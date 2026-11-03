"""
API Test 2: frequency/power/gate SAFE ALIASES

Default: DOES NOT enable RF. Flip DO_RF_TOGGLE=True if you want to test gate toggling.

Run:
    python api_test_freq_power_gate_aliases.py
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
DO_RF_TOGGLE = False

TEST_FREQS = [600.0, 200.1]
TEST_POWERS = [-30.0, -31.0]
TOL_MHZ = 1e-4
TOL_DBM = 0.05


def assert_true(c, m):
    if not c:
        raise AssertionError(m)


def main():
    print("[API2] freq/power/gate aliases")
    dev = None
    try:
        dev = SynthNVPro(PORT, timeout_s=1.2, command_delay_s=0.01, query_retries=2)
        dev.safe_mute()

        orig_f = dev.get_freq()
        orig_p = dev.get_power()

        for f in TEST_FREQS:
            dev.set_freq(f)
            time.sleep(0.05)
            rb = dev.get_freq()
            print("f set->get:", f, rb)
            assert_true(abs(rb - f) <= TOL_MHZ, f"freq mismatch {f} vs {rb}")

        for p in TEST_POWERS:
            dev.set_power(p)
            time.sleep(0.05)
            rb = dev.get_power()
            print("p set->get:", p, rb)
            assert_true(abs(rb - p) <= TOL_DBM, f"power mismatch {p} vs {rb}")

        # keep muted unless explicitly allowed
        if DO_RF_TOGGLE:
            dev.enable_output()
            time.sleep(1.0)
            assert_true(dev.is_output_enabled() is True, "gate didn't enable")
            dev.disable_output()
            time.sleep(1.0)
            assert_true(dev.is_output_enabled() is False, "gate didn't disable")
            print("gate toggle OK")
        else:
            print("gate toggle skipped (DO_RF_TOGGLE=False)")

        # restore
        dev.set_freq(orig_f)
        dev.set_power(orig_p)
        dev.safe_mute()

        print("[PASS] API2")
        return 0
    except Exception as e:
        print("[FAIL] API2", type(e).__name__, e)
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


def test_api_freq_power_gate_aliases() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
