"""
UNIT TEST 13 — Convenience helpers only

Covers ONLY:
- get_identity()
- safe_mute()
- set_reference(mode, ext_ref_mhz=...)

Safety:
- Starts with RAW "h0" as baseline
- safe_mute() is allowed to be tested (it should not raise)
- set_reference is tested but does not touch RF frequency/power (beyond ref chain)
- Restores original x and * at the end (best-effort)

Run:
    python ut13_convenience_helpers.py
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
SETTLE_S = 0.06
TOL_MHZ = 0.02


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print(f"[UT13] Convenience helpers on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=1.4, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    orig_x = None
    orig_star = None

    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        # Snapshot original reference settings using wrappers (not the convenience function)
        orig_x = dev.get_x()
        orig_star = dev.get_star()
        print(f"[INFO] Original ref: x={orig_x}, *={orig_star:.6f} MHz")

        # -------------------------
        # 1) safe_mute()
        # -------------------------
        dev.safe_mute()
        print("[OK] safe_mute() did not raise")

        # -------------------------
        # 2) get_identity()
        # -------------------------
        ident = dev.get_identity()
        print(f"[INFO] get_identity() -> {ident}")

        # Accept either dataclass-like attributes or dict keys
        if isinstance(ident, dict):
            for k in (
                "model_type",
                "serial_number",
                "firmware_version",
                "hardware_version",
            ):
                assert_true(
                    k in ident and str(ident[k]), f"identity missing/empty: {k}"
                )
        else:
            # dataclass or object
            for attr in (
                "model_type",
                "serial_number",
                "firmware_version",
                "hardware_version",
            ):
                assert_true(hasattr(ident, attr), f"identity missing attribute: {attr}")
                assert_true(
                    str(getattr(ident, attr)), f"identity attribute empty: {attr}"
                )

        print("[OK] get_identity() fields look populated")

        # -------------------------
        # 3) set_reference()
        # -------------------------
        # External 10 MHz
        dev.set_reference("ext", ext_ref_mhz=10.0)
        time.sleep(SETTLE_S)
        x = dev.get_x()
        star = dev.get_star()
        print(f"[INFO] After set_reference('ext',10): x={x}, *={star:.6f}")
        assert_true(x == 0, f"Expected x=0 for ext, got {x}")
        assert_true(approx(star, 10.0, TOL_MHZ), f"Expected *=10.0 MHz, got {star}")

        # Internal 27 MHz
        dev.set_reference("int27")
        time.sleep(SETTLE_S)
        x = dev.get_x()
        star = dev.get_star()
        print(f"[INFO] After set_reference('int27'): x={x}, *={star:.6f}")
        assert_true(x == 1, f"Expected x=1 for int27, got {x}")
        assert_true(approx(star, 27.0, TOL_MHZ), f"Expected *=27.0 MHz, got {star}")

        # Internal 10 MHz
        dev.set_reference("int10")
        time.sleep(SETTLE_S)
        x = dev.get_x()
        star = dev.get_star()
        print(f"[INFO] After set_reference('int10'): x={x}, *={star:.6f}")
        assert_true(x == 2, f"Expected x=2 for int10, got {x}")
        assert_true(approx(star, 10.0, TOL_MHZ), f"Expected *=10.0 MHz, got {star}")

        print("[PASS] UT13 convenience helpers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT13 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        # Restore original reference settings
        try:
            if orig_x is not None:
                dev.set_x(int(orig_x))
                time.sleep(0.04)
        except Exception:
            pass
        try:
            if orig_star is not None:
                dev.set_star(float(orig_star))
                time.sleep(0.04)
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


def test_helper() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
