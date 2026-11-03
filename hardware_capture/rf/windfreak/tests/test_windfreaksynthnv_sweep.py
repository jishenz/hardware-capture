"""
UNIT TEST 11 — Sweep control ONLY (l/u/s/t/[ ]/^/X/r/d/g/c/y)

Goal:
- Verify sweep *parameter* wrappers work (set/get round-trip) without doing any long sweep.
- Optionally start/stop sweep briefly with RF muted (still only validating g/c behavior).

Covers ONLY these wrappers:
- set_l / get_l         (lower MHz)
- set_u / get_u         (upper MHz)
- set_s / get_s         (step; interpretation depends on X, but we only round-trip)
- set_t / get_t         (step time ms)
- set_left_bracket / get_left_bracket   ([ low amp dBm)
- set_right_bracket / get_right_bracket (] high amp dBm)
- set_caret / get_caret (^ direction)
- set_X / get_X         (type)
- set_r / get_r         (reading while sweeping)
- set_d / get_d         (display)
- set_c / get_c         (continuous)
- set_y / get_y         (trigger mode)
- set_g / get_g         (run)

Safety:
- Forces RF low using RAW command "h0" (not wrapper).
- Keeps g=0 except for an optional brief start/stop (default enabled but very short).
- Restores original sweep params at end (best-effort).

Run:
    python ut11_sweep_control_only.py
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

# Minimal, benign sweep setup (centered around common NV freq)
L_MHZ = 2869.0
U_MHZ = 2871.0
S_STEP = 0.50000000  # units depend on X; for X=0 linear MHz step
T_MS = 5.0

AMP_LOW_DBM = -30.0
AMP_HIGH_DBM = -20.0

DIR_UP = 1
SWEEP_TYPE = 0  # 0=linear
READ_WHILE_SWEEP = 0
DISPLAY_MODE = 0
CONTINUOUS = 0
TRIG_MODE = 0

# Optional: start/stop sweep very briefly
DO_BRIEF_RUN = True
BRIEF_RUN_S = 0.05

# Tolerances
TOL_MHZ = 1e-4
TOL_DBM = 0.05
TOL_MS = 0.05


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print(f"[UT11] Sweep control only on {PORT}")

    try:
        dev = SynthNVPro(PORT, timeout_s=1.5, command_delay_s=0.01, query_retries=2)
    except Exception as e:
        print("[FAIL] Connect")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    # Snapshot originals (best-effort)
    orig = {}

    def snap(name: str, fn):
        try:
            orig[name] = fn()
        except Exception:
            orig[name] = None

    def restore():
        # Always stop sweep at end
        try:
            dev.set_g(0)
        except Exception:
            pass

        # Restore params in a safe order
        for k, setter in [
            ("X", lambda v: dev.set_X(int(v))),
            ("l", lambda v: dev.set_l(float(v))),
            ("u", lambda v: dev.set_u(float(v))),
            ("s", lambda v: dev.set_s(float(v))),
            ("t", lambda v: dev.set_t(float(v))),
            ("[", lambda v: dev.set_left_bracket(float(v))),
            ("]", lambda v: dev.set_right_bracket(float(v))),
            ("^", lambda v: dev.set_caret(int(v))),
            ("r", lambda v: dev.set_r(int(v))),
            ("d", lambda v: dev.set_d(int(v))),
            ("c", lambda v: dev.set_c(int(v))),
            ("y", lambda v: dev.set_y(int(v))),
        ]:
            v = orig.get(k, None)
            if v is None:
                continue
            try:
                setter(v)
                time.sleep(0.02)
            except Exception:
                pass

    try:
        # Safety baseline (raw)
        dev.write_raw("h0")
        time.sleep(0.05)

        # Snapshot original sweep params (wrappers under test)
        snap("l", dev.get_l)
        snap("u", dev.get_u)
        snap("s", dev.get_s)
        snap("t", dev.get_t)
        snap("[", dev.get_left_bracket)
        snap("]", dev.get_right_bracket)
        snap("^", dev.get_caret)
        snap("X", dev.get_X)
        snap("r", dev.get_r)
        snap("d", dev.get_d)
        snap("c", dev.get_c)
        snap("y", dev.get_y)
        snap("g", dev.get_g)

        print("[INFO] Snapshotted original sweep params (best-effort).")

        # Ensure sweep not running
        dev.set_g(0)
        time.sleep(0.03)
        assert_true(dev.get_g() == 0, "Expected g=0 at start")

        # Set + verify parameters
        dev.set_X(SWEEP_TYPE)
        time.sleep(0.03)
        assert_true(dev.get_X() == SWEEP_TYPE, "X mismatch")

        dev.set_l(L_MHZ)
        time.sleep(0.03)
        assert_true(approx(dev.get_l(), L_MHZ, TOL_MHZ), "l mismatch")

        dev.set_u(U_MHZ)
        time.sleep(0.03)
        assert_true(approx(dev.get_u(), U_MHZ, TOL_MHZ), "u mismatch")

        dev.set_s(S_STEP)
        time.sleep(0.03)
        # s meaning depends on X; round-trip should be close
        assert_true(approx(dev.get_s(), S_STEP, 1e-3), "s mismatch")

        dev.set_t(T_MS)
        time.sleep(0.03)
        assert_true(approx(dev.get_t(), T_MS, TOL_MS), "t mismatch")

        dev.set_left_bracket(AMP_LOW_DBM)
        time.sleep(0.03)
        assert_true(approx(dev.get_left_bracket(), AMP_LOW_DBM, TOL_DBM), "[ mismatch")

        dev.set_right_bracket(AMP_HIGH_DBM)
        time.sleep(0.03)
        assert_true(
            approx(dev.get_right_bracket(), AMP_HIGH_DBM, TOL_DBM), "] mismatch"
        )

        dev.set_caret(DIR_UP)
        time.sleep(0.03)
        assert_true(dev.get_caret() == DIR_UP, "^ mismatch")

        dev.set_r(READ_WHILE_SWEEP)
        time.sleep(0.03)
        assert_true(dev.get_r() == READ_WHILE_SWEEP, "r mismatch")

        dev.set_d(DISPLAY_MODE)
        time.sleep(0.03)
        assert_true(dev.get_d() == DISPLAY_MODE, "d mismatch")

        dev.set_c(CONTINUOUS)
        time.sleep(0.03)
        assert_true(dev.get_c() == CONTINUOUS, "c mismatch")

        dev.set_y(TRIG_MODE)
        time.sleep(0.03)
        assert_true(dev.get_y() == TRIG_MODE, "y mismatch")

        print("[OK] Sweep parameter set/get round-trips look good.")

        # Optional: brief start/stop (still with RF muted)
        if DO_BRIEF_RUN:
            dev.set_g(1)
            time.sleep(BRIEF_RUN_S)
            # We do NOT assert get_g()==1 (some firmware updates g? behavior); we just stop cleanly.
            dev.set_g(0)
            time.sleep(0.05)
            assert_true(dev.get_g() == 0, "Expected g=0 after stopping sweep")
            print("[OK] Brief sweep run/stop executed (RF muted).")

        print("[PASS] UT11 sweep control wrappers OK")
        return 0

    except Exception as e:
        print("[FAIL] UT11 failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    finally:
        try:
            restore()
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


def test_sweep() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
