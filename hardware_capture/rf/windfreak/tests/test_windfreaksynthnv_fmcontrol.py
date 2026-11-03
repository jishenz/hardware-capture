"""
Test script for SynthNV Pro FM controls:
  - '<' / '<?' : FM frequency (Hz)
  - '>' / '>?' : FM deviation (Hz)
  - ',' / ',?' : FM repetitions
  - ';' / ';?' : FM type (0=sin, 1=chirp)
  - '/' / '/?' : FM continuous mode (0/1)

Usage:
  python test_fm_controls.py --port COM4
  python test_fm_controls.py --port /dev/ttyACM0

Notes:
- This script snapshots a few settings and restores them on exit.
- It keeps RF output gated OFF (h0) throughout.
"""

from __future__ import annotations

import argparse
import sys
import time

import pytest

pytest.importorskip("serial")

from hardware_capture.config import WF_NV_PORT
# Adjust import if your file/module name differs.
# If windfreaksynthnv.py is on your PYTHONPATH, this works:
from hardware_capture.rf.windfreak.windfreaksynthnv import SynthNVPro

pytestmark = pytest.mark.hardware
PORT = WF_NV_PORT


def approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--port",
        default=PORT,
        help="Serial port, e.g. COM4 or /dev/ttyACM0 (defaults to hardware_capture.config.WF_NV_PORT)",
    )
    ap.add_argument("--timeout", type=float, default=0.9, help="Read timeout (s)")
    ap.add_argument(
        "--cmd-delay", type=float, default=0.01, help="Delay after each write (s)"
    )
    args = ap.parse_args(argv)
    if not args.port:
        ap.error("provide --port or set hardware_capture.config.WF_NV_PORT")

    dev = SynthNVPro(
        args.port,
        timeout_s=args.timeout,
        command_delay_s=args.cmd_delay,
        query_retries=2,
    )

    # Best-effort: keep output muted
    dev.safe_mute()

    # Snapshot (extend if you want)
    snapshot = {}
    try:
        snapshot["lt"] = dev.get_lt()
    except Exception:
        snapshot["lt"] = None
    try:
        snapshot["gt"] = dev.get_gt()
    except Exception:
        snapshot["gt"] = None
    try:
        snapshot["comma"] = dev.get_comma()
    except Exception:
        snapshot["comma"] = None
    try:
        snapshot["semicolon"] = dev.get_semicolon()
    except Exception:
        snapshot["semicolon"] = None
    try:
        snapshot["slash"] = dev.get_slash()
    except Exception:
        snapshot["slash"] = None

    print("=== Connected ===")
    try:
        print("Identity:", dev.get_identity())
    except Exception as e:
        print("Identity read failed:", repr(e))

    print("\n=== FM controls test ===")
    try:
        # Put FM in a known non-running state first
        dev.set_slash(0)  # continuous off
        time.sleep(0.03)

        # 1) FM type
        for typ in (0, 1):
            dev.set_semicolon(typ)
            time.sleep(0.02)
            got = dev.get_semicolon()
            print(f"; set {typ}  get {got}")
            assert got == typ

        # 2) FM frequency (Hz) (<?)
        # Device formatting is .3 Hz. Allow small tolerance.
        test_lt = 123.456
        dev.set_lt(test_lt)
        time.sleep(0.02)
        got_lt = dev.get_lt()
        print(f"< set {test_lt:.3f} Hz  get {got_lt:.3f} Hz")
        assert approx_equal(got_lt, test_lt, tol=1.0)

        # 3) FM deviation (Hz) (>?)
        test_gt = 2500.000
        dev.set_gt(test_gt)
        time.sleep(0.02)
        got_gt = dev.get_gt()
        print(f"> set {test_gt:.3f} Hz  get {got_gt:.3f} Hz")
        assert approx_equal(got_gt, test_gt, tol=0.01)

        # 4) FM repetitions (,?)
        test_reps = 17
        dev.set_comma(test_reps)
        time.sleep(0.02)
        got_reps = dev.get_comma()
        print(f", set {test_reps}  get {got_reps}")
        assert got_reps == test_reps

        # 5) FM continuous mode (/?)
        for on in (0, 1, 0):
            dev.set_slash(on)
            time.sleep(0.02)
            got_on = dev.get_slash()
            print(f"/ set {on}  get {got_on}")
            assert got_on == on

        print("\nPASS: FM controls basic set/get.")

        # 6) Negative tests (argument validation in your wrappers)
        print("\n=== Negative tests (expected failures) ===")
        try:
            dev.set_semicolon(2)
            raise AssertionError("set_semicolon(2) should have raised ValueError")
        except ValueError:
            print("OK: set_semicolon(2) raised ValueError")

        try:
            dev.set_slash(2)
            raise AssertionError("set_slash(2) should have raised ValueError")
        except ValueError:
            print("OK: set_slash(2) raised ValueError")

        print("\nAll tests completed successfully.")

    except Exception as e:
        print("\nFAIL:", repr(e))
        return 2

    finally:
        # Always stop FM continuous + mute output
        try:
            dev.set_slash(0)
        except Exception:
            pass
        dev.safe_mute()

        # Restore what we changed (best-effort)
        try:
            if snapshot.get("semicolon") is not None:
                dev.set_semicolon(int(snapshot["semicolon"]))
            if snapshot.get("lt") is not None:
                dev.set_lt(float(snapshot["lt"]))
            if snapshot.get("gt") is not None:
                dev.set_gt(float(snapshot["gt"]))
            if snapshot.get("comma") is not None:
                dev.set_comma(int(snapshot["comma"]))
            if snapshot.get("slash") is not None:
                dev.set_slash(int(snapshot["slash"]))
        except Exception as e:
            print("Restore failed (non-fatal):", repr(e))

        try:
            dev.close()
        except Exception:
            pass

    return 0


def test_fmcontrol() -> None:
    assert main(["--port", PORT]) == 0


if __name__ == "__main__":
    sys.exit(main())
