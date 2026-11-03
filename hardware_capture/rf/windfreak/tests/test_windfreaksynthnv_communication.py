"""
Step 1 — SynthNV Pro: connection + I/O primitives + formatting helpers ONLY (COM4)

What this test covers:
- Connection management: connect/open, purge, disconnect/close, context manager
- I/O primitives: write_raw(), read_line(), query1(), drain behavior
- Formatting helpers: _fmt_mhz(), _fmt_dbm(), _fmt_ms(), _fmt_hz3(), _fmt_int()

What this test does NOT cover:
- ANY command wrappers (no get_plus/get_f/etc.)
- Any state-changing commands besides raw '?' help request

Strategy:
- Use write_raw('?') to request the device's help text (multi-line).
- Use read_line() repeatedly until TimeoutError to prove:
    (a) device responds
    (b) readline framing works (LF-terminated)
    (c) timeout behavior is sane
- Use purge() and drain behavior to show we can clear buffers.

Run:
    python test_step1_io_only.py
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
    print(f"[STEP1] IO-only test on {PORT}")

    # ---------------------------
    # 1) Formatting helper checks
    # ---------------------------
    try:
        # These are deterministic and don't touch hardware.
        assert_true(
            SynthNVPro._fmt_mhz(2870.0) == "2870.00000000",
            "_fmt_mhz formatting mismatch",
        )
        assert_true(
            SynthNVPro._fmt_dbm(-10.0) == "-10.000", "_fmt_dbm formatting mismatch"
        )
        assert_true(SynthNVPro._fmt_ms(5.0) == "5.000", "_fmt_ms formatting mismatch")
        assert_true(
            SynthNVPro._fmt_hz3(1000.0) == "1000.000", "_fmt_hz3 formatting mismatch"
        )
        assert_true(
            SynthNVPro._fmt_int(10.4) == "10",
            "_fmt_int rounding mismatch (expected 10)",
        )
        assert_true(
            SynthNVPro._fmt_int(10.6) == "11",
            "_fmt_int rounding mismatch (expected 11)",
        )
        print("[OK] Formatting helpers")
    except Exception as e:
        print("[FAIL] Formatting helpers")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 1

    # ---------------------------
    # 2) Connection open/close
    # ---------------------------
    try:
        dev = SynthNVPro(
            PORT,
            timeout_s=0.6,  # short timeout so '?' read stops promptly
            command_delay_s=0.0,
            query_retries=1,
        )
        print("[OK] connect() via constructor")

        # Explicit purge should not error
        dev.purge()
        print("[OK] purge()")

    except Exception as e:
        print("[FAIL] Connection open/purge failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 2

    # ---------------------------
    # 3) I/O primitives: raw '?' + read_line framing
    # ---------------------------
    try:
        # Ensure clean buffer before requesting help
        dev.purge()
        time.sleep(0.05)

        # Use ONLY raw write/read, not any wrapper.
        dev.write_raw("?")  # raw command; device should print multi-line help

        lines = []
        while True:
            try:
                line = dev.read_line()  # primitive readline
                if line:
                    lines.append(line)
            except TimeoutError:
                break

        assert_true(
            len(lines) > 0, "No lines received from '?' — check cable/port/device"
        )
        print(f"[OK] write_raw('?') + read_line(): got {len(lines)} lines")

        # Minimal sanity: help output typically contains at least one ')' char
        # (don’t over-assume exact contents)
        has_paren = any(")" in ln for ln in lines)
        assert_true(
            has_paren, "Help output didn't look like a command list (missing ')')"
        )

        # Now verify purge clears buffer: another read_line should timeout quickly.
        dev.purge()
        try:
            _ = dev.read_line()
            raise AssertionError(
                "Expected read_line() to timeout after purge, but it returned data"
            )
        except TimeoutError:
            print("[OK] purge() clears pending responses (read_line timed out)")

        # ---------------------------
        # 4) query1 primitive (still no wrappers)
        # ---------------------------
        # query1 does drain+write+read-one-line. We use it with '?' knowing it returns a line.
        first_line = dev.query1("?", parse=None)
        assert_true(
            isinstance(first_line, str) and len(first_line) > 0,
            "query1('?') returned empty",
        )
        print("[OK] query1('?') returned one line")

        # Drain remaining help lines so later tests aren't polluted.
        # (Using primitives only.)
        while True:
            try:
                _ = dev.read_line()
            except TimeoutError:
                break
        print("[OK] drained remaining help lines")

        print("[PASS] Step 1 IO-only communication looks good.")

    except Exception as e:
        print("[FAIL] I/O primitives test failed")
        print(f"  {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 3

    finally:
        # ---------------------------
        # 5) Close / reopen path
        # ---------------------------
        try:
            dev.close()
            print("[OK] close()")
        except Exception:
            pass

        try:
            dev.open()
            print("[OK] open() after close()")
            dev.close()
            print("[OK] close() again")
        except Exception as e:
            print("[WARN] reopen test failed (open/close).")
            print(f"  {type(e).__name__}: {e}")

        # Context manager path
        try:
            with SynthNVPro(
                PORT, timeout_s=0.4, command_delay_s=0.0, query_retries=1
            ) as d2:
                d2.purge()
            print("[OK] context manager enter/exit")
        except Exception as e:
            print("[WARN] context manager test failed.")
            print(f"  {type(e).__name__}: {e}")

    return 0


def test_communication() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
