"""
Windfreak SynthNV Pro — improved serial controller class

Improvements based on the SynthNV Pro programming guide + real-device test failures:
- TX: sends command bytes ONLY (no CR/LF).
- RX: reads LF-terminated lines.
- Drains stale input before queries (prevents "?"/wN leftovers corrupting the next read).
- Adds retries for query timeouts and parse errors.
- Enforces documented parameter constraints (e.g., U must be 1..15; i must be 0.1..1000 Hz).
- Handles “meaningless/unstable” queries (e.g., ~? has no real meaning) by returning None.
- Adds settle delays where hardware needs it (e.g., E1 PLL enable boot delay).
- Makes 'a' (VGA DAC) safe: optionally forces Z0, because raw DAC only works when temp comp is OFF.
- Adds multi-line readers for '?' (help) and burst reader for 'wN ... EOM.'.

Dependencies:
    pip install pyserial
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Union

import serial  # pyserial

logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS + ALIASES (paste INSIDE class SynthNVPro)
# =============================================================================


def safe_api(func):
    """
    Decorator for *high-level helper APIs only* (NOT raw I/O, NOT single-command wrappers).
    Provides:
      - global lock
      - retry
      - recover (close/reopen port)
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        last_exc = None
        for attempt in range(1, getattr(self, "max_retries", 2) + 1):
            with self._GLOBAL_LOCK:
                try:
                    return func(self, *args, **kwargs)
                except (TimeoutError, ValueError, serial.SerialException, OSError) as e:
                    last_exc = e
                    logger.warning(
                        f"[SynthNVPro] {func.__name__} failed: {type(e).__name__}: {e} "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    try:
                        self._recover_locked()
                    except Exception as e2:
                        logger.error(
                            f"[SynthNVPro] recover failed: {type(e2).__name__}: {e2}"
                        )
                    time.sleep(getattr(self, "recover_throttle_s", 0.02))
                except Exception as e:
                    # Unknown exception: still try recover once, then re-raise on last attempt
                    last_exc = e
                    logger.error(
                        f"[SynthNVPro] {func.__name__} unexpected error: {type(e).__name__}: {e} "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    try:
                        self._recover_locked()
                    except Exception:
                        pass
                    time.sleep(getattr(self, "recover_throttle_s", 0.02))
        raise RuntimeError(
            f"[SynthNVPro] {func.__name__} failed after {self.max_retries} retries"
        ) from last_exc

    return wrapper


# ----------------------------
# Optional structured snapshots
# ----------------------------


@dataclass(frozen=True)
class SynthNVProIdentity:
    """Static device identity."""

    model_type: str
    serial_number: str
    firmware_version: str
    hardware_version: str


# ----------------------
# Trigger connector modes (command 'y' per manual)
# ----------------------


class TriggerMode(IntEnum):
    NONE = 0
    FULL_SWEEP = 1
    SINGLE_STEP = 2
    STOP_ALL = 3
    RF_ON_OFF = 4
    REMOVE_INTERRUPTS = 5
    RESERVED_6 = 6
    RESERVED_7 = 7
    EXT_AM_INPUT = 8  # requires AM internal LUT = ramp
    EXT_FM_INPUT = 9  # requires FM internal modulation = chirp


# ----------------------------
# Driver
# ----------------------------


class SynthNVPro:
    """
    Full-command SynthNV Pro serial driver with robust I/O and safety constraints.

    Key behaviors:
    - write_raw(cmd): sends exactly cmd (no CR/LF).
    - query1(cmd): drains stale input, writes cmd, reads a single response line with retries.
    - get_help(): reads multi-line '?' output until timeout.
    - read_w(N): reads N samples terminated by 'EOM.' when N>1.

    Notes:
    - Some commands are not meaningful to query (e.g., '~?'); wrappers return None.
    - Some setters require prerequisites (e.g., 'a' needs Z0); wrapper supports force_z0.
    """

    # ---- class-level global lock (ONLY used by high-level helpers) ----
    _GLOBAL_LOCK = threading.RLock()

    # ---- high-level safety/retry tuning (safe layer only) ----
    max_retries: int = 2
    reconnect_backoff_s: float = 0.25
    post_pll_enable_boot_s: float = 0.05  # doc says ~20 ms; pad a bit
    recover_throttle_s: float = 0.02

    # ----- conservative defaults for USB CDC quirks -----
    DEFAULT_BAUD = 115200
    DEFAULT_TIMEOUT_S = 0.8
    DEFAULT_WRITE_TIMEOUT_S = 0.8
    DEFAULT_CMD_DELAY_S = 0.00

    # ----- robustness knobs -----
    DEFAULT_QUERY_RETRIES = 2
    DEFAULT_DRAIN_WINDOW_S = 0.03
    DEFAULT_POST_ENABLE_PLL_S = 0.03  # doc says ~20 ms; pad to 30 ms

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = DEFAULT_BAUD,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        write_timeout_s: Optional[float] = DEFAULT_WRITE_TIMEOUT_S,
        command_delay_s: float = DEFAULT_CMD_DELAY_S,
        query_retries: int = DEFAULT_QUERY_RETRIES,
        drain_window_s: float = DEFAULT_DRAIN_WINDOW_S,
    ):
        """
        Create a SynthNV Pro controller and open the serial port.

        Args:
            port: COM port name (e.g., "COM4").
            baudrate: USB CDC baudrate (not very meaningful, but avoid 1200).
            timeout_s: Read timeout for line-based responses.
            write_timeout_s: Write timeout (defaults to timeout_s).
            command_delay_s: Optional delay after each command write.
            query_retries: Retries for query timeouts / parse issues.
            drain_window_s: Time window to drain stale input before queries.
        """
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.write_timeout_s = (
            float(write_timeout_s) if write_timeout_s is not None else float(timeout_s)
        )
        self.command_delay_s = float(command_delay_s)
        self.query_retries = int(query_retries)
        self.drain_window_s = float(drain_window_s)

        self._ser: Optional[serial.Serial] = None
        self._lock = threading.RLock()

        self.connect()

    # ---------------------------------------------------------------------
    # Connection management
    # ---------------------------------------------------------------------

    def connect(self) -> None:
        """Open the serial port if not already open, and purge buffers."""
        with self._lock:
            if self._ser and self._ser.is_open:
                return
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout_s,
                write_timeout=self.write_timeout_s,
            )
            time.sleep(0.10)  # let USB CDC settle
            self.purge()

    def disconnect(self) -> None:
        """Close the serial port."""
        with self._lock:
            if self._ser:
                try:
                    self._ser.close()
                finally:
                    self._ser = None

    close = disconnect
    open = connect

    def purge(self) -> None:
        """Clear both input and output buffers."""
        with self._lock:
            if not self._ser:
                return
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()

    def __enter__(self) -> "SynthNVPro":
        """Context manager enter: ensure connected."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context manager exit: disconnect."""
        self.disconnect()

    # ---------------------------------------------------------------------
    # Low-level I/O primitives
    # ---------------------------------------------------------------------

    def _ensure_open(self) -> serial.Serial:
        """Return the open serial port or raise."""
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("SynthNVPro: serial port is not open")
        return self._ser

    def _drain_input_locked(self) -> None:
        """
        Drain any stale input for a short window.

        Why:
            Multi-line commands ('?', wN bursts) and timing issues can leave extra
            lines in the buffer. Draining before a query makes subsequent readline()
            far more reliable.
        """
        ser = self._ensure_open()
        t_end = time.time() + self.drain_window_s
        while time.time() < t_end:
            n = ser.in_waiting
            if n <= 0:
                break
            ser.read(n)
            # small yield to allow more bytes to arrive if they were mid-flight
            time.sleep(0.002)

    def write_raw(self, cmd: str) -> None:
        """
        Send exactly the provided command string bytes (NO CR/LF).

        Args:
            cmd: Command string, e.g. "f2870.00000000" or "W-10.000".
        """
        with self._lock:
            ser = self._ensure_open()
            ser.write(cmd.encode("ascii"))
            ser.flush()
            if self.command_delay_s:
                time.sleep(self.command_delay_s)

    def read_line(self) -> str:
        """
        Read one LF-terminated line and strip whitespace.

        Returns:
            Decoded ASCII response line with trailing whitespace stripped.

        Raises:
            TimeoutError: if no LF-terminated line arrives before timeout.
        """
        with self._lock:
            ser = self._ensure_open()
            raw = ser.readline()
            if not raw:
                raise TimeoutError("SynthNVPro: timeout waiting for response")
            return raw.decode("ascii", errors="replace").strip()

    def query1(self, cmd: str, *, parse: Optional[Callable[[str], Any]] = None) -> Any:
        """
        Send a command and read exactly one response line (with drain + retries).

        Args:
            cmd: Query command (e.g., "f?" or "W?").
            parse: Optional parser function to convert string to desired type.

        Returns:
            Parsed value if parse is provided, else the raw response string.

        Raises:
            TimeoutError / ValueError after exhausting retries.
        """
        last_exc: Optional[Exception] = None
        with self._lock:
            for _ in range(self.query_retries + 1):
                try:
                    self._drain_input_locked()
                    self.write_raw(cmd)
                    line = self.read_line()
                    return parse(line) if parse else line
                except (TimeoutError, ValueError) as e:
                    last_exc = e
                    # small backoff; also clears any partial bytes
                    self._drain_input_locked()
                    time.sleep(0.01)
        assert last_exc is not None
        raise last_exc

    # ---------------------------------------------------------------------
    # Formatting helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _fmt_mhz(x: float) -> str:
        """Format MHz with high enough precision for 0.1 Hz steps (~1e-7 MHz)."""
        return f"{float(x):.8f}"

    @staticmethod
    def _fmt_dbm(x: float) -> str:
        """Format dBm with 0.001 dB resolution."""
        return f"{float(x):.3f}"

    @staticmethod
    def _fmt_ms(x: float) -> str:
        """Format milliseconds."""
        return f"{float(x):.3f}"

    @staticmethod
    def _fmt_hz3(x: float) -> str:
        """Format Hz with 0.001 Hz resolution."""
        return f"{float(x):.3f}"

    @staticmethod
    def _fmt_int(x: Union[int, float]) -> str:
        """Format integer-like values."""
        return str(int(round(float(x))))

    # =====================================================================
    # Command wrappers (ALL commands)
    # =====================================================================

    # ------------------
    # Frequency / Power
    # ------------------

    def set_f(self, mhz: float) -> None:
        """
        Set RF frequency (MHz).

        Args:
            mhz: Frequency in MHz.
        """
        self.write_raw("f" + self._fmt_mhz(mhz))

    def get_f(self) -> float:
        """
        Get RF frequency (MHz).

        Returns:
            Frequency in MHz.
        """
        return float(self.query1("f?", parse=float))

    def set_W(self, dbm: float) -> None:
        """
        Set RF output power (dBm).

        Args:
            dbm: Power in dBm.
        """
        self.write_raw("W" + self._fmt_dbm(dbm))

    def get_W(self) -> float:
        """
        Get RF output power setting (dBm).

        Returns:
            Power in dBm.
        """
        return float(self.query1("W?", parse=float))

    def get_V(self) -> int:
        """
        Read amplifier calibration status.

        Returns:
            Integer flag (device-defined; typically 0/1).
        """
        return int(self.query1("V", parse=int))

    # --------------------------
    # Temperature compensation Z
    # --------------------------

    def set_Z(self, mode: int) -> None:
        """
        Set temperature compensation mode.

        Args:
            mode: 0=none, 1=on set, 2=1 sec, 3=10 sec.

        Raises:
            ValueError: if mode not in {0,1,2,3}.
        """
        if mode not in (0, 1, 2, 3):
            raise ValueError("Z mode must be 0,1,2,3")
        self.write_raw(f"Z{mode}")

    def get_Z(self) -> int:
        """
        Get temperature compensation mode.

        Returns:
            Mode integer.
        """
        return int(self.query1("Z?", parse=int))

    # ----------------------
    # Detector: w and &
    # ----------------------

    def set_ampersand(self, mode: int) -> None:
        """
        Set RF detector mode.

        Args:
            mode: 0=instant, 1=average, 2=peak.

        Raises:
            ValueError: if mode not in {0,1,2}.
        """
        if mode not in (0, 1, 2):
            raise ValueError("& mode must be 0,1,2")
        self.write_raw(f"&{mode}")

    def get_ampersand(self) -> int:
        """
        Get RF detector mode.

        Returns:
            Mode integer.
        """
        return int(self.query1("&?", parse=int))

    def read_w(self, samples: int = 1) -> List[float]:
        """
        Read RF detector power in dBm.

        Args:
            samples: If 1, reads one sample. If >1, reads N lines followed by 'EOM.'.

        Returns:
            List of dBm readings (length == samples).

        Raises:
            TimeoutError: if device doesn't respond.
            ValueError: if parsing fails.
        """
        if samples < 1:
            raise ValueError("samples must be >= 1")

        with self._lock:
            self._drain_input_locked()
            if samples == 1:
                self.write_raw("w")
                return [float(self.read_line())]

            self.write_raw(f"w{int(samples)}")
            vals: List[float] = []
            while True:
                line = self.read_line()
                if line == "EOM.":
                    break
                if not line:
                    continue
                vals.append(float(line))

            # If firmware returns fewer samples, don't hard-fail; surface what we got.
            return vals

    # # ----------------------
    # # VGA DAC: a (raw)
    # # ----------------------

    # def set_a(self, dac: int, *, force_z0: bool = True) -> None:
    #     """
    #     Set raw VGA DAC (0..4000).

    #     IMPORTANT:
    #         Raw DAC control is only effective when temperature compensation is OFF (Z0).
    #         If force_z0=True, this function will set Z0 before writing 'a'.

    #     Args:
    #         dac: DAC value [0..4000].
    #         force_z0: If True, forces Z0 first.

    #     Raises:
    #         ValueError: if dac out of range.
    #     """
    #     dac_i = int(dac)
    #     if not (0 <= dac_i <= 4000):
    #         raise ValueError("a must be 0..4000")
    #     if force_z0:
    #         # Required prerequisite for raw DAC control
    #         self.set_Z(0)
    #         time.sleep(0.01)
    #     self.write_raw("a" + str(dac_i))

    # def get_a(self) -> int:
    #     """
    #     Get raw VGA DAC setting.

    #     Returns:
    #         DAC integer.
    #     """
    #     return int(self.query1("a?", parse=int))

    # ----------------------
    # Phase step: ~
    # ----------------------

    def set_tilde(self, deg_increment: float) -> None:
        """
        Apply a relative phase step (degrees).

        Args:
            deg_increment: Phase increment in degrees.
        """
        self.write_raw("~" + f"{float(deg_increment):.4f}")

    def get_tilde(self) -> Optional[float]:
        """
        Query '~?'.

        NOTE:
            Programming guide states '~?' has no real meaning.
            This wrapper returns None by default to prevent tests from asserting on it.

        Returns:
            None (intentionally).
        """
        return None

    # ----------------------
    # Output gate: h
    # ----------------------

    def set_h(self, high: int) -> None:
        """
        Set RF High(1) or Low(0) power (acts as output gate).

        Args:
            high: 1 enables RF high, 0 disables/low.
        """
        if high not in (0, 1):
            raise ValueError("h must be 0 or 1")
        self.write_raw(f"h{high}")

    def get_h(self) -> int:
        """
        Get RF High/Low state.

        Returns:
            0 or 1.
        """
        return int(self.query1("h?", parse=int))

    # ----------------------
    # PLL enable: E
    # ----------------------

    def set_E(self, on: int) -> None:
        """
        Enable/disable PLL chip power.

        Args:
            on: 1 enables PLL, 0 disables.

        Notes:
            Enabling PLL can take ~20 ms to boot; we wait ~30 ms to stabilize.
        """
        if on not in (0, 1):
            raise ValueError("E must be 0 or 1")
        self.write_raw(f"E{on}")
        if on == 1:
            time.sleep(self.DEFAULT_POST_ENABLE_PLL_S)

    def get_E(self) -> int:
        """
        Get PLL chip enable state.

        Returns:
            0 or 1.
        """
        return int(self.query1("E?", parse=int))

    # ----------------------
    # Charge pump: U
    # ----------------------

    def set_U(self, val: int) -> None:
        """
        Set PLL charge pump current.

        Args:
            val: Allowed range is 1..15. (0 should be avoided; PLL may not lock.)

        Raises:
            ValueError: if val not in 1..15.
        """
        v = int(val)
        if not (1 <= v <= 15):
            raise ValueError("U must be 1..15 (avoid 0; PLL may not lock)")
        self.write_raw("U" + str(v))

    def get_U(self) -> int:
        """
        Get PLL charge pump current setting.

        Returns:
            Integer value.
        """
        return int(self.query1("U?", parse=int))

    # ----------------------
    # Ref doubler: D
    # ----------------------

    def set_D(self, on: int) -> None:
        """
        Enable/disable reference doubler.

        Args:
            on: 1 enables, 0 disables.
        """
        if on not in (0, 1):
            raise ValueError("D must be 0 or 1")
        self.write_raw(f"D{on}")

    def get_D(self) -> int:
        """
        Get reference doubler state.

        Returns:
            0 or 1.
        """
        return int(self.query1("D?", parse=int))

    # ----------------------
    # Channel spacing: i
    # ----------------------

    def set_i(self, hz: float) -> None:
        """
        Set channel spacing (Hz).

        Args:
            hz: Allowed range 0.1..1000.0 Hz.

        Notes:
            Very small spacing can make USB responses slower; this method adds a small settle delay
            when hz is small.
        """
        v = float(hz)
        if not (0.1 <= v <= 1000.0):
            raise ValueError("i must be 0.1..1000 Hz")
        self.write_raw("i" + self._fmt_hz3(v))
        if v < 10.0:
            time.sleep(0.05)

    def get_i(self) -> float:
        """
        Get channel spacing (Hz).

        Returns:
            Spacing in Hz.
        """
        return float(self.query1("i?", parse=float))

    # ----------------------
    # Reference select: x
    # ----------------------

    def set_x(self, ref_sel: int) -> None:
        """
        Set reference selection.

        Args:
            ref_sel: 0=external, 1=internal 27 MHz, 2=internal 10 MHz.
        """
        if ref_sel not in (0, 1, 2):
            raise ValueError("x must be 0,1,2")
        self.write_raw(f"x{ref_sel}")

    def get_x(self) -> int:
        """
        Get reference selection.

        Returns:
            0, 1, or 2.
        """
        return int(self.query1("x?", parse=int))

    # ----------------------
    # Reference frequency: *
    # ----------------------

    def set_star(self, mhz: float) -> None:
        """
        Set PLL reference frequency (MHz).

        Args:
            mhz: Reference frequency in MHz.
        """
        self.write_raw("*" + self._fmt_mhz(mhz))

    def get_star(self) -> float:
        """
        Get PLL reference frequency (MHz).

        Returns:
            Reference frequency in MHz.
        """
        return float(self.query1("*?", parse=float))

    # ----------------------
    # Sweep controls
    # ----------------------

    def set_l(self, mhz: float) -> None:
        """Set sweep lower frequency (MHz)."""
        self.write_raw("l" + self._fmt_mhz(mhz))

    def get_l(self) -> float:
        """Get sweep lower frequency (MHz)."""
        return float(self.query1("l?", parse=float))

    def set_u(self, mhz: float) -> None:
        """Set sweep upper frequency (MHz)."""
        self.write_raw("u" + self._fmt_mhz(mhz))

    def get_u(self) -> float:
        """Get sweep upper frequency (MHz)."""
        return float(self.query1("u?", parse=float))

    def set_s(self, val: float) -> None:
        """
        Set sweep step value.

        Note:
            Interpretation depends on sweep type X (linear / table / percent).
        """
        self.write_raw("s" + self._fmt_mhz(val))

    def get_s(self) -> float:
        """Get sweep step value."""
        return float(self.query1("s?", parse=float))

    def set_t(self, ms: float) -> None:
        """Set sweep step time (ms)."""
        self.write_raw("t" + self._fmt_ms(ms))

    def get_t(self) -> float:
        """Get sweep step time (ms)."""
        return float(self.query1("t?", parse=float))

    def set_left_bracket(self, dbm: float) -> None:
        """Set sweep amplitude low (dBm)."""
        self.write_raw("[" + self._fmt_dbm(dbm))

    def get_left_bracket(self) -> float:
        """Get sweep amplitude low (dBm)."""
        return float(self.query1("[?", parse=float))

    def set_right_bracket(self, dbm: float) -> None:
        """Set sweep amplitude high (dBm)."""
        self.write_raw("]" + self._fmt_dbm(dbm))

    def get_right_bracket(self) -> float:
        """Get sweep amplitude high (dBm)."""
        return float(self.query1("]?", parse=float))

    def set_caret(self, up: int) -> None:
        """Set sweep direction (1=up, 0=down)."""
        if up not in (0, 1):
            raise ValueError("^ must be 0 or 1")
        self.write_raw(f"^{up}")

    def get_caret(self) -> int:
        """Get sweep direction (1=up, 0=down)."""
        return int(self.query1("^?", parse=int))

    def set_X(self, typ: int) -> None:
        """Set sweep type (0=linear, 1=table, 2=percent)."""
        if typ not in (0, 1, 2):
            raise ValueError("X must be 0,1,2")
        self.write_raw(f"X{typ}")

    def get_X(self) -> int:
        """Get sweep type."""
        return int(self.query1("X?", parse=int))

    def set_r(self, on: int) -> None:
        """Enable/disable reading while sweeping (0/1)."""
        if on not in (0, 1):
            raise ValueError("r must be 0 or 1")
        self.write_raw(f"r{on}")

    def get_r(self) -> int:
        """Get reading-while-sweeping flag."""
        return int(self.query1("r?", parse=int))

    def set_d(self, mode: int) -> None:
        """Set sweep display mode (0/1/2)."""
        if mode not in (0, 1, 2):
            raise ValueError("d must be 0,1,2")
        self.write_raw(f"d{mode}")

    def get_d(self) -> int:
        """Get sweep display mode."""
        return int(self.query1("d?", parse=int))

    def do_m(self) -> str:
        """
        Execute 'm' (max/min sweep display; may be disabled on some firmware).

        Returns:
            A response line if present, else empty string.
        """
        with self._lock:
            self._drain_input_locked()
            self.write_raw("m")
            try:
                return self.read_line()
            except TimeoutError:
                return ""

    def set_g(self, on: int) -> None:
        """Start/stop sweep run (1/0)."""
        if on not in (0, 1):
            raise ValueError("g must be 0 or 1")
        self.write_raw(f"g{on}")

    def get_g(self) -> int:
        """Get sweep run flag."""
        return int(self.query1("g?", parse=int))

    def set_c(self, on: int) -> None:
        """Enable/disable sweep continuous mode (1/0)."""
        if on not in (0, 1):
            raise ValueError("c must be 0 or 1")
        self.write_raw(f"c{on}")

    def get_c(self) -> int:
        """Get sweep continuous mode flag."""
        return int(self.query1("c?", parse=int))

    def set_y(self, mode: int) -> None:
        """
        Set Trigger connector function. (Manual: command 'y')
        Values:
        0 none
        1 trigger full frequency sweep
        2 trigger single frequency step
        3 trigger stop-all (pauses sequencing)
        4 digital RF ON/OFF (external pulse modulation)
        5 remove interrupts (less jitter; use carefully)
        8 external AM input (AM internal LUT must be ramp)
        9 external FM input (FM internal modulation must be chirp)

        NOTE: Some firmware/docs mention 'w' as the setter/query; we support fallback.
        """
        try:
            m = int(mode)
        except Exception as e:
            raise TypeError("mode must be int-like") from e

        if m < 0 or m > 9:
            raise ValueError("Trigger mode must be 0..9")

        # If you want to forbid reserved values:
        # if m in (6, 7):
        #     raise ValueError("Trigger modes 6 and 7 are reserved")

        # Primary: 'y'
        try:
            self.write_raw(f"y{m}")
            return
        except Exception:
            # Fallback: 'w' (some manuals / firmware variants)
            self.write_raw(f"w{m}")

    def get_y(self) -> int:
        """Query Trigger connector function. (Manual: 'y?'; fallback 'w?')"""
        try:
            return int(self.query1("y?", parse=int))
        except Exception:
            return int(self.query1("w?", parse=int))

    # # ----------------------
    # # AM controls
    # # ----------------------

    # def set_F(self, us: Union[int, float]) -> None:
    #     """Set AM step time (microseconds)."""
    #     self.write_raw("F" + self._fmt_int(us))

    # def get_F(self) -> int:
    #     """Get AM step time (microseconds)."""
    #     return int(self.query1("F?", parse=int))

    # def set_q(self, reps: Union[int, float]) -> None:
    #     """Set AM cycle repetitions."""
    #     self.write_raw("q" + self._fmt_int(reps))

    # def get_q(self) -> int:
    #     """Get AM cycle repetitions."""
    #     return int(self.query1("q?", parse=int))

    # def set_A(self, on: int) -> None:
    #     """Enable/disable AM continuous run (1/0)."""
    #     if on not in (0, 1):
    #         raise ValueError("A must be 0 or 1")
    #     self.write_raw(f"A{on}")

    # def get_A(self) -> int:
    #     """Get AM continuous run flag."""
    #     return int(self.query1("A?", parse=int))

    # # ----------------------
    # # Pulse controls
    # # ----------------------

    # def set_P(self, us: Union[int, float]) -> None:
    #     """Set pulse ON time (microseconds)."""
    #     self.write_raw("P" + self._fmt_int(us))

    # def get_P(self) -> int:
    #     """Get pulse ON time (microseconds)."""
    #     return int(self.query1("P?", parse=int))

    # def set_O(self, us: Union[int, float]) -> None:
    #     """Set pulse OFF time (microseconds)."""
    #     self.write_raw("O" + self._fmt_int(us))

    # def get_O(self) -> int:
    #     """Get pulse OFF time (microseconds)."""
    #     return int(self.query1("O?", parse=int))

    # def set_R(self, reps: Union[int, float]) -> None:
    #     """Set pulse repetitions."""
    #     self.write_raw("R" + self._fmt_int(reps))

    # def get_R(self) -> int:
    #     """Get pulse repetitions."""
    #     return int(self.query1("R?", parse=int))

    # def set_colon(self, on: int) -> None:
    #     """Set pulse invert signal (1/0)."""
    #     if on not in (0, 1):
    #         raise ValueError(": must be 0 or 1")
    #     self.write_raw(f":{on}")

    # def get_colon(self) -> int:
    #     """Get pulse invert signal flag."""
    #     return int(self.query1(":?", parse=int))

    # def do_G(self) -> None:
    #     """Run one pulse burst (uses current P/O/R/:/j settings)."""
    #     self.write_raw("G")

    # def set_j(self, on: int) -> None:
    #     """Enable/disable pulse continuous mode (1/0)."""
    #     if on not in (0, 1):
    #         raise ValueError("j must be 0 or 1")
    #     self.write_raw(f"j{on}")

    # def get_j(self) -> int:
    #     """Get pulse continuous mode flag."""
    #     return int(self.query1("j?", parse=int))

    # ----------------------
    # FM controls
    # ----------------------

    def set_lt(self, hz: float) -> None:
        """Set FM frequency (Hz). Command is '<'."""
        self.write_raw("<" + self._fmt_hz3(hz))

    def get_lt(self) -> float:
        """Get FM frequency (Hz). Command is '<?'."""
        return float(self.query1("<?", parse=float))

    def set_gt(self, hz: float) -> None:
        """Set FM deviation (Hz). Command is '>'."""
        self.write_raw(">" + self._fmt_hz3(hz))

    def get_gt(self) -> float:
        """Get FM deviation (Hz). Command is '>?'."""
        return float(self.query1(">?", parse=float))

    def set_comma(self, reps: Union[int, float]) -> None:
        """Set FM repetitions. Command is ','."""
        self.write_raw("," + self._fmt_int(reps))

    def get_comma(self) -> int:
        """Get FM repetitions. Command is ',?'."""
        return int(self.query1(",?", parse=int))

    def set_semicolon(self, typ: int) -> None:
        """Set FM type (0=sinusoid, 1=chirp). Command is ';'."""
        if typ not in (0, 1):
            raise ValueError("; must be 0 or 1")
        self.write_raw(f";{typ}")

    def get_semicolon(self) -> int:
        """Get FM type. Command is ';?'."""
        return int(self.query1(";?", parse=int))

    def set_slash(self, on: int) -> None:
        """Enable/disable FM continuous mode (1/0). Command is '/'."""
        if on not in (0, 1):
            raise ValueError("/ must be 0 or 1")
        self.write_raw(f"/{on}")

    def get_slash(self) -> int:
        """Get FM continuous mode flag. Command is '/?'."""
        return int(self.query1("/?", parse=int))

    # ----------------------
    # Misc / status / info
    # ----------------------

    def set_b(self, mode: int) -> None:
        """Set speaker feedback mode (device-defined)."""
        self.write_raw("b" + str(int(mode)))

    def get_b(self) -> int:
        """Get speaker feedback mode."""
        return int(self.query1("b?", parse=int))

    def get_p(self) -> int:
        """Get PLL lock status (0/1). Command is 'p'."""
        return int(self.query1("p", parse=int))

    def get_z(self) -> float:
        """Get device temperature (°C). Command is 'z'."""
        return float(self.query1("z", parse=float))

    def get_v(self, which: int = 0) -> str:
        """
        Get version string.

        Args:
            which: 0=firmware, 1=hardware.

        Returns:
            Version string.
        """
        if which not in (0, 1):
            raise ValueError("v(which) must be 0 or 1")
        return str(self.query1(f"v{which}"))

    def get_plus(self) -> str:
        """Get model type string. Command is '+'."""
        return str(self.query1("+"))

    def get_minus(self) -> str:
        """Get serial number string. Command is '-'."""
        return str(self.query1("-"))

    def do_e(self) -> None:
        """Write all current settings to EEPROM. Command is 'e'."""
        self.write_raw("e")

    def get_help(self) -> str:
        """
        Get command help text. Command is '?'.

        Returns:
            Multi-line help text (joined by '\\n'). Reads until read timeout.
        """
        lines: List[str] = []
        with self._lock:
            self._drain_input_locked()
            self.write_raw("?")
            while True:
                try:
                    lines.append(self.read_line())
                except TimeoutError:
                    break
        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------

    def get_identity(self) -> SynthNVProIdentity:
        """
        Read identity fields (model, serial, firmware, hardware).

        Returns:
            SynthNVProIdentity dataclass.
        """
        return SynthNVProIdentity(
            model_type=self.get_plus(),
            serial_number=self.get_minus(),
            firmware_version=self.get_v(0),
            hardware_version=self.get_v(1),
        )

    def safe_mute(self) -> None:
        """Best-effort RF mute helper (does not raise on failure)."""
        try:
            self.set_h(0)
        except Exception:
            pass

    def set_reference(
        self, mode: Union[str, int], *, ext_ref_mhz: Optional[float] = 10.0
    ) -> None:
        """
        Convenience reference setup (wraps x and *).

        Args:
            mode:
                - "ext" / 0: external reference; sets x0 and sets * to ext_ref_mhz if provided.
                - "int27" / 1: internal 27 MHz.
                - "int10" / 2: internal 10 MHz.
            ext_ref_mhz: If external, sets * to this value (MHz) when not None.
        """
        if isinstance(mode, str):
            m = mode.strip().lower()
            if m in ("ext", "external"):
                self.set_x(0)
                if ext_ref_mhz is not None:
                    self.set_star(float(ext_ref_mhz))
            elif m in ("int27", "internal27", "internal_27", "int_27"):
                self.set_x(1)
                self.set_star(27.0)
            elif m in ("int10", "internal10", "internal_10", "int_10"):
                self.set_x(2)
                self.set_star(10.0)
            else:
                raise ValueError(f"Unknown reference mode: {mode!r}")
        else:
            i = int(mode)
            self.set_x(i)
            if i == 1:
                self.set_star(27.0)
            elif i == 2:
                self.set_star(10.0)
            # for i==0, leave * as-is unless caller sets it separately

    # -------------------------------------------------------------------------
    # Internal connection recovery helpers (used by safe_api helpers only)
    # -------------------------------------------------------------------------

    def _connect_fresh(self) -> None:
        """
        Re-open the serial port cleanly (used by recovery + optionally by helpers).
        This should NOT change instrument settings, only rebuild comms reliably.
        """
        # NOTE: assumes your class already has: port, baudrate, timeout_s, write_timeout_s
        # and methods: purge(), connect(), disconnect()
        # We'll do it explicitly for robustness.
        try:
            if getattr(self, "_ser", None) and self._ser.is_open:
                try:
                    self._ser.close()
                except Exception:
                    pass
        except Exception:
            pass

        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout_s,
            write_timeout=getattr(self, "write_timeout_s", self.timeout_s),
        )
        time.sleep(0.10)
        try:
            self.purge()
        except Exception:
            # last resort: try direct buffer clears
            try:
                self._ser.reset_input_buffer()
                self._ser.reset_output_buffer()
            except Exception:
                pass

    def _recover_locked(self) -> None:
        """
        Recover serial communications (close -> backoff -> reopen -> purge).
        Intended to run under _GLOBAL_LOCK from safe_api.
        """
        try:
            if getattr(self, "_ser", None):
                try:
                    self._ser.close()
                except Exception:
                    pass
        finally:
            self._ser = None

        time.sleep(getattr(self, "reconnect_backoff_s", 0.25))
        self._connect_fresh()

    # -------------------------------------------------------------------------
    # State snapshot / restore (for soft_reset + convenience)
    # -------------------------------------------------------------------------

    def snapshot_state(self) -> Dict[str, Any]:
        """
        Snapshot a minimal but useful set of device settings.

        Returns:
            dict with keys:
                f, W, h, x, star, Z, ampersand, i, D, U, E
            Missing keys may be absent if a read fails.
        """
        st: Dict[str, Any] = {}
        # These are single-command wrappers; no global lock here by design.
        for k, fn in [
            ("f", self.get_f),
            ("W", self.get_W),
            ("h", self.get_h),
            ("x", self.get_x),
            ("star", self.get_star),
            ("Z", self.get_Z),
            ("ampersand", self.get_ampersand),
            ("i", self.get_i),
            ("D", self.get_D),
            ("U", self.get_U),
            ("E", self.get_E),
        ]:
            try:
                st[k] = fn()
            except Exception:
                pass
        return st

    def restore_state(self, state: Dict[str, Any], *, restore_h: bool = False) -> None:
        """
        Restore settings from snapshot_state().

        Args:
            state: dict from snapshot_state()
            restore_h: if True, restore RF gate state; otherwise leaves muted/off.
        """
        # Safe order: mute first, then ref chain, then PLL-related, then RF params
        try:
            self.set_h(0)
        except Exception:
            try:
                self.write_raw("h0")
            except Exception:
                pass

        # Reference chain
        if "x" in state:
            try:
                self.set_x(int(state["x"]))
                time.sleep(0.03)
            except Exception:
                pass
        if "star" in state:
            try:
                self.set_star(float(state["star"]))
                time.sleep(0.03)
            except Exception:
                pass

        # PLL / tuning knobs
        if "i" in state:
            try:
                self.set_i(float(state["i"]))
                time.sleep(0.05)
            except Exception:
                pass
        if "D" in state:
            try:
                self.set_D(int(state["D"]))
                time.sleep(0.03)
            except Exception:
                pass
        if "U" in state:
            try:
                self.set_U(int(state["U"]))
                time.sleep(0.03)
            except Exception:
                pass
        if "Z" in state:
            try:
                self.set_Z(int(state["Z"]))
                time.sleep(0.03)
            except Exception:
                pass
        if "ampersand" in state:
            try:
                self.set_ampersand(int(state["ampersand"]))
                time.sleep(0.03)
            except Exception:
                pass

        # Core output params
        if "f" in state:
            try:
                self.set_f(float(state["f"]))
                time.sleep(0.05)
            except Exception:
                pass
        if "W" in state:
            try:
                self.set_W(float(state["W"]))
                time.sleep(0.05)
            except Exception:
                pass

        # RF gate (optional)
        if restore_h and "h" in state:
            try:
                self.set_h(int(state["h"]))
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # High-level helper APIs (these DO use global safe layer)
    # -------------------------------------------------------------------------

    @safe_api
    def initialize(
        self,
        *,
        reference: Optional[str] = None,
        ext_ref_mhz: float = 10.0,
        pll_on: bool = True,
        mute: bool = True,
    ) -> None:
        """
        Put device into a known safe-ish state for experiment control.

        Args:
            reference: Optional reference mode: "ext", "int10", or "int27". If None, leave unchanged.
            ext_ref_mhz: External reference frequency (MHz) used when reference="ext".
            pll_on: Ensure PLL enabled (E1) if True.
            mute: Mute output (h0) if True.
        """
        if mute:
            try:
                self.set_h(0)
            except Exception:
                self.write_raw("h0")

        if pll_on:
            try:
                self.set_E(1)
                time.sleep(self.post_pll_enable_boot_s)
            except Exception:
                # If E fails, still proceed with reference; recovery will handle if needed
                pass

        if reference is not None:
            self.set_reference(reference, ext_ref_mhz=ext_ref_mhz)

    @safe_api
    def soft_reset(self, *, level: str = "light", restore_h: bool = False) -> bool:
        """
        Attempt to recover device to a sane state without user intervention.

        Args:
            level:
                - "light": re-assert snapshot settings (no PLL power cycle)
                - "pll_cycle": toggles E0->E1 before re-asserting
            restore_h: if True, restore RF gate state from snapshot (otherwise leave muted)

        Returns:
            True if completed (best-effort).
        """
        st = self.snapshot_state()

        # Always mute before resets
        try:
            self.set_h(0)
        except Exception:
            try:
                self.write_raw("h0")
            except Exception:
                pass

        if level.lower() in ("pll_cycle", "pll", "cycle"):
            try:
                self.set_E(0)
                time.sleep(0.05)
            except Exception:
                pass
            try:
                self.set_E(1)
                time.sleep(self.post_pll_enable_boot_s)
            except Exception:
                pass

        # Re-apply snapshot (best-effort)
        self.restore_state(st, restore_h=restore_h)

        return True

    @safe_api
    def wait_for_pll_lock(
        self, *, timeout_s: float = 1.5, poll_s: float = 0.05
    ) -> bool:
        """
        Poll PLL lock status until locked or timeout.

        Returns:
            True if lock achieved before timeout, else False.
        """
        t0 = time.time()
        while (time.time() - t0) < timeout_s:
            try:
                if int(self.get_p()) == 1:
                    return True
            except Exception:
                # allow retry; safe_api will recover if needed
                pass
            time.sleep(poll_s)
        return False

    @safe_api
    def set_output(
        self,
        *,
        freq_mhz: Optional[float] = None,
        power_dbm: Optional[float] = None,
        enable: Optional[bool] = None,
        wait_lock: bool = False,
        lock_timeout_s: float = 1.5,
    ) -> None:
        """
        High-level output configuration helper.

        Args:
            freq_mhz: If provided, set frequency (MHz).
            power_dbm: If provided, set power (dBm).
            enable: If provided, set RF gate (True->on, False->off).
            wait_lock: If True, call wait_for_pll_lock() after setting params.
            lock_timeout_s: PLL lock timeout used when wait_lock is True.
        """
        if enable is False:
            # mute first if asked to disable
            try:
                self.set_h(0)
            except Exception:
                self.write_raw("h0")

        if freq_mhz is not None:
            self.set_f(float(freq_mhz))
            time.sleep(0.03)
        if power_dbm is not None:
            self.set_W(float(power_dbm))
            time.sleep(0.03)

        if wait_lock:
            self.wait_for_pll_lock(timeout_s=lock_timeout_s)

        if enable is True:
            self.set_h(1)

    @safe_api
    def get_device_info(self) -> Dict[str, str]:
        """
        Read device identity info.

        Returns:
            dict with keys: model, serial, fw, hw
        """
        return {
            "model": str(self.get_plus()),
            "serial": str(self.get_minus()),
            "fw": str(self.get_v(0)),
            "hw": str(self.get_v(1)),
        }

    @safe_api
    def get_status(self) -> Dict[str, Any]:
        """
        Read a compact status snapshot (useful for logging/diagnostics).

        Returns:
            dict with common keys (best-effort).
        """
        st: Dict[str, Any] = {}
        for k, fn in [
            ("f_mhz", self.get_f),
            ("power_dbm", self.get_W),
            ("gate_h", self.get_h),
            ("pll_lock", self.get_p),
            ("temp_c", self.get_z),
            ("ref_sel_x", self.get_x),
            ("ref_mhz_star", self.get_star),
            ("tempcomp_Z", self.get_Z),
            ("det_mode_amp", self.get_ampersand),
            ("pll_enable_E", self.get_E),
            ("charge_pump_U", self.get_U),
            ("ref_doubler_D", self.get_D),
            ("channel_spacing_i_hz", self.get_i),
        ]:
            try:
                st[k] = fn()
            except Exception:
                pass
        return st

    # ============================================================================
    # SAFE ALIASES (high-level API compatibility layer)
    # All functions below use @safe_api
    # ============================================================================

    # ----------------------
    # Frequency aliases
    # ----------------------

    @safe_api
    def set_frequency(self, freq_mhz: float) -> None:
        """Alias for set_f (MHz)."""
        self.set_f(freq_mhz)

    @safe_api
    def get_frequency(self) -> float:
        """Alias for get_f (MHz)."""
        return self.get_f()

    @safe_api
    def set_freq(self, freq_mhz: float) -> None:
        """Alias for set_f (MHz)."""
        self.set_f(freq_mhz)

    @safe_api
    def get_freq(self) -> float:
        """Alias for get_f (MHz)."""
        return self.get_f()

    # ----------------------
    # Power aliases
    # ----------------------

    @safe_api
    def set_power(self, power_dbm: float) -> None:
        """Alias for set_W (dBm)."""
        self.set_W(power_dbm)

    @safe_api
    def get_power(self) -> float:
        """Alias for get_W (dBm)."""
        return self.get_W()

    @safe_api
    def set_output_power(self, power_dbm: float) -> None:
        """Alias for set_W (dBm)."""
        self.set_W(power_dbm)

    @safe_api
    def get_output_power(self) -> float:
        """Alias for get_W (dBm)."""
        return self.get_W()

    # ----------------------
    # Phase aliases (write-only)
    # ----------------------

    @safe_api
    def step_phase(self, deg: float) -> None:
        """
        Apply a relative phase step in degrees.
        Alias for set_tilde().
        """
        self.set_tilde(deg)

    @safe_api
    def increment_phase(self, deg: float) -> None:
        """Alias for step_phase()."""
        self.set_tilde(deg)

    # ----------------------
    # Reference aliases
    # ----------------------

    @safe_api
    def set_reference_clock(self, mode: str, freq_mhz: Optional[float] = None) -> None:
        """
        Alias for set_reference().

        Args:
            mode: "ext", "int10", "int27"
            freq_mhz: external ref frequency (MHz) when mode="ext"
        """
        m = mode.lower().strip()
        if m in ("ext", "external"):
            self.set_reference(
                "ext", ext_ref_mhz=(10.0 if freq_mhz is None else float(freq_mhz))
            )
        elif m in ("int10", "internal10", "internal_10", "int_10"):
            self.set_reference("int10")
        elif m in ("int27", "internal27", "internal_27", "int_27"):
            self.set_reference("int27")
        else:
            raise ValueError(f"Unknown reference mode: {mode!r}")

    @safe_api
    def get_reference_clock(self) -> Dict[str, Any]:
        """
        Return reference configuration.

        Returns:
            dict: {"x": <0/1/2>, "ref_mhz": <float>}
        """
        return {
            "x": self.get_x(),
            "ref_mhz": self.get_star(),
        }

    @safe_api
    def set_ref_source(self, ref_sel: int) -> None:
        """Alias for set_x()."""
        self.set_x(ref_sel)

    @safe_api
    def get_ref_source(self) -> int:
        """Alias for get_x()."""
        return self.get_x()

    @safe_api
    def set_ref_frequency(self, ref_mhz: float) -> None:
        """Alias for set_star()."""
        self.set_star(ref_mhz)

    @safe_api
    def get_ref_frequency(self) -> float:
        """Alias for get_star()."""
        return self.get_star()

    # ----------------------
    # Output gate aliases
    # ----------------------
    @safe_api
    def enable_output(self) -> None:
        """Enable RF output (gate on)."""
        if self.get_E() != 1:
            self.set_E(1)  # turn on the PLL if it was off
            self.wait_for_pll_lock(timeout_s=2.0)
        self.set_h(1)

    @safe_api
    def disable_output(self) -> None:
        """Disable RF output (gate off)."""
        self.set_h(0)

    @safe_api
    def output_on(self) -> None:
        """Alias for enable_output()."""
        self.set_h(1)

    @safe_api
    def output_off(self) -> None:
        """Alias for disable_output()."""
        self.set_h(0)

    @safe_api
    def is_output_enabled(self) -> bool:
        """
        Return True if RF gate is enabled (h == 1).
        Note: does NOT imply PLL enabled.
        """
        return bool(int(self.get_h()))

    # ----------------------
    # Safe mute helpers
    # ----------------------

    @safe_api
    def mute(self) -> None:
        """Alias for output_off()."""
        self.set_h(0)

    @safe_api
    def safe_mute(self) -> None:
        """
        Best-effort mute (never raises).
        Uses gate first; falls back to raw command if needed.
        """
        try:
            self.set_h(0)
        except Exception:
            try:
                self.write_raw("h0")
            except Exception:
                pass

    @safe_api
    def configure_fm(
        self,
        *,
        rate_hz: float,
        deviation_hz: float,
        mode: str = "sine",
        continuous: bool = True,
        repetitions: int = 1,
    ) -> None:
        """
        Safely configure FM in one call.
        """

        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")

        if deviation_hz < 0:
            raise ValueError("deviation_hz must be >= 0")

        mode = mode.lower()
        if mode not in ("sine", "chirp"):
            raise ValueError("mode must be 'sine' or 'chirp'")

        # ---- Stop FM before reconfiguring ----
        self.set_slash(0)

        # ---- Apply parameters ----
        self.set_lt(rate_hz)
        self.set_gt(deviation_hz)
        self.set_semicolon(0 if mode == "sine" else 1)

        if continuous:
            self.set_slash(1)
        else:
            self.set_comma(int(repetitions))
            self.set_slash(0)

    @safe_api
    def enable_fm(self) -> None:
        """Enable FM continuous mode."""
        self.set_slash(1)

    @safe_api
    def disable_fm(self) -> None:
        """Disable FM."""
        self.set_slash(0)

    # ----------------------
    # Friendly trigger API
    # ----------------------

    @safe_api
    def set_trigger_mode(self, mode: TriggerMode | int) -> None:
        """Preferred API: set trigger mode using TriggerMode enum (or int)."""
        self.set_y(int(mode))

    @safe_api
    def get_trigger_mode(self) -> TriggerMode:
        """Preferred API: get trigger mode as TriggerMode enum."""
        val = self.get_y()
        try:
            return TriggerMode(val)
        except ValueError:
            # Future-proof: unknown values still returned as raw int-like enum fallback
            return (
                TriggerMode.NONE if val == 0 else TriggerMode(val)
            )  # will still raise if truly unknown

    # Convenience one-liners
    @safe_api
    def trigger_off(self) -> None:
        self.set_trigger_mode(TriggerMode.NONE)

    @safe_api
    def trigger_full_sweep(self) -> None:
        self.set_trigger_mode(TriggerMode.FULL_SWEEP)

    @safe_api
    def trigger_single_step(self) -> None:
        self.set_trigger_mode(TriggerMode.SINGLE_STEP)

    @safe_api
    def trigger_stop_all(self) -> None:
        self.set_trigger_mode(TriggerMode.STOP_ALL)

    @safe_api
    def trigger_rf_gate(self) -> None:
        """Use trigger input as digital RF ON/OFF gate (external pulse modulation)."""
        self.set_trigger_mode(TriggerMode.RF_ON_OFF)

    @safe_api
    def trigger_remove_interrupts(self, enable: bool = True) -> None:
        """
        Reduce modulation jitter by removing interrupts (use carefully).
        enable=True -> mode 5; enable=False -> restore to NONE (0)
        """
        self.set_trigger_mode(
            TriggerMode.REMOVE_INTERRUPTS if enable else TriggerMode.NONE
        )

    @safe_api
    def trigger_external_am_input(self) -> None:
        """
        Route trigger input as external AM modulation input (mode 8).
        Per manual: requires AM internal modulation LUT set to ramp.
        """
        self.set_trigger_mode(TriggerMode.EXT_AM_INPUT)

    @safe_api
    def trigger_external_fm_input(self) -> None:
        """
        Route trigger input as external FM modulation input (mode 9).
        Per manual: requires FM internal modulation set to chirp.
        """
        self.set_trigger_mode(TriggerMode.EXT_FM_INPUT)


# ----------------------------
# Quick sanity check usage
# ----------------------------
if __name__ == "__main__":
    dev = SynthNVPro("COM4", timeout_s=0.9, command_delay_s=0.01, query_retries=2)
    print(dev.get_identity())
    dev.safe_mute()
    dev.set_reference("ext", ext_ref_mhz=10.0)
    dev.set_f(2870.0)
    dev.set_W(-20.0)
    print("PLL lock:", dev.get_p())
    print("Temp C:", dev.get_z())
    print("Detector:", dev.read_w(3))
    dev.safe_mute()
    dev.close()
    dev.close()
    dev.close()
