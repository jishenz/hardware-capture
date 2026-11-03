"""
================================
NI-DAQmx AO controller for Windfreak SynthNV external FM input (HARD-LIMITED 0–2.5 V).

Key feature you asked for:
- "Finite burst per hardware trigger, auto-rearm, triggers over and over"
- Every trigger MUST restart at sample 0 (same phase / same start voltage)

Implementation:
- AO task uses FINITE samples per trigger
- Digital start trigger is set to retriggerable=True
- Buffer can be "tiled" so each trigger outputs K sweeps (K * sweep_time_s)

Notes:
- Retriggerable AO start triggers are supported on many NI devices (esp. X-series),
  but not all. If your device rejects retriggerable=True, you’ll need the
  counter-generated sample clock fallback (not included here unless you want it).

Safety:
- AO voltage range is clamped to [0.0, 2.5] V at all cost.
"""

from __future__ import annotations

import atexit
import os
from enum import Enum, auto
from typing import Literal, Optional

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode

# ============================================================
# HARD SAFETY LIMITS — DO NOT MODIFY
# ============================================================
V_MIN: float = 0.0
V_MAX: float = 2.5

TriggerMode = Literal["hardware_capture", "software"]
RunMode = Literal[
    "finite", "continuous"
]  # (continuous not used for retriggerable bursts here)
Direction = Literal["down", "up"]
Waveform = Literal[
    "sawtooth",  # monotonic linear ramp
    "triangle",  # periodic triangle
    "raised_cosine_monotonic",  # monotonic raised-cosine (smooth endpoints)
    "cosine_periodic",  # periodic cosine (smooth wrap)
]


class AOState(Enum):
    IDLE = auto()
    ARMED = auto()  # task started and waiting for triggers / outputting bursts
    RUNNING = auto()  # (not strictly distinguishable without extra reads; kept for API)
    DONE = auto()


class FMChirpAOcontrol:
    """
    Explicit NI-DAQmx AO controller for SynthNV FM input.
    """

    def __init__(
        self, ao_chan: str, trig_chan: str = None, trig_edge: str = Edge.RISING
    ):
        self._ao_chan = ao_chan

        # Trigger defaults
        if trig_chan:
            self.set_hardware_trigger(trig_chan, trig_edge)
        else:
            self._trigger_mode = "software"
            self._trig_src = f"/{ao_chan.split('/')[0]}/PFI0"  # best-effort default
            self._trig_edge = Edge.RISING

        # Burst defaults
        self._retriggerable: bool = False
        self._reps_per_trigger: int = 1

        # Run defaults
        self._run_mode: RunMode = "finite"

        # Waveform defaults
        self._waveform: Waveform = "raised_cosine_monotonic"
        self._direction: Direction = "down"

        # Timing defaults
        self._sweep_time_s: float = 3e-3
        self._sample_rate: float = 500_000
        self._n_samples: Optional[int] = None

        # Behavior
        self._prehold: bool = True
        self._hold_last_value: bool = True  # after each finite burst, hold last sample
        self._posthold_voltage: float = 0.0

        # Internal
        self._wave: Optional[np.ndarray] = None
        self._task: Optional[nidaqmx.Task] = None
        self._state: AOState = AOState.IDLE

        atexit.register(self._safe_close_task)

    # -----------------------------
    # Context manager
    # -----------------------------
    def __enter__(self) -> "FMChirpAOcontrol":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._safe_close_task()
        except Exception:
            pass

    # -----------------------------
    # Public properties
    # -----------------------------
    @property
    def state(self) -> AOState:
        return self._state

    @property
    def ao_chan(self) -> str:
        return self._ao_chan

    # -----------------------------
    # Safety / cleanup
    # -----------------------------
    def _safe_close_task(self) -> None:
        if self._task is None:
            self._state = AOState.IDLE
            return
        try:
            try:
                self._task.stop()
            except Exception:
                pass
        finally:
            try:
                self._task.close()
            except Exception:
                pass
            self._task = None
            self._state = AOState.IDLE

    def close(self) -> None:
        self._safe_close_task()

    # -----------------------------
    # Trigger configuration
    # -----------------------------
    def set_hardware_trigger(self, trig_src: str, edge: Edge = Edge.RISING) -> None:
        self._trigger_mode = "hardware_capture"
        self._trig_src = str(trig_src)
        self._trig_edge = edge

    def set_software_start(self) -> None:
        self._trigger_mode = "software"
        self._retriggerable = False  # retriggering is a HW trigger feature

    # -----------------------------
    # Burst configuration (what you asked for)
    # -----------------------------

    def set_burst_reps(self, reps_per_trigger: int = 1) -> None:
        """
        Set how many sweeps occur per trigger.
        Each "sweep" is exactly sweep_time_s long.
        Implementation: waveform buffer is tiled reps_per_trigger times, so
        each trigger outputs a finite burst of duration reps_per_trigger * sweep_time_s.

        This does NOT start the task; start() must be called.
        """
        reps = int(reps_per_trigger)
        if reps < 1:
            raise ValueError("reps_per_trigger must be >= 1")
        self._reps_per_trigger = reps
        self._wave = None  # invalidate cached waveform

    def enable_retriggerable(self, enable: bool = True) -> None:
        """
        If enabled, each trigger restarts the AO output at sample 0 (same phase),
        outputs the full buffer, then auto-rearms for the next trigger.
        """
        self._retriggerable = bool(enable)

    # -----------------------------
    # Timing configuration
    # -----------------------------
    def set_timing(
        self,
        sweep_time_s: float,
        sample_rate: float = 500_000,
        n_samples: Optional[int] = None,
    ) -> None:
        if sweep_time_s <= 0:
            raise ValueError("sweep_time_s must be > 0")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self._sweep_time_s = float(sweep_time_s)
        self._sample_rate = float(sample_rate)
        self._n_samples = int(n_samples) if n_samples is not None else None
        self._wave = None

    # -----------------------------
    # Behavior configuration
    # -----------------------------
    def set_prehold(self, enable: bool = True) -> None:
        self._prehold = bool(enable)

    def set_post_behavior(
        self, hold_last_value: bool = True, posthold_voltage: float = 0.0
    ) -> None:
        self._hold_last_value = bool(hold_last_value)
        self._posthold_voltage = float(posthold_voltage)

    # -----------------------------
    # Waveform selection
    # -----------------------------
    def set_waveform_sawtooth(self, direction: Direction = "down") -> None:
        self._waveform = "sawtooth"
        self._direction = direction
        self._wave = None

    def set_waveform_triangle(self, direction: Direction = "down") -> None:
        self._waveform = "triangle"
        self._direction = direction
        self._wave = None

    def set_waveform_raised_cosine_monotonic(
        self, direction: Direction = "down"
    ) -> None:
        self._waveform = "raised_cosine_monotonic"
        self._direction = direction
        self._wave = None

    def set_waveform_cosine(self, direction: Direction = "down") -> None:
        self._waveform = "cosine_periodic"
        self._direction = direction
        self._wave = None

    # -----------------------------
    # Direct output helpers
    # -----------------------------
    @staticmethod
    def clamp_voltage(v: float) -> float:
        return float(np.clip(v, V_MIN, V_MAX))

    def hold_voltage(self, v: float) -> None:
        v = self.clamp_voltage(v)
        with nidaqmx.Task() as t:
            t.ao_channels.add_ao_voltage_chan(
                self._ao_chan, min_val=V_MIN, max_val=V_MAX
            )
            t.write(v, auto_start=True)

    # -----------------------------
    # Waveform generation
    # -----------------------------
    def _npts_one_sweep(self) -> int:
        N = (
            self._n_samples
            if self._n_samples is not None
            else int(round(self._sweep_time_s * self._sample_rate))
        )
        return max(int(N), 2)

    def _build_one_sweep(self) -> np.ndarray:
        N = self._npts_one_sweep()
        u = np.linspace(0.0, 1.0, N, endpoint=False)  # [0,1)

        if self._waveform == "sawtooth":
            if self._direction == "down":
                v = V_MAX * (1.0 - u)  # start high -> end low
            elif self._direction == "up":
                v = V_MAX * u
            else:
                raise ValueError("direction must be 'down' or 'up'")

        elif self._waveform == "raised_cosine_monotonic":
            if self._direction == "down":
                v = 0.5 * V_MAX * (1.0 + np.cos(np.pi * u))  # starts Vmax, ends ~> 0
            elif self._direction == "up":
                v = 0.5 * V_MAX * (1.0 - np.cos(np.pi * u))  # starts 0, ends ~> Vmax
            else:
                raise ValueError("direction must be 'down' or 'up'")

        elif self._waveform == "cosine_periodic":
            if self._direction == "down":
                v = 0.5 * V_MAX * (1.0 + np.cos(2.0 * np.pi * u))  # start high
            elif self._direction == "up":
                v = 0.5 * V_MAX * (1.0 - np.cos(2.0 * np.pi * u))  # start low
            else:
                raise ValueError("direction must be 'down' or 'up'")

        elif self._waveform == "triangle":
            tri01 = 1.0 - 2.0 * np.abs(u - 0.5)  # [0,1], starts 0
            if self._direction == "up":
                v = V_MAX * tri01
            elif self._direction == "down":
                v = V_MAX * (1.0 - tri01)
            else:
                raise ValueError("direction must be 'down' or 'up'")

        else:
            raise RuntimeError(f"Unknown waveform: {self._waveform}")

        return np.clip(v, V_MIN, V_MAX).astype(np.float64)

    def build_waveform(self) -> np.ndarray:
        one = self._build_one_sweep()
        if self._reps_per_trigger > 1:
            one = np.tile(one, self._reps_per_trigger)
        self._wave = one
        return self._wave

    # -----------------------------
    # Run control
    # -----------------------------
    def arm(self) -> None:
        """
        Arms the task.

        - If software start: begins immediately (one burst) and ends (not typical).
        - If hardware start + retriggerable: waits for triggers forever; each trigger outputs one burst.
        """
        self._safe_close_task()

        if self._wave is None:
            self.build_waveform()

        # Prehold ensures a deterministic start voltage before the trigger arrives
        if self._prehold:
            # The start voltage should match the first sample for deterministic edges
            start_v = V_MAX if self._direction == "down" else V_MIN
            self.hold_voltage(start_v)

        wave = self._wave
        assert wave is not None
        N = int(wave.size)

        task = nidaqmx.Task()
        try:
            task.ao_channels.add_ao_voltage_chan(
                self._ao_chan, min_val=V_MIN, max_val=V_MAX
            )

            # Finite burst length = N samples at sample_rate
            task.timing.cfg_samp_clk_timing(
                rate=self._sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )

            # Replay buffer without CPU intervention (important for retrigger)
            task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            if self._trigger_mode == "hardware_capture":
                task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source=self._trig_src,
                    trigger_edge=self._trig_edge,
                )
                # The whole point:
                task.triggers.start_trigger.retriggerable = bool(self._retriggerable)

            task.write(wave, auto_start=False)
            task.start()

            self._task = task
            self._state = AOState.ARMED

        except Exception:
            try:
                task.close()
            except Exception:
                pass
            self._task = None
            self._state = AOState.IDLE
            raise

    def start(self):
        # alias function
        return self.arm()

    def wait_done(self, timeout_s: float = 5.0) -> None:
        """
        Wait for ONE burst to finish. (Useful for software start or debugging.)
        In retriggerable mode, this waits for the currently running burst (after a trigger) to complete.
        """
        if self._task is None:
            raise RuntimeError("No active NI task. Call arm() first.")

        try:
            self._task.wait_until_done(timeout=timeout_s)
        except Exception as e:
            if self._trigger_mode == "hardware_capture":
                raise RuntimeError(
                    f"Timed out waiting for burst to complete. If you haven't triggered yet, that's expected.\n"
                    f"Trigger source: {self._trig_src} edge={self._trig_edge} retriggerable={self._retriggerable}\n"
                    f"Make sure triggers occur AFTER arm(), check wiring + common ground."
                ) from e
            raise

        self._state = AOState.DONE

        if not self._hold_last_value:
            self.hold_voltage(self._posthold_voltage)

    def stop(self) -> None:
        """Stop the armed task (ends retriggering)."""
        if self._task is None:
            return
        try:
            self._task.stop()
        finally:
            self._state = AOState.DONE

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def summary(self) -> dict:
        w = self._wave if self._wave is not None else self.build_waveform()
        N = int(w.size)
        oneN = self._npts_one_sweep()
        t_step_s = float(1.0 / self._sample_rate)
        return {
            "ao_chan": self._ao_chan,
            "trigger_mode": self._trigger_mode,
            "trig_src": self._trig_src,
            "trig_edge": str(self._trig_edge),
            "retriggerable": bool(self._retriggerable),
            "reps_per_trigger": int(self._reps_per_trigger),
            "run_mode": self._run_mode,
            "waveform": self._waveform,
            "direction": self._direction,
            "sample_rate_Sa_s": float(self._sample_rate),
            "sweep_time_s": float(self._sweep_time_s),
            "sweep_time_s_per_tile": float(self._sweep_time_s),
            "N": N,
            "N_one_sweep": int(oneN),
            "t_step_s": t_step_s,
            "N_total_burst": N,
            "burst_time_s": float(N / self._sample_rate),
            "v_wave_min": float(np.min(w)),
            "v_wave_max": float(np.max(w)),
            "v_min": float(np.min(w)),
            "v_max": float(np.max(w)),
            "state": self._state.name,
        }


# ============================================================
# Example: retriggerable bursts (your requirement)
# ============================================================
if __name__ == "__main__":
    ao_chan = os.getenv("NI_AO_CH", "Dev1/ao0")
    trig_src = os.getenv("NI_TRIG_SRC", "/Dev1/PFI0")
    reps_per_trigger = int(os.getenv("CHIRP_REPS_PER_TRIGGER", "5"))
    sweep_time_s = float(os.getenv("CHIRP_SWEEP_TIME_S", "0.003"))
    sample_rate = float(os.getenv("CHIRP_SAMPLE_RATE", "500000"))

    ao = FMChirpAOcontrol(ao_chan)
    try:
        # Each trigger outputs `reps_per_trigger` sweeps and restarts at sample 0.
        ao.set_hardware_trigger(trig_src, edge=Edge.RISING)
        ao.set_burst_reps(reps_per_trigger)
        ao.enable_retriggerable(True)

        ao.set_waveform_raised_cosine_monotonic(direction="down")
        ao.set_timing(sweep_time_s=sweep_time_s, sample_rate=sample_rate)
        ao.set_prehold(True)

        print("Arming retriggerable chirp output with:")
        print(ao.summary())
        print(
            f"\nWaiting for triggers on {trig_src}. "
            f"Each trigger outputs {reps_per_trigger} sweep(s)."
        )
        input("Press Enter to arm...\n")
        ao.arm()
        input("Armed. Send triggers now, then press Enter to stop...\n")
    finally:
        ao.stop()
        ao.close()
        print("Stopped/closed.")
