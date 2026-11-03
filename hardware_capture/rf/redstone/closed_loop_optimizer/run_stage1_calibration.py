"""Run only BB1 phase calibration and its independent verification graphs.

This entry point can never perform an Adam update. Edit the settings below on
the lab computer, run it, and inspect the three Stage-1 PNG files before using
the full two-stage optimizer.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_capture.rf.redstone.closed_loop_optimizer.core import bb1_initialization
from hardware_capture.rf.redstone.closed_loop_optimizer.pipeline import (
    ClosedLoopConfig,
    run_closed_loop_optimizer,
)


# ---------------------------------------------------------------------------
# Stage-1 lab settings
# ---------------------------------------------------------------------------

DRY_RUN = True
LIVE_RUN_CONFIRMATION = ""
REQUIRED_LIVE_RUN_CONFIRMATION = "ARM TNMR AND SCOPE STAGE 1"
SHOW_GRAPHS_AFTER_RUN = True

TNT_PATH = Path(r"C:\TNMR\data\closed_loop_xpi.tnt")
AMPLITUDE_TABLE_NAME = "ampmod_p180"
PHASE_TABLE_NAME = "phmod_p180"
LAUNCH_TNMR_IF_NEEDED = False

VISA_RESOURCE = "TCPIP::192.168.1.152::INSTR"
SCOPE_CHANNEL = "CHAN1"
ACQUISITION_WINDOW_START_S = -10e-6
ACQUISITION_WINDOW_END_S = 100e-6
PULSE_START_S: float | None = None
PULSE_DURATION_S = 60e-6

DEMODULATION_FREQUENCY_HZ = 601.0100113e6
BASEBAND_CUTOFF_HZ = 25e6
Q_SIGN = -1

# Replace both placeholders with current, independently measured calibrations.
SCOPE_OMEGA_RAD_S_PER_VOLT = 1_776_212.654495664
OMEGA_AT_AMPLITUDE_100_HZ = 123_968.40469314181

COEFFICIENT_COUNT = 1500
TABLE_ENTRY_COUNT = 6000
PHASE_REFERENCE_CAPTURES = 5

AMPLITUDE_ERRORS = np.linspace(-0.5, 0.5, 21)
DETUNING_RATIOS = np.linspace(-0.5, 0.5, 21)

RESULTS_ROOT = (
    REPO_ROOT
    / "hardware_capture"
    / "oscilloscope"
    / "results"
    / "tnmr_stage1_calibration"
)


def build_config() -> ClosedLoopConfig:
    """Build a configuration with zero optimizer updates by construction."""

    return ClosedLoopConfig(
        tnt_path=TNT_PATH,
        amplitude_table_name=AMPLITUDE_TABLE_NAME,
        phase_table_name=PHASE_TABLE_NAME,
        visa_resource=VISA_RESOURCE,
        results_root=RESULTS_ROOT,
        scope_channel=SCOPE_CHANNEL,
        acquisition_window_start_s=ACQUISITION_WINDOW_START_S,
        acquisition_window_end_s=ACQUISITION_WINDOW_END_S,
        pulse_duration_s=PULSE_DURATION_S,
        pulse_start_s=PULSE_START_S,
        demodulation_frequency_hz=DEMODULATION_FREQUENCY_HZ,
        baseband_cutoff_hz=BASEBAND_CUTOFF_HZ,
        q_sign=Q_SIGN,
        scope_omega_rad_s_per_volt=SCOPE_OMEGA_RAD_S_PER_VOLT,
        omega_at_amplitude_100_rad_s=(
            2.0 * math.pi * OMEGA_AT_AMPLITUDE_100_HZ
        ),
        coefficient_count=COEFFICIENT_COUNT,
        table_entry_count=TABLE_ENTRY_COUNT,
        phase_reference_captures=PHASE_REFERENCE_CAPTURES,
        amplitude_errors=np.asarray(AMPLITUDE_ERRORS, dtype=float),
        detuning_ratios=np.asarray(DETUNING_RATIOS, dtype=float),
        update_count=0,
        launch_tnmr_if_needed=LAUNCH_TNMR_IF_NEEDED,
    )


def _print_dry_run_summary(config: ClosedLoopConfig) -> None:
    initialization = bb1_initialization(
        config.coefficient_count,
        pulse_duration_s=config.pulse_duration_s,
        omega_at_amplitude_100_rad_s=config.omega_at_amplitude_100_rad_s,
    )
    print("STAGE-1 DRY RUN: no TNMR or oscilloscope command was sent.")
    print(f"TNMR file: {config.tnt_path}")
    print(
        "TNMR tables: "
        f"amplitude={config.amplitude_table_name}, phase={config.phase_table_name}"
    )
    print(
        f"Initial split-BB1 amplitude: "
        f"{100.0 * initialization.amplitude_fraction:.9g} percent"
    )
    print(f"Phase-reference captures: {config.phase_reference_captures}")
    print("Adam updates: 0 (fixed by this Stage-1-only script)")
    print(f"Results root: {config.results_root}")
    print(
        "For a live run, verify the README checklist, set DRY_RUN=False, and set "
        f"LIVE_RUN_CONFIRMATION={REQUIRED_LIVE_RUN_CONFIRMATION!r}."
    )


def _show_graphs(run_directory: Path) -> None:
    start_file = getattr(os, "startfile", None)
    if start_file is None:
        print("Automatic graph opening is unavailable; open the PNG files manually.")
        return
    for filename in (
        "stage1_phase_change.png",
        "stage1_final_pulse.png",
        "stage1_final_heatmap.png",
    ):
        start_file(str(run_directory / filename))


def main() -> int:
    config = build_config()
    if DRY_RUN:
        _print_dry_run_summary(config)
        return 0
    if LIVE_RUN_CONFIRMATION != REQUIRED_LIVE_RUN_CONFIRMATION:
        raise SystemExit(
            "Live Stage-1 run blocked. Set LIVE_RUN_CONFIRMATION to the exact "
            f"phrase {REQUIRED_LIVE_RUN_CONFIRMATION!r}."
        )

    result = run_closed_loop_optimizer(config)
    print(f"Stage 1 completed: {result.run_directory}")
    print(
        "Measured phase offset / analysis-frame correction: "
        f"{math.degrees(result.phase_reference.offset_rad):+.9g} / "
        f"{-math.degrees(result.phase_reference.offset_rad):+.9g} deg"
    )
    print("No Adam update was calculated or uploaded.")
    print("Inspect stage1_summary.json and the three stage1_*.png graphs.")
    if SHOW_GRAPHS_AFTER_RUN:
        _show_graphs(result.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
