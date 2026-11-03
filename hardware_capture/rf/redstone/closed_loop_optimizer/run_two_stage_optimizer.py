"""Run BB1 phase referencing followed by live TNMR/scope optimization.

Edit only the settings in this first section on the lab computer. The script is
safe by default: DRY_RUN must be disabled and the live confirmation phrase must
be entered before TNMR or the oscilloscope is opened.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_capture.rf.redstone.closed_loop_optimizer.core import bb1_initialization
from hardware_capture.rf.redstone.closed_loop_optimizer.pipeline import (
    ClosedLoopConfig,
    run_closed_loop_optimizer,
)


# ---------------------------------------------------------------------------
# Lab settings
# ---------------------------------------------------------------------------

DRY_RUN = True
LIVE_RUN_CONFIRMATION = ""
REQUIRED_LIVE_RUN_CONFIRMATION = "ARM TNMR AND SCOPE CLOSED LOOP"

TNT_PATH = Path(r"C:\TNMR\data\closed_loop_xpi.tnt")
AMPLITUDE_TABLE_NAME = "ampmod_p180"
PHASE_TABLE_NAME = "phmod_p180"
LAUNCH_TNMR_IF_NEEDED = False

VISA_RESOURCE = "TCPIP::192.168.1.152::INSTR"
SCOPE_CHANNEL = "CHAN1"
ACQUISITION_WINDOW_START_S = -10e-6
ACQUISITION_WINDOW_END_S = 100e-6
PULSE_START_S: float | None = None  # None: detect once, then freeze it.
PULSE_DURATION_S = 60e-6

DEMODULATION_FREQUENCY_HZ = 601.0100113e6
BASEBAND_CUTOFF_HZ = 25e6
Q_SIGN = -1  # +1 for I+iQ, -1 for I-iQ.

# Replace both values with current, independently measured lab calibrations.
# The first converts scope volts to physical angular Rabi frequency. The second
# is the physical Rabi frequency produced by TNMR amplitude-table value 100.
SCOPE_OMEGA_RAD_S_PER_VOLT = 1_776_212.654495664
OMEGA_AT_AMPLITUDE_100_HZ = 123_968.40469314181

COEFFICIENT_COUNT = 1500
TABLE_ENTRY_COUNT = 6000
PHASE_REFERENCE_CAPTURES = 5

# One update means: capture current pulse, calculate GRAPE/Adam update, upload,
# then capture the updated pulse. Start with 0 in the lab bring-up sequence.
UPDATE_COUNT = 100
AMPLITUDE_LEARNING_RATE = 0.02
PHASE_LEARNING_RATE = 0.02
MAXIMUM_AMPLITUDE_COEFFICIENT_STEP = 0.02
MAXIMUM_PHASE_COEFFICIENT_STEP_RAD = 0.10

AMPLITUDE_ERRORS = tuple(index / 20.0 - 0.5 for index in range(21))
DETUNING_RATIOS = tuple(index / 20.0 - 0.5 for index in range(21))

RESULTS_ROOT = (
    REPO_ROOT / "hardware_capture" / "oscilloscope" / "results" / "tnmr_closed_loop"
)


def build_config() -> ClosedLoopConfig:
    """Translate the top-of-file settings into the checked run configuration."""

    import numpy as np

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
        update_count=UPDATE_COUNT,
        amplitude_learning_rate=AMPLITUDE_LEARNING_RATE,
        phase_learning_rate=PHASE_LEARNING_RATE,
        maximum_amplitude_coefficient_step=(
            MAXIMUM_AMPLITUDE_COEFFICIENT_STEP
        ),
        maximum_phase_coefficient_step_rad=(
            MAXIMUM_PHASE_COEFFICIENT_STEP_RAD
        ),
        launch_tnmr_if_needed=LAUNCH_TNMR_IF_NEEDED,
    )


def _print_dry_run_summary(config: ClosedLoopConfig) -> None:
    initialization = bb1_initialization(
        config.coefficient_count,
        pulse_duration_s=config.pulse_duration_s,
        omega_at_amplitude_100_rad_s=config.omega_at_amplitude_100_rad_s,
    )
    print("DRY RUN: no TNMR or oscilloscope command was sent.")
    print(f"TNMR file: {config.tnt_path}")
    print(
        "TNMR tables: "
        f"amplitude={config.amplitude_table_name}, phase={config.phase_table_name}"
    )
    print(
        f"Controls: {config.coefficient_count} amplitude + "
        f"{config.coefficient_count} phase coefficients -> "
        f"{config.table_entry_count} TNMR entries"
    )
    print(f"Pulse duration: {config.pulse_duration_s * 1e6:.6g} us")
    print(
        "Initial split-BB1 amplitude: "
        f"{100.0 * initialization.amplitude_fraction:.9g} percent"
    )
    print(f"Phase-reference captures: {config.phase_reference_captures}")
    print(f"Requested Adam updates: {config.update_count}")
    print(f"Results root: {config.results_root}")
    print(
        "For a live run, verify the README checklist, set DRY_RUN=False, and set "
        f"LIVE_RUN_CONFIRMATION={REQUIRED_LIVE_RUN_CONFIRMATION!r}."
    )


def main() -> int:
    config = build_config()
    if DRY_RUN:
        _print_dry_run_summary(config)
        return 0
    if LIVE_RUN_CONFIRMATION != REQUIRED_LIVE_RUN_CONFIRMATION:
        raise SystemExit(
            "Live run blocked. Set LIVE_RUN_CONFIRMATION to the exact phrase "
            f"{REQUIRED_LIVE_RUN_CONFIRMATION!r}."
        )

    result = run_closed_loop_optimizer(config)
    print(f"Completed run: {result.run_directory}")
    print(f"Frozen phase offset: {result.phase_reference.offset_rad:+.12g} rad")
    print(f"Best measured epoch: {result.best_epoch}")
    print(f"Best measured loss: {result.best_loss:.12g}")
    print("TNMR now contains the best measured amplitude and phase tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
