# TNMR + RIGOL closed-loop X(pi) optimizer

This folder contains two operator-run entry points. Use the Stage-1-only script
first:

```powershell
uv run python hardware_capture\rf\redstone\closed_loop_optimizer\run_stage1_calibration.py
```

It uploads and captures BB1, calibrates the constant phase, takes an independent
verification capture, saves the final pulse and fixed-X(pi) heatmap, and exits
without any Adam update. It calls the same tested acquisition pipeline as the
full script but constructs the configuration with `update_count=0`, so the two
entry points cannot drift into different hardware behavior. On Windows it opens
the three diagnostic graphs after a successful live run unless
`SHOW_GRAPHS_AFTER_RUN` is disabled.

Use the full pipeline only after those results look correct:

```powershell
uv run python hardware_capture\rf\redstone\closed_loop_optimizer\run_two_stage_optimizer.py
```

For a fast, hardware-free check of the Stage-2 update direction, run:

```powershell
uv run python hardware_capture\rf\redstone\closed_loop_optimizer\demo_stage2_offline.py
```

This small deterministic demo uses 40 spline coefficients, 160 waveform
entries, a 7-by-7 robustness grid, and three updates. A synthetic capture adds a
known amplitude loss and phase distortion. `PASS` means a new synthetic capture
after the GRAPE/spline/Adam updates has lower fixed-X(pi) objective loss and
higher center, mean, and worst-case fidelity. It does not contact or validate
either instrument.

This standalone script implements the lab acquisition loop: configure it once, start it once, and let it upload TNMR tables, arm the
scope, send `ZG`, capture the RF waveform, calculate the heatmap and gradient,
and upload the next epoch.

The Stage-1-only outputs are:

- `stage1_phase_change.png`: measured phase before and after the frozen
  correction, plus phase residuals for all reference captures
- `stage1_final_pulse.png`: commanded versus independently measured amplitude
  and corrected phase, including the measured-area / `5*pi` ratio
- `stage1_final_heatmap.png`: the independent measured waveform compared with
  one fixed ideal X(pi) target
- `stage1_summary.json`: phase offset/change, first/last agreement, capture
  spread, all five BB1 residuals, amplitude repeatability, measured area, and
  center/mean/worst heatmap fidelity. It records whether the automatic phase
  checks passed but still requires graph review for full Stage-1 acceptance.

## How to accept Stage 1

Reaching normal completion means all automatic phase checks passed. With the
default configuration, first/last zero-phase disagreement is at most 10 deg,
capture-to-capture phase spread is at most 5 deg, and the largest corrected BB1
segment residual is at most 15 deg. A violation stops the run instead of
writing a misleading pass result.

Then inspect the three graphs. The corrected residual should be centered near
zero with the repeated captures tightly clustered. The independent pulse
should follow the reference captures without clipping, truncation, or a large
area error; its area / `5*pi` value should be near one. Finally, the heatmap
center should be high-fidelity and the BB1 robust region should have the
expected shape. These last judgments remain visible rather than being hidden
behind arbitrary amplitude or heatmap pass thresholds.

## What the two stages do

Stage 1 uploads a normal five-segment split BB1 X(pi) pulse. Its phase pattern
is `0, phi, 3*phi, phi, 0`, where `phi = acos(-1/4)`, and its duration weights
are `1:2:4:2:1`. The constant phase offset is measured independently from the
stable interiors of the first and last zero-phase segments across repeated
captures. The two estimates must agree. After correction, the three middle
segments must also match the BB1 phase pattern; this catches a wrong Q sign or
demodulation convention. That offset is then frozen.

The offset only rotates captured complex I/Q into the analysis frame. It is
never added to or subtracted from the TNMR phase table. No extra square pulse is
introduced, and the same BB1 remains the training initial condition.

This RF phase offset is not the unobservable global phase of a 2x2 unitary. A
constant RF phase changes the equatorial rotation axis (for example, X toward
Y), so it must be tied to the X-axis analysis frame even though process fidelity
itself is invariant to an overall unitary phase factor.

Stage 2 repeats this loop:

1. Arm the RIGOL in single-shot mode.
2. Send TNMR `ZG` only after the scope reports that it is armed.
3. Demodulate and resample the capture onto the 6000-entry table grid.
4. Apply the frozen analysis-frame phase correction.
5. Evaluate process fidelity against one fixed ideal X(pi) gate over the
   amplitude-error/detuning grid.
6. Calculate the analytical GRAPE gradient with respect to captured I/Q.
7. Convert that gradient to captured amplitude and phase, transpose it through
   the cubic-spline map, and apply an Adam step to the 1500 amplitude and 1500
   unwrapped-phase coefficients.
8. Wrap phase to degrees only for the 6000-entry TNMR upload, verify both tables
   by readback, and start the next capture.

The captured-waveform GRAPE derivative is exact. The first implementation uses
a straight-through identity approximation for the unknown hardware Jacobian
between a commanded table change and the next measured waveform change. Every
new capture closes that loop, but this approximation is still the main reason
to begin with small step limits and short runs.

## Lab checklist

Edit the lab settings at the top of `run_stage1_calibration.py` first. After
Stage 1 is accepted, copy those verified values to `run_two_stage_optimizer.py`.

- Set `TNT_PATH` to a disposable working copy of the intended TNMR document.
- Confirm that the sequence uses `AMPLITUDE_TABLE_NAME` and `PHASE_TABLE_NAME`
  for the same RF event and consumes exactly 6000 entries over 60 us (10 ns per
  entry with the default settings).
- Confirm that amplitude-table units are 0 to 100 and phase-table units are
  degrees. The script keeps internal phase unwrapped in radians.
- Replace `SCOPE_OMEGA_RAD_S_PER_VOLT` with the current fixed square-pulse
  voltage calibration. There is no per-capture peak normalization.
- Replace `OMEGA_AT_AMPLITUDE_100_HZ` with the current physical Rabi frequency
  produced by TNMR amplitude 100. This value sets both the initial BB1 amplitude
  and the detuning normalization. The first implementation assumes the TNMR
  amplitude scale is locally linear in these normalized units.
- Verify `DEMODULATION_FREQUENCY_HZ`, `Q_SIGN`, scope channel, acquisition
  window, and trigger level/slope on the lab setup.
- Leave `PULSE_START_S=None` only if threshold detection cleanly finds the
  pulse. The first detected pulse center defines a fixed 60 us interval that is
  frozen for every later capture. Otherwise, enter the known trigger-relative
  start explicitly.
- Make sure RF tune is off and the RF chain can safely tolerate the configured
  amplitude range.

Use this bring-up order:

1. In `run_stage1_calibration.py`, keep `DRY_RUN=True` and run once. It only
   prints the derived BB1 amplitude and configuration; it does not import a
   VISA backend or send COM commands.
2. On the lab computer, disable dry run in the Stage-1-only script, enter its
   exact confirmation phrase, and run. This exercises table upload/readback,
   repeated BB1 phase referencing, one independent measured heatmap, and
   best-table restoration with no optimizer update.
3. Inspect `stage1_summary.json`, `stage1_phase_change.png`,
   `stage1_final_pulse.png`, `stage1_final_heatmap.png`, and the TNMR readback.
4. Move the verified lab settings into `run_two_stage_optimizer.py`, run with
   `UPDATE_COUNT=1`, compare initial and final captures, and verify the
   update direction and size before attempting a longer run.
5. Increase the update count only after those checks pass.

## Outputs and recovery

Stage-1-only runs create timestamped folders under
`hardware_capture/oscilloscope/results/tnmr_stage1_calibration/`. Full runs use
`hardware_capture/oscilloscope/results/tnmr_closed_loop/`. Both locations are covered
by the repository's ignored oscilloscope-results path. A run stores the full
config, phase-reference captures, history, checkpoint, initial/best/final
measured controls, TNMR-ready CSV tables, and fixed-X(pi) heatmaps.

On normal completion, the script restores the best *measured* candidate to
TNMR, not an unmeasured final update. If phase referencing, capture, readback,
or optimization raises an error, it attempts to stop TNMR and restore the two
tables that were present before the run. The source TNT file is opened but not
saved or overwritten by this script.

This home computer cannot validate COM/VISA behavior. The no-hardware tests
cover spline construction, BB1 initialization, phase estimation, analytical
gradient direction, table round trips, and a fake end-to-end acquisition.
