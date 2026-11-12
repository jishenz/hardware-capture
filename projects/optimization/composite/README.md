# Repeated square/target X(pi) optimizer

This is the `optimize-composite` workflow. Both it and `optimize-live` use
paired full-loss SPSA gradients followed by Adam. Its notebook and default CLI invocation only preview.

## Input

One raw RF scope record per candidate contains **N repeated pairs**, each a
100%-amplitude, phase-zero square reference followed by the target to optimize.
The reference defines **160 kHz at amplitude 100%**. The repetition count comes
from the saved TNMR loop parameter (`N_repeat` in the current local script),
not an independent optimizer setting.

The local preset extends the ignored `configs/lab.local.json`. Create it from
`configs/lab.example.json` and set `tnmr.path` to the operator-selected sequence.
Both `P_aux` and `P(90)` are required to be **59.78 us**; each gate also has a
trailing **220 ns**, giving **60 us total** for both reference and target.
The preset selects the literal parser field **`value`** as requested by the
operator: `N_repeat=6`, `P_aux=59.78u`, and `P(90)=59.78u`. The duplicate
`current_value` fields contain 20, 10u, and 2u; their names do not establish
which values TNMR actually executes. The old `current`/`default` configuration
labels are replaced by `value`/`current_value` to avoid that unsupported claim.
All schedule parameters come from the selected field, without pulse-duration
overrides. Runtime TNMR readbacks must still agree before tables are uploaded.
This does not edit the TNT file or TNMR parameters.

With the saved `value` timing, the square starts at 7.22 us and target at
71.44 us relative to the sequence's `Scope_Trig` event; the pair period is
136.44 us. That event is not assumed to be scope waveform time zero.
The preset sets `composite.capture_reference_start_s=0.0`, matching the first
square edge near zero in the saved 20261106_172907 capture. Thus capture-time
expectations are square 0 us, target 64.22 us, and last target end 806.42 us
for six pairs. The preset now requests -10 us to 826.42 us. Each square must still
have exactly one rising half-height edge within +/-1 us of its expected time.
This is an explicit anchor for this setup, not unrestricted pulse detection;
recheck the anchor if the trigger setup changes. Omitting the option retains
the sequence-origin timing. This anchor does not change sequence parameters
or the scope trigger setup.

`composite.capture_window="sequence"` derives the download window from the
anchored sequence on every run: first square minus `capture_pre_margin_s`
(10 us) through last target plus `capture_post_margin_s` (20 us). Both margins
must exceed the edge-alignment search. The window therefore follows the parsed
repetition count and pulse durations, overriding inherited `scope.window_*`
bounds. Preview and `effective_config.json` show the derived bounds, which are
passed directly to the scope reader for initial, probe, and final captures.
The sample rate and table resolution are unchanged. Setting `capture_window`
to `configured` (the library default) retains explicit scope bounds instead.

### Initial optimized pulse

The auxiliary reference remains square at amplitude 100% and phase zero.
The **second pulse** starts from a standard symmetric BB1 pi pulse of **60 us**
with **6000 equal-duration samples**, giving a nominal sample dwell of **10 ns**.
Its constant Rabi frequency is 41.666667 kHz, or 26.041667% of the 160 kHz
amplitude-100 calibration. Its total rotation area is 5*pi, producing X(pi):

| Rotation | Duration | Samples | Phase (degrees) |
|---|---:|---:|---:|
| pi/2 | 6 us | 600 | 0 |
| pi | 12 us | 1200 | 104.477512186 |
| 2*pi | 24 us | 2400 | 313.432536558 |
| pi | 12 us | 1200 | 104.477512186 |
| pi/2 | 6 us | 600 | 0 |

`composite.seed="configured"` replaces both target tables with these 6000
entries, even if their old lengths differ, and checks full numeric readback
before the first capture. The first candidate preserves the exact phase steps;
it is not a smoothed spline approximation. Subsequent phases are the fixed BB1
baseline plus spline corrections initialized to zero. Adam optimizes 125
amplitude logits and 125 phase-correction coefficients (250 parameters total).
The coefficient count divides the 6000 samples exactly: 6000 / 125 = 48.
Spline controls overlap; this does not split the pulse into constant 48-sample blocks.
Original tables,
including their original lengths, are retained for failure restoration.

## Processing

For **every pair on every candidate evaluation**:

1. Demodulate the raw record and remove an off-pulse I/Q baseline.
2. Locate the square's rising half-height edge near its anchored capture time. Apply
   the measured time shift to the following target's sampling grid.
   Saved `schedule_offset_s` records the common origin correction;
   `timing_offsets_s` includes that correction and each measured residual shift.
3. Use the square's central 60% plateau to estimate its phase and amplitude.
   Reject missing, ambiguous, low-SNR, phase-incoherent, or uneven references.
4. Rotate that target by the negative reference phase and scale it by
   `2*pi*160000 / measured_square_amplitude`, obtaining angular Rabi I/Q in rad/s.
5. Average the corrected target **complex I/Q samples** across all repetitions.

Only this mean target waveform reaches spin propagation and the scalar loss.
The square is excluded from the target's X(pi) objective. Wrapped phase angles,
per-pair losses, and fidelities are not averaged. The target phase and area are
not fitted away.

The initial capture measures the exact configured BB1 seed. Each update then
measures opposite simultaneous perturbations of amplitude logits and phase
coefficients about the same center. Each probe goes through its own complete
N-pair calibration and averaging. Their complete losses give the SPSA gradient:
`g_j = (loss_plus - loss_minus) / (2 * c_j * Delta_j)`. All requested pairs must
succeed before Adam proposes an update. No GRAPE derivative or identity
hardware Jacobian is used.

### Ideal-condition fidelity acceptance (99%)

The project preset enables `composite.minimum_center_fidelity=0.99`. Both scan
axes must contain exact zero, so the check uses zero amplitude error and zero
detuning, not a nearby grid point. The initial measured seed must pass before
optimization starts.

SPSA uses `search_loss = robustness_loss + 10 * max(0, 0.99 - center_fidelity)`
with the configured `center_penalty_weight=10`. The original robustness loss
and its weights remain unchanged and are saved separately. The penalty guides
the search; measured acceptance enforces the floor.

After each Adam proposal, measure the proposed center. If its fidelity is below
99%, retry at half, quarter, and one-eighth of the original parameter change
(`maximum_backtracks=3`). Commit the first passing proposal. If all fail,
retain both previous parameters and Adam moments/step, and restore the last
accepted command tables. These checks enforce fidelity, not monotonic loss.
Temporary probes and failed trials can fall below 99%; they are not accepted
updates or eligible final pulses.

Select the lowest robustness loss only among passing seeds, probes, and accepted
centers. Before exporting a successful result, independently capture the chosen
pulse three times (`verification_captures=3`); every capture must pass. Failure
aborts export and follows the existing stop/completion/original-table cleanup.
This fidelity is computed from captured RF and the spin model.

`history.csv` plots accepted centers (or retained previous measurements after a
rejection) and the best passing candidate. Probe robustness losses and penalized
search losses are separate columns. Every trial has its own measurement archive;
`epoch_NNNN_acceptance.json` records trial scales and decisions. Checkpoints
include the accepted center's measurement path and fidelity.

The acceptance settings are enabled in this project's preset, including its
15-update child preset. Custom configurations that omit all four acceptance
settings retain the legacy unconstrained behavior.

The shared `spsa` defaults match DiamondGUI's initial scales: 0.08 in amplitude
logit units, 0.05 radians in phase coefficients, decay exponent 0.101, seed
20260820, and one direction per update. This project's preset overrides the
direction count to two, averaging two independent gradients from four probe
captures before each Adam update. Scales decrease with the update index;
probe order alternates. More directions average independent gradient estimates.
Passing probes are eligible for best-candidate selection along with measured,
accepted centers. At completion, the best passing candidate is restored and
verified without an update.

With N=6 and 10 updates, two directions give 54–84 scope records: one initial,
40 probes, 10–40 center trials, and three final verifications. Each record contains six pairs.
An incomplete/invalid pair retains available evidence but cannot update Adam
or advance the last complete checkpoint.

### Runtime and progress

Each candidate prints stage starts and elapsed times and writes
`<candidate>_timing.json`, including the current stage before blocking work.
Stages separate upload/readback, acquisition plus scope transfer, raw saving,
completion wait, demodulation, calibration, loss evaluation, and reduced saving.
A stage marked `running` in an interrupted run does not establish device state.
Progress is reported only at stage start and exit, with no periodic heartbeat
or background reporter. `progress.json` records the latest stage transition,
process ID, elapsed time, and UTC update timestamp. During a blocking stage,
the files retain the start record until the stage exits; they do not establish
whether a device call is advancing or the physical device state.
Failures and keyboard interruption are labeled explicitly for the active stage,
and successful completion prints the total updates and captures.

Raw NPZ files use lossless level-1 ZIP compression and remain readable with
`numpy.load`; the faster compression trades a modest file-size increase for
lower CPU time. Calibration searches only local timestamp slices and computes
off-pulse noise once per record. The forward-only loss batches spin steps and
uses ordered matrix products without GRAPE derivative calculations; the
scenario grid, waveform resolution, objective, and SPSA settings are unchanged.

An offline replay of the 20261106_173950 initial capture measured raw saving
12.81 -> 3.57 s, calibration 3.05 -> 0.63 s, and loss 1.88 -> 0.72 s.
Demodulation remained about 2.1 s. Raw and calibrated arrays were unchanged;
the loss differed by 1.1e-16 from floating-point product ordering. Raw storage
grew from 96.02 MB to 101.53 MB. These are local processing measurements,
not verified hardware acquisition or end-to-end run timings.

The previous -10 us to 4 ms read window contained 16,040,001 samples at 4 GS/s,
requiring 161 ASCII chunks. The sequence-derived six-pair window contains about
3.35 million samples (34 chunks), a 79% reduction. Actual transfer speedup still
requires operator-run timing evidence. The reader now defaults to 16-bit WORD
binary transfer; `scope.waveform_format="ASC"` selects the ASCII fallback.

Only the calibrated target pulses contribute to the spin-propagation loss.
However, reduction subtracts an I/Q baseline estimated from off-pulse samples
throughout the captured record before calibrating those targets. Shortening
the record changes that estimate. On the saved 20261106_173950 initial capture,
all six pairs passed with the short window; loss changed from 0.5386842949 to
0.5388169181. Holding the demodulated samples fixed while shortening the baseline
record reproduced the change to within 8e-10. No noise-window loss term was
introduced. Comparisons within a run use the same derived window.

### Sigmoid amplitude coefficients

Both hardware optimizers use sigmoid amplitude coefficients. The offline
optimizer retains its existing amplitude-space update. With normalized ceiling
`a_max = pulse.max_rabi_hz / calibration.rabi_at_100_hz`, the commanded
coefficient is `a_j = a_max * sigmoid(z_j)`. The waveform amplitude is `B @ a`,
where the nonnegative B-spline rows sum to one, so it stays within the same
bounds. Adam updates unrestricted `z`. SPSA measures its gradient directly
from complete perturbed losses; sigmoid and spline derivatives must not be
applied again. Both probe commands are sigmoid-decoded and bounded, including
when the current amplitude approaches its ceiling.

There is **no post-update amplitude clipping**. A numerically stable sigmoid
and derivative avoid exponential overflow/cancellation. Extreme finite logits
can round to endpoints in floating point. Independent finite-value/bounds
checks and TNMR table readback remain; they are not a measured RF safety limit.

Seed amplitudes are converted using the inverse sigmoid. Out-of-range
coefficients are rejected. Only at initialization, fractions within `1e-6` of
either endpoint are moved to that interior margin to obtain finite logits.
The 26.041667% BB1 seed is unaffected. Existing-table seeds use a bounded
least-squares spline fit before this conversion, not clipping of fitted
coefficients.

`composite.amplitude_logit_learning_rate` and `amplitude_logit_step` both default
to **0.02 in logit units**. The latter limits each Adam step, not the permitted
range of logits. A 0.02 step near the initial BB1 amplitude changes amplitude
by approximately 0.39 percentage points at `a_max=1`, not 2 percentage points.
The old `optimization.amplitude_learning_rate` and `amplitude_step` keys are
rejected for this workflow to prevent silent unit changes. Phase update
settings retain their existing meanings. Sigmoid gradients become small near
the bounds; numerical tests do not establish improved hardware convergence.

The current project preset overrides both learning rates to **0.04**:
`composite.amplitude_logit_learning_rate=0.04` and
`optimization.phase_learning_rate=0.04`. The amplitude-logit step cap is
**0.04**, with the phase step cap retained at **0.1 radians** and the
nominal-fidelity floor at **99%**. After comparing learning rates, a separate
three-seed offline comparison at fixed rates of 0.04 tested amplitude caps
0.02, 0.03, and 0.04. Cap 0.04 achieved lower best loss in all three seeds after
1000 updates, with more fidelity rejections. These ideal-model results do not
establish hardware performance. The preset requests 4000 updates.

## Output

Each run creates a timestamped output folder containing:

- `epoch_0000.npz` and `epoch_0000_raw.npz`: the initial measured seed.
- `epoch_NNNN_direction_DDD_plus/minus.npz` and corresponding `_raw.npz`:
  each probe's raw RF, corrected targets, mean and complex standard deviation,
  per-pair calibration, command parameters, loss, fidelity grid, and scan axes.
- `epoch_NNNN_spsa.npz`: center, signs, scales, plus/minus parameters and losses,
  capture order, and the mean full-loss gradient.
- `final_verification.npz`, `final_verification_002.npz`,
  `final_verification_003.npz`, and their `_raw.npz` files: independent captures
  after restoring the best passing candidate. Summary reports all verification
  losses/fidelities separately from selection loss.
- `composite_schedule.json`, resolved/effective configuration, and original
  numeric TNMR tables for diagnosis or operator-led recovery.
- `initial_seed/`: initial pulse CSV/metadata and TNMR amplitude/phase text
  tables. Epochs and checkpoints also retain `phase_baseline_rad`, which must
  be added to the spline-expanded phase correction to reconstruct commands.
- Epochs/checkpoints contain `amplitude_logits`, decoded
  `amplitude_coefficients`, `amplitude_maximum`, and
  `amplitude_parameterization="sigmoid"`. Checkpoint moments are named
  `amplitude_logit_first` and `amplitude_logit_second` and belong to logit space.
  With fidelity acceptance enabled, checkpoints link the accepted center's
  measured fidelity and retain its parameters and moments after a rejected update.
  Legacy unconstrained checkpoints mark the proposed center as unmeasured.
  Old amplitude-space checkpoints are not interchangeable; automatic resume
  remains unsupported.
- `history.csv`, `checkpoint_latest.npz` after updates, `best_heatmap.png`, and
  summary/result JSON.
  All fidelity heatmaps use a fixed color bar from 0.99 (bottom) to 1.00 (top),
  including optimizer diagnostics. Values below 0.99 use the lowest color;
  saved fidelity arrays and objective calculations retain their full values.
- Best measured candidate exported as `pulse.csv`, pulse metadata, and TNMR
  amplitude/phase text tables. A successful live run leaves those best tables
  loaded; it does not save or rewrite the source TNT.

## Assumptions

The parser supports a single parameter-controlled loop, exactly two RF gates,
and a preceding scope trigger. Unsupported or ambiguous scripts are rejected.
The supported embedded TNT PSEQ format is checked; an exported TPS can instead
be supplied using `composite.sequence_path`. Runtime checks compare resolved
parameters and reference phase tables with TNMR. They do not prove that a
separately edited event matrix matches the saved file.

Reference and target must share the calibrated measurement chain and RF gain
within each pair. The configured carrier and Q sign must be correct. The scope
must already have suitable triggering, RF sampling rate, record memory, and
voltage range for the entire record. No waveform interpolation recovers missing
RF bandwidth. The sequence's table playback/dwell, the trailing 220 ns behavior,
and physical timing still require operator validation. The algorithm assumes
each target table spans the full 60 us gate, including its trailing event.
The operator must verify that the 6000-entry replacement plays over that gate
with the intended 10 ns nominal dwell. Table-value/length readback does not
verify playback rate; this workflow does not change table timing metadata.

The objective remains fixed X(pi), with existing robustness weights. Full-loss
SPSA includes the measured hardware response through both perturbed losses.
No separate BB1 phase-reference capture or fixed volts-to-Rabi artifact is used.
`phase_reference_captures`, old `window` settings, and `use_configured_seed` do
not control this workflow. To choose a generated seed explicitly, use
`composite.seed="configured"` and the shared `pulse` fields.

Acquisition completion must be reported before initial uploads and after every
capture. Failure after modification attempts Stop, checks completion, then
attempts to restore original tables. If completion cannot be confirmed, table
restoration is skipped and recorded in `cleanup_errors.json`. None of these
software statuses establishes a physically safe state. Checkpoints are for
inspection, not automatic resume.

## Run

### BB1 experiment (human operator; 4000 updates by default)

`run_15_epochs.py` now uses `config.json`: 4000 SPSA updates starting from
configured symmetric BB1 X(pi), 60 us, 6000 samples, and 125 amplitude plus
125 phase coefficients. The current saved sequence resolves six square/target
pairs per record. With the 99% acceptance rule, two directions mean 20004–32004
records: initial seed, 16000 probes, 4000–16000 center trials, and three independent
final verifications. Previews report these bounds; summaries report actual counts.
The legacy script filename is retained for existing commands. It honors
`optimization.updates` in the selected configuration; there is no early
convergence stop. The optional `--config projects/optimization/composite/config-15-epochs.json`
still selects the original short 15-update experiment explicitly.

From the repository root, preview the exact instrument/document/table settings:

```powershell
.\.venv\Scripts\python.exe projects/optimization/composite/run_15_epochs.py
```

Before hardware execution, the human operator must approve the exact scope
resource/channel, TNMR hardware/document/table ownership, intended uploads and
acquisitions, 6000-entry playback over 60 us (10 ns nominal dwell), approved
safe state, laboratory setup, and cleanup prerequisites. Read the exact scope
resource/channel and TNMR document from the local profile; target tables are
`ampmod_p180` and `phmod_p180`. Those configuration labels are not independent
physical evidence. Agents must never execute the live command, even after
human approval. After approval, the human operator may run:

```powershell
.\.venv\Scripts\python.exe projects/optimization/composite/run_15_epochs.py --live
```

The run prints its output folder and per-epoch losses. `history.csv` and
`loss_curve.png` update after each completed epoch. Epoch zero is the initial
BB1; subsequent epochs show the measured accepted center or the previous accepted
measurement retained after rejection. The plot also shows each direction-mean
probe robustness loss, best loss meeting the fidelity floor, and the last final
verification marker. All final verification measurements remain in the CSV.
A failed pair
leaves the last complete loss curve, CSV, available raw records, and checkpoint.

On success the best measured target tables remain loaded. On failure the
workflow attempts Stop, checks reported completion, then attempts original-table
restoration; skipped/failed cleanup is recorded. The operator must review the
readbacks and logs and apply the approved physical safe state as required.
Software completion does not verify a physically safe RF state.

Regenerate a curve from a saved or partial run without device access:

```powershell
.\.venv\Scripts\python.exe projects/optimization/composite/run_15_epochs.py --plot-only "outputs/<run-folder>"
```

For penalized-loss, nominal-fidelity and parameter-step plots, and a paired-seed
comparison of current versus half learning rates in a fixed numerical model,
see [optimizer diagnostics](../../../docs/optimizer-diagnostics.md).

### General workflow

From the repository root, these commands are offline and never connect:

```powershell
uv run hc check optimize-composite -c projects/optimization/composite/config.json
uv run hc run optimize-composite -c projects/optimization/composite/config.json
```

Open `notebook.ipynb` for the same preview through Python. Hardware execution
requires an explicit human decision to add `--live` (or Python `live=True`).
Before doing so, the human operator must confirm the exact devices/channel,
working script, approved safe state, laboratory setup, and cleanup procedure.
Agents must not perform that hardware run. Synthetic/fake-device tests do not
validate physical timing, calibration, wiring, duty cycle, or safety.

## Software validation

### DiamondGUI resolution settings (2026-11-06)

At the time of this validation, the local/example hardware profiles used 1500
coefficients per family and 6000 samples. Device-free previews for composite,
live, and phase-reference resolved 3000 adjustable parameters and 6000 samples.
The current composite preset overrides this with 125 coefficients per family.
The composite BB1 remains
60 us at 26.041667% amplitude, with nominal 10 ns sample dwell. Three focused
BB1 initialization, table-length restoration, and invalid-grid tests passed
in 4.52 s with physical-device constructors blocked. Notebook JSON/code-cell
syntax, scoped Ruff check/format-check, and `git diff --check` passed.
These checks do not verify physical table playback or dwell.

### Full-loss SPSA rewrite (2026-11-06)

The regression selection `tests/optimization` plus `tests/test_workflows.py`
passed **127 tests in 57.82 s**. The run used fake devices and numerical
responses, with TNMR/RIGOL constructors replaced by raising guards, pytest
plugin autoload disabled, and a separate writable test-output directory.
Coverage includes complete scalar losses with an extra penalty, objective
agreement with the existing numerical core, exact initial BB1 commands,
fresh calibration for both probes, logit bounds, capture order, checkpoint
moments, restored-best commands, and failed-pair checkpoint preservation.

Device-free plans resolved the composite preset to 6 repetitions, 300 optimized
parameters, 2000 waveform samples, and 22 candidate evaluations for 10 updates.
The live preview also selects total-loss SPSA and sigmoid amplitude parameters.

Ruff checks pass for the SPSA module, composite workflow, and their updated
integration tests. Nine edited numerical/workflow/test files pass format-check.
The remaining 20 lint findings in shared files were compared with their HEAD
versions; none is newly introduced. `git diff --check` passed. No physical
hardware, real-device test, or hardware-capable live entrypoint was run.

The historical validation notes below predate the full-loss SPSA rewrite.
Current SPSA tests cover scalar-loss gradients, forward-objective equivalence,
probe calibration, sigmoid bounds, failed-pair checkpoint preservation, and
fake-device live/composite workflows. These are software checks only.

### Sigmoid amplitude update (2026-11-05)

```powershell
uv run --no-sync --with pytest python -m pytest tests/optimization/test_amplitude.py tests/optimization/test_composite.py tests/optimization/test_core.py tests/optimization/test_pipeline.py tests/test_workflows.py -q
uv run --no-sync --with ruff ruff check hardware_capture/optimization/amplitude.py hardware_capture/workflows/composite_optimize.py tests/optimization/test_amplitude.py tests/optimization/test_composite.py
uv run --no-sync --with ruff ruff format --check hardware_capture/optimization/amplitude.py hardware_capture/workflows/composite_optimize.py tests/optimization/test_amplitude.py tests/optimization/test_composite.py
uv run --no-sync python -m py_compile hardware_capture/optimization/amplitude.py hardware_capture/workflows/composite_optimize.py hardware_capture/configuration.py tests/optimization/test_amplitude.py tests/optimization/test_composite.py
```

Results: **94 tests passed in 51.62 s**, and the listed lint, formatting, and
syntax checks passed. Coverage includes the complete sigmoid/spline/GRAPE
chain-rule gradient against finite differences at two amplitude ceilings,
extreme logits, initial endpoint treatment, synthetic bounded optimization,
logit-space Adam moments, checkpoint-to-upload consistency, and invalid or
legacy-unit configuration rejection. The saved-file preview still resolves
6 pairs, 60 us, 2000 samples, and unchanged 26.041667% BB1 amplitude.

Checks including shared `configuration.py` still report its five pre-existing
lint findings and pre-existing formatting differences; unrelated lines were
not changed. Scoped `git diff --check` passed. Validation used the same uv
executable noted below. No hardware was accessed; hardware convergence and
physical RF bounds remain unverified.

### Prior value-field and BB1 update

The value-field/BB1 update adds checks for exact five-block phase tables, the
nominal X(pi) propagator, upload-before-capture, 6000-to-2000 table replacement,
original-length restoration after failure, and invalid BB1 duration/grid.
The focused regression command below was rerun: **84 passed in 51.59 s**,
including **17 composite tests**. Ruff check and format-check passed for the
three changed Python files. The updated notebook passed JSON/code-cell syntax
checks without execution, and scoped `git diff --check` passed.

The offline pulse-generation API (`run("pulse", config_path)`) also exported
the requested 60 us / 2000-sample seed to a new timestamped `outputs` folder.
This generated files only; no TNMR upload or instrument operation was performed.

### Earlier composite implementation (2026-11-05)

Commands below used `C:/Research/diamondgui/.tmp_uv_two_stage_live/uv.exe`
because `uv` was not on this session's PATH.

```powershell
uv run --no-sync --with pytest python -m pytest tests/optimization/test_composite.py tests/optimization/test_core.py tests/optimization/test_pipeline.py tests/test_workflows.py -q
```

Result: **80 passed** (49.86 s), including 13 new composite tests. Coverage
includes noisy pair-dependent gain/phase/timing, averaging before GRAPE, raw RF
demodulation, both 220 ns trailing events, parameter-field selection,
duration mismatch, reference-table ownership, incomplete capture, bad reference,
unknown acquisition state, fake-device cleanup, and device-free public preview.

```powershell
uv run --no-sync --with ruff ruff check hardware_capture/optimization/composite.py hardware_capture/workflows/composite_optimize.py tests/optimization/test_composite.py
uv run --no-sync --with ruff ruff format --check hardware_capture/optimization/composite.py hardware_capture/workflows/composite_optimize.py tests/optimization/test_composite.py
```

Both passed. The same checks including changed `hardware_capture/api.py` and
`hardware_capture/configuration.py` reported six pre-existing lint findings and
pre-existing formatting differences in those two shared files. Their HEAD
versions were checked to confirm the baseline; unrelated code was not reformatted.

An offline `api.plan("optimize-composite", config_path)` using the actual local
TNT originally resolved 20 pairs using the then-selected `current_value` field.
The updated preview now resolves 6 pairs from `value`, both pulses 60 us,
and a 2000-sample BB1 seed at 26.041667%. Notebook JSON
and all code cells passed syntax parsing without execution. Scoped
`git diff --check` passed. No physical device, real-device test, or live optimizer
was run. Physical calibration and timing remain unverified.
