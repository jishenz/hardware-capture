# Configuration and data contracts

Use `Experiment.load(path)` for JSON/TOML files, `Experiment({...})` for Python dictionaries, and `config.with_updates(section={...})` for programmatic overrides. `plan(workflow, config)` validates and previews; `run(workflow, config)` executes offline operations. Hardware execution requires `live=True` (CLI `--live`). Configuration loading, package imports and plans do not open devices.

## Inheritance and paths

```json
{
  "extends": ["pulse.json", "my-lab.json"],
  "pulse": {"duration_s": 0.00008},
  "optimization": {"updates": 20}
}
```

Tables merge recursively; arrays replace their inherited values. Paths are resolved relative to the file declaring each path, even across inheritance. Python dictionaries resolve relative paths from `Experiment(..., directory=...)`, defaulting to the current directory. Unknown top-level/section fields, nonfinite values and inheritance cycles are errors. A loaded config is not a guarantee that a connected device is available or physically configured correctly.

## Raw RF input and scope

| Field | Meaning |
|---|---|
| `input.kind` | `csv`, `scope`, or `pulse` for ideal-pulse analysis only |
| `input.path` | Raw two-column RF CSV for capture/calibration/analysis |
| `input.read_current` | Scope: true reads existing record; false arms a new single acquisition |
| `input.shots` | Positive integer, capture workflow only |
| `input.shot_command` | Optional trusted local shell command invoked after arming the scope; does not apply to existing-record reads |
| `scope.resource`, `channel` | Explicit VISA address and channel, e.g. `CHAN1` |
| `scope.window_start_s`, `window_end_s` | Requested raw acquisition interval in seconds |
| `scope.waveform_format` | Network waveform encoding: `WORD` (default, unsigned little-endian 16-bit), `ASC`/`ASCII` (text fallback), or `BYTE` (8-bit) |
| `scope.timeout_ms` | VISA command timeout |
| `scope.trigger_timeout_s`, `armed_timeout_s`, `poll_s` | Acquisition synchronization timeouts/polling |
| `scope.allow_clamp` | Ordinary capture may clamp a requested interval to available memory; default false |
| `scope.rearm` | Ordinary capture may rearm after reading; default false |

Raw CSV uses `time_s,chan1_voltage_V` (or another channel-voltage header). Comment rows beginning with `#` are accepted. Samples must have finite increasing uniformly spaced times and finite voltages. The first two columns are raw time in seconds and voltage in volts, not pre-demodulated I/Q or a TNMR command table. Capture saves that format. Raw acquisition assumes the scope trigger, sampling/memory and voltage scales are already configured.

The live optimizer uses its own synchronized single-shot sequence and strict window coverage; `read_current`, `shot_command`, `shots`, `allow_clamp` and `rearm` are ordinary-capture settings, not optimizer controls.

Capture and both live optimizers default to binary `WORD` transfer for the RIGOL
DHO4804. Time windows remain specified in seconds and are converted to point
indices from the fresh waveform preamble. The reader selects the format before
fetching that preamble, checks that the scope reports the requested encoding,
and converts binary codes to volts as `(code - YORigin - YREFerence) * YINCrement`.
VISA handles IEEE binary block headers and terminators; each block must contain
exactly the requested point count. The SCPI block size remains 100,000 points.
The VISA read buffer is 1 MiB for binary transfers.

Set `"scope": {"waveform_format": "ASC"}` in a profile to explicitly use the
previous text transfer. The scope's front-panel Save Wave `.bin`/`.csv` selection
does not control this network encoding. `WORD` preserves the 12-bit ADC codes
in a 16-bit container; `BYTE` has only eight bits and is not the default for RF
fidelity analysis. Instrument/firmware behavior and speed require human-operated
validation; offline tests cannot establish those physical results.

References: [RIGOL DHO1000/DHO4000 Programming Guide, section 3.29](https://www.batronix.com/files/Rigol/Oszilloskope/DHO1000/dho10004000_programmingguide_en.pdf#page=431)
and [PyVISA binary-transfer documentation](https://pyvisa.readthedocs.io/en/latest/introduction/rvalues.html#reading-binary-values).

## Demodulation and windows

- `demodulation.frequency_hz`: carrier used for mixing. Must match the capture.
- `cutoff_hz`: positive baseband low-pass cutoff below sampled Nyquist; default 25 MHz.
- `filter_order`: positive integer; default 4.
- `q_sign`: +1 or -1, default -1. Use one experimentally verified phase convention for a data chain. The historical BB1 example explicitly uses +1.
- `window.start_s`, `end_s`: optional measured-pulse bounds; omit both for detection. Live optimization uses start plus `pulse.duration_s` and freezes the timing after reference acquisition.
- `window.threshold_fraction`, `padding_s`: envelope detection threshold (default 0.1) and padding (default 0.25 microseconds).

The square calibration detects the flat pulse using its separate calibration threshold/smoothing parameters. BB1 and analysis use `window`. Live optimization has its own reference detection and consistency checks.

## Shared pulse conditions

| Field | Meaning |
|---|---|
| `pulse.kind` | `bb1`, `square`/`constant`, `sine`, `triangle`, `linear`, `chirp`, `hs1`, `sahs1`, `wurst`, `wursttanh`, `csv` |
| `duration_s` | Total pulse duration in seconds; required |
| `samples` | Uniform table sample count, default 6000 |
| `angle_deg` | BB1 target/square area angle; default 180. BB1 calibration may describe another measured angle; optimizer target is still X(pi) |
| `axis_phase_deg` | Command-frame phase offset, default 0 |
| `peak_rabi_hz` | Physical amplitude for shaped pulses; square may instead derive it from angle/duration |
| `max_rabi_hz` | Hard commanded amplitude ceiling; optimizer also enforces calibrated amplitude 100 |
| `parameters` | Shape-specific parameters, below |
| `path` | Canonical three-column CSV, for kind=csv |
| `amplitude_path`, `phase_path` | Alternative matching one-column amplitude and phase files |
| `amplitude_unit`, `phase_unit` | One-column import units: amplitude `Hz`, `kHz` or `auto`; phase `deg`, `rad` or `auto` |

BB1 derives its constant amplitude from total area `(4*pi + angle_rad) / duration`; it does not use peak_rabi_hz. Square derives angle/duration unless peak_rabi_hz is explicitly supplied. Pulse limits are checked, not silently rescaled. All waveforms have nonnegative physical amplitudes; phase carries sign/axis changes.

Chirp/HS1/SAHS1/WURST/WURSTtanh share `sweep_width_hz`, `center_hz`, `offset_hz`. HS1 adds `truncation`; WURST adds `order`; WURSTtanh adds `order` and `beta`. SAHS1 uses the shared peak Rabi rate for its nutation scale and retains the original analytical waveform implementation. Sine/triangle/linear expose the parameters of their waveform class. Do not duplicate `points`, `dwell_s`, amplitude mapping, phase or nutation rate inside parameters: they derive from shared pulse conditions. The low-level `hardware_capture.pulses.waveforms` API retains all original constructors, table quantization, display and JSON methods.

Canonical `pulse.csv` header is exactly `time_s,rabi_hz,phase_rad`. Times are **uniform midpoints**, `(i+0.5)*duration/N`, not endpoints. Import requires a matching duration and resamples to the requested sample count. Phase interpolation unwraps phase first. This file is distinct from a raw voltage capture. `pulse.json` describes a sampled export; analytical `Waveform.write_json`/`load_waveform_json` is a separate format.

TNMR exports use amplitude percent of `calibration.rabi_at_100_hz` and phase degrees modulo 360. PhMod frequency sweeps preserve the original negative-integral convention: requested physical RF frequency minus the offset already applied by the sequence. Align offset_hz with the actual sequence.

## Calibration

| Field | Meaning |
|---|---|
| `calibration.path` | Previously generated calibration.json |
| `omega_rad_s_per_volt` | Scope baseband volts to angular Rabi rad/s |
| `rabi_at_100_hz` | Physical Rabi frequency at TNMR amplitude 100 |
| `middle_fraction` | Square calibration: central plateau fraction, default 0.6 |
| `smoothing_s`, `threshold_fraction`, `baseline_guard_s` | Square detection/baseline settings |
| `pi_over_2_duration_s` | Optional BB1 flat-envelope duration cross-check |

Artifact values are loaded first; explicit fields override them. Square artifacts include both scales if the reference is at amplitude 100. BB1 area calibration determines the scope scale; it does not independently establish the amplitude-100 scale. Supply that separately for optimization/table export. The older square artifact field `peak_omega_over_2pi_khz` is also accepted.

## Analysis and scan axes

`analysis.target` is explicit: `nominal` compares against the measured zero-error propagator; `best_axis_pi` finds one best equatorial pi target and fixes it across the scan; `fixed_xpi` compares to ideal X(pi). These answer different physical questions.

`analysis.scaling=fixed` uses the calibrated scope scale; `bb1_area` normalizes the measured envelope to `4*pi + angle_rad` and is valid only for a complete BB1 capture. Ideal pulse input already uses physical units. `analysis.steps` controls measured waveform resampling (default 900), and `method` is `analytic` or `scipy_expm`. `phase_offset_rad` subtracts one constant measured frame offset before analysis.

`analysis.amplitude_path` and `phase_path` optionally overlay one-column ideal traces. Each is interpolated over the detected measured-pulse interval. The phase overlay removes one fitted constant phase offset for visual comparison only. Neither overlay performs trigger alignment or changes the simulated controls. Units can be given explicitly; auto mode uses headers/legacy conventions. Prefer explicit units for your files.

Scan axes can be explicit arrays (`scan.amplitude_errors`, `detuning_hz`) or min/max/count triples (`amplitude_min`, `amplitude_max`, `amplitude_count`; `detuning_min_hz`, `detuning_max_hz`, `detuning_count`). Arrays must be increasing. Amplitude error is a fraction, so 0.1 means +10%; it cannot be below -1. Detuning is always **Hz at the public boundary**. Internal optimizer ratios are derived using amplitude-100 calibration. Results include physical axes; live NPZ additionally retains legacy ratio axes.

## Optimization

`optimization.coefficients` is the cubic spline coefficient count (4 <= coefficients <= samples). `updates` is the number of updates after evaluating the initial candidate. The offline optimizer uses `amplitude_learning_rate` and `amplitude_step` in normalized amplitude-100 units. Hardware `optimize-live` uses `amplitude_logit_learning_rate` and `amplitude_logit_step` (both default 0.02) in unrestricted logit units, and rejects the old amplitude-space keys. The composite workflow retains its `composite` logit settings described below. Phase settings remain `phase_learning_rate` and `phase_step_rad` in radians.

Objective controls are `fidelity_threshold`, `threshold_smoothing`, `mean_weight`, `threshold_weight`, `cvar_weight`, `cvar_confidence`. The nonnegative mean/threshold/CVaR weights must sum to one. The loss combines mean infidelity, a smoothed threshold penalty and worst-tail CVaR. The target is fixed X(pi). The offline optimizer clips amplitude coefficients. Both hardware optimizers use `a_max * sigmoid(z)` and update unrestricted logits without post-update amplitude clipping. Seeds at an endpoint move inward by a fraction of `1e-6` to obtain finite logits.

Offline `simulation.amplitude_scale` (default 1) and `phase_ripple_rad` (default 0) describe an illustrative response. The offline loop retains its GRAPE/identity-response approximation. Both hardware loops use total-loss SPSA: evaluate the complete scalar objective independently after each perturbed command passes through acquisition and calibration, then estimate the command gradient from the difference of those losses. No GRAPE backward pass or additional sigmoid/spline Jacobian enters the hardware update.

The shared `spsa` section controls hardware perturbations:

| Field | Default | Meaning |
|---|---|---|
| `directions` | 1 | Independent +/- pairs averaged per Adam update |
| `seed` | 20260820 | Nonnegative integer for reproducible directions |
| `amplitude_logit_perturbation` | 0.08 | Initial displacement of amplitude logits |
| `phase_perturbation_rad` | 0.05 | Initial displacement of phase coefficients in radians |
| `perturbation_decay` | 0.101 | Scales decay as `(update_index + 1)^(-decay)`, with decay in [0, 1] |

Both parameter families move simultaneously with independent +/-1 signs. Each
pair shares one unchanged center. The gradient is `(loss_plus-loss_minus) /
(2 * scale * sign)` per parameter, averaged across directions before Adam.
Probe order alternates across updates/directions. Probe sizes are distinct from
Adam learning rates and maximum update steps. Hardware performance and suitable
perturbation sizes still require experimental validation.

Without composite fidelity acceptance, a run measures the initial seed, uses `2 * directions` candidate evaluations
per update, then restores and independently measures the best actual candidate
(initial or perturbed). The final Adam center is saved as unmeasured; it is
never reported as a measured best pulse. The reference-only/zero-update live
workflow reuses its initial measurement instead of taking a final extra one.
Composite always takes an independent final measurement. Each composite
candidate record contains all N square/target pairs, freshly calibrated before
complex averaging. For 10 updates and one direction, composite takes 22 records.
Original live phase-reference captures are additional to these candidate counts.
Dry-run plans report effective settings and candidate counts without connecting.

The composite project preset enables a measured ideal-condition acceptance rule:

| `composite` setting | Preset | Meaning |
|---|---|---|
| `minimum_center_fidelity` | 0.99 | Minimum fidelity at exactly zero amplitude error and detuning; both axes must contain zero |
| `center_penalty_weight` | 10.0 | SPSA adds this times `max(0, minimum_center_fidelity - center_fidelity)` to robustness loss |
| `maximum_backtracks` | 3 | After a failed full update, test half, quarter, and one-eighth changes |
| `verification_captures` | 3 | All independent final captures must meet the floor before success/export |

The floor is expressed as a fraction: 0.99 means 99%, and 99 is rejected.
The penalty, backtrack count, and verification count default to the values above
when the floor is configured. Omitting all four settings preserves legacy behavior.
The initial seed must pass. New centers are measured and accepted only when
they pass; exhausting retries retains the previous parameters and Adam state.
Best-candidate selection excludes subthreshold probes even if their loss is lower.
Final verification failure aborts export and invokes existing failure cleanup.
With acceptance enabled, capture bounds are
`1 + updates * (2 * directions + 1) + verification_captures` through
`1 + updates * (2 * directions + 1 + maximum_backtracks) + verification_captures`.
The accepted-center history keeps penalized search losses separate from raw
robustness losses; per-update acceptance JSON files preserve trial decisions.

Each successful update saves `epoch_NNNN_spsa.npz` with the center, both probe
parameter vectors, signs, scales, losses, capture order, and full gradient.
Every probe has its own raw and processed NPZ. History rows identify the selected
measured candidate and mean +/- losses; detailed per-direction losses remain in
the SPSA archive. Failed pairs do not advance Adam or its checkpoint. Checkpoints
save the next center and logit-space Adam moments; automatic resume remains
unsupported. Old amplitude-space checkpoints are not interchangeable.

Live `phase_reference_captures` must be >= 2. A BB1 reference is measured before the requested seed, so duration/limit must permit 5*pi area even if the desired seed is shorter-area square. `use_configured_seed=true` uses shared pulse conditions; false starts from the internal BB1 reference. TNMR working sequence duration, selected tables and sample counts must match the config. The optimizer changes table values, not TPS timing or attenuator settings. Reference diagnostics still need experimental review. Checkpoints retain numerical state for inspection; automatic resume is not implemented.

## Sequence and TNMR settings

### Repeated square/target optimization

`optimize-composite` uses the separate `composite` section. See its
[input, reduction and output contract](../projects/optimization/composite/README.md).
It reads one scope record containing all repetitions, calibrates each target
against its preceding square, then averages complex targets before each full-loss SPSA evaluation and Adam update.

| Field | Meaning |
|---|---|
| `composite.sequence_path` | Optional exported TPS or supported TNT for offline schedule extraction; defaults to `tnmr.path` |
| `parameter_source` | Literal parser field: `value` (default) or `current_value`; these names do not establish TNMR runtime semantics; repetitions come from the script loop |
| `parameter_overrides` | Explicit required values for existing script parameters; cannot override the repetition count; never writes TNMR parameters |
| `seed` | `existing_tables` (default), fitted to splines, or `configured` using shared pulse conditions and replacing both target tables, including their lengths |
| `alignment_search_s` | Positive timing search around each square onset; default 1 us |
| `capture_reference_start_s` | Optional first-square timestamp in scope coordinates; shifts the complete schedule while preserving pulse spacing. Composite lab preset: 0 s. Omit to use the sequence `Scope_Trig` origin. |
| `capture_window` | `sequence` derives the download bounds from the anchored sequence, overriding inherited scope bounds; enabled in the composite preset. Default `configured` preserves explicit scope bounds. |
| `capture_pre_margin_s` | Time before the first square in sequence mode; default 10 us; must exceed alignment search. |
| `capture_post_margin_s` | Time after the last target in sequence mode; default 20 us; must exceed alignment search. |
| `middle_fraction` | Reference plateau fraction; default 0.6 |
| `minimum_coherence` | Minimum phase coherence on that plateau; default 0.9 |
| `maximum_relative_spread` | Maximum (90th-10th percentile)/median plateau amplitude spread; default 0.2 |
| `amplitude_logit_learning_rate` | Adam learning rate for unrestricted amplitude logits; default 0.02 |
| `amplitude_logit_step` | Maximum per-update change of each amplitude logit; default 0.02; not amplitude fraction/percent |

Composite amplitude coefficients are `a_max * sigmoid(z)`, where `a_max` is the
configured maximum Rabi frequency divided by amplitude-100 calibration. The
SPSA perturbations are made directly in logit space and phase-coefficient space.
The full measured losses already include sigmoid decoding and spline expansion,
so neither `B.T` nor the sigmoid derivative is applied again. Physical
coefficients are decoded after every update;
there is no amplitude projection. Initial endpoint fractions are moved inward
by `1e-6` to obtain finite logits (the BB1 seed is unchanged). Explicit old
`optimization.amplitude_learning_rate`/`amplitude_step` keys are rejected for
this workflow; use the two `composite` logit controls instead. Checkpoints save
logits, decoded amplitudes, the maximum, and logit-space Adam moments.

The effective target duration comes from RF-gate events in the script, including
trailing gated events. For `existing_tables`, `pulse.samples` must match both
existing target tables; `configured` replaces them with `pulse.samples` entries.
For configured symmetric BB1 X(pi), the sample count must be divisible by 10:
exact phase steps form a fixed baseline, with spline phase corrections starting
at zero. The local preset selects BB1, 60 us, 6000 entries with 1500 coefficients per family, and the `value`
parameter field (6 repetitions in the saved script). The original table lengths
are preserved for failure restoration. Playback dwell still requires operator
verification; table timing metadata is not rewritten.
The square is required at amplitude 100 and phase zero;
`calibration.rabi_at_100_hz` sets its known Rabi frequency (160000 in the local
preset). A fixed `omega_rad_s_per_volt` and old BB1 phase-reference settings are
not used. The operator must configure a scope record covering all pairs and
their alignment margins. `scope.trigger_timeout_s` also bounds TNMR acquisition
completion waits. The saved 59.78 us event values plus 220 ns trailing gates
yield 60 us pulses; mismatched runtime parameters fail before
upload. No script timing is automatically rewritten.

### Sequence editing and TNMR control

Offline `sequence.template` selects `pulse_phmod`, `inversion_phase`, `inversion_ampm` or `nvnmr`. Alternatively supply `input_path` and optional matched `tnt_path`. `json_input=true` reads an exported sequence JSON instead of TPS. Templates are package resources; outputs are copies.

`sequence.parameter_updates` maps existing TPS parameter tokens to values. `table_updates` is a list of objects containing `name`, `quantity` (amplitude/phase), either `waveform` (analytical Waveform dictionary) or `pulse` (shared-condition overrides), and optional `allow_shared`, `phase_units`, `precision`. `assignments` is a list of `{line, event_number, table_name, slot_index}` objects; event numbers are one-based, slot_index is optional. `remove_unused=true` removes unused offline definitions. Named pulse templates derive timing/offsets from pulse conditions; inversion templates generate paired opposite sweeps. Phase-only templates derive f1_amp from shared physical Rabi amplitude and reject nonconstant envelopes. Matched TNT output requires exactly one matching embedded TPS payload.

Live `tnmr.path` is a working TNT document; `launch` permits starting TNMR if needed. `amplitude_table` and `phase_table` select optimizer tables. For the `tnmr` workflow, `action` is `inspect`, `update`, `clear_tables`, `zg`, `go`, `stop`, `abort`, or `reset`. Update supports `parameters` (dashboard names), `tables` (numeric lists or strings), optional `sequence_path`, and optional `save_path`. Clear requires explicit `clear_tables` names and a separate save_path; referenced tables are protected. It zeroes contents because the COM interface does not delete definitions. `inspect` opens and reads the configured document; it is a live operation. The low-level TNMRControl API retains save/export/status/table and dashboard operations.

## Outputs

`output.directory` is the parent of unique per-run folders. `show_plots` defaults false. Each successful run writes `resolved_config.json` and `result.json`; individual workflows add documented artifacts. A failure may leave partial files for diagnosis. No outputs overwrite input captures or templates. Hardware dry-run produces a plan without output files.

The historical `inversion_phase` TPS and TNT are not byte-matched: its preset emits TPS/JSON/tables and a `tnmr_update.json` configuration instead of patching binary TNT. Run `hc run tnmr -c <output>/tnmr_update.json --live` to load the sequence, synchronize parameters/tables and save a new TNT through TNMR. Other named templates also emit this optional config. Binary patched TNT outputs replace the embedded sequence; use TNMR export when dashboard synchronization is needed.

Sine/triangle `parameters` can set `cycles`, `offset_deg`, `phase_min`, `phase_max`; linear supports phase_min/phase_max. All constructors also retain their low-level quantization controls. Shared peak_rabi_hz maps normalized amplitudes into Hz.
