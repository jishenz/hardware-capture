# Refactor validation

Validated November 3, 2026 on Windows/Python 3.12. Validation reached **unit tests, fake-device integration, offline workflows, installed-package checks and live previews**. No physical instrument was connected or run.

## Evidence

| Check | Result |
|---|---|
| `uv run --no-sync pytest -q --tb=short` | **163 passed**, about 30 seconds |
| `uv sync --extra scope --extra tnmr` | Passed; installed CLI entry point |
| `uv run --no-sync hc list` | All 10 public workflows listed |
| All 26 project configs | Plan validation passed without devices |
| Python and notebook syntax | All source modules and 26 thin notebooks passed; saved notebook outputs empty |
| Documentation links | Local Markdown links resolve |
| Offline square/BB1 calibration | Passed using synthetic square and retained BB1 capture |
| Measured and ideal pulse analysis | Passed; produced data, JSON and plots |
| Offline optimization demo | Best loss 0.429710, down from initial 0.672945 with the configured gain/ripple model |
| Template preparation | All seven named-template presets exported/round-tripped, source bytes unchanged; inversion-phase uses explicit COM export config |
| Wheel build and isolated import | Passed outside checkout; all four TPS/TNT pairs included; no retired module paths in wheel |
| Original DiamondGUI files | All 65 manifest source hashes unchanged |

The original retained suite had 119 tests. Thirteen assertions tied to retired script parsers/globals were replaced by shared workflow contracts. The new suite includes the retained numerical/fake-device tests plus configuration inheritance and file-relative paths, target semantics, physical detuning equivalence, all waveform-family CSV round-trips, calibration artifact reuse, raw CSV versus fake-scope equality, amplitude ceilings, custom optimizer seeds, failure restoration, template round-trips, TNMR update/clear guards and every project preset.

A Windows Tk backend issue appeared during file-only figure rendering. File-only plots now construct matplotlib Figure objects without initializing Tk; interactive plotting remains an explicit option. The final suite passed after that fix.

## Hardware lifecycle review

| Surface | Entry / normal behavior | Failure behavior |
|---|---|---|
| Scope acquisition | Explicit live run; open, arm/read, close owned VISA session | Close owned session in finally, including failed open/read |
| TNMR workflow | Explicit live run; connect/open configured document and perform selected action | Close owned COM adapter in finally; an already-applied edit is not a transaction rollback |
| Live optimization | Explicit live run; verify table readback, reference phase, capture each candidate; leave best measured tables loaded | Attempt original-table restoration and TNMR stop; close owned connections |
| Offline sequence editing | Output copies only; strict matched payload check for binary TNT patching | Reject mismatched pair; inversion-phase emits a separate live export config |
| Offline API / previews | No device constructors called by plans; no hardware output files on preview | Configuration errors before dispatch where inputs are inspectable offline |

Timing, trigger polarity, scope sampling, RF attenuator/duty-cycle settings, physical calibration, native VISA/COM behavior and real amplifier response still require validation on the instrument setup. The live optimizer does not rewrite a TNMR sequence's timing, enforce a laboratory-wide duty cycle, or verify a hardware transfer function. The identity-response approximation is retained and documented. Checkpoints can be inspected; automatic resume is not provided.

Historical extraction evidence is recorded in Git history and the extraction manifest. Old module paths are intentionally removed after migration; current callers use the functional package layout.
