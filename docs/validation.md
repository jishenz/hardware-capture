# Extraction validation

Validated on November 3, 2026, on Windows with an independent Python 3.12 environment.

## Results

| Check | Result |
| --- | --- |
| `uv sync --extra scope --extra tnmr --extra windfreak` | Passed; independent editable installation and `uv.lock` created |
| `uv run --no-sync pytest -q --tb=short` | **126 passed**, including optional NI waveform logic tests |
| Stage 2 synthetic optimizer demo | PASS; objective loss decreased from 0.663474 to 0.430251 |
| Stage 1 and full optimizer default entry points | Both completed in dry-run mode; no instrument commands sent |
| All five scope CLI `--help` entry points | Passed from outside the repository |
| Seven core package imports using isolated Python (`-I`) | Passed from outside the repository |
| Python source and 19 notebook code-cell syntax checks | Passed |
| Notebook outputs | Cleared; execution counts reset and portable Python kernel metadata selected |
| Wheel build | Passed; required TPS templates, vendor TPS fixture, and capture CSV included |
| Source repository verification | All 89 copied source file hashes and source Git status unchanged |

The initial test run had 118 passes and eight setup errors because Windows denied access to the existing system `pytest-of-jd200` directory. The project now uses `.pytest_cache/tmp` as its disposable pytest temporary directory. The complete selected suite then passed. Parallel pytest invocations should use distinct `--basetemp` paths.

## Hardware behavior review

The extraction changes imports, paths, packaging, notebook setup, and local configuration. Instrument commands, timing, voltage limits, calibration defaults, and cleanup logic are inherited unchanged.

| Component | Default / activation | Retained cleanup or protection |
| --- | --- | --- |
| Closed-loop TNMR + RIGOL entry points | Dry run by default; live mode requires the existing confirmation string | Pipeline table readback, best measured candidate restoration, and error recovery paths retained |
| RIGOL scope | Connect/acquire only through explicit calls | Existing trigger synchronization, timeout, and resource cleanup behavior retained |
| TNMR COM wrapper | Explicit connection; fake COM used in tests | Single owner-thread dispatch and close behavior retained |
| Windfreak serial controller | Explicit device construction and output control | Existing mute/gate methods and limits retained; live serial diagnostics not run |
| NI FM analog output | Explicit start | 0–2.5 V clamp and stop/close behavior retained; waveform logic tested without output |

No migration-specific hardware behavior regression was identified by these checks. Validation reached **unit tests, fake-backed logic, isolated imports, and dry-run entry points**. No physical instrument connection or acquisition was performed. Live notebooks and hardware diagnostics were not executed. Lab calibration values and native driver/COM/VISA behavior still require verification on the instrument computer.
