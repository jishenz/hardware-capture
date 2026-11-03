# Extraction validation

Validated on November 3, 2026, on Windows with an independent Python 3.12 environment.

## Results

| Check | Result |
| --- | --- |
| `uv sync --extra scope --extra tnmr` | Passed after subsystem removal; lockfile refreshed and removed subsystem dependencies uninstalled |
| `uv run --no-sync pytest -q --tb=short` | **119 passed** after subsystem removal |
| Stage 2 synthetic optimizer demo | PASS; objective loss decreased from 0.663474 to 0.430251 |
| Stage 1 and full optimizer default entry points | Both completed in dry-run mode; no instrument commands sent |
| All five scope CLI `--help` entry points | Passed from outside the repository |
| Original isolated package import check | Passed from outside the repository at extraction; surviving core modules also covered by the current test run |
| Python source and 18 remaining notebook code-cell syntax checks | Passed after subsystem removal |
| Notebook outputs | Cleared; execution counts reset and portable Python kernel metadata selected |
| Initial wheel build | Passed at extraction; obsolete build/wheel artifacts removed with the retired subsystem |
| Source repository verification | All 65 currently retained source file hashes and original source Git status unchanged |
| Removal audit | No retired subsystem directory or active code/dependency/documentation links remain; historical exclusion records retained |

The initial extraction test run had 118 passes and eight setup errors because Windows denied access to the existing system `pytest-of-jd200` directory. The project now uses `.pytest_cache/tmp` as its disposable pytest temporary directory. The original selected suite then passed all 126 tests. After the requested removal, the remaining suite passes all 119 tests. Parallel pytest invocations should use distinct `--basetemp` paths.

## Hardware behavior review

The extraction changes imports, paths, packaging, notebook setup, and local configuration. Instrument commands, timing, voltage limits, calibration defaults, and cleanup logic are inherited unchanged.

| Component | Default / activation | Retained cleanup or protection |
| --- | --- | --- |
| Closed-loop TNMR + RIGOL entry points | Dry run by default; live mode requires the existing confirmation string | Pipeline table readback, best measured candidate restoration, and error recovery paths retained |
| RIGOL scope | Connect/acquire only through explicit calls | Existing trigger synchronization, timeout, and resource cleanup behavior retained |
| TNMR COM wrapper | Explicit connection; fake COM used in tests | Single owner-thread dispatch and close behavior retained |

No migration-specific hardware behavior regression was identified by these checks. Validation reached **unit tests, fake-backed logic, isolated imports, and dry-run entry points**. No physical instrument connection or acquisition was performed. Live notebooks and hardware diagnostics were not executed. Lab calibration values and native driver/COM/VISA behavior still require verification on the instrument computer.
