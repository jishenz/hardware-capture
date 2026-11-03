# Hardware Capture

Python tools for oscilloscope acquisition, RF instrument control, and captured-waveform analysis, extracted from DiamondGUI.

The main workflow is **prepare RF tables → upload through TNMR → trigger and capture with RIGOL → analyze the measured waveform**.

See the [complete project and input catalog](docs/project-catalog.md) for every script, notebook, numerical component, input format, and experimental assumption.

## Setup

Use Python 3.12 and run commands from this repository:

```powershell
uv sync
uv run pytest
```

The default environment supports offline analysis and fake-backed tests. Add the extras needed for your instruments:

```powershell
uv sync --extra scope --extra tnmr --extra notebooks
```

Live RIGOL access requires a suitable VISA runtime. TNMR automation requires Windows, Tecmag TNMR, and its registered COM server. These native applications and drivers are not installed by `uv`.

## Repository map

| Location | Purpose |
| --- | --- |
| `hardware_capture/oscilloscope/rigol_dho.py` | RIGOL DHO transport, trigger synchronization, waveform capture, and CSV I/O |
| `hardware_capture/oscilloscope/rf_demodulation.py` | RF demodulation and pulse-window extraction |
| `hardware_capture/oscilloscope/pulse_robustness.py` | Captured-pulse models and robustness calculations |
| `hardware_capture/oscilloscope/plotting.py` | Waveform diagnostics and heatmaps |
| `hardware_capture/oscilloscope/examples/` | Acquisition, calibration, and analysis scripts plus a connection notebook |
| `hardware_capture/rf/redstone/` | TNMR COM control, TPS sequence editing, waveform construction, and template pairs |
| `hardware_capture/rf/redstone/closed_loop_optimizer/` | TNMR upload / scope capture loop, phase calibration, and pulse optimization |
| `hardware_capture/rf/redstone/examples/` | Standalone sequence-editing and TNMR notebooks |
| `hardware_capture/rf/redstone/references/` | Selected Tecmag manuals and the sequence fixture used by examples/tests |
| `hardware_capture/config.py` | Minimal connection settings, with environment overrides |
| `docs/` | Workflow guide, extraction scope, provenance, and validation results |

Tests remain next to their subsystem. Generated outputs and notebook execution results are excluded from version control. Python imports use `hardware_capture`, independently of DiamondGUI's `hardware` package.

## Start here

Run the synthetic optimizer demo without connecting any instruments:

```powershell
uv run python -m hardware_capture.rf.redstone.closed_loop_optimizer.demo_stage2_offline
```

Print the Stage 1 configuration with its retained `DRY_RUN=True` default:

```powershell
uv run python -m hardware_capture.rf.redstone.closed_loop_optimizer.run_stage1_calibration
```

Inspect the scope capture options:

```powershell
uv run python -m hardware_capture.oscilloscope.examples.capture_time_window --help
```

For live operation and offline analysis, see [Workflows](docs/workflows.md). Read the [closed-loop guide](hardware_capture/rf/redstone/closed_loop_optimizer/README.md) before enabling its live entry points.

## Extraction status

This is an independent working-tree snapshot. DiamondGUI was left intact. The snapshot includes its current local oscilloscope test edits and required locally available sequence templates. It does not retain the original Git history or introduce a synchronization link back to DiamondGUI.

See [Extraction notes](docs/extraction.md) for the source mapping and omitted material. No new license or authorship assignment is introduced by this extraction; existing source and vendor attribution remain applicable.
