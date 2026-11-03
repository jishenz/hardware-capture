# Workflows

## Offline analysis and calibration

Use the scripts in `hardware_capture/oscilloscope/examples/`:

- `calibrate_rf_from_square.py`: square-pulse voltage-to-rotation calibration.
- `calibrate_rf_from_bb1.py`: BB1-based calibration and phase diagnostics.
- `scope_rf_robustness_heatmap.py`: measured-pulse robustness analysis.
- `scope_rf_best_axis_xpi_heatmap.py`: best-axis X(pi) analysis.

Each supports `--help`. Review the input CSV and calibration values before running. A sample capture is included at `hardware_capture/oscilloscope/tests/pulse1_008.csv`. Individual scripts retain their original lab-specific defaults; these are not universal calibration values.

## Scope acquisition

After installing the `scope` extra and a VISA runtime, set the actual instrument address, channel, trigger, and desired capture interval. For example:

```powershell
uv run python -m hardware_capture.oscilloscope.examples.capture_time_window --ip YOUR_SCOPE_IP --channel 1 --t1=-1us --t2=3us
```

This command performs live acquisition. The connection notebook provides an interactive version of the workflow. Some acquisition scripts can launch a configured shot command; configure that command explicitly for the lab setup.

## Redstone / TNMR

`Sequence` and the waveform classes support offline TPS editing without opening TNMR:

```python
from hardware_capture.rf.redstone import Sequence

sequence = Sequence.load(
    "hardware_capture/rf/redstone/20260620_NVNMR_template.tps"
)
print(sequence.table_references())
```

The notebooks cover waveform construction, TPS/JSON export, paired TNT/TPS templates, and live COM parameter/table editing. Their saved outputs have been cleared. Notebooks that call `TNMRControl.connect()` or start acquisition are live workflows; inspect their file paths and settings first. Historical `C:\TNMR\data\...` inputs must be replaced with your own working files.

Run notebooks with this repository's kernel:

```powershell
uv sync --extra notebooks --extra tnmr --extra scope --extra windfreak
uv run jupyter lab
```

## Closed-loop capture and optimization

The [closed-loop guide](../hardware_capture/rf/redstone/closed_loop_optimizer/README.md) describes the existing Stage 1 phase calibration and Stage 2 optimizer. Both live entry points retain `DRY_RUN=True` and their confirmation checks. Configure the TNT path, amplitude/phase table names, scope settings, and physical calibration values in the entry-point script before lab use.

The synthetic demo and fake-backed tests validate software behavior only. The extraction has not been validated on physical instruments.

## Windfreak and NI analog output

```python
from hardware_capture.rf.windfreak.windfreaksynthnv import SynthNVPro
from hardware_capture.rf.windfreak.chirp import FMChirpAOcontrol
```

Connection settings can be overridden before starting Python:

```powershell
$env:WF_NV_PORT = "COM4"
$env:NI_CH_FM_ANALOG = "Dev1/ao3"
$env:NI_CH_FM_TRIG = "/Dev1/PFI1"
$env:TNMR_APPLICATION_PROGID = "NTNMR.Application"
```

The port and channels shown are inherited examples. The AO controller retains its 0–2.5 V clamp. The serial controller retains its existing RF gate/mute methods. The Windfreak tests folder also contains operator-run hardware diagnostics; these are outside default test discovery and have not been executed during extraction.

## Test selection

`uv run pytest` selects scope unit tests, Redstone offline/fake-COM tests, closed-loop fake-device tests, and the optional NI waveform logic test. The NI test is skipped when its Python dependency is absent. No hardware connection is required. Hardware/integration markers are excluded by default.
