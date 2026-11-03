# Workflows

## Analysis and calibration entry points

Use the scripts in `hardware_capture/oscilloscope/examples/`:

- `calibrate_rf_from_square.py`: square-pulse voltage-to-rotation calibration.
- `calibrate_rf_from_bb1.py`: BB1-based calibration and phase diagnostics.
- `scope_rf_robustness_heatmap.py`: measured-pulse robustness analysis.
- `scope_rf_best_axis_xpi_heatmap.py`: best-axis X(pi) analysis.

Each supports `--help`. Only the BB1 calibration entry point currently has a raw-waveform CSV input path, selected by the source-level `INPUT_CSV` setting. The square calibration and both heatmap entry points acquire from a live scope (or read its current stored trace); they do not provide a raw-waveform CSV CLI option. Their underlying numerical functions can be called with in-memory data. The optional heatmap amplitude/phase CSV files are intended-command overlays, not measured-waveform inputs.

A sample capture is included at `hardware_capture/oscilloscope/tests/pulse1_008.csv`. Individual scripts retain their original lab-specific defaults; these are not universal calibration values. Read the [complete input catalog](project-catalog.md) before selecting an entry point.

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
uv sync --extra notebooks --extra tnmr --extra scope
uv run jupyter lab
```

## Closed-loop capture and optimization

The [closed-loop guide](../hardware_capture/rf/redstone/closed_loop_optimizer/README.md) describes the existing Stage 1 phase calibration and Stage 2 optimizer. Both live entry points retain `DRY_RUN=True` and their confirmation checks. Configure the TNT path, amplitude/phase table names, scope settings, and physical calibration values in the entry-point script before lab use.

The synthetic demo and fake-backed tests validate software behavior only. The extraction has not been validated on physical instruments.

## Test selection

`uv run pytest` selects scope unit tests, Redstone offline/fake-COM tests, and closed-loop fake-device tests. No hardware connection is required. Hardware/integration markers are excluded by default.
