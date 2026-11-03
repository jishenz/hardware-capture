# Nominal robustness analysis

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

input.kind=csv, scope or pulse; explicit target, scan.amplitude_errors (fractions), scan.detuning_hz (Hz). Measured input needs demodulation plus fixed volts-to-Rabi or explicit bb1_area scaling.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

analysis.npz (controls and physical scan axes), analysis.json, heatmap.png; measured-input pulse diagnostics and optional overlays. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Compares errors to the measured zero-error propagator; center fidelity is one by definition, even for an incorrectly calibrated gate. The demo uses a generated BB1 X(pi); for measured data switch input and set the calibration artifact. Optional analysis.amplitude_path/phase_path compare one-column ideal traces after stretching them over the detected window.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check analyze -c projects/analysis/nominal/config.json
uv run hc run analyze -c projects/analysis/nominal/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.
