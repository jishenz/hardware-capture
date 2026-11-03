# Waveform generation

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

Shared pulse duration, sample count, peak_rabi_hz, axis_phase_deg and kind; shape-specific parameters.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Canonical pulse.csv; amplitude_khz.csv, phase_deg.csv; metadata; calibrated TNMR percent/degree tables. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Supported kinds: square/constant, bb1, sine, triangle, linear, chirp, hs1, sahs1, wurst, wursttanh, csv. Rabi amplitudes are Hz and nonnegative.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check pulse -c projects/pulses/waveforms/config.json
uv run hc run pulse -c projects/pulses/waveforms/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `waveform_example.ipynb`.
