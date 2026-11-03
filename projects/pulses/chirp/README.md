# Linear frequency sweep

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

pulse duration/samples/peak_rabi_hz; parameters.sweep_width_hz, center_hz and offset_hz.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Canonical pulse and TNMR table files. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

TNMR PhMod phase is the negative integral of requested frequency minus the sequence offset; offset_hz must match the sequence.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check pulse -c projects/pulses/chirp/config.json
uv run hc run pulse -c projects/pulses/chirp/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `chirp_waveform_example.ipynb`.
