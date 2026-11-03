# Square pulse amplitude calibration

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

Raw RF CSV or scope capture; demodulation frequency/cutoff/Q sign; known calibration.rabi_at_100_hz.

The project config extends `configs/synthetic-square.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

calibration.json with omega_rad_s_per_volt and rabi_at_100_hz; diagnostic PNG. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

The square reference was commanded at amplitude 100 and has a flat plateau with baseline samples outside it. The median of the middle plateau defines volts-to-Rabi. The bundled input is synthetic.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check calibrate-square -c projects/calibration/square/config.json
uv run hc run calibrate-square -c projects/calibration/square/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.
