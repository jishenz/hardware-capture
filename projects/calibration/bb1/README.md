# BB1 envelope area calibration

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

Complete raw RF BB1 capture; pulse.angle_deg; demodulation; optional manual window and pi_over_2_duration_s cross-check.

The project config extends `configs/sample-bb1.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

calibration.json with volts-to-Rabi conversion, baseline/window metadata and optional duration cross-check; diagnostic PNG. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

The full BB1 envelope has area 4*pi + target_angle. Include quiet baseline before/after. The bundled historical capture uses 90 degrees; this calibration input does not change the X(pi) optimization target.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check calibrate-bb1 -c projects/calibration/bb1/config.json
uv run hc run calibrate-bb1 -c projects/calibration/bb1/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.
