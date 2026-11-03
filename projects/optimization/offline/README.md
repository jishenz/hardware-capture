# Simulated X(pi) optimization

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

Shared pulse, calibration.rabi_at_100_hz, physical scan axes, optimization coefficients/update count/Adam steps; optional simulation gain and phase ripple.

The project config extends `configs/offline-demo.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Best pulse exports, history.json, final coefficient checkpoint.npz and result.json. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Target is fixed X(pi). BB1 or any supported pulse/CSV can seed the same cubic-spline optimizer. The simulation uses a simplified response and an identity response Jacobian; it does not establish hardware performance.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check optimize-offline -c projects/optimization/offline/config.json
uv run hc run optimize-offline -c projects/optimization/offline/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.
