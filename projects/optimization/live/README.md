# Live X(pi) workflow

LIVE: default invocation returns a preview. Add --live only when ready to connect.

## Input

Shared pulse/calibration/scan settings, TNMR working file and two table names, scope window, demodulation, phase_reference_captures >= 2.

The project config extends `configs/lab.example.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Reference diagnostics and captures; live mode also produces history, checkpoints and measured heatmaps. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

A complete BB1 5*pi reference must fit within duration and amplitude limit. Only X(pi) is supported. The TNMR sequence must play the selected tables with exactly the configured duration and sample count; this workflow does not rebuild its timing. Verify the working sequence and trigger setup first. Phase referencing uses repeated BB1 captures, and live optimization then uses the requested seed. Failed runs attempt original-table restoration and stop; successful runs leave the best measured candidate loaded.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check optimize-live -c projects/optimization/live/config.json
uv run hc run optimize-live -c projects/optimization/live/config.json
```

For hardware execution install `uv sync --extra scope --extra tnmr`, configure a working experiment and measured calibration, then add `--live`. The lab profile contains placeholders and example calibration values, not current instrument settings.

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.
