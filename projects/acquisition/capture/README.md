# Raw RF capture

LIVE: default invocation returns a preview. Add --live only when ready to connect.

## Input

A configured VISA resource/channel and acquisition time window; input.shots and optional shot_command.

The project config extends `configs/lab.example.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

One time_s, chanN_voltage_V CSV per shot. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

The scope must already have suitable sampling, trigger, voltage scale and memory settings. Raw samples must resolve the RF carrier. read_current=true reads the existing record; otherwise the scope is armed for a new shot. shot_command is an optional trusted local shell command executed after arming.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check capture -c projects/acquisition/capture/config.json
uv run hc run capture -c projects/acquisition/capture/config.json
```

For hardware execution install `uv sync --extra scope --extra tnmr`, configure a working experiment and measured calibration, then add `--live`. The lab profile contains placeholders and example calibration values, not current instrument settings.

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `oscilloscope_connection_pipeline.ipynb`.
