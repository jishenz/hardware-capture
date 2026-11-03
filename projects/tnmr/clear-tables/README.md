# TNMR clear-tables

LIVE: default invocation returns a preview. Add --live only when ready to connect.

## Input

tnmr.path to a working experiment; optional launch, parameters, tables, sequence_path and save_path.

The project config extends `configs/lab.example.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

tnmr.json snapshot/action report and optional saved TNT. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Clear only explicitly listed tables unused by the currently exported sequence. Used tables are rejected. A separate save_path is mandatory. COM cannot delete table definitions, so contents become zero. Other supported actions are zg, go, stop, abort and reset; these operate the instrument.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check tnmr -c projects/tnmr/clear-tables/config.json
uv run hc run tnmr -c projects/tnmr/clear-tables/config.json
```

For hardware execution install `uv sync --extra scope --extra tnmr`, configure a working experiment and measured calibration, then add `--live`. The lab profile contains placeholders and example calibration values, not current instrument settings.

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `tnmrcontrol_clean_sequence_tables.ipynb`.
