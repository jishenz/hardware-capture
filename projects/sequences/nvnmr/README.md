# NVNMR parameter and table editing

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

sequence.template=nvnmr; parameter_updates, table_updates and assignments.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

TPS, JSON, tables and a patched TNT copy. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Parameter and table names must already exist; use the summary or Python Sequence API to inspect names. Named templates provide matching TNT/TPS files.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check sequence -c projects/sequences/nvnmr/config.json
uv run hc run sequence -c projects/sequences/nvnmr/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `nvnmr_template_parameter_table_waveform_editing.ipynb`.
