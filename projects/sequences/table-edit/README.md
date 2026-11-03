# Edit a table with a waveform

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

An existing TPS and sequence.table_updates with table name, quantity and waveform or shared pulse overrides.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Edited TPS, round-trip JSON, summary, table text. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Target table must exist. Shared table edits require allow_shared=true; assignments address visible one-based TNMR events.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check sequence -c projects/sequences/table-edit/config.json
uv run hc run sequence -c projects/sequences/table-edit/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `apply_waveform_to_sequence_example.ipynb`.
