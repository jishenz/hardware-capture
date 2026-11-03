# Remove unused TPS tables offline

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

TPS input and sequence.remove_unused=true.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Cleaned TPS, JSON and remaining table texts. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Only table definitions with no parsed sequence references are removed. This edits an output copy and does not connect to TNMR.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check sequence -c projects/sequences/prune/config.json
uv run hc run sequence -c projects/sequences/prune/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `remove_unused_sequence_tables_example.ipynb`.
