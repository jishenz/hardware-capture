# TPS parsing and JSON round-trip

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

sequence.input_path to TPS, or sequence.json_input=true and input_path to an exported JSON.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

sequence.tps, sequence.json including raw bytes, sequence_summary.json and each table text. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

JSON intended for lossless round-trip must contain raw bytes. Matched TNT output requires the original TPS, not JSON.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check sequence -c projects/sequences/inspect/config.json
uv run hc run sequence -c projects/sequences/inspect/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `parse_sequence_to_json.ipynb`.
