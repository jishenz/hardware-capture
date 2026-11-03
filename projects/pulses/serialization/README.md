# Waveform serialization

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

The same pulse conditions as generation, or canonical pulse CSV with pulse.kind=csv, path, duration_s and samples.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Pulse CSV, metadata JSON and command tables. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Canonical pulse.csv stores midpoint times, Hz and radians. pulse.json describes the sampled export; it is not a Waveform.from_dict specification. Low-level Waveform.write_json/ from_dict preserve analytical shape serialization (see below).

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check pulse -c projects/pulses/serialization/config.json
uv run hc run pulse -c projects/pulses/serialization/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `write_waveform_json_example.ipynb`.

Analytical shape round-trip (offline):

```python
from hardware_capture.pulses.waveforms import Chirp, Waveform
shape = Chirp(points=600, dwell_s=1e-7, sweep_width_hz=200000)
shape.write_json("chirp.json")
import json
restored = Waveform.from_dict(json.load(open("chirp.json")))
```
