# Inversion-Chirp sequence preparation

OFFLINE with the supplied config. For analysis/calibration, input.kind=scope switches to live capture and requires --live.

## Input

Shared pulse conditions and the inversion_phase template; optional sequence.parameter_updates.

The project config extends `configs/pulse.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Edited TPS, JSON, table text and tnmr_update.json. No hardware connection. The historical TPS/TNT pair is not byte-matched, so binary TNT patching is unavailable. To produce a TNT through TNMR, inspect the generated tnmr_update.json and run `uv run hc run tnmr -c <output>/tnmr_update.json --live`; this loads the TPS, synchronizes parameters/tables and saves a new file. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Assumptions

Duration must exceed t_mod_offset. Positive/negative inversion tables use opposite sweep signs. Phase-only templates require a constant envelope and derive f1_amp from calibrated Rabi Hz. Amplitude-modulated templates convert physical Rabi Hz to calibrated percent and set f1_amp=100.

## Run

From the repository root after `uv sync`:

```powershell
uv run hc check sequence -c projects/sequences/inversion-chirp/config.json
uv run hc run sequence -c projects/sequences/inversion-chirp/config.json
```

Open `notebook.ipynb` for the same workflow through Python. It contains only configuration and API calls. See [configuration reference](../../../docs/configuration.md) for units, input schemas and calibration chaining.

Replaces the former `apply_chirp_waveform_to_nvnmr_sensing_sequence_example.ipynb`.
