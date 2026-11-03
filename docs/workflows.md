# Common workflows

## Calibrate once, analyze multiple captures

```python
from hardware_capture.api import run
from hardware_capture.configuration import Experiment

cal = run("calibrate-square", "projects/calibration/square/config.json")
# This demo uses a synthetic square; use your own measured reference in the lab.
analysis = Experiment.load("configs/synthetic-square.json").with_updates(
    calibration={"path": cal["calibration_path"]},
    analysis={"target": "fixed_xpi", "scaling": "fixed"},
)
result = run("analyze", analysis)
```

For repeated captures, update only `input.path`. Set `input.kind="scope"` and add your scope profile to run the same analysis on an acquisition; use `live=True`. Choose a target appropriate to the question rather than interpreting nominal fidelity as X(pi) fidelity.

## Change pulse conditions without editing optimizer code

```python
config = Experiment.load("configs/offline-demo.json")
config = config.with_updates(
    pulse={"kind": "square", "duration_s": 80e-6, "max_rabi_hz": 100000},
    scan={"detuning_min_hz": -20000, "detuning_max_hz": 20000},
)
result = run("optimize-offline", config)
# Feed the optimized waveform into the same generator or analyzer.
optimized = config.with_updates(
    pulse={"kind": "csv", "path": result["pulse_path"]},
    input={"kind": "pulse"},
)
analysis_result = run("analyze", optimized)
```

Live optimization uses the same pulse/scan/optimization sections plus a configured working TNMR file, table names, scope and calibration. Copy/extend `configs/lab.example.json`, then check the resolved plan. Sample count and duration must match the actual TNMR sequence. Phase referencing remains BB1 even with another initial guess.

## Prepare RF sequences offline

```powershell
uv run hc run sequence -c projects/sequences/inversion-hs1/config.json
```

This produces TPS, table exports, JSON and a matched TNT copy. Review the generated sequence summary and actual instrument timing before loading it through the TNMR workflow. For custom tables use `sequence.table_updates`; for custom phase/amplitude files use the same `pulse.kind=csv` contract.

## Hardware operations

The TNMR presets cover inspection, dashboard parameter updates, table updates and clearing unused table contents. They default to previews. Choose a working copy in `tnmr.path`, set explicit edits and optionally `save_path`, then add `--live`. Acquisition uses the scope preset with the same explicit flag. Imports, checks, waveform generation, parsing and offline analysis never need a device.

See the [26-project index](../projects/README.md) for exact inputs, outputs, assumptions and commands for each task.
