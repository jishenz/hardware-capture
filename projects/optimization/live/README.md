# Live X(pi) workflow

LIVE: default invocation returns a preview. Add --live only when ready to connect.

## Input

Shared pulse/calibration/scan settings, TNMR working file and two table names, scope window, demodulation, phase_reference_captures >= 2.

The project config extends `configs/lab.example.json`. Paths resolve relative to the file declaring them. Change shared conditions in a common profile, or override them here.

## Output

Reference diagnostics and captures; live mode also produces history, checkpoints, measured heatmaps, raw/processed +/- probes, and `epoch_NNNN_spsa.npz` gradient evidence. Each run also records `resolved_config.json` and `result.json` in a new timestamped output folder.

## Optimization

The hardware profile uses 1500 amplitude logits and 1500 phase coefficients,
expanded to 6000 table entries per family. At 60 us the nominal dwell is 10 ns.

After BB1 phase referencing, the optimizer measures the requested initial seed.
Each update perturbs amplitude logits and phase coefficients simultaneously,
evaluates the complete scalar robustness loss on both measured waveforms, and
passes their SPSA gradient to Adam. There is no GRAPE backward pass or identity
hardware-response approximation in this workflow. The loss and robustness
weights remain unchanged.

The `spsa` section defaults to one direction, seed 20260820, amplitude-logit
perturbation 0.08, phase perturbation 0.05 rad, and decay 0.101. Capture order
alternates. Both amplitude probe commands use `a_max * sigmoid(z)`. Configure
`optimization.amplitude_logit_learning_rate` and `amplitude_logit_step` (default
0.02 each); old amplitude-space rate/step keys are rejected to avoid silently
changing units. Phase settings keep their existing units.

The best actual initial/probe candidate is restored and independently measured
at completion. The updated Adam center is explicitly marked unmeasured in its
checkpoint. Each update costs two candidate captures per direction; initial and
final measurements and phase-reference captures are additional. A failed probe
cannot advance Adam or its checkpoint. Automatic resume is not implemented.

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
