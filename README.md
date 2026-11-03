# Hardware Capture

A standalone Python toolkit for raw RF capture, calibration, pulse analysis, waveform generation, TNMR sequence editing and X(pi) optimization. Extracted from DiamondGUI; no DiamondGUI installation is needed. Windfreak is excluded.

Configure physical conditions once, then use the same API and CLI across workflows. Notebooks are thin examples. All generated results go into a new timestamped directory with the resolved configuration.

## Start offline

```powershell
uv sync
uv run hc list
uv run hc run pulse -c configs/pulse.json
uv run hc run calibrate-square -c projects/calibration/square/config.json
uv run hc run analyze -c projects/analysis/fixed-xpi/config.json
uv run hc run optimize-offline -c configs/offline-demo.json
```

Python 3.12 is required. `uv run python -m hardware_capture` is equivalent to `uv run hc`. Optional notebook support: `uv sync --extra notebooks`, then `uv run jupyter lab`.

## Configure once

```python
from hardware_capture.configuration import Experiment
from hardware_capture.api import plan, run

conditions = Experiment.load("configs/offline-demo.json")
conditions = conditions.with_updates(
    pulse={"duration_s": 80e-6, "max_rabi_hz": 100000},
    optimization={"updates": 5},
)
print(plan("optimize-offline", conditions))
result = run("optimize-offline", conditions)
```

The same `pulse` section supplies generation, ideal-pulse analysis, sequence preparation and optimizer initialization. Set `pulse.kind="csv"` to use your own amplitude/phase samples. Optimization currently targets **X(pi) only**. Choose BB1, square, a shaped pulse or imported pulse as the initial guess; duration, amplitude bounds, calibration and scan axes remain shared.

JSON and TOML configurations support `extends`. Relative paths resolve from the file declaring them. Shared profiles live in `configs/`; per-project presets override only their differences. See the [configuration reference](docs/configuration.md) and [workflow guide](docs/workflows.md).

## Hardware entry points

```powershell
uv sync --extra scope --extra tnmr
uv run hc check optimize-live -c projects/optimization/live/config.json
uv run hc run optimize-live -c projects/optimization/live/config.json
# After replacing the example profile with actual device/working-file/calibration settings:
uv run hc run optimize-live -c projects/optimization/live/config.json --live
```

`check` never connects. Hardware `run` defaults to a preview; Python uses `live=True` for the same explicit execution choice. The example lab profile contains a placeholder VISA resource and historical/example calibration numbers. Replace them with your setup. TNMR needs Windows, TNMR COM registration and the appropriate instrument software; the scope needs a working VISA backend. Hardware acquisition settings and TNMR sequence timing must match the selected pulse conditions.

## Functional organization

| Directory | Responsibility |
|---|---|
| `hardware_capture/devices` | RIGOL VISA and TNMR COM adapters; no import-time connection |
| `hardware_capture/signal` | RF demodulation and pulse-window detection |
| `hardware_capture/calibration` | Square plateau and BB1 envelope-area calibration |
| `hardware_capture/analysis` | Spin propagation, explicit gate targets and ideal/measured overlays |
| `hardware_capture/pulses` | Shared pulse conditions, physical waveform generation, CSV interchange |
| `hardware_capture/sequences` | TPS parsing/editing/JSON round-trip and matched TNT updates |
| `hardware_capture/optimization` | Cubic splines, GRAPE gradients, Adam and measured closed loop |
| `hardware_capture/workflows` | Shared orchestration behind `api.run` and `hc` |
| `hardware_capture/resources` | Capture fixtures, matched sequence templates and selected references |
| `projects` | 26 task presets, English READMEs and thin notebooks |
| `tests` | Numerical and fake-device regression tests |

The [project catalog](projects/README.md) lists every small project. The [migration map](docs/project-catalog.md) explains where each former notebook/script went. Historical extraction provenance is in [extraction notes](docs/extraction.md).

## Validation

```powershell
uv run pytest -q
```

Default tests exclude physical hardware and instrument integration. See [validation notes](docs/validation.md) for what was exercised and what remains hardware-dependent.
