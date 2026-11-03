# Project catalog and migration map

Projects are grouped by function, not source author or device folder. Every linked README documents input, output, assumptions and commands. `projects/<function>/<name>/config.json` is the editable entry point; notebooks call the same API.

| Functional project | Former entry point | Workflow |
|---|---|---|
| [Raw RF capture](../projects/acquisition/capture/README.md) | `oscilloscope_connection_pipeline.ipynb` | `capture` |
| [Square pulse amplitude calibration](../projects/calibration/square/README.md) | Scope script / optimizer entry point | `calibrate-square` |
| [BB1 envelope area calibration](../projects/calibration/bb1/README.md) | Scope script / optimizer entry point | `calibrate-bb1` |
| [Nominal robustness analysis](../projects/analysis/nominal/README.md) | Scope script / optimizer entry point | `analyze` |
| [Best-Axis robustness analysis](../projects/analysis/best-axis/README.md) | Scope script / optimizer entry point | `analyze` |
| [Fixed-Xpi robustness analysis](../projects/analysis/fixed-xpi/README.md) | Scope script / optimizer entry point | `analyze` |
| [Waveform generation](../projects/pulses/waveforms/README.md) | `waveform_example.ipynb` | `pulse` |
| [Linear frequency sweep](../projects/pulses/chirp/README.md) | `chirp_waveform_example.ipynb` | `pulse` |
| [Waveform serialization](../projects/pulses/serialization/README.md) | `write_waveform_json_example.ipynb` | `pulse` |
| [TPS parsing and JSON round-trip](../projects/sequences/inspect/README.md) | `parse_sequence_to_json.ipynb` | `sequence` |
| [Edit a table with a waveform](../projects/sequences/table-edit/README.md) | `apply_waveform_to_sequence_example.ipynb` | `sequence` |
| [Remove unused TPS tables offline](../projects/sequences/prune/README.md) | `remove_unused_sequence_tables_example.ipynb` | `sequence` |
| [NVNMR parameter and table editing](../projects/sequences/nvnmr/README.md) | `nvnmr_template_parameter_table_waveform_editing.ipynb` | `sequence` |
| [Bb1 sequence preparation](../projects/sequences/bb1/README.md) | `apply_bb1_waveform_to_sequence_example.ipynb` | `sequence` |
| [Chirp sequence preparation](../projects/sequences/chirp/README.md) | `apply_chirp_waveform_to_sequence_example.ipynb` | `sequence` |
| [Inversion-Chirp sequence preparation](../projects/sequences/inversion-chirp/README.md) | `apply_chirp_waveform_to_nvnmr_sensing_sequence_example.ipynb` | `sequence` |
| [Inversion-Hs1 sequence preparation](../projects/sequences/inversion-hs1/README.md) | `apply_hs1_waveform_to_nvnmr_sensing_sequence_example.ipynb` | `sequence` |
| [Inversion-Wurst sequence preparation](../projects/sequences/inversion-wurst/README.md) | `apply_wurst20_waveform_to_nvnmr_sensing_sequence_example.ipynb` | `sequence` |
| [Inversion-Wursttanh sequence preparation](../projects/sequences/inversion-wursttanh/README.md) | `apply_wursttanh_waveform_to_nvnmr_sensing_sequence_example.ipynb` | `sequence` |
| [TNMR inspect](../projects/tnmr/inspect/README.md) | `tnmrcontrol_basic_usage.ipynb` | `tnmr` |
| [TNMR parameters](../projects/tnmr/parameters/README.md) | `tnmrcontrol_parameter_editing.ipynb` | `tnmr` |
| [TNMR tables](../projects/tnmr/tables/README.md) | `tnmrcontrol_sequence_tables.ipynb` | `tnmr` |
| [TNMR clear-tables](../projects/tnmr/clear-tables/README.md) | `tnmrcontrol_clean_sequence_tables.ipynb` | `tnmr` |
| [Simulated X(pi) optimization](../projects/optimization/offline/README.md) | Scope script / optimizer entry point | `optimize-offline` |
| [Phase-Reference X(pi) workflow](../projects/optimization/phase-reference/README.md) | Scope script / optimizer entry point | `phase-reference` |
| [Live X(pi) workflow](../projects/optimization/live/README.md) | Scope script / optimizer entry point | `optimize-live` |

## Former scripts

| Old source script | Current entry |
|---|---|
| capture_time_window.py | acquisition/capture, `hc run capture` |
| calibrate_rf_from_square.py | calibration/square, `hc run calibrate-square` |
| calibrate_rf_from_bb1.py | calibration/bb1, `hc run calibrate-bb1` |
| scope_rf_robustness_heatmap.py | analysis/nominal, shared overlays and explicit scaling |
| scope_rf_best_axis_xpi_heatmap.py | analysis/best-axis |
| demo_stage2_offline.py | optimization/offline |
| run_stage1_calibration.py | optimization/phase-reference |
| run_two_stage_optimizer.py | optimization/live |

## Preserved capabilities and deliberate interface changes

Capture keeps current-record and newly armed single-shot operation, optional shot command, repeated shots and waveform metadata. Square and BB1 calibration share a raw input contract. All three analysis target definitions remain distinct; fixed-scope calibration and BB1 area normalization are explicit choices. Ideal/measured amplitude and phase overlays remain available.

The waveform library retains analytical shapes, phase conventions, quantization, JSON and table conversion. TPS APIs retain parameter editing, shared-table checks, event assignments, table views, unused-table pruning and lossless raw-byte JSON. TNMRControl retains dashboard enumeration, acquisition commands, sequence export/import, table operations and COM thread ownership. The live optimizer retains repeated phase referencing, diagnostic plots, GRAPE/Adam, measured candidate selection, checkpoints and failed-run restoration.

Former module paths and script flags are intentionally retired. New workflows use one config schema; script globals and duplicated notebook algorithms are removed. Raw CSV, canonical pulse CSV and analytical waveform JSON are documented as separate inputs. Physical detuning is Hz at the public boundary. X(pi) is the only optimizer target for now. `check` and live previews never connect; actual hardware execution is explicit.

The inverted-phase historical TPS/TNT pair is not byte-matched. Its offline workflow emits a TPS plus a TNMR update config; actual TNT export uses the preserved explicit COM path. Other paired templates also provide this path when dashboard synchronization is needed.

The extraction manifest describes the pre-refactor snapshot. [Module map](refactor-module-map.json) maps old library modules; templates/manuals are now in `hardware_capture/resources/tnmr`, captures in `hardware_capture/resources/captures`, and tests in `tests/`.
