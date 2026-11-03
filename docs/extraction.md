# Extraction notes

## Source and scope

Extracted on November 3, 2026 from `C:\Users\jd200\diamondgui` at source HEAD `7e23a6f3c4ea1f1662c9d95a98b6ec3c8a23ede4`.

The source working tree contained a modified `test_rf_analysis.py` and two untracked scope tests. Those current files are included. Required TPS/TNT templates were read from the working tree, including files ignored by the source repository. No source file was moved, deleted, or edited.

| Original path | New path |
| --- | --- |
| `hardware/oscilloscope/` | `hardware_capture/oscilloscope/` |
| `hardware/rf/redstone/` | `hardware_capture/rf/redstone/` |
| Selected values from `hardware/config.py` | Minimal `hardware_capture/config.py` with environment overrides |

## Adaptations

- Changed package imports, repository-relative paths, and command examples to `hardware_capture`.
- Retained subsystem layouts and template locations so existing path-relative fixtures keep working.
- Replaced machine-specific repository import roots in notebooks with local repository discovery.
- Cleared saved notebook outputs and execution counts.
- Selected portable Python kernel metadata and made the unused-table example default to the included TPS fixture.
- Added an independent Python project, optional instrument dependencies, offline test selection, and English documentation.
- Preserved existing instrument commands, numeric defaults, timing, calibration values, cleanup behavior, dry-run defaults, and live confirmation checks.

The oscillator capture and RF control code retain their separate source provenance. This extraction does not reassign authorship. Original history remains in DiamondGUI.

## Excluded material

- DiamondGUI Dash pages, task runtime, hardware manager, and unrelated instruments.
- All Windfreak content, including its serial controller, NI analog-output controller, examples, tests, configuration, and optional dependencies, removed at the user's request on November 3, 2026. The initial extraction commit remains in Git history; the current repository no longer contains that subsystem.
- Generated captures, plots, exported sequences, caches, and notebook output payloads.
- The larger vendor example library and manuals outside the selected TNMR control/file-format references. One vendor TPS example required by tests is included.
- Three notebooks tied to DiamondGUI's measurement contracts and experiment data:
  - `apply_bb1_waveform_to_nvnmr_freqsweep_ampmod_tables.ipynb`
  - `apply_sahs1_beta6_waveform_to_nvnmr_freqsweep_tables.ipynb`
  - `nvnmr_nutation_update_2d_tables.ipynb`

The omitted notebooks remain available in the source repository. Their functionality belongs to experiment orchestration and would require a separate extraction of `measurement/tnmr_file.py` and its contracts/data.

The [machine-readable manifest](extraction-manifest.json) records the currently retained source paths and their original SHA-256, source Git status, and selected exclusions. Removed source paths are recorded as exclusions. Checksums refer to the files before import/path adaptation. Generated caches and capture outputs are excluded without itemizing every file.

## Independence

This repository has its own Git history and environment. It does not import DiamondGUI or require that checkout to exist. Future changes in either repository are not synchronized automatically. The new package can be installed using `uv sync` or `pip install -e .`; examples and vendor documents are best used from a checkout.
