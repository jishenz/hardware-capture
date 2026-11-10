# Shared profiles

- `pulse.json`: example BB1 X(pi), 60 microseconds, 600 samples, 125 kHz example amplitude-100 calibration; physical scan axes.
- `offline-demo.json`: inherits pulse conditions, uses 160 table samples/40 spline coefficients, three optimization updates and a deterministic gain/ripple response.
- `lab.example.json`: inherits pulse conditions and adds explicit scope, demodulation, TNMR and live optimizer settings. VISA resource is a placeholder. Calibration values are examples, not measurements for your setup. Make a local copy and update the project `extends` path.
- `sample-bb1.json`: historical raw capture with its original 90-degree BB1 calibration assumption. The target of optimization remains X(pi).
- `synthetic-square.json`: deterministic synthetic 2 MHz square RF fixture for a complete offline calibration example.

Do not replace your measured calibration with example values. A calibration result can be reused through `calibration.path`; explicit fields override values read from that artifact. Remove inherited example coefficient fields from your local lab profile when you want the artifact to supply them.

Project presets are under `projects/<function>/<project>/config.json`. All documented path strings resolve relative to the file that declares them, including parent profiles. A child profile can extend multiple parents; later parents and then child values win. Arrays are replaced, objects are merged.

The capture and composite optimizer presets extend the ignored `configs/lab.local.json`.
Create it by copying `configs/lab.example.json`, then have the human operator set
the instrument address, sequence path, and measured calibration for the setup.
Keep this machine-specific profile out of version control.
