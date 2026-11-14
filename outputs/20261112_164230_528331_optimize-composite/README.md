# November 12, 2026 composite optimization

This is a curated snapshot of `20261112_164230_528331_optimize-composite`, which started on November 12 and
failed during raw-capture saving on November 13 at 00:01:48 CST. It completed
852 of 4,000 planned updates (849 accepted; three rejected). Final independent
verification and successful pulse export were not reached.

| Metric | Initial BB1 seed | Best eligible candidate | Last completed center |
|---|---:|---:|---:|
| Epoch | 0 | 723 | 852 |
| Robustness loss | 0.5384767447 | 0.4141327209 | 0.4352009140 |
| Nominal center fidelity | 99.577923% | 99.345044% | 99.104341% |

The best eligible loss improved by **23.09%**. The selected
measurement is direction 1 (zero-based), plus probe at epoch 723. It satisfies
the 99% nominal fidelity floor, but has not received the required three final
verification captures. Its mean scan fidelity is 68.0581% and its
worst scan fidelity is 0.016523%; this is not a 99% robust gate over
the full scan.

## Retained evidence

- [Loss history plot](loss_curve.png): accepted/retained centers, probe means,
  and best eligible loss through epoch 852.
- [Best fidelity heatmap](best_heatmap.png): selected epoch 723 measurement.
  The color scale is fixed at 0.99-1.00; lower values use the lowest color.
- [Optimizer diagnostics](optimizer_diagnostics.png): nominal fidelity floor,
  search loss, and proposed/applied update sizes.
- [Original scalar history](history.csv): all completed updates, including
  acceptance decisions and step metrics.
- [Best reduced measurement](epoch_0723_direction_001_plus.npz): calibrated I/Q, command parameters,
  fidelity grid, and scan axes for the selected candidate.
- [Last complete checkpoint](checkpoint_latest.npz): optimizer state after
  epoch 852; `next_epoch=853`, `adam_step=849`. Automatic resume is unsupported.
- [Machine-readable review](run_review.json): exact metrics, scientific
  configuration, interruption details, and SHA-256 hashes of retained evidence.

## Interpretation and limits

The initial command is symmetric BB1 X(pi): 60 us, 6,000 samples, constant
26.041667% amplitude, five phase segments, and 125 amplitude plus 125 phase
spline controls. Two SPSA directions were used per update, with learning rates
0.04/0.04, amplitude-logit step cap 0.04, and phase step cap 0.1 radians.
The robustness grid has 21 amplitude errors from -50% to +50% and 21 detunings
from -62.5 to +62.5 kHz. Each measurement averages six individually calibrated
target waveforms before spin propagation. The reference is assigned 160 kHz
at amplitude 100%.

The last stage record reports acquisition/transfer complete, followed by
`save_raw: failed` during epoch 853, direction 1 minus probe, capture 4,293.
Disk exhaustion is the leading diagnosis; the original exception was not
preserved, so the precise cause is unresolved.

These are RF-derived model results. Physical playback, actual spin fidelity,
and cleanup state remain unverified. Historical file paths in the CSV and
checkpoint identify local captures that are mostly excluded from version
control. Raw RF records and device connection configuration are excluded.

The current source implements selected retention in RAM; this historical run
predates that change and saved every capture. Its plots are preserved as
recorded, rather than regenerated with the new retention behavior.
