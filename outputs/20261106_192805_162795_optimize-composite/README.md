# November 6 composite optimizer snapshot

Selected summary artifacts from `20261106_192805_162795_optimize-composite`,
the latest run started on November 6, 2026. Its saved history extends into November 7.

This is a partial run: 594 of 1000 planned updates are present. The best selected
candidate is epoch 594. No final verification or completed result JSON is available.
Metrics in `summary.json` are derived from the saved history; physical state is unverified.

- `history.csv`: complete saved scalar history, including paired SPSA losses.
- `loss_curve.png`: measured loss and best loss over the saved updates.
- `best_heatmap.png`: best saved candidate, using a fixed color scale from 0.99 to 1.00.
- `summary.json`: compact metrics and run completeness.
- `composite_schedule.json`: saved sequence timing context.

Per-epoch processed measurements, raw captures, SPSA archives, checkpoints,
original tables, seed exports, and machine-specific configurations remain local.
History measurement paths are reduced to filenames to omit local filesystem paths;
the referenced files are not in this archive.
The selected best measurement is also excluded. The heatmap was regenerated
offline from that local measurement without operating hardware.
