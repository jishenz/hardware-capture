# Composite optimizer diagnostics and learning-rate comparison

Both modes below operate only on saved files and numerical arrays. They have no
hardware mode and do not call an acquisition workflow.

## Inspect a saved run

```powershell
.\.venv\Scripts\python.exe projects/optimization/composite/analyze_run.py diagnostics "outputs/<run-folder>" --output "outputs/<analysis-folder>/measured_run"
```

The output contains `diagnostics.csv`, `optimizer_diagnostics.png`, and provenance
in `diagnostics_metadata.json`. Source captures and history are preserved.
Unfinished updates are excluded. Missing archive evidence stays unavailable.

The panels show robustness and penalized search losses; nominal fidelity of
accepted centers and the lowest probe fidelity for each sign; and proposed and
applied RMS parameter steps. Amplitude logits and phase radians use separate
panels. A rejected update has zero applied step. Backtracking reduces the
applied step relative to the proposed step. Initial steps are zero; final
verification captures are excluded from training-step plots.

The searched objective is `robustness_loss + center_penalty_weight *
max(0, minimum_center_fidelity - center_fidelity)`. Probe curves are means over
directions. Accepted centers meet the floor, so their search and robustness
losses normally overlap. Probe penalties can still influence the next step.

New constrained composite runs record these diagnostics directly in
`history.csv` and refresh `optimizer_diagnostics.png` with the loss curve. This
logging does not alter the update or acceptance rules.

## Compare learning rates in a fixed simulation

```powershell
.\.venv\Scripts\python.exe projects/optimization/composite/analyze_run.py compare "outputs/<run-folder>" --output "outputs/<analysis-folder>/paired_comparison" --updates 1000 --samples 600 --seeds 20260820 20260821 20260822 --workers 3
```

Each seed runs both the configured learning rates and half those rates, starting
from the archived epoch-zero logits, phase corrections and BB1 baseline. The
same seed uses the same SPSA random directions in both arms. Parameter count,
scan grid, objective weights, perturbations, Adam step caps, penalty and
backtracking policy are unchanged. Updates are accepted on nominal fidelity,
without requiring monotonic robustness improvement. Rejected updates retain
parameters and Adam state. Passing probes remain eligible for best selection.

This uses the live optimizer's SPSA/Adam numerical rules, rather than the older
`optimize-offline` workflow's GRAPE gradients and clipped amplitude controls.
The response model is ideal command playback, with no measurement noise,
calibration variation, hardware distortion or drift. This is an algorithm
sensitivity study, not a reconstruction of physical acquisition.

The default simulation uses 600 time samples for speed while retaining the
archived parameter count and scenario grid. A manifest records the loss and
nominal-fidelity difference versus original resolution on available archived
commands. Every final and selected-best pulse is checked again at original
resolution. Use `--samples 6000` for the full temporal grid of a 6000-sample
source run. Coarse and full discretizations can make different decisions near
the fidelity floor; endpoint checks do not prove trajectory equivalence.

`comparison.png` overlays all paired histories; solid/dashed lines distinguish
current/half learning rates, with one color per seed. `summary.csv` reports loss,
uphill-update counts, largest center excursion above the previous best center,
rejections, backtracking, penalty-active epochs and objective evaluation counts.
Each seed/setting retains its history, diagnostics, final parameters and summary.
The selected best includes passing probes, whereas the excursion metric uses
accepted centers only. Judge loss at equal updates and evaluation budgets;
smaller fluctuations alone are not sufficient to prefer a setting.

The simulator never exports commands to devices or edits the live preset.

To compare the configured rates with modest and large increases, add
`--learning-rate-factors 1 1.5 2`. With base rates of 0.02, these test 0.02,
0.03, and 0.04 for both parameter blocks while retaining the configured caps.
The default factors remain 1 and 0.5. Each arm saves `checkpoints.csv` and
parameter archives at updates 100, 300, and the final update, when reached.
Checkpoint losses and nominal fidelities are also checked at the original
temporal resolution. These extra checks are excluded from the optimization
evaluation budget because their results do not feed back into the trajectory.

## Compare configured step caps with uncapped updates

```powershell
.\.venv\Scripts\python.exe projects/optimization/composite/analyze_run.py compare "outputs/<run-folder>" --comparison step-caps --output "outputs/<analysis-folder>/cap_comparison" --updates 1000 --samples 600 --seeds 20260820 20260821 20260822 --workers 3
```

Both arms use the saved learning rates. The capped arm uses the saved
amplitude-logit cap and phase-radian cap (defaults 0.02 and 0.1). The uncapped
arm passes no maximum update to Adam for either block. Sigmoid amplitude
bounds, SPSA settings, fidelity penalty, 99% acceptance (when configured), and
backtracking remain identical. This setting affects only the numerical simulator.

Histories record how many coordinates are clipped and how many would exceed
the reference caps, along with maximum proposed and pre-clipping changes for
each block. These are measured before fidelity backtracking. The uncapped arm
can exceed a reference cap without any clipping being applied. The comparison
plot uses solid lines for capped updates and dashed lines for uncapped updates.
The manifest includes hashes of the source seed/configuration and numerical
implementation for reproducibility. Every final and selected-best command is
also evaluated at the source run's original temporal resolution.

## Compare amplitude caps at fixed learning rates

For the archived run with base rates of 0.02, this tests amplitude-logit caps
0.02, 0.03 and 0.04 at learning rates of 0.04 for both blocks:

```powershell
.\.venv\Scripts\python.exe projects/optimization/composite/analyze_run.py compare "outputs/<run-folder>" --comparison amplitude-caps --learning-rate-factors 2 --amplitude-step-caps 0.02 0.03 0.04 --output "outputs/<analysis-folder>/amplitude_caps" --updates 1000 --samples 600 --seeds 20260820 20260821 20260822 --workers 3
```

The factor multiplies the archived rates, not the current live preset. Exactly
one factor is required for this comparison. Phase caps, perturbations, the
fidelity floor and penalty remain fixed. Actual amplitude clipping counts use
each arm's cap; the separate reference-cap counts use the archived cap.
Full-resolution checkpoint checks and parameter archives are saved at updates
100, 300 and the final update. The simulator does not change live settings.
