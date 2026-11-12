"""File-only optimizer diagnostics. Importing this module never opens devices."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hardware_capture.figures import subplots


STEP_FIELDS = (
    "amplitude_logit_step_rms", "phase_step_rms_rad",
    "proposed_amplitude_logit_step_rms", "proposed_phase_step_rms_rad",
)


def step_metrics(before, proposed, after):
    """Keep amplitude logits and phase radians separate instead of mixing units."""
    before, proposed, after = (np.asarray(value, dtype=float) for value in (before, proposed, after))
    if before.ndim != 1 or before.size % 2 or not before.size:
        raise ValueError("Expected equal-length amplitude-logit and phase blocks")
    if proposed.shape != before.shape or after.shape != before.shape:
        raise ValueError("Parameter vectors must have matching shapes")
    if not all(np.all(np.isfinite(value)) for value in (before, proposed, after)):
        raise ValueError("Parameter vectors must be finite")
    actual_a, actual_p = np.split(after - before, 2)
    proposed_a, proposed_p = np.split(proposed - before, 2)
    return dict(zip(STEP_FIELDS, (float(np.sqrt(np.mean(value**2))) for value in
                                (actual_a, actual_p, proposed_a, proposed_p))))


def write_diagnostics(history, path):
    """Plot the objective actually searched, the fidelity floor, and update sizes."""
    rows = [row for row in history if row.get("kind") != "final_verification"]
    if not rows:
        raise ValueError("No training history to diagnose")
    x = np.array([int(row["epoch"]) for row in rows])
    def values(key):
        return np.array([float(row.get(key, np.nan)) for row in rows])
    figure, axes = subplots(4, 1, figsize=(12, 12), sharex=True)
    for key, label in (("loss", "Center robustness loss"), ("search_loss", "Center penalized search loss"),
                       ("spsa_plus_search_loss_mean", "SPSA + search loss (direction mean)"),
                       ("spsa_minus_search_loss_mean", "SPSA - search loss (direction mean)")):
        y = values(key)
        if np.any(np.isfinite(y)):
            axes[0].plot(x, y, label=label, alpha=0.8, linewidth=1)
    axes[0].set(ylabel="Loss", title="Optimizer diagnostics: accepted centers and probe direction means")
    axes[1].plot(x, values("center_fidelity"), label="Center nominal fidelity", linewidth=1)
    for key, label in (("spsa_plus_center_fidelity_min", "Lowest + probe nominal fidelity"),
                       ("spsa_minus_center_fidelity_min", "Lowest - probe nominal fidelity")):
        y = values(key)
        if np.any(np.isfinite(y)):
            axes[1].plot(x, y, label=label, alpha=0.5, linewidth=0.8)
    floor = values("minimum_center_fidelity")
    if np.any(np.isfinite(floor)):
        axes[1].plot(x, floor, "r--", label="Required fidelity")
    axes[1].set(ylabel="Nominal fidelity")
    for ax, actual, proposed, label in (
        (axes[2], STEP_FIELDS[0], STEP_FIELDS[2], "Amplitude-logit step RMS"),
        (axes[3], STEP_FIELDS[1], STEP_FIELDS[3], "Phase step RMS (rad)"),
    ):
        ax.plot(x, values(proposed), label="Proposed", alpha=0.7, linewidth=1)
        ax.plot(x, values(actual), label="Applied (zero after rejection)", linewidth=1)
        ax.set(ylabel=label)
    axes[-1].set_xlabel("Completed SPSA updates (0 = initial seed)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small", loc="best")
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=150)
    return path


def _parameters(path):
    with np.load(path, allow_pickle=False) as saved:
        return np.r_[saved["amplitude_logits"], saved["phase_coefficients_rad"]]


def saved_diagnostics(directory, output_directory=None):
    """Reconstruct applied/proposed steps from saved commands, including rejection.

    The source history and captures are never modified. Missing evidence stays NaN.
    Old absolute archive paths are resolved by basename within the chosen run.
    """
    directory = Path(directory)
    destination = Path(output_directory) if output_directory else directory
    destination.mkdir(parents=True, exist_ok=True)
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No saved history to diagnose")
    config_path = directory / "effective_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    composite = config.get("composite", {})
    weight = composite.get("center_penalty_weight", 10.0) if "minimum_center_fidelity" in composite else None
    missing = []
    for row in rows:
        row.update({key: np.nan for key in STEP_FIELDS})
        row.setdefault("search_loss", np.nan)
        for member in ("plus", "minus"):
            fidelities = []
            for direction in range(config.get("spsa", {}).get("directions", 1)):
                archive = directory / f"epoch_{int(row['epoch']):04d}_direction_{direction:03d}_{member}.npz"
                if archive.exists():
                    with np.load(archive, allow_pickle=False) as saved:
                        fidelities.append(float(saved["center_fidelity"]))
            key = f"spsa_{member}_center_fidelity_min"
            row[key] = min(fidelities) if fidelities else row.get(key, np.nan)
        if row.get("kind") in {"initial", "accepted_center", "retained_center", "final_verification"}:
            if weight is not None and "minimum_center_fidelity" in row:
                row["search_loss"] = float(row["loss"]) + weight * max(
                    0.0, float(row["minimum_center_fidelity"]) - float(row["center_fidelity"]))
        if row.get("kind") == "initial":
            row.update(dict.fromkeys(STEP_FIELDS, 0.0))
        if row.get("kind") not in {"accepted_center", "retained_center"}:
            continue
        epoch = int(row["epoch"])
        # Split both separators so archived Windows paths also work on Unix.
        basename = row["measurement_path"].replace("\\", "/").split("/")[-1]
        paths = (directory / f"epoch_{epoch:04d}_spsa.npz",
                 directory / f"epoch_{epoch:04d}_trial_000.npz", directory / basename)
        if not all(path.exists() for path in paths):
            missing.append(epoch)
            continue
        with np.load(paths[0], allow_pickle=False) as saved:
            before = saved["center_parameters"]
        after = before if row["kind"] == "retained_center" else _parameters(paths[2])
        row.update(step_metrics(before, _parameters(paths[1]), after))
    with (destination / "diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_diagnostics(rows, destination / "optimizer_diagnostics.png")
    report = {"source_run": str(directory.resolve()), "rows": len(rows),
              "missing_step_epochs": missing, "step_units": "RMS per coefficient, separately in amplitude logits and phase radians",
              "scope": "Completed training history only; unfinished candidates are excluded"}
    (destination / "diagnostics_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
