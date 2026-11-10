"""File-only plots of measured optimizer losses; no device access."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

from hardware_capture.figures import subplots


def write_loss_curve(history: list[dict], path: Path) -> Path:
    """Plot actual selected/probe losses and keep verification separate.

    Epoch zero is the measured seed. The final measurement of the restored
    best command is a separate marker at the last completed update. Adam's
    unmeasured centers are never plotted as measured losses.
    """
    training = [row for row in history if row.get("kind") != "final_verification"]
    if not training:
        raise ValueError("Loss history has no measured candidates")
    epochs = np.array([int(row["epoch"]) for row in training])
    losses = np.array([float(row["loss"]) for row in training])
    if not np.all(np.isfinite(losses)):
        raise ValueError("Measured losses must be finite")
    figure, ax = subplots(figsize=(9, 5))
    ax.plot(epochs, losses, "o-", label="Selected measured candidate", linewidth=1.5)
    for key, label in (
        ("spsa_plus_loss_mean", "SPSA + loss (direction mean)"),
        ("spsa_minus_loss_mean", "SPSA - loss (direction mean)"),
    ):
        values = np.array([float(row.get(key, "nan")) for row in training])
        valid = np.isfinite(values)
        if np.any(valid):
            ax.plot(epochs[valid], values[valid], ".--", alpha=0.6, label=label)
    ax.step(
        epochs,
        np.minimum.accumulate(losses),
        where="post",
        label="Best measured so far",
        linewidth=2,
    )
    verified = [row for row in history if row.get("kind") == "final_verification"]
    if verified:
        loss = float(verified[-1]["loss"])
        if not np.isfinite(loss):
            raise ValueError("Final verification loss must be finite")
        ax.scatter(
            [epochs[-1]],
            [loss],
            marker="*",
            s=130,
            zorder=5,
            label="Restored best: independent verification",
        )
    ax.set(
        xlabel="Completed SPSA updates (0 = initial seed)",
        ylabel="Full robustness loss (lower is better)",
        title="Measured full-loss SPSA optimization",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=160)
    return path


def plot_saved_history(directory: Path) -> Path:
    """Regenerate a plot from a successful or partial run's CSV, offline."""
    directory = Path(directory)
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    return write_loss_curve(history, directory / "loss_curve.png")
