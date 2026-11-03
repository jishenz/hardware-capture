"""Plotting helpers for oscilloscope RF pulse diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hardware_capture.oscilloscope.pulse_robustness import PulseModel, RobustnessResult
from hardware_capture.oscilloscope.rf_demodulation import DemodulatedWaveform, PulseWindow


def _save_or_show(fig, path: Path | str, *, save: bool, show: bool) -> None:
    if save:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220)
    import matplotlib.pyplot as plt

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_heatmap(
    result: RobustnessResult,
    *,
    save_figure: bool = False,
    figure_path: Path | str = "scope_rf_robustness_heatmap.png",
    show_plot: bool = True,
    vmin: float = 0.99,
    vmax: float = 1.0,
) -> None:
    """Plot process fidelity versus detuning and amplitude error."""

    import matplotlib.pyplot as plt

    extent = (
        float(result.detuning_error[0]),
        float(result.detuning_error[-1]),
        float(result.amplitude_error[0]),
        float(result.amplitude_error[-1]),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        np.clip(result.fidelity, vmin, vmax).T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.plot(0.0, 0.0, marker="+", markersize=14, markeredgewidth=2.5, color="red")
    ax.set_xlabel("Normalized detuning error (Delta / Omega_ref)")
    ax.set_ylabel("Amplitude error (fractional; amplitude scale = 1 + error)")
    ax.set_title("Robustness relative to nominal measured waveform")
    fig.colorbar(image, ax=ax).set_label("Process fidelity")
    fig.tight_layout()
    _save_or_show(fig, figure_path, save=save_figure, show=show_plot)


def _omega_unit(pulse: PulseModel) -> tuple[float, str]:
    scale = max(float(np.max(np.abs(pulse.omega_x))), float(np.max(np.abs(pulse.omega_y))), 1.0)
    if scale >= 1e6:
        return 1e-6, "Mrad/s"
    if scale >= 1e3:
        return 1e-3, "krad/s"
    return 1.0, "rad/s"


def plot_pulse_diagnostics(
    demod: DemodulatedWaveform,
    window: PulseWindow,
    pulse: PulseModel,
    *,
    save_figure: bool = False,
    figure_path: Path | str = "scope_rf_demodulated_pulse.png",
    show_plot: bool = True,
) -> None:
    """Plot demodulated I/Q, envelope, phase, and normalized controls."""

    import matplotlib.pyplot as plt

    active = slice(window.start_index, window.stop_index)
    time_us = demod.time * 1e6
    active_time_us = demod.time[active] * 1e6
    pulse_time_us = pulse.time_mid * 1e6
    omega_factor, omega_unit = _omega_unit(pulse)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=False)
    axes[0].plot(time_us, demod.i, label="I", linewidth=1.0)
    axes[0].plot(time_us, demod.q, label="Q", linewidth=1.0)
    axes[0].set(ylabel="I/Q (V)", title="Demodulated baseband waveform")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_us, demod.amplitude, color="tab:green", linewidth=1.0)
    axes[1].set(ylabel="|I + iQ| (V)", title="Envelope used for active-pulse detection")

    axes[2].plot(active_time_us, demod.phase[active], color="tab:red", linewidth=1.0)
    axes[2].set(ylabel="Phase (rad)", title="Demodulated phase inside detected pulse")

    axes[3].plot(pulse_time_us, pulse.omega_x * omega_factor, label="Omega_x", linewidth=1.0)
    axes[3].plot(pulse_time_us, pulse.omega_y * omega_factor, label="Omega_y", linewidth=1.0)
    axes[3].plot(
        pulse_time_us,
        np.hypot(pulse.omega_x, pulse.omega_y) * omega_factor,
        label="transverse envelope",
        color="black",
        linewidth=1.0,
        alpha=0.75,
    )
    axes[3].set(xlabel="Time (us)", ylabel=f"Applied control ({omega_unit})")
    axes[3].set_title("Pulse applied in the Hamiltonian")
    axes[3].legend(loc="upper right")

    for ax in axes[:2]:
        ax.axvspan(window.start_time * 1e6, window.end_time * 1e6, color="tab:orange", alpha=0.18)
    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, figure_path, save=save_figure, show=show_plot)
