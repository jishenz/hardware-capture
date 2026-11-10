"""Repeated square/target optimization with fresh calibration at every epoch."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from hardware_capture.configuration import Experiment
from hardware_capture.devices.rigol import RigolDHO, WaveformCaptureConfig
from hardware_capture.devices.tnmr import TNMRControl
from hardware_capture.optimization.acceptance import FidelityAcceptance
from hardware_capture.optimization.amplitude import amplitude_logits, sigmoid_amplitude
from hardware_capture.optimization.composite import (
    capture_schedule,
    load_sequence_file,
    reduce_composite,
    schedule_from_sequence,
)
from hardware_capture.optimization.core import (
    AdamState,
    adam_update,
    clamped_bspline_design_matrix,
    command_tables,
    robustness_objective,
)
from hardware_capture.optimization.history import write_loss_curve
from hardware_capture.optimization.pipeline import (
    _history_row,
    _write_history,
    parse_numeric_table,
    upload_tables_with_readback,
)
from hardware_capture.optimization.recording import measure_stage, save_raw_capture
from hardware_capture.optimization.spsa import SPSAConfig, estimate_spsa
from hardware_capture.pulses.conditions import Pulse, generate
from hardware_capture.signal.demodulation import demodulate_raw_rf

from .common import write_json
from .optimize import initial_coefficients, objective, settings


def prepare(config: Experiment):
    """Resolve saved sequence timing and validate without constructing devices."""
    tnmr = config.section("tnmr")
    options = config.section("composite")
    sequence_path = Path(options.get("sequence_path", config.require("tnmr", "path")))
    schedule = schedule_from_sequence(
        load_sequence_file(sequence_path),
        config.require("tnmr", "amplitude_table"),
        config.require("tnmr", "phase_table"),
        parameter_source=options.get("parameter_source", "value"),
        parameter_overrides=options.get("parameter_overrides"),
    )
    if tnmr["amplitude_table"] == tnmr["phase_table"]:
        raise ValueError("Amplitude and phase tables must differ")
    reference = config.calibration().get("rabi_at_100_hz")
    if reference is None or not math.isfinite(reference) or reference <= 0:
        raise ValueError("Set the known square-reference calibration.rabi_at_100_hz")
    effective = config.with_updates(pulse={"duration_s": schedule.target_duration_s})
    settings(effective)
    robustness = objective(effective)
    acceptance = FidelityAcceptance.from_options(options)
    if acceptance and (
        not np.any(robustness.amplitude_errors == 0)
        or not np.any(robustness.detuning_ratios == 0)
    ):
        raise ValueError(
            "Fidelity acceptance requires exact zero amplitude error and detuning in the scan"
        )
    SPSAConfig(**effective.section("spsa")).validate()
    if {"amplitude_learning_rate", "amplitude_step"} & config.section(
        "optimization"
    ).keys():
        raise ValueError(
            "Composite amplitudes use sigmoid logits; replace optimization.amplitude_learning_rate/amplitude_step "
            "with composite.amplitude_logit_learning_rate/amplitude_logit_step"
        )
    for name in ("amplitude_logit_learning_rate", "amplitude_logit_step"):
        value = options.get(name, 0.02)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"composite.{name} must be finite and positive")
    if options.get("seed", "existing_tables") not in {"existing_tables", "configured"}:
        raise ValueError("composite.seed must be existing_tables or configured")
    if options.get("seed", "existing_tables") == "configured":
        _configured_seed_controls(effective)
    search = options.get("alignment_search_s", 1e-6)
    gap = (
        schedule.target_start_s
        - schedule.reference_start_s
        - schedule.reference_duration_s
    )
    if (
        not math.isfinite(search)
        or not 0 < search < min(schedule.reference_duration_s, gap) / 2
    ):
        raise ValueError("alignment_search_s must be positive and fit between pulses")
    for key, default, inclusive in (
        ("middle_fraction", 0.6, False),
        ("minimum_coherence", 0.9, True),
        ("maximum_relative_spread", 0.2, False),
    ):
        value = options.get(key, default)
        if (
            not math.isfinite(value)
            or not 0 < value <= 1
            or (value == 1 and not inclusive)
        ):
            raise ValueError(f"Invalid composite.{key}")
    captured = capture_schedule(schedule, options.get("capture_reference_start_s"))
    window_mode = options.get("capture_window", "configured")
    if window_mode not in {"sequence", "configured"}:
        raise ValueError("composite.capture_window must be sequence or configured")
    if window_mode == "sequence":
        margins = {}
        for name, default in (
            ("capture_pre_margin_s", 10e-6),
            ("capture_post_margin_s", 20e-6),
        ):
            value = options.get(name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= search
            ):
                raise ValueError(
                    f"composite.{name} must be finite and exceed alignment_search_s"
                )
            margins[name] = value
        effective = effective.with_updates(
            scope={
                "window_start_s": captured.reference_start_s
                - margins["capture_pre_margin_s"],
                "window_end_s": captured.end_s + margins["capture_post_margin_s"],
            }
        )
    scope = effective.section("scope")
    config.require("scope", "resource")
    start, end = scope.get("window_start_s", -10e-6), scope.get("window_end_s", 100e-6)
    if start > captured.reference_start_s - search or end < captured.end_s + search:
        raise ValueError(
            f"Scope window must cover {schedule.repetitions} pairs: start <= {captured.reference_start_s - search:.12g} s; end >= {captured.end_s + search:.12g} s"
        )
    for key, default in (
        ("timeout_ms", 30000),
        ("trigger_timeout_s", 30),
        ("armed_timeout_s", 2),
        ("poll_s", 0.05),
    ):
        if not math.isfinite(scope.get(key, default)) or scope.get(key, default) <= 0:
            raise ValueError(f"Invalid scope.{key}")
    demod = config.section("demodulation")
    frequency = config.require("demodulation", "frequency_hz")
    cutoff = demod.get("cutoff_hz", 25e6)
    order = demod.get("filter_order", 4)
    if (
        not math.isfinite(frequency)
        or frequency <= 0
        or not math.isfinite(cutoff)
        or cutoff <= 0
        or demod.get("q_sign", -1) not in {-1, 1}
        or isinstance(order, bool)
        or not isinstance(order, int)
        or order < 1
    ):
        raise ValueError("Invalid demodulation settings")
    return effective, schedule


def _configured_seed_controls(config):
    """Preserve exact symmetric BB1 phase steps as a fixed sampled baseline.

    Adam varies spline corrections to that baseline, starting from zero. This
    avoids smoothing the standard BB1 boundaries before the first acquisition.
    """
    pulse, _, _, _, entries, count, _, _ = settings(config)
    amplitude, phase = initial_coefficients(config)
    baseline = np.zeros(entries)
    if (
        pulse.get("kind", "bb1") == "bb1"
        and pulse.get("angle_deg", 180) == 180
        and pulse.get("axis_phase_deg", 0) == 0
    ):
        if entries % 10:
            raise ValueError("Symmetric BB1 requires pulse.samples divisible by 10")
        seed = generate(config)
        baseline = np.unwrap(seed.phase_rad)
        phase = np.zeros(count)
    return amplitude, phase, baseline


def _heatmap(path, evaluation, robustness, reference, title):
    """Plot fidelity on the fixed 0.99–1.00 scale shared by all heatmaps."""
    from hardware_capture.figures import subplots

    fig, ax = subplots(figsize=(7, 5))
    image = ax.pcolormesh(
        robustness.amplitude_errors,
        robustness.detuning_ratios * reference / 1000,
        evaluation.objective.fidelity,
        shading="nearest",
        vmin=0.99,
        vmax=1.0,
    )
    ax.set(xlabel="Amplitude error", ylabel="Detuning (kHz)", title=title)
    fig.colorbar(image, ax=ax, label="Process fidelity to X(pi)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _verify_parameters(tnmr, schedule):
    """Do not optimize against stale saved timing or repetition settings."""
    for name, expected in schedule.parameters.items():
        actual = str(tnmr.get_parameter(name)).strip()
        if actual != expected:
            # TNMR may return equivalent numeric duration strings.
            from hardware_capture.optimization.composite import _seconds

            try:
                equivalent = math.isclose(
                    _seconds(actual), _seconds(expected), rel_tol=1e-10, abs_tol=0
                )
            except ValueError:
                equivalent = False
            if not equivalent:
                raise ValueError(
                    f"TNMR parameter {name} differs from the saved sequence: {actual!r} != {expected!r}"
                )
    for name in schedule.reference_phase_tables:
        if not np.allclose(
            np.mod(parse_numeric_table(tnmr.get_table(name)), 360), 0, atol=1e-9, rtol=0
        ):
            raise ValueError(
                f"Reference phase table {name} must contain only phase zero"
            )


def _wait_idle(tnmr, scope_options):
    """Require reported acquisition completion before changing command tables.

    This software status is not evidence of a physically safe RF state.
    """
    deadline = time.monotonic() + scope_options.get("trigger_timeout_s", 30)
    while True:
        state = tnmr.snapshot().get("acquisition_complete")
        if state is True:
            return
        if state is not False:
            raise RuntimeError("TNMR acquisition state unknown; refusing table changes")
        if time.monotonic() >= deadline:
            raise TimeoutError("TNMR acquisition did not complete before the timeout")
        time.sleep(scope_options.get("poll_s", 0.05))


def run_composite(
    config: Experiment, directory: Path, *, tnmr=None, scope=None
) -> dict:
    effective, schedule = prepare(config)
    _, opts, reference, duration, entries, count, updates, maximum = settings(effective)
    controls = effective.section("tnmr")
    scope_options = effective.section("scope")
    demod = effective.section("demodulation")
    reduction_options = effective.section("composite")
    acceptance = FidelityAcceptance.from_options(reduction_options)
    spsa = SPSAConfig(**effective.section("spsa"))
    spsa.validate()
    spline = clamped_bspline_design_matrix(entries, count)
    robustness = objective(effective)
    dt = np.full(entries, duration / entries)
    omega_100 = 2 * math.pi * reference
    write_json(directory / "composite_schedule.json", asdict(schedule))
    write_json(directory / "effective_config.json", effective.data)
    print(f"Composite optimizer output: {directory}", flush=True)
    print(
        f"Scope download window: {scope_options.get('window_start_s', -10e-6) * 1e6:.3f} "
        f"to {scope_options.get('window_end_s', 100e-6) * 1e6:.3f} us "
        f"({schedule.repetitions} square/target pairs)",
        flush=True,
    )
    owns_tnmr, owns_scope = tnmr is None, scope is None
    original_amp = original_phase = None
    modified = False
    complete = False
    cleanup_errors = []
    try:
        with measure_stage(
            directory,
            "setup",
            {},
            "connect_read_tables_and_prepare_seed",
            description="Setup",
        ):
            if tnmr is None:
                tnmr = TNMRControl(connect=True, launch=controls.get("launch", False))
                tnmr.open(controls["path"])
            _wait_idle(tnmr, scope_options)
            _verify_parameters(tnmr, schedule)
            original_amp = tnmr.get_table(controls["amplitude_table"])
            original_phase = tnmr.get_table(controls["phase_table"])
            amp_table = parse_numeric_table(original_amp)
            phase_table = parse_numeric_table(original_phase)
            seed_mode = reduction_options.get("seed", "existing_tables")
            if seed_mode == "existing_tables" and (
                len(amp_table) != entries or len(phase_table) != entries
            ):
                raise ValueError(
                    "pulse.samples must match both existing TNMR table lengths"
                )
            write_json(
                directory / "original_tables.json",
                {
                    controls["amplitude_table"]: amp_table.tolist(),
                    controls["phase_table"]: phase_table.tolist(),
                },
            )
            phase_baseline = np.zeros(entries)
            if seed_mode == "existing_tables":
                if np.min(amp_table) < 0 or np.max(amp_table) > maximum * 100:
                    raise ValueError(
                        "Existing target amplitude table exceeds the configured ceiling"
                    )
                # Fit the existing command to the same spline variables Adam updates.
                matrix = spline.toarray()
                fitted = lsq_linear(matrix, amp_table / 100, bounds=(0, maximum))
                if not fitted.success:
                    raise ValueError(
                        f"Bounded amplitude seed fit failed: {fitted.message}"
                    )
                amplitude = fitted.x
                phase = np.linalg.lstsq(
                    matrix, np.unwrap(np.deg2rad(phase_table)), rcond=None
                )[0]
            else:
                # Configured initialization replaces both tables, including their
                # lengths. Originals above remain intact for failure restoration.
                amplitude, phase, phase_baseline = _configured_seed_controls(effective)
            logits = amplitude_logits(amplitude, maximum)
            amplitude, _ = sigmoid_amplitude(logits, maximum)
            initial_percent, _, initial_phase = command_tables(amplitude, phase, spline)
            Pulse(
                duration,
                initial_percent * reference / 100,
                initial_phase + phase_baseline,
            ).save(directory / "initial_seed", rabi_at_100_hz=reference)
            if scope is None:
                scope = RigolDHO(
                    resource=scope_options["resource"],
                    timeout_ms=scope_options.get("timeout_ms", 30000),
                )
                scope.open()

        candidate_count = 0
        total_candidates = 2 + 2 * spsa.directions * updates
        if acceptance:
            _, total_candidates = acceptance.capture_bounds(updates, spsa.directions)

        def measure(parameters, label, description=None):
            nonlocal modified, candidate_count
            candidate_count += 1
            timings = {}
            display = (
                f"{description or label} | capture {candidate_count}/{total_candidates}"
                + (" maximum" if acceptance else "")
            )

            def stage(name):
                return measure_stage(
                    directory, label, timings, name, description=display
                )

            with stage("preflight"):
                _wait_idle(tnmr, scope_options)
                _verify_parameters(tnmr, schedule)
            candidate_logits, candidate_phase = np.split(parameters, 2)
            candidate_amplitude, _ = sigmoid_amplitude(candidate_logits, maximum)
            percent, _, command_phase = command_tables(
                candidate_amplitude, candidate_phase, spline
            )
            command_phase = command_phase + phase_baseline
            degrees = np.mod(np.rad2deg(command_phase), 360)
            modified = True
            with stage("upload_and_readback"):
                upload_tables_with_readback(
                    tnmr,
                    amplitude_table_name=controls["amplitude_table"],
                    phase_table_name=controls["phase_table"],
                    amplitude_percent=percent,
                    phase_degrees=degrees,
                )
            with stage("acquire_and_transfer"):
                raw = scope.capture_single_waveform(
                    WaveformCaptureConfig(
                        channel=scope_options.get("channel", "CHAN1"),
                        window_start_s=scope_options.get("window_start_s", -10e-6),
                        window_end_s=scope_options.get("window_end_s", 100e-6),
                        allow_clamp=False,
                    ),
                    start_shot=tnmr.zero_and_go,
                    confirm_armed=True,
                    rearm_after_read=False,
                    trigger_timeout_s=scope_options.get("trigger_timeout_s", 30),
                    armed_timeout_s=scope_options.get("armed_timeout_s", 2),
                    poll_s=scope_options.get("poll_s", 0.05),
                )
            # Retain raw evidence even if completion, calibration, or loss fails.
            with stage("save_raw"):
                save_raw_capture(
                    directory / f"{label}_raw.npz",
                    time_s=raw.time,
                    voltage_v=raw.voltage,
                    sample_rate_hz=raw.sample_rate,
                )
            with stage("wait_complete"):
                _wait_idle(tnmr, scope_options)
            with stage("demodulation"):
                measured = demodulate_raw_rf(
                    raw,
                    demodulation_freq_hz=demod["frequency_hz"],
                    baseband_cutoff_hz=demod.get("cutoff_hz", 25e6),
                    filter_order=demod.get("filter_order", 4),
                    q_sign=demod.get("q_sign", -1),
                )
            with stage("calibration"):
                reduced = reduce_composite(
                    measured,
                    schedule,
                    samples=entries,
                    reference_rabi_hz=reference,
                    **{
                        key: reduction_options[key]
                        for key in (
                            "alignment_search_s",
                            "capture_reference_start_s",
                            "middle_fraction",
                            "minimum_coherence",
                            "maximum_relative_spread",
                        )
                        if key in reduction_options
                    },
                )
            with stage("objective"):
                normalized = reduced.mean_iq_rad_s / omega_100
                evaluation = robustness_objective(
                    normalized.real, normalized.imag, dt, robustness
                )
            if not math.isfinite(evaluation.objective.loss):
                raise ValueError("Measured composite loss is not finite")
            search_loss = (
                acceptance.search_loss(
                    evaluation.objective.loss, evaluation.objective.center_fidelity
                )
                if acceptance
                else evaluation.objective.loss
            )
            path = directory / f"{label}.npz"
            with stage("save_reduced"):
                np.savez_compressed(
                    path,
                    **asdict(reduced),
                    amplitude_coefficients=candidate_amplitude,
                    amplitude_logits=candidate_logits,
                    amplitude_maximum=maximum,
                    amplitude_parameterization="sigmoid",
                    gradient_method="spsa",
                    phase_coefficients_rad=candidate_phase,
                    phase_baseline_rad=phase_baseline,
                    amplitude_percent=percent,
                    command_phase_unwrapped_rad=command_phase,
                    fidelity=evaluation.objective.fidelity,
                    loss=evaluation.objective.loss,
                    search_loss=search_loss,
                    center_fidelity=evaluation.objective.center_fidelity,
                    amplitude_errors=robustness.amplitude_errors,
                    detuning_hz=robustness.detuning_ratios * reference,
                )
            return {
                "parameters": parameters.copy(),
                "path": path,
                "evaluation": evaluation,
                "search_loss": search_loss,
                "percent": percent,
                "command_phase": command_phase,
            }

        amplitude_state = phase_state = None
        if acceptance:
            amplitude_state = AdamState(np.zeros(count), np.zeros(count))
            phase_state = AdamState(np.zeros(count), np.zeros(count))
        history = []
        best = None
        best_epoch = 0

        def consider_best(candidate, epoch):
            nonlocal best, best_epoch
            evaluation = candidate["evaluation"]
            if acceptance and not acceptance.accepts(
                evaluation.objective.center_fidelity
            ):
                return
            if (
                best is None
                or evaluation.objective.loss < best["evaluation"].objective.loss
            ):
                best, best_epoch = candidate, epoch
                _heatmap(
                    directory / "best_heatmap.png",
                    evaluation,
                    robustness,
                    reference,
                    title=f"Mean of {schedule.repetitions} calibrated targets, epoch {epoch}",
                )

        def record(
            candidate,
            epoch,
            kind,
            estimate=None,
            *,
            accepted=True,
            step_scale=0.0,
            probes=(),
        ):
            evaluation = candidate["evaluation"]
            if kind != "final_verification":
                consider_best(candidate, epoch)
            history.append(
                {
                    **_history_row(epoch, evaluation),
                    "kind": kind,
                    "measurement_path": str(candidate["path"]),
                    "spsa_plus_loss_mean": float(np.mean(estimate.plus_losses))
                    if estimate
                    else math.nan,
                    "spsa_minus_loss_mean": float(np.mean(estimate.minus_losses))
                    if estimate
                    else math.nan,
                    "spsa_capture_order": ";".join(estimate.capture_order)
                    if estimate
                    else "",
                    **(
                        {
                            "minimum_center_fidelity": acceptance.minimum_center_fidelity,
                            "meets_fidelity_floor": acceptance.accepts(
                                evaluation.objective.center_fidelity
                            ),
                            "update_accepted": accepted,
                            "step_scale": step_scale,
                            "best_feasible_loss": best["evaluation"].objective.loss
                            if best
                            else math.nan,
                            "spsa_plus_search_loss_mean": float(
                                np.mean(estimate.plus_losses)
                            )
                            if estimate
                            else math.nan,
                            "spsa_minus_search_loss_mean": float(
                                np.mean(estimate.minus_losses)
                            )
                            if estimate
                            else math.nan,
                            # Keep plotted probe losses in the same units as robustness loss.
                            "spsa_plus_loss_mean": float(
                                np.mean(
                                    [
                                        p["evaluation"].objective.loss
                                        for sign, p in probes
                                        if sign == "plus"
                                    ]
                                )
                            )
                            if probes
                            else math.nan,
                            "spsa_minus_loss_mean": float(
                                np.mean(
                                    [
                                        p["evaluation"].objective.loss
                                        for sign, p in probes
                                        if sign == "minus"
                                    ]
                                )
                            )
                            if probes
                            else math.nan,
                        }
                        if acceptance
                        else {}
                    ),
                }
            )
            _write_history(directory / "history.csv", history)
            write_loss_curve(history, directory / "loss_curve.png")
            label = f"Epoch {epoch:02d}/{updates:02d}" if kind == "spsa" else kind
            print(f"{label}: loss={evaluation.objective.loss:.9g}", flush=True)

        # Measure the exact seed before any temporary SPSA perturbation.
        current = measure(np.r_[logits, phase], "epoch_0000", "Initial BB1")
        if acceptance and not acceptance.accepts(
            current["evaluation"].objective.center_fidelity
        ):
            raise ValueError(
                "Initial pulse fails the ideal-condition fidelity floor; optimization not started"
            )
        record(current, 0, "initial")
        for epoch in range(1, updates + 1):
            candidates = []
            probes = []

            def probe(
                parameters,
                direction,
                member,
                *,
                epoch=epoch,
                candidates=candidates,
                probes=probes,
            ):
                candidate = measure(
                    parameters,
                    f"epoch_{epoch:04d}_direction_{direction:03d}_{member}",
                    f"Epoch {epoch:02d}/{updates:02d}, direction {direction + 1}/{spsa.directions}, {member} probe",
                )
                candidates.append(candidate)
                probes.append((member, candidate))
                return candidate["search_loss"]

            estimate = estimate_spsa(
                np.r_[logits, phase], probe, config=spsa, epoch=epoch - 1
            )
            # Only a complete set of valid pairs can reach Adam or a checkpoint.
            spsa_path = directory / f"epoch_{epoch:04d}_spsa.npz"
            np.savez_compressed(
                spsa_path, **asdict(estimate), center_parameters=np.r_[logits, phase]
            )
            selected = min(
                candidates, key=lambda item: item["evaluation"].objective.loss
            )
            if not acceptance:
                record(selected, epoch, "spsa", estimate)
            proposed_logits, proposed_amplitude_state = adam_update(
                logits,
                estimate.gradient[:count],
                amplitude_state,
                learning_rate=reduction_options.get(
                    "amplitude_logit_learning_rate", 0.02
                ),
                maximum_step=reduction_options.get("amplitude_logit_step", 0.02),
            )
            proposed_phase, proposed_phase_state = adam_update(
                phase,
                estimate.gradient[count:],
                phase_state,
                learning_rate=opts.get("phase_learning_rate", 0.02),
                maximum_step=opts.get("phase_step_rad", 0.1),
            )
            if acceptance:
                for candidate in candidates:
                    consider_best(candidate, epoch)
                center = np.r_[logits, phase]
                change = np.r_[proposed_logits, proposed_phase] - center
                accepted = False
                attempts = []
                step_scale = 0.0
                for backtrack in range(acceptance.maximum_backtracks + 1):
                    scale = 0.5**backtrack
                    trial = measure(
                        center + scale * change,
                        f"epoch_{epoch:04d}_trial_{backtrack:03d}",
                        f"Epoch {epoch}, proposed update at {scale:g} scale",
                    )
                    passed = acceptance.accepts(
                        trial["evaluation"].objective.center_fidelity
                    )
                    attempts.append(
                        {
                            "measurement_path": str(trial["path"]),
                            "center_fidelity": trial[
                                "evaluation"
                            ].objective.center_fidelity,
                            "step_scale": scale,
                            "accepted": passed,
                        }
                    )
                    if passed:
                        current, accepted, step_scale = trial, True, scale
                        logits, phase = np.split(trial["parameters"].copy(), 2)
                        amplitude_state, phase_state = (
                            proposed_amplitude_state,
                            proposed_phase_state,
                        )
                        break
                if not accepted:
                    # Reject both parameter and Adam-state changes. Leave hardware
                    # at the last accepted command, not the last failed trial.
                    _wait_idle(tnmr, scope_options)
                    _verify_parameters(tnmr, schedule)
                    upload_tables_with_readback(
                        tnmr,
                        amplitude_table_name=controls["amplitude_table"],
                        phase_table_name=controls["phase_table"],
                        amplitude_percent=current["percent"],
                        phase_degrees=np.mod(np.rad2deg(current["command_phase"]), 360),
                    )
                write_json(
                    directory / f"epoch_{epoch:04d}_acceptance.json",
                    {
                        "minimum_center_fidelity": acceptance.minimum_center_fidelity,
                        "accepted": accepted,
                        "attempts": attempts,
                        "retained_measurement_path": str(current["path"]),
                    },
                )
                record(
                    current,
                    epoch,
                    "accepted_center" if accepted else "retained_center",
                    estimate,
                    accepted=accepted,
                    step_scale=step_scale,
                    probes=probes,
                )
            else:
                logits, phase = proposed_logits, proposed_phase
                amplitude_state, phase_state = (
                    proposed_amplitude_state,
                    proposed_phase_state,
                )
            amplitude, _ = sigmoid_amplitude(logits, maximum)
            np.savez_compressed(
                directory / "checkpoint_latest.npz",
                next_epoch=epoch + 1,
                amplitude_coefficients=amplitude,
                amplitude_logits=logits,
                amplitude_maximum=maximum,
                amplitude_parameterization="sigmoid",
                gradient_method="spsa",
                spsa_seed=spsa.seed,
                last_spsa_path=str(spsa_path),
                phase_coefficients_rad=phase,
                phase_baseline_rad=phase_baseline,
                amplitude_logit_first=amplitude_state.first_moment,
                amplitude_logit_second=amplitude_state.second_moment,
                phase_first=phase_state.first_moment,
                phase_second=phase_state.second_moment,
                adam_step=amplitude_state.step,
                center_is_measured=acceptance is not None,
                **(
                    {
                        "center_measurement_path": str(current["path"]),
                        "center_fidelity": current[
                            "evaluation"
                        ].objective.center_fidelity,
                        "minimum_center_fidelity": acceptance.minimum_center_fidelity,
                        "update_accepted": accepted,
                    }
                    if acceptance
                    else {}
                ),
                best_epoch=best_epoch,
                best_measurement_path=str(best["path"]),
            )
        # Restore and independently measure the best actual candidate, not the
        # unmeasured Adam center or an average of the +/- command parameters.
        verifications = []
        for index in range(acceptance.verification_captures if acceptance else 1):
            label = (
                "final_verification"
                if index == 0
                else f"final_verification_{index + 1:03d}"
            )
            final = measure(best["parameters"], label, "Final best-pulse verification")
            record(final, updates + 1, "final_verification")
            verifications.append(
                {
                    "measurement_path": str(final["path"]),
                    "center_fidelity": final["evaluation"].objective.center_fidelity,
                    "loss": final["evaluation"].objective.loss,
                }
            )
            if acceptance and not acceptance.accepts(
                final["evaluation"].objective.center_fidelity
            ):
                write_json(
                    directory / "verification_failure.json",
                    {
                        "measurements": verifications,
                        "minimum_center_fidelity": acceptance.minimum_center_fidelity,
                    },
                )
                raise ValueError(
                    "Final verification fails the ideal-condition fidelity floor; pulse not exported"
                )
        artifacts = Pulse(
            duration, best["percent"] * reference / 100, best["command_phase"]
        ).save(directory, rabi_at_100_hz=reference)
        result = {
            "best_epoch": best_epoch,
            "best_loss": best["evaluation"].objective.loss,
            "best_measurement_path": str(best["path"]),
            "final_verification_loss": final["evaluation"].objective.loss,
            "final_verification_path": str(final["path"]),
            "pulse_path": artifacts["pulse_path"],
            "repetitions": schedule.repetitions,
            "reference_rabi_hz": reference,
            "target": "fixed_xpi",
            "amplitude_parameterization": "sigmoid",
            "amplitude_maximum": maximum,
            "gradient_method": "spsa",
            "gradient_hardware_jacobian": "included through paired full-loss measurements",
            "spsa": asdict(spsa),
            "averaging": "mean of individually calibrated and time-aligned complex target I/Q before each loss evaluation",
            "candidate_evaluations": candidate_count,
            **(
                {
                    "acceptance": asdict(acceptance),
                    "best_center_fidelity": best[
                        "evaluation"
                    ].objective.center_fidelity,
                    "final_verifications": verifications,
                    "minimum_verified_center_fidelity": min(
                        v["center_fidelity"] for v in verifications
                    ),
                }
                if acceptance
                else {}
            ),
            "loss_history_path": str(directory / "history.csv"),
            "loss_curve_path": str(directory / "loss_curve.png"),
        }
        write_json(directory / "summary.json", result)
        complete = True
        print(
            f"Optimization complete: {updates}/{updates} updates; {candidate_count}/{total_candidates} captures. Loss curve: {directory / 'loss_curve.png'}",
            flush=True,
        )
        return result
    finally:
        if not complete and modified:
            try:
                tnmr.stop()
                _wait_idle(tnmr, scope_options)
            except Exception as exc:  # noqa: BLE001 - preserve the primary failure
                cleanup_errors.append(
                    f"Stop/completion check: {exc}; table restoration skipped"
                )
            else:
                for name, original in (
                    (controls["amplitude_table"], original_amp),
                    (controls["phase_table"], original_phase),
                ):
                    try:
                        tnmr.set_table(name, original)
                    except Exception as exc:  # noqa: BLE001 - attempt both restorations
                        cleanup_errors.append(f"Restore {name}: {exc}")
        for device, owned in ((scope, owns_scope), (tnmr, owns_tnmr)):
            if device is not None and owned:
                try:
                    device.close()
                except Exception as exc:  # noqa: BLE001 - attempt all owned closes
                    cleanup_errors.append(f"Close {type(device).__name__}: {exc}")
        if cleanup_errors:
            write_json(
                directory / "cleanup_errors.json",
                {"errors": cleanup_errors, "physical_state": "unverified"},
            )
