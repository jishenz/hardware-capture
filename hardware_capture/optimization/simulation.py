"""Deterministic SPSA/Adam comparison using numerical spin propagation only.

No device classes, workflow runners, COM, VISA, or acquisition code are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from hardware_capture.figures import subplots
from .acceptance import FidelityAcceptance
from .amplitude import sigmoid_amplitude
from .core import (AdamState, RobustnessConfig, _objective_from_fidelities,
                   adam_update, clamped_bspline_design_matrix)
from .diagnostics import step_metrics, write_diagnostics
from .spsa import SPSAConfig, estimate_spsa


def simulated_objective(i, q, dt, config):
    """Exact piecewise-constant SU(2) propagation via real unit quaternions.

    Equivalent to core.robustness_objective, using U=w I-i(x X+y Y+z Z).
    Later time steps multiply on the left; X(pi) fidelity is final x squared.
    """
    config.validate()
    i, q, dt = (np.asarray(value, dtype=float) for value in (i, q, dt))
    if i.ndim != 1 or i.size == 0 or q.shape != i.shape or dt.shape != i.shape:
        raise ValueError("I, Q and durations must be matching nonempty vectors")
    if not all(np.all(np.isfinite(value)) for value in (i, q, dt)) or np.any(dt <= 0):
        raise ValueError("Controls must be finite and durations positive")
    scales = 1 + np.tile(config.amplitude_errors, len(config.detuning_ratios))
    if np.any(scales < 0):
        raise ValueError("Amplitude errors must not make drive scale negative")
    detuning = np.repeat(config.detuning_ratios, len(config.amplitude_errors))
    # Component-first layout keeps each arithmetic operation contiguous.
    x = config.omega_reference_rad_s * i[:, None] * scales
    y = config.omega_reference_rad_s * q[:, None] * scales
    z = np.broadcast_to(config.omega_reference_rad_s * detuning, x.shape)
    half_dt = dt[:, None] / 2
    angle = half_dt * np.sqrt(x*x + y*y + z*z)
    factor = half_dt * np.sinc(angle / math.pi)
    parts = np.array([np.cos(angle), factor*x, factor*y, factor*z])
    while parts.shape[1] > 1:
        paired = parts.shape[1] // 2 * 2
        b = parts[:, :paired:2]  # earlier
        a = parts[:, 1:paired:2]  # later
        product = np.empty_like(a)
        product[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
        product[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
        product[2] = a[0]*b[2] + a[2]*b[0] + a[3]*b[1] - a[1]*b[3]
        product[3] = a[0]*b[3] + a[3]*b[0] + a[1]*b[2] - a[2]*b[1]
        parts = np.concatenate((product, parts[:, -1:]), axis=1) if paired < parts.shape[1] else product
    return _objective_from_fidelities(parts[1, 0]**2, config)


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(destination, histories, summaries):
    """Create one comparison figure/table from serial or independently run seeds."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "summary.csv", summaries)
    figure, axes = subplots(2, 1, figsize=(12, 9), sharex=True)
    seeds = list(dict.fromkeys(row["seed"] for row in summaries))
    comparison = summaries[0].get("comparison")
    cap_comparison = comparison in {"step-caps", "amplitude-caps"}
    for index, seed in enumerate(seeds):
        for arm_index, summary in enumerate(row for row in summaries if row["seed"] == seed):
            arm = summary.get("comparison_arm", f"lr_{summary['learning_rate_factor']:g}")
            rows = histories[f"seed_{seed}_{arm}"]
            description = (f"Amplitude cap {summary['amplitude_logit_step_cap']:g}" if comparison == "amplitude-caps"
                           else arm if cap_comparison else f"LR x{summary['learning_rate_factor']:g}")
            for ax, key in zip(axes, ("loss", "best_feasible_loss")):
                ax.plot([int(row["epoch"]) for row in rows], [float(row[key]) for row in rows],
                        label=f"seed {seed}, {description}", linewidth=1, alpha=0.85,
                        color=f"C{index}", linestyle=("-", "--", ":", "-.")[arm_index % 4])
    title = "Capped versus uncapped SPSA/Adam (same learning rates)" if cap_comparison else "Fixed ideal-model SPSA comparison (paired seeds)"
    if comparison == "amplitude-caps":
        title = "Amplitude step-cap comparison (fixed learning rates and phase cap)"
    axes[0].set(title=title, ylabel="Accepted-center loss")
    axes[1].set(ylabel="Best feasible loss (includes probes)", xlabel="Completed updates")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(destination / "comparison.png", dpi=150)


def compare_saved_run(source, destination, *, updates=1000, seeds=(20260820, 20260821, 20260822), samples=600,
                      comparison="learning-rates", learning_rate_factors=None, amplitude_step_caps=None):
    """Paired seeds under either changed learning rates or removed step caps.

    Starts from the archived seed, not the last measured candidate. Exactly one
    of learning rates or step caps changes; perturbations, objective, sigmoid
    amplitude bounds and acceptance policy stay fixed. Discretization is checked.
    """
    source, destination = Path(source), Path(destination)
    if not isinstance(updates, int) or updates < 1 or not seeds:
        raise ValueError("Require positive updates and at least one seed")
    if comparison not in {"learning-rates", "step-caps", "amplitude-caps"}:
        raise ValueError("Comparison must be learning-rates, step-caps or amplitude-caps")
    if comparison != "amplitude-caps" and amplitude_step_caps is not None:
        raise ValueError("Amplitude step caps require amplitude-caps comparison")
    if comparison == "step-caps" and learning_rate_factors is not None:
        raise ValueError("Learning-rate factors cannot be combined with step-cap comparison")
    default_factors = (1.0,) if comparison == "amplitude-caps" else (1.0, 0.5)
    factors = tuple(float(value) for value in (
        default_factors if learning_rate_factors is None else learning_rate_factors))
    if not factors or any(not math.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("Learning-rate factors must be finite and positive")
    if len({f"lr_{value:g}" for value in factors}) != len(factors):
        raise ValueError("Learning-rate factors must have unique output labels")
    arms = (("capped", 1.0, True), ("uncapped", 1.0, False)) if comparison == "step-caps" else tuple(
        (f"lr_{factor:g}", factor, True) for factor in factors)
    cap_overrides = {}
    if comparison == "amplitude-caps":
        caps = tuple(float(value) for value in (() if amplitude_step_caps is None else amplitude_step_caps))
        if not caps or any(not math.isfinite(value) or value <= 0 for value in caps):
            raise ValueError("Amplitude step caps must be finite and positive")
        if len({f"cap_{value:g}" for value in caps}) != len(caps):
            raise ValueError("Amplitude step caps must have unique output labels")
        if len(factors) != 1:
            raise ValueError("Amplitude-cap comparison requires exactly one fixed learning-rate factor")
        cap_overrides = {f"cap_{cap:g}": cap for cap in caps}
        arms = tuple((name, factors[0], True) for name in cap_overrides)
    config = json.loads((source / "effective_config.json").read_text(encoding="utf-8"))
    options, composite = config["optimization"], config["composite"]
    amplitude_cap = composite.get("amplitude_logit_step", 0.02)
    phase_cap = options.get("phase_step_rad", 0.1)
    policy = FidelityAcceptance.from_options(composite)
    if policy is None:
        raise ValueError("This comparison requires saved fidelity-acceptance settings")
    with np.load(source / "epoch_0000.npz", allow_pickle=False) as saved:
        initial = np.r_[saved["amplitude_logits"], saved["phase_coefficients_rad"]]
        baseline_full = saved["phase_baseline_rad"]
        maximum = float(saved["amplitude_maximum"])
        objective = RobustnessConfig(
            omega_reference_rad_s=2*math.pi*config["calibration"]["rabi_at_100_hz"],
            amplitude_errors=saved["amplitude_errors"],
            detuning_ratios=saved["detuning_hz"] / config["calibration"]["rabi_at_100_hz"],
            **{key: options[key] for key in ("fidelity_threshold", "threshold_smoothing", "mean_weight", "threshold_weight", "cvar_weight", "cvar_confidence") if key in options})
    count = len(initial) // 2
    full_samples = len(baseline_full)
    if not isinstance(samples, int) or not count <= samples <= full_samples or samples % 10:
        raise ValueError("Samples must be divisible by 10 and between coefficient count and original resolution")
    duration = config["pulse"]["duration_s"]
    spline = clamped_bspline_design_matrix(samples, count)
    baseline = baseline_full[np.floor((np.arange(samples)+0.5)*full_samples/samples).astype(int)]
    dt = np.full(samples, duration / samples)

    def evaluate(parameters, matrix=spline, phase_baseline=baseline, durations=dt):
        logits, phase = np.split(parameters, 2)
        amplitude, _ = sigmoid_amplitude(logits, maximum)
        envelope = (matrix @ amplitude) * np.exp(1j * (matrix @ phase + phase_baseline))
        return simulated_objective(envelope.real, envelope.imag, durations, objective)

    # Assess downsampling on archived commands spanning both humps.
    full_spline = clamped_bspline_design_matrix(full_samples, count)
    checks = []
    for epoch in (0, 310, 355, 390, 415, 720, 750, 770):
        archive = source / ("epoch_0000.npz" if epoch == 0 else f"epoch_{epoch:04d}_trial_000.npz")
        if not archive.exists():
            continue
        with np.load(archive, allow_pickle=False) as saved:
            parameters = np.r_[saved["amplitude_logits"], saved["phase_coefficients_rad"]]
        coarse = evaluate(parameters)
        full = evaluate(parameters, full_spline, baseline_full, np.full(full_samples, duration/full_samples))
        checks.append({"epoch": epoch, "full_loss": full.loss, "simulation_loss": coarse.loss,
                       "loss_error": coarse.loss-full.loss,
                       "center_fidelity_error": coarse.center_fidelity-full.center_fidelity})
    destination.mkdir(parents=True, exist_ok=True)
    seed_eval = evaluate(initial)
    if not policy.accepts(seed_eval.center_fidelity):
        raise ValueError("Ideal simulated seed fails the saved fidelity floor")
    manifest = {"status": "running", "source_run": str(source.resolve()), "updates": updates, "seeds": list(seeds),
                "samples": samples, "original_samples": full_samples, "coefficients_per_block": count,
                "model": "Deterministic ideal command response; no noise, drift, calibration or hardware transfer model",
                "comparison": comparison,
                "learning_rate_factors": list(dict.fromkeys(arm[1] for arm in arms)),
                "initialization": "Archived epoch-zero amplitude logits and phase correction; fixed BB1 baseline",
                "step_caps": "Vary amplitude cap only; phase cap fixed" if comparison == "amplitude-caps" else "Configured caps versus no step caps" if comparison == "step-caps" else (
                    "Unchanged in both arms" if len(arms) == 2 else "Unchanged in all arms"),
                "arms": [{"name": name, "learning_rate_factor": factor,
                          "amplitude_logit_step_cap": cap_overrides.get(name, amplitude_cap) if capped else None,
                          "phase_step_cap_rad": phase_cap if capped else None}
                         for name, factor, capped in arms],
                "amplitude_maximum": maximum, "resolution_checks": checks,
                "objective": {"amplitude_errors": objective.amplitude_errors.tolist(), "detuning_ratios": objective.detuning_ratios.tolist()},
                "acceptance": {key: getattr(policy, key) for key in policy.__dataclass_fields__},
                "spsa": config["spsa"],
                "base_amplitude_logit_learning_rate": composite.get("amplitude_logit_learning_rate", 0.02),
                "base_phase_learning_rate": options.get("phase_learning_rate", 0.02)}
    manifest["source_sha256"] = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                 for name in ("effective_config.json", "epoch_0000.npz")}
    manifest["implementation_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                         for name in ("simulation.py", "core.py", "spsa.py", "acceptance.py", "amplitude.py")}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries, histories = [], {}
    for seed in seeds:
        for arm, factor, capped in arms:
            arm_amplitude_cap = cap_overrides.get(arm, amplitude_cap)
            label = f"seed_{seed}_{arm}"
            folder = destination / label
            folder.mkdir(exist_ok=True)
            started = time.perf_counter()
            spsa = replace(SPSAConfig(**config["spsa"]), seed=int(seed))
            parameters = initial.copy()
            a_state = AdamState(np.zeros(count), np.zeros(count))
            p_state = AdamState(np.zeros(count), np.zeros(count))
            current = seed_eval
            best_loss = current.loss
            best_parameters = parameters.copy()
            evaluations = 1
            rows = []
            checkpoints = []

            def record_checkpoint(epoch):
                full_current = evaluate(parameters, full_spline, baseline_full, np.full(full_samples, duration/full_samples))
                full_best = evaluate(best_parameters, full_spline, baseline_full, np.full(full_samples, duration/full_samples))
                checkpoints.append({"epoch": epoch, "evaluations": evaluations,
                    "loss": current.loss, "best_feasible_loss": best_loss,
                    "full_resolution_loss": full_current.loss,
                    "full_resolution_best_loss": full_best.loss,
                    "full_resolution_center_fidelity": full_current.center_fidelity,
                    "full_resolution_best_center_fidelity": full_best.center_fidelity})
                _write_csv(folder / "checkpoints.csv", checkpoints)
                np.savez_compressed(folder / f"checkpoint_{epoch:04d}.npz",
                                    parameters=parameters, best_parameters=best_parameters)
                return full_current, full_best
            cap_metrics = dict.fromkeys((
                "amplitude_coefficients_clipped", "phase_coefficients_clipped",
                "amplitude_coefficients_exceeding_reference_cap", "phase_coefficients_exceeding_reference_cap",
                "max_uncapped_amplitude_logit_step", "max_uncapped_phase_step_rad",
                "max_proposed_amplitude_logit_step", "max_proposed_phase_step_rad"), 0)

            def make_row(epoch, kind, metrics, plus=np.nan, minus=np.nan, plus_search=np.nan, minus_search=np.nan, scale=0.0,
                         plus_fidelity=np.nan, minus_fidelity=np.nan):
                return {"epoch": epoch, "kind": kind, "loss": current.loss,
                        "search_loss": policy.search_loss(current.loss, current.center_fidelity),
                        "center_fidelity": current.center_fidelity,
                        "minimum_center_fidelity": policy.minimum_center_fidelity,
                        "best_feasible_loss": best_loss,
                        "spsa_plus_loss_mean": plus, "spsa_minus_loss_mean": minus,
                        "spsa_plus_search_loss_mean": plus_search, "spsa_minus_search_loss_mean": minus_search,
                        "spsa_plus_center_fidelity_min": plus_fidelity, "spsa_minus_center_fidelity_min": minus_fidelity,
                        "step_scale": scale, "evaluations": evaluations, **metrics, **cap_metrics}

            rows.append(make_row(0, "initial", step_metrics(initial, initial, initial)))
            for epoch in range(1, updates+1):
                probes = {"plus": [], "minus": []}
                def measure(candidate, direction, member):
                    nonlocal evaluations, best_loss, best_parameters
                    result = evaluate(candidate)
                    evaluations += 1
                    probes[member].append(result)
                    if policy.accepts(result.center_fidelity) and result.loss < best_loss:
                        best_loss, best_parameters = result.loss, candidate.copy()
                    return policy.search_loss(result.loss, result.center_fidelity)
                estimate = estimate_spsa(parameters, measure, config=spsa, epoch=epoch-1)
                a, proposed_a_state = adam_update(parameters[:count], estimate.gradient[:count], a_state,
                    learning_rate=factor*composite.get("amplitude_logit_learning_rate", 0.02),
                    maximum_step=arm_amplitude_cap if capped else None)
                p, proposed_p_state = adam_update(parameters[count:], estimate.gradient[count:], p_state,
                    learning_rate=factor*options.get("phase_learning_rate", 0.02), maximum_step=phase_cap if capped else None)
                raw_a, raw_p = a, p
                if capped:
                    raw_a, _ = adam_update(parameters[:count], estimate.gradient[:count], a_state,
                        learning_rate=factor*composite.get("amplitude_logit_learning_rate", 0.02))
                    raw_p, _ = adam_update(parameters[count:], estimate.gradient[count:], p_state,
                        learning_rate=factor*options.get("phase_learning_rate", 0.02))
                a_exceeded = int(np.sum(abs(raw_a-parameters[:count]) > amplitude_cap+1e-12))
                p_exceeded = int(np.sum(abs(raw_p-parameters[count:]) > phase_cap+1e-12))
                cap_metrics = {
                    "amplitude_coefficients_clipped": int(np.sum(abs(raw_a-parameters[:count]) > arm_amplitude_cap+1e-12)) if capped else 0,
                    "phase_coefficients_clipped": p_exceeded if capped else 0,
                    "amplitude_coefficients_exceeding_reference_cap": a_exceeded,
                    "phase_coefficients_exceeding_reference_cap": p_exceeded,
                    "max_uncapped_amplitude_logit_step": float(np.max(abs(raw_a-parameters[:count]))),
                    "max_uncapped_phase_step_rad": float(np.max(abs(raw_p-parameters[count:]))),
                    "max_proposed_amplitude_logit_step": float(np.max(abs(a-parameters[:count]))),
                    "max_proposed_phase_step_rad": float(np.max(abs(p-parameters[count:]))),
                }
                before = parameters.copy()
                proposed = np.r_[a, p]
                accepted, scale = False, 0.0
                for attempt in range(policy.maximum_backtracks+1):
                    trial_scale = 0.5**attempt
                    trial_parameters = before + trial_scale*(proposed-before)
                    trial = evaluate(trial_parameters)
                    evaluations += 1
                    if policy.accepts(trial.center_fidelity):
                        parameters, current = trial_parameters, trial
                        a_state, p_state = proposed_a_state, proposed_p_state
                        accepted, scale = True, trial_scale
                        if current.loss < best_loss:
                            best_loss, best_parameters = current.loss, parameters.copy()
                        break
                rows.append(make_row(epoch, "accepted_center" if accepted else "retained_center",
                    step_metrics(before, proposed, parameters),
                    float(np.mean([v.loss for v in probes["plus"]])), float(np.mean([v.loss for v in probes["minus"]])),
                    float(np.mean(estimate.plus_losses)), float(np.mean(estimate.minus_losses)), scale,
                    min(v.center_fidelity for v in probes["plus"]), min(v.center_fidelity for v in probes["minus"])))
                if epoch % 25 == 0 or epoch == updates:
                    _write_csv(folder / "history.csv", rows)
                    print(f"{label}: {epoch}/{updates}; center={current.loss:.7f}; best={best_loss:.7f}; {time.perf_counter()-started:.1f}s", flush=True)
                if epoch in (100, 300) and epoch != updates:
                    record_checkpoint(epoch)
            losses = np.array([row["loss"] for row in rows])
            # Final and selected-best commands are also checked at the original resolution.
            full_current, full_best = record_checkpoint(updates)
            summary = {"seed": int(seed), "comparison": comparison, "comparison_arm": arm,
                       "amplitude_logit_learning_rate": factor*composite.get("amplitude_logit_learning_rate", 0.02),
                       "phase_learning_rate": factor*options.get("phase_learning_rate", 0.02),
                       "amplitude_logit_step_cap": arm_amplitude_cap if capped else None,
                       "phase_step_cap_rad": phase_cap if capped else None,
                       "step_caps_enabled": capped, "learning_rate_factor": factor, "initial_loss": seed_eval.loss,
                       "final_loss": current.loss, "best_loss": best_loss, "uphill_updates": int(np.sum(np.diff(losses)>0)),
                       "max_center_excursion_above_previous_best": float(np.max(losses-np.minimum.accumulate(losses))),
                       "rejected_updates": sum(row["kind"]=="retained_center" for row in rows),
                       "backtracked_updates": sum(0<float(row["step_scale"])<1 for row in rows),
                       "probe_penalty_epochs": sum(max(row["spsa_plus_search_loss_mean"]-row["spsa_plus_loss_mean"], row["spsa_minus_search_loss_mean"]-row["spsa_minus_loss_mean"])>1e-12 for row in rows[1:]),
                       "evaluations": evaluations, "elapsed_s": time.perf_counter()-started,
                       "full_resolution_final_loss": full_current.loss, "full_resolution_best_loss": full_best.loss,
                       "full_resolution_final_center_fidelity": full_current.center_fidelity,
                       "full_resolution_best_center_fidelity": full_best.center_fidelity}
            for block in ("amplitude", "phase"):
                summary[f"epochs_with_{block}_clipping"] = sum(row[f"{block}_coefficients_clipped"] > 0 for row in rows)
                summary[f"epochs_exceeding_reference_{block}_cap"] = sum(row[f"{block}_coefficients_exceeding_reference_cap"] > 0 for row in rows)
            summary["maximum_proposed_amplitude_logit_step"] = max(row["max_proposed_amplitude_logit_step"] for row in rows)
            summary["maximum_proposed_phase_step_rad"] = max(row["max_proposed_phase_step_rad"] for row in rows)
            summaries.append(summary)
            histories[label] = rows
            np.savez_compressed(folder / "final_parameters.npz", parameters=parameters, best_parameters=best_parameters)
            (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            write_diagnostics(rows, folder / "optimizer_diagnostics.png")
            _write_csv(destination / "summary.csv", summaries)
    write_comparison(destination, histories, summaries)
    manifest["status"] = "complete"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summaries
