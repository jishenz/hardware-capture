"""Pure numerical helpers for measured-waveform closed-loop optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.special import expit


I2 = np.eye(2, dtype=np.complex128)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


@dataclass(frozen=True)
class BB1Initialization:
    """Amplitude and unwrapped-phase spline coefficients for split BB1 X(pi)."""

    amplitude_coefficients: np.ndarray
    phase_coefficients_rad: np.ndarray
    amplitude_fraction: float
    segment_phases_rad: np.ndarray


@dataclass(frozen=True)
class PhaseReference:
    """Frozen analysis-frame phase offset measured from split-BB1 end segments."""

    offset_rad: float
    first_segment_rad: float
    last_segment_rad: float
    segment_disagreement_rad: float
    capture_spread_rad: float
    segment_residuals_rad: np.ndarray
    first_capture_offsets_rad: np.ndarray
    last_capture_offsets_rad: np.ndarray


@dataclass(frozen=True)
class RobustnessConfig:
    """Fixed-target robustness-grid and threshold/CVaR objective settings."""

    omega_reference_rad_s: float
    amplitude_errors: np.ndarray
    detuning_ratios: np.ndarray
    fidelity_threshold: float = 0.99
    threshold_smoothing: float = 1e-3
    mean_weight: float = 0.25
    threshold_weight: float = 0.60
    cvar_weight: float = 0.15
    cvar_confidence: float = 0.95

    def validate(self) -> None:
        if not math.isfinite(self.omega_reference_rad_s) or self.omega_reference_rad_s <= 0:
            raise ValueError("omega_reference_rad_s must be finite and positive.")
        if np.asarray(self.amplitude_errors).size < 1:
            raise ValueError("amplitude_errors must not be empty.")
        if np.asarray(self.detuning_ratios).size < 1:
            raise ValueError("detuning_ratios must not be empty.")
        if not np.all(np.isfinite(np.asarray(self.amplitude_errors, dtype=float))):
            raise ValueError("amplitude_errors must be finite.")
        if not np.all(np.isfinite(np.asarray(self.detuning_ratios, dtype=float))):
            raise ValueError("detuning_ratios must be finite.")
        if not 0.0 < self.fidelity_threshold < 1.0:
            raise ValueError("fidelity_threshold must lie in (0, 1).")
        if self.threshold_smoothing <= 0.0:
            raise ValueError("threshold_smoothing must be positive.")
        if not 0.0 < self.cvar_confidence < 1.0:
            raise ValueError("cvar_confidence must lie in (0, 1).")
        weights = (self.mean_weight, self.threshold_weight, self.cvar_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Objective weights must be nonnegative.")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Mean, threshold, and CVaR weights must sum to one.")


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """Robustness heatmap and scalar objective for one captured pulse."""

    fidelity: np.ndarray
    loss: float
    mean_fidelity: float
    cvar_fidelity: float
    worst_fidelity: float
    center_fidelity: float
    fraction_at_or_above_threshold: float
    scenario_weights: np.ndarray


@dataclass(frozen=True)
class GrapeEvaluation:
    """Objective plus gradients with respect to captured normalized I and Q."""

    objective: ObjectiveEvaluation
    i_gradient: np.ndarray
    q_gradient: np.ndarray
    target_unitary: np.ndarray


@dataclass(frozen=True)
class AdamState:
    """Minimal state for one Adam parameter vector."""

    first_moment: np.ndarray
    second_moment: np.ndarray
    step: int = 0


def _spline_knots(n_coefficients: int, degree: int) -> np.ndarray:
    if n_coefficients <= degree:
        raise ValueError("Spline needs more coefficients than its degree.")
    interior_grid = np.linspace(0.0, 1.0, n_coefficients - degree + 1)
    return np.concatenate(
        (
            np.zeros(degree + 1),
            interior_grid[1:-1],
            np.ones(degree + 1),
        )
    )


def clamped_bspline_design_matrix(
    n_entries: int,
    n_coefficients: int,
    degree: int = 3,
) -> sparse.csr_matrix:
    """Return the sparse cubic B-spline map from coefficients to table entries."""

    if n_entries < 1:
        raise ValueError("n_entries must be positive.")
    knots = _spline_knots(n_coefficients, degree)
    entry_centers = (np.arange(n_entries, dtype=float) + 0.5) / n_entries
    design = BSpline.design_matrix(
        entry_centers,
        knots,
        degree,
        extrapolate=False,
    ).tocsr()
    design.sum_duplicates()
    return design


def _coefficient_locations(n_coefficients: int, degree: int) -> np.ndarray:
    knots = _spline_knots(n_coefficients, degree)
    return np.asarray(
        [
            np.mean(knots[index + 1 : index + degree + 1])
            for index in range(n_coefficients)
        ],
        dtype=float,
    )


def bb1_initialization(
    n_coefficients: int,
    *,
    pulse_duration_s: float,
    omega_at_amplitude_100_rad_s: float,
    degree: int = 3,
) -> BB1Initialization:
    """Build a five-segment split-BB1 X(pi) initial condition.

    The segment flip angles are pi/2, pi, 2*pi, pi, pi/2, so their duration
    weights are 1:2:4:2:1. The first and last phases are exactly zero. The
    amplitude is chosen so the nominal total transverse area is 5*pi.
    """

    if pulse_duration_s <= 0.0:
        raise ValueError("pulse_duration_s must be positive.")
    if omega_at_amplitude_100_rad_s <= 0.0:
        raise ValueError("omega_at_amplitude_100_rad_s must be positive.")
    amplitude_fraction = 5.0 * math.pi / (
        pulse_duration_s * omega_at_amplitude_100_rad_s
    )
    if amplitude_fraction > 1.0 + 1e-12:
        raise ValueError(
            "The calibrated amplitude-100 Rabi rate is too small for a split-BB1 "
            "X(pi) pulse of this duration. Increase the duration or recalibrate."
        )

    phi = math.acos(-0.25)
    wrapped_segment_phases = np.asarray([0.0, phi, 3.0 * phi, phi, 0.0])
    segment_phases = np.unwrap(wrapped_segment_phases)
    boundaries = np.asarray([0.1, 0.3, 0.7, 0.9, 1.0])
    coefficient_locations = _coefficient_locations(n_coefficients, degree)
    segment_indices = np.searchsorted(
        boundaries,
        coefficient_locations,
        side="right",
    )
    segment_indices = np.clip(segment_indices, 0, 4)
    phase_coefficients = segment_phases[segment_indices]
    amplitude_coefficients = np.full(
        n_coefficients,
        min(amplitude_fraction, 1.0),
        dtype=float,
    )
    return BB1Initialization(
        amplitude_coefficients=amplitude_coefficients,
        phase_coefficients_rad=phase_coefficients,
        amplitude_fraction=float(amplitude_fraction),
        segment_phases_rad=segment_phases,
    )


def command_tables(
    amplitude_coefficients: np.ndarray,
    phase_coefficients_rad: np.ndarray,
    spline: sparse.spmatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand coefficients into TNMR amplitude-percent and phase-degree tables."""

    amplitude = np.asarray(spline @ np.asarray(amplitude_coefficients), dtype=float)
    phase_rad = np.asarray(spline @ np.asarray(phase_coefficients_rad), dtype=float)
    if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(phase_rad)):
        raise ValueError("Expanded controls must be finite.")
    if np.min(amplitude) < -1e-12 or np.max(amplitude) > 1.0 + 1e-12:
        raise ValueError("Expanded amplitude lies outside [0, 1].")
    amplitude_percent = 100.0 * np.clip(amplitude, 0.0, 1.0)
    phase_degrees = np.mod(np.rad2deg(phase_rad), 360.0)
    return amplitude_percent, phase_degrees, phase_rad


def _circular_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 1:
        raise ValueError("A circular mean needs at least one value.")
    resultant = np.mean(np.exp(1.0j * values))
    if abs(resultant) < 1e-12:
        raise ValueError("Circular mean is undefined for these phases.")
    return float(np.angle(resultant))


def _circular_distance(left: float, right: float) -> float:
    return float(abs(np.angle(np.exp(1.0j * (left - right)))))


def _circular_spread(values: np.ndarray) -> float:
    resultant_length = float(abs(np.mean(np.exp(1.0j * np.asarray(values)))))
    resultant_length = min(1.0, max(resultant_length, np.finfo(float).tiny))
    return float(math.sqrt(max(0.0, -2.0 * math.log(resultant_length))))


def _segment_offset(
    values: np.ndarray,
    segment_slice: slice,
    *,
    expected_phase_rad: float,
    minimum_amplitude_fraction: float,
) -> float:
    segment = np.asarray(values[segment_slice], dtype=np.complex128)
    amplitude = np.abs(segment)
    threshold = minimum_amplitude_fraction * float(np.max(np.abs(values)))
    reliable = amplitude >= threshold
    if np.count_nonzero(reliable) < 3:
        raise ValueError("A BB1 reference segment has fewer than three reliable points.")
    observed = float(np.angle(np.sum(segment[reliable])))
    return float(np.angle(np.exp(1.0j * (observed - expected_phase_rad))))


def estimate_phase_reference(
    captures: np.ndarray,
    *,
    edge_trim_fraction: float = 0.20,
    minimum_amplitude_fraction: float = 0.20,
    maximum_segment_disagreement_rad: float = math.radians(10.0),
    maximum_capture_spread_rad: float = math.radians(5.0),
    maximum_bb1_pattern_residual_rad: float = math.radians(15.0),
) -> PhaseReference:
    """Estimate one constant phase offset from split-BB1 first/last interiors."""

    values = np.asarray(captures, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 100:
        raise ValueError("captures must have shape (at least 2, at least 100 entries).")
    if not 0.0 <= edge_trim_fraction < 0.5:
        raise ValueError("edge_trim_fraction must lie in [0, 0.5).")
    if not 0.0 < minimum_amplitude_fraction < 1.0:
        raise ValueError("minimum_amplitude_fraction must lie in (0, 1).")

    n_entries = values.shape[1]
    boundaries = np.asarray(
        [0, 0.1, 0.3, 0.7, 0.9, 1.0],
        dtype=float,
    )
    boundary_indices = np.rint(boundaries * n_entries).astype(int)
    segment_slices = []
    for start, stop in zip(boundary_indices[:-1], boundary_indices[1:]):
        trim = int(round(edge_trim_fraction * (stop - start)))
        segment_slices.append(slice(start + trim, stop - trim))
    first_slice = segment_slices[0]
    last_slice = segment_slices[-1]

    first_offsets = np.asarray(
        [
            _segment_offset(
                capture,
                first_slice,
                expected_phase_rad=0.0,
                minimum_amplitude_fraction=minimum_amplitude_fraction,
            )
            for capture in values
        ]
    )
    last_offsets = np.asarray(
        [
            _segment_offset(
                capture,
                last_slice,
                expected_phase_rad=0.0,
                minimum_amplitude_fraction=minimum_amplitude_fraction,
            )
            for capture in values
        ]
    )
    first_mean = _circular_mean(first_offsets)
    last_mean = _circular_mean(last_offsets)
    disagreement = _circular_distance(first_mean, last_mean)
    spread = max(_circular_spread(first_offsets), _circular_spread(last_offsets))
    if disagreement > maximum_segment_disagreement_rad:
        raise ValueError(
            "First/last BB1 phase references disagree by "
            f"{math.degrees(disagreement):.3f} degrees."
        )
    if spread > maximum_capture_spread_rad:
        raise ValueError(
            "Repeated BB1 phase-reference captures vary by "
            f"{math.degrees(spread):.3f} degrees."
        )

    offset = _circular_mean(np.concatenate((first_offsets, last_offsets)))
    phi = math.acos(-0.25)
    expected_segment_phases = np.asarray([0.0, phi, 3.0 * phi, phi, 0.0])
    segment_offsets = np.asarray(
        [
            _circular_mean(
                np.asarray(
                    [
                        _segment_offset(
                            capture,
                            segment_slice,
                            expected_phase_rad=expected_phase,
                            minimum_amplitude_fraction=minimum_amplitude_fraction,
                        )
                        for capture in values
                    ]
                )
            )
            for segment_slice, expected_phase in zip(
                segment_slices,
                expected_segment_phases,
            )
        ]
    )
    segment_residuals = np.asarray(
        [_circular_distance(segment_offset, offset) for segment_offset in segment_offsets]
    )
    if np.max(segment_residuals) > maximum_bb1_pattern_residual_rad:
        raise ValueError(
            "The corrected five-segment BB1 phase pattern has a maximum residual of "
            f"{math.degrees(float(np.max(segment_residuals))):.3f} degrees. "
            "Check the Q-sign convention, demodulation frequency, and pulse window."
        )
    return PhaseReference(
        offset_rad=offset,
        first_segment_rad=first_mean,
        last_segment_rad=last_mean,
        segment_disagreement_rad=disagreement,
        capture_spread_rad=spread,
        segment_residuals_rad=segment_residuals,
        first_capture_offsets_rad=first_offsets,
        last_capture_offsets_rad=last_offsets,
    )


def apply_phase_reference(values: np.ndarray, reference: PhaseReference) -> np.ndarray:
    """Rotate captured I/Q into the frozen BB1 analysis frame."""

    return np.asarray(values, dtype=np.complex128) * np.exp(-1.0j * reference.offset_rad)


def x_pi_target() -> np.ndarray:
    """Return the fixed ideal X(pi) gate, including its conventional -i phase."""

    return -1.0j * SIGMA_X


def _step_and_control_derivatives(
    normalized_i: float,
    normalized_q: float,
    amplitude_scale: np.ndarray,
    detuning_ratio: np.ndarray,
    dt_s: float,
    omega_reference_rad_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drive_derivative_scale = omega_reference_rad_s * amplitude_scale
    omega_x = drive_derivative_scale * normalized_i
    omega_y = drive_derivative_scale * normalized_q
    delta = omega_reference_rad_s * detuning_ratio
    magnitude_squared = omega_x**2 + omega_y**2 + delta**2
    magnitude = np.sqrt(magnitude_squared)
    half_angle = 0.5 * dt_s * magnitude
    cosine = np.cos(half_angle)
    sin_over_magnitude = 0.5 * dt_s * np.sinc(half_angle / math.pi)

    derivative_k_s = np.empty_like(magnitude_squared)
    small = np.abs(half_angle) < 1e-4
    s = magnitude_squared[small]
    derivative_k_s[small] = (
        -(dt_s**3) / 48.0
        + (dt_s**5) * s / 1920.0
        - (dt_s**7) * s**2 / 215040.0
    )
    r = magnitude[~small]
    a = half_angle[~small]
    derivative_k_s[~small] = (
        0.5 * dt_s * r * np.cos(a) - np.sin(a)
    ) / (2.0 * r**3)
    derivative_cosine_s = -0.25 * dt_s * sin_over_magnitude

    h_dot_sigma = np.empty((len(amplitude_scale), 2, 2), dtype=np.complex128)
    h_dot_sigma[:, 0, 0] = delta
    h_dot_sigma[:, 0, 1] = omega_x - 1.0j * omega_y
    h_dot_sigma[:, 1, 0] = omega_x + 1.0j * omega_y
    h_dot_sigma[:, 1, 1] = -delta
    step = cosine[:, None, None] * I2 - 1.0j * sin_over_magnitude[:, None, None] * h_dot_sigma

    derivative_cosine_x = 2.0 * omega_x * derivative_cosine_s
    derivative_cosine_y = 2.0 * omega_y * derivative_cosine_s
    derivative_k_x = 2.0 * omega_x * derivative_k_s
    derivative_k_y = 2.0 * omega_y * derivative_k_s
    derivative_x = (
        derivative_cosine_x[:, None, None] * I2
        - 1.0j
        * (
            derivative_k_x[:, None, None] * h_dot_sigma
            + sin_over_magnitude[:, None, None] * SIGMA_X
        )
    )
    derivative_y = (
        derivative_cosine_y[:, None, None] * I2
        - 1.0j
        * (
            derivative_k_y[:, None, None] * h_dot_sigma
            + sin_over_magnitude[:, None, None] * SIGMA_Y
        )
    )
    derivative_x *= drive_derivative_scale[:, None, None]
    derivative_y *= drive_derivative_scale[:, None, None]
    return step, derivative_x, derivative_y


def _objective_from_fidelities(
    fidelities: np.ndarray,
    config: RobustnessConfig,
) -> ObjectiveEvaluation:
    infidelities = 1.0 - fidelities
    count = fidelities.size
    tail_count = max(1, int(math.ceil((1.0 - config.cvar_confidence) * count)))
    tail_indices = np.argpartition(infidelities, count - tail_count)[-tail_count:]
    deficit = config.fidelity_threshold - fidelities
    threshold_penalty = config.threshold_smoothing * np.logaddexp(
        0.0,
        deficit / config.threshold_smoothing,
    )
    scenario_weights = (
        config.mean_weight
        + config.threshold_weight * expit(deficit / config.threshold_smoothing)
    ) / count
    scenario_weights[tail_indices] += config.cvar_weight / tail_count
    loss = (
        config.mean_weight * float(np.mean(infidelities))
        + config.threshold_weight * float(np.mean(threshold_penalty))
        + config.cvar_weight * float(np.mean(infidelities[tail_indices]))
    )

    amp_axis = np.asarray(config.amplitude_errors, dtype=float)
    det_axis = np.asarray(config.detuning_ratios, dtype=float)
    grid = fidelities.reshape(len(det_axis), len(amp_axis))
    center_amp = int(np.argmin(np.abs(amp_axis)))
    center_det = int(np.argmin(np.abs(det_axis)))
    return ObjectiveEvaluation(
        fidelity=grid,
        loss=loss,
        mean_fidelity=float(np.mean(fidelities)),
        cvar_fidelity=float(1.0 - np.mean(infidelities[tail_indices])),
        worst_fidelity=float(np.min(fidelities)),
        center_fidelity=float(grid[center_det, center_amp]),
        fraction_at_or_above_threshold=float(
            np.mean(fidelities >= config.fidelity_threshold)
        ),
        scenario_weights=scenario_weights,
    )


def robustness_grape_gradient(
    normalized_i: np.ndarray,
    normalized_q: np.ndarray,
    dt_s: np.ndarray,
    config: RobustnessConfig,
) -> GrapeEvaluation:
    """Evaluate a fixed X(pi) objective and its exact per-entry GRAPE gradient."""

    config.validate()
    i_values = np.asarray(normalized_i, dtype=float).reshape(-1)
    q_values = np.asarray(normalized_q, dtype=float).reshape(-1)
    durations = np.asarray(dt_s, dtype=float).reshape(-1)
    if not (len(i_values) == len(q_values) == len(durations)) or len(i_values) < 1:
        raise ValueError("I, Q, and dt arrays must have the same nonzero length.")
    if not np.all(np.isfinite(i_values)) or not np.all(np.isfinite(q_values)):
        raise ValueError("Captured normalized I/Q must be finite.")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("All entry durations must be finite and positive.")

    amp_axis = np.asarray(config.amplitude_errors, dtype=float).reshape(-1)
    det_axis = np.asarray(config.detuning_ratios, dtype=float).reshape(-1)
    amp_flat = np.tile(amp_axis, len(det_axis))
    det_flat = np.repeat(det_axis, len(amp_axis))
    amplitude_scale = 1.0 + amp_flat
    if np.any(amplitude_scale < 0.0):
        raise ValueError("Amplitude errors must not make the drive scale negative.")

    scenario_count = len(amp_flat)
    forward = np.empty(
        (len(i_values) + 1, scenario_count, 2, 2),
        dtype=np.complex128,
    )
    forward[0] = I2
    for index, (i_value, q_value, duration) in enumerate(
        zip(i_values, q_values, durations)
    ):
        step, _, _ = _step_and_control_derivatives(
            float(i_value),
            float(q_value),
            amplitude_scale,
            det_flat,
            float(duration),
            config.omega_reference_rad_s,
        )
        forward[index + 1] = np.einsum(
            "gij,gjk->gik",
            step,
            forward[index],
        )

    target = x_pi_target()
    target_dag = target.conj().T
    overlap = np.einsum("ij,gji->g", target_dag, forward[-1])
    fidelities = np.abs(overlap) ** 2 / 4.0
    objective = _objective_from_fidelities(fidelities, config)

    i_gradient = np.empty_like(i_values)
    q_gradient = np.empty_like(q_values)
    backward_after = np.broadcast_to(I2, (scenario_count, 2, 2)).copy()
    weighted_overlap = objective.scenario_weights * np.conj(overlap)
    for index in range(len(i_values) - 1, -1, -1):
        step, derivative_i, derivative_q = _step_and_control_derivatives(
            float(i_values[index]),
            float(q_values[index]),
            amplitude_scale,
            det_flat,
            float(durations[index]),
            config.omega_reference_rad_s,
        )
        total_derivative_i = np.einsum(
            "gij,gjk,gkl->gil",
            backward_after,
            derivative_i,
            forward[index],
        )
        total_derivative_q = np.einsum(
            "gij,gjk,gkl->gil",
            backward_after,
            derivative_q,
            forward[index],
        )
        overlap_derivative_i = np.einsum(
            "ij,gji->g",
            target_dag,
            total_derivative_i,
        )
        overlap_derivative_q = np.einsum(
            "ij,gji->g",
            target_dag,
            total_derivative_q,
        )
        i_gradient[index] = -0.5 * float(
            np.real(np.sum(weighted_overlap * overlap_derivative_i))
        )
        q_gradient[index] = -0.5 * float(
            np.real(np.sum(weighted_overlap * overlap_derivative_q))
        )
        backward_after = np.einsum(
            "gij,gjk->gik",
            backward_after,
            step,
        )
    return GrapeEvaluation(
        objective=objective,
        i_gradient=i_gradient,
        q_gradient=q_gradient,
        target_unitary=target,
    )


def polar_control_gradients(
    normalized_i: np.ndarray,
    normalized_q: np.ndarray,
    i_gradient: np.ndarray,
    q_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert captured I/Q gradients into amplitude and phase gradients."""

    i_values = np.asarray(normalized_i, dtype=float)
    q_values = np.asarray(normalized_q, dtype=float)
    gradient_i = np.asarray(i_gradient, dtype=float)
    gradient_q = np.asarray(q_gradient, dtype=float)
    if not (
        i_values.shape == q_values.shape == gradient_i.shape == gradient_q.shape
    ):
        raise ValueError("I/Q controls and gradients must have matching shapes.")
    amplitude = np.hypot(i_values, q_values)
    phase = np.arctan2(q_values, i_values)
    amplitude_gradient = gradient_i * np.cos(phase) + gradient_q * np.sin(phase)
    phase_gradient = amplitude * (
        -gradient_i * np.sin(phase) + gradient_q * np.cos(phase)
    )
    return amplitude_gradient, phase_gradient


def adam_update(
    parameters: np.ndarray,
    gradient: np.ndarray,
    state: AdamState | None,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    maximum_step: float | None = None,
) -> tuple[np.ndarray, AdamState]:
    """Apply one gradient-descent Adam update and return its new state."""

    values = np.asarray(parameters, dtype=float)
    derivative = np.asarray(gradient, dtype=float)
    if values.shape != derivative.shape:
        raise ValueError("Adam parameters and gradient must have matching shapes.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if state is None:
        state = AdamState(np.zeros_like(values), np.zeros_like(values), 0)
    step_number = state.step + 1
    first = beta1 * state.first_moment + (1.0 - beta1) * derivative
    second = beta2 * state.second_moment + (1.0 - beta2) * derivative**2
    first_hat = first / (1.0 - beta1**step_number)
    second_hat = second / (1.0 - beta2**step_number)
    update = learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
    if maximum_step is not None:
        if maximum_step <= 0.0:
            raise ValueError("maximum_step must be positive when supplied.")
        update = np.clip(update, -maximum_step, maximum_step)
    return values - update, AdamState(first, second, step_number)
