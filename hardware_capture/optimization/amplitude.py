"""Bounded amplitude coefficients represented by unrestricted sigmoid logits."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, logit

# Finite logits cannot represent exact endpoints. Used only when initializing
# a seed, never to project an Adam iterate or clip a measured waveform.
INITIAL_INTERIOR_FRACTION = 1e-6


def _validate(values, maximum):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Amplitude variables must be a nonempty finite vector")
    if not math.isfinite(maximum) or not 0 < maximum <= 1:
        raise ValueError("Normalized amplitude maximum must be in (0, 1]")
    return values


def amplitude_logits(amplitude: np.ndarray, maximum: float) -> np.ndarray:
    """Initialize z from normalized physical coefficients in [0, maximum].

    Seeds within 1e-6 of an endpoint (as a fraction of maximum) are moved to
    that interior margin. Out-of-range seeds are rejected, not silently capped.
    """
    amplitude = _validate(amplitude, maximum)
    if np.any(amplitude < 0) or np.any(amplitude > maximum):
        raise ValueError("Seed amplitude coefficients exceed the allowed interval")
    fraction = np.clip(
        amplitude / maximum, INITIAL_INTERIOR_FRACTION, 1 - INITIAL_INTERIOR_FRACTION
    )
    return logit(fraction)


def sigmoid_amplitude(
    logits: np.ndarray, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a=maximum*sigmoid(z) and da/dz, with stable tail evaluation.

    No iterate projection is used. Extreme finite logits can round to the
    endpoints in floating point; the resulting amplitude remains bounded.
    """
    logits = _validate(logits, maximum)
    fraction = expit(logits)
    amplitude = maximum * fraction
    derivative = maximum * fraction * expit(-logits)
    return amplitude, derivative
