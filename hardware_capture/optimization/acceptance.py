"""Measured ideal-condition fidelity requirements for composite optimization."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FidelityAcceptance:
    minimum_center_fidelity: float
    center_penalty_weight: float = 10.0
    maximum_backtracks: int = 3
    verification_captures: int = 3

    @classmethod
    def from_options(cls, options):
        names = cls.__dataclass_fields__
        supplied = {name: options[name] for name in names if name in options}
        if not supplied:
            return None
        if "minimum_center_fidelity" not in supplied:
            raise ValueError(
                "Set composite.minimum_center_fidelity for acceptance settings"
            )
        policy = cls(**supplied)
        for name, upper in (
            ("minimum_center_fidelity", 1.0),
            ("center_penalty_weight", math.inf),
        ):
            value = getattr(policy, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= upper
            ):
                raise ValueError(f"composite.{name} must be finite and in (0, {upper}]")
        for name, minimum in (("maximum_backtracks", 0), ("verification_captures", 1)):
            value = getattr(policy, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"composite.{name} must be an integer >= {minimum}")
        return policy

    def accepts(self, fidelity):
        return math.isfinite(fidelity) and fidelity >= self.minimum_center_fidelity

    def search_loss(self, loss, fidelity):
        if not math.isfinite(fidelity):
            raise ValueError("Measured ideal-condition fidelity is not finite")
        return loss + self.center_penalty_weight * max(
            0.0, self.minimum_center_fidelity - fidelity
        )

    def capture_bounds(self, updates, directions):
        base = 1 + 2 * directions * updates + self.verification_captures
        return base + updates, base + updates * (1 + self.maximum_backtracks)
