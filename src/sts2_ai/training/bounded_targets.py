"""Numerically stable aux-v1 bounded mean/variance target primitives.

These functions operate on independently sampled, *fair* outcomes. Oracle-exact
search leaves and truncated episodes are not eligible as labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

AUX_TARGET_VERSION = "aux-v1"
DEFAULT_MIN_VARIANCE_SAMPLES = 8


def normalize_outcome(value: float, lower: float, upper: float) -> float:
    """Map a fixed, versioned outcome interval to [0, 1], rejecting invalid data."""
    if not all(math.isfinite(x) for x in (value, lower, upper)):
        raise ValueError("outcome and bounds must be finite")
    if upper <= lower:
        raise ValueError("outcome upper bound must exceed lower bound")
    if value < lower or value > upper:
        raise ValueError("outcome is outside its declared bounds")
    return (value - lower) / (upper - lower)


@dataclass(frozen=True, slots=True)
class NormalizedMoments:
    """A realizable mean and uniformized population variance on [0, 1]."""

    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mean) and math.isfinite(self.variance)):
            raise ValueError("moments must be finite")
        if not 0.0 <= self.mean <= 1.0:
            raise ValueError("mean must be in [0, 1]")
        if self.variance < 0.0 or self.variance > 4.0 * self.mean * (1.0 - self.mean) + 1e-12:
            raise ValueError("uniformized variance is infeasible for this mean")


def predicted_moments(mean: float, dispersion: float) -> NormalizedMoments:
    """Couple direct mean/dispersion heads without subtracting raw second moments."""
    if not math.isfinite(dispersion) or not 0.0 <= dispersion <= 1.0:
        raise ValueError("dispersion must be in [0, 1]")
    if not math.isfinite(mean) or not 0.0 <= mean <= 1.0:
        raise ValueError("mean must be in [0, 1]")
    return NormalizedMoments(mean=mean, variance=4.0 * mean * (1.0 - mean) * dispersion)


@dataclass(frozen=True, slots=True)
class SampleEstimate:
    count: int
    mean: float
    uniformized_variance: float | None
    variance_clamped: bool

    @property
    def variance_eligible(self) -> bool:
        return self.uniformized_variance is not None


@dataclass(slots=True)
class OnlineBoundedMoments:
    """Stable Welford statistics for independently sampled normalized outcomes."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, normalized_outcome: float) -> None:
        x = float(normalized_outcome)
        if not math.isfinite(x) or x < 0.0 or x > 1.0:
            raise ValueError("normalized sample must be finite and in [0, 1]")
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def estimate(self, *, min_variance_samples: int = DEFAULT_MIN_VARIANCE_SAMPLES) -> SampleEstimate:
        """Return unbiased sample variance, masked for undersampled labels.

        Bessel correction can place 4s² marginally above one. Explicitly record
        that clamp, rather than silently altering the estimator.
        """
        if min_variance_samples < 2:
            raise ValueError("min_variance_samples must be at least two")
        if self.count == 0:
            raise ValueError("cannot estimate from zero samples")
        if self.count < min_variance_samples:
            return SampleEstimate(self.count, self.mean, None, False)
        variance = 4.0 * max(0.0, self.m2) / (self.count - 1)
        clamped = variance > 1.0
        return SampleEstimate(self.count, self.mean, min(1.0, variance), clamped)


def mixture_moments(
    branches: tuple[tuple[float, NormalizedMoments], ...],
) -> NormalizedMoments:
    """Exact mixture via total variance: Σp v_i + 4Σp(μ_i-μ)²."""
    if not branches:
        raise ValueError("mixture must have at least one branch")
    weights = [probability for probability, _ in branches]
    if any(not math.isfinite(p) or p < 0.0 for p in weights):
        raise ValueError("mixture weights must be nonnegative and finite")
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("mixture weights must sum to one")
    mean = math.fsum(probability * value.mean for probability, value in branches)
    variance = math.fsum(
        probability * (value.variance + 4.0 * (value.mean - mean) ** 2)
        for probability, value in branches
    )
    # Arithmetic roundoff at a valid endpoint may exceed the feasible cap.
    mean = min(1.0, max(0.0, mean))
    variance = min(4.0 * mean * (1.0 - mean), max(0.0, variance))
    return NormalizedMoments(mean, variance)
