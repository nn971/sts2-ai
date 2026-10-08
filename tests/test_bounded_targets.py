from __future__ import annotations

import math

import pytest

from sts2_ai.training.bounded_targets import (
    AUX_TARGET_VERSION,
    NormalizedMoments,
    OnlineBoundedMoments,
    mixture_moments,
    normalize_outcome,
    predicted_moments,
)


def test_fixed_normalization_and_bounds() -> None:
    assert AUX_TARGET_VERSION == "aux-v1"
    assert normalize_outcome(5, 0, 10) == 0.5
    for args in ((11, 0, 10), (0, 1, 1), (math.nan, 0, 1)):
        with pytest.raises(ValueError):
            normalize_outcome(*args)


def test_online_variance_mask_and_constant() -> None:
    moments = OnlineBoundedMoments()
    for _ in range(7):
        moments.add(0.4)
    assert not moments.estimate().variance_eligible
    moments.add(0.4)
    estimate = moments.estimate()
    assert estimate.count == 8
    assert estimate.mean == pytest.approx(0.4)
    assert estimate.uniformized_variance == pytest.approx(0.0, abs=1e-15)
    assert not estimate.variance_clamped


def test_bernoulli_variance_and_recorded_bessel_clamp() -> None:
    moments = OnlineBoundedMoments()
    for i in range(10_000):
        moments.add(float(i % 2))
    estimate = moments.estimate()
    assert estimate.mean == pytest.approx(0.5)
    assert estimate.uniformized_variance == 1.0
    assert estimate.variance_clamped


def test_direct_coupled_head_feasibility() -> None:
    for mean in (0.0, 1e-9, 0.2, 0.5, 1.0 - 1e-9, 1.0):
        for dispersion in (0.0, 0.5, 1.0):
            moments = predicted_moments(mean, dispersion)
            assert 0.0 <= moments.variance <= 4 * mean * (1 - mean)
    with pytest.raises(ValueError):
        NormalizedMoments(0.01, 0.3)
    with pytest.raises(ValueError):
        predicted_moments(0.5, math.nan)


def test_exact_mixture_total_variance() -> None:
    zero = NormalizedMoments(0.0, 0.0)
    one = NormalizedMoments(1.0, 0.0)
    fair = mixture_moments(((0.5, zero), (0.5, one)))
    assert fair.mean == 0.5
    assert fair.variance == 1.0
    skew = mixture_moments(((0.75, zero), (0.25, one)))
    assert skew.mean == 0.25
    assert skew.variance == 0.75
    constant = mixture_moments(((0.1, NormalizedMoments(0.3, 0)), (0.9, NormalizedMoments(0.3, 0))))
    assert constant.variance == pytest.approx(0)
    with pytest.raises(ValueError):
        mixture_moments(((0.4, zero), (0.4, one)))


def test_invalid_samples_rejected() -> None:
    moments = OnlineBoundedMoments()
    for bad in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            moments.add(bad)
    with pytest.raises(ValueError):
        moments.estimate()
