"""Strict chance-ratio reference-law gating and numerical stability."""
from __future__ import annotations

import math

import pytest

from sts2_ai.training.chance_importance import (
    ChanceLaw,
    effective_sample_size,
    log_chance_ratio,
    mixture_proposal,
    trajectory_log_weight,
)


def law():
    return ChanceLaw(
        "toy-conditional-v1", "event-1", (0.8, 0.2), "SYNTHETIC_EXACT"
    )


def test_exact_log_ratios_and_trajectory_ess():
    p = law()
    logw = log_chance_ratio(
        p, (0.4, 0.6), event_id="event-1", selected_index=0,
    )
    assert math.exp(logw) == pytest.approx(2.0)
    other = ChanceLaw("toy-conditional-v1", "event-2", (0.25, 0.75),
                      "SYNTHETIC_EXACT")
    total = trajectory_log_weight([
        (p, (0.4, 0.6), "event-1", 0),
        (other, (0.5, 0.5), "event-2", 1),
    ])
    assert math.exp(total) == pytest.approx(3.0)
    assert effective_sample_size([0.0] * 5) == pytest.approx(5)
    assert effective_sample_size([500.0, 499.0, -math.inf]) > 1
    assert effective_sample_size([0.0, -math.inf]) == pytest.approx(1)


def test_reference_status_must_be_explicit_and_verified():
    with pytest.raises(ValueError, match="Unverified"):
        ChanceLaw("native", "event", (0.5, 0.5), "UNVERIFIED")
    with pytest.raises(ValueError, match="sum to one"):
        ChanceLaw("toy", "event", (0.4, 0.4), "SYNTHETIC_EXACT")
    with pytest.raises(ValueError, match="event/provenance"):
        log_chance_ratio(law(), (0.5, 0.5),
                         event_id="wrong", selected_index=0)
    with pytest.raises(ValueError, match="full|support|Invalid"):
        log_chance_ratio(law(), (1.0, 0.0),
                         event_id="event-1", selected_index=0)


def test_full_support_proposal_mixing():
    p = law()
    mix = mixture_proposal(p, (0.0, 1.0), alpha=0.7)
    assert mix == pytest.approx((0.24, 0.76))
    assert math.isclose(sum(mix), 1.0)
    with pytest.raises(ValueError, match="alpha"):
        mixture_proposal(p, (0.4, 0.6), alpha=1.0)
    with pytest.raises(ValueError, match="All importance weights"):
        effective_sample_size([-math.inf, -math.inf])


def test_zero_mass_reference_outcomes_preserve_importance_support():
    # q may put mass on a p=0 outcome, contributing a zero-weight rollout.
    p = ChanceLaw("toy-zero-v1", "event-zero", (1.0, 0.0),
                  "SYNTHETIC_EXACT")
    assert log_chance_ratio(
        p, (0.5, 0.5), event_id="event-zero", selected_index=1,
    ) == -math.inf
    q = mixture_proposal(p, (0.0, 1.0), alpha=0.8)
    assert q == pytest.approx((0.2, 0.8))
    assert log_chance_ratio(
        p, (1.0, 0.0), event_id="event-zero", selected_index=0,
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="zero proposal"):
        log_chance_ratio(
            p, (1.0, 0.0), event_id="event-zero", selected_index=1,
        )
