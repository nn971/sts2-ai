"""Exact finite two-step chance tree, REINFORCE, and MC calibration tests."""
from __future__ import annotations

import pytest

from sts2_ai.training.nature_oracle import (
    ToyChanceTree,
    exact_proposal_weighted_gradient,
    exact_toy_reward_gradient,
    sample_proposal_weighted_gradient,
)


@pytest.mark.parametrize("theta", [-3.0, -0.6, 0.0, 0.8, 2.0])
def test_conditional_chance_exact_gradient_identity(theta: float) -> None:
    toy = ToyChanceTree.example()
    natural = exact_toy_reward_gradient(toy, theta)
    reweighted = exact_proposal_weighted_gradient(toy, theta)
    assert reweighted == pytest.approx(natural, abs=1e-12)


def test_analytic_gradient_matches_independent_finite_difference() -> None:
    toy = ToyChanceTree.example()
    theta, delta = 0.3, 1e-5
    _, analytic = exact_toy_reward_gradient(toy, theta)
    plus, _ = exact_toy_reward_gradient(toy, theta + delta)
    minus, _ = exact_toy_reward_gradient(toy, theta - delta)
    assert analytic == pytest.approx((plus - minus) / (2 * delta), abs=1e-8)


def test_sampled_importance_estimator_converges_to_exact() -> None:
    toy = ToyChanceTree.example()
    exact = exact_toy_reward_gradient(toy, 0.4)
    a = sample_proposal_weighted_gradient(toy, 0.4, seed=17, episodes=80000)
    b = sample_proposal_weighted_gradient(toy, 0.4, seed=17, episodes=80000)
    assert a == b
    assert a == pytest.approx(exact, abs=0.08)


def test_toy_oracle_rejects_bad_reference_or_proposal() -> None:
    toy = ToyChanceTree.example()
    from dataclasses import replace
    with pytest.raises(ValueError, match="support|probabilit"):
        replace(toy, q_first=(0.0, 1.0))
    with pytest.raises(ValueError, match="Positive episode"):
        sample_proposal_weighted_gradient(toy, 0.0, seed=0, episodes=0)
