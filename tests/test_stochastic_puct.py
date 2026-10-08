"""Stochastic PUCT is tested against known analytical toy chance laws."""
from __future__ import annotations

import json
import random

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.search.puct import (
    FAIR_TRANSITION_CAPABILITY_ID,
    FairPuctUnavailable,
    IncompleteChanceRollout,
    PublicSearchNode,
    StochasticPuct,
    TerminalReturn,
)

A = LegalAction("A", "toy-choice")
B = LegalAction("B", "toy-choice")


def _obs(outcome: str) -> Observation:
    return Observation("fair-toy-v1", json.dumps({"outcome": outcome}), outcome)


def root() -> PublicSearchNode:
    return PublicSearchNode("initial-public-history", _obs("unknown"), (A, B))


class BernoulliToy:
    fair_transition_capability_id = FAIR_TRANSITION_CAPABILITY_ID

    def __init__(self, pa: float = 0.8, pb: float = 0.2) -> None:
        self.probs = {"A": pa, "B": pb}

    def sample_transition(
        self,
        node: PublicSearchNode,
        action: LegalAction,
        search_rng: random.Random,
    ) -> PublicSearchNode:
        success = search_rng.random() < self.probs[action.action_id]
        outcome = TerminalReturn(success=success, progress=float(success), hp=float(success))
        reveal = "win" if success else "loss"
        return PublicSearchNode(
            f"{node.information_key}/{action.action_id}/{reveal}",
            _obs(reveal), (), outcome,
        )


def test_puct_favors_better_arm_and_estimates_chance_moments() -> None:
    model = BernoulliToy()
    report = StochasticPuct(model, seed=13).search(root(), simulations=1600)
    by_action = {row.action.action_id: row for row in report.actions}
    assert by_action["A"].visits > by_action["B"].visits * 3
    assert by_action["A"].success_mean == pytest.approx(0.8, abs=0.065)
    assert by_action["B"].success_mean == pytest.approx(0.2, abs=0.16)
    assert by_action["A"].progress_uniformized_variance == pytest.approx(
        4 * 0.8 * 0.2, abs=0.14
    )
    assert by_action["A"].hp_uniformized_variance is not None
    assert report.sampled_transitions == 1600
    assert report.expanded_information_sets == 1
    assert sum(row.visits for row in report.actions) == 1600


def test_deterministic_domain_and_search_seed() -> None:
    model = BernoulliToy(pa=1.0, pb=0.0)
    first = StochasticPuct(model, seed=23).search(root(), simulations=240)
    second = StochasticPuct(model, seed=23).search(root(), simulations=240)
    assert first == second
    assert first.actions[0].visits > first.actions[1].visits
    assert first.actions[0].success_mean == 1.0


def test_legal_masked_priors_are_used_by_search() -> None:
    search = StochasticPuct(
        BernoulliToy(pa=0.5, pb=0.5),
        prior_provider=lambda _obs, _actions: (0.9, 0.1),
        seed=7,
    )
    result = search.search(root(), simulations=1)
    assert result.actions[0].visits == 1
    assert result.actions[0].prior == pytest.approx(0.9)
    assert result.actions[1].prior == pytest.approx(0.1)


def test_exact_fork_like_backend_cannot_be_used_as_fair_transition_adapter() -> None:
    class Oracle:
        def fork(self, state: str) -> str:
            return state

    with pytest.raises(FairPuctUnavailable, match="oracle-only"):
        StochasticPuct(Oracle())  # type: ignore[arg-type]


def test_invalid_priors_are_rejected() -> None:
    search = StochasticPuct(
        BernoulliToy(), prior_provider=lambda _obs, _actions: (0.2, 0.2)
    )
    with pytest.raises(ValueError, match="sum to one"):
        search.search(root(), simulations=1)


def test_censored_rollouts_raise_instead_of_becoming_losses() -> None:
    class UnfinishedToy:
        fair_transition_capability_id = FAIR_TRANSITION_CAPABILITY_ID

        def sample_transition(
            self, node: PublicSearchNode, action: LegalAction, search_rng: random.Random
        ) -> PublicSearchNode:
            del search_rng
            return PublicSearchNode(
                f"{node.information_key}/{action.action_id}",
                _obs("still-playing"), (A, B),
            )

    with pytest.raises(IncompleteChanceRollout):
        StochasticPuct(UnfinishedToy(), max_depth=3).search(root(), simulations=1)


def test_bad_reused_history_key_is_rejected() -> None:
    class CyclicToy:
        fair_transition_capability_id = FAIR_TRANSITION_CAPABILITY_ID

        def sample_transition(
            self, node: PublicSearchNode, action: LegalAction, search_rng: random.Random
        ) -> PublicSearchNode:
            del action, search_rng
            return node

    with pytest.raises(ValueError, match="advance"):
        StochasticPuct(CyclicToy()).search(root(), simulations=1)
