from __future__ import annotations

import json

from sts2_ai.agents import OracleMctsAgent
from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.search import ActionEvaluation, SearchResult


class _FakeSearch:
    def __init__(self, result: SearchResult) -> None:
        self._result = result

    def search(self, state: str, budget: object) -> SearchResult:
        del state, budget
        return self._result


def test_oracle_mcts_never_prefers_unvisited_zero_placeholder() -> None:
    searched = LegalAction("play_card:searched", "play_card")
    unvisited = LegalAction("play_card:unvisited", "play_card")
    observation = Observation(
        policy_id="prototype-fair-v0",
        payload_json='{"phase":3}',
        observation_hash="obs",
    )
    result = SearchResult(
        root_state_hash="state",
        root_observation_hash="obs",
        root_observation_json=observation.payload_json,
        evaluations=(
            ActionEvaluation(
                action=searched,
                value=-0.4,
                visits=8,
                uncertainty=0.1,
            ),
            ActionEvaluation(
                action=unvisited,
                value=0.0,
                visits=0,
                uncertainty=None,
            ),
        ),
        expanded_nodes=1,
        transitions=8,
        transposition_hits=0,
        search_version="test-search",
        rollout_count=8,
        terminal_rollouts=3,
        boundary_rollouts=2,
        cutoff_rollouts=3,
        rollout_steps=40,
    )

    agent = OracleMctsAgent(_FakeSearch(result), simulations=8)  # type: ignore[arg-type]
    decision = agent.choose_state(
        "state-handle",
        observation,
        (searched, unvisited),
    )

    assert decision.action == searched
    metadata = json.loads(decision.metadata_json)
    assert metadata["rollout_count"] == 8
    assert metadata["terminal_rollouts"] == 3
    assert metadata["boundary_rollouts"] == 2
    assert metadata["cutoff_rollouts"] == 3
    assert metadata["rollout_steps"] == 40


def test_oracle_mcts_uses_best_mean_among_visited_actions() -> None:
    first = LegalAction("play_card:first", "play_card")
    second = LegalAction("play_card:second", "play_card")
    observation = Observation(
        policy_id="prototype-fair-v0",
        payload_json='{"phase":3}',
        observation_hash="obs",
    )
    result = SearchResult(
        root_state_hash="state",
        root_observation_hash="obs",
        root_observation_json=observation.payload_json,
        evaluations=(
            ActionEvaluation(first, value=-0.30, visits=20, uncertainty=0.1),
            ActionEvaluation(second, value=-0.20, visits=2, uncertainty=0.2),
        ),
        expanded_nodes=1,
        transitions=22,
        transposition_hits=0,
        search_version="test-search",
    )

    agent = OracleMctsAgent(_FakeSearch(result), simulations=22)  # type: ignore[arg-type]
    decision = agent.choose_state(
        "state-handle",
        observation,
        (first, second),
    )

    assert decision.action == second
