from __future__ import annotations

import json

from sts2_ai.emulator import InformationPolicy, Observation
from sts2_ai.search import BestFirstSearch, SearchBudget
from sts2_ai.testing.mock_backend import MockLinearBackend


class ValueEvaluator:
    def evaluate(self, observation: Observation, *, terminal: bool) -> float:
        del terminal
        return float(json.loads(observation.payload_json)["value"])


def test_best_first_search_prefers_better_depth_limited_root_action() -> None:
    backend = MockLinearBackend(terminal_at=100)
    state = backend.reset("search-test")

    search = BestFirstSearch(
        backend=backend,
        evaluator=ValueEvaluator(),
        information_policy=InformationPolicy("fair"),
    )
    result = search.search(
        state,
        SearchBudget(max_nodes=10, max_depth=2),
    )

    by_action = {item.action.action_id: item for item in result.evaluations}
    assert by_action["inc-2"].value > by_action["inc-1"].value
    assert by_action["inc-1"].visits >= 1
    assert by_action["inc-2"].visits >= 1
    assert result.expanded_nodes <= 10
    assert result.search_version == "best-first-v0"


def test_best_first_search_returns_empty_result_for_terminal_root() -> None:
    backend = MockLinearBackend(terminal_at=0)
    state = backend.reset("terminal")

    search = BestFirstSearch(
        backend=backend,
        evaluator=ValueEvaluator(),
        information_policy=InformationPolicy("fair"),
    )
    result = search.search(state, SearchBudget(max_nodes=10))

    assert result.evaluations == ()
    assert result.expanded_nodes == 0
