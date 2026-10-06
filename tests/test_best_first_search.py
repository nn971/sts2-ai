from __future__ import annotations

import json
from collections.abc import Sequence

from sts2_ai.emulator import InformationPolicy, Observation, StateHandle, Transition
from sts2_ai.search import BestFirstSearch, SearchBudget
from sts2_ai.testing.mock_backend import MockLinearBackend


class ExpandableTrackingBackend(MockLinearBackend):
    def __init__(self, terminal_at: int = 100) -> None:
        super().__init__(terminal_at)
        self.expand_calls = 0
        self.released: list[StateHandle] = []

    def expand(self, state: StateHandle) -> Sequence[Transition]:
        self.expand_calls += 1
        return tuple(
            Transition(
                parent=item.parent,
                action=item.action,
                child=item.child,
                terminal=item.terminal,
                exact_hash=self.exact_hash(item.child),
            )
            for item in (
                self.step(state, action)
                for action in self.legal_actions(state)
            )
        )

    def release_many(self, states: Sequence[StateHandle]) -> int:
        self.released.extend(states)
        return len(states)


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
    assert result.search_version == "best-first-v1"


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


def test_best_first_uses_expand_and_releases_temporary_states() -> None:
    backend = ExpandableTrackingBackend()
    state = backend.reset("managed-search")

    search = BestFirstSearch(
        backend=backend,
        evaluator=ValueEvaluator(),
        information_policy=InformationPolicy("fair"),
    )
    result = search.search(state, SearchBudget(max_nodes=8, max_depth=2))

    assert result.expanded_nodes > 0
    assert backend.expand_calls > 0
    assert backend.released
    assert state not in backend.released
