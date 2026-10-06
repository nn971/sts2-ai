import json

from sts2_ai.emulator import InformationPolicy, Observation
from sts2_ai.search import PrototypeFlatRolloutSearch, SearchBudget
from sts2_ai.testing import MockLinearBackend


def test_flat_rollout_search_prefers_larger_progress() -> None:
    backend = MockLinearBackend(terminal_at=20)

    def value(observation: Observation) -> float:
        payload = json.loads(observation.payload_json)
        return float(payload["value"])

    search = PrototypeFlatRolloutSearch(
        backend,
        InformationPolicy("fair-test"),
        seed=7,
        rollout_depth=1,
        leaf_value=value,
    )
    result = search.search(backend.reset("ignored"), SearchBudget(max_nodes=20))

    assert result.expanded_nodes == 20
    assert len(result.evaluations) == 2

    by_kind = {evaluation.action.payload_json: evaluation for evaluation in result.evaluations}
    assert by_kind['{"amount":2}'].value > by_kind['{"amount":1}'].value
    assert all(evaluation.visits > 0 for evaluation in result.evaluations)


def test_flat_rollout_search_is_deterministic_under_node_budget() -> None:
    policy = InformationPolicy("fair-test")

    def value(observation: Observation) -> float:
        payload = json.loads(observation.payload_json)
        return float(payload["value"])

    first = PrototypeFlatRolloutSearch(
        MockLinearBackend(terminal_at=30),
        policy,
        seed=11,
        rollout_depth=5,
        leaf_value=value,
    ).search("0", SearchBudget(max_nodes=40))

    second = PrototypeFlatRolloutSearch(
        MockLinearBackend(terminal_at=30),
        policy,
        seed=11,
        rollout_depth=5,
        leaf_value=value,
    ).search("0", SearchBudget(max_nodes=40))

    assert first == second
