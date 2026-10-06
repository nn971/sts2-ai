import json

from sts2_ai.emulator import InformationPolicy, Observation
from sts2_ai.search import (
    PrototypeFlatRolloutSearch,
    PrototypeUcbRolloutSearch,
    SearchBudget,
)
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


def test_prototype_leaf_value_uses_canonical_snake_case_fields() -> None:
    victory = Observation(
        policy_id="prototype-fair-v0",
        payload_json=json.dumps(
            {
                "act": 3,
                "floor": 6,
                "hp": 20,
                "max_hp": 70,
                "gold": 100,
                "deck": [{}, {}],
                "terminal_outcome": "victory",
            }
        ),
        observation_hash="victory",
    )
    defeat = Observation(
        policy_id="prototype-fair-v0",
        payload_json=json.dumps(
            {
                "act": 3,
                "floor": 6,
                "hp": 0,
                "max_hp": 70,
                "gold": 100,
                "deck": [{}, {}],
                "terminal_outcome": "defeat",
            }
        ),
        observation_hash="defeat",
    )

    victory_value = PrototypeFlatRolloutSearch._prototype_leaf_value(victory)
    defeat_value = PrototypeFlatRolloutSearch._prototype_leaf_value(defeat)

    assert victory_value > 10_000
    assert defeat_value < 0
    assert victory_value - defeat_value > 19_000


def test_same_search_object_repeats_exact_state_deterministically() -> None:
    backend = MockLinearBackend(terminal_at=30)
    policy = InformationPolicy("fair-test")

    def value(observation: Observation) -> float:
        payload = json.loads(observation.payload_json)
        return float(payload["value"])

    search = PrototypeFlatRolloutSearch(
        backend,
        policy,
        seed=23,
        rollout_depth=5,
        leaf_value=value,
    )

    first = search.search("0", SearchBudget(max_nodes=40))
    second = search.search("0", SearchBudget(max_nodes=40))

    assert first == second
    assert first.search_version == "prototype-flat-rollout-v2"



def test_ucb_rollout_search_concentrates_visits_on_better_root_action() -> None:
    backend = MockLinearBackend(terminal_at=100)
    policy = InformationPolicy("fair-test")

    def value(observation: Observation) -> float:
        payload = json.loads(observation.payload_json)
        return float(payload["value"])

    flat = PrototypeFlatRolloutSearch(
        backend,
        policy,
        seed=31,
        rollout_depth=1,
        rollout_batch_size=1,
        leaf_value=value,
    ).search("0", SearchBudget(max_nodes=30))

    adaptive = PrototypeUcbRolloutSearch(
        backend,
        policy,
        seed=31,
        rollout_depth=1,
        rollout_batch_size=1,
        leaf_value=value,
        exploration=1.0,
    ).search("0", SearchBudget(max_nodes=30))

    flat_by_amount = {
        evaluation.action.payload_json: evaluation
        for evaluation in flat.evaluations
    }
    adaptive_by_amount = {
        evaluation.action.payload_json: evaluation
        for evaluation in adaptive.evaluations
    }

    assert abs(
        flat_by_amount['{"amount":2}'].visits
        - flat_by_amount['{"amount":1}'].visits
    ) <= 1
    assert (
        adaptive_by_amount['{"amount":2}'].visits
        > adaptive_by_amount['{"amount":1}'].visits
    )
    assert adaptive.search_version == "prototype-ucb-rollout-v0"
