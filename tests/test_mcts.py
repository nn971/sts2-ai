import json

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import InformationPolicy, Observation
from sts2_ai.search import SearchBudget, UctMcts
from sts2_ai.testing import MockLinearBackend


def _mock_value(observation: Observation) -> float:
    return float(json.loads(observation.payload_json)["value"]) / 10.0


def test_mcts_runs_budget_and_hits_transpositions() -> None:
    backend = MockLinearBackend(terminal_at=8)
    search = UctMcts(
        backend,
        policy=InformationPolicy("fair-test"),
        rollout_policy=HeuristicAgent(),
        value_fn=_mock_value,
        rollout_depth=8,
        rollout_batch_size=8,
        seed=3,
    )

    result = search.search("0", SearchBudget(max_simulations=32))

    assert result.root_state_hash == backend.exact_hash("0")
    assert sum(item.visits for item in result.evaluations) == 32
    assert result.transitions > 0
    assert result.expanded_nodes > 0
    assert result.transposition_hits > 0
    assert result.search_version == "oracle-exact-fused-batched-uct-v3"
    assert {item.action.action_id for item in result.evaluations} == {"inc-1", "inc-2"}


def test_mcts_batch_size_one_still_consumes_exact_budget() -> None:
    backend = MockLinearBackend(terminal_at=8)
    search = UctMcts(
        backend,
        policy=InformationPolicy("fair-test"),
        rollout_policy=HeuristicAgent(),
        value_fn=_mock_value,
        rollout_depth=4,
        rollout_batch_size=1,
        seed=7,
    )

    result = search.search("0", SearchBudget(max_simulations=17))

    assert sum(item.visits for item in result.evaluations) == 17
