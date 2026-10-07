import json

import pytest

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import InformationPolicy, LegalAction, Observation
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
    assert result.search_version == "oracle-exact-light-rollout-uct-v7-virtual-loss"
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


def test_virtual_loss_reservation_is_replaced_by_rollout_value() -> None:
    import sts2_ai.search.mcts as mcts

    action = LegalAction("inc-1", "increment", '{"amount":1}')
    edge = mcts._Edge(action=action, child_hash="child", visits=2, value_sum=-1.2)
    node = mcts._Node(
        handle="0",
        exact_hash="root",
        observation=Observation(
            policy_id="fair-test",
            payload_json='{"value":0}',
            observation_hash="obs",
        ),
        legal_actions=(action,),
        terminal=False,
        visits=3,
        value_sum=-1.5,
    )
    pending = mcts._PendingSimulation(
        path_nodes=(node,),
        path_edges=(edge,),
        leaf=node,
        immediate_value=None,
    )

    mcts.UctMcts._reserve(pending, -1.0)
    assert node.visits == 4
    assert edge.visits == 3
    assert node.value_sum == pytest.approx(-2.5)
    assert edge.value_sum == pytest.approx(-2.2)

    mcts.UctMcts._backup_reserved(pending, -0.25, -1.0)
    assert node.visits == 4
    assert edge.visits == 3
    assert node.value_sum == pytest.approx(-1.75)
    assert edge.value_sum == pytest.approx(-1.45)


def test_virtual_loss_must_be_bounded() -> None:
    backend = MockLinearBackend(terminal_at=8)
    with pytest.raises(ValueError, match="virtual_loss"):
        UctMcts(
            backend,
            policy=InformationPolicy("fair-test"),
            rollout_policy=HeuristicAgent(),
            virtual_loss=-1.5,
        )


def test_pessimistic_virtual_loss_discourages_reselecting_pending_edge() -> None:
    import sts2_ai.search.mcts as mcts

    backend = MockLinearBackend(terminal_at=8)
    search = UctMcts(
        backend,
        policy=InformationPolicy("fair-test"),
        rollout_policy=HeuristicAgent(),
        virtual_loss=-1.0,
        seed=0,
    )
    action_a = LegalAction("a", "increment", '{"amount":1}')
    action_b = LegalAction("b", "increment", '{"amount":2}')
    edge_a = mcts._Edge(
        action=action_a,
        child_hash="a-child",
        visits=10,
        value_sum=-4.0,
    )
    edge_b = mcts._Edge(
        action=action_b,
        child_hash="b-child",
        visits=10,
        value_sum=-4.1,
    )
    node = mcts._Node(
        handle="0",
        exact_hash="root",
        observation=Observation(
            policy_id="fair-test",
            payload_json='{"value":0}',
            observation_hash="obs",
        ),
        legal_actions=(action_a, action_b),
        terminal=False,
        visits=20,
        value_sum=-8.1,
        expanded=True,
        edges={"a": edge_a, "b": edge_b},
    )

    assert search._choose_unvisited_or_uct(node) is edge_a

    pending = mcts._PendingSimulation(
        path_nodes=(node,),
        path_edges=(edge_a,),
        leaf=None,
        immediate_value=-0.4,
    )
    search._reserve(pending, -1.0)

    assert search._choose_unvisited_or_uct(node) is edge_b
