import json

import pytest

from sts2_ai.agents import RoutePlanner, RoutePlanningAgent
from sts2_ai.emulator import LegalAction, Observation


def _frame(
    *,
    hp: int = 70,
    gold: int = 100,
    history: list[dict[str, object]] | None = None,
    relics: list[dict[str, object]] | None = None,
) -> Observation:
    # Both legal next nodes are Combats; only their visible *successors*
    # differ. This distinguishes DAG lookahead from local room scoring.
    visible = {
        "act": 1,
        "floor": 1,
        "hp": hp,
        "max_hp": 70,
        "gold": gold,
        "completed_rooms": history or [],
        "relics": relics or [],
        "map": [
            {"node_id": "a", "room_type": 0, "floor": 2, "next_node_ids": ["danger"]},
            {"node_id": "b", "room_type": 0, "floor": 2, "next_node_ids": ["recover"]},
            {"node_id": "danger", "room_type": 1, "floor": 3, "next_node_ids": []},
            {"node_id": "recover", "room_type": 4, "floor": 3, "next_node_ids": []},
        ],
    }
    return Observation(
        policy_id="prototype-fair-v0",
        payload_json=json.dumps(visible),
        observation_hash="route-observation",
    )


def _map_actions() -> tuple[LegalAction, ...]:
    return (
        LegalAction("action:a", "choose_map_node", '{"NodeId":"a"}'),
        LegalAction("action:b", "choose_map_node", '{"NodeId":"b"}'),
    )


def test_route_planner_looks_beyond_identical_immediate_rooms() -> None:
    agent = RoutePlanningAgent()
    low = agent.choose(_frame(hp=15), _map_actions())
    high = agent.choose(_frame(hp=65), _map_actions())

    assert low.action.action_id == "action:b"
    assert json.loads(low.metadata_json)["route_node_ids"] == ["b", "recover"]
    assert high.action.action_id == "action:a"
    assert json.loads(high.metadata_json)["route_node_ids"] == ["a", "danger"]


def test_horizon_one_cannot_see_route_successors() -> None:
    one_step = RoutePlanningAgent(horizon=1)
    full = RoutePlanningAgent(horizon=6)
    assert one_step.choose(_frame(hp=15), _map_actions()).action.action_id == "action:b"
    assert full.choose(_frame(hp=65), _map_actions()).action.action_id == "action:a"


def test_ledger_visit_parity_uses_only_completed_current_act_history() -> None:
    planner = RoutePlanner(horizon=1)
    relics = [{"relic_id": "proto.relic.trail_ledger"}]
    first = json.loads(_frame(relics=relics).payload_json)
    one_completed = json.loads(
        _frame(
            relics=relics,
            history=[{"act": 1, "floor": 1, "node_id": "old", "room_type": 0}],
        ).payload_json
    )
    previous_act = json.loads(
        _frame(
            relics=relics,
            history=[{"act": 0, "floor": 1, "node_id": "old", "room_type": 0}],
        ).payload_json
    )

    first_score = planner.evaluate(first, ["a"])["a"].score
    ledger_score = planner.evaluate(one_completed, ["a"])["a"].score
    previous_score = planner.evaluate(previous_act, ["a"])["a"].score
    assert ledger_score > first_score
    assert previous_score == first_score


def test_route_planning_is_public_information_only_and_repeatable() -> None:
    agent = RoutePlanningAgent()
    raw = json.loads(_frame(hp=20).payload_json)
    raw["rng_seed"] = "secret-alpha"
    a = Observation("prototype-fair-v0", json.dumps(raw), "hash-a")
    raw["rng_seed"] = "secret-beta"
    b = Observation("prototype-fair-v0", json.dumps(raw), "hash-b")
    first = agent.choose(a, _map_actions())
    second = agent.choose(b, _map_actions())

    assert first.action == second.action
    assert first.metadata_json == second.metadata_json


def test_nonmap_actions_delegate_to_existing_heuristic() -> None:
    agent = RoutePlanningAgent()
    actions = (
        LegalAction("heal", "rest_heal"),
        LegalAction("upgrade", "rest_upgrade"),
    )
    choice = agent.choose(_frame(hp=12), actions)
    assert choice.action.action_id == "heal"
    assert choice.policy_name.startswith("route-planner-")


def test_incomplete_map_falls_back_without_crashing() -> None:
    agent = RoutePlanningAgent()
    observation = Observation("prototype-fair-v0", '{"hp":70,"max_hp":70}', "empty")
    decision = agent.choose(observation, _map_actions())
    assert decision.action in _map_actions()


def test_planner_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="horizon"):
        RoutePlanner(horizon=0)
    with pytest.raises(ValueError, match="discount"):
        RoutePlanner(discount=0.0)
