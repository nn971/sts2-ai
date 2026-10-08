import json

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import LegalAction, Observation


def _observation(payload: dict[str, object]) -> Observation:
    return Observation(
        policy_id="test",
        payload_json=json.dumps(payload),
        observation_hash="hash",
    )


def test_heuristic_heals_when_low() -> None:
    agent = HeuristicAgent()
    observation = _observation({"hp": 20, "max_hp": 70})
    actions = (
        LegalAction("heal", "rest_heal"),
        LegalAction("upgrade", "rest_upgrade"),
    )
    assert agent.choose(observation, actions).action.kind == "rest_heal"


def test_heuristic_upgrades_when_healthy() -> None:
    agent = HeuristicAgent()
    observation = _observation({"hp": 65, "max_hp": 70})
    actions = (
        LegalAction("heal", "rest_heal"),
        LegalAction("upgrade", "rest_upgrade"),
    )
    assert agent.choose(observation, actions).action.kind == "rest_upgrade"


def test_heuristic_reads_pascal_case_map_payload() -> None:
    agent = HeuristicAgent()
    observation = _observation(
        {
            "hp": 20,
            "max_hp": 70,
            "map": [
                {"node_id": "fight", "room_type": 0},
                {"node_id": "rest", "room_type": 4},
            ],
        }
    )
    actions = (
        LegalAction(
            "choose_map_node:fight",
            "choose_map_node",
            payload_json='{"NodeId":"fight"}',
        ),
        LegalAction(
            "choose_map_node:rest",
            "choose_map_node",
            payload_json='{"NodeId":"rest"}',
        ),
    )

    assert agent.choose(observation, actions).action.action_id == "choose_map_node:rest"


def test_heuristic_reads_pascal_case_combat_payload() -> None:
    agent = HeuristicAgent()
    observation = _observation(
        {
            "hp": 60,
            "max_hp": 70,
            "combat": {
                "energy": 3,
                "hand": [
                    {"instance_id": 10, "card_id": "Strike"},
                    {"instance_id": 11, "card_id": "Defend"},
                ],
                "enemies": [
                    {"instance_id": 2, "enemy_id": "test_enemy", "hp": 6},
                ],
            },
        }
    )
    actions = (
        LegalAction(
            "play_card:strike",
            "play_card",
            payload_json='{"CardInstanceId":10,"TargetEnemyId":2}',
        ),
        LegalAction(
            "play_card:defend",
            "play_card",
            payload_json='{"CardInstanceId":11,"TargetEnemyId":null}',
        ),
    )

    assert agent.choose(observation, actions).action.action_id == "play_card:strike"


def test_heuristic_claims_card_before_native_leave_reward() -> None:
    agent = HeuristicAgent()
    observation = _observation({
        "hp": 50,
        "max_hp": 70,
        "reward": {"card_resolved": False},
        "deck": [{"card_id": "proto.silent.strike"}] * 13,
    })
    actions = (
        LegalAction("leave", "leave_reward"),
        LegalAction("take", "take_reward_card", '{"Index":0}'),
        LegalAction("skip", "skip_reward_card"),
    )
    assert agent.choose(observation, actions).action.kind == "take_reward_card"


def test_heuristic_claims_gold_and_relic_before_leaving_native_reward() -> None:
    agent = HeuristicAgent()
    frame = _observation({"hp": 50, "max_hp": 70, "reward": {}})
    actions = (
        LegalAction("leave", "leave_reward"),
        LegalAction("gold", "take_reward_gold"),
        LegalAction("relic", "take_reward_relic", '{"Index":0}'),
    )
    assert agent.choose(frame, actions).action.kind == "take_reward_relic"
    assert agent.choose(frame, actions[:2]).action.kind == "take_reward_gold"


def test_heuristic_can_leave_after_native_reward_claimed() -> None:
    agent = HeuristicAgent()
    observation = _observation({"reward": {"card_resolved": True}})
    assert agent.choose(
        observation, (LegalAction("leave", "leave_reward"),)
    ).action.kind == "leave_reward"


def test_heuristic_preserves_legacy_ordered_reward_choice() -> None:
    agent = HeuristicAgent()
    frame = _observation({
        "hp": 60, "max_hp": 70,
        "reward": {"independent_selection": False},
        "deck": [{"card_id": "proto.silent.strike"}] * 13,
    })
    actions = (
        LegalAction("take", "take_reward_card", '{"Index":0}'),
        LegalAction("skip", "skip_reward_card"),
    )
    assert agent.choose(frame, actions).action.kind == "take_reward_card"
